"""DS 1017 on several ranks through the whole engine (doc/specs/v1-ds1017.md stage 3): the model's
routed experts and Engram tables split over the workers, each worker with requests of its own.

For each backend the engine is served twice with the same checkpoint, on one rank and on several, and
asked the same prompts, one after another and then all at once. Asked alone, a prompt's text has to
be the same on one rank and on several, token for token. A model cut to a few layers is chaotic (one
changed bit moves its logits by tenths), so equal text over dozens of tokens says the ranks compute
what one rank computes. Asked with the others, the text is the same again only with kernels that
compute a token regardless of the rest of its step (the references; not the reference
implementation's, backend ``tilelang``): where it is on one rank, it has to be on several.
A model one card cannot hold (the whole checkpoint) is served on several ranks only, with
``--no-single``: its text is printed, and compared with ``--expect`` where given.

    # a random small model, two ranks on one card (the 4090 D)
    python tests/gpu/check_ds1017_ranks.py --random /tmp/ds-ranks --ranks 2 --backends reference,tilelang
    # the real checkpoint's first 8 layers, four cards against one (the SM100)
    python tests/gpu/check_ds1017_ranks.py --model <dir> --layers 8 --gpus 0,1,2,3 --backends tilelang
    # the whole checkpoint on four cards
    python tests/gpu/check_ds1017_ranks.py --model <dir> --gpus 0,1,2,3 --backends tilelang --no-single \\
        --prompt "Question: Who wrote Romeo and Juliet?\\nAnswer:" --max-tokens 32

It prints one ``summary`` line a backend and exits 1 when a text differs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "engine"), str(REPO)]

from tests.scenarios.deploy import cli, post, served  # noqa: E402

WORDS = ["w1 w2 w3 w4 w5", "w9 w8 w7", " ".join(f"w{i}" for i in range(20, 39)), "w40 w41 w42 w43", "w50 w51",
         " ".join(f"w{i}" for i in range(60, 200))]
"""Prompts in the test tokenizer's words: short ones, and one longer than the window and a page."""


def random_checkpoint(root: Path, shapes: str, seed: int) -> tuple[Path, Path]:
    """A random checkpoint in the real one's formats, with a word tokenizer; its directory and the tokenizer's."""
    from tokenizers import Tokenizer

    from models.ds1017_config import compressed_token_map
    from tests.gpu.support_ds1017 import real_shapes_config, small_config
    from tests.units.support.tokenizer import save_tokenizer

    tokenizer, src, out = root / "tokenizer", root / "config", root / "ckpt"
    if not (out / "config.json").is_file():
        vocab = 4096
        save_tokenizer(tokenizer, vocab, eos_in_config=False)
        _, compressed = compressed_token_map(Tokenizer.from_file(str(tokenizer / "tokenizer.json")))
        make = real_shapes_config if shapes == "real" else small_config
        cfg = {**make(vocab_size=vocab, engram_compressed_vocab_size=compressed), "eos_token_id": 0}
        src.mkdir(parents=True, exist_ok=True)
        src.joinpath("config.json").write_text(json.dumps(cfg))
        cli("checkpoint", "--config", str(src), "--out", str(out), "--tokenizer", str(tokenizer), "--seed", str(seed),
            timeout=1800)
    return out, tokenizer


def graphs(run: Path) -> list[str]:
    """What each worker's ``core.plan`` said about its decode graphs at start: the buckets it recorded,
    or why it recorded none."""
    out: list[str] = []
    for log in sorted(run.glob("host-a/snapshots/.run/worker-*/log")):
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith("core.plan: {"):
                try:
                    said = json.loads(line[len("core.plan: "):])["cuda_graph"]
                except (ValueError, KeyError):
                    continue
                out.append(f"buckets {said.get('buckets')}" + (f", {said['reason']}" if said.get("reason") else ""))
    return out


def generate(run: Path, serve: list[str], prompts: list[str], max_tokens: int, model: str, ready_s: float
             ) -> tuple[list[str], list[str], float]:
    """The prompts' completions asked one after another, then all at once; and tokens a second of the latter."""
    def ask(prompt: str) -> tuple[str, int]:
        got = post(url + "/v1/completions", {"model": model, "prompt": prompt, "max_tokens": max_tokens,
                                             "temperature": 0.0}, timeout=3600)
        return str(got["choices"][0]["text"]), int(got["usage"]["completion_tokens"])

    with served(run, *serve, ready_s=ready_s) as url:
        alone = [ask(p)[0] for p in prompts]
        started = time.monotonic()
        with ThreadPoolExecutor(len(prompts)) as pool:
            together = list(pool.map(ask, prompts))
        rate = sum(n for _, n in together) / (time.monotonic() - started)
    return alone, [text for text, _ in together], rate


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, help="a checkpoint directory (with its tokenizer)")
    p.add_argument("--random", type=Path, help="write a random checkpoint here and serve it")
    p.add_argument("--shapes", choices=("small", "real"), default="small", help="of the random checkpoint")
    p.add_argument("--layers", type=int, default=0, help="serve the first N layers only")
    p.add_argument("--gpus", help="one rank on each of these cards")
    p.add_argument("--ranks", type=int, default=2, help="without --gpus: that many ranks on one card")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--backends", default="reference,tilelang")
    p.add_argument("--no-single", action="store_true", help="do not serve on one rank (the model does not fit a card)")
    p.add_argument("--prompt", action="append", default=[], help="a prompt (repeatable); default: words of the test tokenizer")
    p.add_argument("--expect", action="append", default=[], help="the i-th prompt's text has to start with this")
    p.add_argument("--max-tokens", type=int, default=24)
    p.add_argument("--max-context", type=int, default=512)
    p.add_argument("--page-size", type=int, default=0)
    p.add_argument("--timeout", type=int, default=1800, help="seconds an engine may take to come up")
    p.add_argument("--serve", action="append", default=[], help="another argument for rsinfer serve (repeatable)")
    p.add_argument("--balanced", default="", metavar="INTERVAL,WINDOW,EXTRA",
                   help="serve on several ranks once more with expert parallel load balancing (eplb.interval, "
                        "eplb.window, eplb.extra_experts): the experts are placed again while it generates, and the "
                        "text has to be what it is without")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if (args.model is None) == (args.random is None):
        raise SystemExit("--model or --random")
    tokenizer = None
    if args.random is not None:
        args.model, tokenizer = random_checkpoint(args.random, args.shapes, seed=3)
    prompts = args.prompt or WORDS
    name = json.loads((args.model / "config.json").read_text()).get("_name_or_path") or "DeepSeek-V4.1-Flash"
    work = (args.random or Path("/tmp")) / f"ranks-{time.strftime('%H%M%S')}"
    work.mkdir(parents=True, exist_ok=True)
    common = ["--model", str(args.model), "--device", args.device, "--max-context", str(args.max_context),
              "--max-rows", "8", "--timeout", str(args.timeout), "--served-model-name", name,
              *(["--tokenizer", str(tokenizer)] if tokenizer else []), *(["--layers", str(args.layers)] if args.layers else []),
              *(["--page-size", str(args.page_size)] if args.page_size else []), *args.serve]
    several = ["--gpus", args.gpus] if args.gpus else ["--ranks", str(args.ranks)]
    one = ["--gpu", args.gpus.split(",")[0]] if args.gpus else []
    report: dict[str, Any] = {"prompts": prompts, "backends": {}}
    failed = False
    for backend in args.backends.split(","):
        serve = [*common, "--ds-backend", backend]
        ranked, together, rate = generate(work / f"{backend}-ranks", [*serve, *several], prompts, args.max_tokens, name,
                                          args.timeout)
        entry: dict[str, Any] = {"ranks": ranked, "ranks_together": together, "tokens_per_s": round(rate, 2),
                                 "graphs": graphs(work / f"{backend}-ranks")}
        print(f"graphs {backend}: {entry['graphs']}", flush=True)
        problems: list[str] = []
        invariant = ""
        if not args.no_single:
            single, single_together, single_rate = generate(work / f"{backend}-one", [*serve, *one], prompts,
                                                            args.max_tokens, name, args.timeout)
            entry.update(one=single, one_together=single_together, one_tokens_per_s=round(single_rate, 2))
            problems += [f"prompt {i}: {a!r} on several ranks, {b!r} on one" for i, (a, b) in
                         enumerate(zip(ranked, single, strict=True)) if a != b]
            if single == single_together:  # these kernels do not let the rest of a step show
                problems += [f"prompt {i} differs when asked with the others, on several ranks only"
                             for i, (a, b) in enumerate(zip(ranked, together, strict=True)) if a != b]
                invariant = "; also asked together"
            else:
                invariant = "; asked together the text differs on one rank too (kernels that depend on the step's rows)"
        problems += [f"prompt {i}: {text!r} does not start with {want!r}" for i, (text, want) in
                     enumerate(zip(ranked, args.expect, strict=False)) if not text.startswith(want)]
        words = sum(len(t.split()) for t in ranked)
        if args.balanced:
            interval, window, extra = args.balanced.split(",")
            eplb = ["--set", f"eplb.interval={interval}", "--set", f"eplb.window={window}", "--set",
                    f"eplb.extra_experts={extra}"]
            moved, moved_together, moved_rate = generate(work / f"{backend}-balanced", [*serve, *several, *eplb], prompts,
                                                         args.max_tokens, name, args.timeout)
            # What rank 0 said each time the experts were placed again.
            placed = [line for log in (work / f"{backend}-balanced").glob("host-a/snapshots/.run/worker-0-*/log")
                      for line in log.read_text(errors="replace").splitlines() if "experts placed again" in line]
            entry.update(balanced=moved, balanced_together=moved_together, balanced_tokens_per_s=round(moved_rate, 2),
                         placed=placed)
            problems += [f"prompt {i}: {a!r} with the experts placed again, {b!r} without" for i, (a, b) in
                         enumerate(zip(moved, ranked, strict=True)) if a != b]
            if not placed:
                problems.append("the experts were never placed again")
            print(f"placed {backend}: {len(placed)} times, {moved_rate:.1f} tokens/s with {len(prompts)} at once; "
                  f"the last: {placed[-1] if placed else ''}", flush=True)
        entry["problems"] = problems
        report["backends"][backend] = entry
        failed |= bool(problems)
        print(f"summary {backend}: {'FAILED ' + '; '.join(problems[:3]) if problems else 'same text'}"
              f"{'' if args.no_single else ' on one rank and on several'}{invariant}, {len(prompts)} prompts, "
              f"{words} words, {rate:.1f} tokens/s with {len(prompts)} at once", flush=True)
        for prompt, text in zip(prompts, ranked, strict=True):
            print(f"  {prompt[:60]!r} -> {text[:200]!r}", flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
