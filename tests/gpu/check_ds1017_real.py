"""Manual checks on an SM 10.0 card (doc/specs/v1-ds1017.md, stage 2): the engine's worker and the
reference implementation shipped with the DS 1017 checkpoint on the same weights. Not a pytest: one
needs the real checkpoint, both need a card whose shared memory holds the attention kernel at the
real shapes.

    python tests/gpu/check_ds1017_real.py --model <checkpoint dir> --layers 4 [--out result.json]
    python tests/gpu/check_ds1017_real.py --random <empty dir> [--out result.json]
    ... --backends tilelang,deepinfer,deepinfer+graph

``--model``: the first layers of the real checkpoint, with all their experts and their Engram
tables. One GPU holds 4 layers (their experts, 28 GiB, and the first Engram table, 97 GiB) once for
each of the two, one after the other. ``--random``: a random checkpoint at the released model's
shapes with every kind of layer (``real_shapes_config``), written to the directory.

``--shadow tilelang``: beside every kernel call of the engine, that backend's kernel on the same
arguments, and how far apart the results are (the kernels compared on a real forward's data).

``--backends``: the engine once for each (``dsv4.backend``), ``+graph`` for decode steps replayed
from CUDA graphs, ``~direct`` for the routed experts computed with DeepInfer's rounding by the
reference kernel (``--compare tilelang~direct``: what is left between DeepInfer's kernels and the
reference implementation's once that one known difference is taken out). With ``tilelang``, that implementation's own kernels, the logits of the prefill and
of every decode step must be the same bits as its. Another backend computes the same function with
other kernels: reported are how far its logits are from the reference implementation's while both
have produced the same tokens (after the first different token they answer different questions),
and how long a decode step takes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "engine"), str(REPO)]

import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from deploy.models.hf import cut, read_config  # noqa: E402
from kernels import ds1017_formats as formats  # noqa: E402
from kernels.ds1017 import ops as reference_ops  # noqa: E402
from platforms.cuda_graph import CudaGraphPlatform  # noqa: E402
from tests.gpu.support import free_gpu  # noqa: E402
from tests.gpu.support_ds1017 import real_shapes_config, reference_greedy, reference_model, worker  # noqa: E402
from tests.units.support.ds1017 import checkpoint, driver  # noqa: E402

def shadow(w: Any, against: str, stats: dict[str, dict[str, float]]) -> None:
    """Every kernel call of the worker's layers also computed by ``against``'s solution of the same
    definition on the same arguments, and how far apart the two results are, per definition: the
    kernels compared on the activations and weights of a real forward, one call at a time (the
    results of the worker's own kernels go on)."""
    layers, catalog = w.service("layers.ds1017@1"), w.service("kernels.catalog@1")
    inner = layers.k

    def k(definition: str) -> Any:
        fn = inner(definition)
        other_id = f"{against}.{definition}"
        if other_id not in catalog.solutions(definition):
            return fn
        other = catalog.resolve(other_id)

        def call(*a: Any) -> Any:
            if definition == "ds.store_cache@1":  # it writes the cache: compared in tests/gpu, not here
                return fn(*a)
            # The compressor writes its ring: the other kernel gets a copy of it (it is small).
            b = tuple(x.clone() if i == 2 and definition.startswith("ds.compress") else x for i, x in enumerate(a))
            want = other(*b)
            out = fn(*a)
            got: list[torch.Tensor] = [t for t in (out if isinstance(out, tuple) else (out,)) if t is not None]
            ref: list[torch.Tensor] = [t for t in (want if isinstance(want, tuple) else (want,)) if t is not None]
            entry = stats.setdefault(definition, {"calls": 0, "worst": 0.0, "unequal": 0.0, "other_choices": 0})
            entry["calls"] += 1
            if not got[0].numel():
                return out
            if definition in ("ds.index_topk@1", "ds.index_topk_paged@1"):  # the latents chosen, as sets
                shared = total = 0
                for x, y in zip(got[0].tolist(), ref[0].tolist(), strict=True):
                    sx, sy = {v for v in x if v >= 0}, {v for v in y if v >= 0}
                    shared, total = shared + len(sx & sy), total + max(len(sx), len(sy))
                entry["unequal"] = max(entry["unequal"], 1 - shared / max(total, 1))
                return out
            if definition == "ds.quant_index_q@1":  # FP4 bytes and scales: compared as the values they hold
                got, ref = [formats.dequant_fp4(got[0], got[1])], [formats.dequant_fp4(ref[0], ref[1])]
            if definition == "ds.compress_decode@1":  # a token that completes no group pools nothing defined
                complete = a[4] % a[5] == 0
                got, ref = [got[0][complete]], [ref[0][complete]]
            if definition == "ds.top@1":  # the values chosen, whichever of equal ones
                got = [a[0].gather(-1, got[0]).sort(dim=-1).values.nan_to_num(neginf=-1e30)]
                ref = [a[0].gather(-1, ref[0]).sort(dim=-1).values.nan_to_num(neginf=-1e30)]
            if definition == "ds.moe_gate@1":  # the experts chosen, then their weights by expert
                gw, gi, rw, ri = got[0], got[1], ref[0], ref[1]
                go, ro = gi.argsort(-1), ri.argsort(-1)
                entry["other_choices"] += int((gi.gather(1, go) != ri.gather(1, ro)).any(-1).sum())
                got, ref = [gw.gather(1, go)], [rw.gather(1, ro)]
            for g, r in zip(got, ref, strict=True):
                if not g.numel():
                    continue
                g, r = g.float(), r.float()
                entry["worst"] = max(entry["worst"], float((g - r).abs().max() / r.abs().max().clamp_min(1e-12)))
                entry["unequal"] = max(entry["unequal"], float((g != r).float().mean()))
            return out

        return call

    layers.k = k


def direct_experts(w: Any) -> None:
    """The worker's routed experts replaced by the reference kernel with DeepInfer's rounding (the
    SwiGLU's result quantized from fp32): what the reference implementation would compute if it did
    not round that result to bf16 first."""
    layers = w.service("layers.ds1017@1")
    inner = layers.k
    layers.k = lambda definition: ((lambda *a: reference_ops.moe_experts(*a, direct=True))
                                   if definition == "ds.moe_experts@1" else inner(definition))


def tap_head(w: Any, vocab: int, sink: list[torch.Tensor]) -> None:
    """What goes into the output head at every step (the last layer's stream, normed), kept on the CPU."""
    layers = w.service("layers.ds1017@1")
    inner = layers.k

    def k(definition: str) -> Any:
        fn = inner(definition)
        if definition != "ds.linear@1":
            return fn

        def call(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            if weight.shape[0] == vocab:
                sink.append(x.float().cpu())
            return fn(x, weight)

        return call

    layers.k = k


def agreement(tokens: list[int], logits: list[torch.Tensor], want_tokens: list[int], want_logits: list[torch.Tensor]
              ) -> tuple[int, float, list[float]]:
    """How many tokens are the same before the first different one, the largest logit difference up
    to there (the runs answer the same question until then), and every step's."""
    diffs = [float((a.float() - b.float()).abs().max()) for a, b in zip(logits, want_logits, strict=True)]
    same = next((i for i, (a, b) in enumerate(zip(tokens, want_tokens, strict=True)) if a != b), len(tokens))
    return same, max(diffs[:same + 1]) if diffs else 0.0, diffs


TEXT = ("The engine keeps a sliding window of the last tokens for every layer, compresses older context into "
        "latents, and lets an indexer choose which of them each query attends to. 引擎为每一层保留最近若干 token 的"
        "滑动窗口，把更早的上下文压缩成 latent，再由 indexer 为每个查询挑选要看的那些。") * 12


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path)
    source.add_argument("--random", type=Path)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--prompt-tokens", type=int, default=400)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--backends", default="tilelang")
    parser.add_argument("--more-steps", type=int, default=32, help="decode steps timed after the compared ones")
    parser.add_argument("--compare", default="", help="one of the backends: the others' logits against its")
    parser.add_argument("--shadow", default="", help="a backend whose kernels run beside the engine's on every call")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.random:
        model, cfg = args.random.resolve(), real_shapes_config()
        started = time.time()
        checkpoint(model, cfg, seed=1)
        print(f"random checkpoint at the real shapes in {time.time() - started:.0f} s", file=sys.stderr, flush=True)
        prompt = torch.randint(3, 4096, (args.prompt_tokens,), generator=torch.Generator().manual_seed(7)).tolist()
    else:
        model = args.model.resolve()
        cfg = cut(read_config(model), args.layers)
        prompt = Tokenizer.from_file(str(model / "tokenizer.json")).encode(TEXT).ids[:args.prompt_tokens]
    layers = cfg["text_config"]["num_hidden_layers"]
    report: dict[str, Any] = {"model": str(model), "layers": layers, "prompt_tokens": len(prompt), "steps": args.steps,
                              "device": torch.cuda.get_device_name(0), "torch": torch.__version__}

    started = time.time()
    reference = reference_model(model, args.context, cfg=cfg)
    report["reference_load_s"] = round(time.time() - started, 1)
    started = time.time()
    want_tokens, want_logits = reference_greedy(reference, prompt, args.steps)
    report["reference_run_s"] = round(time.time() - started, 1)
    del reference
    free_gpu()

    ok = True
    scale = float(want_logits[0].abs().max())
    report["logit_scale"], report["reference_tokens"], report["engines"] = scale, want_tokens, {}
    runs: dict[str, tuple[list[int], list[torch.Tensor]]] = {}
    heads: dict[str, list[torch.Tensor]] = {}
    vocab = int(cfg.get("text_config", cfg)["vocab_size"])
    for name in args.backends.split(","):
        backend, _, graph = name.partition("+")
        backend, _, variant = backend.partition("~")
        # Pages of 128 tokens, DeepInfer's blocks: its paged indexer takes 64 or 128 latents a page.
        config = {"dsv4.backend": backend, "batch.max_context_len": args.context, "state.kv.page_size": 128,
                  "state.kv.pages": -(-args.context // 128) * 4, "state.swa.slots": 2, "batch.max_rows": 2,
                  "holder.mem_fraction": 0.95,
                  "plan.cuda_graph": "required" if graph else "off"}
        started = time.time()
        try:
            w = worker(cfg, model, CudaGraphPlatform() if graph else None, **config)
        except Exception as e:  # the other backends' results are still worth having
            report["engines"][name] = {"error": f"{type(e).__name__}: {str(e)[:600]}"}
            ok = False
            free_gpu()
            continue
        entry: dict[str, Any] = {"load_s": round(time.time() - started, 1)}
        try:
            if args.shadow and not graph:
                entry["kernels_against_" + args.shadow] = {}
                shadow(w, args.shadow, entry["kernels_against_" + args.shadow])
            if variant == "direct":
                direct_experts(w)
            if not graph:
                tap_head(w, vocab, heads.setdefault(name, []))
            d = driver(w)
            d.add(0, prompt)
            started = time.time()
            logits = d.generate([0], 1)[0]
            torch.cuda.synchronize()
            entry["prefill_s"] = round(time.time() - started, 2)
            started = time.time()
            logits += d.generate([0], args.steps)[0]
            torch.cuda.synchronize()
            entry["decode_ms_per_step"] = round((time.time() - started) / max(1, args.steps - 1) * 1000, 1)
            tokens = list(d.requests[0].outputs)
            started = time.time()
            d.generate([0], args.steps + args.more_steps)
            torch.cuda.synchronize()
            entry["warm_decode_ms_per_step"] = round((time.time() - started) / max(1, args.more_steps) * 1000, 1)
            entry["solutions"] = sorted({v.split(".")[0] for table in ("prefill", "decode") for k, v in
                                         w.service("layers.ds1017@1").resolve([])[table].items() if k != "backend"})
        finally:
            w.close()
            free_gpu()
        same, worst, diffs = agreement(tokens, logits, want_tokens, want_logits)
        entry |= {"tokens_equal_before_the_first_different": same, "steps_bitwise_equal": sum(x == 0.0 for x in diffs),
                  "max_logit_difference_while_tokens_agree": worst,
                  "logit_differences": [round(x, 4) for x in diffs], "tokens": tokens}
        runs[name] = (tokens, [x.cpu() for x in logits])
        report["engines"][name] = entry
        if backend == "tilelang" and not graph and not variant:
            ok = ok and entry["steps_bitwise_equal"] == args.steps
    print(json.dumps(report, indent=1))
    for name, e in report["engines"].items():
        if "error" in e:
            print(f"summary: {layers} layers, {name}: {e['error']}")
            continue
        print(f"summary: {layers} layers, {name}: {e['steps_bitwise_equal']}/{args.steps} steps bit-equal, "
              f"{e['tokens_equal_before_the_first_different']} tokens the same, logits within "
              f"{e['max_logit_difference_while_tokens_agree']:.4g} of {scale:.3g} while they are; prefill "
              f"{e['prefill_s']} s, decode {e['warm_decode_ms_per_step']} ms a step; kernels {e['solutions']}")
        if args.compare in runs and name != args.compare:
            same, worst, diffs = agreement(*runs[name], *runs[args.compare])
            e["against_" + args.compare] = {"tokens_equal_before_the_first_different": same,
                                            "max_logit_difference_while_tokens_agree": worst,
                                            "logit_differences": [round(x, 4) for x in diffs]}
            print(f"summary: {layers} layers, {name} against {args.compare}: {sum(x == 0.0 for x in diffs)}/{args.steps} "
                  f"steps bit-equal, {same} tokens the same, logits within {worst:.4g} of {scale:.3g} while they are")
            if name in heads and args.compare in heads:
                # The head's input at the steps where both had the same tokens before them.
                pairs = list(zip(heads[name], heads[args.compare]))[:same + 1]
                apart = [float((a - b).norm() / b.norm()) for a, b in pairs]
                e["against_" + args.compare]["head_input_relative_difference"] = [round(x, 5) for x in apart]
                print(f"summary: {layers} layers, {name} against {args.compare}: the head's input differs by "
                      f"{max(apart):.3g} of its norm at most ({apart[0]:.3g} after the prompt)")
        for definition, c in sorted(e.get("kernels_against_" + args.shadow, {}).items()):
            print(f"kernel: {name} {definition} against {args.shadow}: {int(c['calls'])} calls, the farthest "
                  f"{c['worst']:.3g} of the result's largest value, at most {c['unequal']:.4f} of the elements "
                  f"different" + (f", {int(c['other_choices'])} tokens with other experts" if c["other_choices"] else ""))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
