"""Manual checks on an SM 10.0 card (doc/specs/v1-ds1017.md, stage 2): the engine's worker and the
reference implementation shipped with the DS 1017 checkpoint on the same weights; the logits of a
prefill and of the decode steps after it must be the same bits. Not a pytest: one needs the real
checkpoint, both need a card whose shared memory holds the attention kernel at the real shapes.

    python tests/gpu/check_ds1017_real.py --model <checkpoint dir> --layers 4 [--out result.json]
    python tests/gpu/check_ds1017_real.py --random <empty dir> [--out result.json]

``--model``: the first layers of the real checkpoint, with all their experts and their Engram
tables. One GPU holds 4 layers (their experts, 28 GiB, and the first Engram table, 97 GiB) once for
each of the two, one after the other. ``--random``: a random checkpoint at the released model's
shapes with every kind of layer (``real_shapes_config``), written to the directory.

The engine runs that implementation's kernels (``dsv4.backend = tilelang``)."""

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
from tests.gpu.support import free_gpu  # noqa: E402
from tests.gpu.support_ds1017 import real_shapes_config, reference_greedy, reference_model, worker  # noqa: E402
from tests.units.support.ds1017 import checkpoint, driver  # noqa: E402

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

    started = time.time()
    w = worker(cfg, model, **{"dsv4.backend": "tilelang", "batch.max_context_len": args.context, "state.kv.pages": 2048,
                              "state.swa.slots": 2, "batch.max_rows": 2, "holder.mem_fraction": 0.95})
    report["engine_load_s"] = round(time.time() - started, 1)
    try:
        d = driver(w)
        d.add(0, prompt)
        started = time.time()
        logits = d.generate([0], args.steps)[0]
        report["engine_run_s"] = round(time.time() - started, 1)
        tokens = list(d.requests[0].outputs)
    finally:
        w.close()
        free_gpu()
    diffs = [float((a.float() - b.float()).abs().max()) for a, b in zip(logits, want_logits, strict=True)]
    report |= {"tokens_equal": sum(a == b for a, b in zip(tokens, want_tokens, strict=True)),
               "steps_bitwise_equal": sum(x == 0.0 for x in diffs), "max_logit_difference": max(diffs),
               "logit_scale": float(want_logits[0].abs().max()), "tokens": tokens, "reference_tokens": want_tokens}
    print(json.dumps(report, indent=1))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
    return 0 if report["steps_bitwise_equal"] == args.steps else 1


if __name__ == "__main__":
    raise SystemExit(main())
