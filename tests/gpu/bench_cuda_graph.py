"""Decode throughput with graphs on and off, at several ``batch.max_context_len``.

Run manually with the GPU environment; this is not a pytest timing assertion.
Model loading, capture, prefill and warm-up are outside the measured interval.
Each case first primes its entire sequence-length trajectory, then measures repeats.

The context length is varied because a replayed step's attention may cost by it: with FlashInfer
(``kernels.flashinfer``, what ``auto`` picks for this bf16 model) a replayed and an eager step
both cost by the rows' lengths and run the same kernels; without it a replayed step runs the
fixed-extent attention of ``kernels.torch_graph``, which costs by ``max_context`` and may round
another way than the eager kernel. The result says whether the greedy tokens of the two parted.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any

import torch

from core.contracts.records import Resync
from platforms.cuda_graph import CudaGraphPlatform
from platforms.torch_eager import TorchEagerPlatform
from tests.gpu.support import REPO, TEXT, free_gpu
from tests.units.support.worker import Driver, Worker

MODES = ("eager", "graph")


def checkpoint_at(root: Path) -> Path:
    config = root / "config"
    config.mkdir(parents=True, exist_ok=True)
    config_file = config / "config.json"
    expected = {"architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5", "text_config": TEXT}
    if config_file.exists() and json.loads(config_file.read_text()) != expected:
        raise ValueError("benchmark cache has a different model configuration")
    config_file.write_text(json.dumps(expected))
    checkpoint = root / "checkpoint"
    marker = root / "checkpoint-ready.json"
    if not marker.exists():
        subprocess.run([sys.executable, str(REPO / "deploy/agxinfer.py"), "checkpoint", "--config", str(config),
                        "--layers", "4", "--out", str(checkpoint), "--dtype", "bf16", "--seed", "1"], check=True)
        marker.write_text(json.dumps({"layers": 4, "seed": 1, "dtype": "bf16"}))
    return checkpoint


def decode(w: Worker, rows: int, depth: int, prompt: int, warmup: int, steps: int) -> dict[str, Any]:
    driver = Driver(w)
    executor = w.service("executor@1")
    for row in range(rows):
        driver.add(row, [(1000 + 37 * row + 11 * i) % 248000 for i in range(prompt)])
    driver.step({r: prompt for r in range(rows)})
    pending: deque[Any] = deque()
    tokens: list[list[int]] = []
    marks: list[tuple[float, Any, Any, Any]] = []

    def finish() -> None:
        step = pending.popleft()
        result = step.finish()
        tokens.append([int(t) for t in result.tokens.values()])
        # Keep only timing marks, not logits: retaining eager outputs changes allocator use.
        marks.append((step.host_us, step.begin, step.end, step.previous))

    def run(count: int) -> None:
        for _ in range(count):
            delta, plan, state = driver.records({r: 1 for r in range(rows)})
            assert plan.phase == "decode" and all(plan.from_previous.values())
            driver.epoch += 1
            step = executor.submit(driver.epoch, delta, plan, state)
            if isinstance(step, Resync):
                raise RuntimeError(f"unexpected resync: {step}")
            driver.full, driver.removed = False, []
            for request in driver.requests.values():
                request.in_batch = True
                request.computed += 1
                request.known.append(-1)
            pending.append(step)
            if len(pending) >= depth:
                finish()
        while pending:
            finish()

    run(warmup)
    tokens.clear()
    marks.clear()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    start = time.perf_counter()
    run(steps)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    device_ms = [begin.us_until(end) / 1000 for _, begin, end, _ in marks]
    idle_ms = [max(previous.us_until(begin), 0) / 1000 for _, begin, _, previous in marks[1:]
               if previous is not None]
    return {"batch": rows, "overlap_depth": depth, "seconds": elapsed,
            "tokens_per_second": rows * steps / elapsed, "ms_per_step": elapsed * 1000 / steps,
            "median_submit_ms": statistics.median(h / 1000 for h, *_ in marks),
            "median_device_span_ms": statistics.median(device_ms),
            "median_idle_before_ms": statistics.median(idle_ms),
            "allocated_bytes": allocated, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "token_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--prompt", type=int, default=32)
    parser.add_argument("--max-context", type=int, nargs="+", default=[256, 4096, 16384],
                        help="batch.max_context_len of each run; the sequences stay prompt + warmup + steps long")
    parser.add_argument("--graph-memory-bytes", type=int, default=1 << 30)
    args = parser.parse_args()
    if min(args.repeats, args.steps, args.warmup, args.prompt) < 1 or args.steps < 2:
        parser.error("positive counts and at least two measured steps are required")
    if args.prompt + args.warmup + args.steps > min(args.max_context):
        parser.error("prompt + warmup + steps exceeds max-context")
    torch.set_num_threads(1)
    checkpoint = checkpoint_at(args.work_dir)
    cfg = json.loads((checkpoint / "config.json").read_text())
    source_file = REPO / ".ossrun-source.json"
    result: dict[str, Any] = {
        "source": json.loads(source_file.read_text()) if source_file.exists() else {},
        "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "python": platform.python_version(),
                        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                        "cpu_threads": torch.get_num_threads(), "tf32": torch.backends.cuda.matmul.allow_tf32},
        "config": {**vars(args), "model": "Qwen3.8-27B first 4 layers, random seed 1", "dtype": "bf16"},
        "trials": [], "priming": [], "builds": [],
    }
    result["environment"]["nvidia_smi_before"] = subprocess.check_output(["nvidia-smi"], text=True)
    eager_platform, graph_platform = TorchEagerPlatform(cuda=True), CudaGraphPlatform()
    cases = [(b, d) for b in (1, 4, 8) for d in (1, 2)]
    runs = [(repeat, context, mode) for repeat in range(args.repeats) for context in args.max_context
            # Rotate the mode order and reverse the case order to reduce warm-cache and order bias.
            for mode in MODES[repeat % 2:] + MODES[:repeat % 2]]
    for repeat, context, mode in runs:
        free_gpu()
        started = time.perf_counter()
        w = Worker(cfg, platform=(eager_platform if mode == "eager" else graph_platform), **{
            "weights.path": str(checkpoint), "model.dtype": "bf16", "state.gdn.conv_dtype": "bf16",
            "state.kv.pages": 2048, "state.kv.page_size": 4, "state.gdn.slots": 8,
            "batch.max_rows": 8, "batch.max_context_len": context,
            "plan.graph_memory_bytes": args.graph_memory_bytes,
            "plan.cuda_graph": "required" if mode == "graph" else "off"})
        plans: Any = None
        try:
            plans = w.service("plan@1")
            assert plans.buckets("decode") == ([1, 2, 4, 8] if mode == "graph" else [])
            result["builds"].append({"repeat": repeat, "max_context": context, "mode": mode,
                                     "load_seconds": time.perf_counter() - started, "resolution": plans.resolved})
            for batch, depth in cases if repeat % 2 == 0 else list(reversed(cases)):
                if repeat == 0:
                    prime = {"max_context": context, "mode": mode,
                             **decode(w, batch, depth, args.prompt, args.warmup, args.steps)}
                    result["priming"].append(prime)
                    print("BENCH_PRIME " + json.dumps(prime), flush=True)
                trial = {"repeat": repeat, "max_context": context, "mode": mode,
                         **decode(w, batch, depth, args.prompt, args.warmup, args.steps)}
                result["trials"].append(trial)
                print("BENCH_TRIAL " + json.dumps(trial), flush=True)
        finally:
            w.close()
            del plans, w
            free_gpu()
    hashes = {(t["repeat"], t["max_context"], t["batch"], t["overlap_depth"], t["mode"]): t["token_sha256"]
              for t in result["trials"]}
    result["tokens_equal"] = {f"ctx{c}/b{b}/depth{d}": all(hashes[r, c, b, d, "graph"] == hashes[r, c, b, d, "eager"]
                                                           for r in range(args.repeats))
                              for c in args.max_context for b, d in cases}
    result["environment"]["nvidia_smi_after"] = subprocess.check_output(["nvidia-smi"], text=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print("BENCH_RESULT " + json.dumps(result, default=str), flush=True)
    print("AGXINFER_GRAPH_BENCH_OK", flush=True)


if __name__ == "__main__":
    main()
