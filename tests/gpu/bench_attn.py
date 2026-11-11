"""Standalone paged decode comparison; does not claim engine integration or serving throughput.

Each mode/case runs in a fresh process with identical seeded BF16 inputs. Report both
prepared attention execution and prepare + KV append + execution. Inputs and lengths are
stationary (hot-cache microbenchmark), not an autoregressive generation workload.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from typing import Any

import torch

from kernels.reference.ops import attention_decode as reference_decode
from kernels.reference.ops import kv_store
from kernels.torch_graph.ops import attention_decode as fixed_decode

MODES = ("reference_eager", "torch_graph", "flashinfer_eager", "flashinfer_graph")
HEADS, KV_HEADS, DIM, PAGE = 24, 4, 256, 16
REPO = Path(__file__).resolve().parents[2]


def summary(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {"median_us": statistics.median(values), "p90_us": ordered[math.ceil(len(values) * .9) - 1],
            "samples_us": values}


def measure(fn: Callable[[], None], warmup: int, iterations: int, repeats: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    wall: list[float] = []
    device: list[float] = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t = time.perf_counter()
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - t) * 1e6 / iterations)
        device.append(start.elapsed_time(end) * 1000 / iterations)
    return {"wall": summary(wall), "stream_elapsed": summary(device)}


def oracle(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, slots: torch.Tensor,
           lengths: list[int]) -> torch.Tensor:
    out = torch.empty_like(q, dtype=torch.float32)
    for row, length in enumerate(lengths):
        key, value = k[slots[row, :length].long()].float(), v[slots[row, :length].long()].float()
        query = q[row].float().reshape(KV_HEADS, HEADS // KV_HEADS, DIM)
        scores = torch.einsum("hgd,thd->hgt", query, key) * DIM ** -.5
        out[row] = torch.einsum("hgt,thd->hgd", scores.softmax(-1), value).reshape(HEADS, DIM)
    return out


def case(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    batch, length, context = args.batch, args.length, args.max_context
    lengths = [max(1, length - (r % 4) * (length // 8)) if args.varlen else length for r in range(batch)]
    pages_per_row = [(n + PAGE - 1) // PAGE for n in lengths]
    pages = sum(pages_per_row)
    generator = torch.Generator().manual_seed(args.seed)
    page_ids = (torch.randperm(pages, generator=generator) + 1).to(torch.int32)
    offsets = [0]
    for n in pages_per_row:
        offsets.append(offsets[-1] + n)
    slots_cpu = torch.zeros(batch, context, dtype=torch.int32)
    for r, n in enumerate(lengths):
        ids = page_ids[offsets[r]:offsets[r + 1]].long()
        slots_cpu[r, :n] = (ids[:, None] * PAGE + torch.arange(PAGE)).flatten()[:n].int()
    q = torch.randn(batch, HEADS, DIM, generator=generator, dtype=torch.bfloat16).cuda()
    k = torch.randn((pages + 1) * PAGE, KV_HEADS, DIM, generator=generator, dtype=torch.bfloat16).cuda()
    v = torch.randn(k.shape, generator=generator, dtype=torch.bfloat16).cuda()
    output = torch.empty_like(q)
    slots = slots_cpu.cuda()
    row_cpu = torch.arange(batch, dtype=torch.int64).pin_memory()
    len_cpu = torch.tensor(lengths, dtype=torch.int64).pin_memory()
    loc_cpu = torch.tensor([int(slots_cpu[r, n - 1]) for r, n in enumerate(lengths)],
                           dtype=torch.int64).pin_memory()
    rows, lens, loc = row_cpu.cuda(), len_cpu.cuda(), loc_cpu.cuda()
    # Appending the same values keeps the stationary benchmark's state identical across repetitions.
    new_k, new_v = k[loc].clone(), v[loc].clone()
    expected = oracle(q, k, v, slots, lengths)
    torch.cuda.synchronize()
    base_bytes = torch.cuda.memory_allocated()
    wrapper: Any = None
    graph: Any = None
    setup_started = time.perf_counter()

    def prepare_reference() -> None:
        rows.copy_(row_cpu, non_blocking=True)
        lens.copy_(len_cpu, non_blocking=True)
        loc.copy_(loc_cpu, non_blocking=True)

    def compute_reference() -> None:
        fn = fixed_decode if args.mode == "torch_graph" else reference_decode
        fn(q, output, k, v, slots, rows, lens, DIM ** -.5, True, False)

    prepare, compute = prepare_reference, compute_reference
    details: dict[str, Any] = {}
    if args.mode.startswith("flashinfer"):
        fi = importlib.import_module("flashinfer")
        use_graph = args.mode == "flashinfer_graph"
        workspace = torch.empty(args.workspace_mib << 20, device="cuda", dtype=torch.uint8)
        graph_args: dict[str, Any] = {}
        if use_graph:
            graph_args = {"paged_kv_indptr_buffer": torch.empty(batch + 1, device="cuda", dtype=torch.int32),
                          "paged_kv_indices_buffer": torch.empty(pages, device="cuda", dtype=torch.int32),
                          "paged_kv_last_page_len_buffer": torch.empty(batch, device="cuda", dtype=torch.int32)}
        wrapper = fi.BatchDecodeWithPagedKVCacheWrapper(
            workspace, "NHD", use_cuda_graph=use_graph, use_tensor_cores=True, backend=args.backend, **graph_args)
        indptr = torch.tensor(offsets, dtype=torch.int32).pin_memory()
        last = torch.tensor([(n - 1) % PAGE + 1 for n in lengths], dtype=torch.int32).pin_memory()
        indices_cpu = page_ids.pin_memory()
        indices = torch.empty_like(page_ids, device="cuda")
        paged_k = k.view(pages + 1, PAGE, KV_HEADS, DIM)
        paged_v = v.view_as(paged_k)

        def prepare_flashinfer() -> None:
            loc.copy_(loc_cpu, non_blocking=True)
            indices.copy_(indices_cpu, non_blocking=True)
            wrapper.plan(indptr, indices, last, HEADS, KV_HEADS, DIM, PAGE,
                         q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
                         pos_encoding_mode="NONE", sm_scale=DIM ** -.5)

        def compute_flashinfer() -> None:
            wrapper.run(q, (paged_k, paged_v), out=output)

        prepare, compute = prepare_flashinfer, compute_flashinfer
        details = {"flashinfer": importlib.metadata.version("flashinfer-python"),
                   "backend_requested": args.backend, "tensor_cores": True,
                   "workspace_mib": args.workspace_mib}

    def append_and_compute() -> None:
        kv_store(k, v, loc, new_k, new_v)
        compute()

    prepare()
    append_and_compute()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), expected, atol=1e-2, rtol=1e-2)
    max_error = (output.float() - expected).abs().max().item()
    setup_ms = (time.perf_counter() - setup_started) * 1000
    capture_ms = 0.0
    run = compute
    run_layer = append_and_compute
    graphs: list[Any] = []
    if args.mode.endswith("graph"):
        # Capture compute and the layer separately so run-only never includes KV append by accident.
        capture_started = time.perf_counter()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                append_and_compute()
        torch.cuda.current_stream().wait_stream(stream)
        for fn in (compute, append_and_compute):
            graph = torch.cuda.CUDAGraph()
            # The two graphs replay sequentially and may safely share temporary storage.
            with torch.cuda.graph(graph, stream=stream, pool=graphs[0].pool() if graphs else None):
                fn()
            graphs.append(graph)
        run, run_layer = graphs[0].replay, graphs[1].replay
        run_layer()
        torch.cuda.synchronize()
        capture_ms = (time.perf_counter() - capture_started) * 1000
        torch.testing.assert_close(output.float(), expected, atol=1e-2, rtol=1e-2)

    def layer() -> None:
        prepare()
        run_layer()

    torch.cuda.reset_peak_memory_stats()
    run_stats = measure(run, args.warmup, args.iterations, args.repeats)
    layer_stats = measure(layer, args.warmup, args.iterations, args.repeats)
    torch.testing.assert_close(output.float(), expected, atol=1e-2, rtol=1e-2)
    memory = {"base_allocated_bytes": base_bytes, "allocated_bytes": torch.cuda.memory_allocated(),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "reserved_bytes": torch.cuda.memory_reserved()}
    # A content change verifies that graph execution reads current Q, not capture-time results.
    q.mul_(.5)
    expected2 = oracle(q, k, v, slots, lengths)
    output.fill_(float("nan"))
    run()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), expected2, atol=1e-2, rtol=1e-2)
    output.fill_(float("nan"))
    layer()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), expected2, atol=1e-2, rtol=1e-2)
    # Distinct writes catch a missing append; compare the whole cache to detect stray writes.
    new_k.add_(.25)
    new_v.sub_(.5)
    expected_k, expected_v = k.clone(), v.clone()
    expected_k[loc], expected_v[loc] = new_k, new_v
    layer()
    torch.cuda.synchronize()
    torch.testing.assert_close(k, expected_k, atol=0, rtol=0)
    torch.testing.assert_close(v, expected_v, atol=0, rtol=0)
    expected3 = oracle(q, expected_k, expected_v, slots, lengths)
    torch.testing.assert_close(output.float(), expected3, atol=1e-2, rtol=1e-2)
    result = {"mode": args.mode, "batch": batch, "lengths": lengths, "max_context": context,
              "heads": HEADS, "kv_heads": KV_HEADS, "head_dim": DIM, "page_size": PAGE, "dtype": "bf16",
              "seed": args.seed, "varlen": args.varlen, "correct": True, "max_abs_error": max_error,
              "prepared_run": run_stats, "prepare_store_run": layer_stats,
              "setup_and_first_run_ms": setup_ms, "capture_two_graphs_ms": capture_ms,
              **memory, **details}
    for graph in graphs:
        graph.reset()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--lengths", type=int, nargs="+", default=[32, 256, 4096])
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--backend", default="fa2")
    parser.add_argument("--workspace-mib", type=int, default=128)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--length", type=int, default=32)
    args = parser.parse_args()
    lengths = [args.length] if args.mode else args.lengths
    batches = [args.batch] if args.mode else args.batches
    if min(*lengths, *batches, args.iterations, args.repeats, args.workspace_mib) < 1 or args.warmup < 0:
        parser.error("sizes, iterations and repeats must be positive; warmup must be nonnegative")
    if max(lengths) > args.max_context:
        parser.error("max-context must cover every sequence length")
    with torch.inference_mode():
        if args.mode:
            print("ATTN_CASE " + json.dumps(case(args)), flush=True)
            return
    props = torch.cuda.get_device_properties(0)
    source = REPO / ".ossrun-source.json"
    packages: dict[str, str] = {}
    for package in ("flashinfer-python", "apache-tvm-ffi", "nvidia-cutlass-dsl", "cuda-python", "ninja"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not installed"
    result: dict[str, Any] = {"scope": "stationary hot-cache decode attention microbenchmark; not engine throughput",
                             "environment": {"python": sys.version, "torch": torch.__version__,
                                             "cuda": torch.version.cuda, "gpu": props.name,
                                             "sm": [props.major, props.minor],
                                             "packages": packages,
                                             "source": json.loads(source.read_text()) if source.exists() else {},
                                             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                             "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True)},
                             "iterations": args.iterations, "repeats": args.repeats, "warmup": args.warmup,
                             "trials": []}
    jobs = [(b, n, m) for b in batches for n in lengths for m in args.modes]
    random.Random(args.seed).shuffle(jobs)
    for batch, length, mode in jobs:
        cmd = [sys.executable, "-m", "tests.gpu.bench_attn", "--mode", mode,
               "--batch", str(batch), "--length", str(length), "--max-context", str(args.max_context),
               "--iterations", str(args.iterations), "--repeats", str(args.repeats), "--warmup", str(args.warmup),
               "--seed", str(args.seed), "--backend", args.backend, "--workspace-mib", str(args.workspace_mib)]
        if args.varlen:
            cmd.append("--varlen")
        print(f"ATTN_START {mode} batch={batch} length={length} max_context={args.max_context}", flush=True)
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=None)
        print(proc.stdout, end="", flush=True)
        proc.check_returncode()
        trial = next(json.loads(line.removeprefix("ATTN_CASE ")) for line in proc.stdout.splitlines()
                     if line.startswith("ATTN_CASE "))
        result["trials"].append(trial)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        gc.collect()
    print("ATTN_RESULT " + json.dumps(result), flush=True)
    print("AGXINFER_ATTN_BENCH_OK", flush=True)


if __name__ == "__main__":
    main()
