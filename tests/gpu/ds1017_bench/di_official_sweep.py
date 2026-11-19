"""Resident multi-point DeepInfer official Prefill / Decode sweep driver.

Wraps upstream DeepInfer ``examples/benchmark.py`` so multiple budgets (prefill) or decode
concurrencies (with matching construction-time request capacity) share a single model load.

Environment variables (set by ``run_di_official.sh``):
  SWEEP_MODE      "prefill" or "decode" (default: prefill)
  SWEEP_POINTS    Comma-separated budgets (e.g. 4096,8192,16384,32768,65536) or D values (e.g. 4,8,16,32,64,128)
  SWEEP_OUT       Output directory for per-point JSONL files
  SWEEP_PREFIX    Output file prefix (default: di-prefill or di-decode)
  ENGRAM_QPS      Optional integer overriding active/allocated DeepEP Engram QPs (unset = stock default)
  SWEEP_COPY_DIR  Optional directory to mirror each completed JSONL file (e.g. /mnt/artifacts/runs)
  SWEEP_COPY_TAG  Optional filename prefix when mirroring into SWEEP_COPY_DIR
"""
# pyright: reportMissingImports=false

from __future__ import annotations

import json
import os
from pathlib import Path
from shutil import copyfile
from typing import Any

import benchmark as bench
from _launch import launch_dp


class Resident:
    """Keep a loaded DeepInfer LLM resident across points and drain between runs."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def __getattr__(self, name: str) -> Any:
        return getattr(self.llm, name)

    def close(self) -> None:
        core = self.llm.llm_engine
        while core.has_unfinished_requests():
            core.step()


def _maybe_override_engram_qps(rank: int) -> int | None:
    raw = os.environ.get("ENGRAM_QPS", "").strip()
    if not raw:
        return None
    qp = int(raw)
    from deep_ep import EngramBuffer

    orig_cfg = EngramBuffer.get_theoretical_config
    orig_fetch = EngramBuffer.fetch

    def config(*args: Any, **kwargs: Any) -> tuple[int, Any, Any]:
        orig_q, depth, restrict = orig_cfg(*args, **kwargs)
        print("QP_ALLOCATION", rank, "original", orig_q, "allocated", qp, "depth", depth, "restrict", restrict, flush=True)
        return qp, depth, restrict

    def fetch(self: Any, indices: Any, num_qps: int = 0, use_tma_aligned_col_major_sf: bool = True) -> Any:
        return orig_fetch(self, indices, num_qps=qp, use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf)

    EngramBuffer.get_theoretical_config = staticmethod(config)
    EngramBuffer.fetch = fetch
    return qp


def _mirror(path: Path) -> None:
    copy_dir = os.environ.get("SWEEP_COPY_DIR", "").strip()
    if not copy_dir:
        return
    tag = os.environ.get("SWEEP_COPY_TAG", "").strip()
    name = f"{tag}-{path.name}" if tag else path.name
    dst = Path(copy_dir) / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    copyfile(path, dst)


def run(dp_size: int, rank: int) -> None:
    mode = os.environ.get("SWEEP_MODE", "prefill").strip().lower()
    points = [int(x) for x in os.environ["SWEEP_POINTS"].split(",") if x.strip()]
    out = Path(os.environ["SWEEP_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    prefix = os.environ.get("SWEEP_PREFIX", f"di-{mode}")
    override_qp = _maybe_override_engram_qps(rank)

    llm = bench.setup_llm(rank)
    active = Resident(llm)
    core = llm.llm_engine
    bench.setup_llm = lambda _: active
    try:
        if mode == "prefill":
            for budget in points:
                active.close()
                assert not core.has_unfinished_requests()
                bench.args.prefill_batch_size = budget
                bench.full_prefill_tokens = budget
                bench.prefill_num_requests = min(
                    512,
                    (bench.prefill_total_steps * budget + bench.args.non_cached_tokens - 1)
                    // bench.args.non_cached_tokens,
                )
                bench.batch_num_requests = max(bench.prefill_num_requests, bench.args.decode_num_requests)
                bench.total_non_cached_tokens = bench.batch_num_requests * bench.args.non_cached_tokens
                core.scheduler_config.max_num_batched_tokens = budget
                core.scheduler_config.max_num_seqs = bench.batch_num_requests
                core.scheduler = core._create_scheduler()
                if core.multi_modal_processor is not None:
                    core.multi_modal_processor.scheduler = core.scheduler
                out_file = out / f"{prefix}-{budget}.jsonl"
                bench.args.result_jsonl = out_file
                print("OFFICIAL_PREFILL_POINT", rank, budget, bench.batch_num_requests, flush=True)
                bench.main(dp_size, rank)
                if rank == 0:
                    row = json.loads(out_file.read_text(encoding="utf-8").splitlines()[-1])
                    assert row["prefill_window_tokens"] == row["prefill_window_steps"] * budget and row["prefill_window_steps"] > 0, row
                    if override_qp is not None:
                        row["active_engram_qps"] = override_qp
                        row["allocated_engram_qps"] = override_qp
                        out_file.write_text(json.dumps(row) + "\n", encoding="utf-8")
                    _mirror(out_file)
                    print("OFFICIAL_PREFILL_RESULT", budget, row["prefill_tokens_per_s_per_gpu"], flush=True)
        elif mode == "decode":
            for d in points:
                active.close()
                assert not core.has_unfinished_requests()
                assert max(bench.prefill_num_requests, d) == bench.batch_num_requests, (
                    f"D={d} exceeds resident batch_num_requests={bench.batch_num_requests}; run D={d} cold"
                )
                bench.args.decode_num_requests = d
                out_file = out / f"{prefix}-{d}.jsonl"
                bench.args.result_jsonl = out_file
                print("OFFICIAL_DECODE_POINT", rank, d, flush=True)
                bench.main(dp_size, rank)
                if rank == 0:
                    row = json.loads(out_file.read_text(encoding="utf-8").splitlines()[-1])
                    if override_qp is not None:
                        row["active_engram_qps"] = override_qp
                        row["allocated_engram_qps"] = override_qp
                        out_file.write_text(json.dumps(row) + "\n", encoding="utf-8")
                    _mirror(out_file)
                    print("OFFICIAL_DECODE_RESULT", d, row["decode_tokens_per_s_per_gpu"], flush=True)
        else:
            raise SystemExit(f"unsupported SWEEP_MODE={mode!r} (expected prefill or decode)")
    finally:
        llm.close()


if __name__ == "__main__":
    launch_dp(run, bench.dp_size)
