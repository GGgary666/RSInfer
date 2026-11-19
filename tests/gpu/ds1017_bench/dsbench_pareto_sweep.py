"""4-GPU Pareto sweep driver for agxinfer matching DeepInfer's ``examples/sweep/run_sweep.sh``.

Drives a running ``agxinfer`` engine directly over ``StepChannel`` (pre-tokenized ``NewRequest`` ->
``TokenEvents`` shared-memory IPC, matching ``examples/benchmark.py``'s in-process ``_add_request`` +
``step()`` boundary) and appends JSONL records in the exact schema consumed by DeepInfer's
``examples/sweep/plot_pareto.py``:

- **Prefill mode** (``--mode prefill --prefill-batch-size B``): feeds uniform random non-special
  token sequences of ``--context-length`` in atomic quads. By default, the entire prompt is
  uncached. ``--non-cached-tokens`` primes a shared prefix per DP rank, checks the actual cache
  hits using the admission probe, and counts only the newly computed suffix. ``--yoco-clip``
  records the running engine's bounded-replay setting (it does not change engine configuration).
  The driver submits requests
  across all ``dp_size`` domains, warms up full-budget prefill steps, measures steady-state prefill
  step latency (``ms / prefill step``) and ``prefill_tokens_per_s_per_gpu``, while keeping trailing
  requests queued so ``sched.overlap`` remains saturated throughout the measured window.
- **Decode mode** (``--mode decode --decode-num-requests D1 D2 ...``): for each ``d`` (requests per
  DP rank, ``dp_size * d`` total concurrent requests across all GPUs), prefills ``--context-length``
  non-cached tokens per request, runs ``110`` warmup pure-decode steps (matching ``benchmark.py``'s
  ``50 + 60`` warmup steps), then measures ``60`` consecutive pure-decode steps to report
  ``decode_avg_sps`` (``tokens/s/user`` at ``nextn=0``) and ``decode_tokens_per_s_per_gpu``.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import itertools
import json
import os
import random
import secrets
import sys
import time
from pathlib import Path
from typing import Any

import msgspec
import zmq

REPO = next(p for p in Path(__file__).resolve().parents if (p / "engine").is_dir())
sys.path[:0] = [str(REPO / "tests" / "gpu"), str(REPO), str(REPO / "engine")]

from core.common.ids import ProcId  # noqa: E402
from core.contracts.columns import Column  # noqa: E402
from core.contracts.requests import NewRequest, Requests  # noqa: E402
from core.contracts.sampling import SamplingParams  # noqa: E402
from core.ipc.channels import ChannelReader  # noqa: E402
from core.ipc.config import IpcTunables  # noqa: E402
from core.ipc.stream import open_reader  # noqa: E402
from frontend.transport import Directory, StepChannel  # noqa: E402


def _load_non_special_vocab(model_dir: Path) -> tuple[list[int], int]:
    """Return ``(non_special_token_ids, n_routed_experts)`` from the HF checkpoint directory."""
    cfg_raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tcfg = cfg_raw.get("text_config", cfg_raw)
    vocab_size = int(tcfg["vocab_size"])
    n_routed = int(tcfg.get("n_routed_experts", 384))
    special: set[int] = set()
    tok_path = model_dir / "tokenizer.json"
    if tok_path.is_file():
        tok_raw = json.loads(tok_path.read_text(encoding="utf-8"))
        for item in tok_raw.get("added_tokens", []):
            if "id" in item:
                special.add(int(item["id"]))
    tok_cfg_path = model_dir / "tokenizer_config.json"
    if tok_cfg_path.is_file():
        tcr = json.loads(tok_cfg_path.read_text(encoding="utf-8"))
        for k in (tcr.get("added_tokens_decoder") or {}):
            try:
                special.add(int(k))
            except ValueError:
                pass
    vocab = [tid for tid in range(vocab_size) if tid not in special]
    return vocab, n_routed


_MAX_SHM_BATCH_BYTES = 3 << 20


async def _send_new(channel: Any, reqs: list[NewRequest]) -> bool:
    try:
        return (await channel.send(Requests(new=reqs))) is not None
    except ValueError:
        pass
    batches: list[list[NewRequest]] = []
    cur: list[NewRequest] = []
    cur_bytes = 0
    for r in reqs:
        sz = len(r.prompt.data) + 256
        if cur and cur_bytes + sz > _MAX_SHM_BATCH_BYTES:
            batches.append(cur)
            cur = []
            cur_bytes = 0
        cur.append(r)
        cur_bytes += sz
    if cur:
        batches.append(cur)
    futs = [channel.send(Requests(new=b)) for b in batches]
    results = await asyncio.gather(*futs)
    return all(x is not None for x in results)


def _make_prompt_columns(vocab: list[int], count: int, length: int, seed: int) -> list[Column]:
    """Generate ``count`` deterministic uniform-random non-special prompt columns of ``length``."""
    rng = random.Random(seed)
    # Build a small pool of distinct random sequences if count is large, or unique per request:
    # unique per request guarantees zero accidental content correlation across requests.
    return [Column.of("i64", rng.choices(vocab, k=length)) for _ in range(count)]


async def _open_step_channel(
    run_dir: Path,
) -> tuple[StepChannel, ChannelReader, zmq.Context[zmq.Socket[bytes]], dict[str, Any]]:
    supervisor = str(json.loads((run_dir / "control" / "control.json").read_text())["control"])
    cfg: dict[str, Any] = json.loads((run_dir / "config.json").read_text())
    sched_proc = msgspec.convert(cfg["frontend.scheduler"], ProcId)
    tun = IpcTunables(rpc_timeout_s=300.0, close_timeout_s=10.0)
    zctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
    shm_name = f"agx-pareto-{os.getpid()}-{secrets.token_hex(4)}"
    raw_reader, addr = open_reader(
        zctx, local=sched_proc.host, peer=sched_proc.host, bind_host="127.0.0.1",
        shm_name=shm_name, tun=tun, replaceable=True,
    )
    tokens_reader = ChannelReader(raw_reader, addr, sched_proc.host)
    directory = Directory(
        supervisor=supervisor, scheduler=sched_proc,
        me=ProcId(sched_proc.host, "pareto", 0),
    )
    channel = StepChannel(sched_proc.host, tokens_reader, directory, tun=tun)
    await channel.start()
    return channel, tokens_reader, zctx, cfg


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _step_window(run_dir: Path, first: int, last: int, dp_size: int) -> dict[str, Any]:
    """Actual prefill work and device timeline, including idle gaps caused by the host."""
    steps: dict[int, tuple[list[int], dict[int, tuple[float, float, float]]]] = {}
    retired_at: dict[int, float] = {}
    for path in run_dir.glob("*/snapshots/.run/sched-*/log"):
        for line in path.read_text().splitlines():
            if line.startswith("DSBENCH_STEP "):
                _, epoch, stamp, payload = line.split(" ", 3)
                tokens, stats = payload.split("] ", 1)
                steps[int(epoch)] = ast.literal_eval(tokens + "]"), ast.literal_eval(stats)
                retired_at[int(epoch)] = float(stamp)
    missing = set(range(first, last + 1)) - steps.keys()
    if missing:
        raise RuntimeError(f"missing step records {sorted(missing)}; apply dsbench_step_probe.py before serving")
    window = [steps[e] for e in range(first + 1, last + 1)]
    if any(len(tokens) != dp_size or set(stats) != set(range(dp_size)) for tokens, stats in window):
        raise RuntimeError("native step timing requires one worker and timing record per DP domain")
    per_rank_tokens = [sum(tokens[r] for tokens, _ in window) for r in range(dp_size)]
    per_rank_seconds = [sum(stats[r][1] + stats[r][2] for _, stats in window) / 1e6 for r in range(dp_size)]
    return {
        "tokens_per_gpu": sum(per_rank_tokens) / dp_size,
        "seconds": max(per_rank_seconds),
        "scheduler_retirement_seconds": retired_at[last] - retired_at[first],
        "tokens_by_step": [tokens for tokens, _ in window],
        "device_us_by_step": [[stats[r][1] for r in range(dp_size)] for _, stats in window],
        "idle_us_by_step": [[stats[r][2] for r in range(dp_size)] for _, stats in window],
    }


async def _run_prefill_point(
    channel: StepChannel,
    vocab: list[int],
    *,
    dp_size: int,
    ep_size: int,
    n_routed: int,
    extra_experts: int,
    model_name: str,
    context_length: int,
    prefill_batch_size: int,
    result_jsonl: Path,
    non_cached_tokens: int,
    yoco_clip: int,
    run_dir: Path,
    seed_base: int = 10_000,
    measure_steps: int = 12,
    tail_steps: int = 1,
    step_stats: bool = False,
) -> None:
    if context_length % prefill_batch_size != 0:
        raise ValueError(
            f"context_length ({context_length}) must be divisible by prefill_batch_size ({prefill_batch_size})"
        )
    if not 1 <= non_cached_tokens <= context_length:
        raise ValueError("non-cached tokens must be between 1 and the full context length")
    if max(non_cached_tokens, prefill_batch_size) % min(non_cached_tokens, prefill_batch_size):
        raise ValueError("the suffix and batch budget must divide one another")
    warmup_reqs = max(2, -(-(6 * prefill_batch_size) // non_cached_tokens))
    measure_reqs = max(2, -(-(measure_steps * prefill_batch_size) // non_cached_tokens))
    tail_reqs = max(2, -(-(tail_steps * prefill_batch_size) // non_cached_tokens))
    reqs_per_rank = warmup_reqs + measure_reqs + tail_reqs
    total_reqs = dp_size * reqs_per_rank
    sampling = SamplingParams(0.3, 0.95, -1, 7)
    uid_base = secrets.randbits(30) << 32
    prefix = context_length - non_cached_tokens
    if prefix:
        rng = random.Random(seed_base + prefill_batch_size)
        # Distinct prefixes give each DP rank its own retained state. Unique suffixes
        # prevent the online cache from making later measured requests shorter than intended.
        bases = [rng.choices(vocab, k=prefix) for _ in range(dp_size)]
        priming = [NewRequest(uid_base - rank - 1, Column.of("i64", base + rng.choices(vocab, k=max(1, yoco_clip))),
                              sampling, 1, []) for rank, base in enumerate(bases)]
        pending = {r.request for r in priming}
        if not await _send_new(channel, priming):
            raise RuntimeError("scheduler unreachable during prefix priming")
        while pending:
            _, event = await channel.receive()
            if event.errors:
                raise RuntimeError(str(event.errors))
            pending.difference_update(event.finished)
        cols = [Column.of("i64", bases[i % dp_size] + rng.choices(vocab, k=non_cached_tokens))
                for i in range(total_reqs)]
    else:
        cols = _make_prompt_columns(vocab, total_reqs, context_length, seed=seed_base + prefill_batch_size)
    expected_uids = {uid_base + i for i in range(total_reqs)}
    finished_uids: set[int] = set()
    errors: list[str] = []

    t_start: float | None = None
    t_end: float | None = None
    epoch_start: int | None = None
    epoch_end: int | None = None
    wall_start = wall_end = 0.0
    completion_events: list[tuple[int, float]] = []
    done_ev = asyncio.Event()

    warmup_target = dp_size * warmup_reqs
    measure_target = dp_size * (warmup_reqs + measure_reqs)

    async def receiver() -> None:
        nonlocal t_start, t_end, epoch_start, epoch_end, wall_start, wall_end
        while not done_ev.is_set():
            epoch, events = await channel.receive()
            now = time.perf_counter()
            previous_finished = len(finished_uids)
            for uid, err in events.errors.items():
                if int(uid) in expected_uids:
                    errors.append(f"uid={uid}: {err}")
                    finished_uids.add(int(uid))
            for uid in events.finished:
                iuid = int(uid)
                if iuid in expected_uids and iuid not in finished_uids:
                    finished_uids.add(iuid)
                    cnt = len(finished_uids)
                    if cnt == warmup_target and t_start is None:
                        t_start = now
                        wall_start = time.time()
                        epoch_start = epoch
                    elif cnt == measure_target and t_end is None:
                        t_end = now
                        wall_end = time.time()
                        epoch_end = epoch
            if len(finished_uids) > previous_finished:
                completion_events.append((epoch, now))
            if len(finished_uids) >= total_reqs:
                done_ev.set()

    recv_task = asyncio.create_task(receiver())
    try:
        for base in range(0, total_reqs, dp_size):
            quad = [
                NewRequest(uid_base + base + r, cols[base + r], sampling, 1, [])
                for r in range(dp_size)
            ]
            if not await _send_new(channel, quad):
                raise RuntimeError("scheduler unreachable while sending prefill quad")
        await done_ev.wait()
    finally:
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass

    if errors:
        raise RuntimeError(f"prefill errors ({len(errors)}): {errors[0]}")
    assert t_start is not None and t_end is not None
    assert epoch_start is not None and epoch_end is not None

    if prefix:
        admitted = {}
        for path in run_dir.glob("*/snapshots/.run/sched-*/log"):
            for line in path.read_text().splitlines():
                if line.startswith("AGENTX_ADMIT "):
                    _, rid, cached = line.split()
                    if int(rid) in expected_uids:
                        admitted[int(rid)] = int(cached)
        if len(admitted) != total_reqs or any(n != prefix for n in admitted.values()):
            raise RuntimeError(f"expected {total_reqs} admissions at {prefix} cached tokens, got "
                               f"{len(admitted)} admissions, prefixes {sorted(set(admitted.values()))}")

    expected_steps = measure_reqs * non_cached_tokens // prefill_batch_size
    observed_epochs = epoch_end - epoch_start
    window_steps = observed_epochs if observed_epochs > 0 else expected_steps
    window_tokens: int | float = measure_reqs * non_cached_tokens
    window_seconds = t_end - t_start
    completion_seconds = window_seconds
    completion_tokens = window_tokens
    native = _step_window(run_dir, epoch_start, epoch_end, dp_size) if step_stats else None
    if native is not None:
        window_tokens, window_seconds = native["tokens_per_gpu"], native["scheduler_retirement_seconds"]
    tps_per_gpu = window_tokens / window_seconds
    step_ms = window_seconds / window_steps * 1000.0

    record: dict[str, Any] = {
        "model_name": model_name,
        "ep_size": ep_size,
        "dp_size": dp_size,
        "cp": 1,
        "context_length": context_length,
        "non_cached_tokens": non_cached_tokens,
        "cached_tokens": prefix,
        "prefill_batch_size": prefill_batch_size,
        "seed": seed_base + prefill_batch_size,
        "decode_num_requests": 0,
        "yoco_clip": yoco_clip,
        "nextn": 0,
        "mtp_varlen": 0,
        "num_routed_experts": n_routed,
        "num_extra_experts": extra_experts,
        "prefill_num_requests": reqs_per_rank,
        "batch_num_requests": reqs_per_rank,
        "prefill_window_steps": window_steps,
        "prefill_expected_steps": expected_steps,
        "prefill_window_tokens": window_tokens,
        "prefill_window_seconds": window_seconds,
        "prefill_timing": "scheduler_retirement" if step_stats else "completion_events",
        "prefill_first_epoch": epoch_start,
        "prefill_last_epoch": epoch_end,
        "prefill_completion_window_seconds": completion_seconds,
        "prefill_completion_tokens_per_s_per_gpu": completion_tokens / completion_seconds,
        "prefill_window_start_unix": wall_start,
        "prefill_window_end_unix": wall_end,
        "prefill_completion_intervals": [
            {"epochs": e1 - e0, "ms": (t1 - t0) * 1000}
            for (e0, t0), (e1, t1) in itertools.pairwise(completion_events)
            if t0 >= t_start and t1 <= t_end
        ],
        "prefill_tokens_per_s_per_gpu": tps_per_gpu,
    }
    if native is not None:
        record["prefill_step_stats"] = native
        record["prefill_device_tokens_per_s_per_gpu"] = window_tokens / native["seconds"]
    _append_jsonl(result_jsonl, record)
    print(
        f"PARETO-AGX prefill batch_size={prefill_batch_size}: "
        f"step={step_ms:.2f} ms ({window_steps} steps, expected {expected_steps}), "
        f"throughput={tps_per_gpu:,.1f} tokens/s/GPU (4-GPU total={tps_per_gpu * dp_size:,.0f} tok/s)",
        flush=True,
    )


async def _run_decode_point(
    channel: StepChannel,
    vocab: list[int],
    *,
    dp_size: int,
    ep_size: int,
    n_routed: int,
    extra_experts: int,
    model_name: str,
    context_length: int,
    prefill_batch_size: int,
    decode_num_requests: int,
    warmup_steps: int,
    measure_steps: int,
    result_jsonl: Path,
) -> None:
    total_reqs = dp_size * decode_num_requests
    max_tokens = 1 + warmup_steps + measure_steps + 20
    cols = _make_prompt_columns(vocab, total_reqs, context_length, seed=20_000 + decode_num_requests)
    sampling = SamplingParams(0.3, 0.95, -1, 7)
    uid_base = secrets.randbits(30) << 32
    expected_uids = {uid_base + i for i in range(total_reqs)}
    seen_first: set[int] = set()
    finished_uids: set[int] = set()
    errors: list[str] = []

    pure_decode_seen = 0
    stamps: list[tuple[int, float]] = []
    done_ev = asyncio.Event()

    async def receiver() -> None:
        nonlocal pure_decode_seen
        while not done_ev.is_set():
            epoch, events = await channel.receive()
            now = time.perf_counter()
            req_ids = [int(x) for x in events.requests.values()]
            all_were_ready = len(seen_first) == total_reqs

            for iuid in req_ids:
                if iuid in expected_uids:
                    seen_first.add(iuid)

            if all_were_ready and len(req_ids) == total_reqs and not finished_uids:
                pure_decode_seen += 1
                if pure_decode_seen > warmup_steps and len(stamps) < measure_steps + 1:
                    stamps.append((epoch, now))

            for uid, err in events.errors.items():
                if int(uid) in expected_uids:
                    errors.append(f"uid={uid}: {err}")
                    finished_uids.add(int(uid))
            for uid in events.finished:
                if int(uid) in expected_uids:
                    finished_uids.add(int(uid))
            if len(finished_uids) >= total_reqs:
                done_ev.set()

    recv_task = asyncio.create_task(receiver())
    try:
        for base in range(0, total_reqs, dp_size):
            quad = [
                NewRequest(uid_base + base + r, cols[base + r], sampling, max_tokens, [])
                for r in range(dp_size)
            ]
            sent = await channel.send(Requests(new=quad))
            if sent is None:
                raise RuntimeError("scheduler unreachable while sending decode quad")
        await done_ev.wait()
    finally:
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass

    if errors:
        raise RuntimeError(f"decode errors ({len(errors)}): {errors[0]}")
    if len(stamps) < measure_steps + 1:
        raise RuntimeError(
            f"decode_num_requests={decode_num_requests}: collected {len(stamps)}/{measure_steps + 1} "
            f"decode stamps (pure_decode_seen={pure_decode_seen})"
        )

    dts = [t2 - t1 for (_, t1), (_, t2) in itertools.pairwise(stamps)]
    avg_sps = len(dts) / sum(dts)
    max_step_ms = max(dts) * 1000.0
    tps_per_gpu = avg_sps * decode_num_requests

    record: dict[str, Any] = {
        "model_name": model_name,
        "ep_size": ep_size,
        "dp_size": dp_size,
        "cp": 1,
        "context_length": context_length,
        "non_cached_tokens": context_length,
        "cached_tokens": 0,
        "prefill_batch_size": prefill_batch_size,
        "decode_num_requests": decode_num_requests,
        "yoco_clip": 0,
        "nextn": 0,
        "mtp_varlen": 0,
        "num_routed_experts": n_routed,
        "num_extra_experts": extra_experts,
        "batch_num_requests": decode_num_requests,
        "max_tokens": max_tokens,
        "decode_window_steps": len(dts),
        "decode_avg_sps": avg_sps,
        "decode_max_step_ms": max_step_ms,
        "decode_tokens_per_s_per_gpu": tps_per_gpu,
    }
    _append_jsonl(result_jsonl, record)
    print(
        f"PARETO-AGX decode reqs/GPU={decode_num_requests} (4-GPU={total_reqs}): "
        f"avg_SPS={avg_sps:.2f} tok/s/user ({1000.0 / avg_sps:.2f} ms/step, max={max_step_ms:.2f} ms), "
        f"throughput={tps_per_gpu:,.1f} tokens/s/GPU (4-GPU total={tps_per_gpu * dp_size:,.0f} tok/s)",
        flush=True,
    )


async def _amain(args: argparse.Namespace) -> None:
    vocab, n_routed = _load_non_special_vocab(args.model.resolve())
    channel, tokens_reader, zctx, cfg = await _open_step_channel(args.run.resolve())
    dp_size = int(cfg.get("sched.domains", 4))
    ep_size = int(cfg.get("comm.size", dp_size))
    extra_experts = int(cfg.get("eplb.extra_experts", 0)) if ep_size > 1 else 0
    if ep_size > 1 and (extra_experts or int(cfg.get("eplb.interval", 0)) > 0):
        extra_experts = -(-(n_routed + extra_experts) // ep_size) * ep_size - n_routed
    prefill_budget = int(cfg.get("sched.token_budget", 32768))
    try:
        if args.mode == "prefill":
            if args.prefill_batch_size is None:
                raise SystemExit("--mode prefill requires --prefill-batch-size")
            await _run_prefill_point(
                channel, vocab,
                dp_size=dp_size, ep_size=ep_size, n_routed=n_routed, extra_experts=extra_experts,
                model_name=args.model_name, context_length=args.context_length,
                prefill_batch_size=args.prefill_batch_size, result_jsonl=args.result_jsonl,
                non_cached_tokens=args.non_cached_tokens or args.context_length,
                yoco_clip=args.yoco_clip, run_dir=args.run,
                seed_base=args.seed_base,
                measure_steps=args.prefill_measure_steps, tail_steps=args.prefill_tail_steps,
                step_stats=args.step_stats,
            )
        else:
            if not args.decode_num_requests:
                raise SystemExit("--mode decode requires --decode-num-requests")
            for idx, d in enumerate(args.decode_num_requests):
                if idx > 0:
                    await asyncio.sleep(args.pause)
                await _run_decode_point(
                    channel, vocab,
                    dp_size=dp_size, ep_size=ep_size, n_routed=n_routed, extra_experts=extra_experts,
                    model_name=args.model_name, context_length=args.context_length,
                    prefill_batch_size=prefill_budget, decode_num_requests=d,
                    warmup_steps=args.decode_warmup_steps, measure_steps=args.decode_measure_steps,
                    result_jsonl=args.result_jsonl,
                )
    finally:
        await channel.close()
        tokens_reader.close()
        zctx.destroy(linger=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="Running agxinfer run directory")
    p.add_argument("--model", type=Path, required=True, help="Local HF checkpoint directory (for vocab/config)")
    p.add_argument("--model-name", default="deepseek-ai/DeepSeek-V4.1-Flash-1017")
    p.add_argument("--mode", required=True, choices=["prefill", "decode"])
    p.add_argument("--context-length", type=int, default=32768)
    p.add_argument("--non-cached-tokens", type=int, default=None,
                   help="Shared-prefix workload; requires prefix caching and the admission-log probe.")
    p.add_argument("--yoco-clip", type=int, default=0, help="The measured engine's bounded replay length.")
    p.add_argument("--prefill-batch-size", type=int, default=None)
    p.add_argument("--seed-base", type=int, default=10_000,
                   help="Prefill RNG seed is this plus batch size; change it for uncached repeat suffixes.")
    p.add_argument("--prefill-measure-steps", type=int, default=12)
    p.add_argument("--prefill-tail-steps", type=int, default=1,
                   help="Extra queued work after the measurement target; use 4 with overlap to avoid draining.")
    p.add_argument("--step-stats", action="store_true",
                   help="Count actual tokens between scheduler retirements and record device time; needs dsbench_step_probe.py.")
    p.add_argument("--decode-num-requests", type=int, nargs="*", default=[])
    p.add_argument("--decode-warmup-steps", type=int, default=110)
    p.add_argument("--decode-measure-steps", type=int, default=60)
    p.add_argument("--pause", type=float, default=2.0)
    p.add_argument("--result-jsonl", type=Path, required=True)
    args = p.parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
