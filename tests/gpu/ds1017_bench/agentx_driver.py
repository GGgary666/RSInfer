"""SemiAnalysis AgentX multi-turn + subagent trace replay benchmark for DeepInfer and agxinfer.

Replays real Claude Code agentic session structures from ``semianalysisai/cc-traces-weka-062126-256k``
on ``DP4 + EP4`` with both engines reusing prefixes reconstructed from the published session-local block hashes:

1. ``prepare`` builds deterministic ``agentx_dev.json`` (16 stratified sessions for fast iteration)
   and ``agentx_eval.json`` (48 stratified sessions for full evaluation) under ``--max-context``
   (default ``131072``), sampling mid-trajectory slices (``t* in [0.25, 0.75]``, ``seed = 42``) as in
   ``agentx-harness --scenario agentx``, preserving sequential multi-turn dependencies, recorded
   ``think_time`` gaps (capped by ``--idle-cap-s``), and concurrent ``SPAWN`` / ``SPAWN_JOIN``
   subagent fan-outs, backed by a real-code token corpus (``corpus_ids.json``) so both engines run
   identical token IDs and real model expert routing (``force_random = False``).
2. ``deepinfer`` drives DeepInfer in-process across all DP/EP ranks using a shared atomic session
   queue in ``/dev/shm`` so ``C`` concurrent session lanes dynamically pull sessions just like a
   real serving load balancer.
3. ``agx-direct`` drives a running ``agxinfer`` instance over ``StepChannel`` (pre-tokenized shared
   memory IPC, matching DeepInfer's ``_add_request`` + ``step()`` boundary) on the exact same
   session suite and concurrency levels.
"""
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import random
import secrets
import statistics
import struct
import sys
import sysconfig

import numpy as np
import time
from pathlib import Path
from typing import Any

BOS = 0


# ---------------------------------------------------------------------------
# Metrics (shared across DeepInfer and agxinfer)
# ---------------------------------------------------------------------------

def _pct(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    idx = min(len(ordered) - 1, max(0, int(q * len(ordered))))
    return ordered[idx]


def _spread(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    s = sorted(values)
    return {
        "n": len(s),
        "mean": statistics.fmean(s),
        "p10": _pct(s, 0.10),
        "p50": _pct(s, 0.50),
        "p90": _pct(s, 0.90),
        "p99": _pct(s, 0.99),
        "max": s[-1],
    }


def summarize_agentx(
    records: list[dict[str, Any]],
    wall_s: float,
    num_gpus: int,
    concurrency: int,
    sessions_total: int,
) -> dict[str, Any]:
    good = [r for r in records if r.get("complete") and r.get("times")]
    in_toks = sum(int(r["prompt_tokens"]) for r in good)
    out_toks = sum(len(r["times"]) for r in good)
    tot_toks = in_toks + out_toks

    ttfts = [float(r["times"][0]) - float(r["sent"]) for r in good]
    e2es = [float(r["times"][-1]) - float(r["sent"]) for r in good]
    tpots: list[float] = []
    tps_user: list[float] = []
    e2e_tps_user: list[float] = []
    gaps: list[float] = []
    for r in good:
        ts = r["times"]
        e2e = max(1e-6, float(ts[-1]) - float(r["sent"]))
        e2e_tps_user.append(len(ts) / e2e)
        if len(ts) >= 2:
            dec_s = max(1e-6, float(ts[-1]) - float(ts[0]))
            tpot_ms = 1000.0 * dec_s / (len(ts) - 1)
            tpots.append(tpot_ms)
            tps_user.append((len(ts) - 1) / dec_s)
            gaps.extend(1000.0 * (float(b) - float(a)) for a, b in zip(ts, ts[1:]))

    return {
        "cached_input_tokens": sum(r["cache_hit"] for r in good) if good and "cache_hit" in good[0] else None,
        "concurrency": concurrency,
        "num_gpus": num_gpus,
        "sessions_total": sessions_total,
        "requests_total": len(records),
        "requests_completed": len(good),
        "subagent_requests": sum(1 for r in good if r.get("is_subagent")),
        "wall_s": wall_s,
        "input_tokens": in_toks,
        "output_tokens": out_toks,
        "total_tokens": tot_toks,
        "total_tok_s": tot_toks / wall_s if wall_s > 0 else 0.0,
        "total_tok_s_per_gpu": tot_toks / wall_s / max(num_gpus, 1) if wall_s > 0 else 0.0,
        "input_tok_s": in_toks / wall_s if wall_s > 0 else 0.0,
        "input_tok_s_per_gpu": in_toks / wall_s / max(num_gpus, 1) if wall_s > 0 else 0.0,
        "output_tok_s": out_toks / wall_s if wall_s > 0 else 0.0,
        "output_tok_s_per_gpu": out_toks / wall_s / max(num_gpus, 1) if wall_s > 0 else 0.0,
        "decode_tps_per_user": _spread(tps_user),
        "e2e_tps_per_user": _spread(e2e_tps_user),
        "ttft_s": _spread(ttfts),
        "tpot_ms": _spread(tpots),
        "gap_ms": _spread(gaps),
        "e2e_s": _spread(e2es),
    }


def format_summary_line(engine: str, s: dict[str, Any]) -> str:
    return (
        f"AGENTX {engine} c={s['concurrency']}: wall={s['wall_s']:.1f}s, "
        f"total={s['total_tok_s_per_gpu']:,.1f} tok/s/GPU (4GPU={s['total_tok_s']:,.0f}), "
        f"out={s['output_tok_s_per_gpu']:,.1f} tok/s/GPU (4GPU={s['output_tok_s']:,.1f}), "
        f"interactivity p50={s['decode_tps_per_user']['p50']:.1f} p10={s['decode_tps_per_user']['p10']:.1f} tok/s/u, "
        f"ttft p50={s['ttft_s']['p50']:.3f}s p90={s['ttft_s']['p90']:.3f}s, "
        f"tpot p50={s['tpot_ms']['p50']:.2f}ms, gap p99={s['gap_ms']['p99']:.1f}ms, "
        f"done={s['requests_completed']}/{s['requests_total']}"
    )


# ---------------------------------------------------------------------------
# 1. Dataset preparation
# ---------------------------------------------------------------------------

def _build_code_corpus(repo_root: Path, min_chars: int = 24_000_000) -> str:
    roots = [Path(sysconfig.get_paths()["stdlib"]), repo_root / "engine", repo_root / "tests"]
    chunks: list[str] = []
    size = 0
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts or ".venv" in path.parts:
                continue
            try:
                txt = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if len(txt) < 1500:
                continue
            chunks.append(f"# ==== {path.name} ====\n{txt}\n")
            size += len(txt)
            if size >= min_chars:
                break
        if size >= min_chars:
            break
    body = "".join(chunks)
    if len(body) < min_chars:
        body = body * ((min_chars // max(len(body), 1)) + 1)
    return body


def _slice_trajectory(
    trace: dict[str, Any],
    rng: random.Random,
    *,
    max_context: int,
    max_main_turns: int,
    max_subagent_turns: int,
    idle_cap_s: float,
    corpus_len: int,
    req_counter: list[int],
) -> dict[str, Any] | None:
    raw_reqs = trace.get("requests", [])
    main_indices = [idx for idx, r in enumerate(raw_reqs) if r.get("type") != "subagent"]
    if not main_indices:
        return None

    if len(main_indices) > max_main_turns:
        max_start = len(main_indices) - max_main_turns
        start_ord = int(rng.uniform(0.25, 0.75) * max_start)
        chosen_ord = main_indices[start_ord:start_ord + max_main_turns]
        lo, hi = chosen_ord[0], chosen_ord[-1]
        window = raw_reqs[lo:hi + 1]
    else:
        window = list(raw_reqs)

    def make_turn(r: dict[str, Any], first_in_chain: bool, is_sub: bool) -> dict[str, Any] | None:
        in_tok = max(16, int(r["in"]))
        out_tok = max(1, int(r["out"]))
        if in_tok + out_tok > max_context:
            return None
        rid = req_counter[0]
        req_counter[0] += 1
        raw_think = 0.0 if first_in_chain else max(0.0, float(r.get("think_time") or 0.0))
        think_s = min(raw_think, idle_cap_s)
        span = max(1, corpus_len - in_tok - 16)
        offset = (rid * 104_729 + in_tok * 7_919) % span
        return {
            "req_id": rid,
            "in": in_tok,
            "out": out_tok,
            "think_s": round(think_s, 4),
            "offset": offset,
            "hash_ids": r["hash_ids"],
            "is_subagent": is_sub,
        }

    steps: list[dict[str, Any]] = []
    pending_subs: list[list[dict[str, Any]]] = []
    first_main = True
    total_in = 0
    total_out = 0
    total_reqs = 0

    for item in window:
        if item.get("type") == "subagent":
            chain: list[dict[str, Any]] = []
            for j, ir in enumerate(item.get("requests", [])[:max_subagent_turns]):
                t = make_turn(ir, first_in_chain=(j == 0), is_sub=True)
                if t is not None:
                    chain.append(t)
                    total_in += t["in"]
                    total_out += t["out"]
                    total_reqs += 1
            if chain:
                pending_subs.append(chain)
        else:
            if pending_subs:
                steps.append({"kind": "subagents", "chains": pending_subs})
                pending_subs = []
            t = make_turn(item, first_in_chain=first_main, is_sub=False)
            if t is not None:
                steps.append({"kind": "turn", "turn": t})
                first_main = False
                total_in += t["in"]
                total_out += t["out"]
                total_reqs += 1

    if pending_subs:
        steps.append({"kind": "subagents", "chains": pending_subs})

    if not steps or total_reqs == 0:
        return None
    return {
        "trace_id": trace["id"],
        "total_in": total_in,
        "total_out": total_out,
        "total_reqs": total_reqs,
        "has_subagents": any(s["kind"] == "subagents" for s in steps),
        "steps": steps,
    }


def _stratified_pick(sessions: list[dict[str, Any]], k: int, seed: int) -> list[dict[str, Any]]:
    ordered = sorted(sessions, key=lambda s: (s["total_in"] + s["total_out"], s["trace_id"]))
    if len(ordered) <= k:
        picked = list(ordered)
    else:
        picked = [ordered[int(round(i * (len(ordered) - 1) / (k - 1)))] for i in range(k)]
    # Snake-order shuffle with fixed seed so early & late sessions have balanced work
    rng = random.Random(seed)
    # Group into quartiles and interleave so concurrency lanes see uniform mix throughout the run
    q_size = max(1, len(picked) // 4)
    buckets = [picked[i:i + q_size] for i in range(0, len(picked), q_size)]
    for b in buckets:
        rng.shuffle(b)
    interleaved: list[dict[str, Any]] = []
    max_len = max(len(b) for b in buckets)
    for idx in range(max_len):
        for b in buckets:
            if idx < len(b):
                interleaved.append(b[idx])
    return interleaved[:k]


def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "engine").is_dir():
            return parent
    return Path(__file__).resolve().parent


def cmd_prepare(args: argparse.Namespace) -> None:
    import copy
    from transformers import AutoTokenizer

    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    from agentx_prepare_bank import materialize_session_bank

    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus_ids.json"
    if corpus_path.is_file() and not args.rebuild_corpus:
        corpus_ids: list[int] = json.loads(corpus_path.read_text())
        print(f"Loaded existing corpus: {len(corpus_ids):,} tokens from {corpus_path}", flush=True)
    else:
        tok: Any = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, use_fast=True)
        repo_root = _find_repo_root()
        text = _build_code_corpus(repo_root)
        corpus_ids = []
        for start in range(0, len(text), 200_000):
            corpus_ids.extend(tok.encode(text[start:start + 200_000], add_special_tokens=False))
        corpus_path.write_text(json.dumps(corpus_ids))
        print(f"Saved real-code token corpus: {len(corpus_ids):,} tokens -> {corpus_path}", flush=True)

    if len(corpus_ids) < args.max_context + 1024:
        raise SystemExit(f"Corpus ({len(corpus_ids)}) shorter than max_context ({args.max_context})")

    raw_traces: list[dict[str, Any]] = []
    with args.traces_jsonl.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            tr = json.loads(line)
            peak = 0
            for r in tr.get("requests", []):
                if r.get("type") == "subagent":
                    for ir in r.get("requests", []):
                        peak = max(peak, int(ir["in"]) + int(ir["out"]))
                else:
                    peak = max(peak, int(r["in"]) + int(r["out"]))
            if 0 < peak <= args.max_context:
                raw_traces.append(tr)

    rng = random.Random(args.seed)
    req_counter = [1]
    all_sessions: list[dict[str, Any]] = []
    for tr in raw_traces:
        sess = _slice_trajectory(
            tr, rng,
            max_context=args.max_context,
            max_main_turns=args.max_main_turns,
            max_subagent_turns=args.max_subagent_turns,
            idle_cap_s=args.idle_cap_s,
            corpus_len=len(corpus_ids),
            req_counter=req_counter,
        )
        if sess is not None:
            all_sessions.append(sess)

    for tag, count in (("dev", args.dev_sessions), ("eval", args.eval_sessions)):
        picked = copy.deepcopy(_stratified_pick(all_sessions, count, seed=args.seed + len(tag)))
        for sid, s in enumerate(picked):
            s["session_id"] = sid
        payload = {
            "source": "semianalysisai/cc-traces-weka-062126-256k",
            "preset": tag,
            "seed": args.seed,
            "max_context": args.max_context,
            "max_main_turns": args.max_main_turns,
            "max_subagent_turns": args.max_subagent_turns,
            "idle_cap_s": args.idle_cap_s,
            "sessions": picked,
        }
        target_prefix = out_dir / f"agentx_{tag}"
        bank_tokens = materialize_session_bank(payload, corpus_ids, target_prefix)
        tot_reqs = sum(s["total_reqs"] for s in picked)
        tot_in = sum(s["total_in"] for s in picked)
        tot_out = sum(s["total_out"] for s in picked)
        n_sub = sum(1 for s in picked if s["has_subagents"])
        print(
            f"Prepared agentx_{tag}.{{json,i32}}: {len(picked)} sessions ({n_sub} with subagents), "
            f"{tot_reqs} requests, {tot_in / 1e6:.2f}M input tokens ({bank_tokens:,} banked), "
            f"{tot_out / 1e3:.1f}K output tokens",
            flush=True,
        )


def _prompt_tokens(bank: Any, offset: int, length: int) -> Any:
    return bank[offset:offset + length]


# ---------------------------------------------------------------------------
# Shared lane state machine (used by DeepInfer synchronous step loop)
# ---------------------------------------------------------------------------

class _AtomicCounter:
    """Cross-process atomic session counter backed by an 8-byte file in ``/dev/shm``."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def reset(self) -> None:
        self.path.write_bytes(struct.pack("<q", 0))

    def next(self) -> int:
        with self.path.open("r+b") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                val = struct.unpack("<q", f.read(8))[0]
                f.seek(0)
                f.write(struct.pack("<q", val + 1))
                f.flush()
                return int(val)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class _LaneRunner:
    """Deterministic state machine for one AgentX lane with sequential turns & SPAWN_JOIN subagents."""

    def __init__(self, lane_id: int, sessions: list[dict[str, Any]], counter: _AtomicCounter) -> None:
        self.lane_id = lane_id
        self.sessions = sessions
        self.counter = counter
        self.session: dict[str, Any] | None = None
        self.step_idx = 0
        self.due_turns: list[tuple[float, dict[str, Any], tuple[int, int] | None]] = []
        self.in_flight = 0
        self.active_chains: dict[int, tuple[list[dict[str, Any]], int]] = {}
        self.exhausted = False

    def _claim_next_session(self, now: float) -> bool:
        idx = self.counter.next()
        if idx >= len(self.sessions):
            self.session = None
            self.exhausted = True
            return False
        self.session = self.sessions[idx]
        self.step_idx = 0
        self._enter_step(now)
        return True

    def _enter_step(self, now: float) -> None:
        while self.session is not None and self.step_idx < len(self.session["steps"]):
            step = self.session["steps"][self.step_idx]
            if step["kind"] == "turn":
                t = step["turn"]
                self.due_turns.append((now + float(t["think_s"]), t, None))
                return
            chains = [c for c in step["chains"] if c]
            if not chains:
                self.step_idx += 1
                continue
            self.active_chains = {cid: (c, 0) for cid, c in enumerate(chains)}
            for cid, (c, _) in self.active_chains.items():
                t0 = c[0]
                self.due_turns.append((now + float(t0["think_s"]), t0, (cid, 0)))
            return
        # Current session completed — immediately claim next session
        self._claim_next_session(now)

    def start(self, now: float) -> None:
        self._claim_next_session(now)

    def pop_ready(self, now: float) -> list[tuple[dict[str, Any], tuple[int, int] | None]]:
        if not self.due_turns:
            return []
        ready: list[tuple[dict[str, Any], tuple[int, int] | None]] = []
        still_due: list[tuple[float, dict[str, Any], tuple[int, int] | None]] = []
        for fire_at, turn, chain_pos in self.due_turns:
            if fire_at <= now:
                ready.append((turn, chain_pos))
                self.in_flight += 1
            else:
                still_due.append((fire_at, turn, chain_pos))
        self.due_turns = still_due
        return ready

    def on_turn_finished(self, now: float, chain_pos: tuple[int, int] | None) -> None:
        self.in_flight -= 1
        if self.session is None:
            return
        if chain_pos is None:
            self.step_idx += 1
            self._enter_step(now)
            return
        cid, pos = chain_pos
        chain, _ = self.active_chains[cid]
        nxt = pos + 1
        if nxt < len(chain):
            self.active_chains[cid] = (chain, nxt)
            tn = chain[nxt]
            self.due_turns.append((now + float(tn["think_s"]), tn, (cid, nxt)))
        else:
            del self.active_chains[cid]
            if not self.active_chains and self.in_flight == 0 and not self.due_turns:
                self.step_idx += 1
                self._enter_step(now)

    @property
    def active(self) -> bool:
        return not self.exhausted or self.in_flight > 0 or bool(self.due_turns)


# ---------------------------------------------------------------------------
# 2. DeepInfer driver
# ---------------------------------------------------------------------------

_di_args: argparse.Namespace | None = None


def _di_rank_file(out: Path, rank: int) -> Path:
    return Path(f"{out}.rank{rank}.json")


def _di_counter_file(out: Path, conc: int) -> Path:
    return Path(f"/dev/shm/agx_agentx_{out.stem}_c{conc}.cnt")


def _di_worker(dp_size: int, dp_rank: int) -> None:
    args = _di_args if _di_args is not None else _build_parser().parse_args()
    yoco_clip = int(getattr(args, "yoco_clip", 140))
    if yoco_clip > 0:
        os.environ["DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE"] = "compact"
        os.environ["DEEPINFER_PREFILL_STORE_EXTRA_STATE"] = "1"
        os.environ["DEEPINFER_YOCO_PREFILL_KEEP_TOKENS"] = str(yoco_clip)
    else:
        os.environ["DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE"] = "off"
        os.environ["DEEPINFER_YOCO_PREFILL_KEEP_TOKENS"] = "0"
    os.environ["DEEPINFER_USE_NEXTN_DECODE"] = "0"

    sys.path.insert(0, str(Path(args.deepinfer) / "examples"))
    from _config import create_llm, setup_common_env  # noqa: E402

    setup_common_env(model_name=args.model, sampler_force_topk=0, set_override_experts=args.eplb,
                     enable_eplb=args.eplb)
    os.environ["DEEPINFER_MAX_BACKTRACKING_TOKENS"] = "1"
    if args.force_random:
        os.environ["DEEPINFER_FORCE_LOAD_BALANCE"] = "1"

    ep_size = len((os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(","))
    llm = create_llm(
        args.model, dp_rank, ep_size=ep_size, cp_size=1, data_parallel_size=dp_size,
        max_num_seqs=args.max_seqs, max_num_batched_tokens=args.token_budget,
        max_model_len=args.max_model_len, gpu_memory_utilization=args.mem, lookahead_tokens=0,
    )

    import numpy as np
    import torch
    from deepinfer import envs
    from deepinfer._C.engine import RequestOutput
    from deepinfer.model_executor.parallel_utils.parallel_state import PG
    from deepinfer.sampling_params import SamplingParams

    suite = json.loads(args.suite.read_text())
    sessions: list[dict[str, Any]] = suite["sessions"]
    corpus_ids = np.memmap(args.corpus, dtype="<i4", mode="r")
    engine = llm.llm_engine
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    from di_prefix_directory import Directory
    cache = Directory(
        engine, Path(__file__).resolve().parent,
        max_context=args.max_model_len, reserve_slots=args.max_seqs, yoco_clip=yoco_clip,
    )

    if dp_rank == 0:
        print(
            f"AGENTX-DI init: ep={ep_size} dp={dp_size} max_len={args.max_model_len} "
            f"budget={args.token_budget} zos={envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING} "
            f"nextn={envs.DEEPINFER_USE_NEXTN_DECODE} force_random={envs.DEEPINFER_FORCE_LOAD_BALANCE}",
            flush=True,
        )

    sync_group = PG.cpu_data_parallel_group() if envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING else PG.data_parallel_group()
    sync_dev = "cpu" if envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING else "cuda"

    def any_rank_busy(local_running: bool, local_lanes_active: bool) -> tuple[bool, bool]:
        flags = torch.tensor([int(local_running), int(local_lanes_active)], dtype=torch.int64, device=sync_dev)
        if dp_size > 1:
            torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.MAX, group=sync_group)
        return bool(flags[0].item()), bool(flags[1].item())

    rank_results: list[dict[str, Any]] = []
    for conc in args.concurrencies:
        if conc % dp_size != 0:
            raise SystemExit(f"concurrency {conc} must be divisible by dp_size {dp_size}")
        cache.clear()
        cnt_file = _di_counter_file(args.out, conc)
        if dp_rank == 0:
            _AtomicCounter(cnt_file).reset()
        if ep_size > 1:
            torch.distributed.barrier(PG.data_parallel_group())

        counter = _AtomicCounter(cnt_file)
        lanes = [
            _LaneRunner(lane_id, sessions, counter)
            for lane_id in range(dp_rank, conc, dp_size)
        ]
        live: dict[str, tuple[dict[str, Any], _LaneRunner, tuple[int, int] | None, int]] = {}
        records: list[dict[str, Any]] = []

        def dispatch_ready() -> None:
            now = time.time()
            for lane in lanes:
                for turn, chain_pos in lane.pop_ready(now):
                    toks = _prompt_tokens(corpus_ids, int(turn["offset"]), int(turn["in"]))
                    gen = int(turn["out"])
                    rec: dict[str, Any] = {
                        "req_id": int(turn["req_id"]),
                        "lane": lane.lane_id,
                        "rank": dp_rank,
                        "sent": now,
                        "prompt_tokens": len(toks),
                        "expected_out": gen,
                        "is_subagent": bool(turn["is_subagent"]),
                        "times": [],
                        "complete": False,
                    }
                    sp = SamplingParams(max_tokens=gen, temperature=0.0, ignore_eos=True)
                    rid = str(next(llm.request_counter))
                    seq = cache.add(rid, toks, sp)
                    rec["cache_hit"] = int(seq.num_cached_tokens)
                    live[rid] = (rec, lane, chain_pos, gen)
                    records.append(rec)

        def consume(outputs: Any) -> None:
            now = time.time()
            for ptr in outputs:
                rid = RequestOutput.get_request_id(ptr)
                entry = live.get(rid)
                if entry is None:
                    continue
                rec, lane, chain_pos, gen = entry
                rec["times"].append(now)
                if RequestOutput.get_finished(ptr):
                    rec["complete"] = len(rec["times"]) == gen
                    del live[rid]
                    lane.on_turn_finished(now, chain_pos)

        if ep_size > 1:
            torch.distributed.barrier(PG.data_parallel_group())
        t0 = time.time()
        for lane in lanes:
            lane.start(t0)
        dispatch_ready()

        flushed_idle = False
        while True:
            dispatch_ready()
            local_lanes = any(l.active for l in lanes)
            any_running, any_lanes = any_rank_busy(engine.scheduler.has_unfinished_seqs(), local_lanes)
            if not any_running:
                if not flushed_idle:
                    consume(engine.flush_inflight_reqs())
                    flushed_idle = True
                    dispatch_ready()
                    local_lanes = any(l.active for l in lanes)
                    any_running, any_lanes = any_rank_busy(engine.scheduler.has_unfinished_seqs(), local_lanes)
                    if any_running:
                        flushed_idle = False
                        continue
                if not any_lanes:
                    break
                time.sleep(0.001)
                continue
            flushed_idle = False
            for _ in range(args.check):
                dispatch_ready()
                consume(engine.step())

        if ep_size > 1:
            torch.distributed.barrier(PG.data_parallel_group())
        t1 = time.time()
        rank_results.append({
            "concurrency": conc,
            "t0": t0,
            "t1": t1,
            "records": records,
        })
        if dp_rank == 0:
            cnt_file.unlink(missing_ok=True)
            print(f"AGENTX-DI rank 0 finished c={conc} in {t1 - t0:.1f}s (rank0 reqs={len(records)})", flush=True)
        time.sleep(args.pause)

    _di_rank_file(args.out, dp_rank).write_text(json.dumps(rank_results))
    llm.close()


def cmd_deepinfer(args: argparse.Namespace) -> None:
    global _di_args
    _di_args = args
    sys.path.insert(0, str(Path(args.deepinfer) / "examples"))
    from _launch import launch_dp

    ep_size = len((os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(","))
    suite = json.loads(args.suite.read_text())
    sessions_total = len(suite["sessions"])
    launch_dp(_di_worker, ep_size)

    per_rank = [json.loads(_di_rank_file(args.out, r).read_text()) for r in range(ep_size)]
    levels_out: list[dict[str, Any]] = []
    for parts in zip(*per_rank, strict=True):
        conc = int(parts[0]["concurrency"])
        t0 = min(float(p["t0"]) for p in parts)
        t1 = max(float(p["t1"]) for p in parts)
        merged = [rec for p in parts for rec in p["records"]]
        summary = summarize_agentx(merged, t1 - t0, ep_size, conc, sessions_total)
        assert all(r["complete"] for r in merged) and len({r["req_id"] for r in merged}) == sum(s["total_reqs"] for s in suite["sessions"])
        assert summary["cached_input_tokens"] > 0
        print(format_summary_line(args.name, summary), flush=True)
        levels_out.append(summary)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "engine": args.name,
        "preset": suite.get("preset", "custom"),
        "max_context": suite.get("max_context", args.max_model_len),
        "levels": levels_out,
    }, indent=2))
    for r in range(ep_size):
        _di_rank_file(args.out, r).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 3. agxinfer StepChannel direct driver
# ---------------------------------------------------------------------------

async def _run_agx_direct(args: argparse.Namespace) -> None:
    import msgspec
    import zmq

    repo_root = _find_repo_root()
    sys.path[:0] = [str(repo_root / "tests" / "gpu"), str(repo_root), str(repo_root / "engine")]

    from core.common.ids import ProcId
    from core.contracts.columns import Column
    from core.contracts.requests import NewRequest, Requests
    from core.contracts.sampling import SamplingParams
    from core.ipc.channels import ChannelReader
    from core.ipc.config import IpcTunables
    from core.ipc.stream import open_reader
    from frontend.transport import Directory, StepChannel

    run_dir = args.run.resolve()
    supervisor = str(json.loads((run_dir / "control" / "control.json").read_text())["control"])
    cfg: dict[str, Any] = json.loads((run_dir / "config.json").read_text())
    assert cfg["alloc.radix_cache"] and not cfg["spec.algorithm"]
    num_gpus = int(cfg.get("sched.domains", 4))
    sched_proc = msgspec.convert(cfg["frontend.scheduler"], ProcId)
    tun = IpcTunables(rpc_timeout_s=300.0, close_timeout_s=10.0)
    zctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
    shm_name = f"agx-agentx-{os.getpid()}-{secrets.token_hex(4)}"
    raw_reader, addr = open_reader(
        zctx, local=sched_proc.host, peer=sched_proc.host, bind_host="127.0.0.1",
        shm_name=shm_name, tun=tun, replaceable=True,
    )
    tokens_reader = ChannelReader(raw_reader, addr, sched_proc.host)
    directory = Directory(
        supervisor=supervisor, scheduler=sched_proc,
        me=ProcId(sched_proc.host, "agentx", 0),
    )
    channel = StepChannel(sched_proc.host, tokens_reader, directory, tun=tun)
    await channel.start()

    suite = json.loads(args.suite.read_text())
    sessions: list[dict[str, Any]] = suite["sessions"]
    corpus_ids = np.memmap(args.corpus, dtype="<i4", mode="r")
    sampling = SamplingParams(0.0, 1.0, -1, None)
    uid_seq = secrets.randbits(30) << 32

    # Pre-materialize Column("i64") per unique req_id so benchmark CPU doesn't spend time packing
    col_by_req: dict[int, tuple[Column, int]] = {}
    for sess in sessions:
        for st in sess["steps"]:
            if st["kind"] == "turn":
                t = st["turn"]
                toks = _prompt_tokens(corpus_ids, int(t["offset"]), int(t["in"]))
                col_by_req[int(t["req_id"])] = (Column.of("i64", toks), len(toks))
            else:
                for ch in st["chains"]:
                    for t in ch:
                        toks = _prompt_tokens(corpus_ids, int(t["offset"]), int(t["in"]))
                        col_by_req[int(t["req_id"])] = (Column.of("i64", toks), len(toks))

    levels_out: list[dict[str, Any]] = []
    try:
        for conc in args.concurrencies:
            live: dict[int, tuple[dict[str, Any], asyncio.Event, int]] = {}
            records: list[dict[str, Any]] = []
            all_lanes_done = False
            recv_stop = asyncio.Event()

            async def receiver() -> None:
                while not recv_stop.is_set():
                    _, events = await channel.receive()
                    now = time.time()
                    req_ids = events.requests.values()
                    offsets = events.offsets.values()
                    for idx, raw_uid in enumerate(req_ids):
                        uid = int(raw_uid)
                        entry = live.get(uid)
                        if entry is None:
                            continue
                        rec, ev, gen = entry
                        n_new = int(offsets[idx + 1]) - int(offsets[idx])
                        if n_new > 0:
                            rec["times"].extend([now] * n_new)
                        if uid in events.errors:
                            rec["error"] = events.errors[uid]
                            del live[uid]
                            ev.set()
                        elif uid in events.finished:
                            rec["complete"] = len(rec["times"]) == gen
                            del live[uid]
                            ev.set()
                    for uid, err in events.errors.items():
                        entry = live.pop(int(uid), None)
                        if entry is not None:
                            entry[0]["error"] = err
                            entry[1].set()
                    for uid in events.finished:
                        entry = live.pop(int(uid), None)
                        if entry is not None:
                            rec, ev, gen = entry
                            rec["complete"] = len(rec["times"]) == gen
                            ev.set()
                    if all_lanes_done and not live:
                        recv_stop.set()

            async def run_turn(turn: dict[str, Any], lane_id: int) -> None:
                nonlocal uid_seq
                think_s = float(turn["think_s"])
                if think_s > 0:
                    await asyncio.sleep(think_s)
                uid = uid_seq
                uid_seq += 1
                col, plen = col_by_req[int(turn["req_id"])]
                gen = int(turn["out"])
                ev = asyncio.Event()
                now = time.time()
                rec: dict[str, Any] = {
                    "uid": uid,
                    "req_id": int(turn["req_id"]),
                    "lane": lane_id,
                    "sent": now,
                    "prompt_tokens": plen,
                    "expected_out": gen,
                    "is_subagent": bool(turn["is_subagent"]),
                    "times": [],
                    "complete": False,
                    "error": "",
                }
                live[uid] = (rec, ev, gen)
                records.append(rec)
                sent_ok = await channel.send(Requests(new=[NewRequest(uid, col, sampling, gen, [])]))
                if sent_ok is None:
                    rec["error"] = "scheduler unreachable"
                    live.pop(uid, None)
                    ev.set()
                await ev.wait()

            async def run_chain(chain: list[dict[str, Any]], lane_id: int) -> None:
                for t in chain:
                    await run_turn(t, lane_id)

            next_idx = 0

            async def lane_task(lane_id: int) -> None:
                nonlocal next_idx
                while True:
                    if next_idx >= len(sessions):
                        return
                    sess = sessions[next_idx]
                    next_idx += 1
                    for st in sess["steps"]:
                        if st["kind"] == "turn":
                            await run_turn(st["turn"], lane_id)
                        else:
                            chains = [c for c in st["chains"] if c]
                            if chains:
                                await asyncio.gather(*(run_chain(c, lane_id) for c in chains))

            recv_task = asyncio.create_task(receiver())
            t0 = time.time()
            try:
                await asyncio.gather(*(lane_task(lid) for lid in range(conc)))
                all_lanes_done = True
                if live:
                    await recv_stop.wait()
            finally:
                recv_task.cancel()
                try:
                    await recv_task
                except asyncio.CancelledError:
                    pass
            t1 = time.time()

            errors = [r["error"] for r in records if r.get("error")]
            if errors:
                print(f"WARNING: {len(errors)} errors in c={conc}: {errors[0]}", flush=True)
            assert not errors and len(records) == sum(s["total_reqs"] for s in sessions)
            source = next(run_dir.glob("*/snapshots/.run/sched-*/log"))
            hits = {int(parts[1]): int(parts[2]) for line in source.read_text().splitlines()
                    if line.startswith("AGENTX_ADMIT ") and (parts := line.split())}
            for rec in records:
                rec["cache_hit"] = hits[rec["uid"]]
            summary = summarize_agentx(records, t1 - t0, num_gpus, conc, len(sessions))
            assert summary["cached_input_tokens"] > 0
            print(format_summary_line(args.name, summary), flush=True)
            levels_out.append(summary)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({
                "engine": args.name,
                "preset": suite.get("preset", "custom"),
                "max_context": suite.get("max_context", 131072),
                "levels": levels_out,
            }, indent=2))
            await asyncio.sleep(args.pause)
    finally:
        await channel.close()
        tokens_reader.close()
        zctx.destroy(linger=0)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare", help="Build deterministic AgentX dev/eval suites and code token corpus")
    prep.add_argument("--traces-jsonl", type=Path, required=True)
    prep.add_argument("--tokenizer", type=Path, required=True)
    prep.add_argument("--out-dir", type=Path, required=True)
    prep.add_argument("--max-context", type=int, default=131072)
    prep.add_argument("--max-main-turns", type=int, default=6)
    prep.add_argument("--max-subagent-turns", type=int, default=4)
    prep.add_argument("--idle-cap-s", type=float, default=1.5)
    prep.add_argument("--dev-sessions", type=int, default=16)
    prep.add_argument("--eval-sessions", type=int, default=48)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--rebuild-corpus", action="store_true")

    di = sub.add_parser("deepinfer", help="Run AgentX suite on DeepInfer (DP+EP in-process)")
    di.add_argument("--deepinfer", type=Path, required=True)
    di.add_argument("--model", required=True)
    di.add_argument("--suite", type=Path, required=True)
    di.add_argument("--corpus", type=Path, required=True)
    di.add_argument("--concurrencies", type=int, nargs="+", default=[8, 24, 48])
    di.add_argument("--out", type=Path, required=True)
    di.add_argument("--name", default="deepinfer")
    di.add_argument("--max-seqs", type=int, default=128)
    di.add_argument("--token-budget", type=int, default=32768)
    di.add_argument("--max-model-len", type=int, default=131072)
    di.add_argument("--mem", type=float, default=0.88)
    di.add_argument("--yoco-clip", type=int, default=140, help="YoCo compact keep tokens (0 disables YoCo)")
    di.add_argument("--eplb", action="store_true")
    di.add_argument("--force-random", action="store_true")
    di.add_argument("--check", type=int, default=4)
    di.add_argument("--pause", type=float, default=3.0)

    agx = sub.add_parser("agx-direct", help="Run AgentX suite on agxinfer over StepChannel")
    agx.add_argument("--run", type=Path, required=True)
    agx.add_argument("--suite", type=Path, required=True)
    agx.add_argument("--corpus", type=Path, required=True)
    agx.add_argument("--concurrencies", type=int, nargs="+", default=[8, 24, 48])
    agx.add_argument("--out", type=Path, required=True)
    agx.add_argument("--name", default="agxinfer-direct")
    agx.add_argument("--pause", type=float, default=3.0)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.cmd == "prepare":
        cmd_prepare(args)
    elif args.cmd == "deepinfer":
        cmd_deepinfer(args)
    else:
        asyncio.run(_run_agx_direct(args))


if __name__ == "__main__":
    main()
