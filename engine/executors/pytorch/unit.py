# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""``executors.pytorch``: the worker's one executor (``executor@1``), on every torch platform.
Adapted from SGLang's ``ModelRunner`` (srt/model_executor/model_runner.py, e838360;
LICENSE.sglang): the step's plan is picked first, a ready graph bucket for a decode step that
fits it, the eager plan otherwise (``_forward_raw``; stage2-contract.md §9.1), before any state
changes; then the state slots of rows starting from position 0 are cleared and snapshots
restored (``mamba_clear_indices``, copy-on-write sources), the ``ForwardBatch`` inputs are built
(``input_ids``, ``positions``), the plan runs the model with its logits processor, and the
sampler draws at the position of each row's last token (``positions`` for decode,
``seq_lens - 1`` for extend). A graph found stale runs eagerly instead. The rows, their lengths
and their state items come from the scheduler's records instead of ``ScheduleBatch``.

Every row samples, as in SGLang; the scheduler drops what a prefill chunk that does not end its
prompt sampled.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from core.contracts.columns import Column
from core.contracts.plan import Plan, PlanKey, PlanService
from core.contracts.records import BatchPlan, Capacity, Resync, RowDelta, Sampled, StateOps, StepStats
from platforms.base import StaleGraph


@dataclass(frozen=True)
class Step:
    """``PendingStep``; on the CPU and with eager CUDA, finished when ``submit`` returns."""

    sampled: Sampled
    step_stats: StepStats
    logits: torch.Tensor
    """Each row's last-token logits; for evaluation and tests."""

    def ready(self) -> bool:
        return True

    def finish(self) -> Sampled:
        return self.sampled

    def stats(self) -> StepStats:
        return self.step_stats


def check_plan(plan: BatchPlan) -> None:
    """The rules of ``BatchPlan`` (stage2-contract.md §4.2); a plan that breaks them is a bug of
    the scheduler, not a reason to resync."""
    rows = list(plan.rows.values())
    num_tokens, positions, prev = (list(c.values()) for c in (plan.num_tokens, plan.positions, plan.from_previous))
    if not len(rows) == len(num_tokens) == len(positions) == len(prev):
        raise ValueError("a plan needs one num_tokens, position and from_previous per row")
    if len(set(rows)) != len(rows):
        raise ValueError("a row runs twice in one plan")
    if any(n < 1 for n in num_tokens) or any(p < 0 for p in positions):
        raise ValueError("every row runs at least one token from a position >= 0")
    if any(f and n != 1 for f, n in zip(prev, num_tokens, strict=True)):
        raise ValueError("a from_previous row runs exactly one token")
    if sum(n for f, n in zip(prev, num_tokens, strict=True) if not f) != len(plan.tokens):
        raise ValueError("tokens holds the input of every row without from_previous")


class PyTorchExecutor:
    """``executor@1``."""

    def __init__(self, platform: Any, batch: Any, plans: PlanService, sampler: Any, stores: dict[str, Any]) -> None:
        self.platform, self.batch, self.plans, self.sampler, self.stores = platform, batch, plans, sampler, stores

    def capacity(self) -> Capacity:
        return Capacity({kind: store.capacity() for kind, store in self.stores.items()})

    def synchronize(self) -> None:
        self.platform.synchronize(0)

    def submit(self, epoch: int, delta: RowDelta, plan: BatchPlan, state: StateOps) -> Step | Resync:
        start = time.perf_counter()
        check_plan(plan)
        reason = self.batch.check(delta, plan)
        if reason:
            return Resync(reason)
        chosen, eager = self._plans(plan)
        self.batch.apply(delta, state)
        rows = [int(r) for r in plan.rows.values()]
        num_tokens = [int(n) for n in plan.num_tokens.values()]
        positions = [int(p) for p in plan.positions.values()]
        # mamba_clear_indices: a row that starts from position 0 has no state yet (its conv runs
        # without an initial state); a row restored from a snapshot or continuing keeps its items.
        restored = {(c.kind, int(d)) for c in state.copy if c.when == "before" for d in c.dst.values()}
        for kind, store in self.stores.items():
            ids = self.batch.ids(kind)
            fresh = [i for r, p in zip(rows, positions, strict=True) if p == 0 for i in ids.get(r, [])
                     if (kind, i) not in restored]
            if fresh:
                store.clear(fresh)
        self._copies(state, "before")

        last_token = self.batch.binding.buffers()["last_token"]
        device = last_token.device
        given = torch.tensor(list(plan.tokens.values()), dtype=torch.int64, device=device)
        parts: list[torch.Tensor] = []
        offset = 0
        for r, n, prev in zip(rows, num_tokens, plan.from_previous.values(), strict=True):
            if prev:
                parts.append(last_token[r + 1:r + 2])
            else:
                parts.append(given[offset:offset + n])
                offset += n
        input_ids = torch.cat(parts)
        pos = torch.cat([torch.arange(p, p + n, device=device) for p, n in zip(positions, num_tokens, strict=True)])
        last_index = torch.cumsum(torch.tensor(num_tokens, device=device), dim=0) - 1

        try:
            logits = chosen.run(self.batch, plan, input_ids, pos, last_index)
        except StaleGraph:
            if chosen is eager:
                raise
            logits = eager.run(self.batch, plan, input_ids, pos, last_index)
        next_token_ids = self.sampler.sample(logits, self.batch.sampling(rows),
                                             [p + n - 1 for p, n in zip(positions, num_tokens, strict=True)])
        self.batch.record(rows, next_token_ids)
        self._copies(state, "after")

        sampled = Sampled(Column.of("i32", rows), Column.of("i32", [1] * len(rows)),
                          Column.of("i64", next_token_ids.tolist()))
        host_us = (time.perf_counter() - start) * 1e6
        return Step(sampled, StepStats(epoch, host_us, 0.0, 0.0), logits)

    def _plans(self, plan: BatchPlan) -> tuple[Plan, Plan]:
        """The step's plan and the eager one. A decode step takes the smallest ready graph bucket
        that holds its rows; any other step, or one no graph holds, the eager plan (bucket 0). The
        scheduler's ``bucket`` is only a hint."""
        phase = "decode" if plan.phase == "decode" else "prefill"
        eager = self.plans.get(PlanKey(phase, 0))
        fits = [b for b in self.plans.buckets(phase) if b >= len(plan.rows)] if phase == "decode" else []
        return (self.plans.get(PlanKey(phase, min(fits))) if fits else eager), eager

    def _copies(self, state: StateOps, when: str) -> None:
        for c in state.copy:
            if c.when == when:
                self.stores[c.kind].copy([int(x) for x in c.src.values()], [int(x) for x in c.dst.values()])


class Unit:
    STORES = (("kv.paged", "state.kv.paged@1"), ("gdn.slots", "state.gdn.slots@1"))

    def __init__(self, ctx: Any, config: Any) -> None:
        stores = {kind: ctx.use(name) for kind, name in self.STORES}
        ctx.provide("executor@1", PyTorchExecutor(ctx.platform, ctx.use("worker.batch@1"), ctx.use("plan@1"),
                                                ctx.use("layers.sampler@1"), stores))
