"""``core.plan``: the worker's execution plans (``plan@1``, stage2-contract.md §9), as SGLang's
``ModelRunner`` keeps its runners (srt/model_executor/runner/, e838360): an ``EagerRunner`` that
runs any step, and graph runners that replay the buckets they captured.

So far there is only the eager plan, bucket 0: every backend prepares for the step
(``init_forward_metadata``), then the model runs. Graph buckets come in M2c; ``buckets`` lists
only the ready ones whose dependencies hold, and the worker falls back to the eager plan for the
rest (§9.1, §9.4.3).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from core.contracts.plan import Backend, PersistentBatch, Phase, Plan, PlanKey, PlanService
from core.contracts.records import BatchPlan
from core.contracts.tensor import Tensor


def phase_of(plan: BatchPlan) -> Phase:
    """A mixed step runs as an extend, as SGLang's ``EagerRunner`` runs MIXED as EXTEND."""
    return "decode" if plan.phase == "decode" else "prefill"


class EagerPlan:
    """``Plan`` for bucket 0: SGLang's ``EagerRunner``."""

    def __init__(self, model: Any, backends: Sequence[Backend]) -> None:
        self.model, self.backends = model, list(backends)

    def run(self, batch: PersistentBatch, plan: BatchPlan, input_ids: Tensor, positions: Tensor,
            last_index: Tensor) -> Tensor:
        key = PlanKey(phase_of(plan), 0)
        for backend in self.backends:
            backend.prepare(batch, plan, key)
        return self.model.forward(input_ids, positions, last_index)


class Plans:
    """``plan@1``."""

    def __init__(self, model: Any, backends: Sequence[Backend]) -> None:
        self.eager = EagerPlan(model, backends)

    def buckets(self, phase: Phase) -> list[int]:
        return []

    def get(self, key: PlanKey) -> Plan:
        if key.bucket == 0:
            return self.eager
        raise LookupError(f"no {key.phase} graph of bucket {key.bucket} is ready")


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        model = ctx.use("model@1")
        backends = [ctx.use(name) for name in dict.fromkeys(model.assembly.layers)]
        ctx.provide("plan@1", Plans(model, backends))


if TYPE_CHECKING:
    _: type[PlanService] = Plans
    __: type[Plan] = EagerPlan
