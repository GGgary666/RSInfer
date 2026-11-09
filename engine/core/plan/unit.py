"""``core.plan``: the worker's execution plans (``plan@1``, stage2-contract.md §9), as SGLang's
``ModelRunner`` keeps its runners (srt/model_executor/runner/, e838360): an ``EagerRunner`` that
runs any step, and graph runners that replay the buckets they captured.

So far there is only the eager plan, bucket 0: every backend prepares for the step
(``init_forward_metadata``), then the model runs. Graph buckets come in M2c; ``buckets`` lists
only the ready ones whose dependencies hold, and the worker falls back to the eager plan for the
rest (§9.1, §9.4.3).

At start the unit has every backend fix its kernel solutions per phase (``Backend.resolve``; the
selections are the plans' by-value dependencies) and logs one record of what runs on what, with
which state formats (§5.5.3, §8.1). A backend named in the config that cannot run here fails the
start.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from core.contracts.plan import Backend, PersistentBatch, Phase, Plan, PlanKey, PlanService
from core.contracts.records import BatchPlan
from core.contracts.state import StateDecl
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


def resolution(platform: Any, states: Sequence[StateDecl],
               backends: Mapping[str, Mapping[str, Mapping[str, str]]]) -> dict[str, Any]:
    """The start-up record: the platform, each state kind's dtypes (and page size), and per phase
    each backend's choice; verify does not run in stage 2 yet."""
    kinds: dict[str, dict[str, Any]] = {}
    for d in states:
        kinds.setdefault(d.kind, {**{name: dtype for name, (_, dtype) in d.tensors.items()},
                                  **({"page_size": d.tokens_per_item} if d.tokens_per_item else {})})
    devices = platform.devices()
    return {"platform": platform.info().name, "device": list(devices[0].capabilities) if devices else [],
            "states": kinds, "prefill": {name: r["prefill"] for name, r in backends.items()},
            "decode": {name: r["decode"] for name, r in backends.items()}, "verify": "disabled"}


class Plans:
    """``plan@1``."""

    def __init__(self, model: Any, backends: Sequence[Backend], resolved: Mapping[str, Any]) -> None:
        self.eager = EagerPlan(model, backends)
        self.resolved = resolved
        """The start-up record (``resolution``)."""

    def buckets(self, phase: Phase) -> list[int]:
        return []

    def get(self, key: PlanKey) -> Plan:
        if key.bucket == 0:
            return self.eager
        raise LookupError(f"no {key.phase} graph of bucket {key.bucket} is ready")


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        model = ctx.use("model@1")
        names = list(dict.fromkeys(model.assembly.layers))
        backends: list[Backend] = [ctx.use(name) for name in names]
        resolved = resolution(ctx.platform, model.assembly.states,
                              {name: b.resolve(model.assembly.states) for name, b in zip(names, backends, strict=True)})
        print(f"core.plan: {json.dumps(resolved, sort_keys=True)}", file=sys.stderr, flush=True)
        ctx.provide("plan@1", Plans(model, backends, resolved))


if TYPE_CHECKING:
    _: type[PlanService] = Plans
    __: type[Plan] = EagerPlan
