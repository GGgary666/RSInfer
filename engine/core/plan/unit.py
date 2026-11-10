"""``core.plan``: the worker's execution plans (``plan@1``, stage2-contract.md §9), as SGLang's
``ModelRunner`` keeps its runners (srt/model_executor/runner/, e838360): an ``EagerRunner`` that
runs any step, and graph runners that replay the buckets they captured.

Bucket 0 prepares metadata and runs eagerly. Decode graph buckets are built and validated at
startup or while paused; ``buckets`` lists only those whose dependencies still hold. ``required``
refuses an unavailable decode graph before the executor changes state.

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

import torch

from core.contracts.plan import Backend, PersistentBatch, Phase, Plan, PlanKey, PlanService
from core.contracts.records import BatchPlan
from core.contracts.state import StateDecl
from core.contracts.tensor import Tensor
from core.plan.graph import Dependencies, GraphPlan, buckets, build
from unit.runtime.proxy import provision_of


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
        self.graphs: dict[int, GraphPlan] = {}
        self.required = False
        """The start-up record (``resolution``)."""

    def buckets(self, phase: Phase) -> list[int]:
        return [size for size, graph in self.graphs.items() if graph.valid()] if phase == "decode" else []

    def get(self, key: PlanKey) -> Plan:
        if key.bucket == 0:
            return self.eager
        if key.phase == "decode" and key.bucket in self.graphs and self.graphs[key.bucket].valid():
            return self.graphs[key.bucket]
        raise LookupError(f"no {key.phase} graph of bucket {key.bucket} is ready")

    def close(self) -> None:
        if self.graphs:
            torch.cuda.synchronize()
        for graph in self.graphs.values():
            graph.close()
        self.graphs.clear()
        for backend in self.eager.backends:
            backend.release_graphs()


class Rebuild:
    """Prepare is host-only. Execute/abort run at the paused dependency boundary."""

    def __init__(self, unit: Unit) -> None:
        self.unit = unit
        self.executed = False

    def prepare(self, args: dict[str, Any]) -> None:
        if args.get("config"):
            raise ValueError("graph configuration changes require reloading core.plan")

    def execute(self, args: dict[str, Any]) -> None:
        self.executed = True
        self.unit.rebuild()

    def commit(self) -> None:
        self.executed = False

    def abort(self) -> None:
        if self.executed:
            self.unit.rebuild()
            self.executed = False


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        self.ctx = ctx
        self.mode = config["plan.cuda_graph"]
        self.budget = config["plan.graph_memory_bytes"]
        self.sizes = buckets(self.mode, config["plan.graph_buckets"], config["batch.max_rows"], self.budget)
        self.max_context = config["batch.max_context_len"]
        if self.max_context < 1:
            raise ValueError("batch.max_context_len must be positive")
        model = ctx.use("model@1")
        names = list(dict.fromkeys(model.assembly.layers))
        backends: list[Backend] = [ctx.use(name) for name in names]
        resolved = resolution(ctx.platform, model.assembly.states,
                              {name: b.resolve(model.assembly.states) for name, b in zip(names, backends, strict=True)})
        self.model, self.names, self.backends = model, names, backends
        self.catalog = ctx.use("kernels.catalog@1")
        self.batch, self.weights = ctx.use("worker.batch@1"), ctx.use("state.weights@1")
        self.stores = {kind: ctx.use(service) for kind, service in
                       (("kv.paged", "state.kv.paged@1"), ("gdn.slots", "state.gdn.slots@1"))
                       if any(d.kind == kind for d in model.assembly.states)}
        self.plans = Plans(model, backends, resolved)
        self.plans.required = self.mode == "required"
        ctx.effect(lambda: self.plans.close, label="decode graphs")
        ctx.on_action("rebuild_graphs", Rebuild(self))
        self.rebuild()
        ctx.provide("plan@1", self.plans)

    def rebuild(self) -> None:
        self.plans.close()
        resolved = resolution(self.ctx.platform, self.model.assembly.states,
                              {name: b.resolve(self.model.assembly.states)
                               for name, b in zip(self.names, self.backends, strict=True)})
        unsupported = [value for choices in resolved["decode"].values() for key, value in choices.items()
                       if key != "backend" and not self.catalog.describe(value.removesuffix(" (fallback)")).capturable]
        reason = ("disabled" if self.mode == "off" else
                  "platform has no CUDA graph support" if self.ctx.platform.info().name != "cuda_graph" else
                  f"uncapturable decode kernels: {unsupported}" if unsupported else "")
        if reason and self.plans.required:
            raise ValueError(f"plan.cuda_graph=required: {reason}")
        if not reason:
            self.batch.prepare(list(self.stores))
            device = self.batch.binding.buffers()["last_token"].device
            if sum(self.sizes) * (self.max_context * 8 + 64) > self.budget:
                raise MemoryError("graph metadata alone exceeds plan.graph_memory_bytes")
            if self.budget > torch.cuda.mem_get_info(device)[0]:
                raise MemoryError("not enough free memory for plan.graph_memory_bytes")
            baseline = torch.cuda.memory_allocated(device)
            candidates: dict[int, GraphPlan] = {}
            bindings = [lambda: self.batch.binding, lambda: self.weights.bind("")]
            bindings.extend(lambda b=b: b.binding for b in self.backends)
            bindings.extend(lambda d=d: self.stores[d.kind].bind(d.layer) for d in self.model.assembly.states)

            def guard() -> Dependencies:
                return Dependencies([self.model, self.batch, self.weights, self.catalog,
                                     *self.backends, *self.stores.values()], bindings, self.catalog)

            try:
                for size in self.sizes:
                    candidates[size] = GraphPlan(size, device, self.backends)
                with self.ctx.plan_scope("decode_graphs"):
                    deps = build(candidates, self.model, self.backends, self.batch, self.stores,
                                 self.ctx.platform.graph(), self.catalog, resolved, self.budget, baseline, guard)
                for graph in candidates.values():
                    graph.deps = deps
                self.plans.graphs = candidates
            except BaseException:
                torch.cuda.synchronize()
                for graph in candidates.values():
                    graph.close()
                raise
        # Register even for auto's unsupported combination: a kernel reload may make it capturable.
        self.ctx.derived("decode_graphs", rebuild="rebuild_graphs",
                         units=[*self.catalog.sources(), str(provision_of(self.catalog).owner.unit)])
        resolved["cuda_graph"] = {"mode": self.mode, "buckets": self.plans.buckets("decode"), "reason": reason,
                                  "build": {str(b): {"seconds": g.build_seconds, "peak_bytes": g.peak_bytes}
                                            for b, g in self.plans.graphs.items()}}
        self.plans.resolved = resolved
        print(f"core.plan: {json.dumps(resolved, sort_keys=True)}", file=sys.stderr, flush=True)


if TYPE_CHECKING:
    _: type[PlanService] = Plans
    __: type[Plan] = EagerPlan
