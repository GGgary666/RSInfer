"""``core.plan``: the worker's execution plans (``plan@1``, stage2-contract.md §9), as SGLang's
``ModelRunner`` keeps its runners (srt/model_executor/runner/, e838360): an ``EagerRunner`` that
runs any step, and graph runners that replay the buckets they captured.

Bucket 0 prepares metadata and runs eagerly. The decode graph buckets are recorded when the unit
loads, at start or while the worker is paused, largest first into one memory pool; ``buckets``
lists them while what they were recorded over still holds. Nothing rebuilds them in place: a
change to what they were built from reloads this unit (``ctx.derived`` without a rebuild action),
after the units it was built from, and a rollback reloads the old one the same way.

``plan.cuda_graph``: ``off`` records nothing; ``auto`` records where it can and says why where it
cannot (no graph platform, a kernel that is not capturable, graphs that do not fit their memory);
``required`` fails the start in those cases, and has the executor refuse a decode step no graph
holds, before it changes any state.

At start the unit has every backend fix its kernel solutions per phase (``Backend.resolve``; the
selections are the plans' by-value dependencies) and logs one record of what runs on what, with
which state formats (§5.5.3, §8.1). A backend named in the config that cannot run here fails the
start.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

from core.contracts.plan import GRAPH, Backend, PersistentBatch, Phase, Plan, PlanKey, PlanService
from core.contracts.records import BatchPlan
from core.contracts.state import StateDecl
from core.contracts.tensor import Tensor
from core.plan.graph import Dependencies, GraphPlan, buckets
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
    each backend's choice, with the one of a decode step replayed from a graph; verify does not run
    in stage 2 yet."""
    kinds: dict[str, dict[str, Any]] = {}
    for d in states:
        kinds.setdefault(d.kind, {**{name: dtype for name, (_, dtype) in d.tensors.items()},
                                  **({"page_size": d.tokens_per_item} if d.tokens_per_item else {})})
    devices = platform.devices()
    return {"platform": platform.info().name, "device": list(devices[0].capabilities) if devices else [],
            "states": kinds, **{phase: {name: r[phase] for name, r in backends.items()}
                                for phase in ("prefill", "decode", GRAPH)}, "verify": "disabled"}


class Plans:
    """``plan@1``."""

    def __init__(self, model: Any, backends: Sequence[Backend], required: bool = False) -> None:
        self.eager = EagerPlan(model, backends)
        self.required = required
        self.resolved: Mapping[str, Any] = {}
        """The start-up record (``resolution``, and what became of the graphs)."""
        self.graphs: dict[int, GraphPlan] = {}
        self.deps: Dependencies | None = None
        """What every bucket was recorded over."""

    def buckets(self, phase: Phase) -> list[int]:
        """A step's one dependency check: the executor asks here before it changes any state."""
        if phase != "decode" or self.deps is None or not self.deps.valid():
            return []
        return list(self.graphs)

    def get(self, key: PlanKey) -> Plan:
        if key.bucket == 0:
            return self.eager
        if key.phase == "decode" and key.bucket in self.graphs:
            return self.graphs[key.bucket]
        raise LookupError(f"no {key.phase} graph of bucket {key.bucket} is ready")

    def close(self) -> None:
        """After the step stream and the consumers of the last logits are done. The graphs' pool
        goes back to the device: PyTorch would keep its blocks cached where the next graphs,
        recorded into a pool of their own, cannot use them."""
        if not self.graphs:
            return
        torch.cuda.synchronize()
        for graph in self.graphs.values():
            graph.close()
        self.graphs, self.deps = {}, None
        for backend in self.eager.backends:
            backend.release_graphs()
        torch.cuda.empty_cache()


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        mode, budget = config["plan.cuda_graph"], config["plan.graph_memory_bytes"]
        sizes = buckets(mode, config["plan.graph_buckets"], config["batch.max_rows"], budget)
        model = ctx.use("model@1")
        names = list(dict.fromkeys(model.assembly.layers))
        backends: list[Backend] = [ctx.use(name) for name in names]
        self.model, self.backends = model, backends
        self.catalog = ctx.use("kernels.catalog@1")
        self.batch, self.weights = ctx.use("worker.batch@1"), ctx.use("state.weights@1")
        self.stores = {kind: ctx.use(service) for kind, service in
                       (("kv.paged", "state.kv.paged@1"), ("gdn.slots", "state.gdn.slots@1"))
                       if any(d.kind == kind for d in model.assembly.states)}
        self.plans = Plans(model, backends, required=mode == "required")
        ctx.effect(lambda: self.plans.close, label="decode graphs")
        resolved = resolution(ctx.platform, model.assembly.states,
                              {name: b.resolve(model.assembly.states) for name, b in zip(names, backends, strict=True)})
        uncapturable = sorted({definition for chosen in resolved[GRAPH].values()
                               for definition, solution in chosen.items() if not solution})
        reason = ("disabled" if mode == "off" else
                  "platform has no CUDA graph support" if "cuda_graph" not in resolved["device"] else
                  f"uncapturable decode kernels: {uncapturable}" if uncapturable else "")
        build: dict[str, Any] = {}
        if not reason:
            try:
                reason, build = self.record(sizes, budget, ctx.platform.graph())
            except (MemoryError, torch.OutOfMemoryError) as e:
                if self.plans.required:
                    raise
                reason = "the graphs do not fit: " + ". ".join(str(e).split(". ")[:2])
        if reason and self.plans.required:
            raise ValueError(f"plan.cuda_graph=required: {reason}")
        # No rebuild action: core.deps reloads this unit when one of these changes, or when a unit
        # that registers kernels is added. Also without graphs: a kernel unit may bring them.
        ctx.derived("decode_graphs", rebuild="",
                    units=[*self.catalog.sources(), str(provision_of(self.catalog).owner.unit)])
        resolved["cuda_graph"] = {"mode": mode, "buckets": list(self.plans.graphs), "reason": reason, **build}
        self.plans.resolved = resolved
        print(f"core.plan: {json.dumps(resolved, sort_keys=True)}", file=sys.stderr, flush=True)
        ctx.provide("plan@1", self.plans)

    def record(self, sizes: Sequence[int], budget: int, recorder: Any) -> tuple[str, dict[str, Any]]:
        """Record a graph per bucket, or say why none is recorded. All or none: a failure frees
        what it recorded, and so does a set that keeps more device memory than ``budget``."""
        states = self.model.assembly.states
        self.batch.prepare(list(self.stores))
        device = self.batch.binding.buffers()["last_token"].device
        stores = list(self.stores.values())
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()  # count what the graphs keep, not what a cached block happened to hold
        before = torch.cuda.memory_reserved(device)
        graphs = {size: GraphPlan(size, device, self.backends) for size in sorted(sizes, reverse=True)}
        seconds: dict[str, float] = {}
        try:
            # What the forward runs besides the backends' kernels (norms, linears, ...): eager's
            # choices, which a captured step shares, so they have to be capturable too.
            with recorder.warmup(), self.catalog.recording() as used:
                next(iter(graphs.values())).padded(self.batch, self.model, stores)
            uncapturable = sorted(s for s in used if not self.catalog.describe(s).capturable)
            if uncapturable:
                for backend in self.backends:
                    backend.release_graphs()
                return f"uncapturable decode kernels: {uncapturable}", {}
            # The largest first, into one pool: the smaller ones fit in what it took.
            for size, graph in graphs.items():
                started = time.perf_counter()
                graph.capture(self.batch, self.model, stores, recorder)
                seconds[str(size)] = round(time.perf_counter() - started, 3)
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            kept = torch.cuda.memory_reserved(device) - before
            if kept > budget:
                raise MemoryError(f"the decode graphs keep {kept} bytes of device memory, more than "
                                  f"plan.graph_memory_bytes = {budget}")
        except BaseException:
            torch.cuda.synchronize(device)
            for graph in graphs.values():
                graph.close()
            for backend in self.backends:
                backend.release_graphs()
            torch.cuda.empty_cache()
            raise
        # One generation per provider: a store counts its own over all its layers.
        layer = {kind: next(d.layer for d in states if d.kind == kind) for kind in self.stores}
        self.plans.deps = Dependencies(
            [self.model, self.batch, self.weights, self.catalog, *self.backends, *stores],
            [lambda: self.batch.binding, lambda: self.weights.bind(""),
             *(lambda b=b: b.binding for b in self.backends),
             *(lambda kind=kind: self.stores[kind].bind(layer[kind]) for kind in self.stores)], self.catalog)
        self.plans.graphs = dict(sorted(graphs.items()))
        return "", {"seconds": seconds, "kept_bytes": kept}


if TYPE_CHECKING:
    _: type[PlanService] = Plans
    __: type[Plan] = EagerPlan
