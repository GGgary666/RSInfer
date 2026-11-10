"""Decode graphs and their dependency guards. Construction only runs while paused."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any
import time

import torch

from core.contracts.columns import Column
from core.contracts.plan import PlanKey
from core.contracts.records import BatchPlan
from platforms.base import StaleGraph
from unit.runtime.proxy import provision_of


def buckets(mode: str, values: Sequence[int], rows: int, budget: int) -> list[int]:
    if mode not in ("auto", "off", "required"):
        raise ValueError("plan.cuda_graph must be auto, off or required")
    if rows < 1 or budget <= 0:
        raise ValueError("batch.max_rows and plan.graph_memory_bytes must be positive")
    if any(type(b) is not int or not 0 < b <= rows for b in values) or len(set(values)) != len(values):
        raise ValueError("plan.graph_buckets must be distinct positive integers <= batch.max_rows")
    chosen = sorted(values) if values else [1 << i for i in range(rows.bit_length())]
    if mode == "required" and chosen[-1] != rows:
        raise ValueError("required graphs must cover batch.max_rows; include its exact capacity")
    return chosen


def scratch_plan() -> BatchPlan:
    return BatchPlan(Column.of("i32", []), Column.of("i32", []), Column.of("i32", []),
                     Column.of("i64", []), Column.of("u8", []), "decode", 0)


def layout(binding: Any) -> tuple[tuple[Any, ...], ...]:
    return tuple((name, t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)
                 for name, t in sorted(binding.buffers().items()))


class Dependencies:
    def __init__(self, services: Sequence[Any], bindings: Sequence[Callable[[], Any]], catalog: Any,
                 *, debug: bool = False) -> None:
        self.provisions = [provision_of(s) for s in services]
        self.bindings = list(bindings)
        self.generations = [b().generation for b in bindings]
        self.layouts = [layout(b()) for b in bindings] if debug else []
        self.catalog, self.version = catalog, catalog.version

    def valid(self) -> bool:
        # A proxy's provision carries InstanceId and revocation, not Python proxy identity.
        if any(p.revoked for p in self.provisions):
            return False
        if self.catalog.version != self.version:
            return False
        try:
            if any(b().generation != g for b, g in zip(self.bindings, self.generations, strict=True)):
                return False
            return not self.layouts or all(layout(b()) == old
                                           for b, old in zip(self.bindings, self.layouts, strict=True))
        except (KeyError, RuntimeError):
            return False  # A provider released its buffers without revoking its service yet.


class GraphPlan:
    def __init__(self, size: int, device: torch.device, backends: Sequence[Any]) -> None:
        self.size, self.backends = size, list(backends)
        self.inputs = torch.zeros(size, dtype=torch.int64, device=device)
        self.positions = torch.zeros_like(self.inputs)
        self.last_index = torch.arange(size, device=device)
        self.output: torch.Tensor | None = None
        self.graph: Any = None
        self.deps: Dependencies | None = None
        self.peak_bytes = 0
        self.build_seconds = 0.0

    def valid(self) -> bool:
        return self.graph is not None and self.deps is not None and self.deps.valid()

    def run(self, batch: Any, plan: BatchPlan, input_ids: torch.Tensor, positions: torch.Tensor,
            last_index: torch.Tensor) -> torch.Tensor:
        if not self.valid():
            raise StaleGraph("decode graph dependency was revoked or changed")
        n = len(plan.rows)
        if plan.phase != "decode" or not 0 < n <= self.size or any(t != 1 for t in plan.num_tokens.values()):
            raise ValueError("graph requires one token per row within its bucket")
        for backend in self.backends:
            backend.prepare(batch, plan, PlanKey("decode", self.size))
        self.inputs.zero_()
        self.positions.zero_()
        self.inputs[:n].copy_(input_ids)
        self.positions[:n].copy_(positions)
        self.graph.replay()
        assert self.output is not None
        return self.output[:n]

    def close(self) -> None:
        if self.graph is not None:
            self.graph.close()
        self.graph, self.output, self.deps = None, None, None


def build(plans: dict[int, GraphPlan], model: Any, backends: Sequence[Any], batch: Any,
          stores: dict[str, Any], recorder: Any, catalog: Any, resolved: dict[str, Any],
          budget: int, baseline: int, guard: Callable[[], Dependencies]) -> Dependencies:
    """Warm up, capture, and compare each candidate before the set is published."""
    torch.cuda.reset_peak_memory_stats()
    empty = scratch_plan()
    # Allocate every backend bucket before taking any generation snapshots.
    for size in plans:
        for backend in backends:
            backend.prepare(batch, empty, PlanKey("decode", size))
    deps = guard()
    for size, plan in plans.items():
        started = time.perf_counter()

        def clear() -> None:
            for store in stores.values():
                store.clear(store.scratch())

        def scratch_values() -> list[torch.Tensor]:
            values: list[torch.Tensor] = []
            for decl in model.assembly.states:
                store = stores[decl.kind]
                index = store.slots(store.scratch()) if decl.tokens_per_item else store.indices(store.scratch())
                values.extend(t[index].clone() for t in store.bind(decl.layer).buffers().values())
            return values

        with recorder.warmup(), torch.inference_mode():
            for backend in backends:
                backend.prepare(batch, empty, PlanKey("decode", size))
            catalog.selected(clear=True)
            for _ in range(3):
                clear()
                model.forward(plan.inputs, plan.positions, plan.last_index)
            descriptions = list(catalog.selected())
            for choices in resolved["decode"].values():
                descriptions.extend(catalog.describe(value.removesuffix(" (fallback)"))
                                    for key, value in choices.items() if key != "backend")
            unsupported = [s.id for s in descriptions if not s.capturable]
            if unsupported:
                raise ValueError(f"decode kernels are not capturable: {unsupported}")
            clear()
            expected = model.forward(plan.inputs, plan.positions, plan.last_index).clone()
            expected_state = scratch_values()
            clear()
        with torch.inference_mode(), recorder.capture() as graph:
            plan.output = model.forward(plan.inputs, plan.positions, plan.last_index)
        plan.graph = graph
        with recorder.warmup(), torch.inference_mode():
            clear()
            graph.replay()
        assert plan.output is not None
        torch.testing.assert_close(plan.output, expected, rtol=1e-3, atol=1e-3)
        for actual, wanted in zip(scratch_values(), expected_state, strict=True):
            torch.testing.assert_close(actual, wanted, rtol=1e-3, atol=1e-3)
        clear()
        peak = torch.cuda.max_memory_allocated() - baseline
        plan.peak_bytes, plan.build_seconds = peak, time.perf_counter() - started
        if peak > budget:
            raise MemoryError(f"graph bucket {size} needs {peak} bytes; plan.graph_memory_bytes={budget}")
    torch.cuda.synchronize()
    if not deps.valid():
        raise RuntimeError("graph dependencies changed while building")
    return deps
