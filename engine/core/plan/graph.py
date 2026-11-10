"""Decode graphs: one per bucket of rows, recorded while the worker is paused, and what they were
recorded over. A graph is SGLang's ``DecodeCudaGraphRunner`` for one batch size: fixed inputs, the
rows padded to the bucket with the padding row, the whole forward replayed."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from core.contracts.columns import Column
from core.contracts.plan import Backend, PersistentBatch, PlanKey
from core.contracts.records import BatchPlan
from core.contracts.tensor import Tensor
from unit.runtime.proxy import provision_of

DEFAULT_MAX_BUCKET = 8
"""Without ``plan.graph_buckets``: powers of two up to this many rows. The decode attention of a
captured step costs by ``batch.max_context_len`` (``kernels.torch_graph``), which more rows than
this did not win back (doc/evaluations/cuda-graph-review-fixes.md); larger steps run eagerly."""


def buckets(mode: str, values: Sequence[int], rows: int, budget: int) -> list[int]:
    if mode not in ("auto", "off", "required"):
        raise ValueError("plan.cuda_graph must be auto, off or required")
    if rows < 1 or budget <= 0:
        raise ValueError("batch.max_rows and plan.graph_memory_bytes must be positive")
    if any(type(b) is not int or not 0 < b <= rows for b in values) or len(set(values)) != len(values):
        raise ValueError("plan.graph_buckets must be distinct positive integers <= batch.max_rows")
    chosen = sorted(values) if values else [1 << i for i in range(min(rows, DEFAULT_MAX_BUCKET).bit_length())]
    if mode == "required" and chosen[-1] != rows:
        raise ValueError("required graphs must cover batch.max_rows; list its exact capacity in plan.graph_buckets")
    return chosen


def no_rows() -> BatchPlan:
    """A decode step without rows: every lane of a bucket is padding."""
    return BatchPlan(Column.of("i32", []), Column.of("i32", []), Column.of("i32", []),
                     Column.of("i64", []), Column.of("u8", []), "decode", 0)


class Dependencies:
    """What the graphs were recorded over, by address: the instances behind the services, the
    registrations of the kernel catalog, and one binding generation per provider (``Binding``)."""

    def __init__(self, services: Sequence[Any], bindings: Sequence[Callable[[], Any]], catalog: Any) -> None:
        self.provisions = [provision_of(s) for s in services]
        self.bindings = list(bindings)
        self.generations = [b().generation for b in bindings]
        self.catalog, self.version = catalog, catalog.version

    def valid(self) -> bool:
        # A provision carries the provider's instance and its revocation: a reloaded provider is
        # another instance, whatever its generation says.
        if any(p.revoked for p in self.provisions) or self.catalog.version != self.version:
            return False
        try:
            return all(b().generation == g for b, g in zip(self.bindings, self.generations, strict=True))
        except (KeyError, RuntimeError):
            return False  # a provider freed its buffers before its service was revoked


class GraphPlan:
    """``Plan`` for one decode bucket."""

    def __init__(self, size: int, device: torch.device, backends: Sequence[Backend]) -> None:
        self.size, self.backends = size, list(backends)
        self.inputs = torch.zeros(size, dtype=torch.int64, device=device)
        self.positions = torch.zeros_like(self.inputs)
        self.last_index = torch.arange(size, device=device)
        self.output: torch.Tensor | None = None
        self.graph: Any = None

    def run(self, batch: PersistentBatch, plan: BatchPlan, input_ids: Tensor, positions: Tensor,
            last_index: Tensor) -> Tensor:
        """The step's dependency check was ``PlanService.buckets``; this only replays."""
        n = len(plan.rows)
        if plan.phase != "decode" or not 0 < n <= self.size or any(t != 1 for t in plan.num_tokens.values()):
            raise ValueError("a decode graph runs one token per row, at most its bucket of rows")
        for backend in self.backends:
            backend.prepare(batch, plan, PlanKey("decode", self.size))
        self.inputs[:n].copy_(input_ids)
        self.positions[:n].copy_(positions)
        self.inputs[n:].zero_()
        self.positions[n:].zero_()
        self.graph.replay()
        assert self.output is not None
        return self.output[:n]

    def close(self) -> None:
        if self.graph is not None:
            self.graph.close()
        self.graph, self.output = None, None

    def padded(self, batch: PersistentBatch, model: Any, stores: Sequence[Any]) -> torch.Tensor:
        """An eager forward of a bucket of padding lanes, from a cleared padding item. The backends
        allocate the bucket's fixed buffers here the first time, outside inference mode: every
        later step updates them in place."""
        for store in stores:
            store.clear(store.scratch())
        for backend in self.backends:
            backend.prepare(batch, no_rows(), PlanKey("decode", self.size))
        with torch.inference_mode():
            return model.forward(self.inputs, self.positions, self.last_index)

    def capture(self, batch: PersistentBatch, model: Any, stores: Sequence[Any], recorder: Any) -> None:
        """Record the forward of a bucket of padding lanes; the replay must give the logits the
        same kernels gave eagerly. That eager forward is the warm-up too."""
        with recorder.warmup():
            expected = self.padded(batch, model, stores).clone()
            for store in stores:
                store.clear(store.scratch())
        with torch.inference_mode(), recorder.capture() as graph:
            self.output = model.forward(self.inputs, self.positions, self.last_index)
        self.graph = graph
        with recorder.warmup():
            graph.replay()
        assert self.output is not None
        torch.testing.assert_close(self.output, expected, rtol=1e-3, atol=1e-3, msg=lambda m: (
            f"decode graph of {self.size} rows replays other logits than the eager forward it recorded: {m}"))
