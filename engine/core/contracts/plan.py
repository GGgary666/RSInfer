"""The plan unit, the backends it records graphs over, and the worker's persistent batch
(stage2-contract.md §8, §9). The CUDA graph rules here come from the GPU probes (#17).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Literal, Protocol

import msgspec

from core.contracts import Tensor
from core.contracts.records import BatchPlan, RowDelta, StateOps
from core.contracts.state import Binding

type Phase = Literal["prefill", "decode"]


class PlanKey(msgspec.Struct, frozen=True):
    phase: Phase
    bucket: int
    """Padded batch size; 0 = eager (prefill always is in v0)."""


class GraphDep(msgspec.Struct, frozen=True):
    """One thing a recorded graph depends on, checked before every replay."""

    kind: Literal["binding", "value"]
    name: str
    """``binding``: a provider's ``Binding`` (compared by generation); ``value``: a solution id,
    a unit instance (its code version), a config key, the batch size."""
    recorded: str
    """The generation or value at capture, as text."""


class Plan(Protocol):
    @property
    def key(self) -> PlanKey: ...
    @property
    def deps(self) -> Sequence[GraphDep]: ...

    def stale(self) -> list[str]:
        """Changed dependencies, without running. Cheap: generations and a few values."""
        ...

    def run(self) -> None:
        """Replay the graph (or run eagerly) over the inputs already in the bound buffers.
        Raises ``platforms.base.StaleGraph`` if a dependency changed."""
        ...


class PlanService(Protocol):
    """``plan@1`` in every worker. The only unit that may record graphs (``captures_graph``)."""

    def buckets(self, phase: Phase) -> list[int]:
        """Buckets with a ready plan; from the bucketing policy (powers of 2 by default)."""
        ...

    def get(self, key: PlanKey) -> Plan: ...

    def rebuild(self, keys: Sequence[PlanKey]) -> Iterator[None]:
        """Record new plans next to the serving ones, during prepare: an iterator that records
        one bucket per ``next()`` (one slice per safe point, change.md 4.6). Each slice is a
        pause window: it first waits for the step still running on the device, since nothing
        else may run in the process while a graph is captured (GPU probe P10). Warm-up runs on
        ``Scratch`` items only. The new graphs never share a memory pool with graphs that may
        still replay; they replace the old ones at the epoch boundary."""
        ...


class Backend(Protocol):
    """An attention or GDN backend (``attn.backend.<name>@1``), split the way sglang splits it.

    ``prepare`` runs outside the graph each step: host-side work that writes the step's
    metadata (page tables, sequence lengths, slot indices) into the backend's own fixed
    buffers. ``forward`` runs inside the graph: fixed-shape device work only. The fixed
    buffers are the graph's by-address dependencies; the batch size is a by-value one."""

    @property
    def binding(self) -> Binding: ...

    def prepare(self, batch: PersistentBatch, plan: BatchPlan, key: PlanKey) -> None: ...
    def forward(self, layer: int, *args: Tensor) -> Tensor: ...


class PersistentBatch(Protocol):
    """``worker.batch@1``: the worker's copy of every running request, updated by deltas
    instead of rebuilt each step (vLLM V1's persistent batch). A cache: rebuilt empty after
    a reload, then filled by a full ``RowDelta`` (it reports ``Resync``), so reloading it
    drains nothing (``drain_on_reload = false``).

    Device-side tables (per row: positions, item ids of each state kind, the last sampled
    token) are updated with copies enqueued on the step stream, so the step still running
    on the device keeps reading the values it was launched with (one step of overlap)."""

    @property
    def binding(self) -> Binding: ...

    def apply(self, delta: RowDelta, state: StateOps) -> None:
        """Rows and item ids; the stores run ``state.copy`` themselves."""
        ...
    def ids(self, kind: str) -> Mapping[int, Sequence[int]]:
        """Host-side item ids per row, for the backends' ``prepare``."""
        ...
