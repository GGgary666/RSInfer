"""The plan unit, the backends it records graphs over, and the worker's persistent batch
(stage2-contract.md §8, §9). The CUDA graph rules here come from the GPU probes (#17).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

import msgspec

from core.contracts.records import BatchPlan, RowDelta, StateOps
from core.contracts.sampling import SamplingParams
from core.contracts.tensor import Binding, Tensor

type Phase = Literal["prefill", "decode"]


class PlanKey(msgspec.Struct, frozen=True):
    phase: Phase
    bucket: int
    """Padded batch size; 0 = eager (prefill always is in v0)."""


class Plan(Protocol):
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
    on the device keeps reading the values it was launched with (one step of overlap).

    A rebuild loses the last sampled tokens too: the scheduler gives each row's next input
    token itself until the row has run again (``BatchPlan.from_previous``). It has them: the
    rebuild happens while paused, after ``quiesce()`` collected every result.

    The device tables follow SGLang's ``HybridReqToTokenPool``: ``binding`` holds ``req_to_token``
    (``[rows + 1, max_context_len]`` token slots of the paged kind, row ``r`` at index ``r + 1``,
    index 0 the padding row), ``slots.<kind>`` (``[rows + 1]``, the item of each slot kind, as
    ``req_index_to_mamba_index_mapping``) and ``last_token`` (``[rows + 1]``)."""

    @property
    def binding(self) -> Binding: ...

    def check(self, delta: RowDelta, plan: BatchPlan) -> str:
        """Why the plan cannot run on the batch the delta leaves (rebuilt and waiting for a full
        delta, an unknown row, a ``from_previous`` row without a token): the ``Resync`` reason;
        ``""`` when it can. Changes nothing."""
        ...
    def apply(self, delta: RowDelta, state: StateOps) -> None:
        """Rows and item ids; the stores run ``state.copy`` themselves."""
        ...
    def ids(self, kind: str) -> Mapping[int, Sequence[int]]:
        """Host-side item ids per row, for the backends' ``prepare``."""
        ...
    def sampling(self, rows: Sequence[int]) -> list[SamplingParams]: ...
    def record(self, rows: Sequence[int], tokens: Tensor) -> None:
        """The tokens just sampled for ``rows``, which ``from_previous`` reads next."""
        ...
