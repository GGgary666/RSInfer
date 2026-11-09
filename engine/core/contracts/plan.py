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
    """An attention or GDN backend (``layers.attn@1``, ``layers.gdn@1``), split the way SGLang
    splits it (stage2-contract.md §8). The backend prepares the step's metadata, runs the layer's
    computation and updates its state kind; the worker orders the steps and the plan owns graphs.

    ``prepare`` runs outside any graph, once per step and for that step's plan (``key``): SGLang's
    ``init_forward_metadata`` for an eager plan, ``init_forward_metadata_replay_cuda_graph`` for a
    graph bucket. It is not limited to the host: it may call a library's planning, enqueue copies
    or update device tables, writing into the backend's own fixed buffers in the order of the
    step stream (§8.4). It never appends the step's K/V, advances a GDN state or restores one; the
    worker's state copies do the restoring. ``forward`` runs the layer eagerly or under capture,
    with the metadata of the last ``prepare``. A mixed step runs as an extend, eagerly (§8.7).

    The fixed buffers are the graph's by-address dependencies (``binding``); the bucket and the
    selected solutions are by-value ones (§8.5, §9.3)."""

    @property
    def binding(self) -> Binding: ...

    def prepare(self, batch: PersistentBatch, plan: BatchPlan, key: PlanKey) -> None: ...


class AttentionBackend(Backend, Protocol):
    """``layers.attn@1``: full attention (SGLang's ``RadixAttention`` and its backend)."""

    def forward(self, layer: int, q: Tensor, k: Tensor, v: Tensor, num_heads: int, num_kv_heads: int,
                head_dim: int, scaling: float) -> Tensor:
        """``q`` ``[tokens, num_heads * head_dim]``, ``k`` and ``v`` ``[tokens, num_kv_heads * head_dim]``,
        the step's tokens in row order. The model has already applied the Q/K RMSNorm and RoPE; the
        backend writes this step's K/V at the slots the scheduler gave the rows (it allocates none)
        and attends. Returns ``[tokens, num_heads * head_dim]``; an output gate, if the model has
        one, is the model's."""
        ...


class GDNBackend(Backend, Protocol):
    """``layers.gdn@1``: gated delta net (SGLang's ``RadixLinearAttention`` and ``GDNAttnBackend``)."""

    def forward(self, layer: int, mixed_qkv: Tensor, a: Tensor, b: Tensor, conv_weights: Tensor,
                bias: Tensor | None, activation: str, A_log: Tensor, dt_bias: Tensor, num_k_heads: int,
                num_v_heads: int, head_k_dim: int, head_v_dim: int) -> Tensor:
        """``mixed_qkv`` ``[tokens, conv_dim]`` is the input projection before the convolution, ``a``
        and ``b`` ``[tokens, num_v_heads]`` the gate inputs. The backend runs everything of the GDN
        in between, once each: the causal conv and its activation, the per-head L2 norm of q and k
        (it may be fused into the kernel), the gating of ``a`` and ``b`` into the decay and the
        update rate, the gated delta rule, and both state updates, conv and recurrent. The weights
        come from the weight store; the backend only reads them. Returns the core output
        ``[1, tokens, num_v_heads, head_v_dim]``; the gated RMSNorm with ``z`` and the output
        projection are the model's."""
        ...


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
    ``req_index_to_mamba_index_mapping``) and ``last_token`` (``[rows + 1]``). Like rows, items are
    shifted by one: item ``i`` is page ``i + 1`` (token slots ``(i + 1) * page_size + j``) or slot
    ``i + 1`` of the store, so the padding row's zeros point at the padding items (``StateStore``)."""

    @property
    def binding(self) -> Binding: ...

    def check(self, delta: RowDelta, plan: BatchPlan) -> str:
        """Why the plan cannot run on the batch the delta leaves (built or rebuilt and not yet
        given a full delta, an unknown row, a ``from_previous`` row without a token): the
        ``Resync`` reason; ``""`` when it can. Changes nothing."""
        ...
    def apply(self, delta: RowDelta, state: StateOps) -> None:
        """Rows and item ids; the stores run ``state.copy`` themselves. A delta beyond the batch's
        limits (its rows, the tokens one row holds) raises and changes nothing: the scheduler
        admitted what this worker cannot hold, which a full delta would not mend."""
        ...
    def ids(self, kind: str) -> Mapping[int, Sequence[int]]:
        """Host-side item ids per row, for the backends' ``prepare``."""
        ...
    def sampling(self, rows: Sequence[int]) -> list[SamplingParams]: ...
    def record(self, rows: Sequence[int], tokens: Tensor) -> None:
        """The tokens just sampled for ``rows``, which ``from_previous`` reads next."""
        ...
