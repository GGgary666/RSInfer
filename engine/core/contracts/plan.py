"""The plan unit, the backends it records graphs over, and the worker's persistent batch
(stage2-contract.md §8, §9). The CUDA graph rules here come from the GPU probes (#17).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

import msgspec

from core.contracts.records import BatchPlan, RowDelta, StateOps
from core.contracts.sampling import SamplingParams
from core.contracts.state import StateDecl
from core.contracts.tensor import Binding, Tensor

type Phase = Literal["prefill", "decode"]

GRAPH = "graph"
"""Beside the phases, in what ``Backend.resolve`` returns: a decode step replayed from a graph. It
calls the decode definitions with the solutions the kernel catalog picks for a captured step
(``Catalog.select(capture=True)``), which may differ from an eager decode step's."""


class PlanKey(msgspec.Struct, frozen=True):
    phase: Phase
    """``prefill`` also for a mixed step, which runs as an extend (§8.7)."""
    bucket: int
    """A graph's captured capacity, in rows for a decode graph; 0 = eager. Prefill and mixed steps
    are eager in v0 (§9.1)."""


class Plan(Protocol):
    """One way to run the model forward for a step (stage2-contract.md §9.1): embedding, every
    layer with its state updates, the final norm and the LM head; sampling is outside. The eager
    plan and the graph plans of ``plan@1`` take the same call, as SGLang's ``EagerRunner`` and
    ``DecodeCudaGraphRunner`` do under one ``ModelRunner``."""

    def run(self, batch: PersistentBatch, plan: BatchPlan, input_ids: Tensor, positions: Tensor,
            last_index: Tensor) -> Tensor:
        """Every backend prepares for this plan's key, then the model runs eagerly or the graph
        replays; a graph copies the inputs into its fixed buffers and pads the rows to its bucket
        itself. ``input_ids`` and ``positions`` ``[tokens]`` are the step's tokens in row order,
        ``last_index`` ``[rows]`` each row's last one. Returns the logits ``[rows, vocab]`` of the
        real rows, valid until the next run of any plan of the service; the work may still be
        running on the step stream. A graph that was closed raises ``platforms.base.StaleGraph``;
        one whose dependencies changed is no longer among ``PlanService.buckets`` (§9.3)."""
        ...


class PlanService(Protocol):
    """``plan@1`` in every worker: the eager plan and the graph plans of the current configuration.
    The only unit that may record graphs (``captures_graph``). The worker picks the plan of each
    step itself, before it changes any state (§9.1)."""

    @property
    def required(self) -> bool:
        """Decode steps must replay a graph (``plan.cuda_graph = required``): the worker refuses a
        decode step no listed bucket holds, before it changes any state, instead of running it
        eagerly."""
        ...

    def buckets(self, phase: Phase) -> list[int]:
        """The graph buckets that are ready and whose dependencies hold, from the bucketing policy
        (powers of 2 by default). The eager plan, bucket 0, is always there and not listed. This is
        the dependency check of a step: the worker asks once, before it changes any state, and runs
        the plan it then gets."""
        ...

    def get(self, key: PlanKey) -> Plan:
        """The plan for ``key``; ``LookupError`` if no such plan is ready."""
        ...

    # Speculation (doc/specs/mtp.md): the forwards of a speculative step, eager. ``view`` is the
    # step as the backends see it: a verify view is a decode plan with ``draft_tokens`` D and D
    # tokens per row from where each row is; a draft view is a decode or prefill plan of the draft
    # model's tokens, with ``draft_tokens`` 0.

    def prefill_hidden(self, batch: PersistentBatch, plan: BatchPlan, input_ids: Tensor, positions: Tensor,
                       last_index: Tensor) -> tuple[Tensor, Tensor]:
        """``run`` for a prefill step that also returns every token's hidden state."""
        ...

    def verify(self, batch: PersistentBatch, view: BatchPlan, input_ids: Tensor,
               positions: Tensor) -> tuple[Tensor, Tensor]:
        """The target model over each row's D tokens: logits and hidden states ``[rows * D, ...]``. The GDN
        layers keep their states after each token in the intermediate buffers and change no row's state
        but its conv window, which ``commit`` puts right."""
        ...

    def draft(self, batch: PersistentBatch, view: BatchPlan, input_ids: Tensor, positions: Tensor, hidden: Tensor,
              last_index: Tensor) -> tuple[Tensor, Tensor]:
        """The draft model (``DraftModel.draft_forward``) after only the attention backend prepared ``view``."""
        ...

    def commit(self, steps: Tensor, track_steps: Tensor) -> None:
        """After the last ``verify``: each lane's GDN state after its step ``steps[i]`` into its slot, after
        ``track_steps[i]`` (``>= 0``) into its tracking slot (``GDNBackend.commit``)."""
        ...


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
    selected solutions are by-value ones (§8.5, §9.3).

    A backend may also have ``fetch(input_ids, positions)``: the plan calls it after ``prepare`` and
    before the forward or the replay, outside any graph, with the step's inputs as the forward gets
    them (a bucket's padded ones). It is for what a step needs from other ranks before it runs and
    no graph can record (DS 1017's Engram rows, doc/specs/v1-ds1017.md stage 3); it writes into fixed
    buffers as ``prepare`` does."""

    def release_graphs(self) -> None:
        """Drop the fixed buffers of the graph buckets (``binding`` moves on). The plan calls it once
        the graphs recorded over them are closed."""
        ...

    @property
    def binding(self) -> Binding: ...

    def resolve(self, states: Sequence[StateDecl]) -> Mapping[str, Mapping[str, str]]:
        """Once, at start (§5.5.3, §8.1): fix the solution of every kernel definition each phase calls,
        from the backend configured for that phase and the shapes in ``states`` (every layer's
        declarations); ``forward`` then calls only those. A backend named explicitly that has no
        fitting solution for one of them fails the start; ``auto`` may fall back to the reference.
        Returns, per phase, the backend asked for and each definition's solution, for the log; and
        under ``GRAPH`` the decode definitions' solutions in a captured step, ``""`` for one that
        has no capturable solution (the plan then records no graph)."""
        ...

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


class GDNBackend(Backend, Protocol):  # the verify step's commit is ``commit`` below
    """``layers.gdn@1``: gated delta net (SGLang's ``RadixLinearAttention`` and ``GDNAttnBackend``)."""

    def forward(self, layer: int, projected_qkvz: Tensor, projected_ba: Tensor, conv_weights: Tensor,
                bias: Tensor | None, activation: str, A_log: Tensor, dt_bias: Tensor, num_k_heads: int,
                num_v_heads: int, head_k_dim: int, head_v_dim: int) -> tuple[Tensor, Tensor]:
        """``projected_qkvz`` ``[tokens, conv_dim + num_v_heads * head_v_dim]`` and ``projected_ba``
        ``[tokens, 2 * num_v_heads]`` are the layer's two input projections as the GEMMs leave them:
        ``[q | k | v | z]`` and the gate inputs ``[b | a]``. The backend runs everything of the GDN
        in between, once each: taking the projections apart (in a kernel, or as views), the causal
        conv of ``[q | k | v]`` and its activation, the per-head L2 norm of q and k (it may be fused
        into the kernel), the gating of ``a`` and ``b`` into the decay and the update rate, the gated
        delta rule, and both state updates, conv and recurrent. The weights come from the weight
        store; the backend only reads them. Returns the core output ``[1, tokens, num_v_heads,
        head_v_dim]`` and ``z`` ``[tokens, num_v_heads, head_v_dim]``, which may be a view of
        ``projected_qkvz``; the gated RMSNorm with ``z`` and the output projection are the model's."""
        ...

    def commit(self, steps: Tensor, track_steps: Tensor) -> None:
        """After a verify step (``prepare`` of a decode plan with ``draft_tokens``; doc/specs/mtp.md):
        every layer's state after lane ``i``'s step ``steps[i]`` into the lane's slot, and where
        ``track_steps[i] >= 0`` the state after that step into its ``StateTrack`` slot. On the device."""
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

    def prepare(self, kinds: Sequence[str]) -> None:
        """Allocate the ``slots.<kind>`` tables of ``kinds`` now rather than at the first item of
        each: a graph records the tables' addresses."""
        ...

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

    def track(self, kind: str) -> tuple[list[int], list[int], list[int]]:
        """The step's ``StateTrack`` of ``kind`` (rows, destination items, positions), for the
        backends' ``prepare``."""
        ...

    def ids(self, kind: str) -> Mapping[int, Sequence[int]]:
        """Host-side item ids per row, for the backends' ``prepare``."""
        ...
    def sampling(self, rows: Sequence[int]) -> list[SamplingParams]: ...
    def record(self, rows: Sequence[int], tokens: Tensor) -> None:
        """The tokens just sampled for ``rows``, which ``from_previous`` reads next."""
        ...

    # Speculation (doc/specs/mtp.md): ``spec.drafts`` ``[rows + 1, D - 1]`` and ``spec.hidden``
    # ``[rows + 1, hidden]`` in ``binding``, a row's drafts for its next verify and the draft model's
    # hidden state they continue from.

    def prepare_spec(self, draft_tokens: int, hidden_size: int, dtype: str) -> None:
        """Allocate the speculation tables (once; they keep their addresses)."""
        ...

    def committed(self, row: int) -> int | None:
        """Where the row is, as its last speculative step left it (``None``: not since it was added)."""
        ...

    def has_drafts(self, row: int) -> bool:
        """The row's ``spec.drafts`` and ``spec.hidden`` are its own, from its last step."""
        ...

    def speculated(self, rows: Sequence[int], committed: Sequence[int]) -> None:
        """A speculative step left ``rows`` at ``committed`` with drafts for their next verify."""
        ...
