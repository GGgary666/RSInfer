"""Per-step records between the scheduler host, the workers and the frontend
(stage2-contract.md §4). Each class is one section of an ``OpenRecord`` (doc/specs/ipc.md §3),
named by ``SECTION``; a receiver decodes the sections it knows.

The epoch of a record is its step number, assigned by the scheduler (change.md 4.2 step 4).
The scheduler may issue step ``N + 1`` before it has the result of step ``N`` (one step of
overlap, §4.4), so nothing in a plan may need a value only the result of the previous step
carries: a decode input token is named, not given (``BatchPlan.from_previous``).
"""

from __future__ import annotations

from typing import ClassVar, Literal

import msgspec

from core.contracts.columns import Column

STEP_PLAN = "sched.step_plan@1"
"""Schema of the scheduler -> worker record, one per rank (P3)."""
STEP_RESULT = "worker.step_result@1"
"""Schema of the worker -> scheduler record."""
REQUESTS = "frontend.requests@1"
"""Schema of the frontend -> scheduler record."""
TOKENS = "sched.tokens@1"
"""Schema of the scheduler -> frontend record."""


class SamplingParams(msgspec.Struct, frozen=True):
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int | None = None
    """Fixed per request: the reference mode samples the same tokens however requests are batched."""


# ------------------------------------------------------------------ scheduler -> worker


class AddedRow(msgspec.Struct, frozen=True):
    row: int
    """Index into the worker's persistent batch (``worker.batch``)."""
    request: int
    sampling: SamplingParams = SamplingParams()


class RowDelta(msgspec.Struct, frozen=True):
    """How the persistent batch changes before this step. Deltas arrive in order on a
    lossless stream; ``full`` drops every row first (the answer to ``Resync``)."""

    SECTION: ClassVar[str] = "sched.rows@1"
    added: list[AddedRow] = []
    removed: Column = Column("i32")
    full: bool = False


class BatchPlan(msgspec.Struct, frozen=True):
    """What every row computes in this step."""

    SECTION: ClassVar[str] = "sched.batch@1"
    rows: Column
    """``i32``: rows of the persistent batch that run in this step."""
    num_tokens: Column
    """``i32`` per row: prompt tokens of this chunk, or 1 for a decode step (more with speculation)."""
    positions: Column
    """``i32`` per row: position of the row's first token in this step."""
    tokens: Column
    """``i64``, flat: the input tokens of the rows whose ``from_previous`` is 0, in row order."""
    from_previous: Column
    """``u8`` per row: 1 = the row's single input token is the token the worker sampled for it
    the last time the row ran, resolved on the device without a host round trip."""
    phase: Literal["prefill", "decode", "mixed"]
    bucket: int
    """Plan bucket (padded batch size) chosen by the bucketing policy; 0 = eager."""


class StateAssign(msgspec.Struct, frozen=True):
    """Items of one state kind newly given to rows: pages for paged kinds, a slot at admission
    for slot kinds. Row ``rows[i]`` appends ``ids[offsets[i]:offsets[i + 1]]``."""

    kind: str
    rows: Column
    offsets: Column
    ids: Column


class StateCopy(msgspec.Struct, frozen=True):
    """Item copies of one state kind: restore a snapshot into a row's slot on a prefix hit
    (``before`` the step), or snapshot a slot at a block boundary (``after`` it)."""

    kind: str
    when: Literal["before", "after"]
    src: Column
    dst: Column


class StateOps(msgspec.Struct, frozen=True):
    """Every state kind's part of a step plan, in one section."""

    SECTION: ClassVar[str] = "state.ops@1"
    assign: list[StateAssign] = []
    copy: list[StateCopy] = []


# ------------------------------------------------------------------ worker -> scheduler


class Sampled(msgspec.Struct, frozen=True):
    """Tokens sampled in step ``epoch`` (the record's epoch). ``counts`` is 1 per row without
    speculation, the accepted length plus one with it."""

    SECTION: ClassVar[str] = "worker.sampled@1"
    rows: Column
    counts: Column
    tokens: Column


class StepStats(msgspec.Struct, frozen=True):
    """Where the time of one step went; how an agent sees that the CPU is the bottleneck.
    Device times come from events, so they may arrive one record late (``step``)."""

    SECTION: ClassVar[str] = "worker.stats@1"
    step: int
    host_us: float
    """Worker CPU time for the step: apply the delta, prepare, launch."""
    device_us: float
    idle_before_us: float
    """How long the device sat idle between the previous step's work and this one's."""


class Capacity(msgspec.Struct, frozen=True):
    """Usable items per state kind, scratch excluded; sent at start and after a grow."""

    SECTION: ClassVar[str] = "state.capacity@1"
    items: dict[str, int]


class Resync(msgspec.Struct, frozen=True):
    """The worker lost its persistent batch (``worker.batch`` was rebuilt): send ``full``."""

    SECTION: ClassVar[str] = "worker.resync@1"
    reason: str = ""


# ------------------------------------------------------------------ frontend <-> scheduler


class NewRequest(msgspec.Struct, frozen=True):
    request: int
    """Assigned by the frontend; unique while the request lives."""
    prompt: Column
    """``i64`` token ids."""
    sampling: SamplingParams = SamplingParams()
    max_tokens: int = 256
    stop_token_ids: list[int] = []


class AbortRequest(msgspec.Struct, frozen=True):
    request: int
    reason: str = ""
    """E.g. a stop string the detokenizer found, or the client went away."""


class Requests(msgspec.Struct, frozen=True):
    SECTION: ClassVar[str] = "frontend.batch@1"
    new: list[NewRequest] = []
    abort: list[AbortRequest] = []


type FinishReason = Literal["stop", "length", "abort", "version_switch"]


class TokenEvents(msgspec.Struct, frozen=True):
    """New tokens of one step for the frontend; ``requests[i]`` owns
    ``tokens[offsets[i]:offsets[i + 1]]``. With overlap a request may get one token past
    its stop condition; the scheduler drops it before it gets here."""

    SECTION: ClassVar[str] = "sched.events@1"
    identity: str
    """The identity the step ran under (the ``x-engine-identity`` header, v0-plan 5.8)."""
    requests: Column
    offsets: Column
    tokens: Column
    finished: dict[int, FinishReason] = {}
