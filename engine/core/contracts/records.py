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
    """Set by the scheduler at admission when the client gave none. The worker draws from
    (seed, position) and keeps no random state: the same tokens however requests are batched,
    and a rebuilt persistent batch samples what the old one would have (§4.3)."""


# ------------------------------------------------------------------ scheduler -> worker


class AddedRow(msgspec.Struct, frozen=True):
    row: int
    """Index into the worker's persistent batch (``worker.batch``)."""
    request: int
    sampling: SamplingParams = SamplingParams()


class RowDelta(msgspec.Struct, frozen=True):
    """How the persistent batch changes before this step. Deltas arrive in order on a
    lossless stream; ``full`` drops every row first (the answer to ``Resync``). A full delta
    re-adds every row the scheduler keeps, and the plan's ``StateOps`` assigns each of them
    every item it holds, not only new ones."""

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
    """``i32`` per row: prompt tokens of this chunk, or 1 for a decode step. Speculation comes
    with its own section version (§4.4 rule 7), which says how drafts count here."""
    positions: Column
    """``i32`` per row: position of the row's first token in this step."""
    tokens: Column
    """``i64``, flat: the input tokens of the rows whose ``from_previous`` is 0, in row order:
    such a row takes the next ``num_tokens`` of them, a ``from_previous`` row none (``check``)."""
    from_previous: Column
    """``u8`` per row: 1 = the row's single input token is the token the worker sampled for it
    the last time the row ran, resolved on the device without a host round trip. Only for a
    row that has run since it was last added (a full delta adds every row again) and whose
    last run ended its prompt or decoded; otherwise its token is given in ``tokens``. A worker
    with no token to resolve answers ``Resync`` and runs nothing."""
    phase: Literal["prefill", "decode", "mixed"]
    bucket: int
    """Plan bucket (padded batch size) chosen by the bucketing policy; 0 = eager."""

    def check(self) -> list[str]:
        """How the columns fit together; empty if they do. Every per-row column has one value
        per row, a ``from_previous`` row has ``num_tokens == 1`` and takes nothing from
        ``tokens``, and ``tokens`` holds exactly the other rows' ``num_tokens``."""
        n = len(self.rows)
        errors = [f"{name} has {len(col)} values for {n} rows"
                  for name, col in (("num_tokens", self.num_tokens), ("positions", self.positions),
                                    ("from_previous", self.from_previous)) if len(col) != n]
        if errors:
            return errors
        given = 0
        for row, count, previous in zip(self.rows.values(), self.num_tokens.values(), self.from_previous.values()):
            if previous and count != 1:
                errors.append(f"row {row} takes the previous token but computes {count} tokens")
            elif not previous:
                given += int(count)
        if len(self.tokens) != given:
            errors.append(f"tokens has {len(self.tokens)} values, the rows without from_previous take {given}")
        return errors


class StateAssign(msgspec.Struct, frozen=True):
    """Items of one state kind newly given to rows: pages for paged kinds, a slot at admission
    for slot kinds. Row ``rows[i]`` appends ``ids[offsets[i]:offsets[i + 1]]``."""

    kind: str
    rows: Column
    offsets: Column
    ids: Column


class StateCopy(msgspec.Struct, frozen=True):
    """Item copies of one state kind: restore a snapshot into a row's slot on a prefix hit
    (``before`` the step, from ``admit``), or snapshot a slot at a block boundary (``after``
    it, from ``snapshot``). ``after`` sees only the state at the end of the row's chunk in
    this step: a snapshot boundary the chunk crosses inside it cannot be copied (§5.4)."""

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
    """The worker lost its persistent batch (``worker.batch`` was rebuilt), or got a plan it
    cannot apply to it: send ``full``.

    Until a full delta arrives the worker runs no plan: it answers each with ``Resync`` and no
    ``Sampled``. The scheduler counts every step answered so as not run (with overlap, the next
    one may already be on its way): its tokens were not computed and its copies not made, the
    requests' lengths go back to the last step that returned ``Sampled``. Items assigned in
    those steps stay with their requests; the full delta assigns them again. Further
    ``Resync`` answers, to plans sent before the full delta, are expected."""

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
