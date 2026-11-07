"""Two-phase commit messages between the coordinator and participants (change.md §4.3).

Every message carries the change id; participants answer duplicate and late
messages idempotently (ipc.md §4).
"""

from __future__ import annotations

from typing import Literal

import msgspec

from core.change.types import ProcActions
from core.common.ids import Identity
from core.common.result import Diagnostic
from core.supervisor.protocol import UnitBrief
from unit.runtime.types import ReloadReport


class Prepare(msgspec.Struct, frozen=True):
    change_id: int
    identity: Identity
    """The candidate identity; its snapshot is already materialized on this host."""
    actions: ProcActions
    coordinator_timeout_s: float


class Prepared(msgspec.Struct, frozen=True):
    change_id: int
    diagnostics: list[Diagnostic] = []
    """E.g. preparations done in one shot, and how long serving stopped for them (§4.6)."""


class Drain(msgspec.Struct, frozen=True):
    change_id: int
    timeout_s: float


class Drained(msgspec.Struct, frozen=True):
    change_id: int
    outcomes: dict[str, int] = {}
    """What the draining units did with the requests left at the deadline, by their own policy
    (``{"aborted": 2, "requeued": 5}``); empty if everything finished in time."""


class Pause(msgspec.Struct, frozen=True):
    change_id: int
    after_epoch: int | None
    """Stop once this epoch is done; ``None``: at the next safe point."""


class Paused(msgspec.Struct, frozen=True):
    change_id: int
    epoch: int | None


class Execute(msgspec.Struct, frozen=True):
    change_id: int


class Executed(msgspec.Struct, frozen=True):
    change_id: int
    report: ReloadReport | None
    diagnostics: list[Diagnostic] = []


class Commit(msgspec.Struct, frozen=True):
    change_id: int


class Committed(msgspec.Struct, frozen=True):
    change_id: int
    identity: Identity


class Resume(msgspec.Struct, frozen=True):
    change_id: int
    identity: Identity


class Abort(msgspec.Struct, frozen=True):
    change_id: int


class Resumed(msgspec.Struct, frozen=True):
    change_id: int
    identity: Identity
    rolled_back: bool = False
    snapshots: list[Identity] = []
    """Snapshots the process still loads code from; its node agent must keep them."""
    units: list[UnitBrief] = []
    """The units it runs now: in-place changes reload units after the check-in, and ``status``
    shows these."""


class KeepAlive(msgspec.Struct, frozen=True):
    """The coordinator is alive, only busy: renews the participant's lease (§4.7)."""

    change_id: int


class Alive(msgspec.Struct, frozen=True):
    change_id: int


type DecisionKind = Literal["commit", "abort", "pending", "unknown"]


class QueryDecision(msgspec.Struct, frozen=True):
    """A participant whose lease ran out after ``Prepared`` asks for the outcome."""

    change_id: int


class Decision(msgspec.Struct, frozen=True):
    change_id: int
    decision: DecisionKind


MESSAGES: list[tuple[str, type[msgspec.Struct]]] = [
    ("change.prepare@1", Prepare),
    ("change.prepared@1", Prepared),
    ("change.drain@1", Drain),
    ("change.drained@1", Drained),
    ("change.pause@1", Pause),
    ("change.paused@1", Paused),
    ("change.execute@1", Execute),
    ("change.executed@1", Executed),
    ("change.commit@2", Commit),
    ("change.committed@1", Committed),
    ("change.resume@1", Resume),
    ("change.abort@1", Abort),
    ("change.resumed@1", Resumed),
    ("change.keepalive@1", KeepAlive),
    ("change.alive@1", Alive),
    ("change.query_decision@1", QueryDecision),
    ("change.decision@1", Decision),
]
