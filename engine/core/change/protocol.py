"""Two-phase commit messages between the coordinator and participants (change.md §4.3).

Every message carries the change id; participants answer duplicates idempotently.
"""

from __future__ import annotations

import msgspec

from core.change.types import ProcActions
from core.common.ids import Identity
from unit.runtime.types import ReloadReport


class Prepare(msgspec.Struct, frozen=True):
    change_id: int
    identity: Identity
    """The candidate identity; its snapshot is already materialized on this host."""
    actions: ProcActions
    coordinator_timeout_s: float


class Prepared(msgspec.Struct, frozen=True):
    change_id: int


class Drain(msgspec.Struct, frozen=True):
    change_id: int
    timeout_s: float


class Drained(msgspec.Struct, frozen=True):
    change_id: int
    aborted_requests: int


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


class Commit(msgspec.Struct, frozen=True):
    change_id: int


class Abort(msgspec.Struct, frozen=True):
    change_id: int


class Resumed(msgspec.Struct, frozen=True):
    change_id: int
    identity: Identity
    rolled_back: bool = False


MESSAGES: list[tuple[str, type[msgspec.Struct]]] = [
    ("change.prepare@1", Prepare),
    ("change.prepared@1", Prepared),
    ("change.drain@1", Drain),
    ("change.drained@1", Drained),
    ("change.pause@1", Pause),
    ("change.paused@1", Paused),
    ("change.execute@1", Execute),
    ("change.executed@1", Executed),
    ("change.commit@1", Commit),
    ("change.abort@1", Abort),
    ("change.resumed@1", Resumed),
]
