"""Messages of the control interface (doc/specs/control-plane.md §1, §5).

Clients (an agent, an operator, the RSI plane) and units that submit internal
changes talk to the control-plane process with these. They arrive at the
supervisor's control endpoint; the control host hands them to ``core.api``.
"""

from __future__ import annotations

from typing import Any

import msgspec

from core.change.eventlog import ChangeRecord
from core.change.types import ChangeMeta, ChangeResult, ChangeSet, StateRequest
from core.common.ids import Identity
from core.common.result import Diagnostic
from core.supervisor.protocol import ProcIntrospection, UnitBrief
from core.supervisor.topology import Topology


class ProcRow(msgspec.Struct, frozen=True):
    """One process, as the supervisor sees it."""

    proc: str
    state: str
    identity: Identity | None
    units: list[UnitBrief]
    local_pid: int | None = None
    """For operators and tests on that host only; never used to address the process."""


class ApiStatus(msgspec.Struct, frozen=True):
    pass


class ApiStatusReport(msgspec.Struct, frozen=True):
    identity: Identity
    state_version: int
    recovery_required: bool
    busy: bool
    queued: int
    procs: list[ProcRow]
    control_units: list[UnitBrief]


class HaveBlobs(msgspec.Struct, frozen=True):
    """Which of these content hashes the control plane lacks (upload only those)."""

    hashes: list[str]


class MissingBlobs(msgspec.Struct, frozen=True):
    hashes: list[str]


class PutBlobs(msgspec.Struct, frozen=True):
    blobs: dict[str, bytes]
    """Content hash -> content; checked against the hash before it is stored."""


class BlobsStored(msgspec.Struct, frozen=True):
    count: int


class ApiChange(msgspec.Struct, frozen=True):
    """What a client wants the system to become. ``files`` is the whole ``engine/`` tree of
    the workspace (``"engine/..." -> hash``), not a diff: ``core.api`` works out the diff."""

    files: dict[str, str] | None = None
    """``None`` leaves the code as it is."""
    config: dict[str, Any] = {}
    topology: Topology | None = None
    deps: dict[str, str] = {}
    artifacts: dict[str, str] = {}
    base: Identity | None = None
    """``None``: the identity running now."""
    meta: ChangeMeta = ChangeMeta()
    state: list[StateRequest] = []


class ApiCheck(msgspec.Struct, frozen=True):
    """Static checks only; nothing starts (modules.html ⑥: rejected before it takes a GPU)."""

    change: ApiChange


class ApiChecked(msgspec.Struct, frozen=True):
    candidate: Identity
    diagnostics: list[Diagnostic] = []


class ApiApply(msgspec.Struct, frozen=True):
    change: ApiChange
    dry_run: bool = False


class ApiApplied(msgspec.Struct, frozen=True):
    result: ChangeResult


class SubmitChange(msgspec.Struct, frozen=True):
    """A change-set a unit submits (EPLB): queued, not waited for."""

    cs: ChangeSet


class ChangeQueued(msgspec.Struct, frozen=True):
    queued: int


class ApiIntrospect(msgspec.Struct, frozen=True):
    pass


class ApiIntrospection(msgspec.Struct, frozen=True):
    procs: list[ProcIntrospection]
    control: ProcIntrospection


class ApiLogs(msgspec.Struct, frozen=True):
    since: int = 0
    """Change ids greater than this."""


class ApiLogRecords(msgspec.Struct, frozen=True):
    changes: list[ChangeRecord]
    events: list[tuple[int | None, str, dict[str, Any]]]


MESSAGES: list[tuple[str, type[msgspec.Struct]]] = [
    ("api.status@1", ApiStatus),
    ("api.status_report@1", ApiStatusReport),
    ("api.have_blobs@1", HaveBlobs),
    ("api.missing_blobs@1", MissingBlobs),
    ("api.put_blobs@1", PutBlobs),
    ("api.blobs_stored@1", BlobsStored),
    ("api.check@1", ApiCheck),
    ("api.checked@1", ApiChecked),
    ("api.apply@1", ApiApply),
    ("api.applied@1", ApiApplied),
    ("api.submit_change@1", SubmitChange),
    ("api.change_queued@1", ChangeQueued),
    ("api.introspect@1", ApiIntrospect),
    ("api.introspection@1", ApiIntrospection),
    ("api.logs@1", ApiLogs),
    ("api.log_records@1", ApiLogRecords),
]

API_REQUESTS: tuple[type[msgspec.Struct], ...] = (
    ApiStatus, HaveBlobs, PutBlobs, ApiCheck, ApiApply, SubmitChange, ApiIntrospect, ApiLogs)
"""Requests the control host hands to ``core.api``."""
