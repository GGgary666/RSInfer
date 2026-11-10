"""Change-sets, action plans and results (doc/specs/change.md §3, §5)."""

from __future__ import annotations

from typing import Any, Literal

import msgspec

from core.common.ids import Identity, ProcId, UnitId
from core.common.result import Diagnostic
from core.supervisor.topology import Topology
from unit.runtime.types import ProcessPlan, ReloadPlan, ReloadReport, UnitSpec

ChangeSource = Literal["apply", "config", "internal", "startup", "operator"]
ChangeStatus = Literal["committed", "aborted", "failed", "planned", "dropped"]
WhenBusy = Literal["refuse", "wait", "force"]

EMPTY = Identity("")
"""The identity of the empty system; startup changes start here."""


class ChangeMeta(msgspec.Struct, frozen=True):
    submitter: str = ""
    """Self-reported; v0 has no authentication."""
    intent: str = ""
    evidence: list[str] = []
    handles_fault: bool = False
    """Responds to a lost process or machine: goes to the front of the queue."""
    when_busy: WhenBusy = "refuse"
    """A change applies to an empty engine (doc/specs/empty-apply.md): what to do when requests
    are in it. ``refuse``: fail and say how to wait or force; ``wait``: take no new requests and
    apply once the ones in it ended; ``force``: end them at once with an error. The units that
    hold requests do it (their ``drain`` action); a change of live config keys only does not wait."""
    wait_s: float = 3600.0
    """With ``wait``: how long; then the change fails, nothing is ended and requests come in again."""


class StateRequest(msgspec.Struct, frozen=True):
    """Ask the state units ``unit`` to change what they hold, through their own action
    (``relocate`` for EPLB): state contents, not part of what the system is."""

    unit: UnitId
    action: str
    args: dict[str, Any] = {}


class ChangeSet(msgspec.Struct, frozen=True):
    base: Identity
    files: dict[str, str | None] = {}
    """Snapshot path -> content hash of the new content; ``None`` deletes."""
    config: dict[str, Any] = {}
    deps: dict[str, str] = {}
    topology: Topology | None = None
    artifacts: dict[str, str] = {}
    source: ChangeSource = "apply"
    meta: ChangeMeta = ChangeMeta()
    computed_under: Identity | None = None
    """Internal changes: the identity they were computed under; stale ones are dropped."""
    state_only: bool = False
    """Internal changes that only change state contents (EPLB): no new identity."""
    state: list[StateRequest] = []
    """State requests; the only content a ``state_only`` change may carry."""


class UnitAction(msgspec.Struct, frozen=True):
    """A unit-provided action (rebuild graphs, refit, grow, relocate, drain, ...)."""

    unit: UnitId
    action: str
    args: dict[str, Any] = {}


class ProcActions(msgspec.Struct, frozen=True):
    """What one participant process does for one change."""

    proc: ProcId
    config: dict[str, Any] = {}
    """Level 1: values taken at the epoch boundary."""
    stage: list[UnitSpec] = []
    """New units loaded during prepare, next to the running ones."""
    reload: ReloadPlan | None = None
    """Level 2: unit reloads at the epoch boundary."""
    actions: list[UnitAction] = []
    """Levels 3–4: derived artifacts and state actions, by the units that own them."""
    drain: list[UnitAction] = []
    """Drain the affected requests before the pause."""
    restart: ProcessPlan | None = None
    """Level 5: replace the process instead of changing it in place."""
    leads_epochs: bool = False
    """This process decides epochs (the scheduler host); it pauses first."""
    plan_after: ProcessPlan | None = None
    """What the process is once the change commits; the supervisor restarts it from this."""


class ActionPlan(msgspec.Struct, frozen=True):
    """The plan ``core.deps`` derives; ``--dry-run`` returns it, and it is what runs."""

    procs: list[ProcActions] = []
    control: ProcActions | None = None
    """For the control-plane process itself (its control-plane units), run by the coordinator."""
    spawn: list[ProcessPlan] = []
    """New processes (startup, topology changes)."""
    stop: list[ProcId] = []
    topology: Topology | None = None
    restart_engine: bool = False
    drain_timeout_s: float | None = None
    notes: list[str] = []

    def participants(self) -> list[ProcActions]:
        return [p for p in self.procs if p.restart is None]


class ChangeResult(msgspec.Struct, frozen=True):
    status: ChangeStatus
    change_id: int
    old_identity: Identity
    new_identity: Identity | None
    plan: ActionPlan | None
    timings: dict[str, float]
    drain_outcomes: dict[str, int] = {}
    """Requests the drain did not see finish, by what the draining units did with them (§4.2)."""
    restarted: list[ProcId] = []
    reports: dict[str, ReloadReport] = {}
    """Per participant (``str(ProcId)``): what its unit runtime actually did."""
    state_version: int = 0
    diagnostics: list[Diagnostic] = []
