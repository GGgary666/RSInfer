"""The participant side of a change, inside every process host (change.md §4).

Requests are served at safe points only. Replies that must wait (drained,
paused at the epoch boundary) are deferred and sent from a later safe point.
Before execution, coordinator timeout rolls back and resumes serving. After
execution, a decision may already exist elsewhere: stay paused until an explicit
Abort or Commit/Resume arrives. Recovery of a lost coordinator requires an engine
restart; timeout alone cannot establish that rollback is safe.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import msgspec

from core.change.protocol import (
    Abort,
    Commit,
    Committed,
    Drain,
    Drained,
    Execute,
    Executed,
    Pause,
    Paused,
    Prepare,
    Prepared,
    Resume,
    Resumed,
)
from core.change.types import ProcActions, UnitAction
from core.common.ids import Identity, UnitId
from core.common.messages import Envelope
from core.common.result import Diagnostic, Failed, Ok, Result, diag, fail
from core.ipc.rpc import Deferred
from unit.runtime.types import ReloadPlan, ReloadReport, UnitSpec

if TYPE_CHECKING:
    from unit.runtime.host import ProcessHost

COMPONENT = "change.participant"

Phase = Literal["prepared", "draining", "drained", "pausing", "paused", "executed", "committed"]


class ChangeTunables(msgspec.Struct, frozen=True):
    coordinator_timeout_s: float = 60.0
    drain_timeout_s: float = 10.0
    blob_retention: int = 20
    commit_attempts: int = 3
    commit_retry_delay_s: float = 0.1


@dataclass
class _Undo:
    """What execute changed, so that abort can put it back."""

    reverse_reload: ReloadPlan | None = None
    old_config: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass
class _Change:
    id: int
    identity: Identity
    actions: ProcActions
    deadline: float
    timeout_s: float
    phase: Phase = "prepared"
    prepared_actions: list[UnitAction] = field(default_factory=list[UnitAction])
    executed_actions: list[UnitAction] = field(default_factory=list[UnitAction])
    undo: _Undo = field(default_factory=_Undo)
    report: ReloadReport | None = None
    drain_reply: Deferred | None = None
    drain_deadline: float = 0.0
    aborted_requests: int = 0
    pause_reply: Deferred | None = None
    pause_after: int | None = None


class Participant:
    def __init__(self, host: ProcessHost, tun: ChangeTunables) -> None:
        self.host = host
        self.rt = host.runtime
        self.tun = tun
        self.change: _Change | None = None
        self.finished: dict[int, Resumed] = {}
        """Replies to Commit/Abort of finished changes, for duplicate messages."""
        self._initial_resume_pending = host.spec.start_paused
        self.diagnostics: list[Diagnostic] = []
        for type_, handler in [(Prepare, self.on_prepare), (Drain, self.on_drain), (Pause, self.on_pause),
                               (Execute, self.on_execute), (Commit, self.on_commit), (Resume, self.on_resume),
                               (Abort, self.on_abort)]:
            host.handlers[type_] = handler
        host.safe_point_hooks.append(self.tick)

    # ----------------------------------------------------------- helpers

    def _handler(self, a: UnitAction) -> Any:
        h = self.rt.action_handler(a.unit, a.action)
        if h is None:
            raise LookupError(f"{a.unit} has no handler for action {a.action!r}")
        return h

    def _code_root_for(self, identity: Identity) -> str | None:
        """Snapshots of one host are siblings: ``<root>/<identity>/engine``."""
        root = self.rt.code_root
        if root is None:
            return None
        candidate = Path(root).parent.parent / identity / "engine"
        return str(candidate) if candidate.is_dir() else None

    def _current(self, change_id: int) -> _Change | Failed:
        c = self.change
        if c is None or c.id != change_id:
            return fail(COMPONENT, "protocol", f"change {change_id} is not in progress here",
                        subject=str(self.host.spec.proc), code="change.unknown_change")
        c.deadline = time.monotonic() + c.timeout_s  # the coordinator is alive
        return c

    def _fail(self, phase: str, e: BaseException | str) -> Failed:
        reason = e if isinstance(e, str) else f"{type(e).__name__}: {e}"
        return fail(COMPONENT, phase, reason, subject=str(self.host.spec.proc), code=f"change.{phase}_failed")

    # ----------------------------------------------------------- prepare

    def on_prepare(self, env: Envelope, msg: Prepare) -> Result[msgspec.Struct]:
        c = self.change
        if c is not None:
            if c.id == msg.change_id:
                return Ok(Prepared(msg.change_id))
            return fail(COMPONENT, "prepare", f"busy with change {c.id}", code="change.busy")
        if msg.change_id in self.finished:
            return fail(COMPONENT, "prepare", "change already finished", code="change.finished")
        c = _Change(msg.change_id, msg.identity, msg.actions, time.monotonic() + msg.coordinator_timeout_s,
                    msg.coordinator_timeout_s)
        a = msg.actions
        staged = False
        try:
            if a.config:
                r = self.rt.config.stage(a.config)
                if isinstance(r, Failed):
                    return r
            if a.stage:
                changed = a.reload.changed_modules if a.reload else []
                r = self.rt.stage_new_units(a.stage, code_root=self._code_root_for(msg.identity),
                                            changed_modules=changed)
                if isinstance(r, Failed):
                    self.rt.config.discard_staged()
                    return r
                staged = True
            for ua in a.actions:
                self._handler(ua).prepare(ua.args)
                c.prepared_actions.append(ua)
        except Exception as e:  # noqa: BLE001 - a failing unit fails the change, not the process
            self._abort_prepared(c, staged)
            return self._fail("prepare", e)
        self.change = c
        return Ok(Prepared(msg.change_id))

    def _abort_prepared(self, c: _Change, staged: bool) -> None:
        for ua in reversed(c.prepared_actions):
            try:
                self._handler(ua).abort()
            except Exception as e:  # noqa: BLE001
                self.diagnostics.append(diag(COMPONENT, "abort", f"{ua.unit}.{ua.action}: {e}",
                                             code="change.abort_failed"))
        self.rt.config.discard_staged()
        if staged:
            self.rt.discard_staged()

    # ------------------------------------------------------- drain/pause

    def on_drain(self, env: Envelope, msg: Drain) -> Result[msgspec.Struct] | Deferred:
        c = self._current(msg.change_id)
        if isinstance(c, Failed):
            return c
        if c.phase != "prepared":
            return Ok(Drained(c.id, c.aborted_requests))
        try:
            for ua in c.actions.drain:
                self._handler(ua).start_drain(ua.args)
        except Exception as e:  # noqa: BLE001
            return self._fail("drain", e)
        c.phase = "draining"
        c.drain_deadline = time.monotonic() + msg.timeout_s
        reply = c.drain_reply = self.host.rpc.defer()
        self._check_drain(c)  # may already resolve it
        return reply

    def _check_drain(self, c: _Change) -> None:
        try:
            done = all(self._handler(ua).drained() for ua in c.actions.drain)
            if not done and time.monotonic() >= c.drain_deadline:
                c.aborted_requests = sum(int(self._handler(ua).abort_remaining()) for ua in c.actions.drain)
                done = True
        except Exception as e:  # noqa: BLE001
            if c.drain_reply is not None:
                c.drain_reply.resolve(self._fail("drain", e))
            return
        if done and c.drain_reply is not None:
            c.phase = "drained"
            c.drain_reply.resolve(Ok(Drained(c.id, c.aborted_requests)))
            c.drain_reply = None

    def on_pause(self, env: Envelope, msg: Pause) -> Result[msgspec.Struct] | Deferred:
        c = self._current(msg.change_id)
        if isinstance(c, Failed):
            return c
        if c.phase in ("paused", "executed", "committed"):
            return Ok(Paused(c.id, self.host.epoch))
        c.phase = "pausing"
        c.pause_after = msg.after_epoch
        reply = c.pause_reply = self.host.rpc.defer()
        self._check_pause(c)  # may already resolve it
        return reply

    def _check_pause(self, c: _Change) -> None:
        epoch = self.host.epoch
        if c.pause_after is not None and epoch is not None and epoch < c.pause_after:
            return  # keep stepping until the agreed epoch is done
        self.host.paused = True
        c.phase = "paused"
        if c.pause_reply is not None:
            c.pause_reply.resolve(Ok(Paused(c.id, epoch)))
            c.pause_reply = None

    # ----------------------------------------------------------- execute

    def on_execute(self, env: Envelope, msg: Execute) -> Result[msgspec.Struct]:
        c = self._current(msg.change_id)
        if isinstance(c, Failed):
            return c
        if c.phase == "executed":
            return Ok(Executed(c.id, c.report))
        if c.phase != "paused":
            return fail(COMPONENT, "execute", f"not paused (phase {c.phase})", code="change.not_paused")
        a = c.actions
        try:
            if a.config:
                c.undo.old_config = {k: self.rt.config.entries()[k].value for k in a.config}
                commit = self.rt.config.commit()
                if commit.hook_failures:
                    raise RuntimeError("; ".join(d.reason for d in commit.hook_failures))
            if a.reload is not None:
                reverse = self._reverse(a.reload)
                plan = a.reload
                if plan.new_code_root is None and (root := self._code_root_for(c.identity)) is not None \
                        and root != self.rt.code_root:
                    plan = msgspec.structs.replace(plan, new_code_root=root)
                r = self.rt.reload(plan)
                if isinstance(r, Failed):
                    raise _Failed(r)
                c.report = r.value
                c.undo.reverse_reload = reverse
            for ua in a.actions:
                self._handler(ua).execute(ua.args)
                c.executed_actions.append(ua)
        except Exception as e:  # noqa: BLE001
            self._undo_execute(c)
            c.phase = "paused"
            return e.failed if isinstance(e, _Failed) else self._fail("execute", e)
        c.phase = "executed"
        return Ok(Executed(c.id, c.report))

    def _reverse(self, plan: ReloadPlan) -> ReloadPlan:
        old_reload = [s for u in plan.units_to_reload if (s := self.rt.spec_of(u.id)) is not None]
        removed: list[UnitSpec] = [s for u in plan.units_to_remove if (s := self.rt.spec_of(u)) is not None]
        added = [UnitId(s.id) for s in plan.units_to_add]
        staged = [UnitId(u.unit) for u in self.rt.snapshot().units if u.staged]
        return ReloadPlan(
            changed_modules=plan.changed_modules, units_to_reload=old_reload, units_to_add=removed,
            units_to_remove=sorted(set(added) | set(staged)), new_code_root=self.rt.code_root,
        )

    def _undo_execute(self, c: _Change) -> None:
        for ua in reversed(c.executed_actions):
            try:
                self._handler(ua).abort()
            except Exception as e:  # noqa: BLE001
                self.diagnostics.append(diag(COMPONENT, "abort", f"{ua.unit}.{ua.action}: {e}",
                                             code="change.abort_failed"))
        c.executed_actions.clear()
        if c.undo.reverse_reload is not None:
            r = self.rt.reload(c.undo.reverse_reload)
            if isinstance(r, Failed):
                self.diagnostics += r.diagnostics
            c.undo.reverse_reload = None
        if c.undo.old_config:
            self.rt.config.stage(c.undo.old_config)
            self.rt.config.commit()
            c.undo.old_config = {}

    # ---------------------------------------------------- commit / abort

    def on_commit(self, env: Envelope, msg: Commit) -> Result[msgspec.Struct]:
        if msg.change_id in self.finished:
            done = self.finished[msg.change_id]
            if done.rolled_back:
                return fail(COMPONENT, "commit", "change was already aborted", code="change.already_aborted")
            return Ok(Committed(done.change_id, done.identity))
        c = self._current(msg.change_id)
        if isinstance(c, Failed):
            return c
        if c.phase == "committed":
            return Ok(Committed(c.id, c.identity))
        if c.phase != "executed":
            return fail(COMPONENT, "commit", f"not executed (phase {c.phase})", code="change.not_executed")
        for ua in c.actions.actions:
            try:
                self._handler(ua).commit()
            except Exception as e:  # noqa: BLE001 - commit cannot fail any more; report it
                self.diagnostics.append(diag(COMPONENT, "commit", f"{ua.unit}.{ua.action}: {e}",
                                             code="change.commit_hook_failed"))
        self.host.identity = c.identity
        c.phase = "committed"
        return Ok(Committed(c.id, c.identity))

    def on_resume(self, env: Envelope, msg: Resume) -> Result[msgspec.Struct]:
        done = self.finished.get(msg.change_id)
        if done is not None:
            if done.rolled_back or done.identity != msg.identity:
                return fail(COMPONENT, "resume", "change did not commit this identity", code="change.not_committed")
            return Ok(done)
        c = self.change
        if c is None and self._initial_resume_pending and self.host.identity == msg.identity:
            # A newly spawned/restarted process was initialized from the candidate snapshot.
            self.host.paused = False
            self._initial_resume_pending = False
            done = Resumed(msg.change_id, msg.identity)
            self.finished[msg.change_id] = done
            return Ok(done)
        if c is None or c.id != msg.change_id or c.phase != "committed" or c.identity != msg.identity:
            return fail(COMPONENT, "resume", "change has not committed", code="change.not_committed")
        return Ok(self._resume(c, rolled_back=False))

    def on_abort(self, env: Envelope, msg: Abort) -> Result[msgspec.Struct]:
        if msg.change_id in self.finished:
            return Ok(self.finished[msg.change_id])
        c = self._current(msg.change_id)
        if isinstance(c, Failed):
            return Ok(Resumed(msg.change_id, self.host.identity, rolled_back=True))
        if c.phase == "committed":
            return fail(COMPONENT, "abort", "committed decision cannot be rolled back", code="change.already_committed")
        self._rollback(c)
        return Ok(self._resume(c, rolled_back=True))

    def _rollback(self, c: _Change) -> None:
        for d in (c.drain_reply, c.pause_reply):
            if d is not None:
                d.resolve(fail(COMPONENT, "abort", "change aborted", code="change.aborted"))
        if c.phase == "executed":
            self._undo_execute(c)
        self._abort_prepared(c, staged=bool(c.actions.stage))

    def _resume(self, c: _Change, *, rolled_back: bool) -> Resumed:
        self.host.paused = False
        self.change = None
        reply = Resumed(c.id, self.host.identity, rolled_back)
        self.finished[c.id] = reply
        return reply

    # ---------------------------------------------------------- safe point

    def tick(self) -> None:
        c = self.change
        if c is None:
            return
        if c.phase == "draining":
            self._check_drain(c)
        elif c.phase == "pausing":
            self._check_pause(c)
        if time.monotonic() > c.deadline:
            if c.phase in ("executed", "committed"):
                # The decision may already be committed elsewhere. Stay fenced until
                # an explicit decision/resume arrives, or the engine is restarted.
                c.deadline = float("inf")
                self.diagnostics.append(diag(
                    COMPONENT, "timeout", f"change {c.id} is in doubt; remaining paused",
                    subject=str(self.host.spec.proc), code="change.in_doubt"))
                return
            self.diagnostics.append(diag(
                COMPONENT, "timeout", f"lost the coordinator of change {c.id} for {c.timeout_s}s; rolled back",
                subject=str(self.host.spec.proc), code="change.coordinator_lost"))
            self._rollback(c)
            self._resume(c, rolled_back=True)


class _Failed(Exception):
    def __init__(self, failed: Failed) -> None:
        super().__init__(str(failed))
        self.failed = failed
