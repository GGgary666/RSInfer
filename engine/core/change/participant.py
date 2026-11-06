"""The participant side of a change, inside every process host (change.md §4).

Requests are served at safe points only. Replies that must wait (prepared,
drained, paused at the epoch boundary) are deferred and sent from a later safe
point.

Prepare runs in slices (§4.6): a unit's ``prepare(args)`` may return an
iterator, and every safe point advances each unfinished one by one ``next()``,
so the old version keeps serving in between. A ``prepare`` that returns
``None`` was done in one shot; how long serving stopped for it is reported.

Any coordinator message renews the participant's lease (§4.7). When the lease
runs out while still preparing, the participant withdraws by itself: it has
promised nothing yet. After ``Prepared`` it must not decide alone: it asks the
coordinator, and rolls back only if the answer is ``abort``/``unknown`` or no
answer comes. A ``Commit`` that arrives after such a rollback is refused with
``change.already_rolled_back``.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import msgspec

from core.change.config import ChangeTunables
from core.change.protocol import (
    Abort,
    Alive,
    Commit,
    Decision,
    DecisionKind,
    Drain,
    Drained,
    Execute,
    Executed,
    KeepAlive,
    Pause,
    Paused,
    Prepare,
    Prepared,
    QueryDecision,
    Resumed,
)
from core.change.types import ProcActions, UnitAction
from core.common.ids import Identity, UnitId
from core.common.messages import Envelope
from core.common.result import Diagnostic, Failed, Ok, Result, Severity, diag, fail
from core.ipc.rpc import Deferred
from unit.runtime.types import ReloadPlan, ReloadReport, UnitSpec

if TYPE_CHECKING:
    from unit.runtime.host import ProcessHost

COMPONENT = "change.participant"

Phase = Literal["preparing", "prepared", "draining", "drained", "pausing", "paused", "executed"]
type _Slices = tuple[UnitAction, Iterator[object]]


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
    phase: Phase = "preparing"
    staged: bool = False
    prepared_actions: list[UnitAction] = field(default_factory=list[UnitAction])
    """``prepare()`` was called on these: ``abort()`` undoes them."""
    slicing: list[_Slices] = field(default_factory=list[_Slices])
    """Sliced preparations not finished yet."""
    prepare_reply: Deferred | None = None
    prepare_diags: list[Diagnostic] = field(default_factory=list[Diagnostic])
    executed_actions: list[UnitAction] = field(default_factory=list[UnitAction])
    undo: _Undo = field(default_factory=_Undo)
    report: ReloadReport | None = None
    drain_reply: Deferred | None = None
    drain_deadline: float = 0.0
    aborted_requests: int = 0
    pause_reply: Deferred | None = None
    pause_after: int | None = None

    def renew(self) -> None:
        self.deadline = time.monotonic() + self.timeout_s


class Participant:
    def __init__(self, host: ProcessHost, tun: ChangeTunables) -> None:
        self.host = host
        self.rt = host.runtime
        self.tun = tun
        self.change: _Change | None = None
        self.finished: dict[int, Resumed] = {}
        """Replies to Commit/Abort of finished changes, for duplicate messages."""
        self.diagnostics: list[Diagnostic] = []
        for type_, handler in [(Prepare, self.on_prepare), (Drain, self.on_drain), (Pause, self.on_pause),
                               (Execute, self.on_execute), (Commit, self.on_commit), (Abort, self.on_abort),
                               (KeepAlive, self.on_keepalive)]:
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
        c.renew()  # the coordinator is alive
        return c

    def _fail(self, phase: str, e: BaseException | str) -> Failed:
        reason = e if isinstance(e, str) else f"{type(e).__name__}: {e}"
        return fail(COMPONENT, phase, reason, subject=str(self.host.spec.proc), code=f"change.{phase}_failed")

    def _note(self, phase: str, reason: str, code: str, severity: Severity = "warning") -> None:
        self.diagnostics.append(diag(COMPONENT, phase, reason, subject=str(self.host.spec.proc), code=code,
                                     severity=severity))

    def on_keepalive(self, env: Envelope, msg: KeepAlive) -> Result[msgspec.Struct]:
        c = self._current(msg.change_id)
        return c if isinstance(c, Failed) else Ok(Alive(c.id))

    # ----------------------------------------------------------- prepare

    def on_prepare(self, env: Envelope, msg: Prepare) -> Result[msgspec.Struct] | Deferred:
        c = self.change
        if c is not None:
            if c.id != msg.change_id:
                return fail(COMPONENT, "prepare", f"busy with change {c.id}", code="change.busy")
            c.renew()
            if c.phase != "preparing":
                return Ok(Prepared(c.id, list(c.prepare_diags)))
            if c.prepare_reply is not None:
                c.prepare_reply.resolve(fail(COMPONENT, "prepare", "superseded by a repeated Prepare",
                                             code="change.duplicate"))
            reply = c.prepare_reply = self.host.rpc.defer()
            return reply
        if msg.change_id in self.finished:
            return fail(COMPONENT, "prepare", "change already finished", code="change.finished")
        c = _Change(msg.change_id, msg.identity, msg.actions, 0.0, msg.coordinator_timeout_s)
        a = msg.actions
        try:
            if a.config:
                r = self.rt.config.stage(a.config)
                if isinstance(r, Failed):
                    return r
            if a.stage:
                t0 = time.monotonic()
                changed = a.reload.changed_modules if a.reload else []
                r = self.rt.stage_new_units(a.stage, code_root=self._code_root_for(msg.identity),
                                            changed_modules=changed)
                if isinstance(r, Failed):
                    self.rt.config.discard_staged()
                    return r
                c.staged = True
                c.prepare_diags.append(self._one_shot(f"staging {len(a.stage)} new unit(s)", t0))
            for ua in a.actions:
                t0 = time.monotonic()
                out = self._handler(ua).prepare(ua.args)
                c.prepared_actions.append(ua)
                if out is None:
                    c.prepare_diags.append(self._one_shot(
                        f"{ua.unit}.{ua.action}", t0,
                        hint="return an iterator from prepare() to prepare in slices between steps"))
                else:
                    c.slicing.append((ua, iter(out)))
        except Exception as e:  # noqa: BLE001 - a failing unit fails the change, not the process
            self._abort_prepared(c)
            return self._fail("prepare", e)
        self.change = c
        c.renew()  # work done for the coordinator does not count against its lease
        if not c.slicing:
            c.phase = "prepared"
            return Ok(Prepared(c.id, list(c.prepare_diags)))
        reply = c.prepare_reply = self.host.rpc.defer()
        return reply

    def _one_shot(self, what: str, since: float, hint: str = "") -> Diagnostic:
        took = time.monotonic() - since
        return diag(COMPONENT, "prepare", f"{what} prepared in one shot: serving stopped for {took:.3f}s",
                    subject=str(self.host.spec.proc), code="change.prepare_one_shot", hint=hint, severity="info")

    def _advance(self, c: _Change) -> None:
        """One slice of every unfinished preparation; ``Prepared`` once all are done."""
        try:
            for item in list(c.slicing):
                try:
                    next(item[1])
                except StopIteration:
                    c.slicing.remove(item)
        except Exception as e:  # noqa: BLE001
            if c.prepare_reply is not None:
                c.prepare_reply.resolve(self._fail("prepare", e))
            self._rollback(c)
            self._resume(c, rolled_back=True)
            return
        if not c.slicing:
            c.phase = "prepared"
            c.renew()
            if c.prepare_reply is not None:
                c.prepare_reply.resolve(Ok(Prepared(c.id, list(c.prepare_diags))))
                c.prepare_reply = None

    def _abort_prepared(self, c: _Change) -> None:
        for ua, it in c.slicing:
            close = getattr(it, "close", None)
            try:
                if close is not None:
                    close()
            except Exception as e:  # noqa: BLE001
                self.diagnostics.append(diag(COMPONENT, "abort", f"{ua.unit}.{ua.action}: {e}",
                                             code="change.abort_failed"))
        c.slicing.clear()
        for ua in reversed(c.prepared_actions):
            try:
                self._handler(ua).abort()
            except Exception as e:  # noqa: BLE001
                self.diagnostics.append(diag(COMPONENT, "abort", f"{ua.unit}.{ua.action}: {e}",
                                             code="change.abort_failed"))
        c.prepared_actions.clear()
        self.rt.config.discard_staged()
        if c.staged:
            self.rt.discard_staged()
            c.staged = False

    # ------------------------------------------------------- drain/pause

    def on_drain(self, env: Envelope, msg: Drain) -> Result[msgspec.Struct] | Deferred:
        c = self._current(msg.change_id)
        if isinstance(c, Failed):
            return c
        if c.phase == "preparing":
            return self._fail("drain", "not prepared yet")
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
        if c.phase in ("paused", "executed"):
            return Ok(Paused(c.id, self.host.epoch))
        if c.phase == "preparing":
            return self._fail("pause", "not prepared yet")
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
        c.renew()
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
        done = self.finished.get(msg.change_id)
        if done is not None:
            if done.rolled_back:
                return fail(COMPONENT, "commit", f"rolled back change {msg.change_id} before Commit arrived",
                            subject=str(self.host.spec.proc), code="change.already_rolled_back",
                            hint="restart this process at the new identity")
            return Ok(done)
        c = self._current(msg.change_id)
        if isinstance(c, Failed):
            return c
        if c.phase != "executed":
            return fail(COMPONENT, "commit", f"not executed (phase {c.phase})", code="change.not_executed")
        return Ok(self._commit(c))

    def _commit(self, c: _Change) -> Resumed:
        for ua in c.actions.actions:
            try:
                self._handler(ua).commit()
            except Exception as e:  # noqa: BLE001 - commit cannot fail any more; report it
                self.diagnostics.append(diag(COMPONENT, "commit", f"{ua.unit}.{ua.action}: {e}",
                                             code="change.commit_hook_failed"))
        self.host.identity = c.identity
        return self._resume(c, rolled_back=False)

    def on_abort(self, env: Envelope, msg: Abort) -> Result[msgspec.Struct]:
        if msg.change_id in self.finished:
            return Ok(self.finished[msg.change_id])
        c = self._current(msg.change_id)
        if isinstance(c, Failed):
            return Ok(Resumed(msg.change_id, self.host.identity, rolled_back=True))
        self._rollback(c)
        return Ok(self._resume(c, rolled_back=True))

    def _rollback(self, c: _Change) -> None:
        for d in (c.prepare_reply, c.drain_reply, c.pause_reply):
            if d is not None:
                d.resolve(fail(COMPONENT, "abort", "change aborted", code="change.aborted"))
        if c.phase == "executed":
            self._undo_execute(c)
        self._abort_prepared(c)

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
        if c.phase == "preparing":
            self._advance(c)
        elif c.phase == "draining":
            self._check_drain(c)
        elif c.phase == "pausing":
            self._check_pause(c)
        if self.change is c and time.monotonic() > c.deadline:
            self._lease_expired(c)

    def _lease_expired(self, c: _Change) -> None:
        silent = f"heard nothing from the coordinator of change {c.id} for {c.timeout_s}s"
        if c.phase == "preparing":  # it has promised nothing yet
            self._give_up(c, f"{silent} while preparing; withdrew", "change.coordinator_lost")
            return
        answer = self._query(c)
        if answer == "pending":
            c.renew()
            self._note("timeout", f"{silent}; the coordinator answered 'pending'; waiting", "change.decision_pending",
                       severity="info")
        elif answer == "commit" and c.phase == "executed":
            self._commit(c)
            self._note("timeout", f"{silent}; the coordinator answered 'commit'; committed",
                       "change.committed_on_query")
        elif answer is None:
            self._give_up(c, f"{silent} and could not reach it; rolled back", "change.coordinator_lost")
        else:
            self._give_up(c, f"{silent}; the coordinator answered {answer!r} (phase {c.phase}); rolled back",
                          "change.rolled_back_on_query")

    def _query(self, c: _Change) -> DecisionKind | None:
        r = self.host.client.call(self.host.spec.control, QueryDecision(c.id), timeout=self.host.ipc.rpc_timeout_s)
        return r.value.decision if isinstance(r, Ok) and isinstance(r.value, Decision) else None

    def _give_up(self, c: _Change, reason: str, code: str) -> None:
        self._note("timeout", reason, code, severity="error" if code == "change.coordinator_lost" else "warning")
        self._rollback(c)
        self._resume(c, rolled_back=True)


class _Failed(Exception):
    def __init__(self, failed: Failed) -> None:
        super().__init__(str(failed))
        self.failed = failed
