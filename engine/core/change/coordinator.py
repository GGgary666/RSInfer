"""The change coordinator (doc/specs/change.md §4): every change goes through it.

RECEIVED → CHECKED → PLANNED → PREPARING → PREPARED → PAUSING → EXECUTING → COMMITTED,
and from any of them to ABORTING → ABORTED. One change at a time; the rest wait in
the queue. It does not decide what to touch (``core.deps`` plans) and executes
nothing itself (participants do); it sequences phases and tells the truth about
the outcome.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

import msgspec

from core.change.eventlog import EventLog
from core.change.participant import ChangeTunables
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
    Resume,
    Resumed,
)
from core.change.queue import ChangeQueue, Queued
from core.change.state import IdentityFn, IdentityManifest, apply_changeset, identity_of
from core.change.types import EMPTY, ActionPlan, ChangeResult, ChangeSet, ChangeStatus, ProcActions
from core.common.ids import HostId, Identity, ProcId
from core.common.result import Diagnostic, Failed, Ok, Result, diag
from core.supervisor.supervisor import ProcStatus, Supervisor
from core.supervisor.topology import Topology
from unit.runtime.types import ReloadReport

COMPONENT = "change"


class PlanInput(msgspec.Struct, frozen=True):
    cs: ChangeSet
    base_identity: Identity
    base: IdentityManifest
    new: IdentityManifest
    procs: list[ProcStatus]


class Checker(Protocol):
    """``core.manifest``: static checks; nothing runs if they fail."""

    def check(self, inp: PlanInput) -> Awaitable[Result[None]]: ...


class Planner(Protocol):
    """``core.deps``: what each process has to do."""

    def plan(self, inp: PlanInput) -> Awaitable[Result[ActionPlan]]: ...


class AcceptAll:
    async def check(self, inp: PlanInput) -> Result[None]:
        return Ok(None)


class _Abort(Exception):
    def __init__(self, diags: list[Diagnostic]) -> None:
        super().__init__("; ".join(str(d) for d in diags))
        self.diags = diags


@dataclass
class _Run:
    """Book-keeping of one change in flight, for the abort path."""

    id: int
    cs: ChangeSet
    plan: ActionPlan
    identity: Identity
    prepared: list[ProcActions] = field(default_factory=list[ProcActions])
    restarted: list[ProcId] = field(default_factory=list[ProcId])
    spawned: list[ProcId] = field(default_factory=list[ProcId])
    timings: dict[str, float] = field(default_factory=dict[str, float])
    reports: dict[str, ReloadReport] = field(default_factory=dict[str, ReloadReport])
    aborted_requests: int = 0
    _t: float = field(default_factory=time.monotonic)

    def lap(self, phase: str) -> None:
        now = time.monotonic()
        self.timings[phase] = now - self._t
        self._t = now


class Coordinator:
    def __init__(
        self, supervisor: Supervisor, log: EventLog, *, planner: Planner, checker: Checker | None = None,
        identity_fn: IdentityFn = identity_of, tun: ChangeTunables | None = None,
    ) -> None:
        self.sup = supervisor
        self.log = log
        self.planner = planner
        self.checker = checker or AcceptAll()
        self.identity_fn = identity_fn
        self.tun = tun or ChangeTunables()
        if self.tun.commit_attempts < 1 or self.tun.commit_retry_delay_s < 0:
            raise ValueError("commit_attempts must be positive and commit_retry_delay_s nonnegative")
        self.recovery_required = False
        self.identity: Identity = EMPTY
        self.manifest = IdentityManifest()
        self.state_version = 0
        self.queue = ChangeQueue()
        self.faults: set[str] = set()
        """One-shot fault points: ``after_check``, ``after_prepare``, ``after_pause``,
        ``after_execute``, and ``vanish_after_prepare`` (the coordinator goes silent)."""
        self.busy = False
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self.on_result: list[Callable[[ChangeResult], None]] = []

    # ------------------------------------------------------------ queue

    async def start(self) -> None:
        self._task = asyncio.create_task(self._worker())

    def close(self) -> None:
        if self._task is not None:
            self._task.cancel()

    def enqueue(self, cs: ChangeSet, *, dry_run: bool = False) -> asyncio.Future[ChangeResult]:
        fut: asyncio.Future[ChangeResult] = asyncio.get_running_loop().create_future()
        self.queue.push(Queued(cs, dry_run, data={"future": fut}))
        self._wake.set()
        return fut

    async def submit(self, cs: ChangeSet, *, dry_run: bool = False) -> ChangeResult:
        return await self.enqueue(cs, dry_run=dry_run)

    async def _worker(self) -> None:
        while True:
            item, dropped = self.queue.pop(self.identity)
            for d in dropped:
                self._finish_dropped(d)
            if item is None:
                self._wake.clear()
                await self._wake.wait()
                continue
            self.busy = True
            try:
                result = await self._run(item)
            except Exception as e:  # noqa: BLE001 - the coordinator must outlive any change
                result = self._result("failed", -1, item.cs, None, None,
                                      [diag(COMPONENT, "internal", f"{type(e).__name__}: {e}",
                                            code="change.internal_error")])
            finally:
                self.busy = False
            fut = item.data["future"]
            assert isinstance(fut, asyncio.Future)
            if not fut.done():
                fut.set_result(result)  # type: ignore[union-attr]
            for cb in self.on_result:
                cb(result)

    def _finish_dropped(self, item: Queued) -> None:
        cid = self.log.begin(item.cs)
        result = self._result("dropped", cid, item.cs, None, None, [diag(
            COMPONENT, "queue", f"computed under {item.cs.computed_under}, current is {self.identity}",
            code="change.stale_internal", severity="info", hint="the unit recomputes it next cycle")])
        self.log.finish(result)
        self.log.event(cid, "dropped_stale_internal", {"computed_under": item.cs.computed_under})
        fut = item.data["future"]
        assert isinstance(fut, asyncio.Future)
        if not fut.done():
            fut.set_result(result)  # type: ignore[union-attr]
        for cb in self.on_result:
            cb(result)

    def _result(self, status: ChangeStatus, cid: int, cs: ChangeSet, new: Identity | None,
                run: _Run | None, diags: list[Diagnostic], plan: ActionPlan | None = None) -> ChangeResult:
        return ChangeResult(
            status=status, change_id=cid, old_identity=self.identity if status != "committed" else cs.base,
            new_identity=new, plan=run.plan if run else plan, timings=dict(run.timings) if run else {},
            aborted_requests=run.aborted_requests if run else 0,
            restarted=list(run.restarted) if run else [], reports=dict(run.reports) if run else {},
            state_version=self.state_version, diagnostics=diags,
        )

    def _fault(self, point: str) -> None:
        if point in self.faults:
            self.faults.discard(point)
            raise _Abort([diag(COMPONENT, point, f"fault injected at {point}", code="change.injected_fault")])

    # ------------------------------------------------------------- one change

    async def _run(self, item: Queued) -> ChangeResult:
        cs = item.cs
        cid = self.log.begin(cs)
        old = self.identity
        t0 = time.monotonic()
        if self.recovery_required:
            return self._done(cid, self._result("failed", cid, cs, None, None, [diag(
                COMPONENT, "check", "a previous commit is unresolved; restart the engine",
                code="change.recovery_required")]))
        # RECEIVED -> CHECKED
        if cs.source != "internal" and cs.base != self.identity:
            return self._done(cid, self._result("failed", cid, cs, None, None, [diag(
                COMPONENT, "check", f"base {cs.base!r} is not the running identity {self.identity!r}",
                code="change.stale_base", hint="rebase the change on the current identity")]))
        new_manifest = apply_changeset(self.manifest, cs)
        missing = self.sup.blobs.missing(h for h in cs.files.values() if h is not None)
        if missing:
            return self._done(cid, self._result("failed", cid, cs, None, None, [diag(
                COMPONENT, "check", f"{len(missing)} file contents are not in the blob store",
                code="change.missing_blobs")]))
        inp = PlanInput(cs, self.identity, self.manifest, new_manifest, self.sup.status())
        checked = await self.checker.check(inp)
        if isinstance(checked, Failed):
            return self._done(cid, self._result("failed", cid, cs, None, None, checked.diagnostics))
        # CHECKED -> PLANNED
        planned = await self.planner.plan(inp)
        if isinstance(planned, Failed):
            return self._done(cid, self._result("failed", cid, cs, None, None, planned.diagnostics))
        plan = planned.value
        if item.dry_run:
            return self._done(cid, self._result("planned", cid, cs, None, None, [], plan=plan))
        if plan.restart_engine:
            return self._done(cid, self._result("failed", cid, cs, None, None, [diag(
                COMPONENT, "plan", "the change touches the framework core", code="escalate.restart_engine",
                hint="restart the engine from the new source")], plan=plan))
        new_identity = old if cs.state_only else self.identity_fn(new_manifest)
        run = _Run(cid, cs, plan, new_identity)
        run.timings["check+plan"] = time.monotonic() - t0
        self.log.record_identity(new_identity, new_manifest, old if old != EMPTY else None)
        try:
            self._fault("after_check")
            await self._prepare(run, new_manifest)
            if "vanish_after_prepare" in self.faults:
                self.faults.discard("vanish_after_prepare")
                return self._done(cid, self._result("aborted", cid, cs, None, run, [diag(
                    COMPONENT, "prepare", "coordinator vanished (injected)", code="change.injected_fault")]))
            self._fault("after_prepare")
            await self._pause(run)
            self._fault("after_pause")
            await self._execute(run, new_manifest)
            self._fault("after_execute")
        except _Abort as e:
            undo = await self._abort(run)
            return self._done(cid, self._result("aborted", cid, cs, None, run, [*e.diags, *undo]))
        # Persist the irreversible decision before sending any Commit. Once sent,
        # neither RPC failure nor timeout is evidence that rollback is safe.
        self.log.event(cid, "commit_decided", {"identity": new_identity,
                       "participants": [str(pa.proc) for pa in run.prepared]})
        self.recovery_required = True
        committed = await self._commit(run)
        if committed:
            return self._done(cid, self._result("failed", cid, cs, new_identity, run, committed))
        self.recovery_required = False
        self.identity = new_identity
        self.manifest = new_manifest
        if cs.state_only:
            self.state_version += 1
        return self._done(cid, self._result("committed", cid, cs, new_identity, run, []))

    def _done(self, cid: int, result: ChangeResult) -> ChangeResult:
        self.log.finish(result)
        return result

    async def _call(self, proc: ProcId, msg: msgspec.Struct, timeout: float | None = None
                    ) -> Result[msgspec.Struct]:
        return await self.sup.call(proc, msg, timeout=timeout or self.sup.tun.rpc_timeout_s)

    async def _gather(self, calls: list[tuple[ProcId, Awaitable[Result[msgspec.Struct]]]]
                      ) -> dict[ProcId, msgspec.Struct]:
        results = await asyncio.gather(*(c for _, c in calls))
        errors = [d for r in results if isinstance(r, Failed) for d in r.diagnostics]
        if errors:
            raise _Abort(errors)
        return {p: r.unwrap() for (p, _), r in zip(calls, results, strict=True)}

    def _hosts(self, run: _Run) -> set[HostId]:
        hosts = {p.proc.host for p in run.plan.procs} | {p.proc.host for p in run.plan.spawn}
        return hosts

    # ---------------------------------------------------------------- phases

    async def _prepare(self, run: _Run, manifest: IdentityManifest) -> None:
        hosts = sorted(self._hosts(run))
        mats = await asyncio.gather(*(self.sup.materialize(h, run.identity, manifest.files) for h in hosts))
        errors = [d for m in mats if isinstance(m, Failed) for d in m.diagnostics]
        if errors:
            raise _Abort(errors)
        participants = run.plan.participants()
        calls: list[tuple[ProcId, Awaitable[Result[msgspec.Struct]]]] = []
        for pa in participants:
            msg = Prepare(run.id, run.identity, pa, self.tun.coordinator_timeout_s)
            calls.append((pa.proc, self._call(pa.proc, msg)))
        results = await asyncio.gather(*(c for _, c in calls))
        for (proc, _), r in zip(calls, results, strict=True):
            if isinstance(r, Ok):
                run.prepared.append(next(pa for pa in participants if pa.proc == proc))
        errors = [d for r in results if isinstance(r, Failed) for d in r.diagnostics]
        if errors:
            raise _Abort(errors)
        run.lap("prepare")

    async def _pause(self, run: _Run) -> None:
        drain_timeout = run.plan.drain_timeout_s if run.plan.drain_timeout_s is not None else self.tun.drain_timeout_s
        draining = [pa for pa in run.prepared if pa.drain]
        if draining:
            drained = await self._gather([
                (pa.proc, self._call(pa.proc, Drain(run.id, drain_timeout),
                                     timeout=drain_timeout + self.sup.tun.rpc_timeout_s))
                for pa in draining])
            run.aborted_requests = sum(d.aborted_requests for d in drained.values() if isinstance(d, Drained))
            run.lap("drain")
        leaders = [pa for pa in run.prepared if pa.leads_epochs]
        followers = [pa for pa in run.prepared if not pa.leads_epochs]
        paused = await self._gather([(pa.proc, self._call(pa.proc, Pause(run.id, None))) for pa in leaders])
        epochs = [p.epoch for p in paused.values() if isinstance(p, Paused) and p.epoch is not None]
        after = max(epochs) if epochs else None
        await self._gather([(pa.proc, self._call(pa.proc, Pause(run.id, after))) for pa in followers])
        run.lap("pause")

    async def _execute(self, run: _Run, manifest: IdentityManifest) -> None:
        executed = await self._gather([(pa.proc, self._call(pa.proc, Execute(run.id))) for pa in run.prepared])
        for proc, e in executed.items():
            if isinstance(e, Executed) and e.report is not None:
                run.reports[str(proc)] = e.report
        topo = run.plan.topology or self.sup.topology
        for proc in run.plan.stop:
            r = await self.sup.stop(proc)
            if isinstance(r, Failed):
                raise _Abort(r.diagnostics)
            self.sup.forget(proc)
        for pa in run.plan.procs:
            if pa.restart is None:
                continue
            assert topo is not None
            run.restarted.append(pa.proc)
            r = await self.sup.relaunch(pa.restart, run.identity, topo, start_paused=True)
            if isinstance(r, Failed):
                raise _Abort(r.diagnostics)
        for plan in run.plan.spawn:
            assert topo is not None
            run.spawned.append(plan.proc)
            r = await self.sup.launch(plan, run.identity, topo, start_paused=True)
            if isinstance(r, Failed):
                raise _Abort(r.diagnostics)
        if run.plan.topology is not None:
            self.sup.topology = run.plan.topology
        run.lap("execute")

    async def _confirm(self, run: _Run, procs: list[ProcId], *, resume: bool) -> list[Diagnostic]:
        pending = list(procs)
        phase = "resume" if resume else "commit"
        for attempt in range(self.tun.commit_attempts):
            msg = Resume(run.id, run.identity) if resume else Commit(run.id)
            results = await asyncio.gather(*(self._call(p, msg) for p in pending))
            retry: list[ProcId] = []
            for p, r in zip(pending, results, strict=True):
                ack = r.value if isinstance(r, Ok) else None
                valid = (isinstance(ack, Resumed) and not ack.rolled_back) if resume else isinstance(ack, Committed)
                if not valid or not isinstance(ack, (Committed, Resumed)) \
                        or ack.change_id != run.id or ack.identity != run.identity:
                    retry.append(p)
                    self.log.event(run.id, f"{phase}_unacknowledged", {"proc": str(p), "attempt": attempt + 1})
            pending = retry
            if not pending:
                return []
            if attempt + 1 < self.tun.commit_attempts:
                await asyncio.sleep(self.tun.commit_retry_delay_s)
        return [diag(COMPONENT, phase, f"{phase} not confirmed; restart engine to recover the committed decision",
                     subject=str(p), code="change.recovery_required") for p in pending]

    async def _commit(self, run: _Run) -> list[Diagnostic]:
        errors = await self._confirm(run, [pa.proc for pa in run.prepared], resume=False)
        if errors:
            return errors  # All existing participants remain paused; do not send Abort.
        for pa in run.prepared:
            self.sup.committed(pa.proc, run.identity, pa.plan_after)
        for proc in [*run.restarted, *run.spawned]:
            self.sup.committed(proc, run.identity, None)
        # Candidate processes also started paused; none serves before the barrier.
        errors = await self._confirm(run, [pa.proc for pa in run.prepared] + run.restarted + run.spawned, resume=True)
        run.lap("commit")
        return errors

    async def _abort(self, run: _Run) -> list[Diagnostic]:
        out: list[Diagnostic] = []
        results = await asyncio.gather(*(self._call(pa.proc, Abort(run.id)) for pa in run.prepared))
        for pa, r in zip(run.prepared, results, strict=True):
            if isinstance(r, Failed):
                out.append(diag(COMPONENT, "abort", f"no answer to Abort: {r}", subject=str(pa.proc),
                                code="change.abort_unacknowledged",
                                hint="executed participants stay paused until an explicit decision or engine restart"))
        for proc in run.spawned:
            await self.sup.stop(proc)
            self.sup.forget(proc)
        for proc in run.restarted:
            r = await self.sup.relaunch_previous(proc)
            if isinstance(r, Failed):
                out += r.diagnostics
        run.lap("abort")
        return out

    # ------------------------------------------------------------- startup

    async def startup(self, files: dict[str, str], topology: Topology, *, config: dict[str, object] | None = None
                      ) -> ChangeResult:
        """From the empty system to an identity: also a change (modules.html scenario ②)."""
        return await self.submit(ChangeSet(base=EMPTY, files=dict(files), config=dict(config or {}),
                                           topology=topology, source="startup"))

