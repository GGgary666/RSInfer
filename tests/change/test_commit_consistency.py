"""Fault injection at the transport boundary, with real coordinator/participants.

The original implementation must run these tests too: do not import new protocol
messages at collection time. A timeout is simulated by advancing the participant's
deadline, not by sleeping. Unit behavior is checked independently of identity labels.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Protocol, cast

import msgspec
import pytest
import zmq.asyncio

from core.change.blobs import BlobStore
from core.change.coordinator import Coordinator, PlanInput
from core.change.eventlog import EventLog
from core.change.types import ActionPlan, ChangeResult, ChangeSet, ProcActions, UnitAction
from core.common.ids import HostId, Identity, ProcId
from core.common.result import Ok, Result, fail
from core.ipc.rpc import Deferred
from core.supervisor.protocol import Materialized, ProcSpec
from core.supervisor.supervisor import Supervisor, _Proc
from core.supervisor.topology import DesiredProc, HostSpec, Mesh, RankPlacement, RoleSpec, Topology
from tests.support import spec, write_tree
from unit.runtime import ProcessPlan
from unit.runtime.host import ProcessHost

OLD = Identity("old")
UNIT = '''
class Unit:
    def __init__(self, ctx, config):
        self.version = 1
        self.commits = 0
        ctx.on_action("install", self)
    def prepare(self, args):
        pass
    def execute(self, args):
        self.version = 2
    def abort(self):
        self.version = 1
    def commit(self):
        self.commits += 1
'''


class VersionUnit(Protocol):
    version: int
    commits: int


class Capture(Deferred):
    def __init__(self) -> None:
        self.result: Result[msgspec.Struct] | None = None

    def resolve(self, result: Result[msgspec.Struct]) -> None:
        self.result = result


class Harness:
    def __init__(self, tmp: Path, pkg: str, monkeypatch: pytest.MonkeyPatch) -> None:
        self.ctx = zmq.asyncio.Context()
        self.sup = Supervisor(self.ctx, host=HostId("control"), bind_address="127.0.0.1",
                              blobs=BlobStore(tmp / "blobs"))
        self.hosts: dict[ProcId, ProcessHost] = {}
        self.messages: list[tuple[ProcId, str]] = []
        self.drop_kind = "Commit"
        self.drop_count = 0
        self.drop_reply = False
        self.observe: Callable[[], None] = lambda: None
        root = write_tree(tmp / "code", {f"{pkg}/versioned.py": UNIT})
        self.unit = spec(f"{pkg}.versioned", f"{pkg}.versioned:Unit")
        for i in range(2):
            proc = ProcId(HostId(f"host-{i}"), "worker", i)
            plan = ProcessPlan(proc, OLD, [self.unit], code_root=str(root))
            ps = ProcSpec(proc, OLD, plan, self.sup.address, "127.0.0.1")
            host = self.hosts[proc] = ProcessHost(ps)
            host.runtime.load_process_plan(plan).unwrap()
            monkeypatch.setattr(host.rpc, "defer", Capture)
            desired = DesiredProc(proc, [self.unit.id], [], {})
            self.sup.procs[proc] = _Proc(desired, ps, "RUNNING")
        self.first, self.target = self.hosts
        self.action_plan = ActionPlan(procs=[ProcActions(p, actions=[UnitAction(self.unit.id, "install")])
                                     for p in self.hosts])
        monkeypatch.setattr(self.sup, "call", self.call)
        monkeypatch.setattr(self.sup, "materialize", self.materialize)
        self.coord = Coordinator(self.sup, EventLog(tmp / "events.sqlite", self.sup.blobs), planner=self)
        self.coord.identity = OLD

    async def plan(self, inp: PlanInput) -> Result[ActionPlan]:
        return Ok(self.action_plan)

    async def materialize(self, host: HostId, identity: Identity, files: dict[str, str]) -> Result[msgspec.Struct]:
        # This change modifies unit state only; no filesystem materialization is needed.
        return Ok(Materialized(identity, "unused", 0, 0))

    def expire(self) -> None:
        for host in self.hosts.values():
            c = host.participant.change
            if c is not None:
                c.deadline = 0
            host.participant.tick()

    def unit_of(self, proc: ProcId) -> VersionUnit:
        return cast(VersionUnit, self.hosts[proc].runtime.instance(self.unit.id))

    async def call(self, proc: ProcId, msg: msgspec.Struct, *, timeout: float | None = None) -> Result[msgspec.Struct]:
        kind = type(msg).__name__
        self.messages.append((proc, kind))
        drop = proc == self.target and kind == self.drop_kind and self.drop_count != 0
        if drop:
            self.drop_count -= 1
        host = self.hosts[proc]
        if drop and not self.drop_reply:
            self.expire()
            self.observe()
            return fail("test", "rpc", "request lost", code="ipc.timeout")
        env = host.registry.wrap(msg, self.sup.me)
        result = host.handlers[type(msg)](env, msg)
        if isinstance(result, Capture):
            assert result.result is not None
            result = result.result
        assert not isinstance(result, Deferred)
        self.observe()
        if drop:
            self.expire()
            return fail("test", "rpc", "reply lost", code="ipc.timeout")
        return result

    async def submit(self) -> ChangeResult:
        return await self.coord.submit(ChangeSet(base=OLD, config={"revision": 2}))

    def close(self) -> None:
        self.coord.close()
        for host in self.hosts.values():
            host.shutdown()
        self.sup.close()
        self.ctx.term()


@pytest.fixture
def harness(tmp_path: Path, pkg: str, monkeypatch: pytest.MonkeyPatch) -> Generator[Harness]:
    h = Harness(tmp_path, pkg, monkeypatch)
    try:
        yield h
    finally:
        h.close()


@pytest.mark.parametrize("drop_reply", [False, True], ids=["lost-request", "lost-reply"])
def test_lost_commit_is_retried_until_every_participant_serves_new_version(harness: Harness, drop_reply: bool) -> None:
    h = harness
    h.drop_count = 1
    h.drop_reply = drop_reply

    async def main() -> None:
        await h.coord.start()
        result = await h.submit()
        h.expire()
        assert result.status == "committed", result.diagnostics
        for p, host in h.hosts.items():
            assert host.identity == result.new_identity != OLD
            assert not host.paused
            assert h.unit_of(p).version == 2
            assert h.unit_of(p).commits == 1
            assert h.sup.procs[p].spec.identity == host.identity
        assert h.messages.count((h.target, "Commit")) == 2

    asyncio.run(main())


@pytest.mark.parametrize("lost_target", [0, 1])
@pytest.mark.parametrize("drop_reply", [False, True], ids=["lost-request", "lost-reply"])
def test_no_participant_serves_before_global_commit_confirmation(
    harness: Harness, lost_target: int, drop_reply: bool,
) -> None:
    """Observe the interval between deliveries, not just the final state.

    The early-resume mutant passes the initial tests by fencing again at timeout.
    Serving during that interval is already a violation that a later pause cannot undo.
    """
    h = harness
    h.target = list(h.hosts)[lost_target]
    h.drop_count = 1
    h.drop_reply = drop_reply
    premature_service: list[tuple[ProcId, int]] = []

    def observe() -> None:
        if any(host.identity == OLD for host in h.hosts.values()):
            premature_service.extend((p, h.unit_of(p).version) for p, host in h.hosts.items()
                                     if host.identity != OLD and not host.paused)

    h.observe = observe

    async def main() -> None:
        await h.coord.start()
        result = await h.submit()
        assert not premature_service, f"served before all participants committed: {premature_service}"
        assert result.status == "committed", result.diagnostics
        assert all(not host.paused and host.identity == result.new_identity for host in h.hosts.values())

    asyncio.run(main())


def test_persistent_commit_loss_is_not_reported_as_success(harness: Harness) -> None:
    h = harness
    h.drop_count = -1

    async def main() -> None:
        await h.coord.start()
        result = await h.submit()
        assert result.status == "failed", result
        assert any(d.code == "change.recovery_required" for d in result.diagnostics)
        assert h.coord.identity == OLD
        assert all(host.paused for host in h.hosts.values())
        assert all(h.unit_of(p).version == 2 for p in h.hosts)
        assert not any(kind in ("Resume", "Abort") for _, kind in h.messages)
        count = len(h.messages)
        blocked = await h.submit()
        assert blocked.status == "failed"
        assert len(h.messages) == count  # No subsequent transaction enters an unresolved state.
        assert h.coord.log.events("commit_decided")

    asyncio.run(main())


@pytest.mark.parametrize("drop_reply", [False, True], ids=["lost-request", "lost-reply"])
def test_resume_loss_retries_without_committing_twice(harness: Harness, drop_reply: bool) -> None:
    h = harness
    h.drop_kind = "Resume"
    h.drop_count = 1
    h.drop_reply = drop_reply

    async def main() -> None:
        await h.coord.start()
        result = await h.submit()
        assert result.status == "committed", result.diagnostics
        assert h.messages.count((h.target, "Resume")) == 2
        assert all(not host.paused and host.identity == result.new_identity for host in h.hosts.values())
        assert all(h.unit_of(p).commits == 1 for p in h.hosts)

    asyncio.run(main())


def test_persistent_resume_loss_blocks_following_changes(harness: Harness) -> None:
    h = harness
    h.drop_kind = "Resume"
    h.drop_count = -1

    async def main() -> None:
        await h.coord.start()
        result = await h.submit()
        assert result.status == "failed"
        assert any(d.code == "change.recovery_required" for d in result.diagnostics)
        assert h.hosts[h.target].paused
        assert all(host.identity == result.new_identity for host in h.hosts.values())
        assert all(h.unit_of(p).version == 2 for p in h.hosts)
        assert not any(kind == "Abort" for _, kind in h.messages)
        count = len(h.messages)
        assert (await h.submit()).status == "failed"
        assert len(h.messages) == count

    asyncio.run(main())


@pytest.mark.parametrize("restart", [False, True], ids=["spawn", "restart"])
def test_candidate_process_starts_paused_until_commit_barrier(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, restart: bool,
) -> None:
    h = harness
    candidate = h.target if restart else ProcId(HostId("host-new"), "worker", 2)
    existing = list(h.hosts)
    desired_procs = existing if restart else [*existing, candidate]
    topo = Topology(
        hosts=[HostSpec(p.host, "127.0.0.1", "/unused") for p in desired_procs],
        mesh=Mesh({"tp": len(desired_procs)}),
        placement=[RankPlacement(i, p.host, 0) for i, p in enumerate(desired_procs)],
        roles=[RoleSpec("worker", [h.unit.id], per_rank=True)],
    )
    plan = msgspec.structs.replace(h.hosts[h.first].spec.plan, proc=candidate)
    participants = [pa for pa in h.action_plan.procs if pa.proc != candidate]
    h.action_plan = ActionPlan(
        procs=participants + ([ProcActions(candidate, restart=plan)] if restart else []),
        spawn=[] if restart else [plan], topology=topo,
    )
    born_paused: list[bool] = []
    before_barrier: list[bool] = []

    async def spawn(desired: DesiredProc, ps: ProcSpec) -> Result[None]:
        old = h.hosts.pop(desired.proc, None)
        if old is not None:
            old.shutdown()
        host = h.hosts[desired.proc] = ProcessHost(ps)
        host.runtime.load_process_plan(ps.plan).unwrap()
        # A candidate snapshot starts with the new behavior already installed.
        h.unit_of(desired.proc).version = 2
        monkeypatch.setattr(host.rpc, "defer", Capture)
        h.sup.procs[desired.proc] = _Proc(desired, ps, "RUNNING")
        born_paused.append(host.paused)
        observe()
        return Ok(None)

    async def stop(proc: ProcId, timeout_s: float | None = None) -> Result[None]:
        return Ok(None)

    def observe() -> None:
        if born_paused and any(h.hosts[pa.proc].identity == OLD for pa in participants):
            before_barrier.append(h.hosts[candidate].paused)

    monkeypatch.setattr(h.sup, "spawn", spawn)
    monkeypatch.setattr(h.sup, "stop", stop)
    h.observe = observe

    async def main() -> None:
        await h.coord.start()
        result = await h.submit()
        assert born_paused == [True], result.diagnostics
        assert before_barrier and all(before_barrier)
        assert result.status == "committed", result.diagnostics
        assert not h.hosts[candidate].paused
        assert h.hosts[candidate].identity == result.new_identity
        assert h.unit_of(candidate).version == 2
        assert (candidate, "Resume") in h.messages

    asyncio.run(main())
