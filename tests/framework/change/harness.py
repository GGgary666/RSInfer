"""In-process coordinator over two real ProcessHosts, with faults injected at the transport boundary."""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path
from typing import Protocol, cast

import msgspec
import pytest
import zmq.asyncio

from core.change.blobs import BlobStore
from core.change.coordinator import Coordinator, PlanInput
from core.change.eventlog import EventLog
from core.change.protocol import Decision, QueryDecision
from core.change.types import ActionPlan, ChangeResult, ChangeSet, ProcActions, UnitAction
from core.common.ids import HostId, Identity, ProcId
from core.common.result import Ok, Result, fail
from core.ipc.rpc import Deferred
from core.supervisor.protocol import Materialized, ProcSpec
from core.supervisor.supervisor import Supervisor, _Proc
from core.supervisor.topology import DesiredProc
from tests.framework.support import spec, write_tree
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
        self.coord.tun = msgspec.structs.replace(self.coord.tun, commit_retry_interval_s=0.001)
        self.coord.identity = OLD
        for host in self.hosts.values():
            monkeypatch.setattr(host.client, "call", self.query)

    def query(self, address: str, msg: msgspec.Struct, *, timeout: float) -> Result[msgspec.Struct]:
        assert isinstance(msg, QueryDecision)
        return Ok(Decision(msg.change_id, self.coord.decision_of(msg.change_id)))

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
