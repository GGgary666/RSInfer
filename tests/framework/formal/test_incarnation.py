"""Real lifecycle handlers and generation filters; only OS child creation is simulated."""
from __future__ import annotations

import asyncio
from pathlib import Path
import signal
import time
from typing import Any, cast

import msgspec
import pytest

from core.common.ids import Identity, ProcId
from core.common.messages import Envelope
from core.common.result import Failed, Ok, Result
from core.nodeagent.agent import AgentTunables, NodeAgent
from core.supervisor.protocol import CheckIn, CheckedIn, Exited, Heartbeat, ProcBeat, ProcSpec, Spawn, Spawned, Stop
from core.supervisor.supervisor import Supervisor, _Host
from platforms.base import PlatformInfo
from tests.framework.change.harness import Harness, harness as harness
from tests.framework.support import PROC
from unit.runtime import ProcessPlan


class Child:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.exited = asyncio.Event()

    def send_signal(self, sig: int) -> None:
        self.returncode = -sig
        self.exited.set()

    def kill(self) -> None:
        self.send_signal(signal.SIGKILL)

    async def wait(self) -> int:
        await self.exited.wait()
        assert self.returncode is not None
        return self.returncode


def test_duplicate_spawn_and_late_stop_are_serialized_at_the_real_await(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def main() -> None:
        identity = Identity("snapshot")
        agent = NodeAgent(host=PROC.host, address="unused", supervisor="unused",
                          snapshot_root=tmp_path / "snapshots", run_root=tmp_path / "runs",
                          platform=PlatformInfo("fake"), tun=AgentTunables())
        agent.snapshot_dir(identity).mkdir(parents=True)
        children: list[Child] = []
        reports: list[msgspec.Struct] = []
        entered, release = asyncio.Event(), asyncio.Event()

        class Client:
            async def call(self, address: str, msg: msgspec.Struct, *, timeout: float) -> Result[msgspec.Struct]:
                reports.append(msg)
                return Ok(CheckedIn())

        monkeypatch.setattr(agent, "client", Client(), raising=False)

        async def create(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
            requested = msgspec.msgpack.decode(Path(args[-1]).read_bytes(), type=ProcSpec)
            if requested.incarnation == 2:
                entered.set()
                await release.wait()  # real NodeAgent.spawn has yielded after checking old state
            child = Child(100 + len(children))
            children.append(child)
            assert sum(c.returncode is None for c in children) == 1, "overlapping live instances"
            return cast(asyncio.subprocess.Process, child)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
        spec = ProcSpec(PROC, identity, ProcessPlan(PROC, identity, []), "unused", "127.0.0.1", incarnation=1)

        async def send(msg: msgspec.Struct) -> Result[msgspec.Struct]:
            return await agent._dispatch(agent.registry.wrap(msg, agent.me), msg)

        tasks_before = asyncio.all_tasks()
        requests: list[asyncio.Task[Result[msgspec.Struct]]] = []
        try:
            (await send(Spawn(spec))).unwrap()
            (await send(Stop(PROC, 1.0, 1))).unwrap()
            successor = msgspec.structs.replace(spec, incarnation=2)
            requests.append(asyncio.create_task(send(Spawn(successor))))
            await asyncio.wait_for(entered.wait(), 2)
            requests.append(asyncio.create_task(send(Spawn(successor))))
            requests.append(asyncio.create_task(send(Stop(PROC, 1.0, 1))))
            requests.append(asyncio.create_task(send(Spawn(spec))))
            await asyncio.sleep(0)  # let all three requests reach the lifecycle lock
            release.set()
            first, duplicate, late_stop, late_spawn = await asyncio.wait_for(asyncio.gather(*requests), 2)
            assert isinstance(first, Ok) and isinstance(first.value, Spawned)
            assert duplicate == first
            assert isinstance(late_stop, Failed) and late_stop.has_code("agent.stale_stop")
            assert isinstance(late_spawn, Failed) and late_spawn.has_code("agent.stale_spawn")
            assert len(children) == 2 and children[1].returncode is None
            assert agent.procs[PROC].process.pid == children[1].pid
            (await send(Stop(PROC, 1.0, 2))).unwrap()
            again = await send(Spawn(successor))
            assert isinstance(again, Failed) and again.has_code("agent.already_exited")
            assert len(children) == 2
            assert {r.incarnation for r in reports if isinstance(r, Exited)} == {1, 2}
        finally:
            release.set()
            for request in requests:
                if not request.done():
                    request.cancel()
            if requests:
                await asyncio.gather(*requests, return_exceptions=True)
            for child in children:
                child.kill()
            watchers = asyncio.all_tasks() - tasks_before
            if watchers:
                await asyncio.wait_for(asyncio.gather(*watchers, return_exceptions=True), 2)

    asyncio.run(main())


@pytest.mark.parametrize("report", ["exit", "heartbeat", "check-in"])
@pytest.mark.parametrize("state", ["SPAWNING", "RUNNING"])
def test_old_reports_cannot_change_successor_records(harness: Harness, report: str, state: str) -> None:
    h = harness
    rec = h.sup.procs[h.target]
    rec.spec = msgspec.structs.replace(rec.spec, incarnation=2)
    rec.state = cast(Any, state)
    h.sup.hosts[h.target.host] = _Host(h.target.host, "unused", PlatformInfo("fake"), time.monotonic())
    latest = CheckIn(h.target, 2, rec.spec.identity, {}, [], [], [], [])
    if state == "RUNNING":
        rec.checkin = latest
    before = rec.state, rec.checkin, rec.exit_code, rec.checked_in.is_set(), h.sup.losses.qsize()
    if report == "exit":
        h.sup._exited(Exited(h.target, 1, 9))
    elif report == "heartbeat":
        h.sup._heartbeat(Heartbeat(h.target.host, [ProcBeat(h.target, 1, "exited", 9)]))
    else:
        old = msgspec.structs.replace(latest, incarnation=1)
        assert isinstance(h.sup._check_in(old), Failed)
    assert (rec.state, rec.checkin, rec.exit_code, rec.checked_in.is_set(), h.sup.losses.qsize()) == before
    # Positive control: current-generation reports must still take effect.
    if report == "check-in":
        assert isinstance(h.sup._check_in(latest), Ok)
        assert rec.state == "RUNNING" and rec.checkin == latest
    elif report == "exit":
        h.sup._exited(Exited(h.target, 2, 9))
        assert rec.state == "LOST" and rec.exit_code == 9
    else:
        h.sup._heartbeat(Heartbeat(h.target.host, [ProcBeat(h.target, 2, "exited", 9)]))
        assert rec.state == "LOST"


@pytest.mark.parametrize("mutation", ["remove-lock", "remove-stop-generation"])
def test_lifecycle_oracles_reject_implementation_mutants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    """Temporary C implementations must fail the same test that passed on B."""
    dispatch, stop = NodeAgent._dispatch, NodeAgent.stop
    if mutation == "remove-lock":
        async def unlocked(self: NodeAgent, env: Envelope, msg: msgspec.Struct) -> Result[msgspec.Struct]:
            if isinstance(msg, Spawn):
                return await self.spawn(msg.spec)
            if isinstance(msg, Stop):
                return await self.stop(msg.proc, msg.timeout_s, msg.incarnation)
            return await dispatch(self, env, msg)
        monkeypatch.setattr(NodeAgent, "_dispatch", unlocked)
    else:
        async def unfenced(self: NodeAgent, proc: ProcId, timeout_s: float, incarnation: int) -> Result[msgspec.Struct]:
            return await stop(self, proc, timeout_s, self.procs[proc].spec.incarnation)
        monkeypatch.setattr(NodeAgent, "stop", unfenced)
    with pytest.raises(AssertionError):
        test_duplicate_spawn_and_late_stop_are_serialized_at_the_real_await(tmp_path, monkeypatch)


def test_report_oracle_rejects_implementation_mutant(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    exited = Supervisor._exited

    def unfenced(self: Supervisor, msg: Exited) -> Result[msgspec.Struct]:
        return exited(self, msgspec.structs.replace(msg, incarnation=self.procs[msg.proc].spec.incarnation))

    monkeypatch.setattr(Supervisor, "_exited", unfenced)
    with pytest.raises(AssertionError):
        test_old_reports_cannot_change_successor_records(harness, "exit", "RUNNING")
