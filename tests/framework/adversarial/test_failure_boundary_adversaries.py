"""Stronger oracles: unsafe implementations must not pass by changing result labels."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from pathlib import Path

import msgspec
import pytest

from core.change.protocol import Abort, Commit, Execute, Pause, Resume, Resumed
from core.change.queue import ChangeQueue, Queued
from core.change.types import ActionPlan, ChangeMeta, ChangeSet
from core.common.ids import Identity, ProcId
from core.common.result import Failed, Ok, Result, fail
from core.nodeagent.agent import child_env
from core.supervisor.protocol import ProcSpec
from core.supervisor.supervisor import _Host
from core.supervisor.topology import DesiredProc, Mesh, Topology
from platforms.base import PlatformInfo
from tests.framework.adversarial.peer import Peer, peer as peer
from tests.framework.change.harness import Harness, OLD, harness as harness
from tests.framework.support import PROC
from unit.runtime import ProcessPlan


def test_unconfirmed_stop_freezes_changes_and_preserves_process_record(harness: Harness) -> None:
    h = harness
    h.sup.hosts[h.target.host] = _Host(h.target.host, "unused", PlatformInfo("fake"), time.monotonic(), lost=True)
    rec = h.sup.procs[h.target]
    rec.state = "LOST"
    h.action_plan = ActionPlan(procs=[h.action_plan.procs[0]], stop=[h.target])

    async def main() -> None:
        await h.coord.start()
        result = await h.submit()
        assert result.status == "failed" and h.coord.recovery_required
        assert "change.recovery_required" in {d.code for d in result.diagnostics}
        assert h.sup.procs[h.target] is rec and rec.state == "LOST"
        assert h.coord.identity == OLD and h.unit_of(h.first).version == 1
        h.action_plan = ActionPlan()
        assert (await h.submit()).status == "failed"

    asyncio.run(main())


@pytest.mark.parametrize("operation", ["restart", "relaunch_previous", "relaunch"])
def test_failed_stop_cannot_spawn_a_replacement(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    h = harness
    rec = h.sup.procs[h.target]
    h.sup._previous[h.target] = (rec.desired, rec.spec)
    spawned: list[ProcId] = []

    async def stop(proc: ProcId, timeout_s: float | None = None) -> Result[None]:
        return fail("test", "stop", "unconfirmed", code="ipc.timeout")

    async def spawn(desired: DesiredProc, spec: ProcSpec) -> Result[None]:
        spawned.append(spec.proc)
        return Ok(None)

    monkeypatch.setattr(h.sup, "stop", stop)
    monkeypatch.setattr(h.sup, "spawn", spawn)

    async def main() -> None:
        if operation == "restart":
            result = await h.sup.restart([h.target])
        elif operation == "relaunch_previous":
            result = await h.sup.relaunch_previous(h.target)
        else:
            # stop must fail before topology is needed
            result = await h.sup.relaunch(rec.spec.plan, OLD, Topology([], Mesh({}), [], []))
        assert isinstance(result, Failed) and not spawned
        assert h.sup.procs[h.target] is rec and h.target in h.sup._previous

    asyncio.run(main())


@pytest.mark.parametrize("phase", ["prepare", "slice", "lease", "execute", "abort"])
def test_failed_undo_stays_paused_and_repeated_messages_cannot_hide_it(
    peer: Peer, monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    aborts = 0

    def abort() -> None:
        nonlocal aborts
        aborts += 1
        raise RuntimeError("resource still acquired")

    monkeypatch.setattr(peer.state, "abort", abort)
    request = peer.prepare(1, phase if phase in ("prepare", "execute") else "")
    if phase in ("slice", "lease"):
        def prepare(args: object) -> Iterator[None]:
            yield
            raise RuntimeError("slice failed")

        monkeypatch.setattr(peer.state, "prepare", prepare)
        env = peer.host.registry.wrap(request, peer.host.me)
        peer.host.participant.on_prepare(env, request)
        c = peer.host.participant.change
        assert c is not None
        if phase == "lease":
            c.deadline = 0
        peer.host.participant.tick()
        if phase == "slice":
            peer.host.participant.tick()
    elif phase == "prepare":
        assert isinstance(peer.send(request), Failed)
    else:
        peer.send(request).unwrap()
        peer.send(Pause(1, None)).unwrap()
        executed = peer.send(Execute(1))
        assert isinstance(executed, Failed) if phase == "execute" else isinstance(executed, Ok)

    for _ in range(2):
        r = peer.send(Abort(1))
        assert isinstance(r, Failed) and r.has_code("change.abort_failed")
        assert peer.host.paused and peer.host.participant.change is not None
        assert 1 not in peer.host.participant.finished
    assert aborts == 1, "do not rerun a partially failed cleanup"
    for message in (request, peer.prepare(2), Pause(1, None), Execute(1), Commit(1), Resume(1, Identity("v1"))):
        assert isinstance(peer.send(message), Failed)
    assert peer.host.paused


@pytest.mark.parametrize("receipt", ["missing", "wrong-id", "not-rolled-back"])
def test_abort_needs_an_actual_rollback_acknowledgement(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, receipt: str,
) -> None:
    h = harness
    original = h.sup.call

    async def call(proc: ProcId, msg: msgspec.Struct, *, timeout: float | None = None) -> Result[msgspec.Struct]:
        if proc == h.target and isinstance(msg, Abort):
            if receipt == "missing":
                return fail("test", "rpc", "Abort lost", code="ipc.timeout")
            return Ok(Resumed(msg.change_id + (receipt == "wrong-id"), OLD,
                              rolled_back=receipt != "not-rolled-back"))
        return await original(proc, msg, timeout=timeout)

    monkeypatch.setattr(h.sup, "call", call)

    async def main() -> None:
        await h.coord.start()
        h.coord.faults.add("after_execute")
        r = await h.submit()
        assert r.status == "failed" and h.coord.recovery_required
        assert "change.abort_unacknowledged" in {d.code for d in r.diagnostics}
        assert h.hosts[h.target].paused
        assert (await h.submit()).status == "failed"

    asyncio.run(main())


def test_discarding_stale_faults_does_not_reset_internal_fairness() -> None:
    current = Identity("current")
    queue = ChangeQueue()
    normal = queue.push(Queued(ChangeSet(base=current)))
    first = queue.push(Queued(ChangeSet(base=current, source="internal", computed_under=current)))
    second = queue.push(Queued(ChangeSet(base=current, source="internal", computed_under=current)))
    assert queue.pop(current) == (first, [])
    stale = queue.push(Queued(ChangeSet(base=current, source="internal", computed_under=Identity("old"),
                                       meta=ChangeMeta(handles_fault=True))))
    assert queue.pop(current) == (normal, [stale])
    assert queue.pop(current) == (second, [])
    assert queue.pop(current) == (None, [])


def test_environment_keeps_legitimate_overrides_and_cleans_both_sources(tmp_path: Path) -> None:
    inherited = {"PATH": "/bin", "PYTHONHASHSEED": "1", "PYTHONHOME": "/wrong", "CUDA_VISIBLE_DEVICES": "8"}
    overrides = {"PYTHONHASHSEED": "2", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
                 "PYTHONPATH": "/outside", "PYTHONSTARTUP": "/script", "CUDA_VISIBLE_DEVICES": "9"}
    spec = ProcSpec(PROC, OLD, ProcessPlan(PROC, OLD, []), "unused", "unused", env=overrides, gpus=[])
    result = child_env(inherited, spec, tmp_path)
    assert result == {"PATH": "/bin", "PYTHONHASHSEED": "2", "PYTHONUNBUFFERED": "1",
                      "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": str(tmp_path),
                      "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    assert inherited["PYTHONHOME"] == "/wrong" and overrides["PYTHONPATH"] == "/outside"
