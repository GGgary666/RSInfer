"""Initial regression cases for stop, rollback, queue and environment boundaries."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import msgspec
import pytest

from core.change.queue import ChangeQueue, Queued
from core.change.types import ActionPlan, ChangeMeta, ChangeSet
from core.common.ids import Identity
from core.common.result import Failed, Result, fail
from core.nodeagent.agent import child_env
from core.supervisor.protocol import ProcSpec
from core.supervisor.supervisor import _Host
from core.supervisor.topology import HostSpec, Mesh, RankPlacement, RoleSpec, Topology
from platforms.base import PlatformInfo
from tests.change.test_commit_consistency import Harness, OLD, harness as harness
from tests.support import PROC
from unit.runtime import ProcessPlan, ReloadPlan
from unit.runtime.types import ReloadReport


@pytest.mark.parametrize("mode", ["stop", "transaction"])
def test_lost_host_is_not_evidence_that_its_process_stopped(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    h = harness
    target = h.target
    h.sup.hosts[target.host] = _Host(target.host, "unreachable", PlatformInfo("fake"), time.monotonic(), lost=True)
    h.sup.procs[target].state = "LOST"
    calls: list[str] = []

    async def call(address: str, msg: msgspec.Struct, *, timeout: float) -> Result[msgspec.Struct]:
        calls.append(type(msg).__name__)
        return fail("audit", "rpc", "host partitioned", code="ipc.timeout")

    monkeypatch.setattr(h.sup.client, "call", call)

    async def main() -> None:
        if mode == "stop":
            result = await h.sup.stop(target)
            print("STOP", result, "state", h.sup.proc_state(target), "requests", calls)
            assert isinstance(result, Failed), "a LOST record without stop acknowledgement is not a stopped process"
        else:
            small = Topology([HostSpec(h.first.host, "127.0.0.1", "/unused")], Mesh({"tp": 1}),
                             [RankPlacement(0, h.first.host, 0)], [RoleSpec("worker", [h.unit.id], per_rank=True)])
            h.action_plan = ActionPlan(procs=[h.action_plan.procs[0]], stop=[target], topology=small)
            await h.coord.start()
            result = await h.submit()
            serving = {str(p): (host.identity, h.unit_of(p).version) for p, host in h.hosts.items() if not host.paused}
            print("TOPOLOGY", result.status, "forgotten", target not in h.sup.procs,
                  "requests", calls, "serving", serving, "recovery", h.coord.recovery_required)
            assert result.status != "committed", "unconfirmed stop must not produce committed topology removal"

    asyncio.run(main())


@pytest.mark.parametrize("failure", ["action-abort", "reverse-reload"])
def test_failed_rollback_is_reported_and_blocks_following_changes(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    h = harness
    host = h.hosts[h.target]
    if failure == "action-abort":
        def abort() -> None:
            raise RuntimeError("injected resource rollback failure")

        monkeypatch.setattr(h.unit_of(h.target), "abort", abort)
    else:
        h.action_plan = msgspec.structs.replace(h.action_plan, procs=[
            msgspec.structs.replace(pa, reload=ReloadPlan(units_to_reload=[h.unit])) for pa in h.action_plan.procs])
        original = host.runtime.reload
        calls = 0

        def reload(plan: ReloadPlan, *, values: dict[str, Any] | None = None) -> Result[ReloadReport]:
            nonlocal calls
            calls += 1
            if calls == 2:
                return fail("audit", "reload", "reverse reload requires process restart", code="escalate.restart_process")
            return original(plan, values=values)

        monkeypatch.setattr(host.runtime, "reload", reload)

    async def main() -> None:
        await h.coord.start()
        h.coord.faults.add("after_execute")
        result = await h.submit()
        local_codes = [d.code for d in host.participant.diagnostics]
        reply_codes = [d.code for d in result.diagnostics]
        frozen = h.coord.recovery_required
        print("ROLLBACK", failure, "result", result.status, "local", local_codes, "returned", reply_codes,
              "paused", host.paused, "value", h.unit_of(h.target).version, "recovery", frozen)
        # Even a subsequent empty plan must be rejected while rollback is unresolved.
        h.action_plan = ActionPlan()
        following = await h.coord.submit(ChangeSet(base=OLD, config={"revision": 3}))
        print("FOLLOWING", following.status)
        assert result.status == "failed" and frozen and following.status == "failed", (
            "rollback failure must not be acknowledged as restored and admit another transaction")

    asyncio.run(main())


@pytest.mark.parametrize("fault", [False, True])
@pytest.mark.parametrize("stale", [False, True])
def test_internal_staleness_is_checked_independently_of_priority(fault: bool, stale: bool) -> None:
    current = Identity("current")
    queue = ChangeQueue()
    item = Queued(ChangeSet(base=current, source="internal", computed_under=Identity("old") if stale else current,
                            meta=ChangeMeta(handles_fault=fault)))
    queue.push(item)
    got, dropped = queue.pop(current)
    print("QUEUE", "fault", fault, "stale", stale, "returned", got is item, "dropped", len(dropped))
    assert (got is None and dropped == [item]) if stale else (got is item and dropped == [])


def test_operator_fault_is_not_an_internal_computation() -> None:
    queue = ChangeQueue()
    item = Queued(ChangeSet(base=Identity("old"), source="operator", meta=ChangeMeta(handles_fault=True)))
    queue.push(item)
    assert queue.pop(Identity("current")) == (item, [])


@pytest.mark.parametrize("key,value", [
    ("PYTHONPATH", "/outside/engine"), ("PYTHONHOME", "/outside/python"),
    ("PYTHONSTARTUP", "/outside/start.py"), ("PYTHONUSERBASE", "/outside/user"),
    ("CUDA_VISIBLE_DEVICES", "7"), ("PYTHONDONTWRITEBYTECODE", "0"), ("EXTRA", "ok"),
])
def test_process_overrides_do_not_replace_snapshot_or_gpu_settings(tmp_path: Path, key: str, value: str) -> None:
    root = tmp_path / "snapshot" / "engine"
    ps = ProcSpec(PROC, Identity("i"), ProcessPlan(PROC, Identity("i"), []), "unused", "unused",
                  env={key: value}, gpus=[2, 3])
    env = child_env({}, ps, root)
    print("ENV", key, env.get(key))
    expected = {"PYTHONPATH": str(root), "CUDA_VISIBLE_DEVICES": "2,3", "PYTHONDONTWRITEBYTECODE": "1", "EXTRA": "ok"}
    if key in expected:
        assert env[key] == expected[key]
    else:
        assert key not in env
