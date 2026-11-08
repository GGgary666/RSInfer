"""nodeagent.md §7 and supervisor.md §6 on two fake hosts on one machine (P7)."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import pytest
from msgspec.structs import replace

from core.common.ids import HostId, Identity
from core.common.result import Failed, Ok
from core.nodeagent.agent import AgentTunables, NodeAgent, child_env
from core.supervisor.protocol import Materialized, ProcSpec, ProcStatusReport, Spawn, Spawned, Status, Stop
from tests.framework.cluster.harness import ENGINE, REPO, build_tree, cluster, proc, role, topology
from platforms.base import PlatformInfo
from unit.runtime import InstanceState, ProcessPlan


def run(coro: Coroutine[Any, Any, None]) -> None:
    asyncio.run(coro)


def tree_hashes(root: Path) -> dict[str, int]:
    return {str(p.relative_to(root)): hash(p.read_bytes()) for p in sorted(root.rglob("*")) if p.is_file()}


def test_cold_start_on_two_hosts_loads_only_snapshot_code(tmp_path: Path) -> None:
    async def main() -> None:
        async with cluster(tmp_path, ["host-a", "host-b"]) as c:
            identity, files = c.ingest(build_tree(tmp_path))
            topo = topology(tmp_path, ["host-a", "host-b"], ["host-a", "host-b"], [
                role("sched", ["toyc.idle", "toyc.loop"], hosts=["host-a"]),
                role("worker", ["toyc.idle", "toyc.loop"]),
            ])
            plan = (await c.sup.apply_topology(topo, identity, files, c.plan_for(identity))).unwrap()
            assert plan.start == sorted([proc("host-a", "sched", 0), proc("host-a", "worker", 0),
                                         proc("host-b", "worker", 1)])
            for st in c.sup.status():
                assert st.state == "RUNNING"
                assert {u.unit: u.state for u in st.units} == {"toyc.idle": InstanceState.ACTIVE,
                                                               "toyc.loop": InstanceState.ACTIVE}
                checkin = c.sup.checkin_of(st.proc)
                assert checkin is not None
                snapshot = Path(c.agents[st.proc.host].root).resolve() / identity
                assert str(ENGINE) not in checkin.sys_path and str(REPO) not in checkin.sys_path
                assert checkin.module_files, "the process reported no loaded engine files"
                outside = [f for f in checkin.module_files if not f.startswith(str(snapshot) + os.sep)]
                assert outside == []

            worker = proc("host-b", "worker", 1)
            first = (await c.sup.call(worker, Status())).unwrap()
            for _ in range(100):  # checked in, but a safe point (this Status) comes before the first step
                if not isinstance(first, ProcStatusReport) or first.epoch is not None:
                    break
                await asyncio.sleep(0.05)
                first = (await c.sup.call(worker, Status())).unwrap()
            await asyncio.sleep(0.2)
            second = (await c.sup.call(worker, Status())).unwrap()
            assert isinstance(first, ProcStatusReport) and isinstance(second, ProcStatusReport)
            assert first.epoch is not None and second.epoch is not None and second.epoch > first.epoch

            a = Path(c.agents["host-a"].root) / identity
            b = Path(c.agents["host-b"].root) / identity
            assert tree_hashes(a) == tree_hashes(b)
            await c.sup.stop_all()

    run(main())


def test_child_env_replaces_only_where_code_comes_from(tmp_path: Path) -> None:
    w = proc("host-a", "worker", 0)
    spec = ProcSpec(w, Identity("i"), ProcessPlan(w, Identity("i"), []), "tcp://control", "addr",
                    env={"EXTRA": "1"}, gpus=[2, 3])
    environ = {"PYTHONPATH": "/repo/engine", "PYTHONHOME": "/x", "PYTHONSTARTUP": "/y", "PYTHONUSERBASE": "/z",
               "PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1", "PYTHONFAULTHANDLER": "1",
               "CUDA_VISIBLE_DEVICES": "0", "HOME": "/home/u"}
    env = child_env(environ, spec, tmp_path / "engine")
    assert env["PYTHONPATH"] == str(tmp_path / "engine")
    assert not {"PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE"} & set(env)
    assert (env["PYTHONHASHSEED"], env["PYTHONUNBUFFERED"], env["PYTHONFAULTHANDLER"]) == ("0", "1", "1")
    assert (env["CUDA_VISIBLE_DEVICES"], env["EXTRA"], env["HOME"]) == ("2,3", "1", "/home/u")


def test_materialize_reuses_blobs_and_never_leaves_half_snapshots(tmp_path: Path) -> None:
    async def main() -> None:
        async with cluster(tmp_path, ["host-a"]) as c:
            identity, files = c.ingest(build_tree(tmp_path / "v1"))
            first = (await c.sup.materialize(HostId("host-a"), identity, files)).unwrap()
            assert first.fetched == len(set(files.values()))  # type: ignore[attr-defined]

            v2 = build_tree(tmp_path / "v2", {"toyc/extra.py": "X = 2\n"})
            identity2, files2 = c.ingest(v2)
            second = (await c.sup.materialize(HostId("host-a"), identity2, files2)).unwrap()
            assert second.fetched == 1  # type: ignore[attr-defined]  # only the new file crosses the wire

            v3 = build_tree(tmp_path / "v3", {"toyc/extra.py": "X = 3\n"})
            identity3, files3 = c.ingest(v3)
            bad = files3["engine/toyc/extra.py"]
            path = c.blobs.root / bad[:2] / bad[2:]
            path.write_bytes(b"corrupted")
            r = await c.sup.materialize(HostId("host-a"), identity3, files3)
            assert isinstance(r, Failed) and r.has_code("agent.blob_corrupt")
            root = Path(c.agents["host-a"].root)
            assert not (root / identity3).exists()
            assert not list(root.glob(".tmp-*"))

    run(main())


def test_a_snapshot_is_written_once_and_never_rewritten(tmp_path: Path) -> None:
    """stage2-contract.md §10.8: a snapshot path's content never changes once written, so an
    in-process reload never imports stale bytecode (GPU probe P7). Materializing the same identity
    again leaves the files alone, and the snapshot is read-only."""
    async def main() -> None:
        async with cluster(tmp_path, ["host-a"]) as c:
            identity, files = c.ingest(build_tree(tmp_path / "v1"))
            first = (await c.sup.materialize(HostId("host-a"), identity, files)).unwrap()
            assert isinstance(first, Materialized)
            snap = Path(first.path)
            module = snap / "engine" / "toyc" / "units.py"
            before = (module.stat().st_ino, module.stat().st_mtime_ns)
            again = (await c.sup.materialize(HostId("host-a"), identity, files)).unwrap()
            assert isinstance(again, Materialized) and again.path == first.path and again.fetched == 0
            assert (module.stat().st_ino, module.stat().st_mtime_ns) == before
            assert all(d.stat().st_mode & 0o222 == 0 for d in [snap, *(p for p in snap.rglob("*") if p.is_dir())])

    run(main())


def test_stop_kills_a_process_that_does_not_respond(tmp_path: Path) -> None:
    async def main() -> None:
        async with cluster(tmp_path, ["host-a"]) as c:
            identity, files = c.ingest(build_tree(tmp_path))
            topo = topology(tmp_path, ["host-a"], ["host-a"], [role("worker", ["toyc.tunable", "toyc.stubborn"])])
            (await c.sup.apply_topology(topo, identity, files, c.plan_for(identity))).unwrap()
            w = proc("host-a", "worker", 0)
            # Running is not stuck yet: a SIGTERM that arrives before the first step() is honoured at
            # the safe point (exit code 0). Stop it once it no longer reaches safe points.
            deadline = time.monotonic() + 20
            while "stubborn: stuck in step" not in c.log_of(w):
                assert time.monotonic() < deadline, "the stubborn loop never started its step"
                await asyncio.sleep(0.05)
            t0 = time.monotonic()
            (await c.sup.stop(w, timeout_s=0.5)).unwrap()
            assert time.monotonic() - t0 < 5
            assert c.sup.proc_state(w) == "STOPPED"
            assert c.sup.status()[0].exit_code == -signal.SIGKILL

    run(main())


def test_late_and_repeated_control_messages_are_harmless(tmp_path: Path) -> None:
    """ipc.md §4: a timed-out request may still arrive; receivers must cope."""

    async def main() -> None:
        async with cluster(tmp_path, ["host-a"]) as c:
            identity, files = c.ingest(build_tree(tmp_path))
            topo = topology(tmp_path, ["host-a"], ["host-a"], [role("worker", ["toyc.loop"])])
            (await c.sup.apply_topology(topo, identity, files, c.plan_for(identity))).unwrap()
            w = proc("host-a", "worker", 0)
            (await c.sup.restart([w])).unwrap()
            spec = c.sup.procs[w].spec
            agent = c.sup.hosts[HostId("host-a")].agent

            late_stop = await c.sup.client.call(agent, Stop(w, 1.0, spec.incarnation - 1), timeout=5)
            assert isinstance(late_stop, Failed) and late_stop.has_code("agent.stale_stop")
            assert isinstance(await c.sup.call(w, Status()), Ok)  # the successor still runs
            again = await c.sup.client.call(agent, Spawn(spec), timeout=5)
            assert again == Ok(Spawned(w, c.sup.local_pid(w)))
            late_spawn = await c.sup.client.call(agent, Spawn(replace(spec, incarnation=1)), timeout=5)
            assert isinstance(late_spawn, Failed) and late_spawn.has_code("agent.stale_spawn")
            await c.sup.stop_all()

    run(main())


def test_killed_worker_is_reported_lost(tmp_path: Path) -> None:
    async def main() -> None:
        async with cluster(tmp_path, ["host-a"]) as c:
            identity, files = c.ingest(build_tree(tmp_path))
            topo = topology(tmp_path, ["host-a"], ["host-a"], [role("worker", ["toyc.loop"])])
            (await c.sup.apply_topology(topo, identity, files, c.plan_for(identity))).unwrap()
            w = proc("host-a", "worker", 0)
            pid = c.sup.local_pid(w)
            assert pid is not None
            os.kill(pid, signal.SIGKILL)
            loss = await c.wait_loss("process_lost", timeout=5)
            assert loss.procs == [w]
            assert c.sup.proc_state(w) == "LOST"

    run(main())


def test_silent_agent_means_the_machine_is_lost(tmp_path: Path) -> None:
    async def main() -> None:
        async with cluster(tmp_path, ["host-a", "host-b"]) as c:
            identity, files = c.ingest(build_tree(tmp_path))
            topo = topology(tmp_path, ["host-a", "host-b"], ["host-a", "host-b"], [
                role("sched", ["toyc.idle"], hosts=["host-a"]),
                role("worker", ["toyc.loop"]),
            ])
            (await c.sup.apply_topology(topo, identity, files, c.plan_for(identity))).unwrap()
            c.agents["host-b"].kill()
            t0 = time.monotonic()
            loss = await c.wait_loss("machine_lost", timeout=10)
            assert time.monotonic() - t0 < 5
            assert loss.host == "host-b" and loss.procs == [proc("host-b", "worker", 1)]
            assert c.sup.proc_state(proc("host-b", "worker", 1)) == "LOST"
            assert c.sup.proc_state(proc("host-a", "worker", 0)) == "RUNNING"
            assert c.sup.proc_state(proc("host-a", "sched", 0)) == "RUNNING"
            await c.sup.stop(proc("host-a", "worker", 0))
            await c.sup.stop(proc("host-a", "sched", 0))

    run(main())


def test_host_with_a_different_platform_is_refused(tmp_path: Path) -> None:
    async def main() -> None:
        async with cluster(tmp_path, ["host-a"]) as c:
            agent = await c.start_agent("host-c", platform={"name": "fake", "versions": {"driver": "1"}})
            assert await asyncio.wait_for(agent.process.wait(), 20) == 2
            assert HostId("host-c") not in c.sup.hosts

    run(main())


def test_topology_change_only_rebuilds_rank_processes(tmp_path: Path) -> None:
    async def main() -> None:
        async with cluster(tmp_path, ["host-a", "host-b"]) as c:
            identity, files = c.ingest(build_tree(tmp_path))
            roles = [role("frontend", ["toyc.idle"], hosts=["host-a"]),
                     role("sched", ["toyc.loop"], hosts=["host-a"]),
                     role("worker", ["toyc.idle", "toyc.loop"])]  # toyc.idle owns toyc.mesh
            # Workers are built for a mesh; other roles do not depend on it.
            mesh: dict[str, object] = {}
            plan_for = c.plan_for(identity, lambda d: {"toyc.mesh": dict(mesh)} if d.rank is not None else {})
            tp1 = topology(tmp_path, ["host-a", "host-b"], ["host-a"], roles)
            mesh.update(tp1.mesh.dims)
            (await c.sup.apply_topology(tp1, identity, files, plan_for)).unwrap()
            pids = {p: c.sup.local_pid(p) for p in (proc("host-a", "frontend", 0), proc("host-a", "sched", 0))}

            tp2 = topology(tmp_path, ["host-a", "host-b"], ["host-a", "host-b"], roles)
            mesh.update(tp2.mesh.dims)
            plan = (await c.sup.apply_topology(tp2, identity, files, plan_for)).unwrap()

            assert plan.restart == [proc("host-a", "worker", 0)]
            assert plan.start == [proc("host-b", "worker", 1)]
            assert plan.keep == sorted(pids)
            assert {p: c.sup.local_pid(p) for p in pids} == pids
            assert all(st.state == "RUNNING" for st in c.sup.status())
            await c.sup.stop_all()

    run(main())


def test_failed_unit_is_reported_by_check_in(tmp_path: Path) -> None:
    async def main() -> None:
        async with cluster(tmp_path, ["host-a"]) as c:
            src = build_tree(tmp_path, {"toyc/units.py": "class Idle:\n    def __init__(self, ctx, config):\n"
                                                        "        raise RuntimeError('broken')\n"})
            identity, files = c.ingest(src)
            topo = topology(tmp_path, ["host-a"], ["host-a"], [role("worker", ["toyc.idle"])])
            r = await c.sup.apply_topology(topo, identity, files, c.plan_for(identity))
            assert isinstance(r, Failed) and r.has_code("supervisor.unit_failed")
            assert "broken" in r.diagnostics[0].reason
            assert isinstance(await c.sup.call(proc("host-a", "worker", 0), Status()), Ok)
            await c.sup.stop_all()

    run(main())


def test_concurrent_repeated_spawns_start_one_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two copies of one Spawn arriving together (a retry overtaking a slow start)."""

    class Child:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        async def wait(self) -> int:
            await asyncio.Event().wait()
            return 0

    async def main() -> None:
        started: list[int] = []
        slow = asyncio.Event()

        async def create(*args: object, **kwargs: object) -> Child:
            started.append(len(started))
            await slow.wait()
            return Child(1000 + len(started))

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
        agent = NodeAgent(host=HostId("host-a"), address="127.0.0.1", supervisor="unused",
                          snapshot_root=tmp_path / "snapshots", run_root=tmp_path / "runs",
                          platform=PlatformInfo("fake"), tun=AgentTunables())
        (tmp_path / "snapshots" / "i1").mkdir(parents=True)
        w = proc("host-a", "worker", 0)
        spec = ProcSpec(w, Identity("i1"), ProcessPlan(w, Identity("i1"), []), "unused", "127.0.0.1", incarnation=1)
        env = agent.registry.wrap(Spawn(spec), agent.me)
        dispatch = agent._dispatch  # pyright: ignore[reportPrivateUsage] - where the RPC server hands requests
        both = asyncio.gather(dispatch(env, Spawn(spec)), dispatch(env, Spawn(spec)))
        await asyncio.sleep(0.05)
        slow.set()
        first, second = await both
        assert started == [0] and first == second == Ok(Spawned(w, 1001))

    run(main())
