"""The scheduler host and two fixture workers as real processes started by a real control plane,
on the hosts' data channels: worker 0 shares the scheduler's host (shared memory), worker 1 is on
another (TCP). Requests keep going while ``core.sched`` is reloaded through the control interface.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import msgspec
import zmq
import zmq.asyncio

from core.change.config import ChangeTunables
from core.common.ids import HostId, ProcId
from core.common.manifest import MANIFEST_FILE, load_manifest
from core.common.result import Failed, Ok
from core.control.client import ControlClient
from core.contracts.columns import Column
from core.contracts.requests import TokenEvents
from core.control.host import ControlPlane, ingest
from core.ipc.channels import Channels
from core.supervisor import HostSpec, Mesh, RankPlacement, RoleSpec, SupervisorTunables, Topology
from tests.scenarios.harness import ADDRESS, PROC_TUNABLES, rungs, start_agent
from tests.units.scheduler.support import DECLS, PAGES, TUN, ThreadClient, request
from tests.units.scheduler.units.schedfix.worker.unit import expected

ENGINE = Path(__file__).resolve().parents[3] / "engine"
FIXTURES = Path(__file__).resolve().parent / "units" / "schedfix"
HOSTS = ("host-a", "host-b")
SCHED = ProcId(HostId("host-a"), "sched", 0)
WORKERS = [ProcId(HostId("host-a"), "worker", 0), ProcId(HostId("host-b"), "worker", 1)]
CAPACITY = {"kv.paged": 600, "gdn.slots": 8}
KEEP = {"sched.reqs", "core.state.alloc", "policies.fcfs", "policies.prefill_first", "core.sched"}
"""The repo's units outside the control plane that the workspace keeps (no frontend, no kernels)."""


def build_workspace(root: Path) -> Path:
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")

    def skip(directory: str, names: list[str]) -> set[str]:
        out = set(ignore(directory, names))
        for name in names:
            path = Path(directory, name, MANIFEST_FILE)
            r = load_manifest(path) if path.is_file() else None
            if isinstance(r, Ok) and "control" not in r.value.unit.process and str(r.value.unit.id) not in KEEP:
                out.add(name)
        return out

    shutil.copytree(ENGINE, root / "engine", ignore=skip)
    shutil.copytree(FIXTURES, root / "engine" / "schedfix", ignore=ignore)
    return root


def topology(tmp: Path) -> Topology:
    return Topology(hosts=[HostSpec(HostId(h), ADDRESS, str(tmp / h / "snapshots")) for h in HOSTS],
                    mesh=Mesh({"tp": 2}), placement=[RankPlacement(i, HostId(h), i) for i, h in enumerate(HOSTS)],
                    roles=[RoleSpec("sched", [], hosts=[HostId("host-a")], leads_epochs=True),
                           RoleSpec("worker", [], per_rank=True)])


@dataclass
class SupervisorLookup:
    """The client's channels find the scheduler's reader through the supervisor's endpoint table."""

    cp: ControlPlane

    def channels(self, zctx: zmq.Context[zmq.Socket[bytes]], proc: ProcId) -> Channels:
        return Channels(zctx, local=proc.host, bind=ADDRESS, tun=TUN, allocate=lambda name: name,
                        lookup=lambda p, name: self.cp.sup.endpoint(p, name), publish=lambda name, address: None)


@dataclass
class Running:
    workspace: Path
    cp: ControlPlane
    control: ControlClient
    client: ThreadClient


@asynccontextmanager
async def engine(tmp: Path, *, overlap: bool) -> AsyncGenerator[Running]:
    # The control plane in this process drops library modules imported from elsewhere, so that
    # its units load them from the snapshot; msgspec resolves a type's annotations in its module the
    # first time it decodes one. Decode the client's tokens once before, while the module is there.
    events = TokenEvents("", Column.of("i64"), Column.of("i32", [0]), Column.of("i64"))
    msgspec.msgpack.decode(msgspec.msgpack.encode(events), type=TokenEvents)
    zctx = zmq.asyncio.Context()
    cp = ControlPlane(zctx, host=HostId("host-a"), bind=ADDRESS, state_dir=tmp / "control",
                      sup_tun=SupervisorTunables(heartbeat_timeout_s=10.0, monitor_interval_s=0.05,
                                                 checkin_timeout_s=30.0, rpc_timeout_s=10.0, stop_timeout_s=5.0),
                      change_tun=ChangeTunables(coordinator_timeout_s=30.0, keepalive_interval_s=1.0),
                      proc_tunables=dict(PROC_TUNABLES))
    await cp.start()
    agents = [await start_agent(tmp, h, cp.address) for h in HOSTS]
    control = ControlClient(zctx, cp.address)
    client: ThreadClient | None = None
    try:
        (await cp.sup.wait_for_hosts([HostId(h) for h in HOSTS], 20)).unwrap()
        workspace = build_workspace(tmp / "workspace")
        # As a client sends it: JSON values (a tuple would compare unequal to the list the process
        # reports back, and look like a new value to the next change).
        config = {"alloc.states": msgspec.json.decode(msgspec.json.encode(DECLS)), "sched.workers": msgspec.to_builtins(WORKERS),
                  "sched.frontend_host": "client", "sched.max_batch": 8, "sched.token_budget": 16,
                  "sched.overlap": overlap, "schedfix.scheduler": msgspec.to_builtins(SCHED), "schedfix.capacity": CAPACITY,
                  "schedfix.pages": dict(PAGES), "schedfix.max_batch": 8, "schedfix.delay_s": 0.01}
        started = (await cp.boot(ingest(cp.blobs, workspace), topology(tmp), config)).unwrap()
        assert started.status == "committed", started.diagnostics
        client = ThreadClient(SupervisorLookup(cp))
        yield Running(workspace, cp, control, client)
    finally:
        if client is not None:
            client.close()
        control.close()
        await cp.sup.stop_all()
        cp.close()
        for a in agents:
            if a.returncode is None:
                os.killpg(a.pid, signal.SIGKILL)
            await a.wait()
        zctx.destroy(linger=0)


async def until(cond: Callable[[], bool], what: Callable[[], object], timeout: float = 30.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not cond():
        assert asyncio.get_running_loop().time() < deadline, f"condition not reached: {what()}"
        await asyncio.sleep(0.02)


def test_requests_run_on_two_worker_processes_through_a_reload_of_the_scheduler(tmp_path: Path) -> None:
    async def main() -> None:
        async with engine(tmp_path, overlap=True) as r:
            out = r.client.outputs
            long = request(1, [5, 6, 7], 1500)  # outlasts the change, however slow it is; aborted then
            reqs = [request(2, list(range(1, 25)), 60), request(3, [9], 70)]
            r.client.send([long, *reqs])
            seen = lambda: {rid: (len(o.tokens), o.finished, o.error) for rid, o in out.items()}  # noqa: E731
            await until(lambda: len(out[1].tokens) >= 3, seen)
            unit = r.workspace / "engine" / "core" / "sched" / "unit.py"
            unit.write_text(unit.read_text() + "\n# a new version of the scheduler logic\n")
            change = await r.control.change(r.workspace)
            assert not isinstance(change, Failed), change
            applied = await r.control.apply(change.value)
            assert not isinstance(applied, Failed), applied
            result = applied.value.result
            assert result.status == "committed", result.diagnostics
            assert 2 in rungs(result.plan, SCHED) and all(not rungs(result.plan, w) for w in WORKERS)  # a reload
            await until(lambda: str(result.new_identity) in out[1].identities, seen)
            assert len(out[1].identities) == 2  # the same request, before and after the reload
            r.client.send(abort=[1])
            await until(lambda: all(out[rid].finished for rid in (1, 2, 3)), seen, timeout=60)
            assert out[1].finished == "abort" and len(out[1].tokens) < long.max_tokens
            for q in [long, *reqs]:
                assert q.sampling.seed is not None
                prompt = [int(x) for x in q.prompt.values()]
                assert out[q.request].tokens == expected(prompt, q.sampling.seed, q.max_tokens)[:len(out[q.request].tokens)]
            assert [len(out[q.request].tokens) for q in reqs] == [q.max_tokens for q in reqs]

    asyncio.run(main())
