"""The scenario test bed: a real control plane (in the test's event loop) with its
control-plane units, two fake hosts (node agents as child processes), and a
workspace = this repo's ``engine/`` plus the scenario units as package ``scn``.

Changes go through the control interface (``ControlClient``) exactly as an agent
would send them: edit the workspace, upload, apply.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgspec
import zmq.asyncio

from core.change.config import ChangeTunables
from core.change.types import ActionPlan, ChangeResult
from core.common.ids import HostId, ProcId
from core.common.result import Failed
from core.control.client import ControlClient
from core.control.host import ControlPlane, ingest
from core.control.protocol import ApiChange, ApiStatusReport
from core.supervisor import HostSpec, Mesh, RankPlacement, RoleSpec, SupervisorTunables, Topology

REPO = Path(__file__).resolve().parents[2]
ENGINE = REPO / "engine"
UNITS = Path(__file__).resolve().parent / "units"
ADDRESS = "127.0.0.1"  # test data: every fake host lives on this machine
HOSTS = ("host-a", "host-b")

SCHED = ProcId(HostId("host-a"), "sched", 0)
W0 = ProcId(HostId("host-a"), "worker", 0)
W1 = ProcId(HostId("host-b"), "worker", 1)
ALL = (SCHED, W0, W1)

AGENT_TUNABLES = {"nodeagent.heartbeat_interval_s": 0.1, "nodeagent.rpc_timeout_s": 5.0}
PROC_TUNABLES = {"host.idle_poll_s": 0.01}


def build_workspace(root: Path) -> Path:
    """What an agent works on: a source tree whose ``engine/`` holds the framework and the units."""
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ENGINE, root / "engine", ignore=ignore)
    shutil.copytree(UNITS / "scn", root / "engine" / "scn", ignore=ignore)
    return root


def topology(tmp: Path, ranks: tuple[str, ...] = HOSTS) -> Topology:
    """A scheduler host on host-a and one worker per rank; role units come from the manifests."""
    return Topology(
        hosts=[HostSpec(HostId(h), ADDRESS, str(tmp / h / "snapshots")) for h in HOSTS],
        mesh=Mesh({"tp": len(ranks)}),
        placement=[RankPlacement(i, HostId(h), i) for i, h in enumerate(ranks)],
        roles=[RoleSpec("sched", [], hosts=[HostId("host-a")], leads_epochs=True),
               RoleSpec("worker", [], per_rank=True)],
    )


@dataclass
class Scenario:
    tmp: Path
    workspace: Path
    cp: ControlPlane
    client: ControlClient
    topo: Topology
    agents: list[asyncio.subprocess.Process] = field(default_factory=list[asyncio.subprocess.Process])

    @property
    def logs(self) -> Path:
        return self.tmp / "logs"

    def log_of(self, p: ProcId) -> list[str]:
        f = self.logs / f"{p.host}-{p.role}-{p.index}.log"
        return f.read_text().splitlines() if f.exists() else []

    def lines(self, p: ProcId, prefix: str) -> list[str]:
        return [line for line in self.log_of(p) if line.startswith(prefix)]

    def edit(self, rel: str, old: str, new: str) -> None:
        """Change a file of the workspace, as an agent would."""
        path = self.workspace / "engine" / rel
        text = path.read_text()
        assert old in text, (rel, old)
        path.write_text(text.replace(old, new))

    def write(self, rel: str, text: str) -> None:
        path = self.workspace / "engine" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    async def change(self, *, files: bool = True, **kw: Any) -> ApiChange:
        r = await self.client.change(self.workspace if files else None, **kw)
        assert not isinstance(r, Failed), r
        return r.value

    async def apply(self, *, dry_run: bool = False, files: bool = True, **kw: Any) -> ChangeResult:
        r = await self.client.apply(await self.change(files=files, **kw), dry_run=dry_run)
        assert not isinstance(r, Failed), r
        return r.value.result

    async def status(self) -> ApiStatusReport:
        r = await self.client.status()
        assert not isinstance(r, Failed), r
        return r.value

    def pid(self, p: ProcId) -> int | None:
        return self.cp.sup.local_pid(p)


async def start_agent(tmp: Path, host: str, control: str) -> asyncio.subprocess.Process:
    root = tmp / host / "snapshots"
    return await asyncio.create_subprocess_exec(
        sys.executable, "-m", "core.nodeagent", "--host", host, "--address", ADDRESS, "--supervisor", control,
        "--root", str(root), "--platform", json.dumps({"name": "fake", "versions": {"driver": "0"}}),
        "--tunables", json.dumps(AGENT_TUNABLES),
        env={**os.environ, "PYTHONPATH": str(ENGINE)}, start_new_session=True,
        stdout=open(tmp / f"agent-{host}.log", "wb"), stderr=asyncio.subprocess.STDOUT,  # noqa: SIM115
    )


@asynccontextmanager
async def scenario(tmp: Path, *, ranks: tuple[str, ...] = HOSTS, files: dict[str, str] | None = None,
                   **change_tun: Any) -> AsyncGenerator[Scenario]:
    """A started engine. ``files``: more ``engine/`` files of the workspace it starts from
    (``{"core/x/unit.py": text}``). ``change_tun``: ``ChangeTunables`` fields for the coordinator."""
    zctx = zmq.asyncio.Context()
    tun = msgspec.structs.replace(ChangeTunables(coordinator_timeout_s=30.0, keepalive_interval_s=1.0), **change_tun)
    cp = ControlPlane(zctx, host=HostId("host-a"), bind=ADDRESS, state_dir=tmp / "control",
                      sup_tun=SupervisorTunables(heartbeat_timeout_s=10.0, monitor_interval_s=0.05,
                                                 checkin_timeout_s=30.0, rpc_timeout_s=10.0, stop_timeout_s=5.0),
                      change_tun=tun, proc_tunables=dict(PROC_TUNABLES))
    await cp.start()
    agents = [await start_agent(tmp, h, cp.address) for h in HOSTS]
    client = ControlClient(zctx, cp.address)
    try:
        (await cp.sup.wait_for_hosts([HostId(h) for h in HOSTS], 20)).unwrap()
        workspace = build_workspace(tmp / "workspace")
        for rel, text in (files or {}).items():
            (workspace / "engine" / rel).parent.mkdir(parents=True, exist_ok=True)
            (workspace / "engine" / rel).write_text(text)
        (tmp / "logs").mkdir()
        topo = topology(tmp, ranks)
        started = (await cp.boot(ingest(cp.blobs, workspace), topo, {"scn.log_dir": str(tmp / "logs")})).unwrap()
        assert started.status == "committed", started.diagnostics
        yield Scenario(tmp, workspace, cp, client, topo, agents)
    finally:
        client.close()
        await cp.sup.stop_all()
        cp.close()
        for a in agents:
            if a.returncode is None:
                os.killpg(a.pid, signal.SIGKILL)
            await a.wait()
        zctx.destroy(linger=0)


STATE_ACTIONS = frozenset({"refit", "grow", "relocate", "invalidate"})


def rungs(plan: ActionPlan | None, proc: ProcId | None = None) -> set[int]:
    """The rungs of the action ladder a plan uses (for one process, or all of them)."""
    if plan is None:
        return set()
    if plan.restart_engine:
        return {6}
    out: set[int] = set()
    if proc is None and (plan.spawn or plan.stop):
        out.add(5)
    if proc is not None and proc in plan.stop:
        out.add(5)
    for pa in [*plan.procs, *([plan.control] if plan.control else [])]:
        if proc is not None and pa.proc != proc:
            continue
        if pa.restart is not None:
            out.add(5)
            continue
        if pa.config:
            out.add(1)
        if pa.reload is not None and (pa.reload.units_to_reload or pa.reload.units_to_add or pa.reload.units_to_remove):
            out.add(2)
        for a in pa.actions:
            out.add(4 if a.action in STATE_ACTIONS else 3)
        if pa.stage:
            out.add(0)
    return out


async def generations(s: Scenario, p: ProcId) -> dict[str, int]:
    """Unit generations as the process reports them now (status rows date from its check-in)."""
    r = await s.client.introspect()
    assert not isinstance(r, Failed), r
    row = next(i for i in r.value.procs if i.proc == p)
    return {u.unit: u.generation for u in row.units}
