"""The engine under test: a coordinator over real processes on two fake hosts."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgspec

from core.change.config import ChangeTunables
from core.change.coordinator import Coordinator, PlanInput
from core.change.eventlog import EventLog
from core.change.types import EMPTY, ActionPlan, ChangeSet, ProcActions, UnitAction
from core.common.ids import ProcId, UnitId
from core.common.result import Ok, Result
from core.supervisor import Topology, desired_procs
from core.supervisor.protocol import ProcStatusReport, Status
from tests.framework.cluster.harness import SPECS, VERSIONED_V2, Cluster, build_tree, cluster, proc, process_plan, role, topology
from unit.runtime import ReloadPlan

SCHED = proc("host-a", "sched", 0)
W0 = proc("host-a", "worker", 0)
W1 = proc("host-b", "worker", 1)
ALL = [SCHED, W0, W1]
UNITS = ["toyc.loop", "toyc.tunable", "toyc.versioned"]

type Script = Callable[[PlanInput], ActionPlan | Awaitable[ActionPlan]]


class Scripted:
    """Stands in for core.deps: startup spawns everything; other plans come from the test."""

    def __init__(self, c: Cluster) -> None:
        self.c = c
        self.script: Script | None = None

    async def plan(self, inp: PlanInput) -> Result[ActionPlan]:
        if inp.cs.source == "startup":
            topo = inp.cs.topology
            assert topo is not None
            return Ok(ActionPlan(spawn=[process_plan(d, EMPTY, {"toyc.log_dir": str(self.c.logs)})
                                        for d in desired_procs(topo).values()], topology=topo))
        assert self.script is not None
        out = self.script(inp)
        return Ok(await out if isinstance(out, Awaitable) else out)


@dataclass
class Engine:
    c: Cluster
    coord: Coordinator
    planner: Scripted
    topo: Topology

    async def status(self, p: ProcId) -> ProcStatusReport:
        r = (await self.c.sup.call(p, Status())).unwrap()
        assert isinstance(r, ProcStatusReport)
        return r

    def versioned(self, p: ProcId) -> str:
        return [line for line in self.c.log_of(p) if line.startswith("versioned")][-1]

    def ops(self, p: ProcId) -> list[str]:
        """The change phases ``toy.op`` went through (slices excluded)."""
        return [line for line in self.c.log_of(p) if line.startswith("op ") and not line.startswith("op slice")]

    def slices(self, p: ProcId) -> list[str]:
        return [line for line in self.c.log_of(p) if line.startswith("op slice")]

    def code_change(self) -> ChangeSet:
        h = self.c.blobs.put(VERSIONED_V2.encode())
        return ChangeSet(base=self.coord.identity, files={"engine/toyc/versioned.py": h})

    async def epochs_advance(self, p: ProcId) -> bool:
        a = (await self.status(p)).epoch or 0
        await asyncio.sleep(0.1)
        return ((await self.status(p)).epoch or 0) > a


def reload_everywhere(fail: dict[ProcId, str] | None = None) -> Callable[[PlanInput], ActionPlan]:
    fail = fail or {}

    def script(inp: PlanInput) -> ActionPlan:
        return ActionPlan(procs=[
            ProcActions(p, reload=ReloadPlan(changed_modules=["toyc.versioned"],
                                             units_to_reload=[SPECS["toyc.versioned"]]),
                        actions=[UnitAction(UnitId("toyc.tunable"), "toy.op",
                                            {"fail_at": fail[p]} if p in fail else {})],
                        leads_epochs=p == SCHED)
            for p in ALL])

    return script


@asynccontextmanager
async def engine(tmp: Path, *, agent: dict[str, float] | None = None, sup: dict[str, float] | None = None,
                 **tun: Any) -> AsyncGenerator[Engine]:
    """``tun``: coordinator tunables (``ChangeTunables`` fields); ``agent``, ``sup``: node agent
    and supervisor tunables."""
    async with cluster(tmp, ["host-a", "host-b"], agent_tunables=agent, sup_tunables=sup) as c:
        planner = Scripted(c)
        coord = Coordinator(c.sup, EventLog(tmp / "events.sqlite", c.blobs), planner=planner,
                            tun=msgspec.structs.replace(ChangeTunables(coordinator_timeout_s=30.0, drain_timeout_s=5.0),
                                                        **tun))
        await coord.start()
        topo = topology(tmp, ["host-a", "host-b"], ["host-a", "host-b"],
                        [role("sched", UNITS, hosts=["host-a"]), role("worker", UNITS)])
        started = await coord.startup(c.blobs.ingest_tree(build_tree(tmp)), topo)
        assert started.status == "committed", started.diagnostics
        try:
            yield Engine(c, coord, planner, topo)
        finally:
            coord.close()
            await c.sup.stop_all()


def run(coro: Coroutine[Any, Any, None]) -> None:
    asyncio.run(coro)


async def wait_until(cond: Callable[[], bool], timeout: float = 20.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not cond():
        assert asyncio.get_running_loop().time() < deadline, "condition not reached"
        await asyncio.sleep(0.02)
