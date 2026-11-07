"""The nine scenarios of modules.html, on two fake hosts, through the control interface.

Each test exercises what its scenario needs from the framework and the control-plane
units, and leaves out what belongs to the model domain (tokenizer, HTTP, real
admission, real collectives); the docstrings say what is in and what is out.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import pytest

from core.change.types import ChangeResult
from core.common.ids import UnitId
from core.common.result import Failed
from core.supervisor.protocol import ProcIntrospection
from tests.scenarios.harness import ALL, SCHED, W0, W1, Scenario, generations, rungs, scenario, topology
from unit.runtime import DerivedArtifact


def run(coro: Coroutine[Any, Any, None]) -> None:
    asyncio.run(coro)


async def until(cond: Callable[[], bool], timeout: float = 20.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not cond():
        assert asyncio.get_running_loop().time() < deadline, "condition not reached"
        await asyncio.sleep(0.05)


async def introspect(s: Scenario) -> dict[str, ProcIntrospection]:
    r = await s.client.introspect()
    assert not isinstance(r, Failed), r
    return {str(p.proc): p for p in r.value.procs}


def no_replay_failures(s: Scenario) -> None:
    for p in (W0, W1):
        assert not s.lines(p, "worker replay failed"), s.log_of(p)


async def everyone_on(s: Scenario, r: ChangeResult) -> None:
    assert r.status == "committed", r.diagnostics
    st = await s.status()
    assert st.identity == r.new_identity
    assert {row.proc: row.identity for row in st.procs} == {str(p): r.new_identity for p in ALL}


@pytest.mark.covers(edges=("call", "state", "derive", "decl"), kinds=("logic", "state"))
def test_1_one_decode_step(tmp_path: Path) -> None:
    """① A step: the worker loop asks the plan for the graph and replays it; the graph was built from
    the kernel in the catalog (registered there) and the config by value, and reads the memory unit's
    buffers by address. Out: frontend, tokenizer, StepPlan transport (covered by tests/framework/ipc)."""

    async def main() -> None:
        async with scenario(tmp_path) as s:
            intro = await introspect(s)
            for p in (W0, W1):
                w = intro[str(p)]
                assert w.artifacts == [DerivedArtifact(UnitId("scn.plan"), "decode", "scn.rebuild",
                                                       ["scn.kernel", "scn.max_batch"], [UnitId("scn.kernel_basic")])]
                built = {(r.key, r.phase, r.artifact) for r in w.reads if r.unit == "scn.plan"}
                assert {("scn.kernel", "plan_build", "decode"), ("scn.max_batch", "plan_build", "decode")} <= built
            st1 = await s.status()
            await asyncio.sleep(0.3)
            st2 = await s.status()
            assert st1.identity == st2.identity
            no_replay_failures(s)

    run(main())


@pytest.mark.covers(edges=("call",), ladder=(2,), kinds=("logic",))
def test_3_replacing_a_policy_reloads_only_that_unit(tmp_path: Path) -> None:
    """③ apply: a policy's code changed -> reload that unit only; the scheduling logic that
    calls it, the request table and every worker stay as they are."""

    async def main() -> None:
        async with scenario(tmp_path) as s:
            before = {p: await generations(s, p) for p in ALL}
            s.edit("scn/policy_fcfs/unit.py", "VERSION = 1", "VERSION = 2")
            dry = await s.apply(dry_run=True)
            r = await s.apply()
            assert r.plan == dry.plan and dry.new_identity == r.new_identity
            await everyone_on(s, r)
            assert rungs(r.plan, SCHED) == {2} and rungs(r.plan, W0) == rungs(r.plan, W1) == set()
            pa = next(pa for pa in r.plan.procs if pa.proc == SCHED)  # type: ignore[union-attr]
            assert pa.reload is not None and [u.id for u in pa.reload.units_to_reload] == ["scn.policy_fcfs"]
            after = {p: await generations(s, p) for p in ALL}
            assert {u for u, g in after[SCHED].items() if g != before[SCHED][u]} == {"scn.policy_fcfs"}
            assert after[W0] == before[W0] and after[W1] == before[W1]
            assert s.lines(SCHED, "policy")[-1] == "policy v2 gen2"

    run(main())


KERNEL_FAST = '''"""A faster decode kernel (scenario ④)."""

from __future__ import annotations

from typing import Any

NAME = "fast"


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        ctx.register("scn.kernels@1", (NAME, ctx.id))
'''
KERNEL_FAST_TOML = '''[unit]
id = "scn.kernel_fast"
kind = "logic"
process = ["worker"]
entry = "scn.kernel_fast.unit:Unit"

[requires]
services = ["scn.kernels@1"]
'''


@pytest.mark.covers(edges=("derive", "decl"), ladder=(0, 3), kinds=("logic",))
def test_4_a_new_kernel_rebuilds_only_the_graph_that_uses_it(tmp_path: Path) -> None:
    """④ apply: a new kernel unit, and the config selecting it -> the kernel unit is staged during
    prepare (registering into the catalog), the plan unit rebuilds its graph next to the old one;
    nothing is reloaded, the memory (KV) stays. Out: second buffers on a real GPU."""

    async def main() -> None:
        async with scenario(tmp_path) as s:
            before = {p: await generations(s, p) for p in (W0, W1)}
            s.write("scn/kernel_fast/__init__.py", "")
            s.write("scn/kernel_fast/unit.toml", KERNEL_FAST_TOML)
            s.write("scn/kernel_fast/unit.py", KERNEL_FAST)
            dry = await s.apply(dry_run=True, config={"scn.kernel": "fast"})
            r = await s.apply(config={"scn.kernel": "fast"})
            assert r.plan == dry.plan
            await everyone_on(s, r)
            for p in (W0, W1):
                assert rungs(r.plan, p) == {0, 1, 3}, r.plan
                pa = next(pa for pa in r.plan.procs if pa.proc == p)  # type: ignore[union-attr]
                assert [u.id for u in pa.stage] == ["scn.kernel_fast"] and pa.reload is None
                assert [(a.unit, a.action) for a in pa.actions] == [("scn.plan", "scn.rebuild")]
                assert s.lines(p, "plan switched")[-1] == "plan switched to {'scn.kernel': 'fast', 'scn.max_batch': 8}"
            for p in (W0, W1):
                after = await generations(s, p)
                assert after.pop("scn.kernel_fast") == 1 and after == before[p]
            intro = await introspect(s)
            assert intro[str(W0)].artifacts[0].units == ["scn.kernel_fast"]
            await asyncio.sleep(0.2)
            no_replay_failures(s)

    run(main())


@pytest.mark.covers(edges=("state", "ctrl"), ladder=(1, 4), kinds=("state",))
def test_5_rebalancing_rewrites_state_in_place_and_keeps_the_identity(tmp_path: Path) -> None:
    """⑤ EPLB: the balancer (asked through a live config key) submits an internal, state-only change;
    every worker's memory unit rewrites the placement table in place, at the same epoch boundary.
    The graph reads it by address: nothing is rebuilt. Identity unchanged, state version + 1.
    Out: load counting, the all-reduce, moving expert weights between ranks."""

    async def main() -> None:
        async with scenario(tmp_path) as s:
            r = await s.apply(files=False, config={"scn.rebalance": 1})
            await everyone_on(s, r)
            assert rungs(r.plan, SCHED) == {1}
            identity = r.new_identity
            await until(lambda: all(s.lines(p, "memory placement") for p in (W0, W1)))
            st = await s.status()
            for _ in range(200):  # the placement is written at Execute; wait for the commit
                if st.state_version == 1 and not st.busy:
                    break
                await asyncio.sleep(0.05)
                st = await s.status()
            assert st.identity == identity and st.state_version == 1 and not st.recovery_required
            for p in (W0, W1):
                assert s.lines(p, "memory placement") == ["memory placement [1, 2, 3, 0]"]
                assert not s.lines(p, "plan switched") and len(s.lines(p, "plan recorded")) == 1
            logs = await s.client.logs()
            assert not isinstance(logs, Failed)
            internal = [c for c in logs.value.changes if c.source == "internal"]
            assert [c.status for c in internal] == ["committed"]
            assert internal[0].plan is not None and {pa["proc"]["role"] for pa in internal[0].plan["procs"]
                                                     if pa["actions"]} == {"worker"}
            await asyncio.sleep(0.2)
            no_replay_failures(s)

    run(main())


BROKEN_IMPORT = "from scn.memory.unit import Unit as Memory  # reaches into another unit\n"


@pytest.mark.covers(edges=("ctrl",), kinds=("framework",))
def test_6_a_patch_is_checked_before_anything_runs(tmp_path: Path) -> None:
    """⑥ (engine side of patch evaluation): a static check rejects a bad patch before any process is
    touched; a good one gets its candidate identity up front, and applying it yields exactly that
    identity, a function of (base, patch). Out: the evaluation service and lineage (stage 4)."""

    async def main() -> None:
        async with scenario(tmp_path) as s:
            pids = {p: s.pid(p) for p in ALL}
            before = (await s.status()).identity
            s.edit("scn/policy_fcfs/unit.py", "VERSION = 1", "VERSION = 3\n" + BROKEN_IMPORT)
            bad = await s.client.check(await s.change())
            assert isinstance(bad, Failed) and "manifest.cross_unit_import" in {d.code for d in bad.diagnostics}
            r = await s.apply()
            assert r.status == "failed" and "manifest.cross_unit_import" in {d.code for d in r.diagnostics}
            assert (await s.status()).identity == before and {p: s.pid(p) for p in ALL} == pids

            s.edit("scn/policy_fcfs/unit.py", BROKEN_IMPORT, "")
            good = await s.client.check(await s.change())
            assert not isinstance(good, Failed), good
            r = await s.apply()
            assert r.status == "committed" and r.new_identity == good.value.candidate

    run(main())


@pytest.mark.covers(edges=("call", "derive"), ladder=(1, 3, 5), kinds=("logic", "state"))
def test_7_a_config_value_takes_the_cheapest_rung(tmp_path: Path) -> None:
    """⑦ Changing config values, and who reads them when: read every step with a hook -> taken in
    place; recorded into a graph by value -> the graph is rebuilt next to the old one; the mesh ->
    only the rank processes are rebuilt, the scheduler host keeps running."""

    async def main() -> None:
        async with scenario(tmp_path) as s:
            before = await generations(s, W0)
            r = await s.apply(files=False, config={"scn.chunk": 4})
            await everyone_on(s, r)
            assert rungs(r.plan, SCHED) == {1} and rungs(r.plan, W0) == set()
            await until(lambda: "sched chunk 4" in s.log_of(SCHED))

            r = await s.apply(files=False, config={"scn.max_batch": 16})
            await everyone_on(s, r)
            for p in (W0, W1):
                assert rungs(r.plan, p) == {1, 3}
                assert s.lines(p, "plan switched")[-1] == \
                    "plan switched to {'scn.kernel': 'basic', 'scn.max_batch': 16}"
            assert await generations(s, W0) == before

            sched_pid, w0_pid = s.pid(SCHED), s.pid(W0)
            r = await s.apply(files=False, topology=topology(tmp_path, ("host-a",)))
            assert r.status == "committed", r.diagnostics
            assert rungs(r.plan, W0) == {5} and rungs(r.plan, W1) == {5} and rungs(r.plan, SCHED) == set()
            assert s.pid(SCHED) == sched_pid and s.pid(W0) != w0_pid
            assert s.cp.sup.proc_state(W1) is None
            no_replay_failures(s)

    run(main())


@pytest.mark.covers(edges=("call", "state"), ladder=(2,), kinds=("logic", "state"))
def test_8_new_scheduling_logic_keeps_the_requests_in_flight(tmp_path: Path) -> None:
    """⑧ apply: the scheduling logic's code changed -> reload it (and the policies registered into
    it); the request table is a state unit and stays, so in-flight requests carry on."""

    async def main() -> None:
        async with scenario(tmp_path) as s:
            before = await generations(s, SCHED)
            await asyncio.sleep(0.3)  # let requests make progress
            s.edit("scn/sched/unit.py", "STEP_S = 0.005", "STEP_S = 0.004")
            r = await s.apply()
            await everyone_on(s, r)
            assert rungs(r.plan, SCHED) == {2} and rungs(r.plan, W0) == set()
            after = await generations(s, SCHED)
            assert {u for u in after if after[u] != before[u]} == {"scn.sched", "scn.policy_fcfs"}
            gen, progress = s.lines(SCHED, "sched gen")[-1].split()[1::2]
            assert gen == "gen2" and int(progress) > 0

    run(main())


@pytest.mark.covers(edges=("ctrl",), ladder=(5,), kinds=("state",))
def test_9_a_native_extension_change_restarts_the_processes_that_load_it(tmp_path: Path) -> None:
    """⑨ apply: a declared native extension changed -> it cannot be unloaded, so the worker processes
    that load it are restarted at the new identity (their state units start over); the scheduler
    host changes in place. Out: draining, since no unit here holds requests that a restart cuts."""

    async def main() -> None:
        async with scenario(tmp_path) as s:
            pids = {p: s.pid(p) for p in ALL}
            s.write("scn/native/_ext.so", "placeholder for a compiled extension, version 2\n")
            r = await s.apply()
            await everyone_on(s, r)
            assert rungs(r.plan, W0) == {5} and rungs(r.plan, W1) == {5} and rungs(r.plan, SCHED) == set()
            assert s.pid(SCHED) == pids[SCHED]
            assert s.pid(W0) != pids[W0] and s.pid(W1) != pids[W1]
            assert len(s.lines(W0, "plan recorded")) == 2  # the new process built its graph again
            no_replay_failures(s)

    run(main())


@pytest.mark.covers(ladder=(6,), kinds=("framework",))
def test_a_framework_change_needs_an_engine_restart(tmp_path: Path) -> None:
    """Changing the framework core cannot be done in place: the change fails with
    escalate.restart_engine and nothing moves."""

    async def main() -> None:
        async with scenario(tmp_path) as s:
            pids = {p: s.pid(p) for p in ALL}
            before = (await s.status()).identity
            s.edit("core/change/queue.py", '"""Change queue', '"""The change queue')
            r = await s.apply()
            assert r.status == "failed" and r.plan is not None and r.plan.restart_engine
            assert "escalate.restart_engine" in {d.code for d in r.diagnostics}
            assert (await s.status()).identity == before and {p: s.pid(p) for p in ALL} == pids

    run(main())


@pytest.mark.covers(ladder=(2,), kinds=("logic",))
def test_a_control_plane_unit_is_reloaded_in_the_control_plane(tmp_path: Path) -> None:
    """The control-plane units are units too: changing one reloads it in the control-plane process,
    between two changes, and the next change already uses it."""

    async def main() -> None:
        async with scenario(tmp_path) as s:
            s.edit("core/api/unit.py", '"""``core.api``: the engine', '"""``core.api``: the engine\'s')
            r = await s.apply()
            await everyone_on(s, r)
            assert r.plan is not None and r.plan.control is not None
            assert rungs(r.plan, s.cp.proc) == {2}
            st = await s.status()
            assert {u.unit: u.generation for u in st.control_units}["core.api"] == 2
            s.edit("scn/policy_fcfs/unit.py", "VERSION = 1", "VERSION = 2")
            assert (await s.apply()).status == "committed"

    run(main())
