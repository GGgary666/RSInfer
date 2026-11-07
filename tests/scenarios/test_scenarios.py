"""The nine scenarios of modules.html, on two fake hosts, through the control interface.

Each test exercises what its scenario needs from the framework and the control-plane
units, and leaves out what belongs to the model domain (tokenizer, HTTP, real
admission, real collectives); the docstrings say what is in and what is out.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import pytest

from core.change.types import ChangeResult
from core.common.ids import UnitId
from core.common.result import Failed, Result, fail
from core.supervisor.protocol import ProcIntrospection
from tests.scenarios.harness import ALL, SCHED, W0, W1, Scenario, generations, rungs, scenario, topology
from unit.runtime import DerivedArtifact
from unit.runtime.types import ReloadPlan, ReloadReport


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


@pytest.mark.covers(edges=("ctrl",), ladder=(2,), kinds=("framework",))
def test_a_control_plane_change_that_loses_a_control_service_is_rolled_back(tmp_path: Path) -> None:
    """A new ``core.api`` that loads but no longer provides ``ctrl.api@1`` would leave the engine
    without a control interface; the control plane checks its services before the change commits,
    so the change aborts and the old unit serves again."""

    async def main() -> None:
        async with scenario(tmp_path) as s:
            before = (await s.status()).identity
            s.edit("core/api/unit.py", 'ctx.provide("ctrl.api@1", Api(ctx))', "self.api = Api(ctx)")
            r = await s.apply()
            assert r.status == "aborted", r.diagnostics
            assert "control.unit_missing" in {d.code for d in r.diagnostics}
            st = await s.status()  # the control interface still answers
            assert st.identity == before and not st.recovery_required
            s.edit("core/api/unit.py", "self.api = Api(ctx)", 'ctx.provide("ctrl.api@1", Api(ctx))')
            s.edit("scn/policy_fcfs/unit.py", "VERSION = 1", "VERSION = 2")
            assert (await s.apply()).status == "committed"

    run(main())


@pytest.mark.covers(edges=("ctrl",), kinds=("framework",))
def test_a_failed_rollback_in_the_control_plane_requires_recovery(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """If the old control-plane units cannot be reloaded when a change aborts, the new ones keep
    serving under the old identity: the engine needs recovery and refuses further changes, as when
    a process does not confirm its rollback."""

    async def main() -> None:
        async with scenario(tmp_path) as s:
            before = (await s.status()).identity
            s.edit("core/api/unit.py", "len(coord.queue), rows", "len(coord.queue) + 100, rows")
            reload = s.cp.runtime.reload
            calls = 0

            def reverse_fails(plan: ReloadPlan, *, values: dict[str, Any] | None = None) -> Result[ReloadReport]:
                nonlocal calls
                calls += 1
                if calls == 2:  # the reverse reload of the abort
                    return fail("unit.runtime", "rollback", "injected", code="escalate.restart_process")
                return reload(plan, values=values)

            monkeypatch.setattr(s.cp.runtime, "reload", reverse_fails)
            s.cp.coord.faults.add("after_execute")
            r = await s.apply()
            assert calls == 2 and r.status == "failed"
            assert {"control.rollback_failed", "change.recovery_required"} <= {d.code for d in r.diagnostics}
            st = await s.status()
            assert st.identity == before and st.recovery_required and st.queued == 100  # the new core.api
            refused = await s.apply(files=False, config={"scn.chunk": 4})
            assert refused.status == "failed" and "change.recovery_required" in {d.code for d in refused.diagnostics}

    run(main())


READER_TOML = """[unit]
id = "scn.reader"
kind = "logic"
process = ["worker"]
entry = "scn.reader.unit:Unit"

[reads]
config = ["scn.log_dir", "scn.max_batch"]
"""
READER = """from typing import Any

from ..journal import Journal


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        Journal(config["scn.log_dir"], ctx).write(f"reader built with max_batch {config['scn.max_batch']}")
"""


@pytest.mark.covers(edges=("decl",), ladder=(1, 2, 3), kinds=("logic",))
def test_a_new_unit_is_built_with_the_values_of_its_change(tmp_path: Path) -> None:
    """A unit added together with a new value of a key it reads is not staged during prepare (it
    would be built under the old value) but added inside the reload, under the new value."""

    async def main() -> None:
        async with scenario(tmp_path) as s:
            s.write("scn/reader/__init__.py", "")
            s.write("scn/reader/unit.toml", READER_TOML)
            s.write("scn/reader/unit.py", READER)
            r = await s.apply(config={"scn.max_batch": 16})
            await everyone_on(s, r)
            assert r.plan is not None
            for p in (W0, W1):
                assert rungs(r.plan, p) == {1, 2, 3}
                assert s.lines(p, "reader built") == ["reader built with max_batch 16"]
            assert any("scn.reader is added at the safe point: it reads ['scn.max_batch'], which this change sets" in n
                       for n in r.plan.notes)

    run(main())


def _control_module(s: Scenario, name: str) -> Path:
    """Where the control plane loaded ``name`` from, relative to its code root."""
    root = s.cp.runtime.code_root
    file = getattr(sys.modules[name], "__file__", None)
    assert root is not None and file is not None
    assert Path(file).is_relative_to(root), f"{name} is loaded from {file}, not from the code root {root}"
    return Path(file).relative_to(root)


@pytest.mark.covers(edges=("ctrl",), ladder=(2,), kinds=("logic",))
def test_a_library_of_the_control_plane_units_is_reloaded_from_the_new_snapshot(tmp_path: Path) -> None:
    """``core`` was imported where the control plane started, before any snapshot; the control
    units and their library ``core.sourcetree`` come from snapshots through ``core.__path__``.
    Changing the library must load it from the new snapshot, and the units must use it."""

    async def main() -> None:
        async with scenario(tmp_path) as s:
            s.edit("core/sourcetree.py", "    return Tree(root, units, modules, owner, errors)\n",
                   '    if (root / "REJECT").exists():\n'
                   '        errors.append(diag("core.manifest", "manifest", "rejected by the new library",\n'
                   '                           code="manifest.rejected"))\n'
                   "    return Tree(root, units, modules, owner, errors)\n")
            r = await s.apply()
            await everyone_on(s, r)
            assert r.plan is not None and r.plan.control is not None and r.plan.control.reload is not None
            assert "core.sourcetree" in r.plan.control.reload.changed_modules
            assert _control_module(s, "core.sourcetree") == Path("core/sourcetree.py")
            checks = sys.modules["core.manifest.checks"]
            assert getattr(checks, "read_tree").__module__ == "core.sourcetree"
            assert Path(getattr(checks, "read_tree").__code__.co_filename).is_relative_to(str(s.cp.runtime.code_root))
            s.write("REJECT", "")
            checked = await s.client.check(await s.change())
            assert isinstance(checked, Failed) and checked.has_code("manifest.rejected")

    run(main())


PROBE_TOML = """[unit]
id = "core.probe"
kind = "logic"
process = ["control"]
entry = "core.probe.unit:Unit"

[provides]
services = ["ctrl.probe@1"]

[config]
"ctrl.probe_level" = { type = "int", default = 1, binding = "startup" }
"ctrl.probe_mode" = { type = "str", default = "plain", binding = "startup" }

[reads]
config = ["scn.chunk", "scn.log_dir"]
"""
PROBE = """from typing import Any


class Probe:
    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values

    def values(self) -> dict[str, Any]:
        return dict(self._values)


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        ctx.provide("ctrl.probe@1", Probe({
            "ctrl.probe_level": config["ctrl.probe_level"], "ctrl.probe_mode": config["ctrl.probe_mode"],
            "scn.chunk": config["scn.chunk"], "scn.log_dir": config["scn.log_dir"],
        }))
"""
PROBE_FILES = {"core/probe/__init__.py": "", "core/probe/unit.toml": PROBE_TOML, "core/probe/unit.py": PROBE}


def _probe_values(s: Scenario) -> dict[str, Any]:
    probe = s.cp.runtime.lookup("ctrl.probe@1")
    assert probe is not None
    return probe.values()


@pytest.mark.covers(edges=("ctrl", "decl"), ladder=(0, 2), kinds=("logic",))
def test_a_new_control_plane_unit_under_core_is_loaded_from_the_snapshot(tmp_path: Path) -> None:
    """A new control-plane unit in ``core/`` exists only in the new snapshot; ``core`` itself was
    imported from elsewhere, so only its ``__path__`` can find it. Its own keys are declared with
    it (one set by the change, one left at its default), and the keys it reads from another role
    are declared by the reload, before it is built."""

    async def main() -> None:
        async with scenario(tmp_path) as s:
            for rel, text in PROBE_FILES.items():
                s.write(rel, text)
            r = await s.apply(config={"ctrl.probe_level": 5})
            await everyone_on(s, r)
            assert _probe_values(s) == {"ctrl.probe_level": 5, "ctrl.probe_mode": "plain", "scn.chunk": 1,
                                        "scn.log_dir": str(tmp_path / "logs")}
            assert _control_module(s, "core.probe.unit") == Path("core/probe/unit.py")
            assert "core.probe" in {u.unit for u in (await s.status()).control_units}

    run(main())


@pytest.mark.covers(edges=("ctrl", "decl"), kinds=("framework",))
def test_the_control_plane_starts_units_that_read_keys_of_another_role(tmp_path: Path) -> None:
    """At startup a control-plane unit gets the keys it owns and the keys it reads from units of
    other roles, with the values given (``scn.log_dir``) or the owners' defaults (``scn.chunk``)."""

    async def main() -> None:
        async with scenario(tmp_path, files=PROBE_FILES) as s:
            assert _probe_values(s) == {"ctrl.probe_level": 1, "ctrl.probe_mode": "plain", "scn.chunk": 1,
                                        "scn.log_dir": str(tmp_path / "logs")}

    run(main())


CHUNK_READER_TOML = """[unit]
id = "scn.chunk_reader"
kind = "logic"
process = ["worker"]
entry = "scn.chunk_reader.unit:Unit"

[reads]
config = ["scn.chunk", "scn.log_dir", "scn.rebalance"]
"""
CHUNK_READER = """from typing import Any

from ..journal import Journal


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        Journal(config["scn.log_dir"], ctx).write(
            f"chunk reader built with chunk {config['scn.chunk']} rebalance {config['scn.rebalance']}")
"""


@pytest.mark.covers(edges=("decl",), ladder=(1, 2), kinds=("logic",))
def test_a_new_unit_reading_keys_of_another_role_is_built_after_they_are_declared(tmp_path: Path) -> None:
    """A new worker unit reads keys the scheduler's units own: the workers have never declared
    them, so it cannot be staged during prepare; the reload declares them (one set by the change,
    one at its owner's default), then builds it."""

    async def main() -> None:
        async with scenario(tmp_path) as s:
            s.write("scn/chunk_reader/__init__.py", "")
            s.write("scn/chunk_reader/unit.toml", CHUNK_READER_TOML)
            s.write("scn/chunk_reader/unit.py", CHUNK_READER)
            r = await s.apply(config={"scn.chunk": 4})
            await everyone_on(s, r)
            for p in (W0, W1):
                assert rungs(r.plan, p) == {2}
                assert s.lines(p, "chunk reader built") == ["chunk reader built with chunk 4 rebalance 0"]
            assert rungs(r.plan, SCHED) == {1}

    run(main())


@pytest.mark.covers(edges=("ctrl",), ladder=(2,), kinds=("logic",))
def test_a_control_unit_package_init_is_updated_and_rolled_back(tmp_path: Path) -> None:
    """A changed ``core/api/__init__.py`` reloads ``core.api`` from the new snapshot; when a later
    change to it aborts, the package, the unit and ``core.__path__`` are the ones from before."""

    async def main() -> None:
        async with scenario(tmp_path) as s:
            s.write("core/api/__init__.py", '"""``core.api``."""\n\nMARK = 2\n')
            r = await s.apply()
            await everyone_on(s, r)
            assert _control_module(s, "core.api") == Path("core/api/__init__.py")
            assert getattr(sys.modules["core.api"], "MARK") == 2
            committed_root = s.cp.runtime.code_root
            core_path = list(getattr(sys.modules["core"], "__path__"))
            assert core_path[0] == os.path.join(str(committed_root), "core")

            s.write("core/api/__init__.py", '"""``core.api``."""\n\nMARK = 3\n')
            s.cp.coord.faults.add("after_execute")
            r = await s.apply()
            assert r.status == "aborted", r.diagnostics
            assert s.cp.runtime.code_root == committed_root
            assert list(getattr(sys.modules["core"], "__path__")) == core_path
            assert getattr(sys.modules["core"], "api") is sys.modules["core.api"]
            assert getattr(sys.modules["core.api"], "MARK") == 2
            assert _control_module(s, "core.api") == Path("core/api/__init__.py")

            r = await s.apply()
            await everyone_on(s, r)
            assert getattr(sys.modules["core.api"], "MARK") == 3

    run(main())


@pytest.mark.covers(edges=("ctrl",), kinds=("framework",))
def test_a_syntax_error_is_a_static_check_failure_and_the_change_ends_failed(tmp_path: Path) -> None:
    """⑥ for code that does not parse: ``check`` and ``apply`` both report it as a static check
    (``manifest.syntax_error``, with the file and line), and the change record ends as failed."""

    async def main() -> None:
        async with scenario(tmp_path) as s:
            before = (await s.status()).identity
            s.edit("scn/policy_fcfs/unit.py", "VERSION = 1", "VERSION = (1")
            checked = await s.client.check(await s.change())
            assert isinstance(checked, Failed) and checked.has_code("manifest.syntax_error"), checked
            assert any("scn/policy_fcfs/unit.py, line" in d.reason for d in checked.diagnostics)
            r = await s.apply()
            assert r.status == "failed" and r.change_id > 0
            assert "manifest.syntax_error" in {d.code for d in r.diagnostics}
            row = s.cp.log.db.execute("SELECT status, finished_at FROM changes WHERE id = ?",
                                      (r.change_id,)).fetchone()
            assert row[0] == "failed" and row[1] is not None
            assert (await s.status()).identity == before
            s.edit("scn/policy_fcfs/unit.py", "VERSION = (1", "VERSION = 2")
            assert (await s.apply()).status == "committed"

    run(main())


LOGS_WITH_EXTRA = """[unit]
id = "scn.logs"
kind = "logic"
process = ["sched", "worker"]
entry = "scn.logs.unit:Unit"

[provides]
services = ["scn.extra@1"]

[config]
"scn.log_dir" = { type = "str", default = "", binding = "startup" }
"""
LOGS_WITH_EXTRA_CODE = """from typing import Any


class Extra:
    def value(self) -> str:
        return "extra from the reloaded scn.logs"


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        self.directory = config["scn.log_dir"]
        ctx.provide("scn.extra@1", Extra())
"""
EXTRA_USER_TOML = """[unit]
id = "scn.extra_user"
kind = "logic"
process = ["worker"]
entry = "scn.extra_user.unit:Unit"

[requires]
{dependency} = ["scn.extra@1"]

[reads]
config = ["scn.log_dir"]
"""
EXTRA_USER = """from typing import Any

from ..journal import Journal


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        extra = ctx.{fetch}("scn.extra@1")
        self.extra = extra
        Journal(config["scn.log_dir"], ctx).write(f"extra user got {{extra.value() if extra is not None else None}}")
"""


@pytest.mark.parametrize("dependency", ["services", "optional"])
@pytest.mark.covers(edges=("call",), ladder=(2,), kinds=("logic",))
def test_a_new_unit_using_a_service_a_running_unit_gains_waits_for_its_reload(tmp_path: Path,
                                                                               dependency: str) -> None:
    """``scn.logs`` runs without ``scn.extra@1``; the change adds that service to it and, at the same
    time, a new unit that uses it. Staged during prepare the user would find no provider: a required
    service would leave it pending and abort the change, an optional one would be ``None`` for good.
    It is added in the reload, after ``scn.logs`` was rebuilt, and gets the new service."""

    async def main() -> None:
        async with scenario(tmp_path) as s:
            s.write("scn/logs/unit.toml", LOGS_WITH_EXTRA)
            s.write("scn/logs/unit.py", LOGS_WITH_EXTRA_CODE)
            s.write("scn/extra_user/__init__.py", "")
            s.write("scn/extra_user/unit.toml", EXTRA_USER_TOML.format(dependency=dependency))
            s.write("scn/extra_user/unit.py", EXTRA_USER.format(fetch="use" if dependency == "services" else "get"))
            r = await s.apply()
            await everyone_on(s, r)
            assert r.plan is not None
            for p in (W0, W1):
                assert s.lines(p, "extra user got") == ["extra user got extra from the reloaded scn.logs"]
                pa = next(pa for pa in r.plan.procs if pa.proc == p)
                assert pa.stage == [] and pa.reload is not None
                assert [u.id for u in pa.reload.units_to_add] == ["scn.extra_user"]

    run(main())
