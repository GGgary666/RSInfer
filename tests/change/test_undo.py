"""Abort puts back everything a change did, and nothing the engine still runs is deleted."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.change.coordinator import PlanInput
from core.change.state import apply_changeset
from core.change.types import ActionPlan, ChangeSet, ProcActions, UnitAction
from core.common.ids import UnitId
from tests.change.engine import ALL, SCHED, UNITS, W0, W1, Engine, engine, reload_everywhere, run
from tests.cluster.harness import SPECS, TOY, VERSIONED_V2, role, topology
from tests.support import spec

KERNEL = """
from toyc.logutil import log


class Kernel:
    def __init__(self, ctx, config):
        pass
"""
from unit.runtime import ReloadPlan


def everyone_idle(inp: PlanInput) -> ActionPlan:
    return ActionPlan(procs=[ProcActions(p, leads_epochs=p == SCHED) for p in ALL])


def tunable_loads(e: Engine, p: object) -> int:
    return sum(line.startswith("tunable loaded") for line in e.c.log_of(p))  # type: ignore[arg-type]


def test_snapshots_the_processes_still_load_code_from_are_not_pruned(tmp_path: Path) -> None:
    """An in-place change leaves unchanged modules loaded from the snapshot the process started from."""

    async def main() -> None:
        async with engine(tmp_path, agent={"nodeagent.snapshot_retention": 1}) as e:
            start = e.coord.identity
            e.planner.script = reload_everywhere()
            hot = await e.coord.submit(e.code_change())
            assert hot.status == "committed" and hot.new_identity is not None
            current = hot.new_identity
            e.planner.script = everyone_idle
            for i in range(3):  # each materializes another snapshot on both hosts
                e.coord.faults.add("after_prepare")
                h = e.c.blobs.put(f"X = {i}\n".encode())
                r = await e.coord.submit(ChangeSet(base=current, files={f"engine/toyc/extra{i}.py": h}))
                assert r.status == "aborted"
            for host in ("host-a", "host-b"):
                root = Path(e.c.agents[host].root)
                assert (root / current).is_dir() and (root / start).is_dir(), host

            e.planner.script = reload_everywhere()
            v3 = e.c.blobs.put(VERSIONED_V2.replace("VERSION = 2", "VERSION = 3").encode())
            r = await e.coord.submit(ChangeSet(base=current, files={"engine/toyc/versioned.py": v3}))
            assert r.status == "committed", r.diagnostics
            assert all(e.versioned(p) == "versioned v3" for p in ALL)

    run(main())


def test_abort_after_execute_leaves_nothing_loaded_from_the_candidate(tmp_path: Path) -> None:
    """A module the candidate imports for the first time goes with the abort; the candidate
    snapshot can then be pruned and the next update still works."""

    async def main() -> None:
        async with engine(tmp_path, agent={"nodeagent.snapshot_retention": 1}) as e:
            e.planner.script = everyone_idle  # extra.py exists from here on, but nothing imports it
            added = await e.coord.submit(ChangeSet(base=e.coord.identity,
                                                   files={"engine/toyc/extra.py": e.c.blobs.put(b"X = 1\n")}))
            assert added.status == "committed", added.diagnostics
            base = e.coord.identity
            uses_extra = VERSIONED_V2.replace("VERSION = 2", "from toyc import extra\n\nVERSION = 2")
            e.planner.script = reload_everywhere()
            e.coord.faults.add("after_execute")
            cs = ChangeSet(base=base, files={"engine/toyc/versioned.py": e.c.blobs.put(uses_extra.encode())})
            candidate = e.coord.identity_fn(apply_changeset(e.coord.manifest, cs))
            assert (await e.coord.submit(cs)).status == "aborted"
            e.planner.script = everyone_idle
            for i in range(3):  # each materializes another snapshot; retention keeps one spare
                e.coord.faults.add("after_prepare")
                h = e.c.blobs.put(f"Y = {i}\n".encode())
                assert (await e.coord.submit(ChangeSet(base=base, files={f"engine/toyc/other{i}.py": h}))
                        ).status == "aborted"
            for host in ("host-a", "host-b"):
                root = Path(e.c.agents[host].root)
                assert not (root / candidate).exists() and (root / base).is_dir(), host

            e.planner.script = reload_everywhere()
            v3 = e.c.blobs.put(VERSIONED_V2.replace("VERSION = 2", "VERSION = 3").encode())
            r = await e.coord.submit(ChangeSet(base=base, files={"engine/toyc/versioned.py": v3}))
            assert r.status == "committed", r.diagnostics
            assert all(e.versioned(p) == "versioned v3" for p in ALL)

    run(main())


def test_units_staged_by_a_committed_change_are_its_own(tmp_path: Path) -> None:
    """A change that only stages a unit commits it for good: a later aborted change keeps it."""

    async def main() -> None:
        async with engine(tmp_path) as e:
            kernel = spec("toyc.kernel", "toyc.kernel:Kernel", process=["worker"])
            e.planner.script = lambda inp: ActionPlan(procs=[ProcActions(W1, stage=[kernel])])
            r = await e.coord.submit(ChangeSet(base=e.coord.identity,
                                               files={"engine/toyc/kernel.py": e.c.blobs.put(KERNEL.encode())}))
            assert r.status == "committed", r.diagnostics
            assert "toyc.kernel" in {u.unit for u in (await e.status(W1)).units}

            e.planner.script = reload_everywhere()
            e.coord.faults.add("after_execute")
            assert (await e.coord.submit(e.code_change())).status == "aborted"
            assert "toyc.kernel" in {u.unit for u in (await e.status(W1)).units}
            assert (await e.coord.submit(e.code_change())).status == "committed"
            assert "toyc.kernel" in {u.unit for u in (await e.status(W1)).units}

    run(main())


TOPOLOGY_ABORTS = ["after_prepare", "after_pause", "participant_execute", "after_execute"]


@pytest.mark.parametrize("where", TOPOLOGY_ABORTS)
def test_abort_restores_the_topology_and_the_processes_the_change_stopped(tmp_path: Path, where: str) -> None:
    async def main() -> None:
        async with engine(tmp_path) as e:
            old = e.coord.identity
            before = e.c.sup.topology
            smaller = topology(tmp_path, ["host-a", "host-b"], ["host-a"],
                               [role("sched", UNITS, hosts=["host-a"]), role("worker", UNITS)])
            fail_w0 = [UnitAction(UnitId("toyc.tunable"), "toy.op", {"fail_at": "execute"})] \
                if where == "participant_execute" else []
            e.planner.script = lambda inp: ActionPlan(
                procs=[ProcActions(SCHED, leads_epochs=True), ProcActions(W0, actions=fail_w0)], stop=[W1],
                topology=smaller)
            pid = e.c.sup.local_pid(W1)
            if where != "participant_execute":
                e.coord.faults.add(where)
            r = await e.coord.submit(ChangeSet(base=old, topology=smaller))
            assert r.status == "aborted"
            assert e.c.sup.topology == before
            assert e.c.sup.proc_state(W1) == "RUNNING" and (await e.status(W1)).identity == old
            stopped = where == "after_execute"  # only then had the change stopped W1
            assert (e.c.sup.local_pid(W1) != pid) == stopped
            assert ("change.restarted_after_abort" in {d.code for d in r.diagnostics}) == stopped

            e.planner.script = lambda inp: ActionPlan(
                procs=[ProcActions(SCHED, leads_epochs=True), ProcActions(W0)], stop=[W1], topology=smaller)
            r = await e.coord.submit(ChangeSet(base=old, topology=smaller))
            assert r.status == "committed", r.diagnostics
            assert e.c.sup.proc_state(W1) is None and e.c.sup.topology == smaller

    run(main())


def test_abort_after_removing_a_unit_brings_back_its_config(tmp_path: Path) -> None:
    async def main() -> None:
        async with engine(tmp_path) as e:
            remove = ReloadPlan(units_to_remove=[UnitId("toyc.tunable")])
            e.planner.script = lambda inp: ActionPlan(procs=[ProcActions(W1, reload=remove)])
            loads = tunable_loads(e, W1)
            e.coord.faults.add("after_execute")
            r = await e.coord.submit(ChangeSet(base=e.coord.identity))
            assert r.status == "aborted"
            # toyc.log_dir is back to its value, not the default "", or the unit could not log
            assert tunable_loads(e, W1) == loads + 1
            assert "toyc.tunable" in {u.unit for u in (await e.status(W1)).units}

    run(main())


def test_abort_of_a_config_change_with_a_reload_rebuilds_units_with_the_old_values(tmp_path: Path) -> None:
    async def main() -> None:
        async with engine(tmp_path) as e:
            reload = ReloadPlan(units_to_reload=[SPECS["toyc.tunable"]])
            e.planner.script = lambda inp: ActionPlan(procs=[
                ProcActions(W1, config={"toyc.flavor": "spicy"}, reload=reload)])
            e.coord.faults.add("after_execute")
            r = await e.coord.submit(ChangeSet(base=e.coord.identity, config={"toyc.flavor": "spicy"}))
            assert r.status == "aborted"
            loads = [line for line in e.c.log_of(W1) if line.startswith("tunable loaded")]
            assert loads[-2].endswith("flavor spicy") and loads[-1].endswith("flavor plain")  # the instance

            e.planner.script = lambda inp: ActionPlan(procs=[ProcActions(W1, reload=reload)])
            assert (await e.coord.submit(ChangeSet(base=e.coord.identity))).status == "committed"
            assert e.c.log_of(W1)[-1].endswith("flavor plain")  # the config table

    run(main())


def test_an_action_of_a_reloaded_unit_is_prepared_on_the_new_instance(tmp_path: Path) -> None:
    async def main() -> None:
        async with engine(tmp_path) as e:
            reload = ReloadPlan(changed_modules=["toyc.tunable"], units_to_reload=[SPECS["toyc.tunable"]])
            op = UnitAction(UnitId("toyc.tunable"), "toy.op", {"needs_prepare": True})
            e.planner.script = lambda inp: ActionPlan(procs=[
                ProcActions(p, reload=reload, actions=[op], leads_epochs=p == SCHED) for p in ALL])
            src = TOY["toyc/tunable.py"] + "\n# v2\n"
            r = await e.coord.submit(ChangeSet(base=e.coord.identity,
                                               files={"engine/toyc/tunable.py": e.c.blobs.put(src.encode())}))
            assert r.status == "committed", r.diagnostics
            for p in ALL:
                assert e.ops(p) == ["op prepare", "op execute", "op commit"]
            late = [d for d in r.diagnostics if d.code == "change.prepare_one_shot" and "reloaded" in d.reason]
            assert sorted(d.subject for d in late) == sorted(str(p) for p in ALL)

    run(main())


SERVICE = """
from toyc.tunable import Op


class Provider:
    def __init__(self, ctx, config):
        ctx.provide("toy.svc@1", self)


class User:
    def __init__(self, ctx, config):
        self.svc = ctx.use("toy.svc@1")
        ctx.on_action("toy.op", Op(ctx, config))


class Addon:
    def __init__(self, ctx, config):
        ctx.on_action("toy.op", Op(ctx, config))
"""
PROVIDER = spec("toyc.provider", "toyc.svc:Provider", provides=["toy.svc@1"], process=["worker"])
USER = spec("toyc.svc_user", "toyc.svc:User", requires=["toy.svc@1"], reads=["toyc.log_dir"], process=["worker"])
ADDON = spec("toyc.addon", "toyc.svc:Addon", reads=["toyc.log_dir"], process=["worker"])
NEEDS_PREPARE = {"needs_prepare": True}


def test_users_of_a_reloaded_provider_prepare_on_their_new_instance(tmp_path: Path) -> None:
    """Reloading a provider rebuilds its users too, so their actions wait for the safe point."""

    async def main() -> None:
        async with engine(tmp_path) as e:
            e.planner.script = lambda inp: ActionPlan(procs=[ProcActions(W1, stage=[PROVIDER, USER])])
            files: dict[str, str | None] = {"engine/toyc/svc.py": e.c.blobs.put(SERVICE.encode())}
            assert (await e.coord.submit(ChangeSet(base=e.coord.identity, files=files))).status == "committed"
            ops_before = len(e.ops(W1))
            e.planner.script = lambda inp: ActionPlan(procs=[ProcActions(
                W1, reload=ReloadPlan(units_to_reload=[PROVIDER]),
                actions=[UnitAction(UnitId(USER.id), "toy.op", NEEDS_PREPARE)])])
            r = await e.coord.submit(ChangeSet(base=e.coord.identity))
            assert r.status == "committed", r.diagnostics
            assert e.ops(W1)[ops_before:] == ["op prepare", "op execute", "op commit"]

    run(main())


def test_a_unit_added_at_the_safe_point_prepares_there(tmp_path: Path) -> None:
    async def main() -> None:
        async with engine(tmp_path) as e:
            e.planner.script = lambda inp: ActionPlan(procs=[ProcActions(
                W1, reload=ReloadPlan(units_to_add=[ADDON]),
                actions=[UnitAction(UnitId(ADDON.id), "toy.op", NEEDS_PREPARE)])])
            files: dict[str, str | None] = {"engine/toyc/svc.py": e.c.blobs.put(SERVICE.encode())}
            r = await e.coord.submit(ChangeSet(base=e.coord.identity, files=files))
            assert r.status == "committed", r.diagnostics
            assert e.ops(W1) == ["op prepare", "op execute", "op commit"]
            assert ADDON.id in {u.unit for u in (await e.status(W1)).units}

    run(main())


FLAVORED = """
from toyc.logutil import log


class Flavored:
    def __init__(self, ctx, config):
        log(config, ctx, f"flavored {config['toyc.addon_flavor']}")
"""
FLAVORED_SPEC = spec("toyc.flavored", "toyc.flavored:Flavored", reads=["toyc.log_dir"], process=["worker"],
                     config={"toyc.addon_flavor": {"type": "str", "default": "plain", "binding": "startup"}})


@pytest.mark.parametrize("how", ["staged", "added"])
def test_a_new_unit_starts_with_the_config_the_change_sets_for_it(tmp_path: Path, how: str) -> None:
    async def main() -> None:
        async with engine(tmp_path) as e:
            config: dict[str, object] = {"toyc.addon_flavor": "spicy"}
            pa = ProcActions(W1, stage=[FLAVORED_SPEC], config=config) if how == "staged" else \
                ProcActions(W1, reload=ReloadPlan(units_to_add=[FLAVORED_SPEC]), config=config)
            e.planner.script = lambda inp: ActionPlan(procs=[pa])
            files: dict[str, str | None] = {"engine/toyc/flavored.py": e.c.blobs.put(FLAVORED.encode())}
            r = await e.coord.submit(ChangeSet(base=e.coord.identity, files=files, config=config))
            assert r.status == "committed", r.diagnostics
            assert [line for line in e.c.log_of(W1) if line.startswith("flavored")] == ["flavored spicy"]

    run(main())


def test_an_action_whose_prepare_raised_is_still_aborted(tmp_path: Path) -> None:
    async def main() -> None:
        async with engine(tmp_path) as e:
            e.planner.script = lambda inp: ActionPlan(procs=[
                ProcActions(p, actions=[UnitAction(UnitId("toyc.tunable"), "toy.op",
                                                   {"fail_at": "prepare"} if p == W1 else {})],
                            leads_epochs=p == SCHED) for p in ALL])
            r = await e.coord.submit(ChangeSet(base=e.coord.identity))
            assert r.status == "aborted"
            for p in ALL:
                assert e.ops(p) == ["op prepare", "op abort"], p  # once each, the failed one too

    run(main())
