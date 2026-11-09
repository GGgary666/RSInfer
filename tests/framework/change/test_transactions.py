"""change.md §7 with real processes on two fake hosts: 2PC, rollback, queueing, drain, dry-run."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from core.change.coordinator import PlanInput
from core.change.types import EMPTY, ActionPlan, ChangeMeta, ChangeResult, ChangeSet, ProcActions, UnitAction
from core.common.ids import Identity, ProcId, UnitId
from core.supervisor import desired_procs
from tests.framework.change.engine import ALL, SCHED, UNITS, W0, W1, Script, engine, reload_everywhere, run
from tests.framework.cluster.harness import process_plan
from unit.runtime import ReloadPlan


def test_startup_is_a_change(tmp_path: Path) -> None:
    async def main() -> None:
        async with engine(tmp_path) as e:
            assert e.coord.identity != EMPTY
            for p in ALL:
                assert (await e.status(p)).identity == e.coord.identity
                assert e.versioned(p) == "versioned v1"
            rec = e.coord.log.changes()[0]
            assert (rec.source, rec.status, rec.new_identity) == ("startup", "committed", e.coord.identity)

    run(main())


def test_code_change_commits_on_every_participant(tmp_path: Path) -> None:
    async def main() -> None:
        async with engine(tmp_path) as e:
            old = e.coord.identity
            e.planner.script = reload_everywhere()
            r = await e.coord.submit(e.code_change())
            assert r.status == "committed", r.diagnostics
            assert r.old_identity == old and r.new_identity == e.coord.identity != old
            for p in ALL:
                st = await e.status(p)
                assert st.identity == r.new_identity and not st.paused
                assert e.versioned(p) == "versioned v2"
                assert e.ops(p) == ["op prepare", "op execute", "op commit"]
                assert "toyc.versioned" in r.reports[str(p)].reimported
                assert await e.epochs_advance(p)
            assert {"prepare", "pause", "execute", "commit"} <= set(r.timings)
            rec = e.coord.log.changes()[-1]
            assert (rec.status, rec.base, rec.new_identity) == ("committed", old, r.new_identity)
            assert list(e.coord.log.diff(old, r.new_identity or old)) == ["engine/toyc/versioned.py"]

    run(main())


FAILURES = [
    ("participant-prepare", {W1: "prepare"}, None),
    ("participant-execute", {W1: "execute"}, None),
    ("leader-execute", {SCHED: "execute"}, None),
    ("coordinator-after-prepare", {}, "after_prepare"),
    ("coordinator-after-pause", {}, "after_pause"),
    ("coordinator-after-execute", {}, "after_execute"),
]


@pytest.mark.parametrize(("name", "fail", "fault"), FAILURES, ids=[f[0] for f in FAILURES])
def test_any_failure_rolls_back_every_participant(tmp_path: Path, name: str, fail: dict[ProcId, str],
                                                  fault: str | None) -> None:
    async def main() -> None:
        async with engine(tmp_path) as e:
            old = e.coord.identity
            e.planner.script = reload_everywhere(fail)
            if fault:
                e.coord.faults.add(fault)
            r = await e.coord.submit(e.code_change())
            assert r.status == "aborted", name
            assert e.coord.identity == old
            for p in ALL:
                st = await e.status(p)
                assert st.identity == old and not st.paused, p
                assert e.versioned(p) == "versioned v1", p
                assert await e.epochs_advance(p), p
                assert e.ops(p)[0] == "op prepare" and e.ops(p)[-1] == "op abort", p
                assert e.ops(p).count("op abort") == 1, (p, e.ops(p))  # once, even after execute
            # the system is healthy afterwards
            e.planner.script = reload_everywhere()
            assert (await e.coord.submit(e.code_change())).status == "committed"

    run(main())


def test_state_only_internal_change_keeps_the_identity(tmp_path: Path) -> None:
    async def main() -> None:
        async with engine(tmp_path) as e:
            old = e.coord.identity
            e.planner.script = lambda inp: ActionPlan(procs=[
                ProcActions(p, actions=[UnitAction(UnitId("toyc.tunable"), "toy.op")]) for p in (W0, W1)])
            # internal changes do not check their base; the result names the running identity
            r = await e.coord.submit(ChangeSet(base=Identity("whatever"), source="internal", computed_under=old,
                                               state_only=True))
            assert r.status == "committed" and r.new_identity == old and r.state_version == 1
            assert r.old_identity == old
            assert (await e.status(W1)).identity == old
            e.planner.script = reload_everywhere()
            r2 = await e.coord.submit(e.code_change())  # its base is still valid
            assert r2.status == "committed" and r2.state_version == 1

    run(main())


def test_queue_order_and_stale_internal_changes(tmp_path: Path) -> None:
    async def main() -> None:
        async with engine(tmp_path) as e:
            old = e.coord.identity
            gate = asyncio.Event()
            order: list[str] = []
            e.coord.on_result.append(lambda r: order.append(r.status + ":" + (
                e.coord.log.changeset(r.change_id).meta.intent if r.change_id > 0 else "")))  # type: ignore[union-attr]
            plain = reload_everywhere()

            async def blocked(inp: PlanInput) -> ActionPlan:
                if inp.cs.meta.intent == "A":
                    await gate.wait()
                return plain(inp) if inp.cs.files else ActionPlan()

            e.planner.script = blocked
            a = e.coord.enqueue(ChangeSet(base=old, files=e.code_change().files, meta=ChangeMeta(intent="A")))
            await asyncio.sleep(0.2)
            assert e.coord.busy
            futs = [
                e.coord.enqueue(ChangeSet(base=old, meta=ChangeMeta(intent="B-apply"))),
                e.coord.enqueue(ChangeSet(base=old, source="internal", computed_under=old,
                                          meta=ChangeMeta(intent="stale-internal"))),
                e.coord.enqueue(ChangeSet(base=old, source="operator",
                                          meta=ChangeMeta(intent="F-fault", handles_fault=True))),
            ]
            gate.set()
            results: list[ChangeResult] = [await a, *[await f for f in futs]]
            assert results[0].status == "committed"
            assert results[2].status == "dropped"
            assert results[1].status == "failed"  # an apply on an outdated base is rejected, never dropped
            assert results[1].diagnostics[0].code == "change.stale_base"
            # fault handling first, then the (stale, dropped) internal change, then the apply
            assert order == ["committed:A", "failed:F-fault", "dropped:stale-internal", "failed:B-apply"]
            assert e.coord.log.events("dropped_stale_internal")

    run(main())


def test_drain_waits_then_the_unit_settles_the_rest_at_the_deadline(tmp_path: Path) -> None:
    """The drain waits until the unit says it is done, or until the deadline; then the unit settles
    what is left by its own policy (this toy ends them) and the result says what it did."""
    async def main() -> None:
        async with engine(tmp_path) as e:
            def drain(args: dict[str, object], timeout: float) -> Script:
                return lambda inp: ActionPlan(procs=[ProcActions(SCHED, leads_epochs=True, drain=[
                    UnitAction(UnitId("toyc.tunable"), "toy.op", args)])], drain_timeout_s=timeout)

            e.planner.script = drain({"ticks": 3}, 5.0)
            r = await e.coord.submit(ChangeSet(base=e.coord.identity))
            assert r.status == "committed" and r.drain_outcomes == {} and "drain" in r.timings
            e.planner.script = drain({"never": True}, 0.3)
            r = await e.coord.submit(ChangeSet(base=e.coord.identity))
            assert r.status == "committed" and r.drain_outcomes == {"aborted": 3}
            assert "drain timeout" in e.c.log_of(SCHED)

    run(main())


def test_a_drain_hears_how_the_change_ended_also_when_it_refused_to_start(tmp_path: Path) -> None:
    """A drain gets ``commit()`` or ``abort()`` (state-changes.md §5.2), so the unit that stopped
    taking requests for the change takes them again (doc/specs/empty-apply.md). A drain that
    refuses fails the change with its reason."""
    async def main() -> None:
        async with engine(tmp_path) as e:
            def drain(args: dict[str, object]) -> Script:
                return lambda inp: ActionPlan(procs=[ProcActions(SCHED, leads_epochs=True, drain=[
                    UnitAction(UnitId("toyc.tunable"), "toy.op", args)])], drain_timeout_s=5.0)

            e.planner.script = drain({"ticks": 1})
            r = await e.coord.submit(ChangeSet(base=e.coord.identity))
            assert r.status == "committed"
            log = e.c.log_of(SCHED)
            assert log[-2:] == ["drain start", "op commit"], log
            e.planner.script = drain({"refuse": True})
            before = e.coord.identity
            r = await e.coord.submit(ChangeSet(base=e.coord.identity))
            assert r.status != "committed" and e.coord.identity == before
            assert any("the toy holds requests" in d.reason for d in r.diagnostics), r.diagnostics
            log = e.c.log_of(SCHED)
            assert log[-2:] == ["drain start", "op abort"], log

    run(main())


def test_dry_run_returns_the_plan_that_runs(tmp_path: Path) -> None:
    async def main() -> None:
        async with engine(tmp_path) as e:
            old = e.coord.identity
            e.planner.script = reload_everywhere()
            dry = await e.coord.submit(e.code_change(), dry_run=True)
            assert dry.status == "planned" and e.coord.identity == old
            assert all(e.versioned(p) == "versioned v1" for p in ALL)
            real = await e.coord.submit(e.code_change())
            assert real.status == "committed" and real.plan == dry.plan

    run(main())


def test_restart_replaces_only_that_process(tmp_path: Path) -> None:
    async def main() -> None:
        async with engine(tmp_path) as e:
            pids = {p: e.c.sup.local_pid(p) for p in ALL}
            d = desired_procs(e.topo)[W1]

            def script(inp: PlanInput) -> ActionPlan:
                inplace = reload_everywhere()(inp).procs[:2]
                fresh = process_plan(d, Identity(""), {"toyc.log_dir": str(e.c.logs)})
                return ActionPlan(procs=[*inplace, ProcActions(W1, restart=fresh)])

            e.planner.script = script
            r = await e.coord.submit(e.code_change())
            assert r.status == "committed", r.diagnostics
            assert r.restarted == [W1]
            assert e.c.sup.local_pid(W1) != pids[W1]
            assert e.c.sup.local_pid(SCHED) == pids[SCHED] and e.c.sup.local_pid(W0) == pids[W0]
            for p in ALL:
                assert (await e.status(p)).identity == r.new_identity
            assert e.versioned(W1) == "versioned v2"

    run(main())


def test_config_value_change_is_taken_in_place(tmp_path: Path) -> None:
    async def main() -> None:
        async with engine(tmp_path) as e:
            gens = {p: {u.unit: u.generation for u in (await e.status(p)).units} for p in ALL}
            e.planner.script = lambda inp: ActionPlan(procs=[
                ProcActions(p, config={"toyc.chunk": 4096}, leads_epochs=p == SCHED) for p in ALL])
            r = await e.coord.submit(ChangeSet(base=e.coord.identity, config={"toyc.chunk": 4096}, source="config"))
            assert r.status == "committed" and r.new_identity != r.old_identity
            for p in ALL:
                assert "chunk 4096" in e.c.log_of(p)
                assert {u.unit: u.generation for u in (await e.status(p)).units} == gens[p]

    run(main())


def test_status_tells_a_process_that_runs_no_unit_from_one_that_did_not_say(tmp_path: Path) -> None:
    """``Resumed`` reports the units a process runs once a change confirmed there; ``status`` shows
    them. A process left with no unit reports an empty list, and status shows that, not the units it
    checked in with. A reply that says nothing (``None``) leaves the record as it was."""

    async def main() -> None:
        async with engine(tmp_path) as e:
            def row(p: ProcId) -> list[str]:
                return [u.unit for u in next(s for s in e.c.sup.status() if s.proc == p).units]

            assert row(W1) == sorted(UNITS)
            e.planner.script = lambda inp: ActionPlan(procs=[ProcActions(W1, reload=ReloadPlan(units_to_remove=[
                UnitId(u) for u in UNITS]))])
            r = await e.coord.submit(ChangeSet(base=e.coord.identity))
            assert r.status == "committed", r.diagnostics
            assert row(W1) == [] and (await e.status(W1)).units == []
            assert row(W0) == sorted(UNITS)
            e.c.sup.runs_units(W0, None)
            assert row(W0) == sorted(UNITS)

    run(main())
