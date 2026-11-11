"""A unit's build step (``[unit] build``, change.md §4.6): run while the engine serves, before
anything pauses, in a process of its own from the change's code; its failure fails the change
before the pause."""

from __future__ import annotations

import asyncio
from pathlib import Path

import msgspec
import pytest

from core.change.coordinator import PlanInput
from core.change.types import ActionPlan, ProcActions
from tests.framework.change.engine import ALL, SCHED, W1, engine, run
from tests.framework.cluster.harness import ALL_ROLES
from tests.framework.support import spec
from unit.runtime import ReloadPlan

BUILT = spec("toyc.versioned", "toyc.versioned:Versioned", process=ALL_ROLES, reads=["toyc.log_dir"],
             build="toyc.builder:make")


def reload_with_build(inp: PlanInput) -> ActionPlan:
    return ActionPlan(procs=[ProcActions(p, reload=ReloadPlan(changed_modules=["toyc.versioned"],
                                                              units_to_reload=[BUILT]), leads_epochs=p == SCHED)
                             for p in ALL])


def builds(log: Path) -> list[str]:
    return log.read_text().splitlines() if log.exists() else []


def test_a_build_step_runs_from_the_new_code_while_the_old_version_serves(tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    log = tmp_path / "builds.log"
    monkeypatch.setenv("TOYC_BUILD_LOG", str(log))
    monkeypatch.setenv("TOYC_BUILD_S", "1.5")

    async def main() -> None:
        async with engine(tmp_path / "engine") as e:
            e.planner.script = reload_with_build
            change = asyncio.ensure_future(e.coord.submit(e.code_change()))
            await asyncio.sleep(0.5)  # the steps run (1.5 s each, all processes at once)
            assert not change.done() and not builds(log)
            for p in ALL:  # the old version serves meanwhile, nothing is paused
                st = await e.status(p)
                assert not st.paused and e.versioned(p) == "versioned v1"
                assert await e.epochs_advance(p)
            r = await change
            assert r.status == "committed", r.diagnostics
            assert r.new_identity is not None
            lines = builds(log)
            assert len(lines) == len(ALL), lines  # one per process that reloads the unit
            assert all(r.new_identity in line for line in lines), lines  # from the change's snapshot
            for p in ALL:
                assert e.versioned(p) == "versioned v2"
                assert any("build step toyc.builder:make done" in line for line in e.c.output_of(p)), p
            assert r.timings["prepare"] >= 1.5

    run(main())


def test_a_failing_build_step_fails_the_change_before_anything_paused(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    log = tmp_path / "builds.log"
    monkeypatch.setenv("TOYC_BUILD_LOG", str(log))
    monkeypatch.setenv("TOYC_BUILD_FAIL", "1")

    async def main() -> None:
        async with engine(tmp_path / "engine") as e:
            old = e.coord.identity
            e.planner.script = reload_with_build
            r = await e.coord.submit(e.code_change())
            assert r.status == "aborted", r
            codes = {d.code for d in r.diagnostics}
            assert "change.build_failed" in codes, r.diagnostics
            reason = next(d.reason for d in r.diagnostics if d.code == "change.build_failed")
            assert "toyc.versioned" in reason and "exited with 3" in reason, reason
            assert "pause" not in r.timings, r.timings  # it never got that far
            assert e.coord.identity == old
            for p in ALL:
                st = await e.status(p)
                assert st.identity == old and not st.paused and e.versioned(p) == "versioned v1"
                assert await e.epochs_advance(p)
            assert builds(log)  # the steps did run

    run(main())


def test_a_change_without_build_steps_starts_no_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The other units' reloads, and every unit's load at a process's start, run no step."""
    log = tmp_path / "builds.log"
    monkeypatch.setenv("TOYC_BUILD_LOG", str(log))

    async def main() -> None:
        async with engine(tmp_path / "engine") as e:
            plain = msgspec.structs.replace(BUILT, manifest=msgspec.structs.replace(
                BUILT.manifest, unit=msgspec.structs.replace(BUILT.manifest.unit, build="")))

            def script(inp: PlanInput) -> ActionPlan:
                return ActionPlan(procs=[ProcActions(p, reload=ReloadPlan(changed_modules=["toyc.versioned"],
                                                                          units_to_reload=[plain]),
                                                     leads_epochs=p == SCHED) for p in ALL])

            e.planner.script = script
            r = await e.coord.submit(e.code_change())
            assert r.status == "committed", r.diagnostics
            assert not builds(log)
            assert e.versioned(W1) == "versioned v2"

    run(main())


def test_a_bad_build_entry_is_refused_by_the_manifest_check() -> None:
    from core.common.manifest import parse_manifest
    from core.common.result import Failed
    from tests.framework.support import manifest_toml

    bad = parse_manifest(manifest_toml("toyc.x", "toyc.x:X", build="not an entry"))
    assert isinstance(bad, Failed) and "bad build" in str(bad.diagnostics), bad
