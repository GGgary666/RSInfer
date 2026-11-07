"""change.md §4.6 and §7 tests 14–15: prepare in slices while the old version serves."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from core.change.coordinator import PlanInput
from core.change.types import ActionPlan, ChangeSet, ProcActions, UnitAction
from core.common.ids import ProcId, UnitId
from tests.framework.change.engine import ALL, SCHED, W1, Engine, engine, reload_everywhere, run, wait_until


def op_everywhere(args: dict[ProcId, dict[str, Any]]) -> Any:
    def script(inp: PlanInput) -> ActionPlan:
        return ActionPlan(procs=[
            ProcActions(p, actions=[UnitAction(UnitId("toyc.tunable"), "toy.op", args.get(p, {}))],
                        leads_epochs=p == SCHED)
            for p in ALL])

    return script


def has(e: Engine, p: ProcId, line: str) -> bool:
    return line in e.c.log_of(p)


def test_sliced_prepare_keeps_serving_until_everything_is_prepared(tmp_path: Path) -> None:
    async def main() -> None:
        async with engine(tmp_path) as e:
            sliced = {"slices": 20, "slice_s": 0.05}
            e.planner.script = op_everywhere(dict.fromkeys(ALL, sliced))
            change = asyncio.ensure_future(e.coord.submit(ChangeSet(base=e.coord.identity)))
            await wait_until(lambda: has(e, W1, "op slice 2"))
            first = await e.status(W1)
            await wait_until(lambda: has(e, W1, "op slice 12"))
            later = await e.status(W1)
            assert (later.epoch or 0) > (first.epoch or 0) and not later.paused  # the old version serves
            assert not change.done()

            r = await change
            assert r.status == "committed", r.diagnostics
            for p in ALL:
                log = e.c.log_of(p)
                assert log.index("op slice 19") < log.index("op execute")  # paused only once prepared
            assert not [d for d in r.diagnostics if d.code == "change.prepare_one_shot"]

    run(main())


def test_a_failing_slice_aborts_the_change_without_leftovers(tmp_path: Path) -> None:
    async def main() -> None:
        async with engine(tmp_path) as e:
            old = e.coord.identity
            sliced = {"slices": 50, "slice_s": 0.02}
            e.planner.script = op_everywhere({**dict.fromkeys(ALL, sliced), W1: {**sliced, "fail_slice": 3}})
            r = await e.coord.submit(ChangeSet(base=old))
            assert r.status == "aborted"
            assert any("injected failure in slice 3" in d.reason for d in r.diagnostics)
            for p in ALL:
                assert e.ops(p) == ["op prepare", "op abort"], p
                st = await e.status(p)
                assert st.identity == old and not st.paused and not st.diagnostics
                assert not any(u.unit not in ("toyc.loop", "toyc.tunable", "toyc.versioned") for u in st.units)
            sliced_so_far = {p: len(e.slices(p)) for p in ALL}
            await asyncio.sleep(0.3)
            assert {p: len(e.slices(p)) for p in ALL} == sliced_so_far  # nothing keeps preparing
            assert sliced_so_far[W1] == 4

            e.planner.script = reload_everywhere()
            assert (await e.coord.submit(e.code_change())).status == "committed"

    run(main())


def test_a_one_shot_prepare_reports_how_long_serving_stopped(tmp_path: Path) -> None:
    async def main() -> None:
        async with engine(tmp_path) as e:
            e.planner.script = op_everywhere({W1: {"prepare_s": 0.5}})
            r = await e.coord.submit(ChangeSet(base=e.coord.identity))
            assert r.status == "committed", r.diagnostics
            stalls = {d.subject: d for d in r.diagnostics if d.code == "change.prepare_one_shot"}
            assert sorted(stalls) == sorted(str(p) for p in ALL)
            m = re.search(r"serving stopped for ([0-9.]+)s", stalls[str(W1)].reason)
            assert m is not None and float(m.group(1)) >= 0.5
            assert stalls[str(W1)].severity == "info"

    run(main())
