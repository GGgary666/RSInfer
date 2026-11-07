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
from tests.framework.support import spec


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


SLOW_KERNEL = """
import time

from toyc.logutil import log


class Kernel:
    def __init__(self, ctx, config):
        time.sleep(0.5)  # an expensive constructor; the margin for the timing below
        log(config, ctx, f"built {ctx.id}")
"""


def test_new_units_are_staged_one_per_safe_point_while_the_old_version_serves(tmp_path: Path) -> None:
    """change.md §4.6: staging is a sliced prepare too. Each new unit is built at its own safe point,
    and the process keeps stepping and answering in between, instead of stopping until all are built.
    A request sent once unit i is built is answered at the next safe point, so before unit i + 2."""

    async def main() -> None:
        async with engine(tmp_path) as e:
            kernels = [spec(f"toyc.k{i}", "toyc.slow:Kernel", process=["worker"], reads=["toyc.log_dir"])
                       for i in range(5)]
            e.planner.script = lambda inp: ActionPlan(procs=[ProcActions(W1, stage=kernels)])
            files: dict[str, str | None] = {"engine/toyc/slow.py": e.c.blobs.put(SLOW_KERNEL.encode())}
            change = asyncio.ensure_future(e.coord.submit(ChangeSet(base=e.coord.identity, files=files)))
            replies = []
            for built, not_yet in (("built toyc.k0", "built toyc.k2"), ("built toyc.k2", "built toyc.k4")):
                await wait_until(lambda built=built: has(e, W1, built))
                replies.append(await e.status(W1))
                assert not has(e, W1, not_yet), f"answered only after {not_yet!r}: staging did not slice"
            assert not replies[0].paused and not replies[1].paused  # the old version serves
            assert (replies[1].epoch or 0) > (replies[0].epoch or 0)  # and it stepped in between
            r = await change
            assert r.status == "committed", r.diagnostics
            assert not [d for d in r.diagnostics if d.code == "change.prepare_one_shot"]
            assert {f"toyc.k{i}" for i in range(5)} <= {u.unit for u in (await e.status(W1)).units}

    run(main())
