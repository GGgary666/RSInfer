"""The time limit of ``Execute`` is ``change.execute_timeout_s`` (3600 s by default), not the
supervisor's limit for any call (``supervisor.rpc_timeout_s``, 30 s): the reload at the safe point
may read a checkpoint and record CUDA graphs, about 30 s for a 64-layer model on a SM100 (waking
from ``holder.sleep``, doc/evaluations/recovery-sleep-cli.md)."""

from __future__ import annotations

from pathlib import Path

from core.change.coordinator import PlanInput
from core.change.types import ActionPlan, ChangeSet, ProcActions, UnitAction
from core.common.ids import UnitId
from tests.framework.change.engine import ALL, SCHED, W1, engine, run

RPC_S = 1.0
"""``supervisor.rpc_timeout_s`` here."""


def slow_execute_on_w1(inp: PlanInput) -> ActionPlan:
    return ActionPlan(procs=[
        ProcActions(p, actions=[UnitAction(UnitId("toyc.tunable"), "toy.op", {"execute_s": 3 * RPC_S} if p == W1 else {})],
                    leads_epochs=p == SCHED) for p in ALL])


def test_an_execute_longer_than_any_call_may_take_commits(tmp_path: Path) -> None:
    async def main() -> None:
        async with engine(tmp_path, sup={"rpc_timeout_s": RPC_S}) as e:
            e.planner.script = slow_execute_on_w1
            r = await e.coord.submit(ChangeSet(base=e.coord.identity))
            assert r.status == "committed", r.diagnostics
            assert r.timings["execute"] >= 3 * RPC_S

    run(main())


def test_an_execute_past_its_own_limit_is_rolled_back(tmp_path: Path) -> None:
    """The ``Abort`` that follows waits as long again (the participant reads it once its Execute is
    done, then reloads back): an Execute that overran its limit by less than that is rolled back and
    confirmed, the engine needs no recovery. With ``rpc_timeout_s`` for the Abort it did."""
    async def main() -> None:
        async with engine(tmp_path, sup={"rpc_timeout_s": RPC_S}, execute_timeout_s=2 * RPC_S) as e:
            old = e.coord.identity
            e.planner.script = slow_execute_on_w1
            r = await e.coord.submit(ChangeSet(base=old))
            assert r.status == "aborted", r
            assert "change.recovery_required" not in {d.code for d in r.diagnostics}, r.diagnostics
            assert e.coord.identity == old and not e.coord.recovery_required
            for p in ALL:
                assert not (await e.status(p)).paused

    run(main())
