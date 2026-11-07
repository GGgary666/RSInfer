"""change.md §4.7 and §7 tests 3, 12, 13: participants that lose the coordinator."""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Callable
from pathlib import Path

import msgspec
import pytest

from core.change.coordinator import PlanInput
from core.change.state import apply_changeset
from core.change.protocol import Commit
from core.change.types import ActionPlan, ChangeSet, ProcActions, UnitAction
from core.common.ids import Identity, ProcId, UnitId
from core.common.result import Result, fail
from core.supervisor import desired_procs
from tests.framework.change.engine import ALL, SCHED, W0, W1, Engine, engine, reload_everywhere, run, wait_until
from tests.framework.cluster.harness import process_plan

LEASE = 1.0
"""``change.coordinator_timeout_s`` in these tests."""
BEAT = 0.2
"""``change.keepalive_interval_s``, unless a test turns keepalives off."""


def codes(diags: list[object]) -> set[str]:
    return {getattr(d, "code", "") for d in diags}


async def notes(e: Engine, p: ProcId) -> set[str]:
    return codes(list((await e.status(p)).diagnostics))


def restart_w1(e: Engine, init_delay_s: float) -> Callable[[PlanInput], ActionPlan]:
    """SCHED and W0 change in place and then wait for Commit while W1 restarts slowly (loads weights)."""
    d = desired_procs(e.topo)[W1]

    def script(inp: PlanInput) -> ActionPlan:
        inplace = reload_everywhere()(inp).procs[:2]
        fresh = process_plan(d, Identity(""), {"toyc.log_dir": str(e.c.logs), "toyc.init_delay_s": init_delay_s})
        return ActionPlan(procs=[*inplace, ProcActions(W1, restart=fresh)])

    return script


def slow_prepare(slices: int) -> Callable[[PlanInput], ActionPlan]:
    def script(inp: PlanInput) -> ActionPlan:
        return ActionPlan(procs=[
            ProcActions(p, actions=[UnitAction(UnitId("toyc.tunable"), "toy.op", {"slices": slices, "slice_s": 0.05})],
                        leads_epochs=p == SCHED)
            for p in ALL])

    return script


async def assert_rolled_back(e: Engine, old: Identity, code: str) -> None:
    for p in ALL:
        await wait_until(lambda p=p: e.ops(p)[-1:] == ["op abort"])
        st = await e.status(p)
        assert st.identity == old and not st.paused, p
        assert code in codes(list(st.diagnostics)), (p, st.diagnostics)
        assert e.versioned(p) == "versioned v1", p
        assert await e.epochs_advance(p), p


def test_a_participant_still_preparing_withdraws_when_the_coordinator_vanishes(tmp_path: Path) -> None:
    async def main() -> None:
        async with engine(tmp_path, coordinator_timeout_s=LEASE, keepalive_interval_s=BEAT) as e:
            old = e.coord.identity
            e.planner.script = slow_prepare(slices=200)
            e.coord.faults.add("vanish_during_prepare")
            assert (await e.coord.submit(e.code_change())).status == "aborted"
            await assert_rolled_back(e, old, "change.coordinator_lost")
            for p in ALL:
                assert e.ops(p) == ["op prepare", "op abort"]
                assert 0 < len(e.slices(p)) < 200  # it withdrew in the middle
                reasons = [d.reason for d in (await e.status(p)).diagnostics if d.code == "change.coordinator_lost"]
                assert "while preparing" in reasons[-1]

    run(main())


def test_prepared_participants_roll_back_only_when_the_coordinator_cannot_be_reached(tmp_path: Path) -> None:
    async def main() -> None:
        async with engine(tmp_path, coordinator_timeout_s=LEASE, keepalive_interval_s=BEAT) as e:
            old = e.coord.identity
            e.planner.script = reload_everywhere()
            e.coord.faults.add("vanish_after_prepare")
            assert (await e.coord.submit(e.code_change())).status == "aborted"
            await assert_rolled_back(e, old, "change.coordinator_lost")
            for p in ALL:
                assert e.ops(p) == ["op prepare", "op abort"]
                reasons = [d.reason for d in (await e.status(p)).diagnostics if d.code == "change.coordinator_lost"]
                assert "could not reach it" in reasons[-1]

    run(main())


def test_a_participant_that_asks_and_hears_abort_rolls_back(tmp_path: Path) -> None:
    async def main() -> None:
        async with engine(tmp_path, coordinator_timeout_s=LEASE, keepalive_interval_s=BEAT) as e:
            old = e.coord.identity
            e.planner.script = reload_everywhere()
            e.coord.faults |= {"after_pause", "skip_abort"}  # decides abort, but no Abort reaches anyone
            r = await e.coord.submit(e.code_change())
            assert r.status == "failed" and e.coord.decision_of(r.change_id) == "abort"
            assert e.coord.recovery_required
            await assert_rolled_back(e, old, "change.rolled_back_on_query")

    run(main())


def test_a_participant_that_asks_and_hears_commit_commits(tmp_path: Path) -> None:
    async def main() -> None:
        async with engine(tmp_path, coordinator_timeout_s=LEASE, keepalive_interval_s=BEAT) as e:
            e.planner.script = reload_everywhere()
            e.coord.faults.add("delay_commit")  # Commit only comes after every lease ran out
            r = await e.coord.submit(e.code_change())
            assert r.status == "committed", r.diagnostics
            assert e.coord.decision_of(r.change_id) == "commit"  # answered from the event log
            assert e.coord.decision_of(10_000) == "unknown"
            for p in ALL:
                st = await e.status(p)
                assert st.identity == r.new_identity and not st.paused
                assert "change.committed_on_query" in codes(list(st.diagnostics))
                assert e.ops(p) == ["op prepare", "op execute", "op commit"]
                assert e.versioned(p) == "versioned v2"

    run(main())


def test_a_participant_that_hears_pending_keeps_waiting(tmp_path: Path) -> None:
    async def main() -> None:
        async with engine(tmp_path, coordinator_timeout_s=LEASE, keepalive_interval_s=3600.0) as e:
            e.planner.script = restart_w1(e, init_delay_s=3 * LEASE)
            r = await e.coord.submit(e.code_change())
            assert r.status == "committed", r.diagnostics
            for p in (SCHED, W0):
                assert "change.decision_pending" in await notes(e, p)
                assert e.ops(p) == ["op prepare", "op execute", "op commit"]
            for p in ALL:
                assert (await e.status(p)).identity == r.new_identity

    run(main())


def test_keepalives_carry_participants_through_a_long_execute(tmp_path: Path) -> None:
    """§7 test 12: executing takes longer than the lease; nobody rolls back."""

    async def main() -> None:
        async with engine(tmp_path, coordinator_timeout_s=LEASE, keepalive_interval_s=BEAT) as e:
            pids = {p: e.c.sup.local_pid(p) for p in (SCHED, W0)}
            e.planner.script = restart_w1(e, init_delay_s=3 * LEASE)
            r = await e.coord.submit(e.code_change())
            assert r.status == "committed", r.diagnostics
            assert r.timings["execute"] > 2 * LEASE
            for p in ALL:
                st = await e.status(p)
                assert st.identity == r.new_identity
                assert not {c for c in codes(list(st.diagnostics)) if c.startswith("change.")}, st.diagnostics
                assert e.versioned(p) == "versioned v2"
            for p in (SCHED, W0):
                assert e.ops(p) == ["op prepare", "op execute", "op commit"]
                assert e.c.sup.local_pid(p) == pids[p]

    run(main())


def test_executed_participants_wait_when_queries_fail_then_commit_without_restart(tmp_path: Path) -> None:
    """§7 test 13: committed means every participant serves the new identity."""

    async def main() -> None:
        async with engine(tmp_path, coordinator_timeout_s=LEASE, keepalive_interval_s=3600.0) as e:
            pids = {p: e.c.sup.local_pid(p) for p in (SCHED, W0)}
            e.planner.script = restart_w1(e, init_delay_s=3 * LEASE)
            e.coord.faults.add("no_decisions")  # Query failure must not resume old code after Execute.
            r = await e.coord.submit(e.code_change())
            assert r.status == "committed", r.diagnostics
            assert r.restarted == [W1]
            for p in ALL:
                assert (await e.status(p)).identity == r.new_identity == e.coord.identity
                assert e.versioned(p) == "versioned v2"
            for p in (SCHED, W0):
                assert e.c.sup.local_pid(p) == pids[p]
                assert e.ops(p) == ["op prepare", "op execute", "op commit"]
                assert "change.in_doubt" in await notes(e, p)

    run(main())


def test_missing_commit_never_resumes_candidates_or_allows_another_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def main() -> None:
        async with engine(tmp_path, commit_ack_timeout_s=2.0, commit_retry_interval_s=0.01) as e:
            e.planner.script = restart_w1(e, init_delay_s=0.0)
            original = e.c.sup.call

            async def call(proc: ProcId, msg: msgspec.Struct, *, timeout: float | None = None
                           ) -> Result[msgspec.Struct]:
                if proc == W0 and isinstance(msg, Commit):
                    return fail("test", "rpc", "Commit lost", code="ipc.timeout")
                return await original(proc, msg, timeout=timeout)

            monkeypatch.setattr(e.c.sup, "call", call)
            r = await e.coord.submit(e.code_change())
            assert r.status == "failed", r.diagnostics
            assert "change.recovery_required" in codes(list(r.diagnostics))
            assert e.coord.identity == r.new_identity
            summary = [d for d in r.diagnostics if d.code == "change.recovery_required" and not d.subject]
            assert f"decided to commit identity {r.new_identity}" in summary[0].reason
            for p in ALL:
                assert (await e.status(p)).paused
                assert e.versioned(p) == "versioned v2"
            blocked = await e.coord.submit(ChangeSet(base=e.coord.identity))
            assert blocked.status == "failed"
            assert "change.recovery_required" in codes(list(blocked.diagnostics))

    run(main())


def test_a_hung_participant_ends_the_commit_with_recovery_not_a_hang(tmp_path: Path) -> None:
    """§7 test 16: alive but not answering is not lost. Confirmation gives up within
    ``change.commit_ack_timeout_s``, nobody resumes, and later changes are refused."""

    async def main() -> None:
        async with engine(tmp_path, coordinator_timeout_s=LEASE, keepalive_interval_s=BEAT, commit_ack_timeout_s=2.0,
                          commit_retry_interval_s=0.1) as e:
            cs = e.code_change()
            slow = restart_w1(e, init_delay_s=2.0)
            e.planner.script = lambda inp: slow(inp) if inp.cs.files else ActionPlan()
            w1 = e.c.sup.local_pid(W1)
            change = asyncio.ensure_future(e.coord.submit(cs))
            await wait_until(lambda: "op execute" in e.ops(W0))
            # W0 logs "op execute" before its Executed is on the wire; the coordinator relaunches W1
            # only once every Executed is in, so wait for that, then hang W0 during the commit.
            await wait_until(lambda: e.c.sup.local_pid(W1) != w1)
            hung = e.c.sup.local_pid(W0)
            assert hung is not None
            os.kill(hung, signal.SIGSTOP)  # like a rank stuck in NCCL: its agent still sees it running
            follow = e.coord.enqueue(ChangeSet(base=e.coord.identity_fn(apply_changeset(e.coord.manifest, cs))))
            try:
                r = await asyncio.wait_for(change, 30)
                assert r.status == "failed" and "change.recovery_required" in codes(list(r.diagnostics))
                assert r.timings["commit"] < 2.0 + 1.0
                assert e.coord.identity == r.new_identity  # decided, not rolled back
                for p in (SCHED, W1):
                    assert (await e.status(p)).paused  # no one resumes before everyone committed
                blocked = await asyncio.wait_for(follow, 10)
                assert blocked.status == "failed" and "change.recovery_required" in codes(list(blocked.diagnostics))
            finally:
                os.kill(hung, signal.SIGCONT)

    run(main())
