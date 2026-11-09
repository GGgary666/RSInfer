"""``core.sched`` with its policies, request table and ledger, against fixture workers on two ranks.

Every test checks the tokens against what the fixture's model computes directly: a plan that
gives a row a wrong position, token or too few pages fails in the worker (``errors``).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, cast

import pytest

from core.contracts.requests import NewRequest
from core.sched.loop import Scheduler
from tests.framework.support import spec
from tests.units.scheduler.support import Bed, bed, request, specs
from tests.units.scheduler.units.schedfix.worker.unit import expected


def want(r: NewRequest) -> list[int]:
    assert r.sampling.seed is not None
    return expected([int(x) for x in r.prompt.values()], r.sampling.seed, r.max_tokens, r.stop_token_ids)


def check(b: Bed, requests: Sequence[NewRequest]) -> None:
    for r in requests:
        assert b.client.outputs[r.request].tokens == want(r), r.request
    b.run_until(b.idle, what="idle")
    assert b.errors() == [] and b.engine.ledger.check() == []
    assert b.engine.loop.report().fault is None


def added_at(b: Bed, rank: int = 0) -> dict[int, int]:
    """Request -> the first step whose delta added it (full deltas add every row again)."""
    out: dict[int, int] = {}
    for epoch, _, delta in b.workers[rank].fixture.plans_seen:
        for a in delta.added:
            out.setdefault(a.request, epoch)
    return out


def stopping(rid: int, prompt: list[int], max_tokens: int, at: int) -> NewRequest:
    """A request whose ``at``-th token (from 0) is its stop token."""
    seed = rid * 7 + 1
    return request(rid, prompt, max_tokens, seed=seed, stop=[expected(prompt, seed, max_tokens)[at]])


def test_requests_get_what_the_model_computes() -> None:
    with bed() as b:
        reqs = [request(1, [5, 6, 7, 8, 9], 6), request(2, list(range(1, 30)), 4),
                stopping(3, [3, 1, 4], 20, at=4)]
        b.client.send(reqs)
        b.run_until(b.finished(1, 2, 3))
        assert {r: b.client.outputs[r].finished for r in (1, 2, 3)} == {1: "length", 2: "length", 3: "stop"}
        assert len(b.client.outputs[3].tokens) == 5
        # the prompts share a budget of 16 tokens a step: the 29-token one goes in chunks
        prefills = [sum(int(n) for n in plan.num_tokens.values())
                    for _, plan, _ in b.workers[0].fixture.plans_seen if plan.phase == "prefill"]
        assert max(prefills) == 16 and sum(prefills) == 5 + 29 + 3
        check(b, reqs)


def test_first_come_first_served_does_not_let_a_later_request_overtake() -> None:
    """10 pages of 4 tokens: the first request holds 8; the second needs 3 and waits; the third
    needs 1 and would fit, but comes later, so it waits too."""
    with bed(capacity={"kv.paged": 10, "gdn.slots": 8}) as b:
        reqs = [request(1, list(range(1, 21)), 10), request(2, [1, 2, 3, 4, 5, 6], 6), request(3, [7, 8], 2)]
        b.client.send(reqs)
        b.run_until(b.finished(1, 2, 3))
        first = added_at(b)
        assert first[2] >= first[1] and first[3] >= first[2]
        decodes_before_2 = [e for e, plan, _ in b.workers[0].fixture.plans_seen if plan.phase == "decode" and e < first[2]]
        assert decodes_before_2  # the first request decoded while the second could not be admitted
        check(b, reqs)


def test_prefill_runs_first_without_mixing() -> None:
    with bed() as b:
        first = request(1, [1, 2, 3, 4], 30)
        b.client.send([first])
        b.run_until(lambda: len(b.client.outputs[1].tokens) >= 3)
        second = request(2, [9, 8, 7, 6, 5, 4, 3, 2], 2)
        b.client.send([second])
        b.run_until(b.finished(1, 2))
        epoch = added_at(b)[2]
        plan = next(p for e, p, _ in b.workers[0].fixture.plans_seen if e == epoch)
        assert plan.phase == "prefill" and len(plan.rows) == 1  # the decoding request sat out that step
        check(b, [first, second])


def run_set(*, overlap: bool) -> tuple[dict[int, list[int]], int, int]:
    reqs = [request(1, [5, 6, 7], 12), request(2, list(range(10, 40)), 9), stopping(3, [2, 2], 15, at=6),
            request(4, [1], 1)]
    with bed(overlap=overlap, delay_s=0.003) as b:
        b.client.send(reqs[:2])
        b.run_until(lambda: len(b.client.outputs[1].tokens) >= 2)
        b.client.send(reqs[2:])
        b.run_until(b.finished(1, 2, 3, 4))
        check(b, reqs)
        previous = sum(int(x) for w in b.workers for _, plan, _ in w.fixture.plans_seen
                       for x in plan.from_previous.values())
        return ({r.request: b.client.outputs[r.request].tokens for r in reqs},
                max(w.fixture.most_in_flight for w in b.workers), previous)


def test_overlap_gives_the_same_tokens_as_the_synchronous_loop() -> None:
    sync, sync_depth, sync_previous = run_set(overlap=False)
    over, over_depth, over_previous = run_set(overlap=True)
    assert over == sync
    assert (sync_depth, over_depth) == (1, 2)
    assert sync_previous == 0 and over_previous > 0


def test_a_step_retires_once_every_rank_answered() -> None:
    with bed() as b:
        b.workers[1].fixture.held = True
        r = request(1, [4, 5, 6], 3)
        b.client.send([r])
        b.run_until(lambda: bool(b.workers[0].fixture.answered) and bool(b.engine.loop.flights))
        deadline = time.monotonic() + 0.2
        while time.monotonic() < deadline:
            b.turn()
        assert b.client.outputs[1].tokens == [] and b.engine.table.last_retired < 1
        b.workers[1].fixture.held = False
        b.run_until(b.finished(1))
        check(b, [r])


@pytest.mark.parametrize("overlap", [False, True])
def test_resync_sends_the_whole_batch_again(overlap: bool) -> None:
    with bed(overlap=overlap, delay_s=0.002) as b:
        reqs = [request(1, [1, 2, 3], 14), request(2, list(range(1, 21)), 10)]
        b.client.send(reqs)
        b.run_until(lambda: all(len(b.client.outputs[r.request].tokens) >= 2 for r in reqs))
        at = b.engine.table.last_submitted + 2
        for w in b.workers:
            w.fixture.script[at] = "resync"
            # with overlap, step at + 1 goes out while the refusal of step at is on its way, and is
            # still running when it arrives
            w.fixture.slow.update({at: 0.1, at + 1: 0.3})
        b.run_until(b.finished(1, 2))
        full = [e for e, _, d in b.workers[0].fixture.plans_seen if e > at and d.full]
        assert full and full[0] == at + (2 if overlap else 1)  # after every step sent before it
        assert any("not run" in d for d in b.engine.loop.report().diagnostics)
        check(b, reqs)


def test_ranks_that_disagree_whether_a_step_ran_stop_the_engine() -> None:
    with bed() as b:
        b.client.send([request(1, [1, 2, 3], 30)])
        b.run_until(lambda: len(b.client.outputs[1].tokens) >= 2)
        b.workers[0].fixture.script[b.engine.table.last_submitted + 2] = "resync"
        b.run_until(lambda: b.engine.loop.fault is not None)
        assert "disagree" in b.engine.loop.fault.reason  # type: ignore[union-attr]
        assert b.errors() == []


def test_a_failed_worker_stops_the_steps_and_new_requests_get_an_error() -> None:
    with bed(overlap=True, delay_s=0.002) as b:
        held = request(1, [1, 2, 3], 40)
        b.client.send([held])
        b.run_until(lambda: len(b.client.outputs[1].tokens) >= 2)
        at = b.engine.table.last_submitted + 2
        b.workers[1].fixture.script[at] = "fail"
        b.run_until(lambda: b.engine.loop.fault is not None)
        fault = b.engine.loop.fault
        assert fault is not None and (fault.step, fault.rank) == (at, 1) and "scripted failure" in fault.reason
        sent = [len(w.fixture.received) for w in b.workers]
        b.client.send([request(2, [7, 7], 3)])
        b.run_until(b.finished(2))
        assert b.client.outputs[2].error is not None and "stopped serving" in b.client.outputs[2].error
        assert [len(w.fixture.received) for w in b.workers] == sent  # no step went out since
        assert b.client.outputs[1].finished is None  # held, for whoever changes the engine next
        t0 = time.monotonic()
        b.engine.loop.quiesce()  # does not wait for the failed steps: a change can still pause
        assert time.monotonic() - t0 < 1.0
        assert b.engine.loop.report().running == 1


def test_a_step_without_an_answer_counts_as_a_failure() -> None:
    with bed(config={"sched.step_timeout_s": 0.3}) as b:
        b.client.send([request(1, [1, 2], 30)])
        b.run_until(lambda: len(b.client.outputs[1].tokens) >= 1)
        b.workers[0].fixture.script[b.engine.table.last_submitted + 1] = "silent"
        b.run_until(lambda: b.engine.loop.fault is not None)
        assert "no answer" in b.engine.loop.fault.reason and b.engine.loop.fault.rank == 0  # type: ignore[union-attr]


def test_a_pause_that_cannot_finish_the_steps_fails_and_serving_goes_on() -> None:
    with bed(config={"sched.quiesce_timeout_s": 0.2}) as b:
        b.workers[1].fixture.held = True
        r = request(1, [3, 3, 3], 5)
        b.client.send([r])
        b.run_until(lambda: bool(b.engine.loop.flights))
        with pytest.raises(TimeoutError, match="no answer from rank"):
            b.engine.loop.quiesce()
        b.workers[1].fixture.held = False
        b.run_until(b.finished(1))
        check(b, [r])


def test_a_cancelled_request_keeps_its_items_until_its_last_step_retired() -> None:
    with bed(overlap=True, delay_s=0.002) as b:
        b.client.send([request(1, [1, 2, 3, 4], 60)])
        b.run_until(lambda: len(b.client.outputs[1].tokens) >= 3)
        for w in b.workers:
            w.fixture.held = True
        b.run_until(lambda: len(b.engine.loop.flights) == 2)
        b.client.send(abort=[1])
        b.run_until(b.finished(1))
        assert b.client.outputs[1].finished == "abort"
        assert 1 in b.engine.ledger.pools["kv.paged"].holdings  # its steps may still write them
        for w in b.workers:
            w.fixture.held = False
        b.run_until(b.idle)
        assert b.errors() == []


@pytest.mark.parametrize("overlap", [False, True])
def test_reloading_the_scheduler_keeps_requests_running(overlap: bool) -> None:
    with bed(overlap=overlap, delay_s=0.002) as b:
        reqs = [request(1, [1, 2, 3], 16), request(2, list(range(5, 30)), 12)]
        b.client.send(reqs)
        b.run_until(lambda: all(len(b.client.outputs[r.request].tokens) >= 3 for r in reqs))
        old = b.engine.runtime.instance("core.sched")
        b.engine.reload(b.engine.spec("core.sched"))
        assert b.engine.runtime.instance("core.sched") is not old
        boundary = b.engine.table.last_submitted
        b.run_until(b.finished(1, 2))
        after = [d.full for e, _, d in b.workers[0].fixture.plans_seen if e > boundary]
        assert after[0]  # the new instance does not know what the workers hold
        check(b, reqs)


def test_after_a_failure_a_reload_ends_the_requests_of_the_lost_steps() -> None:
    """Until changes can choose what to do with a failed worker (step 5), reloading the
    scheduler is how to go on: what the lost steps held ends with an error, the rest goes on."""
    with bed() as b:
        b.client.send([request(1, [1, 2, 3], 40)])
        b.run_until(lambda: len(b.client.outputs[1].tokens) >= 2)
        b.workers[0].fixture.script[b.engine.table.last_submitted + 1] = "fail"
        b.run_until(lambda: b.engine.loop.fault is not None)
        b.engine.reload(b.engine.spec("core.sched"))
        b.run_until(b.finished(1))
        assert b.client.outputs[1].error is not None and "never came back" in b.client.outputs[1].error
        fresh = request(2, [6, 5, 4], 5)
        b.client.send([fresh])
        b.run_until(b.finished(2))
        check(b, [fresh])


def test_a_policy_decision_out_of_bounds_is_refused_and_reported() -> None:
    units = specs(skip=["policies.prefill_first"], extra=[
        spec("schedtest.compose", "tests.units.scheduler.support:SwitchableCompose",
             provides=["policy.compose@1"], process=["sched"])])
    with bed(units=units) as b:
        r = request(1, [1, 2, 3], 4)
        b.client.send([r])
        b.run_until(lambda: b.engine.loop.policy_error is not None)
        assert "prompt tokens are left" in (b.engine.loop.policy_error or "")
        assert not b.engine.loop.flights and b.engine.loop.fault is None
        compose = b.engine.runtime.lookup("policy.compose@1")
        compose.bad = False  # type: ignore[union-attr]
        b.run_until(b.finished(1))
        assert b.engine.loop.policy_error is None
        check(b, [r])


def test_a_request_larger_than_the_state_is_refused() -> None:
    with bed(capacity={"kv.paged": 4, "gdn.slots": 8}) as b:
        b.client.send([request(1, [1, 2, 3], 100), request(2, [1, 2], 3)])
        b.run_until(b.finished(1, 2))
        assert "need more state" in (b.client.outputs[1].error or "")
        check(b, [request(2, [1, 2], 3)])


def test_a_ledger_rebuilt_under_running_requests_is_reported_not_trusted() -> None:
    """Reloading a state unit rebuilds it empty; until a change can be refused for that (step 5),
    the scheduler says what happened instead of giving the requests' items away again."""
    with bed() as b:
        b.client.send([request(1, [1, 2, 3], 40)])
        b.run_until(lambda: len(b.client.outputs[1].tokens) >= 2)
        b.engine.reload(b.engine.spec("core.state.alloc"))
        b.run_until(lambda: b.engine.loop.fault is not None)
        assert "rebuilt under them" in b.engine.loop.fault.reason  # type: ignore[union-attr]
        b.client.send([request(2, [4, 4], 3)])
        b.run_until(b.finished(2))
        assert b.client.outputs[2].error is not None


def test_a_frontend_that_takes_no_tokens_loses_its_requests_not_the_scheduler() -> None:
    with bed(config={"sched.frontend_timeout_s": 0.3}, delay_s=0.002) as b:
        b.client.send([request(1, [1, 2, 3], 200)])
        b.run_until(lambda: len(b.client.outputs[1].tokens) >= 2)
        b.client.tokens.close()  # the frontend is gone
        deadline = time.monotonic() + 10
        while list(b.engine.table.entries()):
            assert time.monotonic() < deadline, b.engine.loop.report()
            b.engine.loop.step()
        assert any("dropped" in d for d in b.engine.loop.report().diagnostics)
        b.run_until(b.idle)
        assert b.errors() == []


def test_planning_that_fails_after_an_admission_gives_back_what_it_took(monkeypatch: pytest.MonkeyPatch) -> None:
    with bed() as b:
        loop = cast(Scheduler, cast(Any, b.engine.runtime.instance("core.sched")).loop)
        send = loop._send_plan  # pyright: ignore[reportPrivateUsage]
        calls: list[int] = []

        def fail_once(step: Any) -> None:
            calls.append(step.epoch)
            if len(calls) == 1:
                raise RuntimeError("scripted failure while sending")
            send(step)

        monkeypatch.setattr(loop, "_send_plan", fail_once)
        r = request(1, [1, 2, 3], 4)
        b.client.send([r])
        b.run_until(lambda: bool(calls))
        assert b.engine.table.get(1).row is None  # type: ignore[union-attr]
        assert all(not p.holdings and len(p.free) == p.capacity for p in b.engine.ledger.pools.values())
        assert any("planning failed" in d for d in loop.report().diagnostics)
        b.run_until(b.finished(1))
        check(b, [r])


@pytest.mark.parametrize("ranks", [1, 2])
def test_a_worker_replaced_while_idle_serves_the_next_request(ranks: int) -> None:
    """Codex on #31: the results moved first, with nothing on the worker; the first plan to the new
    one then moves the plans too. That is one replacement, and the new request is not touched."""
    with bed(ranks=ranks) as b:
        b.client.send([request(1, [1, 2, 3], 3)])
        b.run_until(b.finished(1))
        b.run_until(b.idle)
        b.replace_worker(0)
        b.run_until(lambda: b.engine.loop.result_gens[0] == 2)
        fresh = request(2, [4, 5, 6], 3)
        b.client.send([fresh])
        b.run_until(b.finished(2))
        assert b.client.outputs[2].error is None
        assert len([e for e, _, d in b.workers[0].fixture.plans_seen if d.full]) == 1
        check(b, [fresh])


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("ranks", [1, 2])
@pytest.mark.parametrize("first", ["plans", "results"])
def test_a_replaced_worker_ends_the_requests_it_held_and_serves_new_ones(first: str, ranks: int, overlap: bool) -> None:
    """Codex on #31: request 1 holds a row on the old worker but is in no step when the replacement
    shows; it must not go on from state the new worker does not have. Request 2, sent to the new
    worker first, runs. With two ranks and the plans first, request 2's first step reaches the new
    rank 0 before anyone knows (it refuses it, waiting for a full delta) and rank 1 runs it
    (Codex's second review): the step counts as not run, and request 2 starts over from 0. With
    overlap its first two chunks are refused so; nothing of it was computed, so it still starts
    over."""
    with bed(ranks=ranks, delay_s=0.002, token_budget=4, overlap=overlap) as b:
        b.client.send([request(1, [1, 2, 3], 40)])
        b.run_until(lambda: len(b.client.outputs[1].tokens) >= 2)
        b.engine.loop.quiesce()  # nothing in flight on the old worker
        b.replace_worker(0, announce_after_s=1.0 if first == "plans" else 0.0)
        if first == "results":
            time.sleep(0.3)  # the new worker's Capacity waits for the scheduler's next turn
        fresh = request(2, list(range(1, 21)), 3)
        b.client.send([fresh])
        b.run_until(b.finished(1, 2))
        assert "gone" in (b.client.outputs[1].error or ""), b.client.outputs[1]
        assert b.client.outputs[2].error is None
        moves = [d for d in b.engine.loop.report().diagnostics if "new connection" in d or "new producer" in d]
        assert ("plans to" if first == "plans" else "results of") in moves[0], moves  # the order this test is for
        if ranks == 2 and first == "plans":
            assert any("refused by the new worker" in d for d in b.engine.loop.report().diagnostics)
        assert len(moves) == 1, moves  # the second sighting is paired, not handled again
        fulls = [e for e, _, d in b.workers[0].fixture.plans_seen if d.full]
        assert len(fulls) == 1, fulls  # one full delta for the new worker, not one per sighting
        check(b, [fresh])
