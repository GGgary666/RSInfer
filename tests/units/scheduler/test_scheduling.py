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


def test_a_failed_worker_stops_the_steps_and_ends_every_request() -> None:
    with bed(overlap=True, delay_s=0.002) as b:
        held = request(1, [1, 2, 3], 40)
        b.client.send([held])
        b.run_until(lambda: len(b.client.outputs[1].tokens) >= 2)
        at = b.engine.table.last_submitted + 2
        b.workers[1].fixture.script[at] = "fail"
        b.run_until(lambda: b.engine.loop.fault is not None)
        fault = b.engine.loop.fault
        assert fault is not None and (fault.step, fault.rank) == (at, 1) and "scripted failure" in fault.reason
        # A step planned before the failure answer may still be on its way to a rank, so count what
        # the scheduler sent, not what the workers have read yet.
        last = b.engine.table.last_submitted
        b.client.send([request(2, [7, 7], 3)])
        b.run_until(b.finished(2))
        assert b.client.outputs[2].error is not None and "stopped serving" in b.client.outputs[2].error
        assert b.engine.table.last_submitted == last  # no step went out since
        assert all(e <= last for w in b.workers for e in w.fixture.received)
        # The running request ended with the failure: no step goes out, so it could never end by
        # itself, and its client is not left waiting (#42 held it for the next change).
        b.run_until(b.finished(1))
        assert "stopped serving" in (b.client.outputs[1].error or "")
        t0 = time.monotonic()
        b.engine.loop.quiesce()  # does not wait for the failed steps: a change can still pause
        assert time.monotonic() - t0 < 1.0
        assert b.engine.loop.occupancy() == (0, 0, 0)  # what it still holds goes back once a change commits


@pytest.mark.parametrize("overlap", [False, True])
def test_a_change_that_commits_after_a_failure_brings_serving_back(overlap: bool) -> None:
    """The change meant to repair the worker commits: the identity served is not the one that
    failed, and the loop starts over as a new instance would. The steps that never came back are
    given up (their items go back), and the first plan is a full delta, which drops the rows the
    workers still had (doc/framework-bugs/2026-11-11-a-repaired-worker-stays-stopped.md)."""
    with bed(overlap=overlap, delay_s=0.002) as b:
        b.client.send([request(1, [1, 2, 3], 40), request(2, list(range(5, 20)), 30)])
        b.run_until(lambda: all(len(b.client.outputs[r].tokens) >= 2 for r in (1, 2)))
        b.workers[0].fixture.script[b.engine.table.last_submitted + 1] = "fail"
        b.run_until(lambda: b.engine.loop.fault is not None)
        b.run_until(b.finished(1, 2))
        b.client.send([request(3, [7, 7], 3)])
        b.run_until(b.finished(3))
        assert all("stopped serving" in (b.client.outputs[r].error or "") for r in (1, 2, 3))
        boundary = b.engine.table.last_submitted
        # With overlap, the step after the failed one was on its way already: rank 0 gets a plan
        # that assumes the failed step ran, and the fixture says so. Nothing after the change may.
        before = b.errors()
        assert all(int(e.split()[1].rstrip(":")) <= boundary for e in before), before
        assert overlap or not before
        for w in b.workers:
            w.fixture.errors.clear()
        b.new_identity("v2")
        fresh = [request(4, [6, 5, 4], 6), request(5, list(range(9, 30)), 5)]
        b.client.send(fresh)
        b.run_until(b.finished(4, 5))
        assert [d.full for e, _, d in b.workers[0].fixture.plans_seen if e > boundary][0]
        assert any("serving again" in d for d in b.engine.loop.report().diagnostics)
        check(b, fresh)


def test_a_change_that_does_not_repair_the_worker_fails_the_same_way_again() -> None:
    with bed() as b:
        b.client.send([request(1, [1, 2, 3], 40)])
        b.run_until(lambda: len(b.client.outputs[1].tokens) >= 1)
        b.workers[1].fixture.script[b.engine.table.last_submitted + 1] = "fail"
        b.run_until(lambda: b.engine.loop.fault is not None)
        b.new_identity("v2")
        b.client.send([request(2, [4, 4, 4], 30)])
        b.run_until(lambda: len(b.client.outputs[2].tokens) >= 1)
        b.workers[1].fixture.script[b.engine.table.last_submitted + 1] = "fail"
        b.run_until(b.finished(2))
        assert "stopped serving" in (b.client.outputs[2].error or "")
        assert b.engine.loop.fault is not None
        b.new_identity("v3")
        fresh = request(3, [2, 2], 4)
        b.client.send([fresh])
        b.run_until(b.finished(3))
        check(b, [fresh])


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
        # Every step in flight lacks some rank's answer, and no worker has one queued to go out: an
        # answer queued before the hold is still written, and would retire the step.
        b.run_until(lambda: len(b.engine.loop.flights) == 2 and all(not w.fixture.outbox for w in b.workers)
                    and not any(all(f.epoch in w.fixture.answered for w in b.workers) for f in b.engine.loop.flights))
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


def test_after_a_failure_a_reload_of_the_scheduler_serves_again() -> None:
    """A new instance gives up the steps the old one lost, as a committed change does."""
    with bed() as b:
        b.client.send([request(1, [1, 2, 3], 40)])
        b.run_until(lambda: len(b.client.outputs[1].tokens) >= 2)
        b.workers[0].fixture.script[b.engine.table.last_submitted + 1] = "fail"
        b.run_until(lambda: b.engine.loop.fault is not None)
        b.run_until(b.finished(1))  # it ended with the failure
        assert "stopped serving" in (b.client.outputs[1].error or "")
        b.engine.reload(b.engine.spec("core.sched"))
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

        def fail_once(steps: Any) -> None:  # one part per scheduling domain
            calls.append(steps[0].epoch)
            if len(calls) == 1:
                raise RuntimeError("scripted failure while sending")
            send(steps)

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


def test_admission_keeps_to_what_a_row_of_the_workers_holds() -> None:
    """#30's ``batch.max_context_len`` and ``batch.max_rows`` (stage2-contract.md §4.3): a request
    longer than a row is refused, not sent to a worker that would raise; plans use the rows the
    workers have, even when ``sched.max_batch`` says more."""
    with bed(max_batch=8, config={"batch.max_context_len": 16, "batch.max_rows": 2}) as b:
        long = request(1, list(range(1, 11)), 10)
        fits = [request(rid, [rid, rid + 1, rid + 2], 13) for rid in (2, 3, 4)]
        b.client.send([long, *fits])
        b.run_until(b.finished(1, 2, 3, 4))
        assert "a row of the workers holds 16" in (b.client.outputs[1].error or "")
        rows = {a.row for _, _, d in b.workers[0].fixture.plans_seen for a in d.added}
        assert rows <= {0, 1}
        check(b, fits)


def test_with_the_prefix_cache_a_token_budget_that_is_not_a_multiple_of_the_page_is_refused() -> None:
    """A chunk that does not end its prompt ends on a page: with a budget below a page none ever ran,
    and a long request waited forever (Codex, #59). Refused when the scheduler loads, as SGLang
    refuses a ``chunked_prefill_size`` that is not a multiple of ``page_size``; without the cache
    chunks end anywhere."""
    with pytest.raises(Exception, match=r"sched.token_budget = 2 must be a multiple of .*page_size = 4"):
        with bed(token_budget=2, config={"alloc.radix_cache": True}):
            pass
    with bed(token_budget=2) as b:
        r = request(1, list(range(1, 12)), 3)
        b.client.send([r])
        b.run_until(b.finished(1))
        check(b, [r])


def positions(b: Bed, rid: int, rank: int = 0) -> list[int]:
    """Where each plan rank ``rank`` was sent started request ``rid``, run or refused."""
    owner: dict[int, int] = {}
    out: list[int] = []
    for _, plan, delta in b.workers[rank].fixture.plans_seen:
        for a in delta.added:
            owner[a.row] = a.request
        out += [int(p) for r, p in zip(plan.rows.values(), plan.positions.values(), strict=True)
                if owner.get(int(r)) == rid]
    return out


def test_a_request_admitted_from_the_cache_of_a_replaced_worker_starts_over_from_0() -> None:
    """Antigravity on #59: request 2 starts at request 1's cached 64 tokens, and its first step is
    the first plan over the new worker's connection. The new worker has none of the old one's
    stores, so request 2 starts over from 0, as a request with nothing computed does; from 64 the
    fixture worker finds its pages and slot empty and fails the step."""
    first = list(range(1, 71))
    with bed(ranks=1, token_budget=128, config={"alloc.radix_cache": True}) as b:
        b.client.send([request(1, first, 2)])
        b.run_until(b.finished(1))
        b.run_until(b.idle)
        b.replace_worker(0, announce_after_s=1.0)  # the plans move first
        later = request(2, [*first[:64], 99, 98], 3)
        b.client.send([later])
        b.run_until(b.finished(2))
        assert b.client.outputs[2].error is None, b.client.outputs[2]
        assert positions(b, 2)[:2] == [64, 0]  # refused from the cache, then run from 0
        assert any("start over from position 0" in d for d in b.engine.loop.report().diagnostics)
        check(b, [later])


def test_a_waiting_request_whose_chunk_reaches_no_page_is_not_admitted_and_evicts_nothing() -> None:
    """Antigravity on #59: request 30 leaves 2 tokens of the budget to request 31, which on pages
    of 4 computes none of them. Admitting it anyway took pages from the cache (request 30 holds
    the free ones), then gave them back with the cached states gone; once request 30 ended, there
    were free pages for request 31."""
    with bed(ranks=1, token_budget=64, capacity={"kv.paged": 64, "gdn.slots": 13},
             config={"alloc.radix_cache": True}) as b:
        ledger: Any = b.engine.ledger
        for rid in (1, 2, 3):  # each leaves its state at 64 and 16 pages in the cache
            b.client.send([request(rid, [rid * 100 + i for i in range(66)], 1)])
            b.run_until(b.finished(rid))
            b.run_until(b.idle)
        assert len(ledger.tree.held()[1]) == 3 and len(ledger.free["kv.paged"]) == 16
        both = [request(30, list(range(1000, 1062)), 1), request(31, list(range(2000, 2040)), 2)]
        b.client.send(both)
        b.run_until(b.finished(30, 31))
        assert len(ledger.tree.held()[1]) == 3
        check(b, both)


def test_multi_domain_prefill_interval_coordinates_chunked_prefills_while_decoding() -> None:
    """With ``sched.prefill_interval > 0`` across DP domains, an isolated multi-chunk prompt pauses
    between chunks once other domains are decoding without prefill work, and resumes in lockstep
    when the other domain also receives prefill work."""
    with bed(ranks=2, token_budget=8,
             config={"sched.domains": 2, "sched.prefill_interval": 16, "sched.balance_prefill": True}) as b:
        r1 = request(1, [10, 11, 12, 13], 35)
        r2 = request(2, [20, 21, 22, 23], 35)
        b.client.send([r1, r2])
        b.run_until(lambda: len(b.client.outputs[1].tokens) >= 2 and len(b.client.outputs[2].tokens) >= 2)

        r3 = request(3, list(range(100, 120)), 3)  # 20 tokens -> 3 chunks at budget 8
        b.client.send([r3])
        # Wait until r3 runs its first 8-token chunk on one domain, then verify decode resumes instead
        # of running r3's second chunk alone while the other domain only decodes.
        b.run_until(lambda: any(
            plan.phase == "prefill" and sum(int(n) for n in plan.num_tokens.values()) == 8
            for w in b.workers for _, plan, _ in w.fixture.plans_seen
        ))
        toks_before = len(b.client.outputs[1].tokens) + len(b.client.outputs[2].tokens)
        b.run_until(lambda: len(b.client.outputs[1].tokens) + len(b.client.outputs[2].tokens) >= toks_before + 4)

        r4 = request(4, list(range(200, 216)), 3)
        b.client.send([r4])
        b.run_until(b.finished(1, 2, 3, 4))
        joint_epochs = {
            e0 for e0, p0, _ in b.workers[0].fixture.plans_seen if p0.phase == "prefill"
        } & {
            e1 for e1, p1, _ in b.workers[1].fixture.plans_seen if p1.phase == "prefill"
        }
        assert len(joint_epochs) >= 2  # initial (r1, r2) plus coordinated (r3 remaining chunk, r4)
        check(b, [r1, r2, r3, r4])
