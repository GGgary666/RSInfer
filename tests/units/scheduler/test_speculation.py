"""``core.sched`` with speculation (doc/specs/mtp.md): decode rows verify ``spec.num_draft_tokens``
(D) tokens and advance by what the fixture worker accepts, 1 to D by seed and position, always tokens
of the same chain. So a request generates what it does without speculation, with overlap or not, on
one rank or two, and a plan that gives a row too few pages for its window fails in the worker."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from core.contracts.requests import NewRequest
from tests.units.scheduler.support import Bed, bed, request
from tests.units.scheduler.units.schedfix.worker.unit import accepted, expected

D = 4
SPEC = {"spec.algorithm": "eagle", "spec.num_draft_tokens": D}


def want(r: NewRequest) -> list[int]:
    assert r.sampling.seed is not None
    return expected([int(x) for x in r.prompt.values()], r.sampling.seed, r.max_tokens, r.stop_token_ids)


def check(b: Bed, requests: Sequence[NewRequest]) -> None:
    for r in requests:
        assert b.client.outputs[r.request].tokens == want(r), r.request
    b.run_until(b.idle, what="idle")
    assert b.errors() == [] and b.engine.ledger.check() == []
    assert b.engine.loop.report().fault is None


def decode_steps(b: Bed) -> int:
    return sum(1 for _, plan, _ in b.workers[0].fixture.plans_seen if plan.phase == "decode")


@pytest.mark.parametrize(("overlap", "ranks"), [(False, 1), (True, 1), (False, 2), (True, 2)])
def test_speculative_decode_generates_what_decoding_one_token_a_step_does(overlap: bool, ranks: int) -> None:
    seed = 7
    stop_at = expected([3, 1, 4], seed, 30)[9]
    reqs = [request(1, [5, 6, 7, 8, 9], 17), request(2, list(range(1, 30)), 9),
            request(3, [3, 1, 4], 30, seed=seed, stop=[stop_at])]
    with bed(overlap=overlap, ranks=ranks, delay_s=0.001, config=SPEC) as b:
        b.client.send(reqs)
        b.run_until(b.finished(1, 2, 3))
        assert b.client.outputs[3].finished == "stop"
        plans = [plan for _, plan, _ in b.workers[0].fixture.plans_seen]
        assert {p.draft_tokens for p in plans} == {D} and {p.phase for p in plans} <= {"prefill", "decode"}
        decoded = sum(len(want(r)) for r in reqs)
        assert decode_steps(b) < decoded - len(reqs)  # steps accepted more than one token
        check(b, reqs)


def test_a_step_that_accepts_past_max_tokens_or_a_stop_token_gives_only_what_the_request_asked() -> None:
    first = accepted(11, 5, D)  # the first decode of a 5-token prompt (seed 11) starts at position 5
    assert first > 2
    reqs = [request(1, [9, 8, 7, 6, 5], 2, seed=11), request(2, [9, 8, 7, 6, 5], 30, seed=11,
                                                              stop=[expected([9, 8, 7, 6, 5], 11, 30)[1]])]
    with bed(config=SPEC) as b:
        b.client.send(reqs)
        b.run_until(b.finished(1, 2))
        assert [len(b.client.outputs[r].tokens) for r in (1, 2)] == [2, 2]
        assert b.client.outputs[2].finished == "stop"
        check(b, reqs)


@pytest.mark.parametrize("overlap", [False, True])
def test_after_a_resync_the_rows_go_on_from_where_the_retired_steps_left_them(overlap: bool) -> None:
    with bed(overlap=overlap, delay_s=0.002, config=SPEC) as b:
        reqs = [request(1, [1, 2, 3], 40), request(2, list(range(1, 21)), 30)]
        b.client.send(reqs)
        b.run_until(lambda: all(len(b.client.outputs[r.request].tokens) >= 4 for r in reqs))
        at = b.engine.table.last_submitted + 2
        for w in b.workers:
            w.fixture.script[at] = "resync"
            w.fixture.slow.update({at: 0.1, at + 1: 0.3})
        b.run_until(b.finished(1, 2))
        full = [(e, plan) for e, plan, d in b.workers[0].fixture.plans_seen if e > at and d.full]
        assert full and not any(int(p) for p in full[0][1].from_previous.values())  # tokens given again
        check(b, reqs)


def test_a_request_without_room_for_the_speculation_window_is_refused() -> None:
    """36 tokens fit a row of 40, but not with the window a decode step writes past them (2 x D)."""
    with bed(config={**SPEC, "batch.max_context_len": 40}) as b:
        b.client.send([request(1, list(range(1, 21)), 16)])
        b.run_until(b.finished(1))
        assert b.client.outputs[1].finished == "abort"
        assert "speculation window" in (b.client.outputs[1].error or "")
    with bed(config={"batch.max_context_len": 40}) as b:
        r = request(1, list(range(1, 21)), 16)
        b.client.send([r])
        b.run_until(b.finished(1))
        check(b, [r])
