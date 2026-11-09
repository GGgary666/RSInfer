"""``core.sched``'s ``drain`` (doc/specs/empty-apply.md): a change applies to an empty engine. The
handler the unit registered, driven as the change participant drives it: ``start_drain`` with the
change's mode, ``drained`` at every safe point, ``drain_deadline`` when the change's wait ran out,
then ``commit`` or ``abort``."""

from __future__ import annotations

from typing import Any

import pytest

from tests.units.scheduler.support import Bed, bed, request


def drain(b: Bed) -> Any:
    handler = b.engine.runtime.action_handler("core.sched", "drain")
    assert handler is not None
    return handler


def held_back(b: Bed, rid: int, turns: int = 50) -> bool:
    """``rid`` waits in the stream: the scheduler has not taken it."""
    for _ in range(turns):
        b.turn()
    return b.engine.table.get(rid) is None and not b.client.outputs[rid].tokens


def test_refuse_fails_the_change_while_requests_are_in_the_engine_and_says_how_to_go_on() -> None:
    with bed() as b:
        b.client.send([request(1, [1, 2, 3], 20)])
        b.run_until(lambda: len(b.client.outputs[1].tokens) >= 1)
        with pytest.raises(RuntimeError, match=r"1 running and 0 waiting request.*--wait-empty.*--force"):
            drain(b).start_drain({"mode": "refuse"})
        drain(b).abort()  # the participant undoes the change
        b.run_until(b.finished(1))
        assert b.client.outputs[1].error is None and b.errors() == []


def test_refuse_on_an_empty_engine_holds_new_requests_until_the_change_ends() -> None:
    with bed() as b:
        h = drain(b)
        h.start_drain({"mode": "refuse"})
        assert h.drained()
        b.client.send([request(1, [1, 2, 3], 4)])
        assert held_back(b, 1)  # it will run on what the change made
        h.commit()
        b.run_until(b.finished(1))
        assert b.client.outputs[1].error is None and b.idle()


def test_wait_serves_what_is_in_the_engine_then_takes_no_more_until_the_change_ends() -> None:
    with bed() as b:
        b.client.send([request(1, [1, 2, 3], 8), request(2, [4, 5], 8)])
        b.run_until(lambda: len(b.client.outputs[1].tokens) >= 1)
        h = drain(b)
        h.start_drain({"mode": "wait"})
        assert not h.drained()
        b.run_until(h.drained, what="the engine empties")
        b.run_until(b.finished(1, 2))  # their last tokens may still be on their way to the client
        assert [b.client.outputs[r].error for r in (1, 2)] == [None, None]
        assert [len(b.client.outputs[r].tokens) for r in (1, 2)] == [8, 8] and b.idle()
        b.client.send([request(3, [7], 2)])
        assert held_back(b, 3)
        h.abort()  # the change was undone: requests come in again
        b.run_until(b.finished(3))
        assert b.client.outputs[3].error is None


def test_force_ends_every_request_at_once_and_gives_its_items_back() -> None:
    with bed(delay_s=0.002) as b:
        b.client.send([request(1, [1, 2, 3], 50), request(2, [9] * 40, 5)])
        b.run_until(lambda: len(b.client.outputs[1].tokens) >= 2)
        h = drain(b)
        h.start_drain({"mode": "force"})
        b.run_until(h.drained, what="the steps in flight retire")
        b.run_until(b.finished(1, 2))
        for r in (1, 2):
            assert b.client.outputs[r].finished == "abort"
            assert "apply --force" in (b.client.outputs[r].error or "")
        assert b.idle()
        h.commit()
        b.client.send([request(3, [7], 2)])
        b.run_until(b.finished(3))
        assert b.client.outputs[3].error is None and b.errors() == []


def test_a_wait_that_runs_out_fails_the_change_and_says_what_is_left() -> None:
    with bed() as b:
        b.client.send([request(1, [1, 2, 3], 30)])
        b.run_until(lambda: len(b.client.outputs[1].tokens) >= 1)
        h = drain(b)
        h.start_drain({"mode": "wait"})
        with pytest.raises(TimeoutError, match=r"did not empty in time: the scheduler holds 1 running.*nothing was ended"):
            h.drain_deadline()
        h.abort()
        b.run_until(b.finished(1))
        assert b.client.outputs[1].error is None
