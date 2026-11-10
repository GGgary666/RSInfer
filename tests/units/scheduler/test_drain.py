"""``core.sched``'s ``drain`` (doc/specs/empty-apply.md): a change applies to an empty engine. The
handler the unit registered, driven as the change participant drives it: ``start_drain`` with the
change's mode, ``drained`` at every safe point, ``drain_deadline`` when the change's wait ran out,
then ``commit`` or ``abort``. The test client stands in for a frontend that closes no door, so the
scheduler is the gate (``gate``) unless a test says there is a frontend in front of it."""

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
        b.client.send([request(1, [1, 2, 3], 200)])
        b.run_until(lambda: len(b.client.outputs[1].tokens) >= 1)
        with pytest.raises(RuntimeError, match=r"1 running and 0 waiting request.*--wait-empty.*--force"):
            drain(b).start_drain({"mode": "refuse", "gate": True})
        drain(b).abort()  # the participant undoes the change
        b.run_until(b.finished(1))
        assert b.client.outputs[1].error is None and b.errors() == []


def test_refuse_on_an_empty_engine_holds_new_requests_until_the_change_ends() -> None:
    with bed() as b:
        h = drain(b)
        h.start_drain({"mode": "refuse", "gate": True})
        assert h.drained()
        b.client.send([request(1, [1, 2, 3], 4)])
        assert held_back(b, 1)  # it will run on what the change made
        h.commit()
        b.new_identity("v2")
        b.run_until(b.finished(1))
        assert b.client.outputs[1].error is None and b.idle()


def test_wait_serves_what_is_in_the_engine_then_takes_no_more_until_the_change_ends() -> None:
    with bed() as b:
        # long enough that they still run when the drain starts, however loaded the machine (Devin, #42);
        # the bed's ledger holds 256 tokens, reserved for prompt plus max_tokens at admission
        b.client.send([request(1, [1, 2, 3], 100), request(2, [4, 5], 100)])
        b.run_until(lambda: len(b.client.outputs[1].tokens) >= 1)
        h = drain(b)
        h.start_drain({"mode": "wait", "gate": True})
        assert not h.drained()
        b.run_until(h.drained, what="the engine empties")
        b.run_until(b.finished(1, 2))  # their last tokens may still be on their way to the client
        assert [b.client.outputs[r].error for r in (1, 2)] == [None, None]
        assert [len(b.client.outputs[r].tokens) for r in (1, 2)] == [100, 100] and b.idle()
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
        h.start_drain({"mode": "force", "gate": True})
        b.run_until(h.drained, what="the steps in flight retire")
        b.run_until(b.finished(1, 2))
        for r in (1, 2):
            assert b.client.outputs[r].finished == "abort"
            assert "apply --force" in (b.client.outputs[r].error or "")
        assert b.idle()
        h.commit()
        b.new_identity("v2")
        b.client.send([request(3, [7], 2)])
        b.run_until(b.finished(3))
        assert b.client.outputs[3].error is None and b.errors() == []


def test_a_wait_that_runs_out_fails_the_change_and_says_what_is_left() -> None:
    with bed() as b:
        b.client.send([request(1, [1, 2, 3], 200)])
        b.run_until(lambda: len(b.client.outputs[1].tokens) >= 1)
        h = drain(b)
        h.start_drain({"mode": "wait", "gate": True})
        with pytest.raises(TimeoutError, match=r"did not empty in time: the scheduler holds 1 running.*nothing was ended"):
            h.drain_deadline()
        h.abort()
        b.run_until(b.finished(1))
        assert b.client.outputs[1].error is None


def test_behind_a_frontend_a_request_that_comes_after_the_engine_emptied_is_served_or_refused() -> None:
    """The frontend took it before it closed its door, and the change waits for the frontend to see
    it end (Antigravity, #42): the scheduler must not stop reading. Waiting, it runs it; forcing,
    it refuses it with the change's error."""
    with bed() as b:
        h = drain(b)
        h.start_drain({"mode": "wait", "gate": False})
        assert h.drained()
        b.client.send([request(1, [1, 2, 3], 4)])
        b.run_until(b.finished(1))
        assert b.client.outputs[1].error is None and len(b.client.outputs[1].tokens) == 4
        b.run_until(h.drained)
        h.commit()
        b.new_identity("v2")
        h.start_drain({"mode": "force", "gate": False})
        b.client.send([request(2, [4, 5], 4)])
        b.run_until(b.finished(2))
        assert b.client.outputs[2].finished == "abort" and "apply --force" in (b.client.outputs[2].error or "")
        b.run_until(h.drained)
        h.commit()
        b.new_identity("v3")
        assert b.idle() and b.errors() == []


def test_after_a_change_no_step_until_every_rank_says_its_capacity_under_the_new_identity() -> None:
    """The change may have rebuilt a worker's stores smaller, and a ``Capacity`` sent before it may
    still be on its way (Codex, Antigravity, #42): one under the old identity does not count."""
    with bed() as b:
        h = drain(b)
        h.start_drain({"mode": "wait", "gate": True})
        assert h.drained()
        h.commit()  # the identity is still the old one: the workers' Capacity under it does not count
        b.client.send([request(1, [1, 2, 3], 3)])
        for _ in range(50):
            b.turn()
        assert b.engine.loop.capacity_wanted == {0, 1} and not b.client.outputs[1].tokens
        b.new_identity("v2")  # the change took effect: each worker says its capacity under it
        b.run_until(b.finished(1))
        assert not b.engine.loop.capacity_wanted and b.client.outputs[1].error is None


def test_after_a_worker_failure_only_force_empties_the_engine_and_the_change_says_so() -> None:
    """No step goes out after a failure, so what the requests hold never ends by itself: refusing
    or waiting fails at once and points to ``--force`` (Devin, #42)."""
    with bed() as b:
        b.client.send([request(1, [1, 2, 3], 200)])
        b.run_until(lambda: len(b.client.outputs[1].tokens) >= 1)
        b.workers[1].fixture.script[b.engine.table.last_submitted + 2] = "fail"
        b.run_until(lambda: b.engine.loop.fault is not None)
        for mode in ("refuse", "wait"):
            with pytest.raises(RuntimeError, match=r"stopped serving at step .*waiting cannot empty it.*--force"):
                drain(b).start_drain({"mode": mode, "gate": True})
            drain(b).abort()
        h = drain(b)
        h.start_drain({"mode": "force", "gate": True})
        b.run_until(h.drained)
        h.abort()
