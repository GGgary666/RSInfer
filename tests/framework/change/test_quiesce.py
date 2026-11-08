"""The loop finishes the steps it has started before its process stops stepping (change.md 4.2
step 4, unit-runtime.md 6.2 step 1): a pause, an undo that leaves the process stuck, a shutdown."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Protocol, cast

import msgspec
import pytest

from core.change.protocol import Abort, Pause, Paused, Prepare, Resumed
from core.change.types import ProcActions
from core.common.ids import Identity
from core.common.result import Failed, Ok, Result
from core.ipc.rpc import Deferred
from core.supervisor.protocol import ProcSpec
from tests.framework.change.harness import Capture
from tests.framework.support import PROC, events, spec, write_tree
from unit.runtime import ProcessPlan
from unit.runtime.host import EXIT_QUIESCE_FAILED, LOOP_SERVICE, ProcessHost

LOOP = '''
from tests.framework.support import events


class Loop:
    """Overlap scheduling in miniature: each step starts the next epoch and finishes the one
    before it, so one epoch is always in flight when step() returns."""

    def __init__(self, ctx, config):
        self.in_flight = []
        self.finished = []
        self.next = 1
        self.fail_quiesce = False
        ctx.provide("proc.loop@1", self)
        ctx.effect(lambda: (lambda: events.append(("unload", list(self.in_flight)))), "log")

    def step(self):
        self.in_flight.append(self.next)
        self.next += 1
        if len(self.in_flight) < 2:
            return None
        done = self.in_flight.pop(0)
        self.finished.append(done)
        return done

    def quiesce(self):
        events.append(("quiesce", list(self.in_flight)))
        if self.fail_quiesce:
            raise RuntimeError("device lost")
        while self.in_flight:
            self.finished.append(self.in_flight.pop(0))
        return self.finished[-1] if self.finished else None


class Follower(Loop):
    """A worker under overlap scheduling: each plan from the scheduler starts that epoch and
    finishes the one before; without a new plan, step() returns at once."""

    def __init__(self, ctx, config):
        super().__init__(ctx, config)
        self.plans = []

    def step(self):
        if not self.plans:
            return None
        self.in_flight.append(self.plans.pop(0))
        if len(self.in_flight) < 2:
            return None
        done = self.in_flight.pop(0)
        self.finished.append(done)
        return done
'''


class LoopState(Protocol):
    in_flight: list[int]
    finished: list[int]
    fail_quiesce: bool
    plans: list[int]


class Proc:
    def __init__(self, tmp: Path, pkg: str, monkeypatch: pytest.MonkeyPatch, cls: str = "Loop") -> None:
        self.unit = spec(f"{pkg}.loop", f"{pkg}.loop:{cls}", provides=[LOOP_SERVICE])
        code = write_tree(tmp / "v1" / "engine", {f"{pkg}/loop.py": LOOP})
        plan = ProcessPlan(PROC, Identity("v1"), [self.unit], code_root=str(code))
        self.host = ProcessHost(ProcSpec(PROC, Identity("v1"), plan, "unused", "127.0.0.1"))
        monkeypatch.setattr(self.host.rpc, "defer", Capture)
        self.host.runtime.load_process_plan(plan).unwrap()
        self.host.paused = False
        self.closed = False

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.host.shutdown()

    @property
    def loop(self) -> LoopState:
        return cast(LoopState, self.host.runtime.instance(self.unit.id))

    def turn(self) -> None:
        """One turn of ``ProcessHost.run``: a safe point, then a step unless paused."""
        self.host.turn()

    def ask(self, msg: msgspec.Struct) -> Capture:
        """Deliver a request whose reply is deferred; the reply lands in ``.result`` later."""
        result = self.host.handlers[type(msg)](self.host.registry.wrap(msg, self.host.me), msg)
        assert isinstance(result, Capture)
        return result

    def send(self, msg: msgspec.Struct) -> Result[msgspec.Struct]:
        result = self.host.handlers[type(msg)](self.host.registry.wrap(msg, self.host.me), msg)
        if isinstance(result, Capture):
            for _ in range(10):
                if result.result is not None:
                    break
                self.turn()
            assert result.result is not None
            return result.result
        assert not isinstance(result, Deferred)
        return result

    def prepare(self, cid: int) -> None:
        assert isinstance(self.send(Prepare(cid, Identity("v1"), ProcActions(PROC), 60.0)), Ok)


@pytest.fixture
def proc(tmp_path: Path, pkg: str, monkeypatch: pytest.MonkeyPatch) -> Generator[Proc]:
    p = Proc(tmp_path, pkg, monkeypatch)
    try:
        yield p
    finally:
        p.close()


@pytest.fixture
def follower(tmp_path: Path, pkg: str, monkeypatch: pytest.MonkeyPatch) -> Generator[Proc]:
    p = Proc(tmp_path, pkg, monkeypatch, cls="Follower")
    try:
        yield p
    finally:
        p.close()


def test_pause_finishes_the_steps_the_loop_started_and_reports_their_epoch(proc: Proc) -> None:
    for _ in range(3):
        proc.turn()
    assert proc.loop.in_flight == [3] and proc.host.epoch == 2
    proc.prepare(1)
    paused = proc.send(Pause(1, None)).unwrap()
    assert paused == Paused(1, 3)
    assert proc.loop.in_flight == [] and proc.host.paused
    assert ("quiesce", [3]) in events
    proc.turn()
    assert proc.loop.in_flight == [], "no step while paused"


def test_a_loop_that_cannot_finish_its_steps_fails_the_pause_and_keeps_serving(proc: Proc) -> None:
    proc.turn()
    proc.prepare(1)
    proc.loop.fail_quiesce = True
    r = proc.send(Pause(1, None))
    assert isinstance(r, Failed) and r.diagnostics[0].code == "host.quiesce_failed"
    assert not proc.host.paused
    resumed = proc.send(Abort(1)).unwrap()
    assert isinstance(resumed, Resumed) and resumed.rolled_back
    proc.turn()
    assert proc.loop.finished == [1], "the old version keeps serving"


def test_a_follower_whose_last_step_only_quiesce_finishes_pauses_at_the_agreed_epoch(follower: Proc) -> None:
    follower.loop.plans += [1, 2, 3]
    for _ in range(3):
        follower.turn()
    assert follower.host.epoch == 2 and follower.loop.in_flight == [3]
    follower.prepare(1)
    # The leader stopped after epoch 3 and sends no step 4: only quiesce() finishes epoch 3.
    assert follower.send(Pause(1, 3)).unwrap() == Paused(1, 3)
    assert follower.loop.in_flight == [] and follower.host.paused


def test_a_pause_before_the_first_finished_step_waits_for_the_agreed_epoch(follower: Proc) -> None:
    follower.loop.plans.append(1)
    follower.turn()
    assert follower.host.epoch is None and follower.loop.in_flight == [1]
    follower.prepare(1)
    reply = follower.ask(Pause(1, 3))
    for _ in range(3):
        follower.turn()
    assert reply.result is None and not follower.host.paused, "epochs 2 and 3 have not run yet"
    follower.loop.plans += [2, 3]
    for _ in range(3):
        follower.turn()
    assert reply.result is not None and reply.result.unwrap() == Paused(1, 3)
    assert follower.loop.finished == [1, 2, 3] and follower.host.paused


def test_shutdown_finishes_the_steps_in_flight_before_unloading(proc: Proc) -> None:
    proc.turn()
    proc.turn()
    proc.close()
    assert events[-2:] == [("quiesce", [2]), ("unload", [])]


def test_a_shutdown_that_cannot_finish_the_steps_unloads_nothing_and_fails(proc: Proc) -> None:
    proc.turn()
    proc.loop.fail_quiesce = True
    proc.host.paused = True  # as a failed undo leaves it: no more steps, the one in flight unsettled
    proc.closed = True
    assert proc.host.shutdown() == EXIT_QUIESCE_FAILED
    assert events[-1] == ("quiesce", [1]) and not [e for e in events if e[0] == "unload"]
    assert proc.host.diagnostics[-1].code == "host.quiesce_failed"
