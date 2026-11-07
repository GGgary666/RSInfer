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
from unit.runtime.host import LOOP_SERVICE, ProcessHost

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
'''


class LoopState(Protocol):
    in_flight: list[int]
    finished: list[int]
    fail_quiesce: bool


class Proc:
    def __init__(self, tmp: Path, pkg: str, monkeypatch: pytest.MonkeyPatch) -> None:
        self.unit = spec(f"{pkg}.loop", f"{pkg}.loop:Loop", provides=[LOOP_SERVICE])
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
        self.host.safe_point()
        if not self.host.paused:
            epoch = self.host.runtime.lookup(LOOP_SERVICE).step()  # type: ignore[union-attr]
            if epoch is not None:
                self.host.epoch = epoch

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


def test_shutdown_finishes_the_steps_in_flight_before_unloading(proc: Proc) -> None:
    proc.turn()
    proc.turn()
    proc.close()
    assert events[-2:] == [("quiesce", [2]), ("unload", [])]
