"""``core.worker``: a worker process's loop unit (``proc.loop@1``, stage2-contract.md §2, §4).

It reads the scheduler's plans (the reader ``plans``), runs each through ``executor@1`` and
writes the answer to the scheduler's reader ``results.<rank>``; the rank is this process's index
(one process per rank). Everything below the loop (persistent batch, backends, model, sampler) is
the executor's (``core.contracts.execution``).

- At start, whenever the results go to a new reader (a new scheduler process, whose ledger is
  empty), and once the process serves a new identity (a change committed), it sends ``Capacity``
  with that identity: the scheduler plans again after a change only once it has it.
- It takes the next plan while a step runs, two at most (one step of overlap, §4.4). Answers go out
  in plan order: ``Sampled`` with ``StepStats``, ``Resync`` (the executor refused the plan before
  doing anything), or ``StepFailure`` (the executor raised: what ran is unknown). After a failure
  it goes on reading plans; the scheduler decides what to send. Nothing is frozen.
- ``step()`` returns the last step answered: a pause waits for a follower to reach the step its
  leader stopped after.
- Answers are written without waiting; what the reader does not take yet is tried next turn.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

import msgspec

from core.common.ids import ProcId
from core.contracts.execution import Executor, PendingStep
from core.contracts.records import (
    STEP_PLAN,
    STEP_RESULT,
    BatchPlan,
    Capacity,
    Resync,
    RowDelta,
    Sampled,
    StateOps,
    StepFailure,
    StepStats,
)
from core.ipc.records import RecordBuilder, decode_record, encode_record, read_sections

DEPTH = 2
"""Plans taken before the oldest one is answered."""


class Reader(Protocol):
    def read(self, timeout: float | None = None) -> bytes | None: ...


class Writer(Protocol):
    generation: int

    def write(self, data: bytes, timeout: float | None = None) -> bool: ...


@dataclass
class Taken:
    epoch: int
    step: PendingStep | None = None
    """Running; ``None`` once answered (``answer`` holds the record)."""
    answer: bytes | None = None


class WorkerLoop:
    def __init__(self, executor: Executor, plans: Reader, results: Writer, *, rank: int,
                 idle_wait_s: float = 0.01, quiesce_timeout_s: float = 30.0, announce_every_s: float = 1.0,
                 identity: Callable[[], str] = lambda: "", clock: Callable[[], float] = time.monotonic) -> None:
        self.executor, self.plans, self.results, self.rank = executor, plans, results, rank
        self.identity = identity
        self.announced_as: str | None = None
        """The identity of the last ``Capacity`` queued."""
        self.idle_wait_s, self.quiesce_timeout_s, self.clock = idle_wait_s, quiesce_timeout_s, clock
        self.announce_every_s = announce_every_s
        self.last_write = clock()
        """When a record last went out."""
        self.taken: deque[Taken] = deque()
        self.outbox: deque[tuple[int, bytes]] = deque()
        self.epoch: int | None = None
        """The last step answered (written to the scheduler)."""
        self.announced_on: int | None = None
        """The results connection the last ``Capacity`` went over; ``None``: none yet."""
        self.announcing = False
        """A ``Capacity`` is queued."""
        self.failures = 0
        # Not at once: built inside a change's Execute, its stores are a candidate the change may
        # still undo (Codex, #42). It sends Capacity at its first step, after the process resumed,
        # under the identity it serves then; the scheduler waits for that after a commit.

    # ------------------------------------------------------------------ the loop

    def step(self) -> int | None:
        self._announce()
        self._finish_ready()
        self._flush()
        if len(self.taken) < DEPTH:
            raw = self.plans.read(timeout=0 if self.taken or self.outbox else self.idle_wait_s)
            if raw is not None:
                self._take(raw)
                self._finish_ready()
                self._flush()
        return self.epoch

    def quiesce(self) -> int | None:
        """Every step taken has finished and its answer is written, within ``quiesce_timeout_s``
        for the writes (the leader collected the answers before, so they normally went out)."""
        while self.taken:
            self._finish(self.taken[0])
            self._release()
        self.executor.synchronize()
        deadline = self.clock() + self.quiesce_timeout_s
        while self.outbox and self.clock() < deadline:
            self._flush()
            if self.outbox:
                time.sleep(self.idle_wait_s)
        if self.outbox:
            raise TimeoutError(f"answers to steps {[e for e, _ in self.outbox]} not taken by the scheduler "
                               f"in {self.quiesce_timeout_s} s")
        return self.epoch

    # ------------------------------------------------------------------ plans

    def _take(self, raw: bytes) -> None:
        try:
            record = decode_record(raw)
        except Exception as e:  # noqa: BLE001 - nothing to answer to: the epoch is unknown
            self._note(f"plan dropped: {type(e).__name__}: {e}")
            return
        taken = Taken(record.epoch)
        self.taken.append(taken)
        try:
            if record.schema != STEP_PLAN or record.rank != self.rank:
                raise ValueError(f"{record.schema} for rank {record.rank} reached rank {self.rank}")
            parts = read_sections(record, {RowDelta.SECTION: RowDelta, BatchPlan.SECTION: BatchPlan,
                                           StateOps.SECTION: StateOps}).unwrap()
            out = self.executor.submit(record.epoch, cast(RowDelta, parts[RowDelta.SECTION]),
                                       cast(BatchPlan, parts[BatchPlan.SECTION]), cast(StateOps, parts[StateOps.SECTION]))
        except Exception as e:  # noqa: BLE001 - what ran is unknown: say so, and go on
            self._failed(taken, e)
            return
        if isinstance(out, Resync):
            taken.answer = self._record(record.epoch, (Resync.SECTION, out))
        else:
            taken.step = out

    def _finish_ready(self) -> None:
        for taken in self.taken:
            if taken.step is None:
                continue
            try:
                ready = taken.step.ready()
            except Exception as e:  # noqa: BLE001 - asking may report the device work's error (a CUDA event)
                self._failed(taken, e)
                continue
            if ready:
                self._finish(taken)
        self._release()

    def _finish(self, taken: Taken) -> None:
        step = taken.step
        if step is None:
            return
        taken.step = None
        try:
            sampled = step.finish()
            stats = step.stats()
        except Exception as e:  # noqa: BLE001
            self._failed(taken, e)
            return
        taken.answer = self._record(taken.epoch, (Sampled.SECTION, sampled), (StepStats.SECTION, stats))

    def _failed(self, taken: Taken, error: Exception) -> None:
        self.failures += 1
        reason = f"{type(error).__name__}: {error}"
        self._note(f"step {taken.epoch} failed: {reason}")
        taken.step = None
        taken.answer = self._record(taken.epoch, (StepFailure.SECTION, StepFailure(reason)))

    def _release(self) -> None:
        """Answers in plan order: the oldest one goes first."""
        while self.taken and self.taken[0].answer is not None:
            taken = self.taken.popleft()
            assert taken.answer is not None
            self.outbox.append((taken.epoch, taken.answer))

    # ------------------------------------------------------------------ results

    def _announce(self) -> None:
        """``Capacity`` at start and on every new results connection. A connection is found new
        only when a write is tried (``ChannelWriter``), so an idle worker also sends it every
        ``announce_every_s``: a scheduler host restarted while nothing ran waits for it, and
        sends no plan until it has it."""
        if self.announcing:
            return
        moved = self.announced_on != self.results.generation
        idle = not self.taken and not self.outbox and self.clock() - self.last_write >= self.announce_every_s
        identity = self.identity()
        if not moved and not idle and identity == self.announced_as:
            return
        if moved and self.announced_on is not None:
            self._note("results go to a new reader (a new scheduler?): Capacity again")
        capacity = msgspec.structs.replace(self.executor.capacity(), identity=identity)
        self.outbox.appendleft((0, self._record(0, (Capacity.SECTION, capacity))))
        self.announcing, self.announced_as = True, identity

    def _flush(self) -> None:
        while self.outbox:
            epoch, data = self.outbox[0]
            try:
                if not self.results.write(data, timeout=0):
                    break
            except TimeoutError:
                break
            self.outbox.popleft()
            self.last_write = self.clock()
            if epoch > 0:
                self.epoch = max(self.epoch or 0, epoch)
            else:
                self.announcing, self.announced_on = False, self.results.generation

    def _record(self, epoch: int, *sections: tuple[str, msgspec.Struct]) -> bytes:
        builder = RecordBuilder(STEP_RESULT, epoch, self.rank)
        for name, value in sections:
            builder.add(name, value)
        return encode_record(builder.build())

    @staticmethod
    def _note(text: str) -> None:
        print(f"core.worker: {text}", file=sys.stderr, flush=True)


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        scheduler = msgspec.convert(config["worker.scheduler"], ProcId)
        streams = ctx.use("proc.streams@1")
        control = ctx.use("proc.control@1")
        rank = ctx.instance.proc.index
        self.loop = WorkerLoop(cast(Executor, ctx.use("executor@1")), streams.reader("plans", scheduler.host),
                               streams.writer(scheduler, f"results.{rank}"), rank=rank,
                               idle_wait_s=config["worker.idle_wait_s"],
                               quiesce_timeout_s=config["worker.quiesce_timeout_s"],
                               announce_every_s=config["worker.announce_every_s"],
                               identity=lambda: str(control.identity))
        ctx.provide("proc.loop@1", self.loop)
