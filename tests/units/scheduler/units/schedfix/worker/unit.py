"""A worker for the scheduler's tests (a test fixture, not ``core.worker``).

It keeps a persistent batch the way the worker will (stage2-contract.md §4.2, §4.3), checks
every plan against it and "computes" a token that depends only on the request's seed, the
position and the input token there. So the same requests give the same tokens with overlap or
without, with a ``Resync`` in between or not, however they are batched; and a plan that breaks a
rule is a test failure (``errors``), answered with ``StepFailure``.

What the state units would hold lives with the items, not the rows, so a rebuilt persistent
batch loses rows and the last sampled tokens only (§4.3). A paged item holds the tokens written
to it and whose they are; a slot holds whose it is and how far its state has gone. A step that
continues a row anywhere but where its items end (a position put back too far, or not far
enough after a step that did run) fails, as a GDN state advanced twice would be wrong.

Each plan's answer is due ``delay_s`` after it arrived; plans are read while answers wait, as
the worker reads the next plan while one runs. ``script`` makes a step answer differently:
``resync`` (the persistent batch was rebuilt), ``fail`` (``StepFailure``), ``silent`` (no answer).
"""

from __future__ import annotations

import copy
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import msgspec

from core.common.ids import ProcId
from core.contracts.columns import Column
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
)
from core.ipc.records import OpenRecord, RecordBuilder, decode_record, encode_record, read_sections

VOCAB = 97


def model(seed: int, position: int, token: int) -> int:
    """The token after ``token`` at ``position``."""
    return (seed * 31 + position * 17 + token * 7) % VOCAB


def expected(prompt: Sequence[int], seed: int, max_tokens: int, stop: Sequence[int] = ()) -> list[int]:
    """What a request generates, computed directly; the stop token is included."""
    out: list[int] = []
    position, token = len(prompt) - 1, prompt[-1]
    while len(out) < max_tokens:
        token = model(seed, position, token)
        out.append(token)
        position += 1
        if token in stop:
            break
    return out


class Reader(Protocol):
    def read(self, timeout: float | None = None) -> bytes | None: ...


class Writer(Protocol):
    def write(self, data: bytes, timeout: float | None = None) -> bool: ...


class PlanError(Exception):
    pass


@dataclass
class Row:
    request: int
    seed: int
    items: dict[str, list[int]] = field(default_factory=dict[str, list[int]])
    last: int | None = None
    """The token sampled the last time the row ran."""
    ran: bool = False
    """Ran since it was last added."""


class FixtureWorker:
    def __init__(self, plans: Reader, results: Writer, *, rank: int, capacity: dict[str, int],
                 pages: dict[str, int], max_batch: int, delay_s: float = 0.0, announce_after_s: float = 0.0,
                 identity: Callable[[], str] = lambda: "", clock: Callable[[], float] = time.monotonic) -> None:
        self.plans, self.results, self.rank = plans, results, rank
        self.capacity, self.pages, self.max_batch, self.delay_s, self.clock = capacity, pages, max_batch, delay_s, clock
        self.announce_after_s = announce_after_s
        """How long after it starts the worker sends ``Capacity``; plans are answered meanwhile."""
        self.started = clock()
        self.rows: dict[int, Row] = {}
        self.pages_held: dict[tuple[str, int], tuple[int, list[int]]] = {}
        """(kind, page) -> (request, tokens written to it)."""
        self.slots_held: dict[tuple[str, int], tuple[int, int]] = {}
        """(kind, slot) -> (request, tokens its state has seen)."""
        self.need_full = True
        self.announced = False
        self.identity = identity
        self.announced_as = ""
        """As ``core.worker``: ``Capacity`` says the identity, and goes out again once it changed."""
        self.due: deque[tuple[float, int, bytes]] = deque()
        self.outbox: deque[tuple[int, bytes]] = deque()
        self.script: dict[int, str] = {}
        self.slow: dict[int, float] = {}
        """Epoch -> seconds its answer takes on top of ``delay_s``."""
        self.errors: list[str] = []
        self.received: list[int] = []
        """Epochs of the plans that arrived."""
        self.answered: set[int] = set()
        self.plans_seen: list[tuple[int, BatchPlan, RowDelta]] = []
        self.most_in_flight = 0
        """The most plans this worker had at once without having sent their answers."""
        self.held = False
        """Keep every answer until this is cleared."""
        self.epoch: int | None = None
        """The last step answered: what the loop returns to its host, so that a pause can wait
        for this follower to reach the step its leader stopped after."""

    def turn(self, wait: float = 0.0) -> None:
        now = self.clock()
        if (not self.announced and now - self.started >= self.announce_after_s) or \
                (self.announced and self.identity() != self.announced_as):
            self.announced_as = self.identity()
            record = RecordBuilder(STEP_RESULT, 0, self.rank).add(
                Capacity.SECTION, Capacity(dict(self.capacity), identity=self.announced_as)).build()
            self.outbox.append((0, encode_record(record)))  # the plans' answers keep their order
            self.announced = True
        while self.due and self.due[0][0] <= now and not self.held:
            _, epoch, data = self.due.popleft()
            self.outbox.append((epoch, data))
        while self.outbox:
            try:
                if not self.results.write(self.outbox[0][1], timeout=0):
                    break
            except TimeoutError:
                break
            epoch = self.outbox.popleft()[0]
            self.answered.add(epoch)
            if epoch > 0:
                self.epoch = max(self.epoch or 0, epoch)
        if self.due and not self.held:
            wait = max(0.0, min(wait, self.due[0][0] - now))
        raw = self.plans.read(timeout=wait)
        if raw is not None:
            self._step(decode_record(raw))

    def _answer(self, epoch: int, section: str, value: msgspec.Struct, due: float | None = None) -> None:
        record = RecordBuilder(STEP_RESULT, epoch, self.rank).add(section, value).build()
        if due is None:
            due = self.clock() + self.delay_s + self.slow.pop(epoch, 0.0)
        self.due.append((due, epoch, encode_record(record)))

    def _step(self, record: OpenRecord) -> None:
        epoch = record.epoch
        self.received.append(epoch)
        self.most_in_flight = max(self.most_in_flight, sum(1 for e in self.received if e not in self.answered))
        try:
            if record.schema != STEP_PLAN or record.rank != self.rank:
                raise PlanError(f"{record.schema} for rank {record.rank}")
            parts = read_sections(record, {RowDelta.SECTION: RowDelta, BatchPlan.SECTION: BatchPlan,
                                           StateOps.SECTION: StateOps}).unwrap()
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"step {epoch}: {e}")
            self._answer(epoch, StepFailure.SECTION, StepFailure(str(e)))
            return
        delta = cast(RowDelta, parts[RowDelta.SECTION])
        plan = cast(BatchPlan, parts[BatchPlan.SECTION])
        state = cast(StateOps, parts[StateOps.SECTION])
        self.plans_seen.append((epoch, plan, delta))
        action = self.script.pop(epoch, None)
        if action == "silent":
            return
        if action == "fail":
            self._answer(epoch, StepFailure.SECTION, StepFailure("scripted failure"))
            return
        if self.need_full and not delta.full:
            self._answer(epoch, Resync.SECTION, Resync("the persistent batch needs a full delta"))
            return
        if action == "resync":  # as if worker.batch were rebuilt just now
            self.rows.clear()
            self.need_full = True
            self._answer(epoch, Resync.SECTION, Resync("scripted rebuild"))
            return
        rows = copy.deepcopy(self.rows)
        pages, slots = copy.deepcopy(self.pages_held), dict(self.slots_held)
        try:
            self._apply(rows, delta, state)
            sampled = self._run(rows, plan, pages, slots)
        except PlanError as e:
            self.errors.append(f"step {epoch}: {e}")
            self._answer(epoch, StepFailure.SECTION, StepFailure(str(e)))
            return
        self.rows, self.pages_held, self.slots_held = rows, pages, slots
        self.need_full = False
        self._answer(epoch, Sampled.SECTION, sampled)

    def _apply(self, rows: dict[int, Row], delta: RowDelta, state: StateOps) -> None:
        if delta.full:
            rows.clear()
        for row in delta.removed.values():
            if rows.pop(int(row), None) is None:
                raise PlanError(f"removed row {row} is not there")
        for a in delta.added:
            if a.row in rows or not 0 <= a.row < self.max_batch:
                raise PlanError(f"added row {a.row} is taken or out of range")
            if a.sampling.seed is None:
                raise PlanError(f"request {a.request} has no seed")
            rows[a.row] = Row(a.request, a.sampling.seed)
        for assign in state.assign:
            offsets = [int(x) for x in assign.offsets.values()]
            ids = [int(x) for x in assign.ids.values()]
            for i, row in enumerate(int(x) for x in assign.rows.values()):
                if row not in rows:
                    raise PlanError(f"{assign.kind} items for row {row}, which is not there")
                rows[row].items.setdefault(assign.kind, []).extend(ids[offsets[i]:offsets[i + 1]])
        for kind, cap in self.capacity.items():
            held = [i for r in rows.values() for i in r.items.get(kind, [])]
            if len(held) != len(set(held)) or any(not 0 <= i < cap for i in held):
                raise PlanError(f"{kind} items held twice or out of range: {sorted(held)}")
        if delta.full and any(set(r.items) != set(self.capacity) for r in rows.values()):
            raise PlanError("a full delta must assign every row all of its items")

    def _run(self, rows: dict[int, Row], plan: BatchPlan, pages: dict[tuple[str, int], tuple[int, list[int]]],
             slots: dict[tuple[str, int], tuple[int, int]]) -> Sampled:
        order = [int(x) for x in plan.rows.values()]
        counts = [int(x) for x in plan.num_tokens.values()]
        positions = [int(x) for x in plan.positions.values()]
        previous = [int(x) for x in plan.from_previous.values()]
        tokens = [int(x) for x in plan.tokens.values()]
        if not len(order) == len(counts) == len(positions) == len(previous) or len(set(order)) != len(order):
            raise PlanError("plan columns do not line up")
        if sum(n for n, p in zip(counts, previous, strict=True) if not p) != len(tokens):
            raise PlanError("tokens do not hold the rows' num_tokens")
        taken = 0
        out: list[int] = []
        for row, n, position, prev in zip(order, counts, positions, previous, strict=True):
            r = rows.get(row)
            if r is None:
                raise PlanError(f"row {row} is not in the batch")
            if n < 1:
                raise PlanError(f"row {row} computes {n} tokens")
            if prev:
                if n != 1 or r.last is None or not r.ran:
                    raise PlanError(f"row {row}: from_previous without a token to take")
                given = [r.last]
            else:
                given = tokens[taken:taken + n]
                taken += n
            for kind, page in self.pages.items():
                have = len(r.items.get(kind, []))
                if (page and have * page < position + n) or (not page and have != 1):
                    raise PlanError(f"row {row} holds {have} {kind} items for {position + n} tokens")
                if page:
                    self._write_pages(kind, page, r, position, given, pages)
                else:
                    self._advance_slot(kind, r, position, n, slots)
            r.ran = True
            r.last = model(r.seed, position + n - 1, given[-1])
            out.append(r.last)
        return Sampled(Column.of("i32", order), Column.of("i32", [1] * len(order)), Column.of("i64", out))

    @staticmethod
    def _write_pages(kind: str, page: int, r: Row, position: int, given: list[int],
                     pages: dict[tuple[str, int], tuple[int, list[int]]]) -> None:
        """The row's pages hold exactly its first ``position`` tokens; then ``given`` follows."""
        ids = r.items[kind]
        for i in range(-(-position // page)):
            owner, tokens = pages.get((kind, ids[i]), (-1, []))
            full = min(page, position - i * page)
            if owner != r.request or len(tokens) < full:
                raise PlanError(f"request {r.request} continues at {position}, but its {kind} item {ids[i]} "
                                f"holds {len(tokens)} of its tokens (owner {owner})")
        for offset, token in enumerate(given):
            at = position + offset
            key = (kind, ids[at // page])
            if at % page == 0:
                pages[key] = (r.request, [])
            pages[key][1][at % page:] = [token]

    @staticmethod
    def _advance_slot(kind: str, r: Row, position: int, n: int, slots: dict[tuple[str, int], tuple[int, int]]) -> None:
        """A slot's state has seen exactly the row's first ``position`` tokens."""
        key = (kind, r.items[kind][0])
        owner, seen = slots.get(key, (-1, 0))
        if position == 0:
            owner, seen = r.request, 0
        if (owner, seen) != (r.request, position):
            raise PlanError(f"request {r.request} continues at {position}, but its {kind} state has seen "
                            f"{seen} tokens (owner {owner})")
        slots[key] = (r.request, position + n)


class Unit:
    """The fixture as the loop unit of a worker process; its rank is its process's index (a role
    with one process per rank)."""

    def __init__(self, ctx: Any, config: Any) -> None:
        streams = ctx.use("proc.streams@1")
        control = ctx.use("proc.control@1")
        scheduler = msgspec.convert(config["schedfix.scheduler"], ProcId)
        rank = ctx.instance.proc.index
        self.worker = FixtureWorker(streams.reader("plans", scheduler.host),
                                    streams.writer(scheduler, f"results.{rank}"), rank=rank,
                                    capacity=dict(config["schedfix.capacity"]), pages=dict(config["schedfix.pages"]),
                                    max_batch=config["schedfix.max_batch"], delay_s=config["schedfix.delay_s"],
                                    identity=lambda: str(control.identity))
        self.reported = 0
        ctx.provide("proc.loop@1", self)

    def step(self) -> int | None:
        self.worker.turn(0.005)
        if len(self.worker.errors) > self.reported:  # into the host's diagnostics, once each
            self.reported = len(self.worker.errors)
            raise RuntimeError(self.worker.errors[-1])
        return self.worker.epoch
