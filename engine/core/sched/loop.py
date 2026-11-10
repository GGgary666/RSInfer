"""The scheduler's step loop: requests in, plans out to every rank, results back, tokens out.

The structure follows mini-sglang (9a91cfa, MIT, see LICENSE.mini-sglang): a prefill batch if one
can be formed, else the decode batch; admission in order, stopping at the first request that does
not fit; requests reserve their prompt plus ``max_tokens`` up front (``PrefillAdder``,
``estimated_len``). The two decisions are policy units (``core.contracts.policy``); whether a
request fits is the ledger's answer (``core.contracts.ledger``).

What is ours (stage2-contract.md §4):

- **Steps in flight.** Up to ``depth`` planned steps have not retired: 1 without overlap, 2 with
  ``sched.overlap`` (§4.4). A decode row whose last run is still in flight takes its input token
  from the worker (``from_previous``); a request may finish in a step whose result has not come
  back, and its token of the next step is dropped (``sched.reqs``). Its items are released only
  once its last step retired.
- **Ranks.** Every rank gets the same plan, its own ``rank`` in the record. A step retires once
  every rank answered it; the tokens are rank 0's, and every rank must have sampled the same.
- **Resync.** A step every rank answered with ``Resync`` did not run. The steps sent after it are
  refused too, so the next plan waits for them and goes out with nothing in flight, as a full
  delta. Ranks that disagree (one ran it, one did not) cannot be put back together: that is a
  failure.
- **Failures.** ``StepFailure``, a step without an answer for ``sched.step_timeout_s``, or ranks
  that disagree: the worker's state is unknown. Then no more steps go out, new requests get an
  error, the requests it holds stay as they are, and ``report()`` says what happened. Nothing is
  frozen: control requests, reloads and config keep working. Until changes can choose how to deal
  with such a state (stage 2's step 5), a new instance of this unit (a reload) ends the requests
  of steps that never came back with an error and carries on with the rest.
- **A replaced worker.** Plans or results of a rank going over a new connection mean a new
  worker process (or a broken connection): what the requests had there is gone. Those requests
  end with an error at once, the steps sent there and not answered are given up, and serving
  goes on; nothing waits for a reload. Which requests: those whose plans went over the old
  connection. The two streams see one replacement at different times, so the second sighting
  is paired with the first, and requests first sent to the new worker are not touched. A step
  already on its way to the new worker is refused there (it waits for a full delta) while the
  other ranks may run it: then it counts as not run. A request with nothing computed starts over
  from position 0 on every rank, which clears its state first (``StateStore.clear``); a request
  further on, whose ranks now differ, ends.
- **Never blocking on a peer.** Plans and tokens are written without waiting; what a reader does
  not take yet stays queued and is tried again next turn.
- **What a row holds.** Admission refuses a request longer than one row of the workers'
  persistent batch (``batch.max_context_len``), and plans use at most ``batch.max_rows`` rows:
  the worker raises on a plan beyond them (stage2-contract.md §4.3). A frontend that takes nothing for
  ``sched.frontend_timeout_s`` loses its requests (they are cancelled), not the scheduler.
- **Changes.** A change applies to an empty engine (doc/specs/empty-apply.md, ``drain.py``): this
  unit says what it holds, ends it all when the change forces it, and once empty takes no
  requests until the change committed or was undone (``held``).
"""

from __future__ import annotations

import sys
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

import msgspec

from core.contracts.columns import Column, ragged
from core.contracts.ledger import Ledger
from core.contracts.policy import Candidate, ComposePolicy, OrderPolicy, StepView, Take
from core.contracts.records import (
    STEP_PLAN,
    STEP_RESULT,
    AddedRow,
    BatchPlan,
    Capacity,
    Resync,
    RowDelta,
    Sampled,
    StateAssign,
    StateOps,
    StepFailure,
)
from core.contracts.requests import ENDED_BY_CHANGE, REQUESTS, TOKENS, Requests, TokenEvents
from core.contracts.serving import ReturnRoute
from core.ipc.records import OpenRecord, RecordBuilder, decode_record, encode_record, read_sections
from core.ipc.stream import StreamAddress

type Answer = Sampled | Resync | StepFailure
FORCED = ENDED_BY_CHANGE
"""The error of a request a change ended at once (doc/specs/empty-apply.md)."""
type Kind = Literal["waiting", "prefilling", "decoding"]


class Reader(Protocol):
    def read(self, timeout: float | None = None) -> bytes | None: ...
    @property
    def generation(self) -> int: ...


class Writer(Protocol):
    generation: int

    def write(self, data: bytes, timeout: float | None = None) -> bool: ...
    def abandon(self) -> None: ...


class Request(Protocol):
    """What this loop uses of an entry of ``sched.reqs@1`` (it sets ``row``)."""

    row: int | None

    @property
    def input(self) -> Any: ...
    @property
    def output_ids(self) -> Sequence[int]: ...
    @property
    def computed(self) -> int: ...
    @property
    def route(self) -> StreamAddress | None: ...
    @property
    def prompt_tokens(self) -> Sequence[int]: ...
    @property
    def reclaimable(self) -> bool: ...
    def finished(self) -> bool: ...


class Table(Protocol):
    """What this loop uses of ``sched.reqs@1``."""

    @property
    def steps(self) -> Mapping[int, Any]: ...
    @property
    def last_submitted(self) -> int: ...
    @property
    def last_retired(self) -> int: ...

    def entries(self) -> Iterable[Request]: ...
    def get(self, request: int) -> Request | None: ...
    def add(self, request: Any, *, route: StreamAddress | None = None) -> Any: ...
    def start_step(self, epoch: int, rows: Mapping[int, int], counts: Mapping[int, int] | None = None) -> None: ...
    def sample(self, epoch: int, identity: str, sampled: Sampled) -> TokenEvents: ...
    def cancel(self, request: int, identity: str) -> TokenEvents: ...
    def retire(self, epoch: int, ran: bool = True) -> list[int]: ...
    def reap(self, request: int) -> Any: ...


@dataclass(frozen=True)
class Settings:
    max_batch: int = 32
    """Rows plans use: ``sched.max_batch``, at most the workers' ``batch.max_rows``."""
    max_context: int = 0
    """Tokens one row of the workers holds (``batch.max_context_len``); 0: no limit."""
    token_budget: int = 2048
    """Prompt tokens per step; sglang's smallest ``chunked_prefill_size``."""
    step_timeout_s: float = 3600.0
    quiesce_timeout_s: float = 30.0
    frontend_timeout_s: float = 5.0
    idle_wait_s: float = 0.01


class Fault(msgspec.Struct, frozen=True):
    step: int
    rank: int | None
    reason: str


class Report(msgspec.Struct, frozen=True):
    """``sched.status@1``: what the scheduler holds, for whoever is about to change the engine."""

    waiting: int
    running: int
    in_flight: list[int]
    """Steps planned and not retired, oldest first."""
    last_retired: int | None
    fault: Fault | None
    policy_error: str | None
    diagnostics: list[str]


class PolicyError(Exception):
    """A policy's decision the loop does not carry out."""


@dataclass
class Flight:
    epoch: int
    rows: dict[int, int]
    """row -> request"""
    counts: dict[int, int]
    samples: frozenset[int]
    """Rows whose chunk ends the prompt or decodes: the rows that sample."""
    since: float
    answers: dict[int, Answer] = field(default_factory=dict[int, Answer])
    void: bool = False
    """Sent after a step the workers refused (``Resync``): they refuse this one too."""
    written: dict[int, int] = field(default_factory=dict[int, int])
    """rank -> the generation of the plan connection it went over."""
    refused_by: set[int] = field(default_factory=set[int])
    """Ranks whose new worker refuses this step: it reached them after a replacement, before the
    full delta. A ``Resync`` from them while other ranks ran it is expected."""


class Outbox:
    """Records for one reader, written in order without waiting. ``since`` is when the oldest
    one was queued; each record has a tag (a plan's epoch)."""

    def __init__(self, writer: Writer) -> None:
        self.writer = writer
        self.queue: deque[tuple[float, bytes, int]] = deque()

    def put(self, data: bytes, now: float, tag: int = 0) -> None:
        self.queue.append((now, data, tag))

    @property
    def since(self) -> float | None:
        return self.queue[0][0] if self.queue else None

    def flush(self) -> list[int]:
        """Write what the reader takes; the tags written. They all went over one connection: a
        connection that breaks stops the flush, and the next one is made by the next write."""
        written: list[int] = []
        while self.queue:
            try:
                if not self.writer.write(self.queue[0][1], timeout=0):
                    break
            except TimeoutError:
                break
            written.append(self.queue.popleft()[2])
        return written

    def drop(self, tags: set[int]) -> None:
        self.queue = deque(e for e in self.queue if e[2] not in tags)


class Scheduler:
    def __init__(self, *, table: Table, ledger: Ledger, order: OrderPolicy, compose: ComposePolicy,
                 requests: Reader, results: Sequence[Reader], plans: Sequence[Writer],
                 tokens_to: Callable[[StreamAddress], Writer], discard: Callable[[StreamAddress], None],
                 identity: Callable[[], str], depth: Callable[[], int] = lambda: 1,
                 settings: Settings = Settings(), clock: Callable[[], float] = time.monotonic) -> None:
        if not plans or len(plans) != len(results):
            raise ValueError("one plan writer and one result reader per rank")
        if settings.max_batch < 1 or settings.token_budget < 1:
            raise ValueError("sched.max_batch and sched.token_budget must be positive")
        self.table, self.ledger, self.order, self.compose = table, ledger, order, compose
        self.requests, self.results = requests, list(results)
        self.plans = [Outbox(w) for w in plans]
        self.tokens_to, self.discard, self.identity = tokens_to, discard, identity
        self.depth, self.settings, self.clock = depth, settings, clock
        self.flights: deque[Flight] = deque()
        self.full = True
        """The next plan resets the workers' persistent batch: it cannot know what an earlier
        instance of this unit sent them."""
        self.removed: list[int] = []
        self.capacity: dict[int, dict[str, int]] = {}
        self.fault: Fault | None = None
        self.policy_error: str | None = None
        self.diagnostics: deque[str] = deque(maxlen=64)
        self.routes: dict[StreamAddress, Outbox] = {}
        # A new instance takes over the connections as they are (they belong to the host).
        self.result_gens = [r.generation for r in self.results]
        self.plan_gens = [w.generation for w in plans]
        self.home: dict[int, dict[int, int]] = {}
        """request -> rank -> the plan connection its first plan to that rank went over."""
        self.expect_results = [False] * len(plans)
        self.expect_plans = [False] * len(plans)
        """One replacement, seen on one stream already: the other stream's change is the same one."""
        self.adopted = False
        self.drain_mode: str | None = None
        """A change is emptying the engine (``drain.py``): ``refuse``, ``wait`` or ``force``."""
        self.held = False
        """Taking no requests: the engine is empty for a change, until it committed or was undone."""
        self.capacity_wanted: set[int] = set()
        """Ranks whose ``Capacity`` under the identity served now has not come since a change committed:
        their stores may have been rebuilt with another size. No step is planned meanwhile."""

    # ------------------------------------------------------------------ the loop

    def step(self) -> int | None:
        before = self.table.last_retired
        if not self.adopted:
            self._adopt()
        busy = self._collect()
        busy = self._ingest() or busy
        self._check_timeout()
        self._check_ledger()
        busy = self._plan() or busy
        self._flush()
        if not busy:
            self._wait()
        return self._epoch() if self.table.last_retired != before else None

    def quiesce(self) -> int | None:
        """Every step sent is answered, every token queued for a frontend written (or its route
        given up). With a failed worker the steps it never answered are not waited for. A step
        without an answer within ``sched.quiesce_timeout_s`` fails the pause: the change is not
        made, and the engine goes on serving."""
        deadline = self.clock() + self.settings.quiesce_timeout_s
        while self.flights and self.fault is None:
            self._flush()
            if not self._collect():
                self._wait_result(min(self.settings.idle_wait_s, max(0.0, deadline - self.clock())))
            self._check_timeout()
            if self.flights and self.fault is None and self.clock() >= deadline:
                head = self.flights[0]
                missing = [r for r in range(len(self.results)) if r not in head.answers]
                raise TimeoutError(f"step {head.epoch} has no answer from rank(s) {missing} after "
                                   f"{self.settings.quiesce_timeout_s} s; the engine goes on serving")
        while self.routes:
            self._flush_routes()
            if self.routes:
                time.sleep(self.settings.idle_wait_s)
        return self._epoch()

    def report(self) -> Report:
        waiting = sum(1 for r in self.table.entries() if r.row is None and not r.finished())
        running = sum(1 for r in self.table.entries() if r.row is not None)
        return Report(waiting, running, [f.epoch for f in self.flights], self._epoch(), self.fault,
                      self.policy_error, list(self.diagnostics))

    def _epoch(self) -> int | None:
        return self.table.last_retired if self.table.last_retired >= 1 else None

    def occupancy(self) -> tuple[int, int, int]:
        """Requests running, requests waiting, steps in flight. A request ended (or cancelled)
        no longer counts, but a step still in flight does: it may write what the request held.
        A failed worker's steps never come back, so they do not count once the loop failed."""
        live = [r for r in self.table.entries() if not r.finished()]
        running = sum(1 for r in live if r.row is not None)
        return running, len(live) - running, 0 if self.fault is not None else len(self.flights)

    def end_all(self, message: str) -> int:
        """End every request with ``message`` as its error; how many."""
        live = [r for r in self.table.entries() if not r.finished()]
        for r in live:
            self._error(r, message)
        return len(live)

    def _note(self, text: str) -> None:
        if not self.diagnostics or self.diagnostics[-1] != text:
            self.diagnostics.append(text)
            print(f"core.sched: {text}", file=sys.stderr, flush=True)

    def _wait(self) -> None:
        if self.flights and self.fault is None:
            self._wait_result(self.settings.idle_wait_s)
            return
        if self.held:
            time.sleep(self.settings.idle_wait_s)
            return
        raw = self.requests.read(timeout=self.settings.idle_wait_s)
        if raw is not None:
            self._take_requests(raw)

    def _wait_result(self, timeout: float) -> None:
        head = self.flights[0] if self.flights else None
        rank = next((r for r in range(len(self.results)) if head is None or r not in head.answers), 0)
        raw = self.results[rank].read(timeout=timeout)
        if raw is not None:
            self._take_result(rank, raw)
            self._retire_ready()

    # ------------------------------------------------------------------ taking over

    def _adopt(self) -> None:
        """Steps the request table knows and this instance did not send were abandoned by an
        earlier instance after a worker failure: their requests end with an error, the rest
        carry on from what retired."""
        self.adopted = True
        for request in self.table.entries():
            if request.row is not None:
                self.home[request.input.request] = dict(enumerate(self.plan_gens))
        for epoch in sorted(self.table.steps):
            rows = dict(cast(Mapping[int, int], self.table.steps[epoch].rows))
            for rid in rows.values():
                request = self.table.get(rid)
                if request is not None and not request.finished() and request.route is not None:
                    self._error(request, f"step {epoch} never came back from the workers; "
                                         "the request's state on the device is unknown")
            self.ledger.retire(epoch, ran=False)
            for rid in self.table.retire(epoch, ran=False):
                self._reap(rid)
            self._note(f"took over: step {epoch} was abandoned, its requests ended")

    # ------------------------------------------------------------------ results

    def _collect(self) -> bool:
        got = False
        for rank, reader in enumerate(self.results):
            for _ in range(8):
                raw = reader.read(timeout=0)
                if raw is None:
                    break
                got = True
                self._take_result(rank, raw)
        if got:
            self._retire_ready()
        return got

    def _take_result(self, rank: int, raw: bytes) -> None:
        reader = self.results[rank]
        if reader.generation != self.result_gens[rank]:
            replaced = self.result_gens[rank] != 0
            self.result_gens[rank] = reader.generation
            if replaced:
                self._results_moved(rank)
        try:
            record = decode_record(raw)
            if record.schema != STEP_RESULT or record.rank != rank:
                raise ValueError(f"{record.schema} for rank {record.rank} on rank {rank}'s results")
            parts = read_sections(record, {}, {Capacity.SECTION: Capacity, Sampled.SECTION: Sampled,
                                               Resync.SECTION: Resync, StepFailure.SECTION: StepFailure}).unwrap()
        except Exception as e:  # noqa: BLE001 - a bad record is dropped, not fatal
            self._note(f"rank {rank}: result dropped: {type(e).__name__}: {e}")
            return
        if Capacity.SECTION in parts:
            capacity = cast(Capacity, parts[Capacity.SECTION])
            self._capacity(rank, capacity.items)
            if capacity.identity == self.identity():
                self.capacity_wanted.discard(rank)
        answer = cast(Answer | None, parts.get(StepFailure.SECTION) or parts.get(Resync.SECTION)
                      or parts.get(Sampled.SECTION))
        if answer is None:
            return
        flight = next((f for f in self.flights if f.epoch == record.epoch), None)
        if flight is None or rank in flight.answers:
            if self.fault is None:
                self._note(f"rank {rank}: answer to step {record.epoch}, which is not waiting for one")
            return
        flight.answers[rank] = answer
        if isinstance(answer, StepFailure):
            self._fail(record.epoch, rank, f"worker failed: {answer.reason}")

    def _capacity(self, rank: int, items: dict[str, int]) -> None:
        self.capacity[rank] = items
        if len(self.capacity) < len(self.results):
            return
        kinds = set.intersection(*(set(c) for c in self.capacity.values()))
        lowest = {k: min(c[k] for c in self.capacity.values()) for k in sorted(kinds)}
        try:
            self.ledger.set_capacity(lowest)
        except ValueError as e:
            self._note(f"capacity {lowest} not taken: {e}")

    def _plans_moved(self, rank: int, gen: int) -> None:
        """Plans to ``rank`` now go over a new connection: to a new worker process, or the old
        one's connection broke and records on it may be lost. Requests sent over the old one
        lose what they had there. If results moved first, it is the same replacement, handled
        then: the plan writer was abandoned at once, so no plan went over the old connection
        since, and the plan just written is the full delta for the new worker."""
        if self.expect_plans[rank]:
            self.expect_plans[rank] = False
            return
        self.expect_results[rank] = True
        self._lose(rank, lambda first: first < gen, f"plans to rank {rank} go over a new connection "
                                                   "(a new worker process?)")

    def _results_moved(self, rank: int) -> None:
        """Results of ``rank`` come from a new producer. If plans moved first, it is the same
        replacement. Otherwise the plan writer may still be connected to the old reader: it looks
        the reader up again at the next write, and that move is paired with this one."""
        if self.expect_results[rank]:
            self.expect_results[rank] = False
            return
        self.expect_plans[rank] = True
        self.plans[rank].writer.abandon()
        self._lose(rank, lambda first: True, f"results of rank {rank} come from a new producer "
                                            "(a new worker process?)")

    def _lose(self, rank: int, old: Callable[[int], bool], why: str) -> None:
        """End the requests whose plans went to ``rank`` over the old connection, and give up the
        steps sent there that it did not answer (their answers may never come)."""
        lost = {rid for rid, first in self.home.items() if rank in first and old(first[rank])}
        void = [f for f in self.flights if rank in f.written and old(f.written[rank]) and rank not in f.answers]
        for f in void:
            lost |= set(f.rows.values())
        self.full = True
        for f in self.flights:  # not given up: on their way to the new worker, which refuses them
            if f not in void and rank not in f.answers:
                f.refused_by.add(rank)
        if not lost:
            self._note(f"{why}; no request was on it")
            return
        for rid in sorted(lost):
            request = self.table.get(rid)
            if request is not None and not request.finished():
                self._error(request, f"{why}: what the request had there is gone")
        tags = {f.epoch for f in void}
        for f in void:
            self.flights.remove(f)
            self.ledger.retire(f.epoch, ran=False)
            for rid in self.table.retire(f.epoch, ran=False):
                self._reap(rid)
        for outbox in self.plans:
            outbox.drop(tags)
        self._note(f"{why}: {len(lost)} request(s) ended" + (f", step(s) {sorted(tags)} given up" if tags else ""))

    def _retire_ready(self) -> None:
        while self.fault is None and self.flights and len(self.flights[0].answers) == len(self.results):
            flight = self.flights.popleft()
            try:
                self._retire(flight)
            except Exception as e:  # noqa: BLE001 - half-retired: the state is unknown
                self._fail(flight.epoch, None, f"retiring step {flight.epoch} failed: {type(e).__name__}: {e}")

    def _retire(self, flight: Flight) -> None:
        answers = [flight.answers[r] for r in range(len(self.results))]
        if all(isinstance(a, Resync) for a in answers):
            self.ledger.retire(flight.epoch, ran=False)
            ready = self.table.retire(flight.epoch, ran=False)
            self.full = True
            for later in self.flights:
                later.void = True
            self._note(f"step {flight.epoch} not run: {cast(Resync, answers[0]).reason or 'Resync'}")
            for rid in ready:
                self._reap(rid)
            return
        refused = {r for r, a in enumerate(answers) if isinstance(a, Resync)}
        if refused and refused <= flight.refused_by and all(isinstance(a, (Resync, Sampled)) for a in answers):
            self._ran_on_some(flight, refused)
            return
        if flight.void:
            self.flights.appendleft(flight)
            self._fail(flight.epoch, None, f"step {flight.epoch} ran after the workers refused the step "
                                           "before it, without a full delta")
            return
        if not all(isinstance(a, Sampled) for a in answers):
            kinds = {r: type(a).__name__ for r, a in enumerate(answers)}
            self.flights.appendleft(flight)
            self._fail(flight.epoch, None, f"ranks disagree whether step {flight.epoch} ran: {kinds}")
            return
        sampled = cast(Sampled, answers[0])
        if any(msgspec.msgpack.encode(a) != msgspec.msgpack.encode(sampled) for a in answers[1:]):
            self.flights.appendleft(flight)
            self._fail(flight.epoch, None, f"ranks sampled different tokens in step {flight.epoch}")
            return
        kept = self._kept(flight, sampled)
        if kept is None:
            self.flights.appendleft(flight)
            self._fail(flight.epoch, 0, f"step {flight.epoch} sampled rows "
                                        f"{sorted(int(x) for x in sampled.rows.values())}, not its rows "
                                        f"{sorted(flight.rows)}")
            return
        events = self.table.sample(flight.epoch, self.identity(), kept)
        self.ledger.retire(flight.epoch)
        ready = self.table.retire(flight.epoch)
        self._emit(events, flight.epoch)
        for rid in ready:
            self._reap(rid)

    def _ran_on_some(self, flight: Flight, refused: set[int]) -> None:
        """A replaced rank's new worker refused ``flight`` and the others ran it. It counts as not
        run. A request with nothing computed starts over from position 0 on every rank, and
        starting from 0 clears its state first. A request further on now has a different state on
        different ranks: it ends."""
        self.ledger.retire(flight.epoch, ran=False)
        ready = self.table.retire(flight.epoch, ran=False)
        for rid in flight.rows.values():
            request = self.table.get(rid)
            if request is not None and not request.finished() and request.computed > 0:
                self._error(request, f"rank(s) {sorted(refused)} were replaced and refused step {flight.epoch}, "
                                     "which the others ran: the request's state differs between ranks")
        self.full = True
        for later in self.flights:
            later.void = True
        self._note(f"step {flight.epoch} not run: refused by the new worker(s) of rank(s) {sorted(refused)}")
        for rid in ready:
            self._reap(rid)

    @staticmethod
    def _kept(flight: Flight, sampled: Sampled) -> Sampled | None:
        """The tokens of the rows that sample: every row of the plan answers, and a row whose
        chunk did not end its prompt has its token dropped (``Sampled``). ``None``: the rows are
        not the plan's."""
        rows = [int(x) for x in sampled.rows.values()]
        counts = [int(x) for x in sampled.counts.values()]
        tokens = [int(x) for x in sampled.tokens.values()]
        if sorted(rows) != sorted(flight.rows) or len(counts) != len(rows) or sum(counts) != len(tokens):
            return None
        keep_rows: list[int] = []
        keep_counts: list[int] = []
        keep_tokens: list[int] = []
        offset = 0
        for row, count in zip(rows, counts, strict=True):
            if row in flight.samples:
                keep_rows.append(row)
                keep_counts.append(count)
                keep_tokens.extend(tokens[offset:offset + count])
            offset += count
        return Sampled(Column.of("i32", keep_rows), Column.of("i32", keep_counts), Column.of("i64", keep_tokens))

    def _check_timeout(self) -> None:
        if self.fault is not None or not self.flights:
            return
        head = self.flights[0]
        if self.clock() - head.since > self.settings.step_timeout_s:
            missing = [r for r in range(len(self.results)) if r not in head.answers]
            self._fail(head.epoch, missing[0] if missing else None,
                       f"no answer to step {head.epoch} from rank(s) {missing} after {self.settings.step_timeout_s} s")

    def _check_ledger(self) -> None:
        """A ledger without capacity while requests hold rows was rebuilt under them (a change
        reloaded ``core.state.alloc``): what they hold is unknown, and admitting anything would
        give their items away again. Until changes refuse that (stage 2's step 5), say so."""
        if self.ledger.ready or self.fault is not None:
            return
        held = sum(1 for r in self.table.entries() if r.row is not None)
        if held:
            self._fail(self.table.last_submitted, None, f"the ledger has no capacity while {held} request(s) hold "
                                                        "rows: it was rebuilt under them")
        elif len(self.capacity) == len(self.results):
            self._capacity(0, self.capacity[0])  # nothing held: what the workers said still holds

    def _fail(self, step: int, rank: int | None, reason: str) -> None:
        if self.fault is not None:
            return
        self.fault = Fault(step, rank, reason)
        for outbox in self.plans:
            outbox.queue.clear()  # nothing more goes to a worker in an unknown state
        self._note(f"stopped at step {step}: {reason}. New requests get an error; the requests "
                   "held stay until the engine is changed")
        for request in list(self.table.entries()):
            if request.row is None and not request.finished():
                self._error(request, f"the engine stopped serving at step {step}: {reason}")

    # ------------------------------------------------------------------ frontend

    def _ingest(self) -> bool:
        if self.held:  # what comes now waits in the stream: it runs on what the change makes
            return False
        got = False
        for _ in range(32):
            raw = self.requests.read(timeout=0)
            if raw is None:
                break
            got = True
            self._take_requests(raw)
        return got

    def _take_requests(self, raw: bytes) -> None:
        try:
            record = decode_record(raw)
            if record.schema != REQUESTS:
                raise ValueError(f"{record.schema} on the requests stream")
            parts = read_sections(record, {Requests.SECTION: Requests, ReturnRoute.SECTION: ReturnRoute}).unwrap()
        except Exception as e:  # noqa: BLE001 - a bad record is dropped, never fatal
            self._note(f"requests record dropped: {type(e).__name__}: {e}")
            return
        requests = cast(Requests, parts[Requests.SECTION])
        route = cast(ReturnRoute, parts[ReturnRoute.SECTION]).tokens
        for new in requests.new:
            if self.drain_mode == "force":
                self._refuse(route, new.request, FORCED)
                continue
            if self.fault is not None:
                self._refuse(route, new.request, f"the engine stopped serving at step {self.fault.step}: "
                                                 f"{self.fault.reason}")
                continue
            added = self.table.add(new, route=route)
            if not added.ok:
                self._refuse(route, new.request, "; ".join(d.reason for d in added.diagnostics))
                continue
            if self.ledger.ready and (why := self._too_long(len(new.prompt) + new.max_tokens)):
                self._error(added.value, why)
        for abort in requests.abort:
            request = self.table.get(abort.request)
            if request is None or request.route != route:
                continue
            self._emit(self.table.cancel(abort.request, self.identity()), self.table.last_submitted)
            if request.reclaimable:
                self._reap(abort.request)

    def _too_long(self, extent: int) -> str:
        """Why a request of ``extent`` tokens (prompt plus max_tokens) can never run, or ``""``."""
        if not self.ledger.fits(extent):
            return f"{extent} tokens (prompt plus max_tokens) need more state than the engine has"
        if self.settings.max_context and extent > self.settings.max_context:
            return f"{extent} tokens (prompt plus max_tokens); a row of the workers holds {self.settings.max_context}"
        return ""

    def _refuse(self, route: StreamAddress, rid: int, message: str) -> None:
        events = TokenEvents(self.identity(), Column.of("i64", [rid]), Column.of("i32", [0, 0]),
                             Column.of("i64", []), {rid: "abort"}, {rid: message})
        self._send(route, events, max(0, self.table.last_submitted))

    def _error(self, request: Request, message: str) -> None:
        """End a live request with an error the client sees."""
        rid = request.input.request
        self.table.cancel(rid, self.identity())
        if request.route is not None:
            self._refuse(request.route, rid, message)
        if request.reclaimable:
            self._reap(rid)

    def _emit(self, events: TokenEvents, epoch: int) -> None:
        ids = [int(x) for x in events.requests.values()]
        if not ids:
            return
        offsets = [int(x) for x in events.offsets.values()]
        tokens = [int(x) for x in events.tokens.values()]
        grouped: dict[StreamAddress, dict[int, list[int]]] = {}
        for i, rid in enumerate(ids):
            request = self.table.get(rid)
            if request is not None and request.route is not None:
                grouped.setdefault(request.route, {})[rid] = tokens[offsets[i]:offsets[i + 1]]
        for route, outputs in grouped.items():
            starts, flat = ragged("i64", outputs.values())
            self._send(route, TokenEvents(events.identity, Column.of("i64", outputs), starts, flat,
                                          {r: f for r, f in events.finished.items() if r in outputs},
                                          {r: e for r, e in events.errors.items() if r in outputs}), epoch)

    def _send(self, route: StreamAddress, events: TokenEvents, epoch: int) -> None:
        outbox = self.routes.get(route)
        if outbox is None:
            outbox = self.routes[route] = Outbox(self.tokens_to(route))
        record = RecordBuilder(TOKENS, epoch).add(TokenEvents.SECTION, events).build()
        outbox.put(encode_record(record), self.clock())

    def _flush_routes(self) -> None:
        for route, outbox in list(self.routes.items()):
            try:
                outbox.flush()
            except Exception as e:  # noqa: BLE001 - only this route is lost
                self._drop(route, f"{type(e).__name__}: {e}")
                continue
            since = outbox.since
            if since is None:
                del self.routes[route]
            elif self.clock() - since > self.settings.frontend_timeout_s:
                self._drop(route, f"took nothing for {self.settings.frontend_timeout_s} s")

    def _drop(self, route: StreamAddress, why: str) -> None:
        """A frontend that does not take its tokens: its requests are cancelled. A later request
        from the same address is served again, over a new connection."""
        self.routes.pop(route, None)
        self.discard(route)
        n = 0
        for request in list(self.table.entries()):
            if request.route == route and not request.finished():
                n += 1
                self.table.cancel(request.input.request, self.identity())
                if request.reclaimable:
                    self._reap(request.input.request)
        self._note(f"frontend {route} dropped ({why}); {n} request(s) cancelled")

    def _reap(self, rid: int) -> None:
        request = self.table.get(rid)
        if request is None or not request.reclaimable:
            return
        self.ledger.release(rid, self.table.last_retired)
        if request.row is not None:
            self.removed.append(request.row)
        self.home.pop(rid, None)
        self.table.reap(rid)

    # ------------------------------------------------------------------ planning

    def _flush(self) -> None:
        for rank, outbox in enumerate(self.plans):
            written: list[int] = []
            try:
                written = outbox.flush()
            except Exception as e:  # noqa: BLE001
                self._fail(self.flights[0].epoch if self.flights else self.table.last_submitted, rank,
                           f"writing a plan to rank {rank} failed: {type(e).__name__}: {e}")
            gen = outbox.writer.generation
            if gen != self.plan_gens[rank]:
                moved = self.plan_gens[rank] != 0
                self.plan_gens[rank] = gen
                if moved:  # before the plans just written are counted: they went to the new reader
                    self._plans_moved(rank, gen)
            flights = {f.epoch: f for f in self.flights}
            for epoch in written:
                flight = flights.get(epoch)
                if flight is not None:
                    flight.written[rank] = gen
                    for rid in flight.rows.values():
                        self.home.setdefault(rid, {}).setdefault(rank, gen)
        self._flush_routes()

    def _plan(self) -> bool:
        if self.fault is not None or not self.ledger.ready or len(self.flights) >= max(1, self.depth()):
            return False
        if self.capacity_wanted:
            return False  # after a change: plan with the workers' capacity of now, not an older one
        if self.full and self.flights:
            return False  # a full delta goes out once every step sent before it is answered
        try:
            planned = self._plan_step()
        except PolicyError as e:
            if self.policy_error != str(e):
                self.policy_error = str(e)
                self._note(f"policy decision not carried out: {e}")
            return False
        except Exception as e:  # noqa: BLE001 - this step is not sent; the next turn tries again
            self._note(f"planning failed: {type(e).__name__}: {e}")
            return False
        self.policy_error = None
        return planned

    def _in_flight(self) -> tuple[dict[int, int], dict[int, int]]:
        tokens: dict[int, int] = {}
        samples: dict[int, int] = {}
        for f in self.flights:
            for row, rid in f.rows.items():
                tokens[rid] = tokens.get(rid, 0) + f.counts[row]
                if row in f.samples:
                    samples[rid] = samples.get(rid, 0) + 1
        return tokens, samples

    def _view(self, tokens: Mapping[int, int], samples: Mapping[int, int]
              ) -> tuple[dict[int, tuple[Kind, Request, int]], list[Candidate], list[Candidate], list[Candidate], int]:
        kinds: dict[int, tuple[Kind, Request, int]] = {}
        waiting: list[Candidate] = []
        prefilling: list[Candidate] = []
        decoding: list[Candidate] = []
        used: set[int] = set()
        for arrival, request in enumerate(self.table.entries()):
            if request.row is not None:
                used.add(request.row)
            if request.finished():
                continue
            rid = request.input.request
            prompt = len(request.prompt_tokens)
            planned = request.computed + tokens.get(rid, 0) if request.row is not None else 0
            output = len(request.output_ids) + samples.get(rid, 0)
            c = Candidate(rid, arrival, prompt, min(planned, prompt), output, request.input.max_tokens)
            if request.row is None:
                kind: Kind = "waiting"
                waiting.append(c)
            elif planned < prompt:
                kind = "prefilling"
                prefilling.append(c)
            elif output < request.input.max_tokens:
                kind = "decoding"
                decoding.append(c)
            else:
                continue  # its last token is being sampled
            kinds[rid] = kind, request, planned
        return kinds, waiting, prefilling, decoding, max(0, self.settings.max_batch - len(used))

    def _decide(self, waiting: list[Candidate], prefilling: list[Candidate], decoding: list[Candidate],
                free_rows: int) -> list[Take]:
        if waiting:
            ordered = list(self.order.order(waiting))
            known = {c.request: c for c in waiting}
            if len(set(ordered)) != len(ordered) or not set(ordered) <= set(known):
                raise PolicyError(f"order {ordered} is not a list of distinct waiting requests")
            waiting = [known[r] for r in ordered]
        view = StepView(waiting, prefilling, decoding, self.settings.token_budget, free_rows)
        takes = list(self.compose.compose(view))
        allowed = {c.request: (kind, c) for kind, cs in (("waiting", waiting), ("prefilling", prefilling),
                                                         ("decoding", decoding)) for c in cs}
        seen: set[int] = set()
        prompt_tokens = new_rows = 0
        for t in takes:
            if t.request in seen or t.request not in allowed:
                raise PolicyError(f"{t} names a request twice, or one that cannot run now")
            seen.add(t.request)
            kind, c = allowed[t.request]
            if kind == "decoding":
                if t.tokens != 1:
                    raise PolicyError(f"{t}: a decoding request takes exactly 1 token")
                continue
            if not 1 <= t.tokens <= c.prompt - c.planned:
                raise PolicyError(f"{t}: {c.prompt - c.planned} prompt tokens are left")
            prompt_tokens += t.tokens
            new_rows += kind == "waiting"
        if prompt_tokens > self.settings.token_budget or new_rows > free_rows:
            raise PolicyError(f"{prompt_tokens} prompt tokens and {new_rows} new rows; the budget is "
                              f"{self.settings.token_budget} tokens and {free_rows} rows")
        return takes

    def _plan_step(self) -> bool:
        tokens, samples = self._in_flight()
        kinds, waiting, prefilling, decoding, free_rows = self._view(tokens, samples)
        if not kinds:
            return False
        epoch = max(self.table.last_submitted, 0) + 1
        used = {r.row for r in self.table.entries() if r.row is not None}
        free = [row for row in range(self.settings.max_batch) if row not in used]
        takes = self._decide(waiting, prefilling, decoding, free_rows)
        step = _Step(epoch)
        try:
            refused = self._carry_out(step, takes, kinds, tokens, free)
            if not step.rows and refused:
                takes = self._decide([], prefilling, decoding, free_rows)
                self._carry_out(step, takes, kinds, tokens, free)
            if not step.rows:
                return False
            self._send_plan(step)
        except BaseException:  # nothing of this step went out: give back what it took
            for rid in {*step.rows.values(), *(r.input.request for r in step.admitted)}:
                self.ledger.cancel(rid, epoch)
            for request in step.admitted:
                request.row = None
            raise
        return True

    def _carry_out(self, step: _Step, takes: list[Take], kinds: Mapping[int, tuple[Kind, Request, int]],
                   in_flight: Mapping[int, int], free: list[int]) -> bool:
        """Admit and reserve for each take, in order; after the first waiting request that does
        not fit, no later waiting request is admitted. Returns whether one did not fit."""
        refused = False
        for take in takes:
            kind, request, planned = kinds[take.request]
            rid = take.request
            if kind == "waiting":
                if refused:
                    continue
                extent = len(request.prompt_tokens) + request.input.max_tokens
                if why := self._too_long(extent):
                    self._error(request, why)
                    continue
                got = self.ledger.admit(rid, extent, step.epoch)
                if got is None:
                    refused = True
                    continue
                request.row = free.pop(0)
                step.admitted.append(request)
                step.added.append(AddedRow(request.row, rid, request.input.sampling))
                step.assign_all(request.row, got)
            row = request.row
            assert row is not None
            extra = self.ledger.reserve(rid, planned + take.tokens, step.epoch)
            if extra is None:  # what step.epoch took for it is back in the ledger
                if kind == "waiting":
                    step.unadmit(request, free)
                continue
            step.assign_all(row, extra)
            step.rows[row], step.counts[row] = rid, take.tokens
            step.positions.append(planned)
            if kind == "decoding":
                previous = rid in in_flight  # its last run is still in flight: the worker has the token
                step.from_previous.append(int(previous))
                if not previous:
                    step.tokens.append(request.output_ids[-1])
                step.decodes += 1
            else:
                step.from_previous.append(0)
                step.tokens.extend(request.prompt_tokens[planned:planned + take.tokens])
                step.prefills += 1
            if planned + take.tokens >= len(request.prompt_tokens):
                step.samples.add(row)
        return refused

    def _send_plan(self, step: _Step) -> None:
        if self.full:
            step.added = []
            step.assign = {}
            for request in self.table.entries():
                if request.row is not None and not request.finished():
                    step.added.append(AddedRow(request.row, request.input.request, request.input.sampling))
                    step.assign_all(request.row, self.ledger.items(request.input.request))
        phase: Literal["prefill", "decode", "mixed"] = (
            "decode" if not step.prefills else "prefill" if not step.decodes else "mixed")
        batch = BatchPlan(Column.of("i32", step.rows), Column.of("i32", step.counts.values()),
                          Column.of("i32", step.positions), Column.of("i64", step.tokens),
                          Column.of("u8", step.from_previous), phase, 0)
        state = StateOps([StateAssign(kind, Column.of("i32", per_row), *ragged("i32", per_row.values()))
                          for kind, per_row in step.assign.items()])
        delta = RowDelta(step.added, Column.of("i32", [] if self.full else self.removed), self.full)
        record = (RecordBuilder(STEP_PLAN, step.epoch).add(RowDelta.SECTION, delta)
                  .add(BatchPlan.SECTION, batch).add(StateOps.SECTION, state).build())
        self.table.start_step(step.epoch, step.rows, step.counts)
        now = self.clock()
        for rank, outbox in enumerate(self.plans):
            outbox.put(encode_record(OpenRecord(record.schema, record.epoch, record.sections, rank)), now, step.epoch)
        self.flights.append(Flight(step.epoch, dict(step.rows), dict(step.counts), frozenset(step.samples), now))
        self.full = False
        self.removed.clear()


@dataclass
class _Step:
    epoch: int
    rows: dict[int, int] = field(default_factory=dict[int, int])
    counts: dict[int, int] = field(default_factory=dict[int, int])
    positions: list[int] = field(default_factory=list[int])
    tokens: list[int] = field(default_factory=list[int])
    from_previous: list[int] = field(default_factory=list[int])
    samples: set[int] = field(default_factory=set[int])
    added: list[AddedRow] = field(default_factory=list[AddedRow])
    assign: dict[str, dict[int, list[int]]] = field(default_factory=dict[str, dict[int, list[int]]])
    admitted: list[Request] = field(default_factory=list[Request])
    prefills: int = 0
    decodes: int = 0

    def assign_all(self, row: int, items: Mapping[str, list[int]]) -> None:
        for kind, ids in items.items():
            if ids:
                self.assign.setdefault(kind, {}).setdefault(row, []).extend(ids)

    def unadmit(self, request: Request, free: list[int]) -> None:
        row = request.row
        assert row is not None
        request.row = None
        self.admitted.remove(request)
        self.added = [a for a in self.added if a.row != row]
        for per_row in self.assign.values():
            per_row.pop(row, None)
        free.insert(0, row)
