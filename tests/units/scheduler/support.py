"""The scheduler host's units in one unit runtime (``Engine``), fixture workers on threads of their
own, and a client standing in for the frontend; each on its own data channels, each on a host of
its own (TCP). Shared memory between the scheduler and a worker on its host is what
``test_processes.py`` runs: in one process, Python's resource tracker would see the same ring
created and attached.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, cast

import msgspec
import zmq

from core.common.ids import HostId, Identity, ProcId
from core.common.manifest import load_manifest
from core.context.config import ConfigEntry
from core.contracts.columns import Column
from core.contracts.policy import StepView, Take
from core.contracts.requests import REQUESTS, TOKENS, AbortRequest, NewRequest, Requests, TokenEvents
from core.contracts.sampling import SamplingParams
from core.contracts.serving import ReturnRoute
from core.contracts.state import StateDecl
from core.ipc.channels import Channels
from core.ipc.config import IpcTunables
from core.ipc.records import RecordBuilder, decode_record, encode_record, read_sections
from core.ipc.stream import StreamAddress
from core.sched.loop import Scheduler
from core.state.alloc.unit import JointLedger
from policies.prefill_first.unit import PrefillFirst
from sched.reqs.unit import RequestTable
from tests.units.scheduler.units.schedfix.worker.unit import FixtureWorker
from unit.runtime import ProcessPlan, ReloadPlan, UnitRuntime, UnitSpec

ENGINE = Path(__file__).resolve().parents[3] / "engine"
BIND = "127.0.0.1"
SCHED = ProcId(HostId("host-a"), "sched", 0)
WORKERS = (ProcId(HostId("host-b"), "worker", 0), ProcId(HostId("host-c"), "worker", 1))
CLIENT = HostId("frontend")
TUN = IpcTunables(peer_timeout_s=5.0, rpc_timeout_s=1.0, close_timeout_s=0.2, lookup_retry_s=0.01)

PAGE = 4
DECLS = [StateDecl("kv.paged", 0, {"k": ((PAGE, 1, 8), "bf16")}, "request", tokens_per_item=PAGE),
         StateDecl("gdn.slots", 1, {"s": ((8, 8), "f32")}, "slot")]
PAGES = {"kv.paged": PAGE, "gdn.slots": 0}
CAPACITY = {"kv.paged": 64, "gdn.slots": 8}
UNITS = ("sched/reqs", "core/state/alloc", "policies/fcfs", "policies/prefill_first", "core/sched")


class Directory:
    """Where each process's readers are: the supervisor's endpoint table, in a dict."""

    def __init__(self) -> None:
        self.addresses: dict[tuple[ProcId, str], StreamAddress] = {}
        self.tag = uuid.uuid4().hex[:8]

    def channels(self, zctx: zmq.Context[zmq.Socket[bytes]], proc: ProcId) -> Channels:
        return Channels(zctx, local=proc.host, bind=BIND, tun=TUN,
                        allocate=lambda name: f"sf{self.tag}{proc.index}{name.replace('.', '')}",
                        lookup=lambda p, name: self.addresses.get((p, name)),
                        publish=lambda name, address: self.addresses.__setitem__((proc, name), address))


class WorkerThread:
    """A fixture worker on a thread, with data channels of its own."""

    def __init__(self, directory: Directory, rank: int, *, max_batch: int, capacity: dict[str, int],
                 delay_s: float) -> None:
        self.directory, self.rank, self.proc = directory, rank, WORKERS[rank]
        self.kw: dict[str, Any] = {"rank": rank, "capacity": capacity, "pages": PAGES, "max_batch": max_batch,
                                   "delay_s": delay_s}
        self.ready, self.stopping = threading.Event(), threading.Event()
        self.error: BaseException | None = None
        self.worker: FixtureWorker | None = None
        self.thread = threading.Thread(target=self.run, name=f"fixture-worker-{rank}", daemon=True)
        self.thread.start()
        assert self.ready.wait(10) and self.error is None, self.error

    def run(self) -> None:
        zctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
        channels = self.directory.channels(zctx, self.proc)
        try:
            self.worker = FixtureWorker(channels.reader("plans", SCHED.host),
                                        channels.writer(SCHED, f"results.{self.rank}"), **self.kw)
            self.ready.set()
            while not self.stopping.is_set():
                self.worker.turn(0.002)
        except BaseException as e:  # noqa: BLE001
            self.error = e
            self.ready.set()
        finally:
            channels.close()
            zctx.destroy(linger=0)

    @property
    def fixture(self) -> FixtureWorker:
        assert self.worker is not None
        return self.worker

    def close(self) -> None:
        self.stopping.set()
        self.thread.join(5)
        assert not self.thread.is_alive(), "worker thread leaked"
        assert self.error is None, self.error


@dataclass
class Output:
    tokens: list[int] = field(default_factory=list[int])
    finished: str | None = None
    error: str | None = None
    identities: set[str] = field(default_factory=set[str])
    """The identities the steps that gave it tokens ran under."""


class Lookup(Protocol):
    def channels(self, zctx: zmq.Context[zmq.Socket[bytes]], proc: ProcId) -> Channels: ...


class Client:
    """The frontend's two streams: requests to the scheduler's reader ``requests``, tokens back."""

    def __init__(self, directory: Lookup) -> None:
        self.zctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
        self.channels = directory.channels(self.zctx, ProcId(CLIENT, "frontend", 0))
        self.tokens = self.channels.reader("tokens", SCHED.host)
        self.out = self.channels.writer(SCHED, "requests")
        self.outputs: dict[int, Output] = {}
        self.pending: list[bytes] = []

    def send(self, new: Sequence[NewRequest] = (), abort: Sequence[int] = ()) -> None:
        for request in new:
            self.outputs[request.request] = Output()
        record = (RecordBuilder(REQUESTS, 0).add(Requests.SECTION, Requests(list(new), [AbortRequest(a) for a in abort]))
                  .add(ReturnRoute.SECTION, ReturnRoute(self.tokens.address)).build())
        self.pending.append(encode_record(record))
        self.poll()

    def poll(self) -> None:
        while self.pending and self.out.write(self.pending[0], timeout=1.0):
            self.pending.pop(0)
        while (raw := self.tokens.read(timeout=0)) is not None:
            record = decode_record(raw)
            assert record.schema == TOKENS
            events = cast(TokenEvents, read_sections(record, {TokenEvents.SECTION: TokenEvents}).unwrap()[TokenEvents.SECTION])
            offsets = [int(x) for x in events.offsets.values()]
            tokens = [int(x) for x in events.tokens.values()]
            for i, rid in enumerate(int(x) for x in events.requests.values()):
                out = self.outputs.setdefault(rid, Output())
                assert out.finished is None, f"tokens for request {rid} after it finished"
                out.tokens.extend(tokens[offsets[i]:offsets[i + 1]])
                if offsets[i + 1] > offsets[i]:
                    out.identities.add(events.identity)
            for rid, reason in events.finished.items():
                self.outputs[rid].finished = reason
            for rid, error in events.errors.items():
                self.outputs[rid].error = error

    def close(self) -> None:
        self.channels.close()
        self.zctx.destroy(linger=0)


class ThreadClient:
    """A ``Client`` on a thread of its own, for a test whose own thread runs the control plane."""

    def __init__(self, directory: Lookup) -> None:
        self.directory = directory
        self.queue: queue.Queue[tuple[list[NewRequest], list[int]]] = queue.Queue()
        self.outputs: dict[int, Output] = {}
        self.ready, self.stopping = threading.Event(), threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self.run, name="client", daemon=True)
        self.thread.start()
        assert self.ready.wait(10) and self.error is None, self.error

    def run(self) -> None:
        client: Client | None = None
        try:
            client = Client(self.directory)
            self.outputs = client.outputs
            self.ready.set()
            while not self.stopping.is_set():
                with suppress(queue.Empty):
                    new, abort = self.queue.get(timeout=0.002)
                    client.send(new, abort)
                client.poll()
        except BaseException as e:  # noqa: BLE001
            traceback.print_exc()
            self.error = e
            self.ready.set()
        finally:
            if client is not None:
                client.close()

    def send(self, new: Sequence[NewRequest] = (), abort: Sequence[int] = ()) -> None:
        for r in new:
            self.outputs.setdefault(r.request, Output())
        self.queue.put((list(new), list(abort)))

    def close(self) -> None:
        self.stopping.set()
        self.thread.join(5)
        assert not self.thread.is_alive() and self.error is None, self.error


def request(rid: int, prompt: Sequence[int], max_tokens: int, *, seed: int | None = None,
            stop: Sequence[int] = ()) -> NewRequest:
    return NewRequest(rid, Column.of("i64", prompt), SamplingParams(seed=seed if seed is not None else rid * 7 + 1),
                      max_tokens, list(stop))


def specs(extra: Sequence[UnitSpec] = (), skip: Sequence[str] = ()) -> list[UnitSpec]:
    out = [UnitSpec(load_manifest(ENGINE / path / "unit.toml").unwrap()) for path in UNITS]
    return [s for s in out if str(s.id) not in skip] + list(extra)


class Engine:
    """The scheduler host: its units in a unit runtime, stepped by the test as its host would."""

    def __init__(self, directory: Directory, *, ranks: int = 2, overlap: bool = False, max_batch: int = 8,
                 token_budget: int = 16, config: dict[str, Any] | None = None, units: list[UnitSpec] | None = None) -> None:
        self.units = units or specs()
        values: dict[str, Any] = {
            "alloc.states": msgspec.json.decode(msgspec.json.encode(DECLS)), "sched.workers": msgspec.to_builtins(list(WORKERS[:ranks])),
            "sched.frontend_host": str(CLIENT), "sched.max_batch": max_batch, "sched.token_budget": token_budget,
            "sched.overlap": overlap, "sched.idle_wait_s": 0.002, "sched.frontend_timeout_s": 2.0,
            "sched.quiesce_timeout_s": 5.0, **(config or {})}
        self.config = {key: ConfigEntry(unit.id, decl, values.get(key, decl.default))
                       for unit in self.units for key, decl in unit.manifest.config.items()}
        self.zctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
        self.channels = directory.channels(self.zctx, SCHED)
        self.runtime = UnitRuntime(SCHED)
        self.runtime.provide_host("proc.control@1", SimpleNamespace(identity=Identity("v1"), address=""))
        self.runtime.provide_host("proc.streams@1", self.channels)
        self.runtime.load_process_plan(ProcessPlan(SCHED, Identity("v1"), self.units, config=self.config)).unwrap()

    @property
    def loop(self) -> Scheduler:
        loop = self.runtime.lookup("proc.loop@1")
        assert loop is not None, self.runtime.snapshot()
        return cast(Scheduler, loop)

    @property
    def table(self) -> RequestTable:
        return cast(RequestTable, self.runtime.lookup("sched.reqs@1"))

    @property
    def ledger(self) -> JointLedger:
        return cast(JointLedger, self.runtime.lookup("ledger@1"))

    def reload(self, *units: UnitSpec) -> None:
        """As a change would: the loop finishes its steps first, then the units reload."""
        self.loop.quiesce()
        self.runtime.reload(ReloadPlan(units_to_reload=list(units))).unwrap()

    def spec(self, unit: str) -> UnitSpec:
        found = self.runtime.spec_of(unit)
        assert found is not None
        return found

    def close(self) -> None:
        self.runtime.unload_all().unwrap()
        self.channels.close()
        self.zctx.destroy(linger=0)


@dataclass
class Bed:
    engine: Engine
    workers: list[WorkerThread]
    client: Client

    def turn(self) -> None:
        self.engine.loop.step()
        self.client.poll()

    def run_until(self, done: Callable[[], bool], timeout: float = 10.0, what: str = "") -> None:
        deadline = time.monotonic() + timeout
        while not done():
            assert time.monotonic() < deadline, f"timed out: {what} {self.engine.loop.report()} " \
                                                f"{[w.fixture.errors for w in self.workers]}"
            self.turn()

    def finished(self, *rids: int) -> Callable[[], bool]:
        return lambda: all(self.client.outputs[r].finished is not None for r in rids)

    def idle(self) -> bool:
        """No request left and every item back in the ledger."""
        ledger = self.engine.ledger
        return not list(self.engine.table.entries()) and not self.engine.loop.flights and \
            all(len(p.free) == p.capacity and not p.holdings for p in ledger.pools.values())

    def errors(self) -> list[str]:
        return [e for w in self.workers for e in w.fixture.errors]


@contextmanager
def bed(*, ranks: int = 2, delay_s: float = 0.0, capacity: dict[str, int] | None = None,
        **engine: Any) -> Generator[Bed]:
    directory = Directory()
    workers: list[WorkerThread] = []
    client: Client | None = None
    e: Engine | None = None
    try:
        e = Engine(directory, ranks=ranks, **engine)
        max_batch = e.config["sched.max_batch"].value
        workers = [WorkerThread(directory, r, max_batch=max_batch, capacity=dict(capacity or CAPACITY),
                                delay_s=delay_s) for r in range(ranks)]
        client = Client(directory)
        b = Bed(e, workers, client)
        b.run_until(lambda: e.ledger.ready, what="capacity")
        yield b
    finally:
        problems: list[AssertionError] = []
        for w in workers:
            try:
                w.close()
            except AssertionError as error:
                problems.append(error)
        if client is not None:
            client.close()
        if e is not None:
            e.close()
        if problems:
            raise problems[0]


class SwitchableCompose:
    """``policy.compose@1`` that, while ``bad``, asks for one prompt token more than is left."""

    def __init__(self, ctx: Any, config: Any) -> None:
        self.good = PrefillFirst()
        self.bad = True
        ctx.provide("policy.compose@1", self)

    def compose(self, view: StepView) -> list[Take]:
        if self.bad and view.waiting:
            c = view.waiting[0]
            return [Take(c.request, c.prompt + 1)]
        return self.good.compose(view)
