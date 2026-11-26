"""Frontend fixtures: a byte tokenizer, a scripted scheduler on the real record streams (and the
real request table), and a stand-in for the supervisor's lookups. No model output: the
scheduler serves a fixed text, so HTTP, routing, cancellation and teardown can be tested
without the scheduler of stage 2's next step, a model or a GPU.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import msgspec
import zmq
import zmq.asyncio

from core.common.ids import Endpoint, HostId, Identity, ProcId
from core.common.manifest import load_manifest
from core.common.messages import Envelope
from core.common.result import Ok, Result, fail
from core.context.config import ConfigEntry
from core.contracts.columns import Column
from core.contracts.records import Sampled
from core.contracts.requests import REQUESTS, TOKENS, AbortRequest, NewRequest, Requests, TokenEvents
from core.contracts.serving import ReturnRoute
from core.ipc.channels import Channels
from core.ipc.config import IpcTunables
from core.ipc.records import RecordBuilder, decode_record, encode_record, read_sections
from core.ipc.rpc import RpcServer
from core.ipc.stream import ShmWriter, StreamAddress, TcpWriter, open_reader, open_writer
from core.supervisor.protocol import Found, Lookup, control_registry
from frontend.http.unit import Server
from frontend.interfaces import Message
from sched.reqs.unit import RequestTable
from tests.framework.support import spec
from unit.runtime import ProcessPlan, UnitRuntime, UnitSpec

ENGINE = Path(__file__).resolve().parents[3] / "engine"
BIND = "127.0.0.1"
SCHEDULER = ProcId(HostId("scheduler"), "sched", 0)
TUN = IpcTunables(peer_timeout_s=5.0, rpc_timeout_s=1.0, close_timeout_s=0.2)


class ByteTokenizer:
    @property
    def stop_token_ids(self) -> Sequence[int]:
        return [256]

    def encode(self, text: str) -> list[int]:
        return list(text.encode())

    def chat(self, messages: Sequence[Message], *, reasoning_effort: str | None = None) -> list[int]:
        return self.encode("".join(f"{m.role}: {m.content}\n" for m in messages) + "assistant: ")

    def decode(self, tokens: Sequence[int]) -> str:
        return bytes(t for t in tokens if t != 256).decode(errors="replace")


class TokenizerUnit:
    def __init__(self, ctx: Any, config: Any) -> None:
        ctx.provide("frontend.tokenizer@1", ByteTokenizer())


class ScriptScheduler:
    """Serves ``text`` one token per step to every request, on a thread of its own. ``hold``:
    accept requests but send nothing; ``refuse``: refuse every request with this message."""

    def __init__(self, text: str = "你好，世界!", *, hold: bool = False, refuse: str = "") -> None:
        self.text, self.hold, self.refuse = text, hold, refuse
        self.identity = "v1"
        self.new: queue.Queue[NewRequest] = queue.Queue()
        self.aborts: queue.Queue[AbortRequest] = queue.Queue()
        self.ready, self.stopping = threading.Event(), threading.Event()
        self.address: StreamAddress | None = None
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self.run, name="scripted-scheduler", daemon=True)
        self.thread.start()
        assert self.ready.wait(10), "scheduler did not start"
        assert self.error is None, self.error

    def run(self) -> None:
        ctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
        writers: dict[StreamAddress, ShmWriter | TcpWriter] = {}
        try:
            reader, self.address = open_reader(ctx, local=SCHEDULER.host, peer=HostId("frontend"), bind_host=BIND,
                                               shm_name="", tun=TUN, replaceable=True)
            table = RequestTable()
            replies: dict[int, list[int]] = {}
            route: StreamAddress | None = None
            epoch = 0
            self.ready.set()

            def emit(events: TokenEvents) -> None:
                assert route is not None
                if route not in writers:
                    writers[route] = open_writer(ctx, local=SCHEDULER.host, address=route, tun=TUN)
                record = RecordBuilder(TOKENS, epoch).add(TokenEvents.SECTION, events).build()
                writers[route].write(encode_record(record), timeout=1.0)

            while not self.stopping.is_set():
                epoch += 1
                data = reader.read(timeout=0.005)
                if data is not None:
                    record = decode_record(data)
                    assert record.schema == REQUESTS
                    parts = read_sections(record, {Requests.SECTION: Requests, ReturnRoute.SECTION: ReturnRoute}).unwrap()
                    requests = cast(Requests, parts[Requests.SECTION])
                    route = cast(ReturnRoute, parts[ReturnRoute.SECTION]).tokens
                    for request in requests.new:
                        self.new.put(request)
                        if self.refuse:
                            rid = request.request
                            emit(TokenEvents(self.identity, Column.of("i64", [rid]), Column.of("i32", [0, 0]),
                                             Column.of("i64", []), {rid: "abort"}, {rid: self.refuse}))
                            continue
                        table.add(request, route=route).unwrap()
                        replies[request.request] = [*self.text.encode(), 256]
                    for abort in requests.abort:
                        self.aborts.put(abort)
                        emit(table.cancel(abort.request, self.identity))
                        replies.pop(abort.request, None)
                        entry = table.get(abort.request)
                        if entry is not None and entry.reclaimable:
                            table.reap(abort.request)
                if self.hold or not replies:
                    continue
                rows = dict(enumerate(replies))
                table.start_step(epoch, rows)
                sampled = Sampled(Column.of("i32", rows), Column.of("i32", [1] * len(rows)),
                                  Column.of("i64", [replies[rid].pop(0) for rid in rows.values()]))
                emit(table.sample(epoch, self.identity, sampled))
                for rid in table.retire(epoch):
                    replies.pop(rid)
                    table.reap(rid)
            reader.close()
        except BaseException as exc:
            if not self.stopping.is_set():
                self.error = exc
            self.ready.set()
        finally:
            for writer in writers.values():
                with suppress(Exception):
                    writer.close()
            ctx.destroy(linger=0)

    def close(self) -> None:
        self.stopping.set()
        self.thread.join(5)
        assert not self.thread.is_alive(), "scheduler thread leaked"
        assert self.error is None, self.error


class FakeSupervisor:
    """Answers the frontend's ``Lookup`` of the scheduler's reader ``requests`` with ``target``."""

    def __init__(self) -> None:
        self.target: StreamAddress | None = None
        self.address = ""
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="fake-supervisor", daemon=True)
        self._thread.start()
        assert self._ready.wait(10)

    def _run(self) -> None:
        async def main() -> None:
            zctx = zmq.asyncio.Context()
            server = RpcServer(zctx, control_registry(), Endpoint(ProcId(HostId("control"), "control", 0),
                                                                  "supervisor"), BIND)
            self.address = server.address

            async def handle(env: Envelope, msg: msgspec.Struct) -> Result[msgspec.Struct]:
                if isinstance(msg, Lookup) and msg.proc == SCHEDULER and msg.channel == "requests":
                    return Ok(Found(self.target))
                return fail("test", "lookup", f"unexpected {msg}", code="test.unexpected")

            self._ready.set()
            try:
                await server.serve(handle)
            finally:
                server.close()
                zctx.destroy(linger=0)

        with suppress(asyncio.CancelledError):
            self._loop.run_until_complete(main())

    def close(self) -> None:
        for task in asyncio.all_tasks(self._loop):
            self._loop.call_soon_threadsafe(task.cancel)
        self._thread.join(5)


class Frontend:
    """The real frontend units in a unit runtime, with the host services they use."""

    def __init__(self, scheduler: ScriptScheduler | None = None, *, supervisor: FakeSupervisor | None = None,
                 response_timeout_s: float = 120.0, io_timeout_s: float = 2.0, paused: bool = False) -> None:
        units = [spec("frontend.tokenizer", "tests.units.frontend.support:TokenizerUnit",
                      provides=["frontend.tokenizer@1"], process=["frontend"]),
                 *(UnitSpec(load_manifest(ENGINE / f"frontend/{name}/unit.toml").unwrap())
                   for name in ("detok", "http"))]
        overrides: dict[str, Any] = {"frontend.model": "toy", "frontend.http_bind": BIND, "frontend.http_port": 0,
                                     "frontend.response_timeout_s": response_timeout_s,
                                     "frontend.io_timeout_s": io_timeout_s, "frontend.shutdown_timeout_s": 5.0}
        if supervisor is None:
            assert scheduler is not None and scheduler.address is not None
            overrides["frontend.requests_address"] = msgspec.to_builtins(scheduler.address)
        else:
            overrides["frontend.scheduler"] = msgspec.to_builtins(SCHEDULER)
        config = {key: ConfigEntry(unit.id, decl, overrides.get(key, decl.default))
                  for unit in units for key, decl in unit.manifest.config.items()}
        proc = ProcId(HostId("frontend"), "frontend", 0)
        self.zctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
        self.published: dict[str, StreamAddress] = {}
        self.channels = Channels(self.zctx, local=proc.host, bind=BIND, tun=TUN, allocate=lambda name: name,
                                 lookup=lambda p, name: None, publish=self.published.__setitem__)
        self.runtime = UnitRuntime(proc)
        self.control = SimpleNamespace(identity=Identity("v1"), address=supervisor.address if supervisor is not None else "",
                                       paused=paused)
        """``proc.control@1``; ``paused``: the process is paused (a change's Execute, or before it started)."""
        self.runtime.provide_host("proc.control@1", self.control)
        self.runtime.provide_host("proc.streams@1", self.channels)
        self.runtime.load_process_plan(ProcessPlan(proc, Identity("v1"), units, config=config)).unwrap()
        server = cast(Server | None, self.runtime.lookup("frontend.http@1"))
        assert server is not None, self.runtime.snapshot()
        self.server = server
        self.url = f"http://{BIND}:{server.port}"

    def close(self) -> None:
        self.runtime.unload_all().unwrap()
        self.channels.close()
        self.zctx.destroy(linger=0)


def wait_empty(manager: Any) -> None:
    deadline = time.monotonic() + 3
    while manager.states and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not manager.states
