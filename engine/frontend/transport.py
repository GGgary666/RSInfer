"""The frontend's two record streams, in place of mini-sglang's ZMQ queues.

The HTTP server runs its own asyncio loop on its own thread, and a ZMQ socket must not be
used from two threads at once, so every stream operation runs on one I/O thread this
channel owns; HTTP handlers never touch a socket:

- The token reader is the host's (``proc.streams@1`` reader ``tokens``): opened on the host
  thread when the unit loads, then used by this I/O thread only. It outlives a reload of the
  frontend, so the return route the scheduler writes to stays the same.
- The request writer is this channel's own ``ChannelWriter``, with its own ZMQ context and its
  own RPC client to the supervisor, so its lookups never touch the host's sockets. When the
  scheduler's process is replaced, it finds the new one's reader ``requests``.

A replaced scheduler lost every request it had. A scheduler instance is known by the
address of its reader ``requests``: a new process opens a new one. ``incarnation`` counts
the instances; it moves when a request goes to a new address, or when a new producer of
tokens appears and the supervisor names a new address (a new producer alone may be the same
scheduler writing again). The manager ends the requests sent to an earlier instance.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import zmq

from core.common.ids import Endpoint, HostId, ProcId
from core.common.result import Failed
from core.contracts.requests import REQUESTS, TOKENS, Requests, TokenEvents
from core.contracts.serving import ReturnRoute
from core.ipc.channels import ChannelReader, ChannelWriter
from core.ipc.config import IpcTunables
from core.ipc.records import RecordBuilder, decode_record, encode_record, read_sections
from core.ipc.rpc import SyncRpcClient
from core.ipc.stream import StreamAddress
from core.supervisor.protocol import Found, Lookup, control_registry


class Channel(Protocol):
    @property
    def incarnation(self) -> int:
        """Which scheduler instance this is, counting from 0 since the channel started."""
        ...

    async def start(self) -> None: ...
    async def send(self, requests: Requests) -> int | None:
        """The incarnation the requests went to; ``None``: no scheduler to send to right now,
        nothing was sent."""
        ...
    async def receive(self) -> tuple[int, TokenEvents]: ...
    async def close(self) -> None: ...


class Directory:
    """Where the scheduler's reader ``requests`` is: a fixed address, or the supervisor."""

    def __init__(self, *, fixed: StreamAddress | None = None, supervisor: str = "",
                 scheduler: ProcId | None = None, me: ProcId | None = None) -> None:
        if fixed is None and (scheduler is None or me is None or not supervisor):
            raise ValueError("give the scheduler's address, or the supervisor and the scheduler's ProcId")
        self.fixed, self.supervisor, self.scheduler, self.me = fixed, supervisor, scheduler, me


LOOKUP_TIMEOUT_S = 0.25
"""A lookup runs on the I/O thread: it must not hold token reading up for long."""


@dataclass
class _Send:
    requests: Requests
    deadline: float
    """Set when the caller queued it: time spent waiting behind other sends counts."""
    done: Callable[[int | BaseException | None], None]


class StepChannel:
    """One I/O thread runs a loop: it takes the requests to send from a queue, writes them in
    order without waiting, reads tokens and hands them to the asyncio loop. A send that cannot
    go out yet (no scheduler, a connection still opening, a full queue) stays queued while
    reading and later turns go on; once past its deadline it is dropped and its caller gets
    ``None``. So one request waiting for a scheduler holds up no other, and no token."""

    def __init__(self, local: HostId, tokens: ChannelReader, directory: Directory, *,
                 tun: IpcTunables | None = None) -> None:
        self.local, self.tokens, self.directory = local, tokens, directory
        self.tun = tun or IpcTunables()
        self.context: zmq.Context[zmq.Socket[bytes]] | None = None
        self.rpc: SyncRpcClient | None = None
        self.writer: ChannelWriter | None = None
        self.sequence = 0
        self.closed = False
        self.incarnation = 0
        self._target: StreamAddress | None = None
        """The scheduler's reader ``requests`` the latest request went to."""
        self._producers = 0
        """Producers of tokens seen."""
        self._outbox: queue.SimpleQueue[_Send] = queue.SimpleQueue()
        self._inbox: asyncio.Queue[tuple[int, TokenEvents] | BaseException] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------ I/O thread

    def _open(self) -> None:
        self.context = zmq.Context()
        d = self.directory
        if d.fixed is None:
            assert d.me is not None
            self.rpc = SyncRpcClient(self.context, control_registry(), Endpoint(d.me, "frontend-lookup"))
        self.writer = ChannelWriter(self.context, self.local, self.tun, self._resolve)

    def _resolve(self) -> StreamAddress | None:
        d = self.directory
        if d.fixed is not None:
            return d.fixed
        assert self.rpc is not None and d.scheduler is not None
        found = self.rpc.call(d.supervisor, Lookup(d.scheduler, "requests"),
                              timeout=min(self.tun.rpc_timeout_s, LOOKUP_TIMEOUT_S))
        if isinstance(found, Failed) or not isinstance(found.value, Found):
            return None
        return found.value.address

    def _moved_to(self, address: StreamAddress | None) -> None:
        if address is None or address == self._target:
            return
        if self._target is not None:
            self.incarnation += 1
        self._target = address
        assert self.writer is not None
        if self.writer.address not in (None, address):
            self.writer.abandon()  # still connected to the replaced one: connect to this one next

    def _flush(self, pending: deque[_Send]) -> None:
        """Write what can go out now, in order; drop what is past its deadline."""
        assert self.writer is not None
        while pending:
            item = pending[0]
            if time.monotonic() > item.deadline:
                pending.popleft()
                item.done(None)
                continue
            try:
                record = (RecordBuilder(REQUESTS, self.sequence).add(Requests.SECTION, item.requests)
                          .add(ReturnRoute.SECTION, ReturnRoute(self.tokens.address)).build())
                if not self.writer.write(encode_record(record), timeout=0):
                    return  # no scheduler right now, or a connection still opening
            except TimeoutError:
                return  # the scheduler is there but its queue is full
            except Exception as e:  # noqa: BLE001 - this request cannot be sent: its caller says why
                pending.popleft()
                item.done(e)
                continue
            pending.popleft()
            self.sequence += 1
            self._moved_to(self.writer.address)
            item.done(self.incarnation)

    def _read(self, wait: float) -> tuple[int, TokenEvents] | None:
        data = self.tokens.read(timeout=wait)
        if data is None:
            return None
        if self.tokens.generation > self._producers:
            if self._producers:  # a new producer: is it a new scheduler, or the same one again?
                self._moved_to(self._resolve())
            self._producers = self.tokens.generation
        record = decode_record(data)
        if record.schema != TOKENS:
            raise ValueError(f"expected {TOKENS}, got {record.schema}")
        events = read_sections(record, {TokenEvents.SECTION: TokenEvents}).unwrap()[TokenEvents.SECTION]
        return record.epoch, events

    def _run(self, started: Callable[[BaseException | None], None]) -> None:
        try:
            self._open()
        except BaseException as e:  # noqa: BLE001 - reported to start()
            started(e)
            return
        started(None)
        pending: deque[_Send] = deque()
        reading = True
        try:
            while not self._stop.is_set():
                while True:
                    try:
                        pending.append(self._outbox.get_nowait())
                    except queue.Empty:
                        break
                self._flush(pending)
                wait = 0.002 if pending else 0.01
                if not reading:
                    time.sleep(wait)
                    continue
                try:
                    got = self._read(wait)
                except BaseException as e:  # noqa: BLE001 - the token stream broke: tell the listener once
                    reading = False
                    self._deliver(e)
                    continue
                if got is not None:
                    self._deliver(got)
        finally:
            for item in pending:
                item.done(None)
            while True:
                try:
                    self._outbox.get_nowait().done(None)
                except queue.Empty:
                    break
            self._close()

    def _deliver(self, item: tuple[int, TokenEvents] | BaseException) -> None:
        assert self._loop is not None and self._inbox is not None
        self._loop.call_soon_threadsafe(self._inbox.put_nowait, item)

    def _close(self) -> None:
        try:
            if self.writer is not None:
                self.writer.close()
        finally:
            if self.rpc is not None:
                self.rpc.close()
            if self.context is not None:
                self.context.destroy(linger=0)
            self.writer, self.rpc, self.context = None, None, None

    # ------------------------------------------------------------ asyncio side

    def _answer(self, future: asyncio.Future[Any]) -> Callable[[Any], None]:
        loop = asyncio.get_running_loop()

        def done(value: Any) -> None:
            def settle() -> None:
                if future.done():
                    return
                if isinstance(value, BaseException):
                    future.set_exception(value)
                else:
                    future.set_result(value)
            loop.call_soon_threadsafe(settle)
        return done

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._inbox = asyncio.Queue()
        started: asyncio.Future[None] = self._loop.create_future()
        self._thread = threading.Thread(target=self._run, args=(self._answer(started),), name="frontend-io",
                                        daemon=True)
        self._thread.start()
        await started

    async def send(self, requests: Requests) -> int | None:
        if self.closed:
            return None
        future: asyncio.Future[int | None] = asyncio.get_running_loop().create_future()
        self._outbox.put(_Send(requests, time.monotonic() + self.tun.rpc_timeout_s, self._answer(future)))
        return await future

    async def receive(self) -> tuple[int, TokenEvents]:
        if self._inbox is None or self.closed:
            raise ConnectionError("frontend channel is closed")
        item = await self._inbox.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._stop.set()
        if self._thread is not None:
            await asyncio.to_thread(self._thread.join)
