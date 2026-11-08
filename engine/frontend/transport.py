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
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

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


class StepChannel:
    def __init__(self, local: HostId, tokens: ChannelReader, directory: Directory, *,
                 tun: IpcTunables | None = None) -> None:
        self.local, self.tokens, self.directory = local, tokens, directory
        self.tun = tun or IpcTunables()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="frontend-io")
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
        found = self.rpc.call(d.supervisor, Lookup(d.scheduler, "requests"), timeout=self.tun.rpc_timeout_s)
        if isinstance(found, Failed) or not isinstance(found.value, Found):
            return None
        return found.value.address

    def _moved_to(self, address: StreamAddress | None) -> None:
        if address is not None and address != self._target:
            if self._target is not None:
                self.incarnation += 1
            self._target = address

    async def start(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self.pool, self._open)

    def _send(self, requests: Requests) -> int | None:
        assert self.writer is not None
        record = (RecordBuilder(REQUESTS, self.sequence).add(Requests.SECTION, requests)
                  .add(ReturnRoute.SECTION, ReturnRoute(self.tokens.address)).build())
        data = encode_record(record)
        deadline = time.monotonic() + self.tun.rpc_timeout_s
        while True:  # a scheduler that was just replaced is found, and connected to, in a moment
            try:
                if self.writer.write(data, timeout=max(0.0, deadline - time.monotonic())):
                    break
            except TimeoutError:
                return None  # the scheduler is there but does not take requests in time
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.01)
        self.sequence += 1
        self._moved_to(self.writer.address)
        return self.incarnation

    async def send(self, requests: Requests) -> int | None:
        if self.closed:
            return None
        return await asyncio.get_running_loop().run_in_executor(self.pool, self._send, requests)

    def _read(self) -> tuple[int, TokenEvents] | None:
        data = self.tokens.read(timeout=0.01)
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

    async def receive(self) -> tuple[int, TokenEvents]:
        while not self.closed:
            result = await asyncio.get_running_loop().run_in_executor(self.pool, self._read)
            if result is not None:
                return result
        raise ConnectionError("frontend channel is closed")

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

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            await asyncio.get_running_loop().run_in_executor(self.pool, self._close)
        finally:
            self.pool.shutdown(wait=True)
