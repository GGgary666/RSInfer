"""Data channels the process host keeps for its units: ``proc.streams@1`` (doc/specs/ipc.md §6).

A unit asks the host for a step-stream reader by name, or for a writer to a named reader of
another process. The host owns them, so they outlive the units that use them: a reloaded
loop unit finds its readers where they were, at the same address.

Peers come and go. A reader takes the next producer when one closes or dies; a writer to a
named reader looks it up again, through the supervisor, when that reader's process is
replaced. Both count how often that happened (``generation``): what was in flight to an
old peer may be lost, and only the unit knows what to do about it.

All calls run on the host's loop thread. Lookups go to the supervisor only while a writer
has no reader to write to, at most every ``ipc.lookup_retry_s``; never per record.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import zmq

from core.common.ids import HostId, ProcId
from core.ipc.config import IpcTunables
from core.ipc.stream import (
    PeerLost,
    RingBusy,
    ShmReader,
    ShmWriter,
    StreamAddress,
    StreamClosed,
    TcpReader,
    TcpWriter,
    backend_for,
    open_reader,
    open_writer,
)

RESERVED = frozenset({"rpc"})
"""Endpoint names the host itself publishes."""


class ChannelReader:
    """A reader the host keeps; a unit reads from it but does not close it."""

    def __init__(self, reader: ShmReader | TcpReader, address: StreamAddress, peer: HostId) -> None:
        self._reader, self.address, self.peer = reader, address, peer

    def read(self, timeout: float | None = None) -> bytes | None:
        """The next record, or ``None`` if none arrived within ``timeout``. A producer that
        left is not an error: the reader waits for the next one."""
        return self._reader.read(timeout)

    @property
    def generation(self) -> int:
        """Which producer wrote the record last returned: 1 for the first, one more per
        replacement. A change means records of an earlier producer may be lost."""
        return self._reader.writer_generation

    def close(self) -> None:
        self._reader.close()


class ChannelWriter:
    """Writes to one reader, and finds it again when the reader's process is replaced."""

    def __init__(self, zctx: zmq.Context[zmq.Socket[bytes]], local: HostId, tun: IpcTunables,
                 resolve: Callable[[], StreamAddress | None]) -> None:
        self._zctx, self._local, self._tun, self._resolve = zctx, local, tun, resolve
        self._writer: ShmWriter | TcpWriter | None = None
        self._connected = False
        self._since = 0.0
        self._next_try = 0.0
        self._dead: StreamAddress | None = None
        self._dead_until = 0.0
        self.address: StreamAddress | None = None
        self.generation = 0
        """Connections established; a change means records sent on an earlier one may be lost."""

    def write(self, data: bytes, timeout: float | None = None) -> bool:
        """Send one record. ``False``: there is no reader to send to right now (not published
        yet, gone, its replacement not published yet, or a TCP connection still opening), and
        nothing was sent; try again later. Never waits for a connection. Raises
        ``TimeoutError`` if a connected reader does not keep up within ``timeout``."""
        if self._writer is None and not self._connect():
            return False
        writer = self._writer
        assert writer is not None
        if not self._connected:
            assert isinstance(writer, TcpWriter)
            if not writer.connected:
                if time.monotonic() - self._since > self._tun.peer_timeout_s:
                    self.abandon()  # nobody accepts the connection there
                return False
            self._connected = True
            self.generation += 1
        try:
            writer.write(data, timeout)
        except (PeerLost, StreamClosed):
            self.abandon()
            return False
        return True

    def _connect(self) -> bool:
        now = time.monotonic()
        if now < self._next_try:
            return False
        self._next_try = now + self._tun.lookup_retry_s
        address = self._resolve()
        if address is None or (address == self._dead and now < self._dead_until):
            return False  # a dead reader's address is tried again only every peer_timeout_s
        try:
            writer = open_writer(self._zctx, local=self._local, address=address, tun=self._tun)
        except RingBusy:
            return False  # the previous producer has not closed or timed out yet
        except (PeerLost, FileNotFoundError):
            self._bury(address)  # the ring's reader is gone, or the ring itself
            return False
        self._writer, self.address, self._since = writer, address, now
        self._connected = isinstance(writer, ShmWriter)  # a TCP connection is made in the background
        if self._connected:
            self.generation += 1
        return True

    def _bury(self, address: StreamAddress | None) -> None:
        self._dead, self._dead_until = address, time.monotonic() + self._tun.peer_timeout_s

    def abandon(self) -> None:
        """The reader is gone: drop the connection without delivering what is queued. The next
        write looks the reader up again, and connects to this address no sooner than
        ``peer_timeout_s`` from now."""
        if self._writer is not None:
            self._writer.abort()
            self._writer = None
            self._bury(self.address)

    def close(self) -> None:
        """Orderly end: the reader gets what is queued, then waits for the next producer."""
        if self._writer is not None:
            try:
                self._writer.close()
            finally:
                self._writer = None


class Channels:
    """``proc.streams@1``: the data channels of one process host."""

    def __init__(self, zctx: zmq.Context[zmq.Socket[bytes]], *, local: HostId, bind: str, tun: IpcTunables,
                 allocate: Callable[[str], str], lookup: Callable[[ProcId, str], StreamAddress | None],
                 publish: Callable[[str, StreamAddress], None]) -> None:
        self._zctx, self.local, self._bind, self._tun = zctx, local, bind, tun
        self._allocate, self._lookup, self._publish = allocate, lookup, publish
        self._readers: dict[str, ChannelReader] = {}
        self._writers: dict[tuple[ProcId, str], ChannelWriter] = {}
        self._direct: dict[StreamAddress, ChannelWriter] = {}

    def reader(self, name: str, peer: HostId) -> ChannelReader:
        """The reader ``name``, for a producer on host ``peer``; made and published the first
        time, the same one afterwards (a reloaded unit gets its reader back)."""
        if name in RESERVED:
            raise ValueError(f"{name!r} is an endpoint of the host itself")
        known = self._readers.get(name)
        if known is not None:
            if known.peer != peer:
                raise ValueError(f"reader {name!r} takes records from {known.peer}, not {peer}")
            return known
        shm = self._allocate(name) if backend_for(self.local, peer) == "shm" else ""
        raw, address = open_reader(self._zctx, local=self.local, peer=peer, bind_host=self._bind,
                                   shm_name=shm, tun=self._tun, replaceable=True)
        reader = ChannelReader(raw, address, peer)
        try:
            self._publish(name, address)
        except Exception:
            reader.close()
            raise
        self._readers[name] = reader
        return reader

    def writer(self, proc: ProcId, name: str) -> ChannelWriter:
        """A writer to the reader ``name`` of process ``proc``, found through the supervisor."""
        key = proc, name
        if key not in self._writers:
            self._writers[key] = ChannelWriter(self._zctx, self.local, self._tun, lambda: self._lookup(proc, name))
        return self._writers[key]

    def writer_to(self, address: StreamAddress) -> ChannelWriter:
        """A writer to a reader whose address came in a record (a frontend's return route);
        it is not looked up again. ``discard`` it once that reader is known to be gone."""
        if address not in self._direct:
            self._direct[address] = ChannelWriter(self._zctx, self.local, self._tun, lambda: address)
        return self._direct[address]

    def discard(self, address: StreamAddress) -> None:
        writer = self._direct.pop(address, None)
        if writer is not None:
            writer.abandon()

    def close(self) -> None:
        for writer in [*self._writers.values(), *self._direct.values()]:
            try:
                writer.close()
            except Exception:  # noqa: BLE001 - closing the rest matters more
                pass
        for reader in self._readers.values():
            reader.close()
        self._writers.clear()
        self._direct.clear()
        self._readers.clear()
