"""Per-step record streams (doc/specs/ipc.md §2): StepPlan, StepResult.

Ordered, lossless, single producer; the producer blocks when the consumer falls
behind. Same host: a shared-memory ring (segment named by the node agent).
Different hosts: ZMQ PUSH/PULL over TCP with sequence numbers. Both expose the
same ``StepWriter`` / ``StepReader``, and the backend is chosen by comparing
``HostId``s, never addresses (P7) — two fake hosts on one machine use TCP.

A peer that stops proving it is alive for ``ipc.peer_timeout_s`` is reported
as ``PeerLost``; the caller reports that to the control plane.

A reader opened ``replaceable`` outlives its producers: when one closes or dies, the
next one takes over, one at a time, and ``writer_generation`` says which producer wrote the
record just returned. The process host keeps such readers for its units
(``core.ipc.channels``, doc/specs/ipc.md §6).
"""

from __future__ import annotations

import fcntl
import math
import secrets
import struct
import threading
import time
from collections import deque
from collections.abc import Callable
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from typing import Literal, Protocol

import msgspec
import zmq
from zmq.utils.monitor import recv_monitor_message

from core.common.ids import HostId
from core.ipc.config import IpcTunables

Backend = Literal["shm", "tcp"]


class PeerLost(ConnectionError):
    pass


class StreamClosed(EOFError):
    """The producer closed the stream in an orderly way."""


class RingBusy(ConnectionError):
    """A live producer already writes to this shared-memory ring."""


class StreamAddress(msgspec.Struct, frozen=True):
    """Where a reader listens. Published through the endpoint directory."""

    host: HostId
    shm: str | None = None
    tcp: str | None = None


class StepWriter(Protocol):
    def write(self, data: bytes, timeout: float | None = None) -> None: ...
    def close(self) -> None: ...


class StepReader(Protocol):
    def read(self, timeout: float | None = None) -> bytes | None:
        """The next record, or None if nothing arrived within ``timeout``."""
        ...

    def close(self) -> None: ...


def backend_for(a: HostId, b: HostId) -> Backend:
    return "shm" if a == b else "tcp"


class _Backoff:
    def __init__(self) -> None:
        self._delay = 2e-5

    def wait(self) -> None:
        time.sleep(self._delay)
        self._delay = min(self._delay * 2, 1e-3)


class _Heartbeat(threading.Thread):
    def __init__(self, interval: float, beat: Callable[[], None]) -> None:
        super().__init__(daemon=True, name="stream-heartbeat")
        self._interval = interval
        self._beat = beat
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.wait(self._interval):
            try:
                self._beat()
            except (ValueError, TypeError):  # segment closed under us
                return

    def stop(self) -> None:
        self._stop_event.set()


# --------------------------------------------------------------------- shm

_MAGIC = 0x41475852494E4732  # "AGXRING2"
_HEADER = 128
_MARK = 0xFFFFFFFF
# offsets in the header
_O_MAGIC, _O_CAP, _O_HEAD, _O_TAIL, _O_WBEAT, _O_RBEAT, _O_WCLOSED, _O_RCLOSED, _O_WGEN = range(0, 72, 8)


def _pad(n: int) -> int:
    return (n + 7) & ~7


class _Ring:
    """A single-producer single-consumer byte ring in a shared-memory segment.

    The producer writes the payload, then the length, then publishes ``head``;
    the consumer reads ``head`` before the length and payload. 8-byte aligned
    stores are single instructions on x86-64, which keeps this sound for v0;
    the ring is a first candidate for a Rust rewrite.
    """

    def __init__(self, shm: SharedMemory, owner: bool) -> None:
        assert shm.buf is not None
        self.shm = shm
        self.buf: memoryview = shm.buf
        self.owner = owner
        self.capacity = self.u64(_O_CAP)

    @classmethod
    def create(cls, name: str, capacity: int) -> _Ring:
        shm = SharedMemory(name=name, create=True, size=_HEADER + capacity)
        assert shm.buf is not None
        struct.pack_into("<QQQQddQQQ", shm.buf, 0, _MAGIC, capacity, 0, 0, 0.0, 0.0, 0, 0, 0)
        return cls(shm, owner=True)

    @classmethod
    def attach(cls, name: str) -> _Ring:
        shm = SharedMemory(name=name, create=False)
        # Python 3.12 registers attached segments with this process's resource
        # tracker, which would unlink them when this process exits.
        resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
        ring = cls(shm, owner=False)
        if ring.u64(_O_MAGIC) != _MAGIC:
            ring.close()
            raise ValueError(f"{name} is not a step-stream ring")
        return ring

    def u64(self, off: int) -> int:
        return struct.unpack_from("<Q", self.buf, off)[0]

    def set_u64(self, off: int, v: int) -> None:
        struct.pack_into("<Q", self.buf, off, v)

    def f64(self, off: int) -> float:
        return struct.unpack_from("<d", self.buf, off)[0]

    def beat(self, off: int) -> None:
        struct.pack_into("<d", self.buf, off, time.monotonic())

    def peer_lost(self, off: int, timeout: float) -> bool:
        last = self.f64(off)
        return last != 0.0 and time.monotonic() - last > timeout

    def hold(self) -> bool | None:
        """Take the producer's lock: an exclusive ``flock`` on the segment, held until this
        mapping closes, released by the kernel when the process ends however it ends. ``False``:
        a live producer holds it; ``None``: this platform's shared memory has no such lock."""
        try:
            fcntl.flock(self.shm._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[attr-defined]
        except BlockingIOError:
            return False
        except OSError:
            return None
        return True

    def close(self) -> None:
        self.buf.release()
        self.shm.close()
        if self.owner:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass


_RECORD = struct.Struct("<II")  # payload length (or _MARK: skip to the ring's end), producer generation


class ShmWriter:
    """The producer of a ring. Opening it makes this producer the ring's owner: its generation
    goes into the header (``_O_WGEN``) and into every record it writes.

    Ownership is the producer's lock on the segment (``_Ring.hold``): a new producer takes
    the ring over only once the last one closed or its process ended, never while it can still
    write, however long it stalls. Where shared memory has no such lock, a producer whose
    heartbeat is ``peer_timeout_s`` old is taken over instead; the taken-over one is fenced as
    far as it can be: before writing, before publishing, before beating and before closing it
    checks that it still owns the ring. A stall between that check and its write can still
    overwrite its successor's records there (doc/specs/ipc.md §6)."""

    def __init__(self, name: str, tun: IpcTunables) -> None:
        self._ring = r = _Ring.attach(name)
        self._tun = tun
        if r.u64(_O_RCLOSED) or r.peer_lost(_O_RBEAT, tun.peer_timeout_s):
            r.close()
            raise PeerLost(f"{name}: the step-stream reader is gone")
        held = r.hold()
        alive = r.f64(_O_WBEAT) != 0.0 and not r.u64(_O_WCLOSED) and not r.peer_lost(_O_WBEAT, tun.peer_timeout_s)
        if held is False or (held is None and alive):
            r.close()
            raise RingBusy(f"{name} has a live producer")
        # The previous producer closed or ended: what it published stays for the reader, and a
        # half-written record is invisible (its length and the head are written last).
        self._gen = r.u64(_O_WGEN) + 1
        r.set_u64(_O_WGEN, self._gen)
        r.set_u64(_O_WCLOSED, 0)
        r.beat(_O_WBEAT)
        self._hb = _Heartbeat(tun.heartbeat_interval_s, self._beat)
        self._hb.start()

    def _owns(self) -> bool:
        return self._ring.u64(_O_WGEN) == self._gen

    def _beat(self) -> None:
        if self._owns():
            self._ring.beat(_O_WBEAT)

    def _check(self) -> None:
        r = self._ring
        if not self._owns():
            raise PeerLost("another producer took this step-stream ring over")
        if r.u64(_O_RCLOSED) or r.peer_lost(_O_RBEAT, self._tun.peer_timeout_s):
            raise PeerLost("step-stream reader is gone")

    def write(self, data: bytes, timeout: float | None = None) -> None:
        r = self._ring
        cap = r.capacity
        size = _pad(_RECORD.size + len(data))
        if size > cap:
            raise ValueError(f"record of {len(data)} bytes exceeds ring capacity {cap}")
        self._check()  # not into a ring nobody reads, or one another producer owns now
        deadline = None if timeout is None else time.monotonic() + timeout
        backoff = _Backoff()
        while True:
            head, tail = r.u64(_O_HEAD), r.u64(_O_TAIL)
            pos = head % cap
            skip = cap - pos if cap - pos < size else 0
            if cap - (head - tail) >= skip + size:
                break
            if skip and cap - (head - tail) >= skip:
                # Publish padding separately: padding + record may exceed capacity.
                # Only the reader advances tail, after consuming any queued records.
                self._check()
                if skip >= 4:
                    struct.pack_into("<I", r.buf, _HEADER + pos, _MARK)
                r.set_u64(_O_HEAD, head + skip)
                continue
            self._check()
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("step-stream reader is not keeping up")
            backoff.wait()
        self._check()
        if skip:
            if skip >= 4:
                struct.pack_into("<I", r.buf, _HEADER + pos, _MARK)
            head += skip
            pos = 0
        r.buf[_HEADER + pos + _RECORD.size:_HEADER + pos + _RECORD.size + len(data)] = data
        _RECORD.pack_into(r.buf, _HEADER + pos, len(data), self._gen)
        if not self._owns():  # taken over meanwhile: this record is never published
            raise PeerLost("another producer took this step-stream ring over")
        r.set_u64(_O_HEAD, head + size)
        r.beat(_O_WBEAT)

    def close(self) -> None:
        self._hb.stop()
        if self._owns():  # a fenced producer must not mark its successor closed
            self._ring.set_u64(_O_WCLOSED, 1)
        self._ring.close()

    def abort(self) -> None:
        self.close()


class ShmReader:
    def __init__(self, name: str, tun: IpcTunables, *, replaceable: bool = False) -> None:
        self._ring = _Ring.create(name, tun.shm_capacity)
        self._tun = tun
        self._replaceable = replaceable
        self.name = name
        self._gen = 0
        self._ring.beat(_O_RBEAT)
        self._hb = _Heartbeat(tun.heartbeat_interval_s, lambda: self._ring.beat(_O_RBEAT))
        self._hb.start()

    def read(self, timeout: float | None = None) -> bytes | None:
        r = self._ring
        cap = r.capacity
        deadline = None if timeout is None else time.monotonic() + timeout
        backoff = _Backoff()
        while True:
            head, tail = r.u64(_O_HEAD), r.u64(_O_TAIL)
            if tail < head:
                pos = tail % cap
                n = struct.unpack_from("<I", r.buf, _HEADER + pos)[0] if cap - pos >= 4 else _MARK
                if n == _MARK:
                    r.set_u64(_O_TAIL, tail + (cap - pos))
                    continue
                gen = _RECORD.unpack_from(r.buf, _HEADER + pos)[1]
                start = _HEADER + pos + _RECORD.size
                data = bytes(r.buf[start:start + n])
                r.set_u64(_O_TAIL, tail + _pad(_RECORD.size + n))
                if gen < self._gen:
                    continue  # a fenced producer's record, after records of its successor
                self._gen = gen
                return data
            if not self._replaceable:
                if r.u64(_O_WCLOSED):
                    raise StreamClosed("producer closed the stream")
                if r.peer_lost(_O_WBEAT, self._tun.peer_timeout_s):
                    raise PeerLost("step-stream writer is gone")
            if deadline is not None and time.monotonic() > deadline:
                return None
            backoff.wait()

    @property
    def writer_generation(self) -> int:
        """The producer of the record last returned: 1 for the ring's first, one more for each
        producer that took over. Records a producer left in the ring keep its generation."""
        return self._gen

    def close(self) -> None:
        self._hb.stop()
        self._ring.set_u64(_O_RCLOSED, 1)
        self._ring.close()


# --------------------------------------------------------------------- tcp

_FRAME = struct.Struct("<QQQB")  # producer start (ns), producer nonce, sequence number, kind (0 data, 1 close)


def _tune(sock: zmq.Socket[bytes], tun: IpcTunables) -> None:
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.SNDHWM, tun.stream_hwm)
    sock.setsockopt(zmq.RCVHWM, tun.stream_hwm)
    sock.setsockopt(zmq.SNDBUF, tun.stream_socket_buffer)
    sock.setsockopt(zmq.RCVBUF, tun.stream_socket_buffer)
    # Liveness comes from the kernel, not ZMTP heartbeats: when the consumer is
    # slow, ZMQ stops reading the socket and PINGs queued behind data would time
    # out a healthy link. A killed peer closes its socket (FIN/RST); a vanished
    # host is caught by keepalive probes or by unacknowledged data (TCP_MAXRT).
    timeout_ms = max(1000, int(tun.peer_timeout_s * 1000))
    sock.setsockopt(zmq.TCP_KEEPALIVE, 1)
    sock.setsockopt(zmq.TCP_KEEPALIVE_IDLE, max(1, timeout_ms // 2000))
    sock.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 1)
    sock.setsockopt(zmq.TCP_KEEPALIVE_CNT, max(1, timeout_ms // 2000))
    sock.setsockopt(zmq.TCP_MAXRT, timeout_ms)


class _Link:
    """Tracks whether the single peer of a socket is connected, from monitor events."""

    def __init__(self, sock: zmq.Socket[bytes]) -> None:
        self._mon = sock.get_monitor_socket(zmq.EVENT_HANDSHAKE_SUCCEEDED | zmq.EVENT_DISCONNECTED)
        self.seen = False
        self.lost = False

    def update(self) -> None:
        while self._mon.poll(0, zmq.POLLIN):
            ev = recv_monitor_message(self._mon)["event"]
            if ev == zmq.EVENT_HANDSHAKE_SUCCEEDED:
                self.seen, self.lost = True, False
            elif ev == zmq.EVENT_DISCONNECTED and self.seen:
                self.lost = True

    def close(self) -> None:
        self._mon.close(0)


class TcpWriter:
    def __init__(self, zctx: zmq.Context[zmq.Socket[bytes]], address: str, tun: IpcTunables) -> None:
        self._sock = zctx.socket(zmq.PUSH)
        _tune(self._sock, tun)
        self._sock.setsockopt(zmq.IMMEDIATE, 1)
        self._link = _Link(self._sock)
        self._sock.connect(address)
        # Orders the producers of one reader: the one that started later replaces the earlier.
        self._key = (time.time_ns(), secrets.randbits(64))
        self._seq = 0
        self._tun = tun

    @property
    def connected(self) -> bool:
        """The reader accepted the connection and has not dropped it."""
        self._link.update()
        return self._link.seen and not self._link.lost

    def _send(self, kind: int, data: bytes, timeout: float | None) -> None:
        frame = _FRAME.pack(*self._key, self._seq, kind) + data
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            self._link.update()
            if self._link.lost:
                raise PeerLost("step-stream reader disconnected")
            try:
                self._sock.send(frame, flags=zmq.NOBLOCK)
                break
            except zmq.Again:
                pass
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("step-stream reader is not keeping up")
            self._sock.poll(10, zmq.POLLOUT)
        self._seq += 1

    def write(self, data: bytes, timeout: float | None = None) -> None:
        self._send(0, data, timeout)

    def close(self) -> None:
        """Orderly end: queued records and the close marker are delivered first."""
        linger = 0
        try:
            self._send(1, b"", timeout=self._tun.close_timeout_s)
            linger = int(self._tun.close_timeout_s * 1000)
        except (PeerLost, TimeoutError):
            pass
        self._link.close()
        self._sock.close(linger)

    def abort(self) -> None:
        """Drop the connection without delivering what is queued (the reader is gone or replaced)."""
        self._link.close()
        self._sock.close(0)


type _Producer = tuple[int, int]
"""(start time in ns, nonce): a later start is a later producer."""


class TcpReader:
    """The consumer end over TCP. ZMQ fair-queues the connections of several producers, so their
    frames may arrive interleaved. A replaceable reader keeps one current producer, in order:
    frames of earlier producers are dropped; frames of a later one wait until the current one
    closed (its close frame was read) or went quiet for ``peer_timeout_s``. Then the later one
    becomes current. So a producer's own records are never mixed with another's, and the tail
    of one that closed in order is not lost to its successor. Waiting frames are bounded by
    ``ipc.stream_hold_bytes``: past that, two producers write at once, and ``read`` raises
    ``PeerLost``."""

    def __init__(self, zctx: zmq.Context[zmq.Socket[bytes]], bind_host: str, tun: IpcTunables, *,
                 replaceable: bool = False) -> None:
        self._sock = zctx.socket(zmq.PULL)
        _tune(self._sock, tun)
        self._link = _Link(self._sock)
        self._sock.bind(f"tcp://{bind_host}:*")
        self.address = self._sock.getsockopt_string(zmq.LAST_ENDPOINT)
        self._tun = tun
        self._replaceable = replaceable
        self._current: _Producer | None = None
        self._expect = 0
        self._closed = False
        """The current producer sent its close frame."""
        self._last = 0.0
        """When the current producer's last frame arrived."""
        self._ready: deque[bytes] = deque()
        self._later: dict[_Producer, list[tuple[int, int, bytes]]] = {}
        self._held = 0
        """Bytes in ``_later``."""
        self.writer_generation = 0
        """The producer of the record last returned: 1 for the first, one more per replacement."""

    def _take(self, seq: int, kind: int, payload: bytes) -> None:
        """A frame of the current producer."""
        if seq != self._expect:
            raise PeerLost(f"step stream lost records: expected #{self._expect}, got #{seq}")
        self._expect += 1
        self._last = time.monotonic()
        if kind == 1:
            if not self._replaceable:
                raise StreamClosed("producer closed the stream")
            self._closed = True
        else:
            self._ready.append(payload)

    def _adopt(self, producer: _Producer) -> None:
        self._current, self._expect, self._closed = producer, 0, False
        self._last = time.monotonic()
        self.writer_generation += 1

    def _frame(self, raw: bytes) -> None:
        start, nonce, seq, kind = _FRAME.unpack_from(raw)
        producer, payload = (start, nonce), raw[_FRAME.size:]
        current = self._current
        if current is None:
            self._adopt(producer)
            current = producer
        if producer == current:
            self._take(seq, kind, payload)
        elif not self._replaceable:
            raise PeerLost("a second producer writes to a single-producer step stream")
        elif producer > current:
            self._later.setdefault(producer, []).append((seq, kind, payload))
            self._held += len(raw)
            if self._held > self._tun.stream_hold_bytes:
                raise PeerLost(f"a later producer wrote {self._held} bytes while the current one did not end: "
                               "two producers write to this step stream at once")
        # else: an earlier producer's frame, queued before it was replaced

    def _replace_if_due(self) -> None:
        """Make the earliest waiting producer current once the current one closed or went quiet.
        How many frames wait proves nothing: the current one's tail may still be queued."""
        while self._later and not self._ready:
            if not (self._closed or time.monotonic() - self._last > self._tun.peer_timeout_s):
                return
            producer = min(self._later)
            frames = self._later.pop(producer)
            self._held -= sum(_FRAME.size + len(payload) for _, _, payload in frames)
            self._adopt(producer)
            for seq, kind, payload in frames:
                self._take(seq, kind, payload)

    def read(self, timeout: float | None = None) -> bytes | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._ready:
                return self._ready.popleft()
            # Never wait past the deadline: read(0) only looks.
            wait_ms = 10 if deadline is None else min(10, max(0, math.ceil((deadline - time.monotonic()) * 1000)))
            if self._sock.poll(wait_ms, zmq.POLLIN):
                self._frame(self._sock.recv())
                self._replace_if_due()
                continue
            self._link.update()
            if self._replaceable:
                self._replace_if_due()
                if self._ready:
                    continue
            elif self._link.lost:
                # Data and monitor events come through separate queues: the last records may
                # be ready after the empty poll, before the disconnect was seen.
                if self._sock.poll(0, zmq.POLLIN):
                    continue
                raise PeerLost("step-stream writer disconnected")
            if deadline is not None and time.monotonic() > deadline:
                return None

    def close(self) -> None:
        self._link.close()
        self._sock.close(0)


# ------------------------------------------------------------- selection

def open_reader(
    zctx: zmq.Context[zmq.Socket[bytes]], *, local: HostId, peer: HostId, bind_host: str, shm_name: str,
    tun: IpcTunables, replaceable: bool = False,
) -> tuple[ShmReader | TcpReader, StreamAddress]:
    """The consumer end. ``shm_name`` comes from the node agent of this host. A ``replaceable``
    reader takes the next producer when one closes or dies, instead of ending the stream."""
    if backend_for(local, peer) == "shm":
        return ShmReader(shm_name, tun, replaceable=replaceable), StreamAddress(local, shm=shm_name)
    reader = TcpReader(zctx, bind_host, tun, replaceable=replaceable)
    return reader, StreamAddress(local, tcp=reader.address)


def open_writer(
    zctx: zmq.Context[zmq.Socket[bytes]], *, local: HostId, address: StreamAddress, tun: IpcTunables,
) -> ShmWriter | TcpWriter:
    if backend_for(local, address.host) == "shm":
        if address.shm is None:
            raise ValueError(f"reader on {address.host} offers no shared-memory ring")
        return ShmWriter(address.shm, tun)
    if address.tcp is None:
        raise ValueError(f"reader on {address.host} offers no TCP endpoint")
    return TcpWriter(zctx, address.tcp, tun)
