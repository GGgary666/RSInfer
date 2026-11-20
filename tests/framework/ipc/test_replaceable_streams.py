"""ipc.md §6: a replaceable reader outlives its producers, on shm and TCP alike."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
import uuid

import msgspec
import pytest
import zmq

from core.common.ids import HostId
from core.ipc import IpcTunables, PeerLost, RingBusy, StreamAddress, open_reader, open_writer
from core.ipc.stream import ShmReader, ShmWriter, TcpReader, TcpWriter
from tests.framework.ipc.peers import BIND, PATIENT, TUN, write_records, write_then_wait

SPAWN = mp.get_context("spawn")
BACKENDS = {"shm": ("host-a", "host-a"), "tcp": ("host-a", "host-b")}


@pytest.fixture(params=sorted(BACKENDS))
def hosts(request: pytest.FixtureRequest) -> tuple[str, str]:
    return BACKENDS[request.param]


@pytest.fixture
def zctx() -> zmq.Context[zmq.Socket[bytes]]:
    return zmq.Context()


def reader_for(zctx: zmq.Context[zmq.Socket[bytes]], hosts: tuple[str, str], *, replaceable: bool = True,
               tun: IpcTunables = PATIENT) -> tuple[ShmReader | TcpReader, StreamAddress]:
    rhost, whost = hosts
    return open_reader(zctx, local=HostId(rhost), peer=HostId(whost), bind_host=BIND,
                       shm_name=f"rsinfer-test-{uuid.uuid4().hex[:12]}", tun=tun,
                       replaceable=replaceable)


def writer_for(zctx: zmq.Context[zmq.Socket[bytes]], hosts: tuple[str, str], address: StreamAddress,
               tun: IpcTunables = PATIENT) -> ShmWriter | TcpWriter:
    return open_writer(zctx, local=HostId(hosts[1]), address=address, tun=tun)


def take(reader: ShmReader | TcpReader, n: int) -> list[bytes]:
    out: list[bytes] = []
    deadline = time.monotonic() + 20
    while len(out) < n and time.monotonic() < deadline:
        data = reader.read(timeout=0.1)
        if data is not None:
            out.append(data)
    return out


def test_the_next_producer_takes_over_after_one_closes(zctx: zmq.Context[zmq.Socket[bytes]],
                                                        hosts: tuple[str, str]) -> None:
    reader, address = reader_for(zctx, hosts)
    try:
        first = writer_for(zctx, hosts, address)
        first.write(b"a0", timeout=5)
        first.write(b"a1", timeout=5)
        first.close()
        assert take(reader, 2) == [b"a0", b"a1"]
        assert reader.read(timeout=0.2) is None  # a producer that left is not the end of the stream
        second = writer_for(zctx, hosts, address)
        second.write(b"b0", timeout=5)
        second.write(b"b1", timeout=5)
        assert take(reader, 2) == [b"b0", b"b1"]
        assert reader.writer_generation == 2
        second.close()
    finally:
        reader.close()


def test_a_killed_producer_is_replaced(zctx: zmq.Context[zmq.Socket[bytes]], hosts: tuple[str, str]) -> None:
    """Shared memory: the kernel drops a producer's lock when its process ends, so the next one
    takes over at once. TCP: the next one's records wait until the killed one is quiet."""
    reader, address = reader_for(zctx, hosts, tun=TUN)
    child = SPAWN.Process(target=write_records, args=(address, hosts[1], [10] * 3, False, 60.0))
    child.start()
    try:
        assert len(take(reader, 3)) == 3
        assert child.pid is not None
        os.kill(child.pid, signal.SIGKILL)
        child.join(10)
        second = writer_for(zctx, hosts, address, TUN)  # no RingBusy: that process has ended
        second.write(b"b0", timeout=5)
        assert take(reader, 1) == [b"b0"]
        assert reader.writer_generation == 2
        second.close()
    finally:
        child.join(10)
        reader.close()


def test_late_records_of_a_replaced_tcp_producer_are_dropped(zctx: zmq.Context[zmq.Socket[bytes]]) -> None:
    """Over TCP the old producer's records can still be queued when the new one starts: once the
    reader took the new one, they are dropped, never interleaved with its records."""
    hosts = BACKENDS["tcp"]
    reader, address = reader_for(zctx, hosts)
    try:
        old = writer_for(zctx, hosts, address)
        old.write(b"a0", timeout=5)
        assert take(reader, 1) == [b"a0"]
        new = writer_for(zctx, hosts, address)
        new.write(b"b0", timeout=5)
        assert take(reader, 1) == [b"b0"]
        old.write(b"a1", timeout=5)
        new.write(b"b1", timeout=5)
        assert take(reader, 1) == [b"b1"]
        assert reader.read(timeout=0.3) is None
        assert reader.writer_generation == 2
        old.abort()
        new.close()
    finally:
        reader.close()


def test_a_single_producer_stream_still_refuses_a_second_producer(zctx: zmq.Context[zmq.Socket[bytes]]) -> None:
    hosts = BACKENDS["tcp"]
    reader, address = reader_for(zctx, hosts, replaceable=False)
    first = writer_for(zctx, hosts, address)
    second: ShmWriter | TcpWriter | None = None
    try:
        first.write(b"a0", timeout=5)
        assert take(reader, 1) == [b"a0"]
        second = writer_for(zctx, hosts, address)
        second.write(b"b0", timeout=5)
        with pytest.raises(PeerLost, match="second producer"):
            reader.read(timeout=10)
    finally:
        first.abort()
        if second is not None:
            second.abort()
        reader.close()


def test_a_shm_producer_notices_a_closed_reader_before_writing(zctx: zmq.Context[zmq.Socket[bytes]]) -> None:
    """Without the check it would fill a ring nobody reads before noticing."""
    hosts = BACKENDS["shm"]
    reader, address = reader_for(zctx, hosts)
    writer = writer_for(zctx, hosts, address)
    writer.write(b"a0", timeout=5)
    reader.close()
    with pytest.raises(PeerLost):
        writer.write(b"a1", timeout=5)
    writer.close()


def test_records_a_producer_left_behind_keep_its_generation(zctx: zmq.Context[zmq.Socket[bytes]],
                                                            hosts: tuple[str, str]) -> None:
    """The first producer closes with records unread and the next one writes before the reader
    gets to them: each record comes with the generation of the producer that wrote it, and over
    TCP the closed producer's tail is not lost to the next one (its records wait)."""
    reader, address = reader_for(zctx, hosts)
    try:
        first = writer_for(zctx, hosts, address)
        first.write(b"old:0", timeout=5)
        first.write(b"old:1", timeout=5)
        first.close()
        second = writer_for(zctx, hosts, address)
        second.write(b"new:0", timeout=5)
        got: list[tuple[bytes, int]] = []
        for _ in range(3):
            got.append((take(reader, 1)[0], reader.writer_generation))
        assert got == [(b"old:0", 1), (b"old:1", 1), (b"new:0", 2)]
        second.close()
    finally:
        reader.close()


def test_a_stalled_shm_producer_keeps_its_ring_until_it_closes(zctx: zmq.Context[zmq.Socket[bytes]]) -> None:
    """A stall is not a death: a producer stopped past ``peer_timeout_s``, heartbeat and all, still
    holds the ring, so nobody can take it over and be overwritten when it goes on."""
    hosts = BACKENDS["shm"]
    reader, address = reader_for(zctx, hosts, tun=TUN)
    events, go = SPAWN.Queue(), SPAWN.Event()
    child = SPAWN.Process(target=write_then_wait, args=(address, hosts[1], events, go))
    child.start()
    assert child.pid is not None
    try:
        assert events.get(timeout=20) == "wrote old:0"
        assert take(reader, 1) == [b"old:0"]
        os.kill(child.pid, signal.SIGSTOP)
        time.sleep(TUN.peer_timeout_s + 0.3)  # its heartbeat is stale now
        with pytest.raises(RingBusy):
            writer_for(zctx, hosts, address, TUN)
        os.kill(child.pid, signal.SIGCONT)
        go.set()
        assert events.get(timeout=20) == "wrote old:1"
        assert events.get(timeout=20) == "closed"
        successor = writer_for(zctx, hosts, address, TUN)
        successor.write(b"new:0", timeout=5)
        got: list[tuple[bytes, int]] = []
        for _ in range(2):
            got.append((take(reader, 1)[0], reader.writer_generation))
        assert got == [(b"old:1", 1), (b"new:0", 2)]
        successor.close()
    finally:
        os.kill(child.pid, signal.SIGCONT)
        child.join(10)
        reader.close()


def test_no_producer_takes_over_between_a_producers_check_and_its_write(
        zctx: zmq.Context[zmq.Socket[bytes]], monkeypatch: pytest.MonkeyPatch) -> None:
    """The interleaving Codex found on #23: the producer passes its ownership check, then stalls
    (heartbeat stopped) past ``peer_timeout_s`` before it writes. A successor trying to take the
    ring over right then is refused, so the stalled write cannot overwrite a successor's record."""
    hosts = BACKENDS["shm"]
    reader, address = reader_for(zctx, hosts, tun=TUN)
    stalled = writer_for(zctx, hosts, address, TUN)
    check = getattr(stalled, "_check")
    refused: list[str] = []

    def check_then_stall() -> None:  # at every check, so also right before the bytes are written
        check()
        getattr(stalled, "_hb").stop()
        time.sleep(TUN.peer_timeout_s + 0.2)
        try:
            writer_for(zctx, hosts, address, TUN).close()
        except RingBusy as e:
            refused.append(str(e))

    monkeypatch.setattr(stalled, "_check", check_then_stall)
    try:
        stalled.write(b"old:late", timeout=5)
        assert refused and len(refused) == 2  # at the check that opens the write, and right before the bytes
        assert take(reader, 1) == [b"old:late"] and reader.writer_generation == 1
        stalled.close()
        successor = writer_for(zctx, hosts, address, TUN)
        successor.write(b"new:0", timeout=5)
        assert take(reader, 1) == [b"new:0"] and reader.writer_generation == 2
        successor.close()
    finally:
        reader.close()


def test_a_closed_tcp_producer_keeps_a_tail_longer_than_the_queue(zctx: zmq.Context[zmq.Socket[bytes]]) -> None:
    """Codex's case on #23: the old producer's tail and the new one's backlog both exceed
    ``stream_hwm``. How many frames wait proves nothing; the reader waits for the close frame."""
    hosts = BACKENDS["tcp"]
    tun = msgspec.structs.replace(TUN, stream_hwm=4, peer_timeout_s=2.0)
    reader, address = reader_for(zctx, hosts, tun=tun)
    try:
        first = writer_for(zctx, hosts, address, tun)
        for i in range(24):
            first.write(f"a{i}".encode(), timeout=5)
        first.close()
        second = writer_for(zctx, hosts, address, tun)
        for i in range(12):
            second.write(f"b{i}".encode(), timeout=5)
        second.close()
        got: list[tuple[bytes, int]] = []
        for _ in range(36):
            got.append((take(reader, 1)[0], reader.writer_generation))
        assert got == [(f"a{i}".encode(), 1) for i in range(24)] + [(f"b{i}".encode(), 2) for i in range(12)]
    finally:
        reader.close()


def test_two_tcp_producers_writing_at_once_are_reported(zctx: zmq.Context[zmq.Socket[bytes]]) -> None:
    """A later producer's frames wait for the current one to end; if it never does, the wait is
    bounded (``stream_hold_bytes``) and the reader says what happened instead of growing."""
    hosts = BACKENDS["tcp"]
    tun = msgspec.structs.replace(PATIENT, stream_hold_bytes=2000)
    reader, address = reader_for(zctx, hosts, tun=tun)
    current = writer_for(zctx, hosts, address, tun)
    later = writer_for(zctx, hosts, address, tun)
    try:
        current.write(b"a0", timeout=5)
        assert take(reader, 1) == [b"a0"]
        for _ in range(40):
            later.write(b"x" * 100, timeout=5)
        with pytest.raises(PeerLost, match="two producers"):
            take(reader, 1)
        for _ in range(20):  # broken for good: it holds nothing and takes no more frames
            with pytest.raises(PeerLost, match="two producers"):
                reader.read(timeout=0.05)
        assert getattr(reader, "_held") == 0
    finally:
        current.abort()
        later.abort()
        reader.close()


def test_a_slow_consumer_does_not_make_a_closed_producer_look_quiet(zctx: zmq.Context[zmq.Socket[bytes]]) -> None:
    """Codex's case on #23: between reads the consumer pauses longer than ``peer_timeout_s``.
    Time away from ``read`` is not the producer's silence: its tail is queued, unread."""
    hosts = BACKENDS["tcp"]
    tun = msgspec.structs.replace(TUN, stream_hwm=4, peer_timeout_s=0.05)
    reader, address = reader_for(zctx, hosts, tun=tun)
    try:
        first = writer_for(zctx, hosts, address, tun)
        first.write(b"a0", timeout=5)
        assert take(reader, 1) == [b"a0"]
        for i in range(1, 24):
            first.write(f"a{i}".encode(), timeout=5)
        first.close()
        second = writer_for(zctx, hosts, address, tun)
        for i in range(12):
            second.write(f"b{i}".encode(), timeout=5)
        second.close()
        got: list[tuple[bytes, int]] = []
        for _ in range(35):
            time.sleep(0.08)  # a consumer busy with something else
            got.append((take(reader, 1)[0], reader.writer_generation))
        assert got == [(f"a{i}".encode(), 1) for i in range(1, 24)] + [(f"b{i}".encode(), 2) for i in range(12)]
    finally:
        reader.close()


def test_any_number_of_replaced_tcp_producers_stay_replaced(zctx: zmq.Context[zmq.Socket[bytes]]) -> None:
    """Producers are ordered by when they started, not remembered in a bounded list: after 17
    replacements a late frame of the first one is still dropped."""
    hosts = BACKENDS["tcp"]
    reader, address = reader_for(zctx, hosts, tun=TUN)
    writers: list[ShmWriter | TcpWriter] = []
    try:
        for i in range(18):
            writers.append(writer_for(zctx, hosts, address, TUN))
            writers[-1].write(f"p{i}".encode(), timeout=5)
            if i >= 2:
                writers[-2].close()  # the first one stays open: it is replaced once it is quiet
            assert take(reader, 1) == [f"p{i}".encode()]
        writers[0].write(b"late", timeout=5)
        writers[-1].write(b"last", timeout=5)
        assert take(reader, 1) == [b"last"]
        assert reader.read(timeout=0.3) is None
        assert reader.writer_generation == 18
    finally:
        for w in writers:
            w.abort()
        reader.close()


def test_an_earlier_tcp_producer_never_replaces_a_later_one(zctx: zmq.Context[zmq.Socket[bytes]]) -> None:
    """ZMQ fair-queues the two connections: the earlier producer's first frame can arrive after
    the later one's. It is dropped; it does not take the stream back."""
    hosts = BACKENDS["tcp"]
    reader, address = reader_for(zctx, hosts)
    earlier = writer_for(zctx, hosts, address)
    later = writer_for(zctx, hosts, address)
    try:
        later.write(b"b0", timeout=5)
        assert take(reader, 1) == [b"b0"]
        earlier.write(b"a0", timeout=5)
        later.write(b"b1", timeout=5)
        assert take(reader, 1) == [b"b1"]
        assert reader.read(timeout=0.3) is None
        assert reader.writer_generation == 1
    finally:
        earlier.abort()
        later.abort()
        reader.close()
