"""ipc.md §6: a replaceable reader outlives its producers, on shm and TCP alike."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
import uuid

import pytest
import zmq

from core.common.ids import HostId
from core.ipc import IpcTunables, PeerLost, RingBusy, StreamAddress, open_reader, open_writer
from core.ipc.stream import ShmReader, ShmWriter, TcpReader, TcpWriter
from tests.framework.ipc.peers import BIND, PATIENT, TUN, write_records

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
                       shm_name=f"agx-test-{uuid.uuid4().hex[:12]}", tun=tun,
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


def test_a_killed_producer_is_replaced_once_it_is_known_dead(zctx: zmq.Context[zmq.Socket[bytes]],
                                                             hosts: tuple[str, str]) -> None:
    """The shm ring refuses a second producer while the first still proves it is alive, so two
    never write it at once; once its heartbeat is ``peer_timeout_s`` old, the next one takes over."""
    reader, address = reader_for(zctx, hosts, tun=TUN)
    child = SPAWN.Process(target=write_records, args=(address, hosts[1], [10] * 3, False, 60.0))
    child.start()
    try:
        assert len(take(reader, 3)) == 3
        assert child.pid is not None
        os.kill(child.pid, signal.SIGKILL)
        if address.shm is not None:
            with pytest.raises(RingBusy):
                writer_for(zctx, hosts, address, TUN)
        deadline = time.monotonic() + 10
        while True:
            try:
                second = writer_for(zctx, hosts, address, TUN)
                break
            except RingBusy:
                assert time.monotonic() < deadline
                time.sleep(0.05)
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
    try:
        writer_for(zctx, hosts, address).write(b"a0", timeout=5)
        writer_for(zctx, hosts, address).write(b"b0", timeout=5)
        with pytest.raises(PeerLost, match="second producer"):
            take(reader, 2)
    finally:
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
