"""ipc.md §5 tests 1, 2, 4 for step streams: identical behaviour on shm and TCP."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import struct
import time
import uuid

import pytest
import zmq

from core.common.ids import HostId
from core.ipc import PeerLost, StreamClosed, backend_for, open_reader, open_writer
from tests.ipc.peers import BIND, TUN, read_forever, write_records

SPAWN = mp.get_context("spawn")

# (reader host, writer host): same host -> shm, two fake hosts -> TCP
BACKENDS = {"shm": ("host-a", "host-a"), "tcp": ("host-a", "host-b")}


@pytest.fixture(params=sorted(BACKENDS))
def hosts(request: pytest.FixtureRequest) -> tuple[str, str]:
    return BACKENDS[request.param]


@pytest.fixture
def zctx() -> zmq.Context[zmq.Socket[bytes]]:
    return zmq.Context()


def shm_name() -> str:
    return f"agx-test-{uuid.uuid4().hex[:12]}"


def test_backend_is_chosen_by_host_id_not_address() -> None:
    assert backend_for(HostId("host-a"), HostId("host-a")) == "shm"
    assert backend_for(HostId("host-a"), HostId("host-b")) == "tcp"


def test_ordered_and_lossless_across_ring_wraps(zctx: zmq.Context[zmq.Socket[bytes]], hosts: tuple[str, str]) -> None:
    rhost, whost = hosts
    reader, addr = open_reader(zctx, local=HostId(rhost), peer=HostId(whost), bind_host=BIND,
                               shm_name=shm_name(), tun=TUN)
    sizes = [(i * 7919) % 6000 for i in range(1500)]
    child = SPAWN.Process(target=write_records, args=(addr, whost, sizes, True, 0.0))
    child.start()
    try:
        got: list[bytes] = []
        with pytest.raises(StreamClosed):
            while True:
                rec = reader.read(timeout=20)
                assert rec is not None, "writer stalled"
                got.append(rec)
        assert [struct.unpack_from("<Q", r)[0] for r in got] == list(range(len(sizes)))
        for i, rec in enumerate(got):
            assert rec[16:] == bytes([i % 251]) * sizes[i]
    finally:
        child.join(10)
        reader.close()


def test_slow_consumer_blocks_the_producer(zctx: zmq.Context[zmq.Socket[bytes]], hosts: tuple[str, str]) -> None:
    rhost, whost = hosts
    reader, addr = open_reader(zctx, local=HostId(rhost), peer=HostId(whost), bind_host=BIND,
                               shm_name=shm_name(), tun=TUN)
    n, size = 400, 4096
    child = SPAWN.Process(target=write_records, args=(addr, whost, [size] * n, True, 0.0))
    child.start()
    try:
        read_at: list[float] = []
        written_at: list[float] = []
        while len(read_at) < n:
            rec = reader.read(timeout=20)
            assert rec is not None
            read_at.append(time.monotonic())
            written_at.append(struct.unpack_from("<d", rec, 8)[0])
            time.sleep(0.002)
        # The producer may run ahead only by what fits in the ring / socket buffers.
        bound = 20 if whost == rhost else 150
        lead = max(i - max((j for j, t in enumerate(read_at) if t <= w), default=-1)
                   for i, w in enumerate(written_at))
        assert lead <= bound, f"producer ran {lead} records ahead"
    finally:
        child.join(10)
        reader.close()


def test_reader_reports_a_killed_writer(zctx: zmq.Context[zmq.Socket[bytes]], hosts: tuple[str, str]) -> None:
    rhost, whost = hosts
    reader, addr = open_reader(zctx, local=HostId(rhost), peer=HostId(whost), bind_host=BIND,
                               shm_name=shm_name(), tun=TUN)
    child = SPAWN.Process(target=write_records, args=(addr, whost, [10] * 3, False, 60.0))
    child.start()
    try:
        for _ in range(3):
            assert reader.read(timeout=20) is not None
        assert child.pid is not None
        os.kill(child.pid, signal.SIGKILL)
        t0 = time.monotonic()
        with pytest.raises(PeerLost):
            while time.monotonic() - t0 < 10:
                reader.read(timeout=0.1)
        assert time.monotonic() - t0 < TUN.peer_timeout_s + 2
    finally:
        child.join(10)
        reader.close()


def test_writer_reports_a_killed_reader(zctx: zmq.Context[zmq.Socket[bytes]], hosts: tuple[str, str]) -> None:
    rhost, whost = hosts
    q = SPAWN.Queue()
    name = shm_name()
    child = SPAWN.Process(target=read_forever, args=(q, rhost, whost, name))
    child.start()
    try:
        addr = q.get(timeout=20)
        writer = open_writer(zctx, local=HostId(whost), address=addr, tun=TUN)
        writer.write(b"x" * 100, timeout=5)
        assert child.pid is not None
        os.kill(child.pid, signal.SIGKILL)
        t0 = time.monotonic()
        with pytest.raises(PeerLost):
            while time.monotonic() - t0 < 10:
                try:
                    writer.write(b"x" * 4096, timeout=0.2)
                except TimeoutError:
                    pass  # ring full; the reader is only declared lost after peer_timeout_s
        assert time.monotonic() - t0 < TUN.peer_timeout_s + 2
    finally:
        child.join(10)
        if hosts[0] == hosts[1]:
            from multiprocessing.shared_memory import SharedMemory
            try:
                SharedMemory(name=name).unlink()
            except FileNotFoundError:
                pass
