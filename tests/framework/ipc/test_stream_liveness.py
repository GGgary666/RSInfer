"""A peer that holds the GIL past ``ipc.peer_timeout_s`` is busy, not gone (shared memory): its
heartbeat thread stops, but the kernel still holds its process's lock on the ring
(doc/framework-bugs/2026-11-10-a-long-gil-hold-looks-like-a-dead-peer.md). A killed peer is still
reported (``test_streams.py``). Over TCP the kernel decides already, so these run on shm only.
Linux only, as the engine (open file description locks)."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
import uuid

import pytest
import zmq

from core.common.ids import HostId
from core.ipc import PeerLost, StreamClosed, open_reader, open_writer
from tests.framework.ipc.peers import BIND, TUN, read_holding_the_gil, write_holding_the_gil

SPAWN = mp.get_context("spawn")
HOST = "host-a"
HOLD_S = 4 * TUN.peer_timeout_s


def shm_name() -> str:
    return f"rsinfer-test-{uuid.uuid4().hex[:12]}"


def test_a_writer_waits_for_a_reader_that_holds_the_gil() -> None:
    zctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
    q = SPAWN.Queue()
    n = 400
    child = SPAWN.Process(target=read_holding_the_gil, args=(q, HOST, HOST, shm_name(), HOLD_S, n))
    child.start()
    try:
        writer = open_writer(zctx, local=HostId(HOST), address=q.get(timeout=20), tun=TUN)
        writer.write(b"0" * 100, timeout=5)
        assert q.get(timeout=20) == "holding"
        t0, sent = time.monotonic(), 1
        while sent < n:  # through the hold: the ring fills, the writer waits, nothing is lost
            try:
                writer.write(b"x" * 1000, timeout=0.2)
                sent += 1
            except TimeoutError:
                pass
        assert time.monotonic() - t0 > HOLD_S / 2  # the hold did stall it
        assert q.get(timeout=20) == "released"
        assert q.get(timeout=20) == n
        writer.close()
    finally:
        assert child.pid is not None
        os.kill(child.pid, signal.SIGKILL)
        child.join(10)


@pytest.mark.parametrize("second", [False, True], ids=["alone", "after a refused second writer"])
def test_a_reader_waits_for_a_writer_that_holds_the_gil(second: bool) -> None:
    """``second``: the writer's process tried a second writer of the ring first, which was refused
    and closed its descriptor of the segment. With POSIX record locks that let the first writer's
    lock go too, and the hold looked like death (Codex's review of #55)."""
    zctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
    reader, addr = open_reader(zctx, local=HostId(HOST), peer=HostId(HOST), bind_host=BIND, shm_name=shm_name(),
                               tun=TUN)
    events = SPAWN.Queue()
    child = SPAWN.Process(target=write_holding_the_gil, args=(addr, HOST, events, HOLD_S, second))
    child.start()
    try:
        assert reader.read(timeout=20) == b"before"
        if second:
            assert events.get(timeout=20) == "second writer refused"
        assert events.get(timeout=20) == "holding"
        t0 = time.monotonic()
        got = None
        while got is None:  # PeerLost here was the bug
            got = reader.read(timeout=0.1)
        assert got == b"after" and time.monotonic() - t0 > TUN.peer_timeout_s
        assert events.get(timeout=20) == "wrote after"
        try:
            while reader.read(timeout=5) is not None:
                pass
        except StreamClosed:
            pass
    finally:
        assert child.pid is not None
        os.kill(child.pid, signal.SIGKILL)
        child.join(10)
        reader.close()


def test_a_reader_holding_the_gil_and_then_killed_is_reported() -> None:
    """The lock goes with the process: a reader killed in the middle of a hold is lost."""
    zctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
    q = SPAWN.Queue()
    child = SPAWN.Process(target=read_holding_the_gil, args=(q, HOST, HOST, shm_name(), 60.0, 10))
    child.start()
    try:
        writer = open_writer(zctx, local=HostId(HOST), address=q.get(timeout=20), tun=TUN)
        writer.write(b"0" * 100, timeout=5)
        assert q.get(timeout=20) == "holding"
        time.sleep(2 * TUN.peer_timeout_s)
        writer.write(b"1" * 100, timeout=5)  # stale heartbeat, held lock: alive
        assert child.pid is not None
        os.kill(child.pid, signal.SIGKILL)
        child.join(10)
        t0 = time.monotonic()
        lost = False
        while time.monotonic() - t0 < 10 and not lost:
            try:
                writer.write(b"x" * 1000, timeout=0.2)
            except TimeoutError:
                pass
            except PeerLost:
                lost = True
        assert lost and time.monotonic() - t0 < TUN.peer_timeout_s + 2
    finally:
        if child.is_alive():
            assert child.pid is not None
            os.kill(child.pid, signal.SIGKILL)
        child.join(10)
