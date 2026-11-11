"""Code that runs in child processes for the IPC tests."""

from __future__ import annotations

import asyncio
import struct
import time
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from typing import Any

import msgspec
import zmq
import zmq.asyncio

from core.common.ids import Endpoint, HostId, ProcId
from core.common.messages import Envelope, SchemaRegistry
from core.common.result import Ok, Result
from core.ipc import IpcTunables, PeerLost, RpcServer, StreamAddress, open_reader, open_writer

BIND = "127.0.0.1"  # test data: the address of every fake host
TUN = IpcTunables(heartbeat_interval_s=0.05, peer_timeout_s=0.5, stream_hwm=4,
                  stream_socket_buffer=1 << 16, shm_capacity=1 << 16)
"""For the tests that kill a peer: it is declared lost soon."""
PATIENT = msgspec.structs.replace(TUN, peer_timeout_s=5.0)
"""For the tests whose peers all live: a runner busy under ``pytest -n auto`` can stall a
process past 0.5 s, and a live peer must not be declared lost then."""
LOADED = msgspec.structs.replace(PATIENT, peer_timeout_s=30.0)
"""For both ends of a stream that carries many records between the test and a child process:
on a loaded CI runner either side's heartbeat thread can go unscheduled for seconds."""


class Ping(msgspec.Struct, frozen=True):
    n: int


class Pong(msgspec.Struct, frozen=True):
    n: int


class Unknown(msgspec.Struct, frozen=True):
    x: int


def registry() -> SchemaRegistry:
    r = SchemaRegistry()
    r.register("test.ping@1", Ping)
    r.register("test.pong@1", Pong)
    return r


def ep(host: str, role: str, channel: str = "rpc") -> Endpoint:
    return Endpoint(ProcId(HostId(host), role, 0), channel)


def stamp(i: int, size: int) -> bytes:
    """Record ``i``: index, completion-of-write time, padding."""
    return struct.pack("<Qd", i, time.monotonic()) + bytes([i % 251]) * size


def write_records(address: StreamAddress, host: str, sizes: list[int], close: bool, then_sleep: float) -> None:
    zctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
    w = open_writer(zctx, local=HostId(host), address=address, tun=LOADED)  # no test kills its reader
    for i, size in enumerate(sizes):
        w.write(stamp(i, size), timeout=30)
    if close:
        w.close()
        zctx.term()  # blocks until queued records are handed to the kernel
    time.sleep(then_sleep)


def write_then_wait(address: StreamAddress, host: str, events: Queue[str], go: Event) -> None:
    """Writes one record, waits for ``go`` (the test stops and continues this process in
    between, as a stall would), then tries one more and closes; reports each outcome."""
    zctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
    w = open_writer(zctx, local=HostId(host), address=address, tun=TUN)
    w.write(b"old:0", timeout=5)
    events.put("wrote old:0")
    go.wait(60)
    try:
        w.write(b"old:1", timeout=1)
        events.put("wrote old:1")
    except PeerLost:
        events.put("fenced")
    w.close()
    events.put("closed")


def read_forever(q: Queue[Any], host: str, peer: str, shm_name: str) -> None:
    zctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
    r, addr = open_reader(zctx, local=HostId(host), peer=HostId(peer), bind_host=BIND, shm_name=shm_name, tun=TUN)
    q.put(addr)
    while True:
        r.read(timeout=1.0)
        time.sleep(1.0)  # a reader that has stopped consuming


def serve_rpc(q: Queue[Any]) -> None:
    async def main() -> None:
        zctx = zmq.asyncio.Context()
        server = RpcServer(zctx, registry(), ep("host-b", "server"), BIND)
        q.put(server.address)

        async def handle(env: Envelope, msg: msgspec.Struct) -> Result[msgspec.Struct]:
            assert isinstance(msg, Ping)
            return Ok(Pong(msg.n + 1))

        await server.serve(handle)

    asyncio.run(main())


def hold_the_gil(seconds: float) -> None:
    """Block this process's every thread, its heartbeat's too, for ``seconds``: a libc call through
    ``ctypes.PyDLL`` keeps the GIL, as a long C call of a unit does (a ``list.extend`` over 10^8
    items, doc/framework-bugs/2026-11-10-a-long-gil-hold-looks-like-a-dead-peer.md)."""
    import ctypes
    import ctypes.util

    libc = ctypes.PyDLL(ctypes.util.find_library("c"))
    libc.usleep(ctypes.c_uint(int(seconds * 1e6)))


def read_holding_the_gil(q: Queue[Any], host: str, peer: str, shm_name: str, hold_s: float, n: int) -> None:
    """A reader that, after the first record, holds the GIL for ``hold_s``; then reads until ``n``
    records came, or the writer was lost; reports how many it read."""
    zctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
    r, addr = open_reader(zctx, local=HostId(host), peer=HostId(peer), bind_host=BIND, shm_name=shm_name, tun=TUN)
    q.put(addr)
    got = 0
    try:
        while got < 1:
            got += r.read(timeout=20) is not None
        q.put("holding")
        hold_the_gil(hold_s)
        q.put("released")
        while got < n:
            rec = r.read(timeout=20)
            assert rec is not None
            got += 1
        q.put(got)
    except PeerLost as e:
        q.put(f"lost: {e}")
    time.sleep(60)


def write_holding_the_gil(address: StreamAddress, host: str, events: Queue[str], hold_s: float) -> None:
    """A writer that writes one record, holds the GIL for ``hold_s``, writes another and closes."""
    zctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
    w = open_writer(zctx, local=HostId(host), address=address, tun=TUN)
    w.write(b"before", timeout=5)
    events.put("holding")
    hold_the_gil(hold_s)
    try:
        w.write(b"after", timeout=5)
        events.put("wrote after")
    except PeerLost as e:
        events.put(f"lost: {e}")
    w.close()
    zctx.term()
    time.sleep(60)
