"""Code that runs in child processes for the IPC tests."""

from __future__ import annotations

import asyncio
import struct
import time
from multiprocessing.queues import Queue
from typing import Any

import msgspec
import zmq
import zmq.asyncio

from core.common.ids import Endpoint, HostId, ProcId
from core.common.messages import Envelope, SchemaRegistry
from core.common.result import Ok, Result
from core.ipc import IpcTunables, RpcServer, StreamAddress, open_reader, open_writer

BIND = "127.0.0.1"  # test data: the address of every fake host
TUN = IpcTunables(heartbeat_interval_s=0.05, peer_timeout_s=0.5, stream_hwm=4,
                  stream_socket_buffer=1 << 16, shm_capacity=1 << 16)


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
    w = open_writer(zctx, local=HostId(host), address=address, tun=TUN)
    for i, size in enumerate(sizes):
        w.write(stamp(i, size), timeout=30)
    if close:
        w.close()
        zctx.term()  # blocks until queued records are handed to the kernel
    time.sleep(then_sleep)


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
