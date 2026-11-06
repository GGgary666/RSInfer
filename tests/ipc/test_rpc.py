"""ipc.md §5 test 4 for RPC, plus envelope rejection and the sync variants."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import signal
import threading
import time

import msgspec
import zmq
import zmq.asyncio

from core.common.messages import Envelope, SchemaRegistry
from core.common.result import Failed, Ok, Result, fail
from core.ipc import Publisher, RpcClient, RpcServer, Subscriber, SyncRpcClient, SyncRpcServer
from tests.ipc.peers import BIND, Ping, Pong, Unknown, ep, registry, serve_rpc

SPAWN = mp.get_context("spawn")


def test_request_reply_and_handler_failures() -> None:
    async def main() -> None:
        zctx = zmq.asyncio.Context()
        server = RpcServer(zctx, registry(), ep("host-a", "server"), BIND)

        async def handle(env: Envelope, msg: msgspec.Struct) -> Result[msgspec.Struct]:
            assert isinstance(msg, Ping) and env.sender.proc.role == "client"
            if msg.n < 0:
                return fail("test", "handle", "negative", code="test.negative")
            if msg.n == 0:
                raise ValueError("zero")
            return Ok(Pong(msg.n + 1))

        task = asyncio.create_task(server.serve(handle))
        client = RpcClient(zctx, registry(), ep("host-b", "client"))
        r = await client.call(server.address, Ping(41), timeout=5)
        assert r == Ok(Pong(42))
        r = await client.call(server.address, Ping(-1), timeout=5)
        assert isinstance(r, Failed) and r.has_code("test.negative")
        r = await client.call(server.address, Ping(0), timeout=5)
        assert isinstance(r, Failed) and r.has_code("ipc.handler_failed")
        concurrent = await asyncio.gather(*(client.call(server.address, Ping(i), timeout=5) for i in range(1, 50)))
        assert concurrent == [Ok(Pong(i + 1)) for i in range(1, 50)]
        task.cancel()
        client.close()
        server.close()

    asyncio.run(main())


def test_a_timed_out_call_does_not_cut_off_other_calls_to_the_same_peer() -> None:
    async def main() -> None:
        zctx = zmq.asyncio.Context()
        server = RpcServer(zctx, registry(), ep("host-a", "server"), BIND)

        async def handle(env: Envelope, msg: msgspec.Struct) -> Result[msgspec.Struct]:
            assert isinstance(msg, Ping)
            await asyncio.sleep(msg.n / 10)
            return Ok(Pong(msg.n))

        task = asyncio.create_task(server.serve(handle))
        client = RpcClient(zctx, registry(), ep("host-b", "client"))
        long = asyncio.create_task(client.call(server.address, Ping(5), timeout=5))
        short = await client.call(server.address, Ping(20), timeout=0.1)
        assert isinstance(short, Failed) and short.has_code("ipc.timeout")
        assert await long == Ok(Pong(5))
        task.cancel()
        client.close()
        server.close()

    asyncio.run(main())


def test_unknown_schema_is_rejected_not_guessed() -> None:
    async def main() -> None:
        zctx = zmq.asyncio.Context()
        server = RpcServer(zctx, registry(), ep("host-a", "server"), BIND)

        async def handle(env: Envelope, msg: msgspec.Struct) -> Result[msgspec.Struct]:
            raise AssertionError("must not be called")

        task = asyncio.create_task(server.serve(handle))
        newer = SchemaRegistry()
        newer.register("test.ping@2", Unknown)
        client = RpcClient(zctx, newer, ep("host-b", "client"))
        r = await client.call(server.address, Unknown(1), timeout=5)
        assert isinstance(r, Failed) and r.has_code("ipc.unknown_schema")
        assert "test.ping@1" in r.diagnostics[0].hint
        task.cancel()
        client.close()
        server.close()

    asyncio.run(main())


def test_call_to_a_killed_peer_fails_within_the_timeout() -> None:
    q = SPAWN.Queue()
    child = SPAWN.Process(target=serve_rpc, args=(q,))
    child.start()
    address = q.get(timeout=20)

    async def main() -> None:
        client = RpcClient(zmq.asyncio.Context(), registry(), ep("host-a", "client"))
        assert await client.call(address, Ping(1), timeout=5) == Ok(Pong(2))
        assert child.pid is not None
        os.kill(child.pid, signal.SIGKILL)
        t0 = time.monotonic()
        r = await client.call(address, Ping(2), timeout=0.5)
        assert isinstance(r, Failed) and r.has_code("ipc.timeout")
        assert time.monotonic() - t0 < 1.5
        client.close()

    try:
        asyncio.run(main())
    finally:
        child.join(10)


def test_sync_server_serves_at_its_own_safe_points() -> None:
    zctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
    server = SyncRpcServer(zctx, registry(), ep("host-a", "worker"), BIND)
    results: list[Result[msgspec.Struct]] = []

    def call() -> None:
        client = SyncRpcClient(zmq.Context(), registry(), ep("host-a", "ctrl"))
        results.append(client.call(server.address, Ping(1), timeout=5))
        client.close()

    t = threading.Thread(target=call)
    t.start()
    time.sleep(0.2)  # the worker is "inside a step": the request waits
    assert results == []
    served = server.poll(lambda env, msg: Ok(Pong(msg.n + 1)), timeout_s=2)  # type: ignore[attr-defined]
    t.join(5)
    assert served == 1 and results == [Ok(Pong(2))]
    client = SyncRpcClient(zctx, registry(), ep("host-a", "ctrl"))
    r = client.call(server.address, Ping(1), timeout=0.3)  # nobody polls now
    assert isinstance(r, Failed) and r.has_code("ipc.timeout")
    server.close()


def test_pubsub_is_at_most_once_and_filters_topics() -> None:
    zctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
    pub = Publisher(zctx, registry(), ep("host-a", "agent", "events"), BIND)
    sub = Subscriber(zctx, registry())
    sub.connect(pub.address, ["heartbeat."])
    deadline = time.monotonic() + 5
    got = None
    while got is None and time.monotonic() < deadline:  # slow joiner: early messages are lost
        pub.publish("heartbeat.host-a", Ping(7))
        pub.publish("other", Ping(8))
        got = sub.recv(timeout_s=0.05)
    assert got == ("heartbeat.host-a", Ping(7))
    pub.close()
    sub.close()
