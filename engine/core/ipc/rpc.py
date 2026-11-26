"""Request/reply over ZMQ (doc/specs/ipc.md §2): change transactions, control
commands, process check-in. Calls carry a timeout; on timeout the result is
``Failed`` with ``ipc.timeout``.

Every control message must be idempotent (ipc.md §4): a request that timed out
may still sit in the send queue and arrive once the peer recovers, and callers
retry. Receivers deduplicate, e.g. by change id or by incarnation.

``RpcServer``/``RpcClient`` are for asyncio processes (control plane, node
agents); ``SyncRpcServer``/``SyncRpcClient`` for single-threaded loops (workers),
which serve requests only at their safe points.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections.abc import Awaitable, Callable
from typing import Literal

import msgspec
import zmq
import zmq.asyncio

from core.common.ids import Endpoint, Epoch
from core.common.messages import Envelope, SchemaRegistry
from core.common.result import Diagnostic, Failed, Ok, Result, diag, fail


class RpcFrame(msgspec.Struct, frozen=True):
    kind: Literal["req", "rep", "err"]
    id: int
    env: Envelope | None = None
    diagnostics: list[Diagnostic] = []


_enc = msgspec.msgpack.Encoder()
_dec = msgspec.msgpack.Decoder(RpcFrame)

type AsyncHandler = Callable[[Envelope, msgspec.Struct], Awaitable[Result[msgspec.Struct]]]
type SyncHandler = Callable[[Envelope, msgspec.Struct], Result[msgspec.Struct] | Deferred]


def _bind(sock: zmq.Socket[bytes] | zmq.asyncio.Socket, host: str) -> str:
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"tcp://{host}:*")
    return sock.getsockopt_string(zmq.LAST_ENDPOINT)


def _reply(registry: SchemaRegistry, me: Endpoint, req: RpcFrame, result: Result[msgspec.Struct]) -> RpcFrame:
    if isinstance(result, Failed):
        return RpcFrame("err", req.id, diagnostics=result.diagnostics)
    try:
        env = registry.wrap(result.value, me, req.env.epoch if req.env else None)
    except KeyError as e:
        return RpcFrame("err", req.id, diagnostics=[diag("ipc", "rpc", str(e), code="ipc.unknown_schema")])
    return RpcFrame("rep", req.id, env=env)


def _decode_request(registry: SchemaRegistry, data: bytes) -> tuple[RpcFrame | None, Result[msgspec.Struct]]:
    try:
        frame = _dec.decode(data)
    except msgspec.DecodeError as e:
        return None, fail("ipc", "rpc", f"bad frame: {e}", code="ipc.malformed")
    if frame.kind != "req" or frame.env is None:
        return frame, fail("ipc", "rpc", "not a request", code="ipc.malformed")
    return frame, registry.unwrap(frame.env)


def _handler_error(e: Exception) -> Failed:
    return fail("ipc", "rpc", f"handler raised {type(e).__name__}: {e}", code="ipc.handler_failed")


def _decode_reply(registry: SchemaRegistry, frame: RpcFrame) -> Result[msgspec.Struct]:
    if frame.kind == "err":
        return Failed(frame.diagnostics)
    if frame.env is None:
        return fail("ipc", "rpc", "reply without body", code="ipc.malformed")
    return registry.unwrap(frame.env)


def _timeout(address: str, msg: msgspec.Struct, timeout: float) -> Failed:
    return fail("ipc", "rpc", f"no reply to {type(msg).__name__} within {timeout}s", subject=address,
                code="ipc.timeout", hint="the peer may be lost")


class RpcServer:
    def __init__(self, zctx: zmq.asyncio.Context, registry: SchemaRegistry, me: Endpoint, bind_host: str) -> None:
        self._registry = registry
        self._me = me
        self._sock = zctx.socket(zmq.ROUTER)
        self.address = _bind(self._sock, bind_host)
        self._tasks: set[asyncio.Task[None]] = set()

    async def serve(self, handler: AsyncHandler) -> None:
        """Serve forever; each request runs in its own task."""
        while True:
            parts = await self._sock.recv_multipart()
            if len(parts) != 2:
                continue
            ident, data = parts
            task = asyncio.create_task(self._handle(ident, data, handler))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _handle(self, ident: bytes, data: bytes, handler: AsyncHandler) -> None:
        frame, msg = _decode_request(self._registry, data)
        if frame is None:
            return
        if isinstance(msg, Ok):
            try:
                result = await handler(frame.env, msg.value)  # type: ignore[arg-type]
            except Exception as e:  # noqa: BLE001 - reported to the caller
                result = _handler_error(e)
        else:
            result = msg
        await self._sock.send_multipart([ident, _enc.encode(_reply(self._registry, self._me, frame, result))])

    def close(self) -> None:
        for t in self._tasks:
            t.cancel()
        self._sock.close(0)


class RpcClient:
    def __init__(self, zctx: zmq.asyncio.Context, registry: SchemaRegistry, me: Endpoint) -> None:
        self._zctx = zctx
        self._registry = registry
        self._me = me
        self._ids = itertools.count(1)
        self._socks: dict[str, tuple[zmq.asyncio.Socket, asyncio.Task[None]]] = {}
        self._pending: dict[int, asyncio.Future[RpcFrame]] = {}
        self._waiting: dict[str, set[int]] = {}
        """Request ids still waiting for a reply, per peer."""

    def _sock(self, address: str) -> zmq.asyncio.Socket:
        entry = self._socks.get(address)
        if entry is None:
            sock = self._zctx.socket(zmq.DEALER)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(address)
            entry = self._socks[address] = (sock, asyncio.create_task(self._read(sock)))
        return entry[0]

    async def _read(self, sock: zmq.asyncio.Socket) -> None:
        while True:
            data = await sock.recv()
            try:
                frame = _dec.decode(data)
            except msgspec.DecodeError:
                continue
            fut = self._pending.pop(frame.id, None)
            if fut is not None and not fut.done():
                fut.set_result(frame)

    def _drop(self, address: str) -> None:
        """Forget a peer that did not answer, so queued requests are not delivered later.

        Only when nothing else waits on it: a short call (a keepalive) timing out
        must not cut off a long one (a deferred ``Prepared``) to the same peer.
        """
        if self._waiting.get(address):
            return
        entry = self._socks.pop(address, None)
        if entry is not None:
            entry[1].cancel()
            entry[0].close(0)

    async def call(self, address: str, msg: msgspec.Struct, *, timeout: float,
                   epoch: Epoch | None = None) -> Result[msgspec.Struct]:
        sock = self._sock(address)
        rid = next(self._ids)
        fut: asyncio.Future[RpcFrame] = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        waiting = self._waiting.setdefault(address, set())
        waiting.add(rid)
        frame: RpcFrame | None = None
        try:
            await sock.send(_enc.encode(RpcFrame("req", rid, self._registry.wrap(msg, self._me, epoch))))
            frame = await asyncio.wait_for(fut, timeout)
        except TimeoutError:
            pass
        finally:
            self._pending.pop(rid, None)
            waiting.discard(rid)
        if frame is None:
            self._drop(address)
            return _timeout(address, msg, timeout)
        return _decode_reply(self._registry, frame)

    def close(self) -> None:
        self._waiting.clear()
        for address in list(self._socks):
            self._drop(address)


class Deferred:
    """A reply the handler will send later, e.g. ``Paused`` once the epoch boundary is reached."""

    def __init__(self, server: SyncRpcServer, ident: bytes, frame: RpcFrame) -> None:
        self._server = server
        self._ident = ident
        self._frame = frame
        self.done = False

    def resolve(self, result: Result[msgspec.Struct]) -> None:
        if self.done:
            return
        self.done = True
        self._server._send(self._ident, self._frame, result)  # pyright: ignore[reportPrivateUsage]


class SyncRpcServer:
    def __init__(self, zctx: zmq.Context[zmq.Socket[bytes]], registry: SchemaRegistry, me: Endpoint,
                 bind_host: str) -> None:
        self._registry = registry
        self._me = me
        self._sock = zctx.socket(zmq.ROUTER)
        self.address = _bind(self._sock, bind_host)
        self._current: tuple[bytes, RpcFrame] | None = None

    def defer(self) -> Deferred:
        """Called inside a handler: the handler returns the ``Deferred`` and resolves it later."""
        if self._current is None:
            raise RuntimeError("defer() outside a request handler")
        return Deferred(self, *self._current)

    def _send(self, ident: bytes, frame: RpcFrame, result: Result[msgspec.Struct]) -> None:
        self._sock.send_multipart([ident, _enc.encode(_reply(self._registry, self._me, frame, result))])

    def poll(self, handler: SyncHandler, timeout_s: float = 0.0) -> int:
        """Serve every request that arrives within ``timeout_s``; returns how many."""
        served = 0
        wait_ms = int(timeout_s * 1000)
        while self._sock.poll(wait_ms, zmq.POLLIN):
            parts = self._sock.recv_multipart()
            wait_ms = 0
            if len(parts) != 2:
                continue
            ident, data = parts
            frame, msg = _decode_request(self._registry, data)
            if frame is None:
                continue
            if isinstance(msg, Ok):
                self._current = (ident, frame)
                try:
                    result = handler(frame.env, msg.value)  # type: ignore[arg-type]
                    if not isinstance(result, Ok | Failed | Deferred):  # pyright: ignore[reportUnnecessaryIsInstance]
                        raise TypeError(f"handler returned {result!r}, not a Result")
                except Exception as e:  # noqa: BLE001
                    result = _handler_error(e)
                finally:
                    self._current = None
            else:
                result = msg
            if not isinstance(result, Deferred):
                self._send(ident, frame, result)
            served += 1
        return served

    def close(self) -> None:
        self._sock.close(0)


class SyncRpcClient:
    def __init__(self, zctx: zmq.Context[zmq.Socket[bytes]], registry: SchemaRegistry, me: Endpoint) -> None:
        self._zctx = zctx
        self._registry = registry
        self._me = me
        self._ids = itertools.count(1)
        self._socks: dict[str, zmq.Socket[bytes]] = {}

    def call(self, address: str, msg: msgspec.Struct, *, timeout: float,
             epoch: Epoch | None = None) -> Result[msgspec.Struct]:
        sock = self._socks.get(address)
        if sock is None:
            sock = self._socks[address] = self._zctx.socket(zmq.DEALER)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(address)
        rid = next(self._ids)
        sock.send(_enc.encode(RpcFrame("req", rid, self._registry.wrap(msg, self._me, epoch))))
        deadline = time.monotonic() + timeout
        while (left := deadline - time.monotonic()) > 0:
            if not sock.poll(max(1, int(left * 1000)), zmq.POLLIN):
                continue
            try:
                frame = _dec.decode(sock.recv())
            except msgspec.DecodeError:
                continue
            if frame.id == rid:
                return _decode_reply(self._registry, frame)
        self._socks.pop(address).close(0)
        return _timeout(address, msg, timeout)

    def close(self) -> None:
        for sock in self._socks.values():
            sock.close(0)
        self._socks.clear()
