"""Publish/subscribe (doc/specs/ipc.md §2): event streams, heartbeats.

At most once: a slow subscriber loses messages, a publisher never blocks.
"""

from __future__ import annotations

import msgspec
import zmq
import zmq.asyncio

from core.common.ids import Endpoint, Epoch
from core.common.messages import SchemaRegistry, decode_envelope, encode_envelope
from core.common.result import Failed


class Publisher:
    def __init__(self, zctx: zmq.Context[zmq.Socket[bytes]], registry: SchemaRegistry, me: Endpoint,
                 bind_host: str) -> None:
        self._registry = registry
        self._me = me
        self._sock = zctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(f"tcp://{bind_host}:*")
        self.address = self._sock.getsockopt_string(zmq.LAST_ENDPOINT)

    def publish(self, topic: str, msg: msgspec.Struct, epoch: Epoch | None = None) -> bool:
        """Returns False when the message was dropped."""
        env = self._registry.wrap(msg, self._me, epoch)
        try:
            self._sock.send_multipart([topic.encode(), encode_envelope(env)], flags=zmq.NOBLOCK)
        except zmq.Again:
            return False
        return True

    def close(self) -> None:
        self._sock.close(0)


def _decode(registry: SchemaRegistry, parts: list[bytes]) -> tuple[str, msgspec.Struct] | None:
    if len(parts) != 2:
        return None
    env = decode_envelope(parts[1])
    if isinstance(env, Failed):
        return None
    msg = registry.unwrap(env.value)
    if isinstance(msg, Failed):
        return None
    return parts[0].decode(), msg.value


class Subscriber:
    def __init__(self, zctx: zmq.Context[zmq.Socket[bytes]], registry: SchemaRegistry) -> None:
        self._registry = registry
        self._sock = zctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.LINGER, 0)

    def connect(self, address: str, topics: list[str]) -> None:
        self._sock.connect(address)
        for t in topics:
            self._sock.setsockopt(zmq.SUBSCRIBE, t.encode())

    def recv(self, timeout_s: float = 0.0) -> tuple[str, msgspec.Struct] | None:
        """Next message whose schema this process knows, or None on timeout."""
        while self._sock.poll(int(timeout_s * 1000), zmq.POLLIN):
            got = _decode(self._registry, self._sock.recv_multipart())
            if got is not None:
                return got
        return None

    def close(self) -> None:
        self._sock.close(0)


class AsyncSubscriber:
    def __init__(self, zctx: zmq.asyncio.Context, registry: SchemaRegistry) -> None:
        self._registry = registry
        self._sock = zctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.LINGER, 0)

    def connect(self, address: str, topics: list[str]) -> None:
        self._sock.connect(address)
        for t in topics:
            self._sock.setsockopt(zmq.SUBSCRIBE, t.encode())

    def disconnect(self, address: str) -> None:
        self._sock.disconnect(address)

    async def recv(self) -> tuple[str, msgspec.Struct]:
        while True:
            got = _decode(self._registry, await self._sock.recv_multipart())
            if got is not None:
                return got

    def close(self) -> None:
        self._sock.close(0)
