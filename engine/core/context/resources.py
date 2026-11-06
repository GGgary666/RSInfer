"""Resource leases handed to units as revocable registrations (context.md §5).

Streams are named by purpose (``"alt"``, ``"offload"``); one stream per
(purpose, device), shared by every unit that leases that purpose. Buffers are
private to the leasing instance. Leases return to the pool when the unit unloads.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.common.ids import InstanceId
from platforms.base import Platform, Stream


@dataclass
class _StreamLease:
    stream: Stream
    holders: set[InstanceId] = field(default_factory=set[InstanceId])


class Resources:
    def __init__(self, platform: Platform | None) -> None:
        self._platform = platform
        self._streams: dict[tuple[str, int], _StreamLease] = {}
        self._buffers: dict[tuple[InstanceId, str], Any] = {}

    def lease_stream(self, owner: InstanceId, name: str, device: int = 0) -> tuple[Stream, Callable[[], None]]:
        if self._platform is None:
            raise RuntimeError("this process has no platform; streams are unavailable")
        lease = self._streams.get((name, device))
        if lease is None:
            lease = self._streams[(name, device)] = _StreamLease(self._platform.stream(device))
        if owner in lease.holders:
            raise RuntimeError(f"{owner} already leases stream {name!r} on device {device}")
        lease.holders.add(owner)

        def release() -> None:
            current = self._streams.get((name, device))
            if current is None:
                return
            current.holders.discard(owner)
            if not current.holders:
                del self._streams[(name, device)]

        return lease.stream, release

    def lease_buffer[T](self, owner: InstanceId, name: str, factory: Callable[[], T]) -> tuple[T, Callable[[], None]]:
        key = (owner, name)
        if key in self._buffers:
            raise RuntimeError(f"{owner} already leases buffer {name!r}")
        buf = self._buffers[key] = factory()

        def release() -> None:
            self._buffers.pop(key, None)

        return buf, release

    def held_by(self, owner: InstanceId) -> list[str]:
        out = [f"stream:{n}@{d}" for (n, d), lease in self._streams.items() if owner in lease.holders]
        out += [f"buffer:{n}" for (o, n) in self._buffers if o == owner]
        return sorted(out)
