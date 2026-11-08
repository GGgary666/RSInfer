"""The platform interface (doc/specs/platform.md): devices, memory, streams,
graph capture/replay, platform info. Hardware differences stop here.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any, Protocol, runtime_checkable

import msgspec


class DeviceInfo(msgspec.Struct, frozen=True):
    index: int
    name: str
    total_bytes: int
    capabilities: list[str] = []


class PlatformInfo(msgspec.Struct, frozen=True):
    """Goes into the system identity; hosts whose info differs may not join."""

    name: str
    versions: dict[str, str] = {}
    """driver, cuda, torch, nic, nvshmem, ..."""


class StaleGraph(RuntimeError):
    """A graph was replayed after one of its recorded dependencies changed."""


class GraphCaptureDenied(PermissionError):
    """Only plan units may record graphs."""


@runtime_checkable
class Buffer(Protocol):
    @property
    def device(self) -> int: ...
    @property
    def addr(self) -> int: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> str: ...


class VirtualRange(Protocol):
    """Reserved virtual addresses; physical memory is mapped on demand (grow)."""

    @property
    def device(self) -> int: ...
    @property
    def base(self) -> int: ...
    @property
    def reserved(self) -> int: ...
    @property
    def mapped(self) -> int: ...
    def view(self, offset: int, shape: tuple[int, ...], dtype: str) -> Buffer: ...


class Stream(Protocol):
    @property
    def device(self) -> int: ...


class Graph(Protocol):
    def replay(self) -> None:
        """Raises ``StaleGraph`` when a recorded dependency changed."""
        ...

    def check(self) -> list[str]:
        """Changed dependencies, without replaying."""
        ...


class GraphRecorder(Protocol):
    def capture(self) -> AbstractContextManager[Graph]: ...


class Platform(Protocol):
    def devices(self) -> list[DeviceInfo]: ...
    def alloc(self, nbytes: int, device: int, *, shape: tuple[int, ...] | None = None,
              dtype: str = "u8") -> Buffer: ...
    def free(self, buf: Buffer) -> None: ...
    def tensor(self, buf: Buffer) -> Any:
        """The tensor behind ``buf`` (a ``torch.Tensor`` on the torch platforms; opaque to the
        framework). The fake platform has no tensors and raises ``TypeError``."""
        ...
    def reserve(self, nbytes: int, device: int) -> VirtualRange: ...
    def map(self, vr: VirtualRange, offset: int, nbytes: int) -> None: ...
    def stream(self, device: int) -> Stream: ...
    def synchronize(self, device: int) -> None:
        """Return once all work queued on ``device`` has finished (a worker's ``quiesce()``)."""
        ...
    def graph(self) -> GraphRecorder: ...
    def info(self) -> PlatformInfo: ...


class GuardedPlatform:
    """What a unit sees as ``ctx.platform``: ``graph()`` only for plan units."""

    __slots__ = ("_inner", "_may_capture", "_who")

    def __init__(self, inner: Platform, *, may_capture: bool, who: str) -> None:
        self._inner = inner
        self._may_capture = may_capture
        self._who = who

    def graph(self) -> GraphRecorder:
        if not self._may_capture:
            raise GraphCaptureDenied(
                f"{self._who} may not record graphs; only plan units "
                "(capabilities.captures_graph) can"
            )
        return self._inner.graph()

    def __getattr__(self, name: str) -> Callable[..., object]:
        return getattr(self._inner, name)
