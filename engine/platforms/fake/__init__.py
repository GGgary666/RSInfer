"""The fake platform for CI (doc/specs/platform.md §2.2).

CPU only and free of model semantics. Its point is the fake CUDA graph: while
recording it notes every buffer it touched (by address: address, shape, dtype)
and every value it baked in (by value: kernel solution, scalars, config keys
and runtime flags read during capture). Replaying after any of those changed
raises ``StaleGraph``; the same buffer with new contents replays normally.

Buffers that a graph must follow across reallocation are passed as
``FakeHandle``: a named reference the owning state unit re-points. A plain
buffer that was freed is also caught, as "the graph reads freed memory".
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from core.context.capture import ValueDep, observe_values
from platforms.base import Buffer, DeviceInfo, Platform, PlatformInfo, StaleGraph, VirtualRange

_ADDRESS_STEP = 1 << 40
"""Each device owns a disjoint fake address space."""
_ALIGN = 256

_DTYPE_BYTES = {"u8": 1, "i8": 1, "bf16": 2, "f16": 2, "f32": 4, "i32": 4, "i64": 8, "f64": 8}


class FakeOOM(MemoryError):
    pass


class FakeKernelError(RuntimeError):
    """Raised by a fake kernel configured to fail."""


def _size(shape: tuple[int, ...], dtype: str) -> int:
    n = _DTYPE_BYTES[dtype]
    for d in shape:
        n *= d
    return n


@dataclass(eq=False)
class FakeBuffer:
    device: int
    addr: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str
    freed: bool = False
    version: int = 0
    """Bumped by ``write()``: content changes, address stays."""

    def write(self) -> None:
        if self.freed:
            raise RuntimeError(f"write to freed buffer at {self.addr:#x}")
        self.version += 1


@dataclass(eq=False)
class FakeHandle:
    """A named reference to the current buffer of some state, e.g. ``"kv_pool"``."""

    name: str
    buffer: FakeBuffer


@dataclass(eq=False)
class FakeVirtualRange:
    device: int
    base: int
    reserved: int
    mapped: int = 0

    def view(self, offset: int, shape: tuple[int, ...], dtype: str) -> FakeBuffer:
        nbytes = _size(shape, dtype)
        if offset + nbytes > self.mapped:
            raise ValueError(f"view [{offset}, {offset + nbytes}) beyond mapped {self.mapped}")
        return FakeBuffer(self.device, self.base + offset, nbytes, shape, dtype)


@dataclass(eq=False)
class FakeStream:
    device: int
    index: int


@dataclass
class FakeKernel:
    sleep_s: float = 0.0
    fail: bool = False
    calls: int = 0


@dataclass(frozen=True)
class _BufferDep:
    label: str
    ref: FakeBuffer | FakeHandle
    addr: int
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class _Step:
    kernel: str
    buffers: tuple[_BufferDep, ...]
    solution: str | None
    scalars: tuple[tuple[str, Any], ...]


@dataclass(eq=False)
class FakeGraph:
    platform: FakePlatform
    steps: list[_Step] = field(default_factory=list[_Step])
    values: list[ValueDep] = field(default_factory=list[ValueDep])
    replays: int = 0

    def check(self) -> list[str]:
        problems: list[str] = []
        for step in self.steps:
            for b in step.buffers:
                buf = b.ref.buffer if isinstance(b.ref, FakeHandle) else b.ref
                if buf.freed:
                    problems.append(f"{step.kernel}: {b.label} at {b.addr:#x} was freed")
                elif buf.addr != b.addr:
                    problems.append(f"{step.kernel}: {b.label} moved {b.addr:#x} -> {buf.addr:#x}")
                elif buf.shape != b.shape or buf.dtype != b.dtype:
                    problems.append(f"{step.kernel}: {b.label} is now {buf.shape}/{buf.dtype}, "
                                    f"recorded {b.shape}/{b.dtype}")
        for dep in self.values:
            if dep.current is not None:
                now = dep.current()
                if now != dep.value:
                    problems.append(f"{dep.source} {dep.name!r} changed {dep.value!r} -> {now!r}")
        return problems

    def replay(self) -> None:
        problems = self.check()
        if problems:
            raise StaleGraph("; ".join(problems))
        for step in self.steps:
            self.platform._run(step.kernel)  # pyright: ignore[reportPrivateUsage]
        self.replays += 1


class FakeGraphRecorder:
    def __init__(self, platform: FakePlatform) -> None:
        self._platform = platform

    @contextmanager
    def capture(self) -> Generator[FakeGraph]:
        p = self._platform
        if p._capturing is not None:  # pyright: ignore[reportPrivateUsage]
            raise RuntimeError("nested graph capture")
        graph = FakeGraph(p)
        p._capturing = graph  # pyright: ignore[reportPrivateUsage]
        try:
            with observe_values(graph.values.append):
                yield graph
        finally:
            p._capturing = None  # pyright: ignore[reportPrivateUsage]


class FakePlatform:
    def __init__(
        self,
        *,
        devices: int = 1,
        total_bytes: int = 1 << 34,
        info: PlatformInfo | None = None,
    ) -> None:
        self._devices = [DeviceInfo(i, "fake", total_bytes, ["vmm", "graph"]) for i in range(devices)]
        self._next = [(i + 1) * _ADDRESS_STEP for i in range(devices)]
        self._used = [0] * devices
        self._streams = itertools.count()
        self._info = info or PlatformInfo("fake", {"driver": "0", "cuda": "0", "torch": "0"})
        self._capturing: FakeGraph | None = None
        self.kernels: dict[str, FakeKernel] = {}
        """Per-kernel behaviour: ``sleep_s`` and ``fail`` (fault injection)."""

    # -- Platform -------------------------------------------------------------

    def devices(self) -> list[DeviceInfo]:
        return list(self._devices)

    def _take(self, nbytes: int, device: int, physical: int) -> int:
        if not 0 <= device < len(self._devices):
            raise ValueError(f"no device {device}")
        if self._used[device] + physical > self._devices[device].total_bytes:
            raise FakeOOM(f"device {device}: {physical} more bytes do not fit")
        addr = self._next[device]
        self._next[device] += -(-nbytes // _ALIGN) * _ALIGN
        self._used[device] += physical
        return addr

    def alloc(self, nbytes: int, device: int, *, shape: tuple[int, ...] | None = None,
              dtype: str = "u8") -> FakeBuffer:
        if shape is None:
            shape = (nbytes // _DTYPE_BYTES[dtype],)
        nbytes = max(nbytes, _size(shape, dtype))
        # Addresses are never reused, so a reallocation always moves.
        return FakeBuffer(device, self._take(nbytes, device, nbytes), nbytes, shape, dtype)

    def free(self, buf: Buffer) -> None:
        if not isinstance(buf, FakeBuffer):
            raise TypeError(f"not a fake buffer: {buf!r}")
        if buf.freed:
            raise RuntimeError(f"double free at {buf.addr:#x}")
        buf.freed = True
        self._used[buf.device] -= buf.nbytes

    def reserve(self, nbytes: int, device: int) -> FakeVirtualRange:
        return FakeVirtualRange(device, self._take(nbytes, device, 0), nbytes)

    def map(self, vr: VirtualRange, offset: int, nbytes: int) -> None:
        if not isinstance(vr, FakeVirtualRange):
            raise TypeError(f"not a fake virtual range: {vr!r}")
        if offset != vr.mapped:
            raise ValueError(f"map must extend the mapped prefix at {vr.mapped}, got {offset}")
        if offset + nbytes > vr.reserved:
            raise ValueError(f"map beyond reserved size {vr.reserved}")
        if self._used[vr.device] + nbytes > self._devices[vr.device].total_bytes:
            raise FakeOOM(f"device {vr.device}: cannot map {nbytes} more bytes")
        self._used[vr.device] += nbytes
        vr.mapped += nbytes

    def stream(self, device: int) -> FakeStream:
        return FakeStream(device, next(self._streams))

    def graph(self) -> FakeGraphRecorder:
        return FakeGraphRecorder(self)

    def info(self) -> PlatformInfo:
        return self._info

    def used(self, device: int = 0) -> int:
        return self._used[device]

    # -- fake kernels ---------------------------------------------------------

    def handle(self, name: str, buffer: FakeBuffer) -> FakeHandle:
        return FakeHandle(name, buffer)

    def launch(
        self,
        kernel: str,
        buffers: list[FakeBuffer | FakeHandle] | None = None,
        *,
        solution: str | None = None,
        scalars: dict[str, Any] | None = None,
    ) -> None:
        """Run a fake kernel now, or record it when a capture is open."""
        graph = self._capturing
        if graph is None:
            for ref in buffers or []:
                buf = ref.buffer if isinstance(ref, FakeHandle) else ref
                if buf.freed:
                    raise RuntimeError(f"{kernel}: reads freed buffer at {buf.addr:#x}")
            self._run(kernel)
            return
        deps: list[_BufferDep] = []
        for i, ref in enumerate(buffers or []):
            buf = ref.buffer if isinstance(ref, FakeHandle) else ref
            label = ref.name if isinstance(ref, FakeHandle) else f"arg{i}"
            deps.append(_BufferDep(label, ref, buf.addr, buf.shape, buf.dtype))
        if solution is not None:
            graph.values.append(ValueDep("solution", kernel, solution))
        for k, v in sorted((scalars or {}).items()):
            graph.values.append(ValueDep("scalar", f"{kernel}.{k}", v))
        graph.steps.append(_Step(kernel, tuple(deps), solution, tuple(sorted((scalars or {}).items()))))

    def _run(self, kernel: str) -> None:
        k = self.kernels.setdefault(kernel, FakeKernel())
        k.calls += 1
        if k.sleep_s:
            time.sleep(k.sleep_s)
        if k.fail:
            raise FakeKernelError(f"fake kernel {kernel} failed (injected)")


if TYPE_CHECKING:
    _: type[Platform] = FakePlatform  # FakePlatform satisfies the Platform protocol
