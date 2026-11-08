"""Eager torch, on the CPU or on this process's CUDA devices (doc/specs/platform.md §2.3).

A buffer is a torch tensor, handed out by ``tensor(buf)``, zero-filled as sglang fills its
pools. The address of a buffer is its tensor's ``data_ptr()``. There is no graph capture:
plan units run eagerly here, and ``cuda_sm100`` adds graphs. ``reserve`` keeps the VMM
contract (a base address that never moves, views only inside the mapped prefix) by
committing the whole range at once, so it saves no memory, and ``vmm`` is not among the
device capabilities.

On CUDA the devices are the ones the process sees: the node agent sets
``CUDA_VISIBLE_DEVICES`` to the process's GPUs. Every stream is the device's current stream;
units do not create streams (stage2-contract.md §10.4).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from platforms.base import Buffer, DeviceInfo, GraphRecorder, Platform, PlatformInfo, VirtualRange

NAME = "torch_eager"

DTYPES: dict[str, torch.dtype] = {
    "u8": torch.uint8, "i8": torch.int8, "bf16": torch.bfloat16, "f16": torch.float16,
    "f32": torch.float32, "i32": torch.int32, "i64": torch.int64, "f64": torch.float64,
}
"""The platform's dtype names and the torch dtypes they stand for."""


def _size(shape: tuple[int, ...], dtype: str) -> int:
    n = DTYPES[dtype].itemsize
    for d in shape:
        n *= d
    return n


@dataclass(eq=False)
class TorchBuffer:
    device: int
    addr: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str
    data: torch.Tensor | None
    """``None`` once freed."""


@dataclass(eq=False)
class TorchVirtualRange:
    device: int
    base: int
    reserved: int
    storage: torch.Tensor
    """``u8``, the whole range, allocated at ``reserve``."""
    mapped: int = 0

    def view(self, offset: int, shape: tuple[int, ...], dtype: str) -> TorchBuffer:
        nbytes = _size(shape, dtype)
        if offset < 0 or offset + nbytes > self.mapped:
            raise ValueError(f"view [{offset}, {offset + nbytes}) beyond mapped {self.mapped}")
        if offset % DTYPES[dtype].itemsize:
            raise ValueError(f"view offset {offset} is not aligned to {dtype}")
        data = self.storage[offset:offset + nbytes].view(DTYPES[dtype]).view(shape)
        return TorchBuffer(self.device, self.base + offset, nbytes, shape, dtype, data)


@dataclass(eq=False)
class TorchStream:
    device: int
    raw: torch.cuda.Stream | None
    """The device's current CUDA stream; ``None`` on the CPU."""


class TorchEagerPlatform:
    def __init__(self, *, cuda: bool = False) -> None:
        if cuda and not torch.cuda.is_available():
            raise RuntimeError("torch_eager was asked for CUDA, but torch sees no CUDA device")
        self._cuda = cuda
        self._devices = self._probe()

    def _probe(self) -> list[DeviceInfo]:
        if self._cuda:
            out: list[DeviceInfo] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                out.append(DeviceInfo(i, props.name, props.total_memory,
                                      ["cuda", f"sm_{props.major}{props.minor}"]))
            return out
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (AttributeError, ValueError, OSError):
            total = 0
        return [DeviceInfo(0, "cpu", total, ["cpu"])]

    def _device(self, device: int) -> torch.device:
        if not 0 <= device < len(self._devices):
            raise ValueError(f"no device {device}")
        return torch.device("cuda", device) if self._cuda else torch.device("cpu")

    # -- Platform -------------------------------------------------------------

    def devices(self) -> list[DeviceInfo]:
        return list(self._devices)

    def alloc(self, nbytes: int, device: int, *, shape: tuple[int, ...] | None = None,
              dtype: str = "u8") -> TorchBuffer:
        if shape is None:
            shape = (nbytes // DTYPES[dtype].itemsize,)
        data = torch.zeros(shape, dtype=DTYPES[dtype], device=self._device(device))
        return TorchBuffer(device, data.data_ptr(), _size(shape, dtype), shape, dtype, data)

    def free(self, buf: Buffer) -> None:
        if not isinstance(buf, TorchBuffer):
            raise TypeError(f"not a torch_eager buffer: {buf!r}")
        if buf.data is None:
            raise RuntimeError(f"double free at {buf.addr:#x}")
        buf.data = None

    def tensor(self, buf: Buffer) -> torch.Tensor:
        if not isinstance(buf, TorchBuffer):
            raise TypeError(f"not a torch_eager buffer: {buf!r}")
        if buf.data is None:
            raise RuntimeError(f"buffer at {buf.addr:#x} was freed")
        return buf.data

    def reserve(self, nbytes: int, device: int) -> TorchVirtualRange:
        storage = torch.zeros(nbytes, dtype=torch.uint8, device=self._device(device))
        return TorchVirtualRange(device, storage.data_ptr(), nbytes, storage)

    def map(self, vr: VirtualRange, offset: int, nbytes: int) -> None:
        if not isinstance(vr, TorchVirtualRange):
            raise TypeError(f"not a torch_eager virtual range: {vr!r}")
        if offset != vr.mapped:
            raise ValueError(f"map must extend the mapped prefix at {vr.mapped}, got {offset}")
        if offset + nbytes > vr.reserved:
            raise ValueError(f"map beyond reserved size {vr.reserved}")
        vr.mapped += nbytes

    def stream(self, device: int) -> TorchStream:
        d = self._device(device)
        return TorchStream(device, torch.cuda.current_stream(d) if self._cuda else None)

    def synchronize(self, device: int) -> None:
        d = self._device(device)
        if self._cuda:
            torch.cuda.synchronize(d)

    def graph(self) -> GraphRecorder:
        raise NotImplementedError("torch_eager runs eagerly: it records no graphs")

    def info(self) -> PlatformInfo:
        return PlatformInfo(NAME, {"torch": torch.__version__, "cuda": torch.version.cuda or "",
                                   "device": "cuda" if self._cuda else "cpu"})


if TYPE_CHECKING:
    _: type[Platform] = TorchEagerPlatform  # TorchEagerPlatform satisfies the Platform protocol
