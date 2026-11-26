"""CANN tensors and process-owned ACL graph capture on Ascend."""

from __future__ import annotations

import sys
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any

import torch

from platforms.base import DeviceInfo, Platform, PlatformInfo, StaleGraph
from platforms.torch_eager import TorchEagerPlatform


@dataclass(frozen=True)
class AscendStream:
    device: int
    raw: object


class AclGraph:
    def __init__(self, npu: Any) -> None:
        self.raw = npu.NPUGraph()
        self.ready = False
        self.replays = 0

    def check(self) -> list[str]:
        return [] if self.ready else ["graph is not ready"]

    def replay(self) -> None:
        if not self.ready:
            raise StaleGraph("ACL graph is not ready")
        self.raw.replay()
        self.replays += 1

    def close(self) -> None:
        if self.ready:
            self.ready = False
            self.raw.reset()
            print(f"ACL graph: {self.replays} replays including the initial validation replay",
                  file=sys.stderr, flush=True)


class AclRecorder:
    def __init__(self, npu: Any) -> None:
        self.npu = npu
        self.stream = npu.Stream(device=0)
        self.pool = npu.graph_pool_handle()

    @contextmanager
    def warmup(self) -> Generator[None]:
        current = self.npu.current_stream()
        self.stream.wait_stream(current)
        try:
            with self.npu.stream(self.stream):
                yield
        finally:
            self.stream.synchronize()
            current.wait_stream(self.stream)

    @contextmanager
    def capture(self) -> Generator[AclGraph]:
        graph = AclGraph(self.npu)
        with self.warmup(), self.npu.graph(graph.raw, pool=self.pool, stream=self.stream):
            yield graph
        graph.ready = True


class AscendPlatform(TorchEagerPlatform):
    def __init__(self) -> None:
        try:
            self.torch_npu = import_module("torch_npu")
        except ModuleNotFoundError as e:
            raise RuntimeError("ascend requires a matching torch_npu/CANN runtime") from e
        self.npu = self.torch_npu.npu
        super().__init__()

    def _probe(self) -> list[DeviceInfo]:
        if not self.npu.is_available():
            raise RuntimeError("ascend requires an available NPU and a matching torch_npu/CANN runtime")
        self.npu.set_device(0)
        devices: list[DeviceInfo] = []
        for i in range(self.npu.device_count()):
            props = self.npu.get_device_properties(i)
            devices.append(DeviceInfo(i, props.name, props.total_memory, ["npu", "ascend", "acl_graph"]))
        return devices

    def _device(self, device: int) -> torch.device:
        if not 0 <= device < len(self._devices):
            raise ValueError(f"no device {device}")
        return torch.device("npu", device)

    def stream(self, device: int) -> AscendStream:
        return AscendStream(device, self.npu.current_stream(self._device(device)))

    def synchronize(self, device: int) -> None:
        self.npu.synchronize(self._device(device))

    def graph(self) -> AclRecorder:
        return AclRecorder(self.npu)

    def info(self) -> PlatformInfo:
        return PlatformInfo("ascend", {"torch": torch.__version__, "torch_npu": self.torch_npu.__version__, "device": "npu"})


if TYPE_CHECKING:
    _: type[Platform] = AscendPlatform
