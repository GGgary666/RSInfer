"""PyTorch CUDA graphs with one process-owned capture stream (TP1).

The kernels are ordinary PyTorch operations, not architecture-specific kernels.
Warm-up and capture must run while the worker is paused.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import torch

from platforms.base import DeviceInfo, PlatformInfo, StaleGraph
from platforms.torch_eager import TorchEagerPlatform


class CudaGraph:
    def __init__(self) -> None:
        self.raw = torch.cuda.CUDAGraph()
        self.ready = False

    def check(self) -> list[str]:
        return [] if self.ready else ["graph is not ready"]

    def replay(self) -> None:
        if not self.ready:
            raise StaleGraph("graph is not ready")
        self.raw.replay()

    def close(self) -> None:
        self.ready = False
        self.raw.reset()


class Recorder:
    def __init__(self, stream: torch.cuda.Stream) -> None:
        self.stream = stream

    @contextmanager
    def warmup(self) -> Generator[None]:
        current = torch.cuda.current_stream(self.stream.device)
        self.stream.wait_stream(current)
        try:
            with torch.cuda.stream(self.stream):
                yield
        finally:
            self.stream.synchronize()
            current.wait_stream(self.stream)

    @contextmanager
    def capture(self) -> Generator[CudaGraph]:
        graph = CudaGraph()
        with self.warmup():
            try:
                graph.raw.capture_begin()
                try:
                    yield graph
                finally:
                    graph.raw.capture_end()
                graph.ready = True
            except BaseException:
                graph.close()
                raise


class CudaGraphPlatform(TorchEagerPlatform):
    def __init__(self) -> None:
        super().__init__(cuda=True)
        self.capture_stream = torch.cuda.Stream(device=0)

    def devices(self) -> list[DeviceInfo]:
        return [DeviceInfo(d.index, d.name, d.total_bytes, [*d.capabilities, "cuda_graph"])
                for d in super().devices()]

    def graph(self) -> Recorder:
        return Recorder(self.capture_stream)

    def info(self) -> PlatformInfo:
        return PlatformInfo("cuda_graph", super().info().versions)
