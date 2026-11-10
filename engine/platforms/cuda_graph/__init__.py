"""PyTorch CUDA graphs with one process-owned capture stream (TP1).

The kernels are ordinary PyTorch operations, not architecture-specific kernels.
Warm-up and capture must run while the worker is paused.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager

import torch

from platforms.base import DeviceInfo, PlatformInfo, StaleGraph
from platforms.torch_eager import TorchEagerPlatform


class CudaGraph:
    def __init__(self, closed: Callable[[], None]) -> None:
        self.raw = torch.cuda.CUDAGraph()
        self.ready = False
        self._closed: Callable[[], None] | None = closed

    def check(self) -> list[str]:
        return [] if self.ready else ["graph is not ready"]

    def replay(self) -> None:
        if not self.ready:
            raise StaleGraph("graph is not ready")
        self.raw.replay()

    def close(self) -> None:
        self.ready = False
        if self._closed is not None:
            closed, self._closed = self._closed, None
            self.raw.reset()
            closed()


class Recorder:
    """The graphs of one recorder share a memory pool, as SGLang's and vLLM's do: record the
    largest first and the others fit in what it took. They must not replay concurrently, and only
    a graph's outputs outlive the replay of another. PyTorch drops a pool with its last graph, so
    a recorder whose graphs are all closed starts a new one."""

    def __init__(self, stream: torch.cuda.Stream) -> None:
        self.stream = stream
        self.pool = torch.cuda.graph_pool_handle()
        self.open = 0
        """Graphs of this recorder not closed yet: those hold ``pool``."""

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

    def _closed(self) -> None:
        self.open -= 1
        if not self.open:
            self.pool = torch.cuda.graph_pool_handle()

    @contextmanager
    def capture(self) -> Generator[CudaGraph]:
        graph = CudaGraph(self._closed)
        self.open += 1
        with self.warmup():
            try:
                graph.raw.capture_begin(pool=self.pool)
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
