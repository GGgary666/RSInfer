"""A captured decode step on the CPU: stand-ins for the platform's graph recorder, so that the path a
graph bucket takes (fixed buffers, padding lanes, the solutions of captured steps) runs without a
GPU. Nothing is recorded: a replay runs the forward again."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager

import torch

from core.plan.graph import Dependencies, GraphPlan
from platforms.base import StaleGraph
from tests.units.support.worker import Worker


class StandInGraph:
    """``Graph`` where nothing was recorded: a replay runs ``forward`` again."""

    def __init__(self, forward: Callable[[], None], replays: list[int], size: int) -> None:
        self.forward, self.replays, self.size = forward, replays, size
        self.closed = False

    def replay(self) -> None:
        if self.closed:
            raise StaleGraph("closed")
        self.replays.append(self.size)
        self.forward()

    def close(self) -> None:
        self.closed = True


class StandInRecorder:
    """``GraphRecorder`` for one stand-in graph. The forward inside ``capture`` runs here, so what
    it did to the padding items is taken back: a real capture records without running."""

    def __init__(self, graph: StandInGraph, undo: Callable[[], None]) -> None:
        self.graph, self.undo = graph, undo

    @contextmanager
    def warmup(self) -> Generator[None]:
        yield

    @contextmanager
    def capture(self) -> Generator[StandInGraph]:
        yield self.graph
        self.undo()


def with_graphs(w: Worker, sizes: list[int]) -> list[int]:
    """Give ``w``'s plans a graph per bucket, recorded by the stand-in; the buckets replayed since."""
    plans, model, batch = w.service("plan@1"), w.service("model@1"), w.service("worker.batch@1")
    stores = list(model.stores.values())
    batch.prepare(list(model.stores))
    replays: list[int] = []
    graphs: dict[int, GraphPlan] = {}

    def undo() -> None:
        for store in stores:
            store.clear(store.scratch())

    for size in sorted(sizes, reverse=True):
        graph = GraphPlan(size, torch.device("cpu"), plans.eager.backends)

        def forward(graph: GraphPlan = graph) -> None:
            assert graph.output is not None
            with torch.inference_mode():
                graph.output.copy_(model.forward(graph.inputs, graph.positions, graph.last_index))

        graph.capture(batch, model, stores, StandInRecorder(StandInGraph(forward, replays, size), undo))
        graphs[size] = graph
    replays.clear()
    plans.graphs = dict(sorted(graphs.items()))
    plans.deps = Dependencies([batch, *stores], [lambda: batch.binding], w.service("kernels.catalog@1"))
    return replays
