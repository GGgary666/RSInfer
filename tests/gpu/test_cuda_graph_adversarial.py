"""Common semantic tests for the baseline, graph implementation and adversarial variants."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
import torch

from core.contracts.kernels import Implementation, Solution
from core.contracts.plan import PlanKey
from platforms.base import StaleGraph
from platforms import PLATFORMS
from platforms.torch_eager import TorchEagerPlatform
from tests.units.support.worker import Driver, Worker, hf_model, save_checkpoint, tiny_config


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return save_checkpoint(hf_model(tiny_config()), tmp_path_factory.mktemp("abc-weights"))


def worker(checkpoint: Path, mode: str = "required") -> Worker:
    # Baseline A has only eager CUDA. It must fail a feature assertion, not an import.
    available: set[str] = set(PLATFORMS)
    platform = (importlib.import_module("platforms.cuda_graph").CudaGraphPlatform()
                if "cuda_graph" in available else TorchEagerPlatform(cuda=True))
    return Worker(tiny_config().to_dict(), platform=platform, **{
        "weights.path": str(checkpoint), "batch.max_rows": 4, "plan.cuda_graph": mode})


def require_graphs(w: Worker) -> Any:
    plans = w.service("plan@1")
    assert plans.buckets("decode") == [1, 2, 4], "decode CUDA graphs must be available"
    return plans


def observe_replays(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    replays: list[Any] = []
    replay = torch.cuda.CUDAGraph.replay

    def observed(raw: Any) -> None:
        replays.append(raw)
        replay(raw)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", observed)
    return replays


def test_decode_graph_matches_eager(checkpoint: Path) -> None:
    graph, eager = worker(checkpoint), worker(checkpoint, "off")
    try:
        require_graphs(graph)
        a, b = Driver(graph), Driver(eager)
        for d in (a, b):
            d.add(0, [5, 17, 3])
        for x, y in zip(a.generate([0], 5)[0], b.generate([0], 5)[0], strict=True):
            torch.testing.assert_close(x, y, rtol=2e-4, atol=2e-4)
        assert a.requests[0].outputs == b.requests[0].outputs
    finally:
        graph.close()
        eager.close()


def test_decode_replays_the_selected_cuda_graph(checkpoint: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    w = worker(checkpoint)
    try:
        plans = require_graphs(w)

        replays = observe_replays(monkeypatch)
        d = Driver(w)
        for row in range(3):
            d.add(row, [5 + row, 17, 3])
        d.step({0: 3, 1: 3, 2: 3})
        assert replays == [], "prefill must use eager"
        for chunks, bucket in (({0: 1}, 1), ({1: 1, 0: 1}, 2), ({2: 1, 0: 1, 1: 1}, 4)):
            raw = plans.get(PlanKey("decode", bucket)).graph.raw
            replays.clear()
            d.step(chunks)
            assert replays == [raw], "decode must replay its selected CUDA graph exactly once"
    finally:
        w.close()


def test_catalog_change_rejects_replay_before_state_writes(checkpoint: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    w = worker(checkpoint)
    undo: Any = None
    try:
        plans = require_graphs(w)

        d = Driver(w)
        d.add(0, [5, 17, 3])
        d.step({0: 3})
        old = plans.get(PlanKey("decode", 1))
        catalog = w.service("kernels.catalog@1")
        version = catalog.version
        batch = w.service("worker.batch@1")
        generation = batch.binding.generation
        fn = catalog.resolve(catalog.definition("gemm.linear@1").reference)
        undo = catalog.register(Implementation(
            Solution("abc.linear@1", "gemm.linear@1", "test:linear", capturable=True),
            lambda *args, **kwargs: fn(*args, **kwargs) * 0))
        assert catalog.version > version and batch.binding.generation == generation
        assert catalog.select("gemm.linear@1", {}).solution == "abc.linear@1"
        # Registration changed in the same catalog instance. No provider reload masks the guard.
        buffers = [*batch.binding.buffers().values()]
        for decl in w.service("model@1").assembly.states:
            service = "state.kv.paged@1" if decl.kind == "kv.paged" else "state.gdn.slots@1"
            buffers.extend(w.service(service).bind(decl.layer).buffers().values())
        before = [t.clone() for t in buffers]
        replays = observe_replays(monkeypatch)
        delta, plan, ops = d.records({0: 1})
        with pytest.raises(StaleGraph):
            w.service("executor@1").submit(2, delta, plan, ops)
        assert replays == [], "rejected graph must not launch GPU work"
        assert plans.buckets("decode") == [] and plans.get(PlanKey("decode", 1)) is old
        for actual, expected in zip(buffers, before, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        if undo is not None:
            undo()
        w.close()
