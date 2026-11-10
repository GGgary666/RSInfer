"""Common semantic tests for the baseline, graph implementation and adversarial variants."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
import torch

from platforms import PLATFORMS
from platforms.torch_eager import TorchEagerPlatform
from tests.units.support.worker import Driver, Worker, hf_model, save_checkpoint, tiny_config


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return save_checkpoint(hf_model(tiny_config()), tmp_path_factory.mktemp("abc-weights"))


def worker(checkpoint: Path, mode: str = "required") -> Worker:
    # Baseline A has only eager CUDA. It must fail a feature assertion, not an import.
    platform = (importlib.import_module("platforms.cuda_graph").CudaGraphPlatform()
                if "cuda_graph" in PLATFORMS else TorchEagerPlatform(cuda=True))
    return Worker(tiny_config().to_dict(), platform=platform, **{
        "weights.path": str(checkpoint), "batch.max_rows": 4, "plan.cuda_graph": mode})


def require_graphs(w: Worker) -> Any:
    plans = w.service("plan@1")
    assert plans.buckets("decode") == [1, 2, 4], "decode CUDA graphs must be available"
    return plans


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
