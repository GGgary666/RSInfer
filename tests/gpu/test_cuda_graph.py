"""Real TP1 capture/replay, changing inputs, state, buckets and providers. No GPU skips."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from contextlib import contextmanager
from collections.abc import Generator
import gc

import pytest
import torch

from core.contracts.plan import PlanKey
from core.contracts.columns import Column
from core.contracts.records import Resync, StateCopy
from platforms.base import StaleGraph
from platforms.cuda_graph import CudaGraph, CudaGraphPlatform, Recorder
from tests.units.support.worker import Driver, Worker, hf_greedy, hf_model, save_checkpoint, spec, tiny_config
from unit.runtime import ReloadPlan


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return save_checkpoint(hf_model(tiny_config()), tmp_path_factory.mktemp("graph-weights"))


def worker(checkpoint: Path, mode: str = "required", platform: CudaGraphPlatform | None = None, **config: Any) -> Worker:
    return Worker(tiny_config().to_dict(), platform=platform or CudaGraphPlatform(), **{
        "weights.path": str(checkpoint), "batch.max_rows": 4, "plan.cuda_graph": mode, **config})


def state(w: Worker) -> list[torch.Tensor]:
    kv, gdn = w.service("state.kv.paged@1"), w.service("state.gdn.slots@1")
    return [*(kv.bind(layer).buffers()[name][kv.page_size:(kv.pages + 1) * kv.page_size].clone()
              for layer in kv.k_buffer for name in ("k", "v")),
            gdn.conv[:, 1:gdn.size + 1].clone(), gdn.temporal[:, 1:gdn.size + 1].clone()]


def assert_state(a: Worker, b: Worker) -> None:
    for x, y in zip(state(a), state(b), strict=True):
        torch.testing.assert_close(x, y, rtol=2e-4, atol=2e-4)


def test_platform_reuses_stream_and_recovers_failed_capture() -> None:
    p = CudaGraphPlatform()
    first, second = p.graph(), p.graph()
    assert first.stream is second.stream
    x = torch.ones(4, device="cuda")
    with first.warmup():
        x.add_(1)
    with first.capture() as graph:
        y = x * 2
    x.fill_(3)
    graph.replay()
    torch.testing.assert_close(y, torch.full_like(y, 6))
    graph.close()
    with pytest.raises(StaleGraph):
        graph.replay()
    with pytest.raises(RuntimeError):
        with second.capture():
            x.sum().item()
    with first.capture() as recovered:
        z = x + 1
    recovered.replay()
    torch.testing.assert_close(z, x + 1)
    recovered.close()


def test_full_forward_state_padding_and_bucket_switches(checkpoint: Path) -> None:
    eager, graph = worker(checkpoint, "off"), worker(checkpoint)
    try:
        assert graph.service("plan@1").buckets("decode") == [1, 2, 4]
        assert graph.service("plan@1").buckets("prefill") == []
        # Construction must not touch any item the scheduler can allocate.
        assert all(torch.count_nonzero(t).item() == 0 for t in state(graph))
        a, b = Driver(eager), Driver(graph)
        for d in (a, b):
            for row, prompt in ((0, [5, 17, 3]), (1, [9, 2, 8, 4, 7]), (2, [6, 1])):
                d.add(row, prompt)
        rounds = [{0: 3, 1: 5, 2: 2}, {2: 1, 0: 1, 1: 1}, {1: 1}, {0: 1, 1: 1},
                  {1: 1, 2: 1, 0: 1}, {2: 1}, {0: 1, 2: 1}]
        for i, chunks in enumerate(rounds):
            copies = ([StateCopy("gdn.slots", "after", Column.of("i32", [0]), Column.of("i32", [4]))] if i == 1 else
                      [StateCopy("gdn.slots", "before", Column.of("i32", [4]), Column.of("i32", [0]))] if i == 4 else [])
            x, y = a.step(chunks, copies), b.step(chunks, copies)
            torch.testing.assert_close(y.logits, x.logits, rtol=2e-4, atol=2e-4)
            assert y.finish() == x.finish()
            assert_state(eager, graph)
        # Eager prefill after replay, then another padded graph with remapped state.
        for d in (a, b):
            d.remove(1)
            d.add(1, [8, 3, 7, 6])
        for chunks in ({1: 4, 0: 1}, {1: 1, 0: 1, 2: 1}):
            x, y = a.step(chunks), b.step(chunks)
            torch.testing.assert_close(y.logits, x.logits, rtol=2e-4, atol=2e-4)
            assert_state(eager, graph)
    finally:
        graph.close()
        eager.close()


def test_hf_guard_refit_and_kernel_reload(checkpoint: Path) -> None:
    w = worker(checkpoint)
    try:
        d = Driver(w)
        d.add(0, [5, 17, 3])
        logits = d.generate([0], 5)[0]
        expected_tokens, expected = hf_greedy(hf_model(tiny_config()), [5, 17, 3], 5)
        assert d.requests[0].outputs == expected_tokens
        for x, y in zip(logits, expected, strict=True):
            torch.testing.assert_close(x.cpu(), y, rtol=2e-4, atol=2e-4)
        plans, weights = w.service("plan@1"), w.service("state.weights@1")
        old = plans.get(PlanKey("decode", 1))
        gen = weights.bind("").generation
        weights.fill("")
        assert weights.bind("").generation == gen and old.valid()
        # Changed contents must be visible without recapturing or moving the bindings.
        d = Driver(w)
        d.add(0, [5, 17, 3])
        refitted = d.generate([0], 3)[0]
        assert not torch.allclose(refitted[0], logits[0])
        eager = worker(checkpoint, "off")
        try:
            eager.service("state.weights@1").fill("")
            reference = Driver(eager)
            reference.add(0, [5, 17, 3])
            for x, y in zip(refitted, reference.generate([0], 3)[0], strict=True):
                torch.testing.assert_close(x, y, rtol=2e-4, atol=2e-4)
            assert plans.get(PlanKey("decode", 1)) is old
        finally:
            eager.close()
        delta, plan, ops = d.records({0: 1})
        batch = w.service("worker.batch@1")
        batch.generation += 1
        before = state(w)
        with pytest.raises(StaleGraph):
            w.service("executor@1").submit(100, delta, plan, ops)
        for x, y in zip(before, state(w), strict=True):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        w.runtime.reload(ReloadPlan(units_to_reload=[spec("kernels.reference"), spec("core.plan")])).unwrap()
        assert not old.valid()
        assert w.service("plan@1").buckets("decode") == [1, 2, 4]
        assert w.service("plan@1").get(PlanKey("decode", 1)) is not old
    finally:
        w.close()


def test_required_and_auto_handle_uncapturable_backend(checkpoint: Path) -> None:
    config: dict[str, Any] = {"attn.decode_backend": "reference"}
    w = worker(checkpoint, "auto", **config)
    try:
        assert w.service("plan@1").buckets("decode") == []
    finally:
        w.close()
    with pytest.raises(Exception, match="uncapturable decode kernels"):
        worker(checkpoint, **config)


def test_candidate_failure_releases_partial_set(checkpoint: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    platform = CudaGraphPlatform()
    warm = worker(checkpoint, platform=platform)
    warm.close()
    del warm
    torch.cuda.synchronize()
    gc.collect()
    baseline = torch.cuda.memory_allocated()
    capture = Recorder.capture
    calls = 0

    @contextmanager
    def fail_second(self: Recorder) -> Generator[CudaGraph]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second-bucket failure")
        with capture(self) as graph:
            yield graph

    monkeypatch.setattr(Recorder, "capture", fail_second)
    with pytest.raises(Exception, match="injected second-bucket failure"):
        worker(checkpoint, platform=platform)
    torch.cuda.synchronize()
    gc.collect()
    assert torch.cuda.memory_allocated() <= baseline + (1 << 20)


def test_overlap_and_repeated_rebuild_keep_outputs_and_memory(checkpoint: Path) -> None:
    w = worker(checkpoint)
    try:
        def run(depth: int) -> list[list[int | float]]:
            d = Driver(w)
            d.add(0, [5, 17, 3])
            d.add(1, [9, 2, 8])
            d.step({0: 3, 1: 3})
            pending: list[Any] = []
            out: list[list[int | float]] = []
            for _ in range(12):
                delta, plan, ops = d.records({0: 1, 1: 1})
                step = w.service("executor@1").submit(d.epoch + 1, delta, plan, ops)
                assert not isinstance(step, Resync)
                d.epoch += 1
                d.full, d.removed = False, []
                for request in d.requests.values():
                    request.computed += 1
                    request.known.append(-1)
                pending.append(step)
                if len(pending) >= depth:
                    out.append(list(pending.pop(0).finish().tokens.values()))
            out.extend(list(s.finish().tokens.values()) for s in pending)
            return out

        assert run(2) == run(1)
        allocated: list[int] = []
        for _ in range(4):
            w.service("executor@1").synchronize()
            w.runtime.reload(ReloadPlan(units_to_reload=[spec("kernels.reference"), spec("core.plan")])).unwrap()
            assert w.service("plan@1").buckets("decode") == [1, 2, 4]
            gc.collect()
            allocated.append(torch.cuda.memory_allocated())
        assert max(allocated[1:]) - min(allocated[1:]) <= (1 << 20), allocated
    finally:
        w.close()
