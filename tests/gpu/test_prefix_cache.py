"""The prefix cache's part in the worker on a CUDA GPU (doc/specs/prefix-cache.md), Qwen3.8-27B's real
structure cut to 4 layers, bf16, the real kernels (``auto``): where a step's ``StateTrack`` says, the
GDN layer keeps a row's state, inside a prefill chunk from the Triton chunk kernel's ``h`` and at a
decode step in one Triton kernel for every layer, also in a replayed graph.

In bf16 a state kept from inside a chunk is ``h`` rounded to bf16, as in SGLang (its Triton GDN kernel
has no fp32 snapshot): close to the fp32 state computed up to there, not equal. Manual (``tests/gpu``
is not collected by default); ``QWEN38_MODEL`` reuses a checkpoint."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from core.contracts.columns import Column
from core.contracts.records import StateCopy, StateTrack
from platforms.cuda_graph import CudaGraphPlatform
from platforms.torch_eager import TorchEagerPlatform
from tests.gpu.support import free_gpu, make_checkpoint
from tests.units.support.worker import Driver, Worker

KIND = "gdn.slots"
PROMPT = [(1000 + 37 * i) % 150000 for i in range(200)]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return Path(os.environ["QWEN38_MODEL"]) if os.environ.get("QWEN38_MODEL") else make_checkpoint(tmp_path_factory, 4)


def worker(checkpoint: Path, graphs: bool, **config: Any) -> Worker:
    """The test driver counts pages of 4 tokens (``Driver``) where these are 16: a row gets more
    pages than it fills, so the rows may be 1024 tokens long."""
    cfg = json.loads((checkpoint / "config.json").read_text())
    return Worker(cfg, platform=CudaGraphPlatform() if graphs else TorchEagerPlatform(cuda=True), **{
        "model.dtype": "bf16", "state.gdn.conv_dtype": "bf16", "weights.path": str(checkpoint),
        "state.kv.page_size": 16, "state.kv.pages": 256, "state.gdn.slots": 8, "batch.max_rows": 4,
        "batch.max_context_len": 1024, "plan.cuda_graph": "required" if graphs else "off", **config})


def track(rows: list[int], dst: list[int], upto: list[int]) -> list[StateTrack]:
    return [StateTrack(KIND, Column.of("i32", rows), Column.of("i32", dst), Column.of("i32", upto))]


def state(w: Worker, item: int) -> tuple[torch.Tensor, torch.Tensor]:
    conv, temporal, _ = w.service("state.gdn.slots@1").pools()
    return conv[:, item + 1].float().clone(), temporal[:, item + 1].float().clone()


def relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


@pytest.mark.timeout(1800)
def test_a_state_kept_inside_or_at_the_end_of_a_prefill_chunk_is_the_state_computed_up_to_there(
        checkpoint: Path) -> None:
    """Against the state of the same 128 tokens computed alone, to bf16 (another batch shape rounds the
    GEMMs otherwise); a state one token off is a hundred times further."""
    w = worker(checkpoint, False)
    try:
        resolved = w.service("plan@1").resolved
        assert resolved["prefill"]["layers.gdn@1"]["gdn.chunk_delta_rule@1"] == "triton.gdn.chunk_delta_rule@1"
        d = Driver(w)
        d.add(0, PROMPT)
        d.step({0: 200}, track=track([0], [6], [128]))  # inside: the conv window from the input, h
        alone = d.add(1, PROMPT[:128])
        d.step({1: 128})
        early = d.add(2, PROMPT[:127])  # the control
        d.step({2: 127})
        want = state(w, alone.slot)
        kept = [relative(a, b) for a, b in zip(state(w, 6), want, strict=True)]
        off = [relative(a, b) for a, b in zip(state(w, early.slot), want, strict=True)]
        print(f"kept inside a chunk (conv, temporal): {kept[0]:.1e}, {kept[1]:.1e}; one token off: "
              f"{off[0]:.1e}, {off[1]:.1e}")
        assert max(kept) < 5e-2 and min(off) > 0.5
        d.add(3, PROMPT[:128])
        d.step({3: 128}, track=track([3], [5], [128]))  # at the end: the state itself
        assert max(relative(a, b) for a, b in zip(state(w, 5), want, strict=True)) < 5e-2
    finally:
        w.close()
        free_gpu()


@pytest.mark.timeout(1800)
def test_a_decode_step_replayed_from_a_graph_keeps_the_states_the_eager_one_keeps(checkpoint: Path) -> None:
    """The bucket's mask and destinations are fixed buffers, the Triton copy is in the graph: the
    kept slot is the row's slot after the step (bit for bit), and the eager worker's to bf16; a
    replay of the same bucket after it keeps nothing."""
    eager, graph = worker(checkpoint, False), worker(checkpoint, True, **{"plan.graph_buckets": [1, 2, 4]})
    try:
        assert graph.service("plan@1").resolved["graph"]["layers.gdn@1"]["gdn.track_states@1"] == \
            "triton.gdn.track_states@1"
        a, b = Driver(eager), Driver(graph)
        for d in (a, b):
            for row, prompt in ((0, PROMPT[:40]), (1, PROMPT[40:48]), (2, PROMPT[48:50])):
                d.add(row, prompt)
        rounds: list[tuple[dict[int, int], list[StateTrack]]] = [
            ({0: 40, 1: 8, 2: 2}, []), ({1: 1}, track([1], [4], [9])), ({2: 1}, []),
            ({0: 1, 2: 1}, track([2, 0], [5, 6], [4, 42])), ({2: 1, 0: 1}, [])]
        for chunks, tracked in rounds:
            before = [state(graph, i) for i in (4, 5, 6)]
            x, y = a.step(chunks, track=tracked), b.step(chunks, track=tracked)
            assert y.finish() == x.finish()
            kept = {int(r): int(t) for t in tracked for r, t in zip(t.rows.values(), t.dst.values(), strict=True)}
            for row, item in kept.items():
                assert all(torch.equal(u, v) for u, v in zip(state(graph, item), state(graph, row), strict=True))
                for u, v in zip(state(graph, item), state(eager, item), strict=True):
                    torch.testing.assert_close(u, v, rtol=2e-2, atol=2e-2)
            for i, item in enumerate((4, 5, 6)):
                if item not in kept.values():  # untouched
                    assert all(torch.equal(u, v) for u, v in zip(state(graph, item), before[i], strict=True))
    finally:
        eager.close()
        graph.close()
        free_gpu()


@pytest.mark.timeout(1800)
def test_a_row_restored_from_a_kept_state_continues_where_the_whole_prompt_does(checkpoint: Path) -> None:
    """Over the shared pages, from the state kept at 128, two more tokens: the state is the whole
    prompt's to bf16, where a row restored from nothing is far off (the GDN state fades within a few
    dozen tokens with these weights, so the check is two tokens after the restore)."""
    w = worker(checkpoint, False)
    try:
        d = Driver(w)
        whole = d.add(0, PROMPT[:130])
        d.step({0: 130})
        want = state(w, whole.slot)
        kept = d.add(1, PROMPT[:128])
        d.step({1: 128}, track=track([1], [6], [128]))
        hit = d.add(2, PROMPT[:130])
        hit.pages, hit.computed = list(kept.pages[:128 // 16]), 128  # 16-token pages
        d.step({2: 2}, copies=[StateCopy(KIND, "before", Column.of("i32", [6]), Column.of("i32", [hit.slot]))])
        control = d.add(3, PROMPT[:130])
        control.pages, control.computed = list(kept.pages[:128 // 16]), 128
        d.step({3: 2})
        got = max(relative(a, b) for a, b in zip(state(w, hit.slot), want, strict=True))
        bad = max(relative(a, b) for a, b in zip(state(w, control.slot), want, strict=True))
        print(f"state two tokens after the restore: {got:.1e}; restored from nothing: {bad:.1e}")
        assert got < 5e-2 and bad > 0.3
        assert hit.outputs == whole.outputs
    finally:
        w.close()
        free_gpu()
