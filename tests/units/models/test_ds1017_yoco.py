"""Bounded replay through real DS1017 worker plans, including chunks and prefix restoration."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from core.contracts.columns import Column
from core.contracts.records import StateCopy, StateTrack
from kernels.ds1017_yoco import gather, retained
from tests.units.support.ds1017 import DSWorker, checkpoint, driver, tiny_config


@pytest.fixture(scope="module")
def model(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], Path]:
    cfg = tiny_config(engram_layer_ids=[1])
    return cfg, checkpoint(tmp_path_factory.mktemp("yoco"), cfg)


def worker(model: tuple[dict[str, Any], Path], mode: str, keep: int = 9) -> DSWorker:
    return DSWorker(model[0], **{"weights.path": str(model[1]), "dsv4.yoco_mode": mode,
                                "dsv4.yoco_keep_tokens": keep, "alloc.prefix_replay_tokens": keep})


@pytest.mark.parametrize("chunk", [None, 1, 5, 11])
def test_compact_matches_full_mask_prefill_and_subsequent_decode(model: tuple[dict[str, Any], Path],
                                                               chunk: int | None) -> None:
    # Ragged rows, one short prompt, sliding-window wrap, and chunks before the retained suffix.
    prompts = {0: [i % 120 for i in range(39)], 1: [7, 21, 4, 9, 11], 2: list(range(20, 46))}
    saved: list[tuple[dict[int, list[int]], dict[int, list[torch.Tensor]], list[torch.Tensor]]] = []
    for mode in ("mask", "compact"):
        w = worker(model, mode)
        try:
            d = driver(w)
            for row, ids in prompts.items():
                d.add(row, ids)
            logits = d.generate(list(prompts), 5, chunk)
            kv = w.service("state.kv.paged@1")
            tensors = [v.clone() for layer in (1, 3) for v in kv.bind(layer).buffers().values()]
            saved.append(({row: list(r.outputs) for row, r in d.requests.items()}, logits, tensors))
        finally:
            w.close()
    a, b = saved
    assert a[0] == b[0]
    for row in prompts:
        torch.testing.assert_close(torch.stack(a[1][row]), torch.stack(b[1][row]), rtol=0, atol=0)
    for u, v in zip(a[2], b[2], strict=True):
        assert torch.equal(u, v)


def test_short_prompts_are_identical_to_full_replay(model: tuple[dict[str, Any], Path]) -> None:
    outputs = []
    for mode in ("off", "mask", "compact"):
        w = worker(model, mode)
        try:
            d = driver(w)
            d.add(0, [1, 2, 6, 9, 3, 8, 5])
            outputs.append(torch.stack(d.generate([0], 5)[0]))
        finally:
            w.close()
    assert all(torch.equal(outputs[0], x) for x in outputs[1:])


def test_an_early_prefix_checkpoint_restores_and_replays_the_decoder_tail(model: tuple[dict[str, Any], Path]) -> None:
    w = worker(model, "compact")
    try:
        prompt = list(range(24))
        cold = driver(w)
        cold.add(2, prompt)
        expected = cold.generate([2], 5)[2]

        d = driver(w)
        parent = d.add(0, list(range(50)))
        snapshot = d.next_slot
        d.next_slot += 1
        # The parent's decoder has only placeholders at p=12, far before its own retained suffix.
        d.step({0: 12}, [StateCopy("swa.slots", "after", Column.of("i32", [parent.slot]),
                                 Column.of("i32", [snapshot]))])
        slots = w.service("state.swa.slots@1")
        for layer in (3, 4, 5):
            assert torch.count_nonzero(slots.bind(layer).buffers()["ring"][parent.slot + 1]) == 0
        child = d.add(1, prompt)
        child.pages = list(parent.pages[:3])
        child.computed = 12  # 12 remaining tokens cover keep=9
        copied = StateCopy("swa.slots", "before", Column.of("i32", [snapshot]), Column.of("i32", [child.slot]))
        first = d.step({1: 12}, [copied]).logits[0].clone()
        actual = [first, *d.generate([1], 5)[1]]
        torch.testing.assert_close(torch.stack(actual), torch.stack(expected), rtol=0, atol=0)
    finally:
        w.close()


def test_mid_chunk_window_track_restores_and_replays_without_splitting_parent_step(
    model: tuple[dict[str, Any], Path],
) -> None:
    w = worker(model, "compact")
    try:
        prompt = list(range(24))
        cold = driver(w)
        cold.add(2, prompt)
        expected = cold.generate([2], 5)[2]

        d = driver(w)
        parent_prompt = [*prompt[:12], *range(50, 88)]
        parent = d.add(0, parent_prompt)
        snapshot = d.next_slot
        d.next_slot += 1
        # Track p=12 inside a single 50-token parent prefill step without splitting at 12.
        delta, plan, state = d.records({0: len(parent_prompt)})
        from msgspec import structs

        tracked = structs.replace(
            state,
            track=[StateTrack("swa.slots", Column.of("i32", [0]), Column.of("i32", [snapshot]), Column.of("i32", [12]))],
        )
        res = w.service("executor@1").submit(d.epoch, delta, plan, tracked)
        res.finish()
        d.epoch += 1
        parent.computed = len(parent_prompt)

        slots = w.service("state.swa.slots@1")
        for layer in (3, 4, 5):
            assert torch.count_nonzero(slots.bind(layer).buffers()["ring"][snapshot + 1]) == 0
        child = d.add(1, prompt)
        child.pages = list(parent.pages[:3])
        child.computed = 12
        copied = StateCopy("swa.slots", "before", Column.of("i32", [snapshot]), Column.of("i32", [child.slot]))
        first = d.step({1: 12}, [copied]).logits[0].clone()
        actual = [first, *d.generate([1], 5)[1]]
        torch.testing.assert_close(torch.stack(actual), torch.stack(expected), rtol=0, atol=0)
    finally:
        w.close()


def test_checkpoint_metadata_retains_a_placeholder_and_absolute_suffix_positions() -> None:
    view = retained([1, 2, 3], [8, 6, 10], [0, 6, 20], [30, 12, 30], 9, torch.device("cpu"), compact=True)
    assert view.counts == [1, 6, 9]
    assert view.starts == [7, 6, 21] and view.lower == [21, 3, 21]
    assert view.indices.tolist() == [7, *range(8, 14), *range(15, 24)]
    x = torch.arange(36).reshape(6, 6)[:, :4]
    got = gather(x, torch.tensor([5, -1, 2, 6], dtype=torch.int32))
    assert torch.equal(got, torch.stack([x[5], torch.zeros_like(x[0]), x[2], torch.zeros_like(x[0])]))


def test_incompatible_shapes_and_cache_replay_are_refused(model: tuple[dict[str, Any], Path]) -> None:
    with pytest.raises(Exception, match="sliding_window"):
        worker(model, "compact", keep=7)
    with pytest.raises(Exception, match="prefix_replay_tokens"):
        DSWorker(model[0], **{"weights.path": str(model[1]), "dsv4.yoco_mode": "compact",
                             "dsv4.yoco_keep_tokens": 9, "alloc.prefix_replay_tokens": 1})


def test_a_plan_without_prompt_lengths_uses_full_replay(model: tuple[dict[str, Any], Path]) -> None:
    from msgspec import structs

    outputs = []
    for mode in ("off", "compact"):
        w = worker(model, mode)
        try:
            d = driver(w)
            d.add(0, list(range(24)))
            delta, plan, state = d.records({0: 24})
            old = structs.replace(plan, prompt_lengths=Column("i32"))
            result = w.service("executor@1").submit(1, delta, old, state)
            result.finish()
            outputs.append(result.logits.clone())
        finally:
            w.close()
    assert torch.equal(*outputs)
