"""The prefix cache's part in the worker (doc/specs/prefix-cache.md), on torch_eager (CPU, fp32):
the GDN layer keeps a row's state where the step's ``StateTrack`` says, inside a prefill chunk (the
chunk kernel's state before a chunk of 64, the conv window from the input), at the end of a chunk,
and when it decodes; the state kept is the one a row computed alone up to there, and a row restored
from it (``StateCopy(before)``) over the shared pages generates what the whole prompt gives."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from core.contracts.columns import Column
from core.contracts.records import StateCopy, StateTrack
from tests.units.support.worker import Driver, Worker, hf_model, save_checkpoint, tiny_config

PROMPT = [(7 * i + 3) % 97 for i in range(100)]
KIND = "gdn.slots"


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, object], Path]:
    cfg = tiny_config()
    return cfg.to_dict(), save_checkpoint(hf_model(cfg), tmp_path_factory.mktemp("ckpt"))


@pytest.fixture
def worker(checkpoint: tuple[dict[str, object], Path]) -> Iterator[Worker]:
    w = Worker(checkpoint[0], **{"weights.path": str(checkpoint[1]), "batch.max_context_len": 128,
                                 "state.kv.pages": 160, "state.gdn.slots": 8})
    yield w
    w.close()


def track(rows: list[int], dst: list[int], upto: list[int]) -> list[StateTrack]:
    return [StateTrack(KIND, Column.of("i32", rows), Column.of("i32", dst), Column.of("i32", upto))]


def state(worker: Worker, item: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Every layer's conv and temporal state of ``item`` (slot ``item + 1``)."""
    conv, temporal, _ = worker.service("state.gdn.slots@1").pools()
    return conv[:, item + 1].clone(), temporal[:, item + 1].clone()


def close(a: tuple[torch.Tensor, torch.Tensor], b: tuple[torch.Tensor, torch.Tensor]) -> None:
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-5)


def test_the_state_kept_inside_a_chunk_at_its_end_and_when_decoding_is_the_state_computed_up_to_there(
        worker: Worker) -> None:
    d = Driver(worker)
    d.add(0, PROMPT)  # item 0
    d.step({0: 100}, track=track([0], [6], [64]))  # inside: 64 of 100 tokens
    alone = d.add(1, PROMPT[:64])  # item 1, the same 64 tokens computed alone
    d.step({1: 64})
    close(state(worker, 6), state(worker, alone.slot))
    d.add(2, PROMPT[:64])  # item 2: kept at the end of its input
    d.step({2: 64}, track=track([2], [5], [64]))
    close(state(worker, 5), state(worker, alone.slot))
    d.step({0: 1}, track=track([0], [7], [101]))  # decoding
    assert all(torch.equal(x, y) for x, y in zip(state(worker, 7), state(worker, 0), strict=True))


def relative(a: tuple[torch.Tensor, torch.Tensor], b: tuple[torch.Tensor, torch.Tensor]) -> float:
    return max(float((x - y).norm() / y.norm()) for x, y in zip(a, b, strict=True))


def test_a_row_restored_from_a_kept_state_over_shared_pages_continues_as_the_whole_prompt(worker: Worker) -> None:
    """Four tokens after the restore: the GDN state fades within a few dozen tokens here, so a row
    restored from nothing (the control) is still far off after four, and right after forty."""
    d = Driver(worker)
    whole = d.add(0, PROMPT[:68])
    want_logits = d.step({0: 68}).logits
    want = state(worker, whole.slot)
    kept = d.add(1, PROMPT[:64])
    d.step({1: 64}, track=track([1], [6], [64]))
    hit = d.add(2, PROMPT[:68])
    hit.pages, hit.computed = list(kept.pages[:64 // 4]), 64  # the cached prefix: 16 pages, the state at 64
    got = d.step({2: 4}, copies=[StateCopy(KIND, "before", Column.of("i32", [6]), Column.of("i32", [hit.slot]))])
    assert relative(state(worker, hit.slot), want) < 1e-5
    torch.testing.assert_close(got.logits, want_logits, rtol=1e-4, atol=1e-4)
    control = d.add(3, PROMPT[:68])
    control.pages, control.computed = list(kept.pages[:64 // 4]), 64
    d.step({3: 4})  # the same pages, its slot not restored
    assert relative(state(worker, control.slot), want) > 1e-2
    assert hit.outputs == whole.outputs  # and greedy, the same next token


def test_a_position_neither_at_the_end_nor_on_a_chunk_is_refused(worker: Worker) -> None:
    d = Driver(worker)
    d.add(0, PROMPT)
    with pytest.raises(ValueError, match="not its end nor a multiple of 64"):
        d.step({0: 100}, track=track([0], [6], [50]))
