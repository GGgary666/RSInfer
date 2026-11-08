"""executors.eager with every worker unit of stage 2, on torch_eager (CPU, fp32): greedy
generation follows the HF model token by token, and the same requests give the same tokens and,
within tolerance, the same logits however they are batched, chunked, rebuilt or restored from a
snapshot (stage2-contract.md §1.3; the comparisons are not bitwise, §6)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from core.contracts.columns import Column
from core.contracts.records import Resync, StateCopy
from tests.units.support.worker import Driver, Worker, hf_greedy, hf_model, save_checkpoint, spec, tiny_config
from unit.runtime import ReloadPlan

PROMPTS = {0: [5, 17, 3, 44, 9, 61, 2], 1: [8, 30, 30, 1, 77, 12, 40, 3, 19, 6, 55]}
STEPS = 6


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, object], Path, object]:
    cfg = tiny_config()
    model = hf_model(cfg)
    return cfg.to_dict(), save_checkpoint(model, tmp_path_factory.mktemp("ckpt")), model


@pytest.fixture
def worker(checkpoint: tuple[dict[str, object], Path, object]) -> Iterator[Worker]:
    w = Worker(checkpoint[0], **{"weights.path": str(checkpoint[1])})
    yield w
    w.close()


def run(worker: Worker, rows: list[int], chunk: int | None = None) -> tuple[dict[int, list[int]], dict[int, list[torch.Tensor]]]:
    d = Driver(worker)
    for row in rows:
        d.add(row, PROMPTS[row])
    logits = d.generate(rows, STEPS, chunk)
    return {row: d.requests[row].outputs for row in rows}, logits


def close(a: torch.Tensor, b: torch.Tensor) -> None:
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


def same(a: dict[int, list[torch.Tensor]], b: dict[int, list[torch.Tensor]]) -> None:
    for row in a:
        close(torch.stack(a[row]), torch.stack(b[row]))


def test_greedy_generation_follows_hf_token_by_token(worker: Worker, checkpoint: tuple[dict[str, object], Path, object]) -> None:
    for row in PROMPTS:
        tokens, logits = run(worker, [row])
        want_tokens, want_logits = hf_greedy(checkpoint[2], PROMPTS[row], STEPS)  # type: ignore[arg-type]
        assert tokens[row] == want_tokens
        close(torch.stack(logits[row]), torch.stack(want_logits))


def test_rows_batched_together_give_what_they_give_alone(worker: Worker) -> None:
    """Row 0 decodes while row 1 still prefills: mixed steps, then decode steps of both."""
    tokens, logits = run(worker, [0, 1], chunk=4)
    for row in PROMPTS:
        alone_tokens, alone_logits = run(worker, [row])
        assert tokens[row] == alone_tokens[row]
        same({row: logits[row]}, alone_logits)


def test_a_chunked_prefill_gives_what_one_prefill_gives(worker: Worker) -> None:
    whole_tokens, whole = run(worker, [1])
    for chunk in (1, 3, 4):
        tokens, logits = run(worker, [1], chunk)
        assert tokens == whole_tokens
        same(logits, whole)


def test_a_rebuilt_batch_answers_resync_until_a_full_delta_then_continues(worker: Worker) -> None:
    want, _ = run(worker, [0, 1])
    d = Driver(worker)
    for row in PROMPTS:
        d.add(row, PROMPTS[row])
    d.generate([0, 1], 2)
    worker.runtime.reload(ReloadPlan(units_to_reload=[spec("worker.batch")])).unwrap()
    resync = d.step({0: 1, 1: 1})  # the rebuilt batch has none of these rows, nor their last tokens
    assert isinstance(resync, Resync) and "row 0 is not in the persistent batch" in resync.reason
    assert worker.service("worker.batch@1").rows == {}  # nothing applied, nothing ran
    d.rebuilt()
    d.generate([0, 1], STEPS)
    assert {row: d.requests[row].outputs for row in PROMPTS} == want


def test_a_request_restored_from_a_snapshot_continues_as_one_computed_from_scratch(worker: Worker) -> None:
    """Request A's GDN state is snapshotted when its prompt ends (StateCopy after); B, whose prompt
    starts with A's, reads A's KV pages and restores the snapshot (StateCopy before) instead of
    computing the shared prefix."""
    prefix, rest = PROMPTS[1][:8], PROMPTS[1][8:]
    scratch_tokens, scratch = run(worker, [1])

    d = Driver(worker)
    a = d.add(0, prefix)
    snapshot_slot = d.next_slot
    d.next_slot += 1
    d.step({0: len(prefix)}, [StateCopy("gdn.slots", "after", Column.of("i32", [a.slot]),
                                        Column.of("i32", [snapshot_slot]))])
    b = d.add(1, PROMPTS[1])
    b.pages, b.computed = list(a.pages[:len(prefix) // 4]), len(prefix)
    restore = StateCopy("gdn.slots", "before", Column.of("i32", [snapshot_slot]), Column.of("i32", [b.slot]))
    first = d.step({1: len(rest)}, [restore])
    logits = [first.logits[0]]
    logits += d.generate([1], STEPS)[1]
    assert b.outputs == scratch_tokens[1]
    close(torch.stack(logits), torch.stack(scratch[1]))


def test_capacity_counts_the_usable_items_of_each_kind(worker: Worker) -> None:
    assert worker.service("executor@1").capacity().items == {"kv.paged": 32, "gdn.slots": 6}
