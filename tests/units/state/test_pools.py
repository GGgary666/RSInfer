"""state.kv_paged, state.gdn_slots and worker.batch: SGLang's layouts (``MHATokenToKVPool``,
``MambaPool``, ``HybridReqToTokenPool``), item copies and clears, and the persistent batch's
deltas, its Resync answers and its tables."""

from __future__ import annotations

import pytest
import torch

from core.contracts.columns import Column, ragged
from core.contracts.records import AddedRow, BatchPlan, RowDelta, StateAssign, StateOps
from models.qwen38.config import Qwen3_5TextConfig as Config
from models.qwen38.unit import states
from platforms.torch_eager import TorchEagerPlatform
from state.gdn_slots.unit import MambaPool
from state.kv_paged.unit import KVPool
from tests.units.support.worker import tiny_config
from worker.batch.unit import PersistentBatch

CFG = Config.from_hf(tiny_config().to_dict())


def test_kv_pages_are_token_slots_with_a_padding_page() -> None:
    pool = KVPool(TorchEagerPlatform(), 4, 8)
    decls = [s for s in states(CFG, "f32", 4) if s.kind == "kv.paged"]
    pool.declare(decls)
    k = pool.bind(3).k
    assert k.shape == (9 * 4, 2, 16) and pool.capacity() == 8 and list(pool.scratch()) == [-1]
    # item i is page i + 1; the padding page 0 is the scratch item
    assert pool.slots([0, 7]).tolist() == [4, 5, 6, 7, 32, 33, 34, 35]
    assert pool.slots(pool.scratch()).tolist() == [0, 1, 2, 3]
    k[pool.slots([2])] = 1.0
    pool.copy([2], [5])
    assert bool((k[24:28] == 1).all()) and not k[:12].any()
    pool.declare(decls)
    assert pool.bind(3).k.data_ptr() == k.data_ptr() and pool.generation == 0
    with pytest.raises(ValueError, match="pages of 16 tokens, not kv.paged 4"):
        KVPool(TorchEagerPlatform(), 16, 8).declare(decls)


def test_gdn_slots_are_one_tensor_per_state_across_layers() -> None:
    pool = MambaPool(TorchEagerPlatform(), 3)
    pool.declare([s for s in states(CFG, "f32", 4) if s.kind == "gdn.slots"])
    assert pool.conv is not None and pool.temporal is not None
    assert pool.conv.shape == (3, 4, 2 * 16 + 48, 3) and pool.temporal.shape == (3, 4, 4, 12, 8)
    assert pool.temporal.dtype == torch.float32 and pool.mamba_layer_ids == [0, 1, 2]
    assert pool.capacity() == 3 and list(pool.scratch()) == [-1]
    pool.bind(2).temporal[1] = 2.0  # slot 1: item 0
    pool.copy([0], [2])
    assert bool((pool.temporal[2, 3] == 2).all())
    pool.clear([0])
    assert not pool.temporal[:, 1].any() and bool((pool.temporal[2, 3] == 2).all())


def test_the_scratch_items_are_outside_the_items_the_ledger_hands_out() -> None:
    """Warm-up and padded rows write the scratch items (§5.2, probe P6). The ledger hands out
    ``0 .. capacity - 1``; requests holding item 0 or the last item keep their state."""
    kv, gdn = KVPool(TorchEagerPlatform(), 4, 8), MambaPool(TorchEagerPlatform(), 3)
    kv.declare([s for s in states(CFG, "f32", 4) if s.kind == "kv.paged"])
    gdn.declare([s for s in states(CFG, "f32", 4) if s.kind == "gdn.slots"])
    assert gdn.conv is not None and gdn.temporal is not None
    k, conv = kv.k_buffer[3].normal_(), gdn.conv.normal_()
    k_before, conv_before = k.clone(), conv.clone()
    for pool in (kv, gdn):
        pool.copy([pool.capacity() - 1], pool.scratch())
        pool.clear(pool.scratch())
    assert torch.equal(k[:4], k_before[32:]) and torch.equal(k[4:], k_before[4:])
    assert not conv[:, 0].any() and torch.equal(conv[:, 1:], conv_before[:, 1:])


def plan(rows: list[int], prev: list[int]) -> BatchPlan:
    return BatchPlan(Column.of("i32", rows), Column.of("i32", [1] * len(rows)), Column.of("i32", [0] * len(rows)),
                     Column.of("i64", [7] * (len(rows) - sum(prev))), Column.of("u8", prev), "decode", 0)


def test_the_batch_applies_deltas_into_req_to_token_and_answers_resync() -> None:
    batch = PersistentBatch(TorchEagerPlatform(), 4, 16, 4)
    rows = [AddedRow(0, 10), AddedRow(2, 12)]
    assert "waits for a full delta" in batch.check(RowDelta(rows), plan([0, 2], [0, 0]))  # a new batch
    added = RowDelta(rows, full=True)
    assert batch.check(added, plan([0, 2], [0, 0])) == ""
    assert "previous token" in batch.check(added, plan([0], [1]))
    assert "row 1 is not" in batch.check(added, plan([1], [0]))
    pages = StateAssign("kv.paged", Column.of("i32", [0, 2]), *ragged("i32", [[3], [0, 2]]))
    slots = StateAssign("gdn.slots", Column.of("i32", [0, 2]), *ragged("i32", [[5], [0]]))
    batch.apply(added, StateOps([pages, slots]))
    t = batch.binding.buffers()
    # item i is page or slot i + 1; index 0 (padding) is no item's
    assert t["req_to_token"][1, :4].tolist() == [16, 17, 18, 19]
    assert t["req_to_token"][3, :8].tolist() == [4, 5, 6, 7, 12, 13, 14, 15]
    assert t["slots.gdn.slots"].tolist() == [0, 6, 0, 1, 0] and batch.ids("kv.paged") == {0: [3], 2: [0, 2]}
    batch.record([2], torch.tensor([42]))
    assert batch.check(RowDelta(), plan([2], [1])) == "" and t["last_token"][3] == 42
    assert "row 3 is removed" in batch.check(RowDelta(removed=Column.of("i32", [3])), plan([2], [1]))
    assert "row 2 is not" in batch.check(RowDelta([AddedRow(0, 10)], full=True), plan([2], [1]))


def kv_pages(row: int, items: list[int]) -> StateOps:
    return StateOps([StateAssign("kv.paged", Column.of("i32", [row]), *ragged("i32", [items]))])


def test_a_delta_beyond_the_limits_of_the_batch_changes_nothing() -> None:
    """More pages than a row holds, a row outside the batch, items for a row it does not have: the
    scheduler admitted what this worker cannot hold, so ``apply`` raises before changing anything."""
    batch = PersistentBatch(TorchEagerPlatform(), 4, 6, 4)  # 6 tokens a row: 2 whole pages
    full = RowDelta([AddedRow(0, 10)], full=True)
    for delta, state, error in ((full, kv_pages(0, [0, 1, 2]), "3 pages of 4 tokens, more than 6 tokens"),
                                (RowDelta([AddedRow(4, 14)], full=True), StateOps(), "outside the batch of 4 rows"),
                                (full, kv_pages(1, [0]), "row 1 is given kv.paged items")):
        with pytest.raises(ValueError, match=error):
            batch.apply(delta, state)
        assert batch.rows == {} and batch.rebuilt and not batch.binding.buffers()["req_to_token"].any()
    batch.apply(full, kv_pages(0, [0, 1]))  # a request of 6 tokens fits
    assert batch.binding.buffers()["req_to_token"].shape == (5, 8) and not batch.rebuilt
    with pytest.raises(ValueError, match="3 pages"):
        batch.apply(RowDelta(), kv_pages(0, [2]))
    assert batch.ids("kv.paged") == {0: [0, 1]}
