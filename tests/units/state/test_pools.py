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
    assert k.shape == (9 * 4, 2, 16) and pool.capacity() == 8 and list(pool.scratch()) == [0]
    k[pool.slots([2])] = 1.0
    pool.copy([2], [5])
    assert bool((k[20:24] == 1).all()) and not k[:8].any()
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
    pool.bind(2).temporal[1] = 2.0
    pool.copy([1], [3])
    assert bool((pool.temporal[2, 3] == 2).all())
    pool.clear([1])
    assert not pool.temporal[:, 1].any() and bool((pool.temporal[2, 3] == 2).all())


def plan(rows: list[int], prev: list[int]) -> BatchPlan:
    return BatchPlan(Column.of("i32", rows), Column.of("i32", [1] * len(rows)), Column.of("i32", [0] * len(rows)),
                     Column.of("i64", [7] * (len(rows) - sum(prev))), Column.of("u8", prev), "decode", 0)


def test_the_batch_applies_deltas_into_req_to_token_and_answers_resync() -> None:
    batch = PersistentBatch(TorchEagerPlatform(), 4, 16, 4)
    added = RowDelta([AddedRow(0, 10), AddedRow(2, 12)])
    assert batch.check(added, plan([0, 2], [0, 0])) == ""
    assert "previous token" in batch.check(added, plan([0], [1]))
    assert "row 1 is not" in batch.check(added, plan([1], [0]))
    pages = StateAssign("kv.paged", Column.of("i32", [0, 2]), *ragged("i32", [[3], [1, 2]]))
    slots = StateAssign("gdn.slots", Column.of("i32", [0, 2]), *ragged("i32", [[5], [6]]))
    batch.apply(added, StateOps([pages, slots]))
    t = batch.binding.buffers()
    assert t["req_to_token"][1, :4].tolist() == [12, 13, 14, 15] and t["req_to_token"][3, :8].tolist() == [4, 5, 6, 7, 8,
                                                                                                         9, 10, 11]
    assert t["slots.gdn.slots"].tolist() == [0, 5, 0, 6, 0] and batch.ids("kv.paged") == {0: [3], 2: [1, 2]}
    batch.record([2], torch.tensor([42]))
    assert batch.check(RowDelta(), plan([2], [1])) == "" and t["last_token"][3] == 42
    assert "row 3 is removed" in batch.check(RowDelta(removed=Column.of("i32", [3])), plan([2], [1]))
    assert "row 2 is not" in batch.check(RowDelta([AddedRow(0, 10)], full=True), plan([2], [1]))
    with pytest.raises(ValueError, match="more than 16"):
        batch.apply(RowDelta(), StateOps([StateAssign("kv.paged", Column.of("i32", [2]), *ragged("i32", [[4, 5, 6]]))]))
