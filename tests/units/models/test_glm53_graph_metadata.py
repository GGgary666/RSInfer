"""The GLM decode graphs keep metadata addresses when live rows and pages change."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import torch

from core.contracts.plan import PersistentBatch
from layers.glm53.metadata import IndexShard, Row, SparseStep, Verify, device_metadata, pad_decode
from layers.glm53.unit import GlmLayers
from models.glm53.weights import EMBED, HEAD, PREFIX, Weights
from models.glm53_config import KDA, KV, Config


class Batch:
    def __init__(self, pages: list[int], slot: int, track: int) -> None:
        self.items = {KV: {0: pages}, KDA: {0: [slot]}}
        self.tracking = ([0], [track], [-128])

    def ids(self, kind: str) -> dict[int, list[int]]:
        return self.items[kind]

    def track(self, kind: str) -> tuple[list[int], list[int], list[int]]:
        assert kind == KDA
        return self.tracking


def test_decode_graph_metadata_preserves_addresses_and_isolates_padding() -> None:
    cpu = torch.device("cpu")
    rows = pad_decode([Row(0, 127, 5, 0)], 2, 5)
    first = cast(PersistentBatch, Batch([2, 8], 2, 4))
    sparse = SparseStep.prepare(first, rows, 128, 8, 256, cpu, context=4096)
    verify = Verify.prepare(first, rows, cpu, 5)
    assert sparse.table.shape == (2, 32)
    assert sparse.max_pool == 512
    assert sparse.seen.tolist() == [132, 5]
    assert sparse.kv_slots.tolist() == [511, 1152, 1153, 1154, 1155, 0, 1, 2, 3, 4]
    assert verify.slots.tolist() == [3, 0] and verify.track.tolist() == [5, 0]

    rows = pad_decode([Row(0, 132, 5, 0)], 2, 5)
    second = cast(PersistentBatch, Batch([3, 9], 6, 10))
    table_ptr, slots_ptr = sparse.table.data_ptr(), verify.slots.data_ptr()
    assert device_metadata(SparseStep.prepare(second, rows, 128, 8, 256, cpu, context=4096), cpu, sparse) is sparse
    assert device_metadata(Verify.prepare(second, rows, cpu, 5), cpu, verify) is verify
    assert (sparse.table.data_ptr(), verify.slots.data_ptr()) == (table_ptr, slots_ptr)
    assert sparse.table[0, :3].tolist() == [4, 10, 0]
    assert sparse.kv_slots.tolist() == [1284, 1285, 1286, 1287, 1288, 0, 1, 2, 3, 4]
    assert verify.slots.tolist() == [7, 0] and verify.track.tolist() == [11, 0]


def test_prefill_index_shards_keep_request_boundaries_and_prefix_lengths() -> None:
    cpu = torch.device("cpu")
    rows = [Row(4, 1021, 7, 0), Row(7, 7, 10, 7), Row(11, 494, 2, 17)]
    middle = IndexShard.prepare(rows, 4, rank=1, size=4, device=cpu)
    assert middle.tokens == slice(5, 10) and middle.requests == slice(0, 2)
    assert middle.ends.tolist() == [2, 5]
    assert middle.pooled.tolist() == [257, 2] and middle.max_pool == 257

    tail = IndexShard.prepare(rows, 4, rank=3, size=4, device=cpu)
    assert tail.width == 5 and tail.tokens == slice(15, 20) and tail.requests == slice(1, 3)
    assert tail.ends.tolist() == [2, 4]
    assert tail.pooled.tolist() == [4, 124]

    empty = IndexShard.prepare(rows, 4, rank=7, size=8, device=cpu)
    assert len(empty.ends) == len(empty.pooled) == empty.max_pool == 0


def test_refit_refreshes_packed_projections_at_their_captured_addresses() -> None:
    layers = GlmLayers.__new__(GlmLayers)
    layers._projection_weights = {}
    config = cast(Config, SimpleNamespace(hidden=3, vocab=4, attention=(), mlp=()))
    weights = Weights(config, 1, 0, refresh_projections=layers.refresh_projections)
    tensors = {decl.name: torch.zeros(decl.shape) for decl in weights.decls}
    left, right = tensors[EMBED], tensors[HEAD][:2]
    packed = layers._packed_projection(left, right)
    address = packed.data_ptr()

    for value in (1, 4, 1):  # Refit abort reloads the old contents through the same loader.
        incoming = [(name if name == HEAD else PREFIX + name, torch.full_like(tensor, value + i))
                    for i, (name, tensor) in enumerate(tensors.items())]
        assert weights.load_weights(incoming, tensors) == set(tensors)
        assert layers._packed_projection(left, right).data_ptr() == address
        torch.testing.assert_close(packed, torch.cat((left, right)))
