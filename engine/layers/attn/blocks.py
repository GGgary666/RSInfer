"""Block-table metadata for eager attention over physical KV pages."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate
from typing import Any

import torch

from core.contracts.plan import PersistentBatch, PlanKey
from core.contracts.records import BatchPlan
from layers.attn.paged import on_device
from layers.solutions import Kernels

BLOCK_PAGED = "attn.block_paged@1"


@dataclass(frozen=True)
class Metadata:
    phase: str
    blocks: torch.Tensor
    slots: torch.Tensor
    ends: list[int]
    lengths: list[int]


class BlockAttnBackend:
    def __init__(self, kernels: Kernels, kv: Any, page_size: int) -> None:
        self.kernels, self.kv, self.page_size = kernels, kv, page_size
        self.metadata: Metadata | None = None

    def prepare(self, batch: PersistentBatch, plan: BatchPlan, key: PlanKey) -> None:
        if key.bucket:
            raise RuntimeError("block-table attention requires an eager plan")
        device = batch.binding.buffers()["req_to_token"].device
        page_size = self.page_size
        pages = batch.ids("kv.paged")
        lengths: list[int] = []
        query_sizes: list[int] = []
        table: list[list[int]] = []
        slots: list[int] = []
        for row, start, count in zip(plan.rows.values(), plan.positions.values(), plan.num_tokens.values(), strict=True):
            start, count = int(start), int(count)
            length = start + count
            n_pages = -(-length // page_size)
            ids = pages[int(row)][:n_pages]
            if len(ids) != n_pages:
                raise ValueError(f"row {row} has {len(ids)} pages for {length} tokens")
            # Storage page 0 is the padding page. Scheduler item 0 starts at storage page 1.
            blocks = [i + 1 for i in ids]
            table.append(blocks)
            slots.extend(blocks[p // page_size] * page_size + p % page_size for p in range(start, length))
            lengths.append(length)
            query_sizes.append(count)
        width = max(map(len, table))
        block_table = on_device(torch.tensor([b + [0] * (width - len(b)) for b in table], dtype=torch.int32), device)
        self.metadata = Metadata(key.phase, block_table, on_device(torch.tensor(slots, dtype=torch.int64), device),
                                 list(accumulate(query_sizes)), lengths)

    def forward(self, layer: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int,
                num_kv_heads: int, head_dim: int, scaling: float) -> torch.Tensor:
        m = self.metadata
        if m is None:
            raise RuntimeError("block-table attention ran before prepare")
        cache = self.kv.bind(layer).buffers()
        keys, values = cache["k"], cache["v"]
        self.kernels(m.phase, "kv.store@1")(keys, values, m.slots, k.reshape(-1, num_kv_heads, head_dim),
                                          v.reshape(-1, num_kv_heads, head_dim))
        out = self.kernels(m.phase, BLOCK_PAGED)(
            q.reshape(-1, num_heads, head_dim), keys.view(-1, self.page_size, num_kv_heads, head_dim),
            values.view(-1, self.page_size, num_kv_heads, head_dim), m.blocks, m.ends, m.lengths, scaling,
            m.phase != "decode")
        return out.reshape(-1, num_heads * head_dim)
