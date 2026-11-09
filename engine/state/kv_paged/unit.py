# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""``state.kv_paged``: paged KV of the full-attention layers (``state.kv.paged@1``). Adapted from
SGLang's ``MHATokenToKVPool`` (srt/mem_cache/memory_pool.py, e838360; LICENSE.sglang): per layer
one K and one V buffer ``[size + page_size, kv_heads, head_dim]`` indexed by token slot, the page of
slot 0 absorbing padded-token writes. An item is a page: item ``i`` is page ``i + 1``, the pages
SGLang's ``PagedTokenToKVPoolAllocator`` hands out, so its slots are ``(i + 1) * page_size + j``
and page 0 is the scratch item ``-1``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from core.contracts.state import StateDecl

KIND = "kv.paged"


def on_device(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Host ``values`` on ``device`` without waiting for the work queued there: from pinned memory,
    ``non_blocking``, as SGLang's ``ForwardBatch`` copies its inputs. A blocking copy would wait
    for the step still running, and so would the next step's preparation."""
    if device.type != "cuda":
        return values.to(device)
    return values.pin_memory().to(device, non_blocking=True)


@dataclass(frozen=True)
class LayerBuffers:
    """``Binding`` of one layer: ``k`` and ``v``."""

    generation: int
    k: torch.Tensor
    v: torch.Tensor

    def buffers(self) -> Mapping[str, torch.Tensor]:
        return {"k": self.k, "v": self.v}


@dataclass
class KVPool:
    """``state.kv.paged@1``."""

    platform: Any
    page_size: int
    pages: int
    """Usable pages; 0: as many as the holder gave ``kv.paged`` (``state.holder@1``)."""
    holder: Any = None
    decls: dict[int, StateDecl] = field(default_factory=dict[int, StateDecl])
    k_buffer: dict[int, torch.Tensor] = field(default_factory=dict[int, torch.Tensor])
    v_buffer: dict[int, torch.Tensor] = field(default_factory=dict[int, torch.Tensor])
    allocated: dict[int, list[Any]] = field(default_factory=dict[int, list[Any]])
    generation: int = 0
    sized_from: dict[str, Any] = field(default_factory=dict[str, Any])
    """The dtype key the model declares the KV in. Read here so that changing it reloads this
    unit, which ``core.deps`` refuses in place while it holds state (§5.5.5); what ``auto`` follows
    (``model.dtype``) the weight store reads."""

    @property
    def size(self) -> int:
        """Token slots requests can use."""
        return self.pages * self.page_size

    def declare(self, decls: Sequence[StateDecl]) -> None:
        if not self.pages and decls:
            self.pages = self.holder.items(KIND) if self.holder is not None else 0
            if not self.pages:
                raise RuntimeError(f"{KIND}: no page count: set state.kv.pages, or size the states first (state.holder@1)")
        for d in decls:
            if d.kind != KIND or d.tokens_per_item != self.page_size:
                raise ValueError(f"{KIND} holds pages of {self.page_size} tokens, not {d.kind} {d.tokens_per_item}")
            if self.decls.get(d.layer) == d:
                continue
            (k_shape, k_dtype), (v_shape, v_dtype) = d.tensors["k"], d.tensors["v"]
            # The padded page (slot 0's page) absorbs dummy padded-token writes.
            total_slots = self.size + self.page_size
            for buf in self.allocated.pop(d.layer, []):
                self.platform.free(buf)
            k = self.platform.alloc(0, 0, shape=(total_slots, *k_shape[1:]), dtype=k_dtype)
            v = self.platform.alloc(0, 0, shape=(total_slots, *v_shape[1:]), dtype=v_dtype)
            self.allocated[d.layer] = [k, v]
            self.generation += d.layer in self.decls
            self.k_buffer[d.layer], self.v_buffer[d.layer] = self.platform.tensor(k), self.platform.tensor(v)
            self.decls[d.layer] = d

    def bind(self, layer: int) -> LayerBuffers:
        return LayerBuffers(self.generation, self.k_buffer[layer], self.v_buffer[layer])

    def capacity(self) -> int:
        return self.pages

    def scratch(self) -> Sequence[int]:
        return [-1]

    def slots(self, items: Sequence[int]) -> torch.Tensor:
        """The token slots of ``items``, in order, on the device of the buffers."""
        device = next(iter(self.k_buffer.values())).device if self.k_buffer else torch.device("cpu")
        return on_device(((torch.tensor(list(items), dtype=torch.int64)[:, None] + 1) * self.page_size
                          + torch.arange(self.page_size)).flatten(), device)

    def copy(self, src: Sequence[int], dst: Sequence[int]) -> None:
        """``move_kv_cache`` by page: every layer's ``buffer[dst slots] = buffer[src slots]``."""
        src_loc, tgt_loc = self.slots(src), self.slots(dst)
        for layer in self.k_buffer:
            self.k_buffer[layer][tgt_loc] = self.k_buffer[layer][src_loc]
            self.v_buffer[layer][tgt_loc] = self.v_buffer[layer][src_loc]

    def clear(self, items: Sequence[int]) -> None:
        """Nothing: attention reads only the slots a request wrote (SGLang never zeroes KV)."""

    def invalidate(self) -> None:
        """Nothing to do here: the ledger drops its prefix index, so no request reads old contents."""

    def free(self) -> None:
        for bufs in self.allocated.values():
            for buf in bufs:
                self.platform.free(buf)
        self.allocated.clear()


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        self.pool = KVPool(ctx.platform, config["state.kv.page_size"], config["state.kv.pages"],
                           ctx.use("state.holder@1"), sized_from={"state.kv.dtype": config["state.kv.dtype"]})
        ctx.effect(lambda: self.pool.free, label="kv buffers")
        ctx.provide("state.kv.paged@1", self.pool)
