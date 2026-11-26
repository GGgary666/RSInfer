# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""``state.kv_paged``: paged KV of the attention layers (``state.kv.paged@1``). Adapted from
SGLang's ``MHATokenToKVPool`` (srt/mem_cache/memory_pool.py, e838360; LICENSE.sglang): per layer
one buffer ``[size + page_size, ...]`` indexed by token slot for each tensor the layer declares, the
page of slot 0 absorbing padded-token writes. An item is a page: item ``i`` is page ``i + 1``, the
pages SGLang's ``PagedTokenToKVPoolAllocator`` hands out, so its slots are ``(i + 1) * page_size + j``
and page 0 is the scratch item ``-1``.

Qwen3.8's full-attention layers declare ``k`` and ``v`` ``[page_size, kv_heads, head_dim]``. DS 1017's
layers that produce compressed KV declare the latent ``kv`` and the indexer's key ``ik`` (2026-11-14):
a latent stands for ``ratio`` tokens, so a page holds ``page_size / ratio`` of them, as DeepInfer's
cache does (a tensor declared ``[entries, ...]`` with ``entries`` dividing ``page_size``): the buffer
is ``[(pages + 1) * entries, ...]``, entry ``e`` of page ``p`` at row ``p * entries + e``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from core.contracts.state import StateDecl
from platforms.torch_eager import release_cached_memory

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
    """``Binding`` of one layer: its tensors by the names it declared them with."""

    generation: int
    tensors: Mapping[str, torch.Tensor]

    @property
    def k(self) -> torch.Tensor:
        return self.tensors["k"]

    @property
    def v(self) -> torch.Tensor:
        return self.tensors["v"]

    def buffers(self) -> Mapping[str, torch.Tensor]:
        return self.tensors


@dataclass
class KVPool:
    """``state.kv.paged@1``."""

    platform: Any
    page_size: int
    pages: int
    """Usable pages; 0: as many as the holder gives ``kv.paged`` (``state.holder@1``), taken again
    at every declaration (``_count``)."""
    holder: Any = None
    decls: dict[int, StateDecl] = field(default_factory=dict[int, StateDecl])
    tensors: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict[int, dict[str, torch.Tensor]])
    """Layer -> name -> its buffer."""
    allocated: dict[int, list[Any]] = field(default_factory=dict[int, list[Any]])
    generation: int = 0
    sized_from: dict[str, Any] = field(default_factory=dict[str, Any])
    """The dtype key the model declares the KV in. Read here so that changing it reloads this
    unit, which ``core.deps`` refuses in place while it holds state (§5.5.5); what ``auto`` follows
    (``model.dtype``) the weight store reads."""
    configured: int = field(init=False, default=0)
    """``state.kv.pages``; 0: the holder's count."""

    def __post_init__(self) -> None:
        self.configured = self.pages

    @property
    def size(self) -> int:
        """Token slots requests can use."""
        return self.pages * self.page_size

    @property
    def k_buffer(self) -> dict[int, torch.Tensor]:
        """Each layer's ``k``, of the layers that declare one."""
        return {layer: t["k"] for layer, t in self.tensors.items() if "k" in t}

    @property
    def v_buffer(self) -> dict[int, torch.Tensor]:
        return {layer: t["v"] for layer, t in self.tensors.items() if "v" in t}

    def _count(self) -> int:
        """The pages to hold. The model sizes the states again each time it allocates its weights
        (``state.holder@1``), and weights of another size (``model.dtype``) leave another budget,
        though this unit is not reloaded: a pool kept at the old count in a larger dtype took more
        than the budget, and a hot ``model.dtype`` change from bf16 to f32 ran out of memory where a
        cold start fits."""
        if self.holder is not None and self.holder.asleep:
            return 1  # asleep (holder.sleep): whatever state.kv.pages says
        if self.configured:
            return self.configured
        return self.holder.items(KIND) if self.holder is not None else 0

    def declare(self, decls: Sequence[StateDecl]) -> None:
        if decls:
            pages = self._count()
            if not pages:
                raise RuntimeError(f"{KIND}: no page count: set state.kv.pages, or size the states first (state.holder@1)")
            if pages != self.pages:
                for layer in list(self.allocated):  # another count: every layer is allocated again
                    self._free_layer(layer)
            self.pages = pages
        changed: list[StateDecl] = []
        for d in decls:
            if d.kind != KIND or d.tokens_per_item != self.page_size:
                raise ValueError(f"{KIND} holds pages of {self.page_size} tokens, not {d.kind} {d.tokens_per_item}")
            for name, (shape, _) in d.tensors.items():
                if not shape or self.page_size % shape[0]:
                    raise ValueError(f"{KIND}: layer {d.layer}'s {name} has {shape[:1]} entries a page, which does not "
                                     f"divide a page of {self.page_size} tokens")
            if self.decls.get(d.layer) != d:
                changed.append(d)
        for d in changed:
            self._free_layer(d.layer)
        if changed:
            # As in state.gdn_slots: give the freed buffers (these, or those of a store the same
            # change rebuilds) back before allocating, or these allocations may split them.
            release_cached_memory(self.platform)
        for d in changed:
            # The padded page (slot 0's page) absorbs dummy padded-token writes.
            bufs = {name: self.platform.alloc(0, 0, shape=((self.pages + 1) * shape[0], *shape[1:]), dtype=dtype)
                    for name, (shape, dtype) in d.tensors.items()}
            self.allocated[d.layer] = list(bufs.values())
            self.tensors[d.layer] = {name: self.platform.tensor(buf) for name, buf in bufs.items()}
            self.decls[d.layer] = d

    def bind(self, layer: int) -> LayerBuffers:
        return LayerBuffers(self.generation, self.tensors[layer])

    def capacity(self) -> int:
        return self.pages

    def scratch(self) -> Sequence[int]:
        return [-1]

    def slots(self, items: Sequence[int], entries: int = 0) -> torch.Tensor:
        """The rows of ``items``, in order, on the device of the buffers: their token slots, or with
        ``entries`` the rows of a tensor holding that many entries a page."""
        per = entries or self.page_size
        first = next(iter(self.tensors.values()), None)
        device = next(iter(first.values())).device if first else torch.device("cpu")
        return on_device(((torch.tensor(list(items), dtype=torch.int64)[:, None] + 1) * per
                          + torch.arange(per)).flatten(), device)

    def copy(self, src: Sequence[int], dst: Sequence[int]) -> None:
        """``move_kv_cache`` by page: every layer's ``buffer[dst rows] = buffer[src rows]``."""
        for layer, tensors in self.tensors.items():
            for name, buffer in tensors.items():
                per = self.decls[layer].tensors[name][0][0]
                buffer[self.slots(dst, per)] = buffer[self.slots(src, per)]

    def clear(self, items: Sequence[int]) -> None:
        """Nothing: attention reads only the slots a request wrote (SGLang never zeroes KV)."""

    def invalidate(self) -> None:
        """Nothing to do here: the ledger drops its prefix index, so no request reads old contents."""

    def _free_layer(self, layer: int) -> None:
        """Its buffers go, and so does its declaration: declared again (a rollback declares the
        old one), the layer is allocated again rather than taken as unchanged."""
        bufs = self.allocated.pop(layer, [])
        for buf in bufs:
            self.platform.free(buf)
        self.generation += bool(bufs)  # its bindings are stale
        self.tensors.pop(layer, None)
        self.decls.pop(layer, None)

    def free(self) -> None:
        for layer in list(self.allocated):
            self._free_layer(layer)


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        self.pool = KVPool(ctx.platform, config["state.kv.page_size"], config["state.kv.pages"],
                           ctx.use("state.holder@1"), sized_from={"state.kv.dtype": config["state.kv.dtype"]})
        ctx.effect(lambda: self.pool.free, label="kv buffers")
        ctx.provide("state.kv.paged@1", self.pool)
