"""``state.swa_slots``: what a DS 1017 request keeps in one fixed-size slot (``state.swa.slots@1``,
doc/specs/v1-ds1017.md §3): per layer the sliding-window KV (a ring of the last ``window`` tokens'
latents), for the layers that compress KV by more than one the tail of a group still filling, and
for the first Engram layer the last tokens an n-gram reaches back to. DeepInfer keeps the same in
its ``StateBlock`` (deepinfer/worker/cache_engine.py).

The layout follows ``state.gdn_slots``: per layer and tensor one buffer ``[slots + 1, ...]``, item
``i`` at index ``i + 1``, index 0 the padding slot (scratch item ``-1``). Unlike the GDN states, the
layers need not declare the same tensors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from core.contracts.state import StateDecl

KIND = "swa.slots"


def on_device(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Host ``values`` on ``device`` without waiting for the work queued there (pinned, non-blocking)."""
    if device.type != "cuda":
        return values.to(device)
    return values.pin_memory().to(device, non_blocking=True)


@dataclass(frozen=True)
class LayerState:
    """``Binding`` of one layer: its tensors by the names it declared them with."""

    generation: int
    tensors: Mapping[str, torch.Tensor]

    def buffers(self) -> Mapping[str, torch.Tensor]:
        return self.tensors


@dataclass
class SlotPool:
    """``state.swa.slots@1``."""

    platform: Any
    size: int
    """Usable slots; 0: as many as the holder gives ``swa.slots`` (``state.holder@1``), taken again at
    every declaration."""
    holder: Any = None
    decls: list[StateDecl] = field(default_factory=list[StateDecl])
    tensors: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict[int, dict[str, torch.Tensor]])
    allocated: list[Any] = field(default_factory=list[Any])
    generation: int = 0
    configured: int = field(init=False, default=0)
    """``state.swa.slots``; 0: the holder's count."""

    def __post_init__(self) -> None:
        self.configured = self.size

    def _count(self) -> int:
        if self.holder is not None and self.holder.asleep:
            return 1
        if self.configured:
            return self.configured
        return self.holder.items(KIND) if self.holder is not None else 0

    def declare(self, decls: Sequence[StateDecl]) -> None:
        decls = sorted(decls, key=lambda d: d.layer)
        size = self._count() if decls else self.size
        if decls == self.decls and size == self.size:
            return
        if not size:
            raise RuntimeError(f"{KIND}: no slot count: set state.swa.slots, or size the states first (state.holder@1)")
        if any(d.kind != KIND or d.lifetime != "slot" for d in decls):
            raise ValueError(f"{KIND} holds slot states only")
        self.free()
        self.size = size
        torch.cuda.empty_cache()  # as state.gdn_slots: give freed blocks back before allocating
        for d in decls:
            bufs = {name: self.platform.alloc(0, 0, shape=(size + 1, *shape), dtype=dtype)
                    for name, (shape, dtype) in d.tensors.items()}
            self.allocated += list(bufs.values())
            self.tensors[d.layer] = {name: self.platform.tensor(buf) for name, buf in bufs.items()}
        self.decls = list(decls)

    def bind(self, layer: int) -> LayerState:
        return LayerState(self.generation, self.tensors[layer])

    def capacity(self) -> int:
        return self.size

    def scratch(self) -> Sequence[int]:
        return [-1]

    def indices(self, items: Sequence[int]) -> torch.Tensor:
        first = next(iter(self.tensors.values()), None)
        device = next(iter(first.values())).device if first else torch.device("cpu")
        return on_device(torch.tensor(list(items), dtype=torch.int64) + 1, device)

    def copy(self, src: Sequence[int], dst: Sequence[int]) -> None:
        src_indices, dst_indices = self.indices(src), self.indices(dst)
        for tensors in self.tensors.values():
            for buffer in tensors.values():
                buffer[dst_indices] = buffer[src_indices]

    def clear(self, items: Sequence[int]) -> None:
        """Zero the slots: a request that starts from position 0 reads nothing it did not write, but a
        cleared slot keeps what it holds defined."""
        index = self.indices(items)
        for tensors in self.tensors.values():
            for buffer in tensors.values():
                buffer.index_fill_(0, index, 0)

    def invalidate(self) -> None:
        """Nothing to do here: a slot is cleared or restored before a request reads it."""

    def free(self) -> None:
        for buf in self.allocated:
            self.platform.free(buf)
        self.generation += bool(self.allocated)
        self.allocated, self.tensors, self.decls = [], {}, []


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        self.pool = SlotPool(ctx.platform, config["state.swa.slots"], holder=ctx.use("state.holder@1"))
        ctx.effect(lambda: self.pool.free, label="swa states")
        ctx.provide("state.swa.slots@1", self.pool)
