# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""``state.gdn_slots``: the conv and temporal states of the GDN layers (``state.gdn.slots@1``).
Adapted from SGLang's ``MambaPool`` (srt/mem_cache/memory_pool.py, e838360; LICENSE.sglang):
conv ``[num_mamba_layers, size + 1, conv_dim, K - 1]`` and temporal
``[num_mamba_layers, size + 1, HV, V, K]``, zero-filled; slot 0 is the padding slot. An item is a
slot: item ``i`` is slot ``i + 1``, the slots SGLang's ``MambaSlotAllocator`` hands out, and slot 0
is the scratch item ``-1``. ``copy`` is ``copy_from`` and ``clear`` is ``clear_slots``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from core.contracts.state import StateDecl

KIND = "gdn.slots"


def on_device(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Host ``values`` on ``device`` without waiting for the work queued there: from pinned memory,
    ``non_blocking``, as SGLang's ``ForwardBatch`` copies its inputs. A blocking copy would wait
    for the step still running, and so would the next step's preparation."""
    if device.type != "cuda":
        return values.to(device)
    return values.pin_memory().to(device, non_blocking=True)


@dataclass(frozen=True)
class LayerState:
    """``Binding`` of one layer: ``conv`` ``[size + 1, conv_dim, K - 1]`` and ``temporal``."""

    generation: int
    conv: torch.Tensor
    temporal: torch.Tensor

    def buffers(self) -> Mapping[str, torch.Tensor]:
        return {"conv": self.conv, "temporal": self.temporal}


@dataclass
class MambaPool:
    """``state.gdn.slots@1``."""

    platform: Any
    size: int
    """Usable slots; 0: as many as the holder gives ``gdn.slots`` (``state.holder@1``), taken again
    at every declaration (as ``state.kv_paged`` does)."""
    holder: Any = None
    mamba_layer_ids: list[int] = field(default_factory=list[int])
    decls: list[StateDecl] = field(default_factory=list[StateDecl])
    conv: torch.Tensor | None = None
    temporal: torch.Tensor | None = None
    allocated: list[Any] = field(default_factory=list[Any])
    generation: int = 0
    sized_from: dict[str, Any] = field(default_factory=dict[str, Any])
    """The dtype keys the model declares the states in. Read here so that changing one reloads this
    unit, which ``core.deps`` refuses in place while it holds state (§5.5.5); what ``auto`` follows
    (the checkpoint's config) the weight store reads."""
    configured: int = field(init=False, default=0)
    """``state.gdn.slots``; 0: the holder's count."""
    spec_steps: int = 0
    """Speculation (doc/specs/mtp.md): the D tokens a verify step keeps a state after, per lane; 0 off."""
    spec_lanes: int = 0
    """Lanes of the intermediate buffers: the rows one step runs at most (``batch.max_rows``)."""
    intermediate: tuple[torch.Tensor, torch.Tensor] | None = None
    """SGLang's ``intermediate_conv_window`` and ``intermediate_ssm``: ``[layers, lanes, D, ...]`` of conv
    and temporal, what a verify step leaves after each of its tokens."""

    def __post_init__(self) -> None:
        self.configured = self.size

    def _count(self) -> int:
        """The slots to hold, as ``state.kv_paged`` counts its pages: the holder's count may change
        while this unit stays (weights of another size)."""
        if self.holder is not None and self.holder.asleep:
            return 1  # asleep (holder.sleep): whatever state.gdn.slots says
        if self.configured:
            return self.configured
        return self.holder.items(KIND) if self.holder is not None else 0

    def declare(self, decls: Sequence[StateDecl]) -> None:
        decls = sorted(decls, key=lambda d: d.layer)
        size = self._count() if decls else self.size
        if decls == self.decls and size == self.size:
            return
        if not size:
            raise RuntimeError(f"{KIND}: no slot count: set state.gdn.slots, or size the states first (state.holder@1)")
        self.size = size
        if any(d.kind != KIND or d.tensors != decls[0].tensors for d in decls):
            raise ValueError(f"{KIND} holds one state shape for every GDN layer")
        (conv_shape, conv_dtype), (temporal_shape, ssm_dtype) = decls[0].tensors["conv"], decls[0].tensors["temporal"]
        self.free()
        # Freed buffers stay in PyTorch's cache, and the next allocation may take part of a freed
        # block: a new pool a little larger than the old one then fits neither there nor in what the
        # device has left. Give them back first (doc/framework-bugs/2026-11-10-gdn-ssm-dtype-rebuild-oom.md).
        torch.cuda.empty_cache()
        n = len(decls)
        conv = self.platform.alloc(0, 0, shape=(n, self.size + 1, *conv_shape), dtype=conv_dtype)
        temporal = self.platform.alloc(0, 0, shape=(n, self.size + 1, *temporal_shape), dtype=ssm_dtype)
        self.allocated = [conv, temporal]
        self.conv, self.temporal = self.platform.tensor(conv), self.platform.tensor(temporal)
        if self.spec_steps:
            lanes = (n, self.spec_lanes, self.spec_steps)
            iconv = self.platform.alloc(0, 0, shape=(*lanes, *conv_shape), dtype=conv_dtype)
            itemporal = self.platform.alloc(0, 0, shape=(*lanes, *temporal_shape), dtype=ssm_dtype)
            self.allocated += [iconv, itemporal]
            self.intermediate = (self.platform.tensor(iconv), self.platform.tensor(itemporal))
        self.mamba_layer_ids, self.decls = [d.layer for d in decls], list(decls)

    def _tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.conv is None or self.temporal is None:
            raise RuntimeError(f"{KIND} has no declarations")
        return self.conv, self.temporal

    def bind(self, layer: int) -> LayerState:
        conv, temporal = self._tensors()
        i = self.mamba_layer_ids.index(layer)
        return LayerState(self.generation, conv[i], temporal[i])

    def capacity(self) -> int:
        return self.size

    def verify_buffers(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``layer``'s intermediate conv windows and states ``[lanes, D, ...]`` for a verify step."""
        if self.intermediate is None:
            raise RuntimeError(f"{KIND} keeps no intermediate states: speculation is off")
        i = self.mamba_layer_ids.index(layer)
        return self.intermediate[0][i], self.intermediate[1][i]

    def verify_pools(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Every layer's intermediate buffers ``[layers, lanes, D, ...]``, for the commit."""
        if self.intermediate is None:
            raise RuntimeError(f"{KIND} keeps no intermediate states: speculation is off")
        return self.intermediate

    def pools(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        """``conv`` and ``temporal`` of every layer, ``[layers, size + 1, ...]``, and the last GDN layer:
        what ``track_mamba_states_all_layers`` copies, once that layer ran (``_track_pools``)."""
        conv, temporal = self._tensors()
        return conv, temporal, self.mamba_layer_ids[-1]

    def scratch(self) -> Sequence[int]:
        return [-1]

    def indices(self, items: Sequence[int]) -> torch.Tensor:
        """The slots of ``items``, on the device of the states (``index_fill_`` wants them there)."""
        conv, _ = self._tensors()
        return on_device(torch.tensor(list(items), dtype=torch.int64) + 1, conv.device)

    def copy(self, src: Sequence[int], dst: Sequence[int]) -> None:
        """``copy_from``: clone conv and temporal state from ``src`` slots into ``dst`` slots."""
        conv, temporal = self._tensors()
        src_indices, dst_indices = self.indices(src), self.indices(dst)
        conv[:, dst_indices] = conv[:, src_indices]
        temporal[:, dst_indices] = temporal[:, src_indices]

    def clear(self, items: Sequence[int]) -> None:
        """``clear_slots``: zero out the state at the given slots."""
        conv, temporal = self._tensors()
        idx64 = self.indices(items)
        conv.index_fill_(1, idx64, 0)
        if temporal.numel() > 0:
            temporal.index_fill_(1, idx64, 0)

    def invalidate(self) -> None:
        """Nothing to do here: a slot is cleared or restored before a request reads it."""

    def free(self) -> None:
        """The buffers go, and so do the declarations: declared again (a rollback declares the old
        ones), the states are allocated again rather than taken as unchanged."""
        for buf in self.allocated:
            self.platform.free(buf)
        self.generation += bool(self.allocated)  # their bindings are stale
        self.allocated, self.conv, self.temporal, self.decls = [], None, None, []
        self.intermediate = None


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        steps = config["spec.num_draft_tokens"] if config["spec.algorithm"] else 0
        self.pool = MambaPool(ctx.platform, config["state.gdn.slots"], holder=ctx.use("state.holder@1"), sized_from={
            "state.gdn.ssm_dtype": config["state.gdn.ssm_dtype"], "state.gdn.conv_dtype": config["state.gdn.conv_dtype"]},
            spec_steps=steps, spec_lanes=config["batch.max_rows"] if steps else 0)
        ctx.effect(lambda: self.pool.free, label="gdn states")
        ctx.provide("state.gdn.slots@1", self.pool)
