# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""``layers.rope``: SGLang's ``RotaryEmbedding`` (srt/layers/rotary_embedding/base.py, e838360;
LICENSE.sglang): an fp32 cos/sin cache per (rotary dim, base, device), grown past
``max_position_embeddings`` as ``_ensure_cos_sin_cache_length`` does, and ``pos.rope@1``."""

from __future__ import annotations

from typing import Any

import torch

ROPE_CACHE_ALIGN = 1024


def rope_cos_sin_cache(rotary_dim: int, max_position: int, base: float) -> torch.Tensor:
    """``RotaryEmbedding._compute_cos_sin_cache``: fp32 ``[max_position, rotary_dim]``, cos then sin."""
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
    t = torch.arange(0, max_position, dtype=torch.float)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


class Rope:
    """``layers.rope@1``."""

    def __init__(self, catalog: Any) -> None:
        self.catalog = catalog
        self.caches: dict[tuple[int, float, str], torch.Tensor] = {}

    def cos_sin_cache(self, rotary_dim: int, max_position: int, base: float, device: torch.device,
                      needed_max_pos: int) -> torch.Tensor:
        key = (rotary_dim, base, str(device))
        cache = self.caches.get(key)
        if cache is None or needed_max_pos >= cache.shape[0]:
            length = max(max_position, (needed_max_pos + ROPE_CACHE_ALIGN) // ROPE_CACHE_ALIGN * ROPE_CACHE_ALIGN)
            cache = self.caches[key] = rope_cos_sin_cache(rotary_dim, length, base).to(device)
        return cache

    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor, head_size: int,
                rotary_dim: int, max_position: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self.cos_sin_cache(rotary_dim, max_position, base, query.device,
                                   int(positions.max()) if positions.numel() else 0)
        return self.catalog.call("pos.rope@1", {"head_dim": head_size, "rotary_dim": rotary_dim}, positions, query,
                                 key, head_size, rotary_dim, cache)


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        ctx.provide("layers.rope@1", Rope(ctx.use("kernels.catalog@1")))
