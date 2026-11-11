# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""``layers.rope``: SGLang's ``RotaryEmbedding`` (srt/layers/rotary_embedding/base.py, e838360;
LICENSE.sglang): an fp32 cos/sin cache per (rotary dim, base, device), grown past
``max_position_embeddings`` as ``_ensure_cos_sin_cache_length`` does. An attention layer applies
it with the Q/K norm and the split of its output gate, in one call (``attn.qk_norm_rope_gate@1``,
``Qwen3_5AttentionDecoderLayer.forward_prepare_cuda_fused``)."""

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

    def __init__(self, catalog: Any, max_context: int | None = None) -> None:
        self.catalog = catalog
        self.max_context = max_context
        self.caches: dict[tuple[int, float, str], torch.Tensor] = {}

    def cos_sin_cache(self, rotary_dim: int, max_position: int, base: float, device: torch.device,
                      needed_max_pos: int) -> torch.Tensor:
        key = (rotary_dim, base, str(device))
        cache = self.caches.get(key)
        if cache is None or needed_max_pos >= cache.shape[0]:
            length = max(max_position, (needed_max_pos + ROPE_CACHE_ALIGN) // ROPE_CACHE_ALIGN * ROPE_CACHE_ALIGN)
            cache = self.caches[key] = rope_cos_sin_cache(rotary_dim, length, base).to(device)
        return cache

    def qk_norm_rope_gate(self, positions: torch.Tensor, q_gate: torch.Tensor, k: torch.Tensor,
                          q_weight: torch.Tensor, k_weight: torch.Tensor, eps: float, num_q_heads: int,
                          num_kv_heads: int, head_dim: int, rotary_dim: int, max_position: int, base: float,
                          has_gate: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """``q_gate`` holds per head the query, then (``has_gate``) its output gate. Returns the
        normed and rotated q and k, and the gate in q's shape."""
        cache = self.cos_sin_cache(rotary_dim, max_position, base, q_gate.device,
                                   self.max_context - 1 if self.max_context is not None else
                                   int(positions.max()) if positions.numel() else 0)
        shape = {"head_dim": head_dim, "rotary_dim": rotary_dim, "fp32": int(q_gate.dtype == torch.float32)}
        return self.catalog.call("attn.qk_norm_rope_gate@1", shape, q_gate, k, q_weight, k_weight, cache, positions,
                                 eps, num_q_heads, num_kv_heads, head_dim, rotary_dim, has_gate)


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        ctx.provide("layers.rope@1", Rope(ctx.use("kernels.catalog@1"), config["batch.max_context_len"]))
