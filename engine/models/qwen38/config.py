# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""The fields of SGLang's ``Qwen3_5TextConfig`` (srt/configs/qwen3_5.py, qwen3_next.py, e838360;
LICENSE.sglang) that the dense text model reads, from an HF ``config.json``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import msgspec


class Qwen3_5TextConfig(msgspec.Struct, frozen=True):
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    linear_num_value_heads: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    full_attention_interval: int
    rope_theta: float
    partial_rotary_factor: float
    max_position_embeddings: int
    attn_output_gate: bool
    tie_word_embeddings: bool
    hidden_act: str

    @staticmethod
    def from_hf(raw: dict[str, Any]) -> Qwen3_5TextConfig:
        """A text config, or the ``text_config`` of a ``Qwen3_5ForConditionalGeneration`` one.
        RoPE as ``get_rope_config`` reads it: ``rope_parameters`` (v5) or ``rope_theta`` (v4)."""
        d = raw.get("text_config", raw)
        rope = d.get("rope_parameters") or d.get("rope_scaling") or {}
        return Qwen3_5TextConfig(
            vocab_size=d["vocab_size"], hidden_size=d["hidden_size"], intermediate_size=d["intermediate_size"],
            num_hidden_layers=d["num_hidden_layers"], num_attention_heads=d["num_attention_heads"],
            num_key_value_heads=d["num_key_value_heads"],
            head_dim=d.get("head_dim") or d["hidden_size"] // d["num_attention_heads"],
            rms_norm_eps=d.get("rms_norm_eps", 1e-6),
            linear_num_value_heads=d["linear_num_value_heads"], linear_num_key_heads=d["linear_num_key_heads"],
            linear_key_head_dim=d["linear_key_head_dim"], linear_value_head_dim=d["linear_value_head_dim"],
            linear_conv_kernel_dim=d["linear_conv_kernel_dim"],
            full_attention_interval=d.get("full_attention_interval", 4),
            rope_theta=rope.get("rope_theta", d.get("rope_theta", 10000)),
            partial_rotary_factor=d.get("partial_rotary_factor", rope.get("partial_rotary_factor", 1.0)),
            max_position_embeddings=d.get("max_position_embeddings", 8192),
            attn_output_gate=d.get("attn_output_gate", True),
            tie_word_embeddings=d.get("tie_word_embeddings", False),
            hidden_act=d.get("hidden_act", "silu"),
        )

    @property
    def layers_block_type(self) -> list[str]:
        """``Qwen3NextConfig.layers_block_type``: every ``full_attention_interval``-th layer is full
        attention, the others linear attention (GDN)."""
        return ["attention" if (layer + 1) % self.full_attention_interval == 0 else "linear_attention"
                for layer in range(self.num_hidden_layers)]


@dataclass(frozen=True)
class Dims:
    """The per-layer sizes ``Qwen3_5AttentionDecoderLayer`` and ``Qwen3_5GatedDeltaNet`` derive (TP1)."""

    q_size: int
    kv_size: int
    key_dim: int
    value_dim: int
    conv_dim: int
    rotary_dim: int

    @staticmethod
    def of(cfg: Qwen3_5TextConfig) -> Dims:
        key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads
        value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads
        rotary_dim = cfg.head_dim
        if cfg.partial_rotary_factor < 1.0:  # get_rope
            rotary_dim = int(rotary_dim * cfg.partial_rotary_factor)
        return Dims(cfg.num_attention_heads * cfg.head_dim, cfg.num_key_value_heads * cfg.head_dim, key_dim,
                    value_dim, key_dim * 2 + value_dim, rotary_dim)
