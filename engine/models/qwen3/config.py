"""Hugging Face configuration and the dense model's KV layout."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from core.contracts.state import StateDecl

KV_KIND = "kv.paged"


@dataclass(frozen=True)
class Config:
    hidden: int
    intermediate: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    max_positions: int
    eps: float
    rope_theta: float
    tie_embeddings: bool
    attention_bias: bool

    @classmethod
    def from_hf(cls, raw: Mapping[str, Any]) -> Config:
        if raw.get("use_sliding_window") or raw.get("sliding_window"):
            raise ValueError("models.qwen3 requires full causal attention")
        if raw.get("hidden_act", "silu") != "silu":
            raise ValueError("models.qwen3 requires hidden_act=silu")
        rope = raw.get("rope_parameters") or raw.get("rope_scaling") or {}
        if (rope.get("rope_type", rope.get("type", "default")) != "default"
                or rope.get("partial_rotary_factor", raw.get("partial_rotary_factor", 1.0)) != 1.0):
            raise ValueError("models.qwen3 requires unscaled rotary embeddings")
        hidden, heads = int(raw["hidden_size"]), int(raw["num_attention_heads"])
        if heads < 1:
            raise ValueError("models.qwen3 requires at least one attention head")
        cfg = cls(hidden, int(raw["intermediate_size"]), int(raw["num_hidden_layers"]), heads,
                  int(raw.get("num_key_value_heads", heads)), int(raw.get("head_dim", hidden // heads)),
                  int(raw["vocab_size"]), int(raw["max_position_embeddings"]),
                  float(raw.get("rms_norm_eps", 1e-6)), float(rope.get("rope_theta", raw.get("rope_theta", 1e6))),
                  bool(raw.get("tie_word_embeddings", False)), bool(raw.get("attention_bias", False)))
        if min(cfg.hidden, cfg.intermediate, cfg.layers, cfg.heads, cfg.kv_heads, cfg.head_dim,
               cfg.vocab, cfg.max_positions) <= 0 or cfg.heads % cfg.kv_heads or cfg.head_dim % 2:
            raise ValueError("models.qwen3 requires positive dimensions, grouped heads and an even head_dim")
        return cfg

    def states(self, dtype: str, page_size: int, kv_dtype: str) -> list[StateDecl]:
        dtypes = {"auto": dtype, "bf16": "bf16", "bfloat16": "bf16", "f16": "f16",
                  "float16": "f16", "f32": "f32", "float32": "f32"}
        if dtype not in ("bf16", "f16", "f32") or kv_dtype not in dtypes or page_size < 1:
            raise ValueError("models.qwen3 requires a positive page size and bf16, f16 or f32 tensors")
        kv = ((page_size, self.kv_heads, self.head_dim), dtypes[kv_dtype])
        return [StateDecl(KV_KIND, layer, {"k": kv, "v": kv}, "request", tokens_per_item=page_size)
                for layer in range(self.layers)]
