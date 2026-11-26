"""GLM-5.3-Flash text configuration and its RSInfer cache layout."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.contracts.state import StateDecl

KV = "kv.paged"
KDA = "gdn.slots"
LINEAR = "linear_attention"
SPARSE = "deepseek_sparse_attention"
CHUNK = 64


@dataclass(frozen=True)
class Config:
    hidden: int
    vocab: int
    max_context: int
    eps: float
    attention: tuple[str, ...]
    mlp: tuple[str, ...]
    intermediate: int
    experts: int
    expert_topk: int
    expert_width: int
    shared_width: int
    routed_scale: float
    swiglu_limit: float
    streams: int
    sinkhorn_iters: int
    hc_eps: float
    post_mult: float
    kda_heads: int
    kda_dim: int
    conv_width: int
    gate_bound: float
    mla_heads: int
    q_dim: int
    v_dim: int
    q_rank: int
    kv_rank: int
    index_heads: int
    index_dim: int
    pool: int
    sparse_topk: int
    draft_layers: int

    @classmethod
    def load(cls, path: str, fallback: dict[str, Any]) -> Config:
        return cls.from_hf(json.loads((Path(path) / "config.json").read_text()) if path else fallback)

    @classmethod
    def from_hf(cls, raw: dict[str, Any]) -> Config:
        cfg = raw.get("text_config", raw)
        if (cfg.get("model_type") != "glm5_next_text" or not cfg.get("mhc") or not cfg.get("mla_use_nope")
                or cfg["qk_rope_head_dim"] != 0 or cfg.get("attention_bias")
                or cfg["hidden_act"] != "silu" or cfg.get("tie_word_embeddings")):
            raise ValueError("GLM-5.3-Flash requires the untied mHC/NoPE text checkpoint")
        if (cfg["scoring_func"], cfg["topk_method"], cfg["n_group"], cfg["topk_group"], cfg["norm_topk_prob"]) != (
                "sigmoid", "noaux_tc", 1, 1, True):
            raise ValueError("GLM-5.3-Flash requires the normalized sigmoid MoE router")
        if not (cfg["index_kpool_compress"] and cfg["index_kpool_always_select_tail"]):
            raise ValueError("GLM-5.3-Flash requires compressed KPool with the causal tail")
        if any(kind != "full" for kind in cfg.get("indexer_types", ())):
            raise ValueError("GLM-5.3-Flash requires an independent indexer per sparse layer")

        layers = int(cfg["num_hidden_layers"])
        attention = tuple(str(kind) for kind in cfg["layer_types"][:layers])
        mlp = tuple(str(kind) for kind in cfg["mlp_layer_types"][:layers])
        if len(attention) != layers or not set(attention) <= {LINEAR, SPARSE}:
            raise ValueError("invalid GLM-5.3-Flash attention layers")
        if len(mlp) != layers or not set(mlp) <= {"dense", "sparse"}:
            raise ValueError("invalid GLM-5.3-Flash MLP layers")
        kda = cfg["linear_attn_config"]
        bound = float(kda["gate_lower_bound"])
        if kda["head_dim"] != 128 or not 2 <= kda["short_conv_kernel_size"] <= 4 or not -5.0 <= bound < 0.0:
            raise ValueError("checkpoint KDA dimensions or gate are outside the CANN operator's range")
        pool, topk = int(cfg["index_kpool"]), int(cfg["index_topk"])
        if pool <= 0 or topk <= 0 or topk % pool:
            raise ValueError("KPool must divide the positive sparse top-k")
        intermediate, expert_width = int(cfg["intermediate_size"]), int(cfg["moe_intermediate_size"])
        return cls(
            hidden=int(cfg["hidden_size"]), vocab=int(cfg["vocab_size"]), max_context=int(cfg["max_position_embeddings"]),
            eps=float(cfg["rms_norm_eps"]), attention=attention, mlp=mlp, intermediate=intermediate,
            experts=int(cfg["n_routed_experts"]), expert_topk=int(cfg["num_experts_per_tok"]), expert_width=expert_width,
            shared_width=int(cfg["n_shared_experts"]) * expert_width, routed_scale=float(cfg["routed_scaling_factor"]),
            swiglu_limit=float(cfg["swiglu_limit"]), streams=int(cfg.get("mhc_num_residual_streams") or cfg["hc_mult"]),
            sinkhorn_iters=int(cfg.get("mhc_sinkhorn_iterations") or cfg["hc_sinkhorn_iters"]),
            hc_eps=float(cfg["hc_eps"]), post_mult=float(cfg.get("mhc_post_mult_value", 2.0)),
            kda_heads=int(kda["num_heads"]), kda_dim=int(kda["head_dim"]),
            conv_width=int(kda["short_conv_kernel_size"]), gate_bound=bound,
            mla_heads=int(cfg["num_attention_heads"]), q_dim=int(cfg["qk_nope_head_dim"]),
            v_dim=int(cfg["v_head_dim"]), q_rank=int(cfg["q_lora_rank"]), kv_rank=int(cfg["kv_lora_rank"]),
            index_heads=int(cfg["index_n_heads"]), index_dim=int(cfg["index_head_dim"]), pool=pool, sparse_topk=topk,
            draft_layers=int(cfg.get("num_nextn_predict_layers", 0)),
        )

    def validate_mesh(self, size: int) -> None:
        widths = (self.kda_heads, self.mla_heads, self.intermediate, self.experts, self.shared_width)
        if any(width % size for width in widths):
            raise ValueError(f"GLM-5.3-Flash heads, experts and MLP widths must divide across {size} ranks")

    def tail_capacity(self, verify: int) -> int:
        if not verify:
            return self.pool
        return 1 << (self.pool + verify - 2).bit_length()

    def states(self, world: int, page: int, verify: int = 0) -> list[StateDecl]:
        self.validate_mesh(world)
        if page < CHUNK or page % CHUNK or page % self.pool:
            raise ValueError(f"GLM-5.3-Flash pages must be multiples of {CHUNK} and KPool width {self.pool}")
        if verify and (not 2 <= verify <= 8 or self.draft_layers != 1 or self.conv_width != 4):
            raise ValueError("GLM-5.3-Flash MTP requires one draft layer, a width-4 conv, and 2..8 verify tokens")
        tail = self.tail_capacity(verify)
        if page % tail:
            raise ValueError(f"GLM-5.3-Flash KV pages must be a multiple of the {tail}-token KPool tail")
        heads = self.kda_heads // world
        attention = (*self.attention, SPARSE) if verify else self.attention
        return [
            StateDecl(KDA, layer, {
                "conv": ((self.conv_width - 1, 3 * heads * self.kda_dim), "bf16"),
                "temporal": ((heads, self.kda_dim, self.kda_dim), "f32"),
            }, "slot") if kind == LINEAR else StateDecl(KV, layer, {
                "kv": ((page, 1, self.kv_rank), "bf16"),
                "index": ((page // self.pool, 1, self.index_dim), "bf16"),
                "tail": ((1, 2, tail, self.index_dim), "f32"),
            }, "request", page)
            for layer, kind in enumerate(attention)
        ]
