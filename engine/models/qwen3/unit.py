"""Dense Qwen3 with fused QKV and gate/up weights in the RSInfer weight store."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

from core.contracts.model import Assembly, Model, WeightDecl
from models.qwen3.config import KV_KIND, Config

ATTN = "layers.attn@1"
EMBEDDINGS = "model.embed_tokens.weight"
FINAL_NORM = "model.norm.weight"
LM_HEAD = "lm_head.weight"


class Qwen3:
    def __init__(self, cfg: Config, dtype: str, kv_dtype: str, page_size: int, max_context: int,
                 services: Mapping[str, Any]) -> None:
        if not 0 < max_context <= cfg.max_positions:
            raise ValueError(f"batch.max_context_len must be in 1..{cfg.max_positions} for this Qwen3")
        self.cfg, self.max_context = cfg, max_context
        self.q_width = cfg.heads * cfg.head_dim
        self.kv_width = cfg.kv_heads * cfg.head_dim
        self.weights = services["state.weights@1"]
        self.kv = services["state.kv.paged@1"]
        self.holder = services["state.holder@1"]
        self.attn = services[ATTN]
        self.linear = services["layers.linear@1"]
        self.embedding = services["layers.embedding@1"]
        self.norm = services["layers.norm@1"]
        self.rope = services["layers.rope@1"]
        self.act = services["layers.activation@1"]
        self.head_name = EMBEDDINGS if cfg.tie_embeddings else LM_HEAD
        self.parts: dict[str, tuple[str, int, int]] = {}
        self.assembly = Assembly([ATTN] * cfg.layers, cfg.states(dtype, page_size, kv_dtype), self.declarations(dtype))
        self.backends = {ATTN: self.attn}
        self.stores = {KV_KIND: self.kv}

    def declarations(self, dtype: str) -> list[WeightDecl]:
        cfg = self.cfg
        decls: list[WeightDecl] = []

        def weight(name: str, *shape: int) -> None:
            decls.append(WeightDecl(name, shape, dtype))

        def packed(dst: str, src: tuple[tuple[str, int], ...], suffix: str, *tail: int) -> None:
            name = dst + suffix
            weight(name, sum(width for _, width in src), *tail)
            offset = 0
            for base, width in src:
                self.parts[base + suffix] = (name, offset, width)
                offset += width

        weight(EMBEDDINGS, cfg.vocab, cfg.hidden)
        weight(FINAL_NORM, cfg.hidden)
        if not cfg.tie_embeddings:
            weight(LM_HEAD, cfg.vocab, cfg.hidden)
        for layer in range(cfg.layers):
            p = f"model.layers.{layer}."
            a = p + "self_attn."
            qkv = ((a + "q_proj", self.q_width), (a + "k_proj", self.kv_width), (a + "v_proj", self.kv_width))
            packed(a + "qkv_proj", qkv, ".weight", cfg.hidden)
            weight(a + "o_proj.weight", cfg.hidden, self.q_width)
            if cfg.attention_bias:
                packed(a + "qkv_proj", qkv, ".bias")
                weight(a + "o_proj.bias", cfg.hidden)
            weight(a + "q_norm.weight", cfg.head_dim)
            weight(a + "k_norm.weight", cfg.head_dim)
            weight(p + "input_layernorm.weight", cfg.hidden)
            weight(p + "post_attention_layernorm.weight", cfg.hidden)
            m = p + "mlp."
            packed(m + "gate_up_proj", ((m + "gate_proj", cfg.intermediate), (m + "up_proj", cfg.intermediate)),
                   ".weight", cfg.hidden)
            weight(m + "down_proj.weight", cfg.hidden, cfg.intermediate)
        return decls

    def wants_weight(self, name: str) -> bool:
        return name in self.parts or name in self.checkpoint_weights

    def load_weights(self, incoming: Iterable[tuple[str, torch.Tensor]], tensors: Mapping[str, torch.Tensor]
                     ) -> set[str]:
        loaded: set[str] = set()
        shards: set[str] = set()
        for name, tensor in incoming:
            if name in self.parts:
                dst, offset, size = self.parts[name]
                target = tensors[dst].narrow(0, offset, size)
                shards.add(name)
            elif name in self.checkpoint_weights:
                target = tensors[name]
                loaded.add(name)
            else:
                continue
            if target.shape != tensor.shape:
                raise ValueError(f"{name}: expected {tuple(target.shape)}, got {tuple(tensor.shape)}")
            target.copy_(tensor)
        return loaded | (self.packed_weights if self.parts.keys() <= shards else set())

    def start(self) -> None:
        self.packed_weights = {dst for dst, _, _ in self.parts.values()}
        self.checkpoint_weights = {w.name for w in self.assembly.weights} - self.packed_weights
        self.weights.declare(self.assembly.weights)
        tensors: Mapping[str, torch.Tensor] = self.weights.bind("").buffers()
        self.holder.size(self.assembly.states, sum(t.numel() * t.element_size() for t in tensors.values()))
        self.kv.declare(self.assembly.states)
        self.weights.load()
        self.top = {name: tensors[name] for name in (EMBEDDINGS, FINAL_NORM, self.head_name)}
        self.layers: list[Mapping[str, torch.Tensor]] = [
            self.weights.bind(f"model.layers.{layer}.").buffers() for layer in range(self.cfg.layers)]

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, last_index: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        hidden = self.embedding.embed(input_ids, self.top[EMBEDDINGS])
        cos, sin = self.rope.step(positions, cfg.head_dim, self.max_context, cfg.rope_theta, hidden.dtype)
        residual = None
        for layer, w in enumerate(self.layers):
            if residual is None:
                residual = hidden
                hidden = self.norm.rms(hidden, w["input_layernorm.weight"], cfg.eps)
            else:
                hidden, residual = self.norm.rms(hidden, w["input_layernorm.weight"], cfg.eps, residual)
            hidden = self.self_attention(layer, hidden, cos, sin, w)
            hidden, residual = self.norm.rms(hidden, w["post_attention_layernorm.weight"], cfg.eps, residual)
            gate_up = self.linear.forward(hidden, w["mlp.gate_up_proj.weight"])
            hidden = self.linear.forward(self.act.silu_and_mul(gate_up), w["mlp.down_proj.weight"])
        assert residual is not None
        if not last_index.numel():
            return hidden.new_empty((0, cfg.vocab))
        # Only the sampled rows need the final norm and vocabulary projection.
        hidden, _ = self.norm.rms(hidden.index_select(0, last_index), self.top[FINAL_NORM], cfg.eps,
                                  residual.index_select(0, last_index))
        return self.embedding.logits(hidden, self.top[self.head_name])

    def self_attention(self, layer: int, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                       w: Mapping[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg
        qkv = self.linear.forward(hidden, w["self_attn.qkv_proj.weight"], w.get("self_attn.qkv_proj.bias"))
        q, k, v = qkv.split((self.q_width, self.kv_width, self.kv_width), dim=-1)
        q = self.norm.rms(q.reshape(-1, cfg.heads, cfg.head_dim), w["self_attn.q_norm.weight"], cfg.eps)
        k = self.norm.rms(k.reshape(-1, cfg.kv_heads, cfg.head_dim), w["self_attn.k_norm.weight"], cfg.eps)
        q, k = self.rope.qk(q, k, cos, sin)
        out = self.attn.forward(layer, q, k, v, cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.head_dim ** -0.5)
        return self.linear.forward(out, w["self_attn.o_proj.weight"], w.get("self_attn.o_proj.bias"))


class Unit:
    SERVICES = ("state.weights@1", "state.holder@1", "state.kv.paged@1", "layers.linear@1", "layers.embedding@1",
                "layers.norm@1", "layers.rope@1", "layers.activation@1", ATTN)

    def __init__(self, ctx: Any, config: Any) -> None:
        raw = json.loads((Path(config["model.path"]) / "config.json").read_text()) if config["model.path"] else config[
            "model.hf_config"]
        model: Model = Qwen3(Config.from_hf(raw), config["model.dtype"], config["state.kv.dtype"],
                             config["state.kv.page_size"], config["batch.max_context_len"],
                             {name: ctx.use(name) for name in self.SERVICES})
        ctx.register("state.weights@1", model.load_weights, label="load_weights")
        model.start()
        ctx.provide("model@1", model)
