# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""``models.qwen38``: Qwen3.8 dense text, ``model@1``. Adapted from SGLang's
``Qwen3_5ForCausalLM`` (srt/models/qwen3_5.py, e838360; LICENSE.sglang) for TP1, without
speculation, LoRA, quantization or fused kernels.

What moved: the parameters are declared to ``state.weights@1`` (``params``, SGLang's names, with
its fused ``qkv_proj``, ``gate_up_proj``, ``in_proj_qkvz``, ``in_proj_ba``) instead of living in
``nn.Module``s; ``load_weights`` is SGLang's, registered with the store; each op goes through a
layer unit (``layers.*``), which runs the kernel the catalog selects; ``RadixAttention`` and
``RadixLinearAttention`` are the backends ``layers.attn`` and ``layers.gdn``, prepared for the step
by the executor. The residual stream is ``GemmaRMSNorm``'s fused add (``forward_native``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

import torch

from core.contracts.model import Assembly, WeightDecl
from core.contracts.state import StateDecl
from models.qwen38.config import Dims, Qwen3_5TextConfig

# (param_name, shard_name, shard_id)
STACKED_PARAMS_MAPPING: tuple[tuple[str, str, str | int | tuple[int, ...]], ...] = (
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
    # GDN
    ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
    ("in_proj_qkvz.", "in_proj_z.", 3),
    ("in_proj_ba.", "in_proj_b.", 0),
    ("in_proj_ba.", "in_proj_a.", 1),
)


def params(cfg: Qwen3_5TextConfig, dtype: str) -> list[WeightDecl]:
    """The parameters of SGLang's ``Qwen3_5ForCausalLM``, by its names. ``A_log`` is fp32 there."""
    d = Dims.of(cfg)
    h = cfg.hidden_size

    def w(name: str, *shape: int, fmt: str = dtype) -> WeightDecl:
        return WeightDecl(name, shape, fmt)

    out = [w("model.embed_tokens.weight", cfg.vocab_size, h)]
    for i, kind in enumerate(cfg.layers_block_type):
        p = f"model.layers.{i}."
        if kind == "attention":
            q_heads = cfg.num_attention_heads * (1 + cfg.attn_output_gate)
            out += [w(p + "qkv_proj.weight", (q_heads + 2 * cfg.num_key_value_heads) * cfg.head_dim, h),
                    w(p + "o_proj.weight", h, d.q_size),
                    w(p + "q_norm.weight", cfg.head_dim), w(p + "k_norm.weight", cfg.head_dim)]
        else:
            nv = cfg.linear_num_value_heads
            out += [w(p + "linear_attn.in_proj_qkvz.weight", 2 * d.key_dim + 2 * d.value_dim, h),
                    w(p + "linear_attn.in_proj_ba.weight", 2 * nv, h),
                    w(p + "linear_attn.conv1d.weight", d.conv_dim, 1, cfg.linear_conv_kernel_dim),
                    w(p + "linear_attn.dt_bias", nv), w(p + "linear_attn.A_log", nv, fmt="f32"),
                    w(p + "linear_attn.norm.weight", cfg.linear_value_head_dim),
                    w(p + "linear_attn.out_proj.weight", h, d.value_dim)]
        out += [w(p + "mlp.gate_up_proj.weight", 2 * cfg.intermediate_size, h),
                w(p + "mlp.down_proj.weight", h, cfg.intermediate_size),
                w(p + "input_layernorm.weight", h), w(p + "post_attention_layernorm.weight", h)]
    out.append(w("model.norm.weight", h))
    if not cfg.tie_word_embeddings:
        out.append(w("lm_head.weight", cfg.vocab_size, h))
    return out


STATE_DTYPES = MappingProxyType({"bf16": "bf16", "bfloat16": "bf16", "f16": "f16", "float16": "f16", "f32": "f32",
                                  "float32": "f32"})
"""The state dtypes stage 2 keeps, in the platform's names and SGLang's."""


def state_dtype(key: str, value: str, auto: str) -> str:
    """A state dtype key (stage2-contract.md §5.5.3): ``auto`` is ``auto``'s value; anything but bf16,
    f16 and f32 (FP8 and FP4 KV among them) is refused at start."""
    if value == "auto":
        return auto
    if value not in STATE_DTYPES:
        raise ValueError(f"{key} = {value!r}: stage 2 keeps states in bf16, f16 or f32 (or auto)")
    return STATE_DTYPES[value]


def states(cfg: Qwen3_5TextConfig, dtype: str, page_size: int, *, kv_dtype: str | None = None,
           conv_dtype: str | None = None, ssm_dtype: str = "f32") -> list[StateDecl]:
    """KV per full-attention layer (``MHATokenToKVPool``), conv and temporal state per GDN layer
    (``Mamba2StateShape``: conv ``(conv_dim, K - 1)``, temporal ``(HV, V, K)``). KV and conv are in
    ``dtype`` unless given, temporal in ``ssm_dtype`` (``Mamba2StateDType``)."""
    d = Dims.of(cfg)
    out: list[StateDecl] = []
    for i, kind in enumerate(cfg.layers_block_type):
        if kind == "attention":
            kv = ((page_size, cfg.num_key_value_heads, cfg.head_dim), kv_dtype or dtype)
            out.append(StateDecl("kv.paged", i, {"k": kv, "v": kv}, "request", tokens_per_item=page_size))
        else:
            temporal = (cfg.linear_num_value_heads, cfg.linear_value_head_dim, cfg.linear_key_head_dim)
            out.append(StateDecl("gdn.slots", i, {"conv": ((d.conv_dim, cfg.linear_conv_kernel_dim - 1), conv_dtype or dtype),
                                                  "temporal": (temporal, ssm_dtype)}, "slot"))
    return out


class Qwen3_5ForCausalLM:
    """``model@1``."""

    def __init__(self, cfg: Qwen3_5TextConfig, dtype: str, page_size: int, services: Mapping[str, Any],
                 state_dtypes: Mapping[str, str] | None = None) -> None:
        self.cfg, self.dims = cfg, Dims.of(cfg)
        s = services
        self.weights, self.kv, self.gdn_states = s["state.weights@1"], s["state.kv.paged@1"], s["state.gdn.slots@1"]
        self.holder = s["state.holder@1"]
        self.linear, self.embedding, self.norm = s["layers.linear@1"], s["layers.embedding@1"], s["layers.norm@1"]
        self.rope, self.act, self.attn, self.gdn = (s["layers.rope@1"], s["layers.activation@1"], s["layers.attn@1"],
                                                    s["layers.gdn@1"])
        self._assembly = Assembly(["layers.attn@1" if k == "attention" else "layers.gdn@1"
                                   for k in cfg.layers_block_type], states(cfg, dtype, page_size, **(state_dtypes or {})),
                                  params(cfg, dtype))
        self._views: list[Any] = []
        self._cached: tuple[int, list[Mapping[str, torch.Tensor]]] = (-1, [])

    @property
    def assembly(self) -> Assembly:
        return self._assembly

    def start(self) -> None:
        a = self._assembly
        self.weights.declare(a.weights)
        # The pools are sized from what is left once the weights are allocated (state.holder@1).
        weights = self.weights.bind("").buffers().values()
        self.holder.size(a.states, sum(t.numel() * t.element_size() for t in weights))
        self.kv.declare([s for s in a.states if s.kind == "kv.paged"])
        self.gdn_states.declare([s for s in a.states if s.kind == "gdn.slots"])
        self.weights.load()
        self._views = [self.weights.bind(f"model.layers.{i}.") for i in range(self.cfg.num_hidden_layers)]
        self._views.append(self.weights.bind(""))

    # -------------------------------------------------------------- weights

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]], params_dict: Mapping[str, torch.Tensor]
                     ) -> set[str]:
        """``Qwen3_5ForCausalLM.load_weights`` with TP1's weight loaders."""
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if "visual" in name:
                continue
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")

            for param_name, weight_name, shard_id in STACKED_PARAMS_MAPPING:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    continue
                self._load_shard(name, params_dict[name], loaded_weight, shard_id)
                break
            else:
                if name not in params_dict:
                    continue
                default_weight_loader(params_dict[name], loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def output_sizes(self, name: str) -> list[int]:
        """``MergedColumnParallelLinear.output_sizes`` of a fused parameter."""
        cfg, d = self.cfg, self.dims
        if "gate_up_proj" in name:
            return [cfg.intermediate_size, cfg.intermediate_size]
        if "in_proj_qkvz" in name:
            return [d.key_dim, d.key_dim, d.value_dim, d.value_dim]
        if "in_proj_ba" in name:
            return [cfg.linear_num_value_heads, cfg.linear_num_value_heads]
        raise KeyError(name)

    def _load_shard(self, name: str, param: torch.Tensor, loaded_weight: torch.Tensor,
                    shard_id: str | int | tuple[int, ...]) -> None:
        if isinstance(shard_id, str):  # QKVParallelLinear.weight_loader
            cfg = self.cfg
            q_size = cfg.num_attention_heads * (1 + cfg.attn_output_gate) * cfg.head_dim
            kv_size = cfg.num_key_value_heads * cfg.head_dim
            offset, size = {"q": (0, q_size), "k": (q_size, kv_size), "v": (q_size + kv_size, kv_size)}[shard_id]
            param.narrow(0, offset, size).copy_(loaded_weight)
            return
        sizes = self.output_sizes(name)
        if isinstance(shard_id, tuple):  # Qwen3_5GatedDeltaNet._make_packed_weight_loader
            chunks = loaded_weight.split([sizes[i] for i in shard_id], dim=0)
            for idx, chunk in zip(shard_id, chunks, strict=True):
                self._load_shard(name, param, chunk, idx)
            return
        param.narrow(0, sum(sizes[:shard_id]), sizes[shard_id]).copy_(loaded_weight)  # MergedColumnParallelLinear

    def _layer_weights(self) -> list[Mapping[str, torch.Tensor]]:
        generation = self.weights.bind("").generation
        if self._cached[0] != generation:
            self._cached = (generation, [v.buffers() for v in self._views])
        return self._cached[1]

    # -------------------------------------------------------------- forward

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, last_index: torch.Tensor) -> torch.Tensor:
        cfg, eps = self.cfg, self.cfg.rms_norm_eps
        weights = self._layer_weights()
        top = weights[-1]
        hidden_states = self.embedding.embed(input_ids, top["model.embed_tokens.weight"])
        residual: torch.Tensor | None = None
        for layer_id, kind in enumerate(cfg.layers_block_type):
            w = weights[layer_id]
            if residual is None:
                residual = hidden_states
                hidden_states = self.norm.gemma_rms(hidden_states, w["input_layernorm.weight"], eps)
            else:
                hidden_states, residual = self.norm.gemma_rms(hidden_states, w["input_layernorm.weight"], eps, residual)
            if kind == "attention":
                hidden_states = self.self_attention(layer_id, positions, hidden_states, w)
            else:
                hidden_states = self.linear_attention(layer_id, hidden_states, w)
            hidden_states, residual = self.norm.gemma_rms(hidden_states, w["post_attention_layernorm.weight"], eps,
                                                          residual)
            hidden_states = self.mlp(hidden_states, w)
        hidden_states, _ = self.norm.gemma_rms(hidden_states, top["model.norm.weight"], eps, residual)
        # LogitsProcessor: the last token of each sequence, then the LM head
        pruned_states = hidden_states[last_index]
        lm_head = top["model.embed_tokens.weight" if cfg.tie_word_embeddings else "lm_head.weight"]
        return self.embedding.logits(pruned_states, lm_head)

    def mlp(self, x: torch.Tensor, w: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """``Qwen2MoeMLP``: ``gate_up_proj``, ``SiluAndMul``, ``down_proj``."""
        gate_up = self.linear.forward(x, w["mlp.gate_up_proj.weight"])
        return self.linear.forward(self.act.silu_and_mul(gate_up), w["mlp.down_proj.weight"])

    def self_attention(self, layer_id: int, positions: torch.Tensor, hidden_states: torch.Tensor,
                       w: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """``Qwen3_5AttentionDecoderLayer.self_attention`` with ``forward_prepare_native``."""
        cfg, d = self.cfg, self.dims
        head_dim, num_heads = cfg.head_dim, cfg.num_attention_heads
        qkv = self.linear.forward(hidden_states, w["qkv_proj.weight"])
        gate: torch.Tensor | None = None
        if cfg.attn_output_gate:
            q_gate, k, v = qkv.split([d.q_size * 2, d.kv_size, d.kv_size], dim=-1)
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([d.q_size, d.kv_size, d.kv_size], dim=-1)

        q_by_head = self.norm.gemma_rms(q.reshape(-1, head_dim), w["q_norm.weight"], cfg.rms_norm_eps)
        k_by_head = self.norm.gemma_rms(k.reshape(-1, head_dim), w["k_norm.weight"], cfg.rms_norm_eps)
        q, k = q_by_head.view(q.shape), k_by_head.view(k.shape)
        q, k = self.rope.forward(positions, q, k, head_dim, d.rotary_dim, cfg.max_position_embeddings,
                                 cfg.rope_theta)

        attn_output = self.attn.forward(layer_id, q, k, v, num_heads, cfg.num_key_value_heads, head_dim,
                                        head_dim ** -0.5)
        if gate is not None:  # fused_sigmoid_mul
            attn_output = attn_output * torch.sigmoid(gate.reshape(attn_output.shape))
        return self.linear.forward(attn_output, w["o_proj.weight"])

    def linear_attention(self, layer_id: int, hidden_states: torch.Tensor, w: Mapping[str, torch.Tensor]
                         ) -> torch.Tensor:
        """``Qwen3_5GatedDeltaNet.forward``: input projections, ``fix_query_key_value_ordering``,
        the core attention (``RadixLinearAttention``), gated norm, output projection."""
        cfg, d = self.cfg, self.dims
        head_v_dim, nv = cfg.linear_value_head_dim, cfg.linear_num_value_heads
        projected_states_qkvz = self.linear.forward(hidden_states, w["linear_attn.in_proj_qkvz.weight"])
        projected_states_ba = self.linear.forward(hidden_states, w["linear_attn.in_proj_ba.weight"])

        query, key, value, z = projected_states_qkvz.split([d.key_dim, d.key_dim, d.value_dim, d.value_dim], dim=-1)
        b, a = projected_states_ba.split([nv, nv], dim=-1)
        value = value.reshape(value.size(0), -1, head_v_dim)
        z = z.reshape(z.size(0), -1, head_v_dim)
        b = b.contiguous()
        a = a.contiguous()
        query, key, value = (x.reshape(x.shape[0], -1) for x in (query, key, value))
        mixed_qkv = torch.cat((query, key, value), dim=-1)

        conv_weight = w["linear_attn.conv1d.weight"]
        core_attn_out = self.gdn.forward(
            layer_id, mixed_qkv, a, b, conv_weight.view(conv_weight.size(0), conv_weight.size(2)), None,
            cfg.hidden_act, w["linear_attn.A_log"], w["linear_attn.dt_bias"], cfg.linear_num_key_heads, nv,
            cfg.linear_key_head_dim, head_v_dim)

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm.rms_gated(core_attn_out, w["linear_attn.norm.weight"], z, cfg.rms_norm_eps)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-2], core_attn_out.shape[-2] * core_attn_out.shape[-1])
        return self.linear.forward(core_attn_out, w["linear_attn.out_proj.weight"])


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    if param.numel() == 1 and loaded_weight.numel() == 1:
        param.fill_(loaded_weight.item())
    else:
        assert param.size() == loaded_weight.size(), (
            f"Attempted to load weight ({loaded_weight.size()}) into parameter ({param.size()})")
        param.copy_(loaded_weight)


def hf_config(path: str, inline: dict[str, Any]) -> dict[str, Any]:
    if path:
        with open(os.path.join(path, "config.json")) as f:
            return json.load(f)
    if not inline:
        raise ValueError("models.qwen38 needs model.path or model.hf_config")
    return inline


class Unit:
    SERVICES = ("state.weights@1", "state.holder@1", "state.kv.paged@1", "state.gdn.slots@1", "layers.linear@1",
                "layers.embedding@1", "layers.norm@1", "layers.rope@1", "layers.activation@1", "layers.attn@1",
                "layers.gdn@1")

    def __init__(self, ctx: Any, config: Any) -> None:
        cfg = Qwen3_5TextConfig.from_hf(hf_config(config["model.path"], dict(config["model.hf_config"])))
        services = {name: ctx.use(name) for name in self.SERVICES}
        dtype = config["model.dtype"]
        state_dtypes = {
            "kv_dtype": state_dtype("state.kv.dtype", config["state.kv.dtype"], dtype),
            "conv_dtype": state_dtype("state.gdn.conv_dtype", config["state.gdn.conv_dtype"], "bf16"),
            "ssm_dtype": state_dtype("state.gdn.ssm_dtype", config["state.gdn.ssm_dtype"],
                                     STATE_DTYPES.get(cfg.mamba_ssm_dtype, "f32"))}
        self.model = Qwen3_5ForCausalLM(cfg, dtype, config["state.kv.page_size"], services, state_dtypes)
        ctx.register("state.weights@1", self.model.load_weights, label="load_weights")
        self.model.start()
        ctx.provide("model@1", self.model)
