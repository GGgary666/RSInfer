# Copyright (c) 2023 DeepSeek
# SPDX-License-Identifier: MIT
"""``models.ds1017``: DeepSeek-V4.1 text (DS 1017), ``model@1``. After ``Transformer`` and ``Block`` of
the reference implementation shipped with the checkpoint (``inference/model.py`` of
deepseek-ai/DeepSeek-V4.1-Flash-1017; LICENSE.deepseek in ``kernels/ds1017``), without vision and
without the draft layers (doc/specs/v1-ds1017.md).

What moved: the parameters are declared to ``state.weights@1`` under the checkpoint's names, in the
checkpoint's bytes where it is quantized (FP8 weights and their ue8m0 scales, FP4 experts, the Engram
tables: ``u8``); ``load_weights`` does what that implementation's ``convert.py`` does for one rank
(``wo_a`` dequantized to bf16, the head and the pooling compressors' projections in fp32). Beside
that, as DeepInfer lays weights out for its GEMM kernels: a layer's routed experts are two stacks,
each expert's ``w1`` over its ``w3`` (one product gives a SwiGLU's two inputs) and ``w2``, and the
shared expert's ``w1`` over its ``w3`` likewise; the
experts' scales are kept as a GEMM kernel reads them (``formats.pack_scales``), and so are each FP8
weight's, beside the checkpoint's block scales. They are declared tensors filled by ``load_weights``,
so a ``refit`` writes them again. The layers are ``layers.ds1017``, which the
executor prepares for the step; the residual stream is ``hc_mult`` copies ``[tokens, hc, dim]``.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from typing import Any

import torch

from core.contracts.model import Assembly, WeightDecl
from kernels import ds1017_formats as formats
from models.ds1017_config import KV_KIND, SLOT_KIND, DSConfig, states

LAYER = re.compile(r"layers\.(\d+)\.")
EXPERT = re.compile(r"(layers\.\d+\.ffn\.experts)\.(\d+)\.(w[123])\.(weight|scale)")
SHARED = re.compile(r"(layers\.\d+\.ffn\.shared_experts)\.(w[13])\.(weight|scale)")
SKIPPED = ("mtp.", "vision.", "aligner.", "image_")
"""Checkpoint tensors the text model without draft layers does not read."""


def fp8(name: str, out: int, into: int) -> list[WeightDecl]:
    """An FP8 weight ``[out, into]`` and its 32 x 32 block scales, both as the checkpoint's bytes, and
    the scales as a GEMM kernel reads them."""
    return [WeightDecl(name + ".weight", (out, into), "u8"),
            WeightDecl(name + ".scale", (-(-out // 32), -(-into // 32)), "u8"),
            WeightDecl(name + ".packed", formats.packed_shape(out, into), "u8")]


def params(cfg: DSConfig) -> list[WeightDecl]:
    """The backbone's parameters, by the checkpoint's names; the routed experts of a layer as
    ``ffn.experts.w13.weight`` ``[experts, 2 * inter, dim / 2]`` (``w1`` over ``w3``) and
    ``ffn.experts.w2.weight``, each with its ``packed`` scales."""
    dim, hd, heads, inter, experts = (cfg.hidden_size, cfg.head_dim, cfg.num_attention_heads,
                                      cfg.moe_intermediate_size, cfg.n_routed_experts)
    mix = (2 + cfg.hc_mult) * cfg.hc_mult

    def w(name: str, *shape: int, fmt: str = "bf16") -> WeightDecl:
        return WeightDecl(name, shape, fmt)

    out = [w("embed.weight", cfg.vocab_size, dim), w("head.weight", cfg.vocab_size, dim, fmt="f32"), w("norm.weight", dim)]
    for i in range(cfg.num_hidden_layers):
        p = f"layers.{i}."
        out += [*fp8(p + "attn.wq_a", cfg.q_lora_rank, dim), *fp8(p + "attn.wq_b", heads * hd, cfg.q_lora_rank),
                *fp8(p + "attn.wkv", hd, dim),
                w(p + "attn.wo_a.weight", cfg.o_groups * cfg.o_lora_rank, heads * hd // cfg.o_groups),
                *fp8(p + "attn.wo_b", dim, cfg.o_groups * cfg.o_lora_rank), w(p + "attn.attn_sink", heads, fmt="f32"),
                w(p + "attn.q_norm.weight", cfg.q_lora_rank), w(p + "attn.kv_norm.weight", hd)]
        if i in cfg.kv_source_layer_ids:
            pooled = cfg.ratio(i) > 1  # the softmax pooling runs in fp32, and so do its projections
            out += [w(p + "attn.compressor.wkv.weight", hd, dim, fmt="f32" if pooled else "bf16"),
                    w(p + "attn.compressor.norm.weight", hd), w(p + "attn.indexer.wk.weight", cfg.index_head_dim, hd),
                    w(p + "attn.indexer.k_norm.weight", cfg.index_head_dim)]
            if pooled:
                out.append(w(p + "attn.compressor.wgate.weight", hd, dim, fmt="f32"))
        if i in cfg.index_source_layer_ids:
            out += [*fp8(p + "attn.indexer.wq_b", cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank),
                    w(p + "attn.indexer.weights_proj.weight", cfg.index_n_heads, dim)]
        out += [w(p + "attn_norm.weight", dim), w(p + "ffn_norm.weight", dim),
                w(p + "ffn.gate.weight", experts, dim, fmt="f32"), w(p + "ffn.gate.bias", experts, fmt="f32")]
        for m, (n, k) in (("w13", (2 * inter, dim)), ("w2", (dim, inter))):
            out += [w(p + f"ffn.experts.{m}.weight", experts, n, k // 2, fmt="u8"),
                    w(p + f"ffn.experts.{m}.packed", experts, *formats.packed_shape(n, k), fmt="u8")]
        # The shared expert's w1 over its w3, as DeepInfer's shared_gate_up_proj (one GEMM).
        out += [*fp8(p + "ffn.shared_experts.w13", 2 * inter, dim), *fp8(p + "ffn.shared_experts.w2", dim, inter)]
        for which in ("attn", "ffn"):
            out += [w(p + f"hc_{which}_fn", mix, cfg.hc_mult * dim, fmt="f32"), w(p + f"hc_{which}_base", mix, fmt="f32"),
                    w(p + f"hc_{which}_scale", 3, fmt="f32")]
        if i in cfg.engram_layer_ids:
            rows, eh = cfg.engram_num_embeddings[cfg.engram_layer_ids.index(i)], cfg.engram_head_dim
            out += [w(p + "engram.embed.weight", rows, eh, fmt="u8"), w(p + "engram.embed.scale", rows, eh // 32, fmt="u8"),
                    *fp8(p + "engram.wkv", dim * (cfg.hc_mult + 1), cfg.engram_hash_cols * eh),
                    w(p + "engram.q_weight", cfg.hc_mult, dim), w(p + "engram.k_weight", cfg.hc_mult, dim)]
    return out


def checkpoint_name(name: str) -> str:
    """A checkpoint tensor's name as ``convert.py`` reads it (older exports name things as HF does)."""
    name = name.removeprefix("model.").replace("self_attn", "attn")
    if not name.startswith("vision."):
        name = name.replace("mlp", "ffn")
    return name.replace("weight_scale_inv", "scale").replace("e_score_correction_bias", "bias")


def as_bytes(t: torch.Tensor) -> torch.Tensor:
    if t.element_size() != 1:
        raise ValueError(f"a quantized tensor has one byte per element, not {t.dtype}")
    return t.contiguous().view(torch.uint8)


class DeepseekV41ForCausalLM:
    """``model@1``."""

    def __init__(self, cfg: DSConfig, dtype: str, page_size: int, services: Mapping[str, Any]) -> None:
        if dtype != "bf16":
            raise ValueError(f"model.dtype = {dtype!r}: DS 1017's activations are bf16 (its kernels quantize from it)")
        self.cfg, self._dtype = cfg, dtype
        s = services
        self.weights, self.holder = s["state.weights@1"], s["state.holder@1"]
        self.kv, self.slots = s["state.kv.paged@1"], s["state.swa.slots@1"]
        self.embedding, self.layers = s["layers.embedding@1"], s["layers.ds1017@1"]
        self._assembly = Assembly(["layers.ds1017@1"] * cfg.num_hidden_layers, states(cfg, page_size), params(cfg))
        self._views: list[Any] = []
        self._cached: tuple[int, list[Mapping[str, torch.Tensor]]] = (-1, [])

    @property
    def assembly(self) -> Assembly:
        return self._assembly

    @property
    def stores(self) -> Mapping[str, Any]:
        return {KV_KIND: self.kv, SLOT_KIND: self.slots}

    @property
    def backends(self) -> Mapping[str, Any]:
        return {"layers.ds1017@1": self.layers}

    @property
    def hidden_size(self) -> int:
        return self.cfg.hidden_size

    @property
    def dtype(self) -> str:
        return self._dtype

    def start(self) -> None:
        a = self._assembly
        self.weights.declare(a.weights)
        weights = self.weights.bind("").buffers().values()
        self.holder.size(a.states, sum(t.numel() * t.element_size() for t in weights))
        self.kv.declare([s for s in a.states if s.kind == KV_KIND])
        self.slots.declare([s for s in a.states if s.kind == SLOT_KIND])
        self.weights.load()
        self._views = [self.weights.bind(f"layers.{i}.") for i in range(self.cfg.num_hidden_layers)]
        self._views.append(self.weights.bind(""))

    # -------------------------------------------------------------- weights

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]], params_dict: Mapping[str, torch.Tensor]
                     ) -> set[str]:
        """``convert.py`` for one rank, into the declared tensors: quantized tensors as their bytes, a
        layer's experts into their stacks, ``wo_a`` dequantized, everything else cast."""
        loaded: set[str] = set()
        stacked: dict[str, set[tuple[int, str, str]]] = {}
        halves: dict[str, torch.Tensor] = {}
        inter = self.cfg.moe_intermediate_size
        for raw, tensor in weights:
            name = checkpoint_name(raw)
            if name.startswith(SKIPPED) or name.endswith("bias_vl"):
                continue
            expert = EXPERT.fullmatch(name)
            if expert:
                # w1 is the upper half of the ``w13`` stack, w3 the lower; a scale goes in packed.
                index, matrix, part = int(expert[2]), expert[3], expert[4]
                stack = f"{expert[1]}.{'w2' if matrix == 'w2' else 'w13'}"
                target = f"{stack}.{'weight' if part == 'weight' else 'packed'}"
                if target in params_dict and index < params_dict[target].shape[0]:
                    rows = slice(inter, 2 * inter) if matrix == "w3" else slice(0, tensor.shape[0])
                    data = as_bytes(tensor)
                    if part == "weight":
                        params_dict[target][index, rows].copy_(data)
                    else:
                        params_dict[target][index, :, rows].copy_(formats.pack_scales(data)[:, :data.shape[0]])
                    stacked.setdefault(stack, set()).add((index, matrix, part))
                continue
            shared = SHARED.fullmatch(name)
            if shared:  # w1 is the upper half of w13, w3 the lower; a scale's rows and its packed rows too
                target = f"{shared[1]}.w13"
                if target + ".weight" in params_dict:
                    lower = shared[2] == "w3"
                    data = as_bytes(tensor)
                    if shared[3] == "weight":
                        params_dict[target + ".weight"][inter * lower:inter * (lower + 1)].copy_(data)
                    else:
                        blocks = inter // formats.BLOCK
                        params_dict[target + ".scale"][blocks * lower:blocks * (lower + 1)].copy_(data)
                        params_dict[target + ".packed"][:, inter * lower:inter * (lower + 1)].copy_(
                            formats.pack_scales(formats.block_rows(data, inter)))
                    halves[f"{target}.{shared[2]}.{shared[3]}"] = data[:0]
                    if all(f"{target}.{m}.{part}" in halves for m in ("w1", "w3") for part in ("weight", "scale")):
                        loaded.update((target + ".weight", target + ".scale", target + ".packed"))
                continue
            if name.endswith(("attn.wo_a.weight", "attn.wo_a.scale")):
                target = name.rsplit(".", 1)[0] + ".weight"
                if target not in params_dict:
                    continue
                if tensor.element_size() != 1:  # already dequantized
                    params_dict[target].copy_(tensor.to(params_dict[target].dtype))
                    loaded.add(target)
                    continue
                halves[name] = as_bytes(tensor)
                weight, scale = halves.get(target), halves.get(target[:-len("weight")] + "scale")
                if weight is not None and scale is not None:
                    params_dict[target].copy_(formats.dequant_fp8(weight, scale).to(params_dict[target].dtype))
                    loaded.add(target)
                    del halves[target], halves[target[:-len("weight")] + "scale"]
                continue
            param = params_dict.get(name)
            if param is None:
                continue
            if tuple(param.shape) != tuple(tensor.shape):
                raise ValueError(f"{raw} is {tuple(tensor.shape)} in the checkpoint, {tuple(param.shape)} in the config")
            param.copy_(as_bytes(tensor) if param.dtype == torch.uint8 else tensor.to(param.dtype))
            loaded.add(name)
            packed = params_dict.get(name[:-len("scale")] + "packed") if name.endswith(".scale") else None
            if packed is not None:
                packed.copy_(formats.pack_scales(formats.block_rows(param, packed.shape[1])))
                loaded.add(name[:-len("scale")] + "packed")
        for stack, seen in stacked.items():
            experts = params_dict[stack + ".weight"].shape[0]
            for matrix in (("w2",) if stack.endswith("w2") else ("w1", "w3")):
                count = len({index for index, m, part in seen if m == matrix and part == "weight"}
                            & {index for index, m, part in seen if m == matrix and part == "scale"})
                if count != experts:
                    raise ValueError(f"the checkpoint has {count} of the {experts} experts of "
                                     f"{stack.rsplit('.', 1)[0]}.{matrix}")
            loaded.update((stack + ".weight", stack + ".packed"))
        return loaded

    def wants_weight(self, raw: str) -> bool:
        """Whether ``load_weights`` reads the checkpoint tensor ``raw``: the weight store does not read
        the others (the layers a cut config leaves out, the draft layers, vision)."""
        name = checkpoint_name(raw)
        if name.startswith(SKIPPED) or name.endswith("bias_vl"):
            return False
        layer = LAYER.match(name)
        return layer is None or int(layer[1]) < self.cfg.num_hidden_layers

    def _layer_weights(self) -> list[Mapping[str, torch.Tensor]]:
        generation = self.weights.bind("").generation
        if self._cached[0] != generation:
            self._cached = (generation, [v.buffers() for v in self._views])
        return self._cached[1]

    # -------------------------------------------------------------- forward

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, last_index: torch.Tensor) -> torch.Tensor:
        """``Transformer.forward``: logits ``[rows, vocab]`` in fp32 at each row's last token."""
        cfg, layers = self.cfg, self.layers
        weights = self._layer_weights()
        top = weights[cfg.num_hidden_layers]
        layers.begin(input_ids, positions)
        h = self.embedding.embed(input_ids, top["embed.weight"]).unsqueeze(1).repeat(1, cfg.hc_mult, 1)
        # Each sublayer's mixes give the next sublayer its input: the first takes the stream's first copy.
        pre = torch.zeros(h.shape[0], cfg.hc_mult, dtype=torch.float32, device=h.device)
        pre[:, 0] = 1.0
        for i in range(cfg.num_hidden_layers):
            w = weights[i]
            if i in cfg.engram_layer_ids:
                h = layers.engram(i, h, w)
            residual = h
            attn_pre, post, comb = layers.hc_mixes(h, w, "attn")
            x = layers.attention(i, layers.norm(layers.hc_pre(h, pre), w["attn_norm.weight"]), w)
            h = layers.hc_post(x, residual, post, comb)
            residual = h
            pre, post, comb = layers.hc_mixes(h, w, "ffn")
            x = layers.moe(layers.norm(layers.hc_pre(h, attn_pre), w["ffn_norm.weight"]), w)
            h = layers.hc_post(x, residual, post, comb)
        x = layers.norm(layers.hc_pre(h, pre), top["norm.weight"])
        return layers.linear(x[last_index], top["head.weight"])


def hf_config(path: str, inline: dict[str, Any]) -> dict[str, Any]:
    if path:
        with open(os.path.join(path, "config.json")) as f:
            return json.load(f)
    if not inline:
        raise ValueError("models.ds1017 needs model.path or model.hf_config")
    return inline


class Unit:
    SERVICES = ("state.weights@1", "state.holder@1", "state.kv.paged@1", "state.swa.slots@1", "layers.embedding@1",
                "layers.ds1017@1")

    def __init__(self, ctx: Any, config: Any) -> None:
        if not config["weights.path"]:
            raise ValueError("weights.path is empty, and DS 1017 takes no dummy weights: they fill floating-point "
                             "tensors only, and its matrices are stored as FP8 and FP4 bytes with their scales "
                             "(agxinfer checkpoint writes random weights in the checkpoint's formats)")
        cfg = DSConfig.from_hf(hf_config(config["model.path"], dict(config["model.hf_config"])))
        cfg.check()
        self.model = DeepseekV41ForCausalLM(cfg, config["model.dtype"], config["state.kv.page_size"],
                                            {name: ctx.use(name) for name in self.SERVICES})
        ctx.register("state.weights@1", self.model.load_weights, label="load_weights")
        self.model.start()
        ctx.provide("model@1", self.model)
