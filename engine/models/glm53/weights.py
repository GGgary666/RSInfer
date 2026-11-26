"""Load the local TP projections and EP expert stacks directly from safetensors."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import import_module

import torch

from core.contracts.model import WeightDecl
from models.glm53_config import LINEAR, Config

PREFIX = "model.language_model."
EMBED = "embed_tokens.weight"
NORM = "norm.weight"
HEAD = "lm_head.weight"
ROT = "rot.weight"
FRACTAL_NZ = 29
type Index = tuple[int | slice, ...]


@dataclass(frozen=True)
class LoadPart:
    dst: str
    source: Index = ()
    destination: Index = ()
    transpose: bool = False
    squeeze: int | None = None


class Weights:
    """A declaration and the checkpoint pieces needed to fill it on this rank."""

    def __init__(self, cfg: Config, world: int, rank: int, *, dfc: bool = False,
                 draft: bool = False, rotate: bool = False, shared_expert_dp: bool = False) -> None:
        self.cfg, self.world, self.rank = cfg, world, rank
        self.dfc = dfc
        self.shared_expert_dp = shared_expert_dp
        self.nz: set[str] = set()
        self.decls: list[WeightDecl] = []
        self.sources: dict[str, LoadPart] = {}
        self.required: dict[str, set[str]] = defaultdict(set)
        self.plain(EMBED, (cfg.vocab, cfg.hidden))
        self.plain(NORM, (cfg.hidden,))
        self.plain(HEAD, (cfg.vocab, cfg.hidden))
        for layer, (attention, mlp) in enumerate(zip(cfg.attention, cfg.mlp, strict=True)):
            p = f"layers.{layer}."
            mix = (cfg.streams + 2) * cfg.streams
            for hc in ("attn_hc", "ffn_hc"):
                self.plain(p + hc + ".base", (mix,), "f32")
                self.plain(p + hc + ".scale", (3,), "f32")
                self.plain(p + hc + ".fn", (mix, cfg.streams * cfg.hidden), "f32")
            for norm in ("input_layernorm.weight", "post_attention_layernorm.weight"):
                self.plain(p + norm, (cfg.hidden,))
            if attention == LINEAR:
                self.kda(p + "self_attn.")
            else:
                self.mla(p + "self_attn.")
            if mlp == "dense":
                self.ffn(p + "mlp.", cfg.intermediate, world)
            else:
                self.moe(p + "mlp.")
        if draft:
            self.draft(len(cfg.attention), rotate)

    def draft(self, layer: int, rotate: bool) -> None:
        cfg = self.cfg
        p = f"layers.{layer}."
        self.plain(p + EMBED, (cfg.vocab, cfg.hidden))
        self.plain(p + "eh_proj.weight", (cfg.hidden, 2 * cfg.hidden))
        for name in ("enorm", "hnorm", "input_layernorm", "post_attention_layernorm", "shared_head.norm"):
            self.plain(p + name + ".weight", (cfg.hidden,))
        if rotate:
            self.plain(ROT, (cfg.hidden, cfg.hidden))
        self.plain(p + "shared_head.head.weight", (cfg.vocab, cfg.hidden))
        self.mla(p + "self_attn.")
        self.moe(p + "mlp.")

    def sliced(self, width: int) -> slice:
        return slice(self.rank * width, (self.rank + 1) * width)

    def decl(self, name: str, shape: tuple[int, ...], dtype: str = "bf16") -> None:
        self.decls.append(WeightDecl(name, shape, dtype))

    def part(self, name: str, dst: str, source: Index = (), destination: Index = (), *,
             transpose: bool = False, squeeze: int | None = None) -> None:
        key = name if name in (HEAD, ROT) else PREFIX + name
        self.sources[key] = LoadPart(dst, source, destination, transpose, squeeze)
        self.required[dst].add(key)

    def plain(self, name: str, shape: tuple[int, ...], dtype: str = "bf16", source: Index = ()) -> None:
        self.decl(name, shape, dtype)
        self.part(name, name, source)

    def kda(self, p: str) -> None:
        cfg = self.cfg
        heads = cfg.kda_heads // self.world
        width = heads * cfg.kda_dim
        projected = p + "input.weight"
        self.decl(projected, (3 * width + heads + 2 * cfg.kda_dim, cfg.hidden))
        offset = 0
        for name, count, tp in (("q_proj", width, True), ("k_proj", width, True), ("v_proj", width, True),
                                ("b_proj", heads, True), ("forget_gate.f_a_proj", cfg.kda_dim, False),
                                ("g_a_proj", cfg.kda_dim, False)):
            self.part(p + name + ".weight", projected, (self.sliced(count),) if tp else (),
                      (slice(offset, offset + count),))
            offset += count
        conv = p + "conv.weight"
        self.decl(conv, (cfg.conv_width, 3 * width))
        for i, kind in enumerate(("q", "k", "v")):
            self.part(p + kind + "_conv1d.weight", conv, (self.sliced(width),),
                      (slice(None), slice(i * width, (i + 1) * width)), squeeze=1, transpose=True)
        self.plain(p + "forget_gate.A_log", (heads,), "f32", (self.sliced(heads),))
        self.plain(p + "forget_gate.dt_bias", (width,), "f32", (self.sliced(width),))
        self.plain(p + "forget_gate.f_b_proj.weight", (width, cfg.kda_dim), source=(self.sliced(width),))
        self.plain(p + "g_b_proj.weight", (width, cfg.kda_dim), source=(self.sliced(width),))
        self.plain(p + "o_norm.weight", (cfg.kda_dim,))
        self.plain(p + "o_proj.weight", (cfg.hidden, width), source=(slice(None), self.sliced(width)))

    def mla(self, p: str) -> None:
        cfg = self.cfg
        heads = cfg.mla_heads // self.world
        q, kv, v = heads * cfg.q_dim, heads * (cfg.q_dim + cfg.v_dim), heads * cfg.v_dim
        self.plain(p + "q_a_proj.weight", (cfg.q_rank, cfg.hidden))
        self.plain(p + "q_a_layernorm.weight", (cfg.q_rank,))
        self.plain(p + "q_b_proj.weight", (q, cfg.q_rank), source=(self.sliced(q),))
        self.plain(p + "kv_a_proj_with_mqa.weight", (cfg.kv_rank, cfg.hidden))
        self.plain(p + "kv_a_layernorm.weight", (cfg.kv_rank,))
        self.plain(p + "kv_b_proj.weight", (kv, cfg.kv_rank), source=(self.sliced(kv),))
        self.plain(p + "o_proj.weight", (cfg.hidden, v), source=(slice(None), self.sliced(v)))
        ip = p + "indexer."
        both = ip + "wk_weights.weight"
        self.decl(both, (cfg.index_dim + cfg.index_heads, cfg.hidden), "f32")
        self.part(ip + "wk.weight", both, destination=(slice(0, cfg.index_dim),))
        self.part(ip + "weights_proj.weight", both, destination=(slice(cfg.index_dim, None),))
        self.plain(ip + "k_norm.weight", (cfg.index_dim,), "f32")
        self.plain(ip + "k_norm.bias", (cfg.index_dim,), "f32")
        self.plain(ip + "wq_b.weight", (cfg.index_heads * cfg.index_dim, cfg.q_rank))
        self.plain(ip + "index_kpool_compress_gate", (cfg.index_dim, cfg.hidden), "f32")
        self.plain(ip + "index_kpool_compress_ape", (cfg.pool, cfg.index_dim), "f32")

    def ffn(self, p: str, intermediate: int, parts: int) -> None:
        width, hidden = intermediate // parts, self.cfg.hidden
        shard = self.sliced(width) if parts > 1 else slice(0, width)
        self.decl(p + "gate_up.weight", (hidden, 2 * width), "i8")
        self.decl(p + "gate_up.scale", (2 * width,))
        self.decl(p + "down.weight", (width, hidden), "i8")
        self.decl(p + "down.scale", (hidden,))
        for i, name in enumerate(("gate_proj", "up_proj")):
            dst = slice(i * width, (i + 1) * width)
            rows = (shard,)
            self.part(p + name + ".weight", p + "gate_up.weight", rows, (slice(None), dst), transpose=True)
            self.part(p + name + ".weight_scale", p + "gate_up.scale", rows, (dst,), squeeze=1)
        self.part(p + "down_proj.weight", p + "down.weight", (slice(None), shard), transpose=True)
        self.part(p + "down_proj.weight_scale", p + "down.scale", squeeze=1)

    def moe(self, p: str) -> None:
        cfg = self.cfg
        self.plain(p + "gate.weight", (cfg.experts, cfg.hidden), "f32")
        self.plain(p + "gate.e_score_correction_bias", (cfg.experts,), "f32")
        if cfg.shared_width:
            self.ffn(p + "shared_experts.", cfg.shared_width, 1 if self.shared_expert_dp else self.world)
        local, width, hidden = cfg.experts // self.world, cfg.expert_width, cfg.hidden
        e = p + "experts."
        scale_dtype = "i64" if self.dfc else "bf16"
        self.decl(e + "gate_up.weight", (local, hidden, 2 * width), "i8")
        self.decl(e + "gate_up.scale", (local, 2 * width), scale_dtype)
        self.decl(e + "down.weight", (local, width, hidden), "i8")
        self.decl(e + "down.scale", (local, hidden), scale_dtype)
        if self.dfc:
            self.nz.update((e + "gate_up.weight", e + "down.weight"))
        for i in range(local):
            source = e + f"{self.rank * local + i}."
            for j, name in enumerate(("gate_proj", "up_proj")):
                dst = slice(j * width, (j + 1) * width)
                self.part(source + name + ".weight", e + "gate_up.weight",
                          destination=(i, slice(None), dst), transpose=True)
                self.part(source + name + ".weight_scale", e + "gate_up.scale", destination=(i, dst), squeeze=1)
            self.part(source + "down_proj.weight", e + "down.weight", destination=(i,), transpose=True)
            self.part(source + "down_proj.weight_scale", e + "down.scale", destination=(i,), squeeze=1)

    def allocate(self, decl: WeightDecl) -> torch.Tensor | None:
        if decl.name not in self.nz:
            return None
        npu = import_module("torch_npu")
        return npu.empty_with_format(decl.shape, dtype=torch.int8, device=torch.device("npu"), acl_format=FRACTAL_NZ)

    def wants_weight(self, name: str) -> bool:
        return name in self.sources

    def weight_rows(self, name: str) -> tuple[int, int] | None:
        source = self.sources[name].source
        if len(source) != 1:
            return None
        rows = source[0]
        assert isinstance(rows, slice) and rows.start is not None and rows.stop is not None
        return rows.start, rows.stop

    def load_weights(self, incoming: Iterable[tuple[str, torch.Tensor]], tensors: Mapping[str, torch.Tensor]
                     ) -> set[str]:
        seen: set[str] = set()
        staged: dict[str, torch.Tensor] = {}
        received: dict[str, set[str]] = defaultdict(set)
        npu = import_module("torch_npu") if self.nz else None
        for key, tensor in incoming:
            if key not in self.sources:
                continue
            p = self.sources[key]
            if p.source and len(p.source) != 1:
                tensor = tensor[p.source]
            if p.squeeze is not None:
                tensor = tensor.squeeze(p.squeeze)
            if p.transpose:
                tensor = tensor.t()
            dest = tensors[p.dst]
            if p.dst in self.nz:
                if p.dst not in staged:
                    staged[p.dst] = torch.empty(dest.shape, device="cpu", dtype=torch.int8)
                dest = staged[p.dst]
            out = dest[p.destination]
            if out.shape != tensor.shape:
                raise ValueError(f"{key}: expected {tuple(out.shape)}, got {tuple(tensor.shape)}")
            if out.dtype == torch.int8 and tensor.dtype != torch.int8:
                raise ValueError(f"{key}: GLM-5.3-Flash requires W8A8 int8 weights")
            if out.dtype == torch.int64:
                # DFC reads the low 32 bits of each word as an FP32 scale.
                tensor = tensor.to(torch.bfloat16).float().contiguous().view(torch.int32).to(torch.int64)
            out.copy_(tensor)
            seen.add(key)
            if p.dst in self.nz:
                received[p.dst].add(key)
                if self.required[p.dst] == received[p.dst]:
                    assert npu is not None
                    target = tensors[p.dst]
                    # Convert on the NPU with the explicit format operator, then copy equal layouts.
                    packed = npu.npu_format_cast(staged.pop(p.dst).to(target.device), FRACTAL_NZ)
                    target.copy_(packed)
        return {name for name, keys in self.required.items() if keys <= seen}
