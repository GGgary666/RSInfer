"""DeepSeek-V4.1 (``DeepseekV41ForCausalLM``, DS 1017): what the engine's units need to serve its text
model (``models.ds1017``, ``layers.ds1017``, ``state.swa_slots``; doc/specs/v1-ds1017.md), and
random-weight checkpoints in its checkpoint's formats for tests.

A recipe is what ``agxinfer`` knows about one model family: ``matches`` a ``config.json``,
``add_arguments`` for the flags only it has, ``config`` for the config keys only its units read,
``WORKER_ONLY`` for the worker units only it runs, and ``checkpoint``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from deploy.models.hf import TOKENIZER_FILES, cut, read_config

NAME = "ds1017"
SERVED_NAME = "DeepSeek-V4.1-Flash"
ARCHITECTURES = frozenset({"DeepseekV41ForCausalLM"})
MODEL_TYPES = frozenset({"deepseek_v41", "deepseek_v41_text"})
WORKER_ONLY = frozenset({"models.ds1017", "layers.ds1017", "state.swa_slots", "kernels.ds1017", "kernels.tilelang"})
"""Worker units of this family alone: a worker serving another family does not load them."""


def matches(cfg: dict[str, Any]) -> bool:
    return bool(ARCHITECTURES & set(cfg.get("architectures") or [])) or cfg.get("model_type") in MODEL_TYPES


def add_arguments(group: argparse._ArgumentGroup) -> None:
    group.add_argument("--ds-backend", default="auto", help="dsv4.backend: the layers' kernels (auto, reference, "
                       "tilelang)")


def config(args: argparse.Namespace, cfg: dict[str, Any], model: Path) -> dict[str, Any]:
    """The keys of this family's units. ``cfg`` is the config served (cut to ``--layers``). No prefix
    cache yet: its rule for this model's states is not written (doc/specs/v1-ds1017.md §3)."""
    import msgspec

    from models.ds1017_config import DSConfig, states

    decls = states(DSConfig.from_hf(cfg), args.page_size)
    return {"alloc.states": msgspec.json.decode(msgspec.json.encode(decls)), "model.dtype": "bf16",
            "alloc.radix_cache": False, "dsv4.backend": args.ds_backend,
            **({"model.hf_config": cfg} if args.layers else {"model.path": str(model)})}


def random_tensors(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Random weights for the text model of ``cfg``, by the checkpoint's names and in its formats:
    FP8 weights with their block scales, FP4 experts one tensor per expert and matrix, the Engram
    tables, bf16 and fp32 for the rest. Every matrix has unit gain, so activations keep their scale
    through the layers, and every vector some noise, so a misplaced one shows."""
    import torch

    from kernels import ds1017_formats as ops
    from models.ds1017_config import DSConfig, engram_layout

    c = DSConfig.from_hf(cfg)
    c.check()
    g = torch.Generator().manual_seed(seed)
    out: dict[str, torch.Tensor] = {}
    dim, hd, heads, inter = c.hidden_size, c.head_dim, c.num_attention_heads, c.moe_intermediate_size

    def matrix(n: int, k: int) -> torch.Tensor:
        return torch.randn(n, k, generator=g) * k ** -0.5

    def vector(n: int, mean: float = 1.0, std: float = 0.1) -> torch.Tensor:
        return mean + std * torch.randn(n, generator=g)

    def fp8(name: str, n: int, k: int) -> None:
        weight, scale = ops.quantize_fp8(matrix(n, k))
        out[name + ".weight"], out[name + ".scale"] = weight.view(torch.float8_e4m3fn), scale.view(torch.float8_e8m0fnu)

    def bf16(name: str, value: torch.Tensor) -> None:
        out[name] = value.to(torch.bfloat16)

    bf16("embed.weight", torch.randn(c.vocab_size, dim, generator=g))
    bf16("head.weight", matrix(c.vocab_size, dim))
    bf16("norm.weight", vector(dim))
    primes, offsets, _ = engram_layout(c)
    for i in range(c.num_hidden_layers):
        p = f"layers.{i}."
        fp8(p + "attn.wq_a", c.q_lora_rank, dim)
        fp8(p + "attn.wq_b", heads * hd, c.q_lora_rank)
        fp8(p + "attn.wkv", hd, dim)
        fp8(p + "attn.wo_a", c.o_groups * c.o_lora_rank, heads * hd // c.o_groups)
        fp8(p + "attn.wo_b", dim, c.o_groups * c.o_lora_rank)
        out[p + "attn.attn_sink"] = torch.randn(heads, generator=g)
        bf16(p + "attn.q_norm.weight", vector(c.q_lora_rank))
        bf16(p + "attn.kv_norm.weight", vector(hd))
        if i in c.kv_source_layer_ids:
            bf16(p + "attn.compressor.wkv.weight", matrix(hd, dim))
            bf16(p + "attn.compressor.norm.weight", vector(hd))
            if c.ratio(i) > 1:
                bf16(p + "attn.compressor.wgate.weight", matrix(hd, dim))
            bf16(p + "attn.indexer.wk.weight", matrix(c.index_head_dim, hd))
            bf16(p + "attn.indexer.k_norm.weight", vector(c.index_head_dim))
        if i in c.index_source_layer_ids:
            fp8(p + "attn.indexer.wq_b", c.index_n_heads * c.index_head_dim, c.q_lora_rank)
            bf16(p + "attn.indexer.weights_proj.weight", matrix(c.index_n_heads, dim))
        bf16(p + "attn_norm.weight", vector(dim))
        bf16(p + "ffn_norm.weight", vector(dim))
        bf16(p + "ffn.gate.weight", matrix(c.n_routed_experts, dim))
        out[p + "ffn.gate.bias"] = 0.1 * torch.randn(c.n_routed_experts, generator=g)
        for m, (n, k) in (("w1", (inter, dim)), ("w2", (dim, inter)), ("w3", (inter, dim))):
            fp8(p + f"ffn.shared_experts.{m}", n, k)
            for e in range(c.n_routed_experts):
                weight, scale = ops.quantize_fp4(matrix(n, k))
                out[p + f"ffn.experts.{e}.{m}.weight"] = weight.view(torch.int8)
                out[p + f"ffn.experts.{e}.{m}.scale"] = scale.view(torch.float8_e8m0fnu)
        mix = (2 + c.hc_mult) * c.hc_mult
        for which in ("attn", "ffn"):
            out[p + f"hc_{which}_fn"] = matrix(mix, c.hc_mult * dim)
            out[p + f"hc_{which}_base"] = 0.5 * torch.randn(mix, generator=g)
            out[p + f"hc_{which}_scale"] = vector(3, std=0.2)
        if i in c.engram_layer_ids:
            at = c.engram_layer_ids.index(i)
            rows, needed = c.engram_num_embeddings[at], offsets[at][-1] + primes[at][-1][-1]
            if rows < needed:
                raise SystemExit(f"engram_num_embeddings[{at}] = {rows}: this layer's buckets take {needed} rows")
            weight, scale = ops.quantize_rows_fp8(torch.randn(rows, c.engram_head_dim, generator=g))
            out[p + "engram.embed.weight"] = weight.view(torch.float8_e4m3fn)
            out[p + "engram.embed.scale"] = scale.view(torch.float8_e8m0fnu)
            fp8(p + "engram.wkv", dim * (c.hc_mult + 1), c.engram_hash_cols * c.engram_head_dim)
            bf16(p + "engram.q_weight", 1 + 0.1 * torch.randn(c.hc_mult, dim, generator=g))
            bf16(p + "engram.k_weight", 1 + 0.1 * torch.randn(c.hc_mult, dim, generator=g))
    return out


def token_map(cfg: dict[str, Any]) -> list[int]:
    """A stand-in for the compressed token map a tokenizer gives: every id onto
    ``engram_compressed_vocab_size`` ids, all of them used."""
    t = cfg.get("text_config", cfg)
    size = t.get("engram_compressed_vocab_size", 0)
    if size > t["vocab_size"]:
        raise SystemExit("engram_compressed_vocab_size is larger than the vocabulary")
    return [i % size for i in range(t["vocab_size"])] if size else []


def write(cfg: dict[str, Any], out: Path, seed: int) -> int:
    """A random checkpoint of ``cfg`` in ``out``: the tensors, ``config.json``, and the token map
    Engram hashes over (``engram_token_map.json``), which a real checkpoint derives from its tokenizer."""
    from safetensors.torch import save_file

    tensors = random_tensors(cfg, seed)
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(tensors)
    half = len(names) // 2
    index: dict[str, str] = {}
    for i, part in enumerate((names[:half], names[half:])):
        file = f"model-{i + 1:05d}-of-00002.safetensors"
        save_file({n: tensors[n].contiguous() for n in part}, str(out / file), metadata={"format": "pt"})
        index |= {n: file for n in part}
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": index}, indent=1))
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    lookup = token_map(cfg)
    if lookup:
        (out / "engram_token_map.json").write_text(json.dumps(lookup))
    return sum(v.numel() * v.element_size() for v in tensors.values())


def checkpoint(args: argparse.Namespace) -> int:
    """Random weights in a DS 1017 checkpoint's layout, the config cut to ``--layers``: for tests that
    compare against the reference implementation shipped with the real checkpoint."""
    source = args.config.resolve()
    cfg = read_config(source)
    if args.layers:
        cfg = cut(cfg, args.layers)
    out = args.out.resolve()
    size = write(cfg, out, args.seed)
    tokenizer = (args.tokenizer or source).resolve()
    for name in TOKENIZER_FILES:
        if (tokenizer / name).is_file() and tokenizer != out:
            shutil.copy(tokenizer / name, out / name)
    layers = cfg.get("text_config", cfg)["num_hidden_layers"]
    print(f"{out}: {layers} layers, {size / 2**30:.2f} GiB of weights in the checkpoint's formats", file=sys.stderr)
    return 0
