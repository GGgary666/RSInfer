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
import uuid
from pathlib import Path
from typing import Any

from deploy.models.hf import TOKENIZER_FILES, cut, read_config

NAME = "ds1017"
SERVED_NAME = "DeepSeek-V4.1-Flash"
PAGE_SIZE = 128
"""``--page-size`` by default: DeepInfer's block of tokens: a page holds 64 latents at a ratio of 2, as its kernels take them."""
ARCHITECTURES = frozenset({"DeepseekV41ForCausalLM"})
MODEL_TYPES = frozenset({"deepseek_v41", "deepseek_v41_text"})
WORKER_ONLY = frozenset({"models.ds1017", "layers.ds1017", "state.swa_slots", "kernels.ds1017", "kernels.tilelang",
                         "kernels.deepinfer", "state.expert_placement", "policies.eplb"})
"""Worker units of this family alone: a worker serving another family does not load them."""
WORKER_ENV = {"CUDA_SCALE_LAUNCH_QUEUES": "4x"}
"""The workers' environment, as DeepInfer sets it before CUDA starts (deepinfer/__init__.py): room for
the many kernel launches of a step that is queued while the one before runs."""


def matches(cfg: dict[str, Any]) -> bool:
    return bool(ARCHITECTURES & set(cfg.get("architectures") or [])) or cfg.get("model_type") in MODEL_TYPES


def add_arguments(group: argparse._ArgumentGroup) -> None:
    group.add_argument("--yoco-clip", nargs="?", const=140, default=0, type=int,
                       help="Keep this prompt suffix in decoder layers (default when set: 140; 0 disables).")
    group.add_argument("--yoco-mask", action="store_true",
                       help="Validate bounded replay with full query rows and the same visibility mask.")
    group.add_argument("--ds-backend", default="auto", help="dsv4.backend: the layers' kernels (auto, reference, "
                       "tilelang, deepinfer)")


def config(args: argparse.Namespace, cfg: dict[str, Any], model: Path) -> dict[str, Any]:
    """The keys of this family's units. ``cfg`` is the config served (cut to ``--layers``). Full models
    can share compressed KV pages and restore the window slot at a checkpoint."""
    import msgspec

    from models.ds1017_config import DSConfig, states

    if args.dummy:  # they fill floating-point tensors only: every quantized matrix and its scales would be zero
        raise SystemExit("DS 1017 takes no dummy weights (its matrices are stored as FP8 and FP4 bytes): write random "
                         "ones in the checkpoint's formats with `agxinfer checkpoint --config <model dir> [--layers N] "
                         "--out <dir>` and serve that directory")
    model_cfg = DSConfig.from_hf(cfg)
    decls = states(model_cfg, args.page_size)
    clip = getattr(args, "yoco_clip", 0)
    mask = getattr(args, "yoco_mask", False)
    if clip < 0 or (mask and not clip):
        raise SystemExit("--yoco-clip must be nonnegative; --yoco-mask needs a positive --yoco-clip")
    # Engram's compressed token map: an explicit --tokenizer overrides the model's directory (as it
    # does for frontend.tokenizer_path); otherwise the model's or the weights' directory, preferring a
    # prebuilt engram_token_map.json to the tokenizer.json it is built from. The worker is not given
    # the model's directory with --layers, and never the tokenizer's.
    tok = args.tokenizer.resolve() if args.tokenizer is not None else None
    places = [d.resolve() for d in (model, args.weights) if d is not None]
    found = (tok if tok is not None and any((tok / name).is_file() for name in ("engram_token_map.json", "tokenizer.json"))
             else next((d for name in ("engram_token_map.json", "tokenizer.json") for d in places if (d / name).is_file()), None))
    # Several workers hold the model between them (its routed experts and Engram tables, split in
    # rank order): one communication group. Its ranks meet through a file of this run's, a new one each
    # time the engine is planned; tensors travel over NCCL where every rank has a card of its own.
    ranks = len(args.rank_gpus)
    comm: dict[str, Any] = {}
    if ranks > 1:
        meet = args.run.resolve() / "comm"
        meet.mkdir(parents=True, exist_ok=True)
        own_cards = args.device == "cuda" and len(set(args.rank_gpus)) == ranks
        comm = {"comm.size": ranks, "comm.rendezvous": f"file://{meet / uuid.uuid4().hex}",
                "comm.device_backend": "nccl" if own_cards else "gloo",
                **({"sched.idle_wait_s": 0.001, "sched.overlap_lead_s": 0.025,
                    "sched.prefill_coalesce_s": 0.006, "sched.prefill_interval": 108,
                    "sched.balance_prefill": True, "policy.min_chunk_tokens": 2048} if own_cards else {})}
    # A plain prompt starts with BOS, as DeepInfer makes it (its tokenizer repair sets add_bos_token):
    # the checkpoint's tokenizer_config says not to add one, and its chat template writes its own.
    settings = (args.tokenizer or model).resolve() / "tokenizer_config.json"
    bos = settings.is_file() and bool(json.loads(settings.read_text()).get("bos_token"))
    return {"alloc.states": msgspec.json.decode(msgspec.json.encode(decls)), "model.dtype": "bf16",
            "dsv4.backend": args.ds_backend, "frontend.add_bos": bos, **comm,
            "dsv4.yoco_mode": ("mask" if mask else "compact") if clip else "off",
            "dsv4.yoco_keep_tokens": clip or 140, "alloc.prefix_replay_tokens": max(clip, 1),
            # A truncated, window-only model has no KV pages to share.
            **({"alloc.radix_cache": False} if not any(d.kind == "kv.paged" for d in decls) else {}),
            **({"dsv4.token_map_path": str(found)} if found else {}),
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


def write(cfg: dict[str, Any], out: Path, seed: int, stand_in_map: bool = True) -> int:
    """A random checkpoint of ``cfg`` in ``out``: the tensors, ``config.json``, and, with
    ``stand_in_map``, a token map for Engram to hash over (``engram_token_map.json``) in place of the
    one a real checkpoint derives from its tokenizer."""
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
    lookup = token_map(cfg) if stand_in_map else []
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
    tokenizer = (args.tokenizer or source).resolve()
    # With a tokenizer the engine derives the token map from it, as for a real checkpoint.
    size = write(cfg, out, args.seed, stand_in_map=not (tokenizer / "tokenizer.json").is_file())
    for name in TOKENIZER_FILES:
        if (tokenizer / name).is_file() and tokenizer != out:
            shutil.copy(tokenizer / name, out / name)
    layers = cfg.get("text_config", cfg)["num_hidden_layers"]
    print(f"{out}: {layers} layers, {size / 2**30:.2f} GiB of weights in the checkpoint's formats", file=sys.stderr)
    return 0
