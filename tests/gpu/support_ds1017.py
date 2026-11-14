"""DS 1017 on the GPU: the reference implementation shipped with the checkpoint (``ds1017_reference``,
its TileLang kernels in ``kernels.tilelang``) on one of this repository's random checkpoints, a small
config whose attention kernel fits an SM 8.9 card, and the engine's worker on the same checkpoint.

The reference needs ``tilelang`` (0.1.12); install it beside the GPU environment without replacing
its torch (doc/specs/v1-ds1017.md)."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from models.ds1017_config import DSConfig, engram_layout
from platforms.torch_eager import TorchEagerPlatform
from tests.units.support.ds1017 import DSWorker

REFERENCE = Path(__file__).resolve().parent / "ds1017_reference"


def small_config(layers: int = 8, **kw: Any) -> dict[str, Any]:
    """The reference's own small shapes (``ModelArgs``' defaults: 1024 wide, 16 heads of 128), which its
    attention kernel can run on an SM 8.9 card (the real 64 heads of 512 need more shared memory per
    block than it has), with every kind of layer of the real model in 8 layers, Engram before two."""
    text: dict[str, Any] = {
        "model_type": "deepseek_v41_text", "vocab_size": 4096, "hidden_size": 1024, "moe_intermediate_size": 1024,
        "num_hidden_layers": layers, "num_attention_heads": 16, "head_dim": 128, "qk_rope_head_dim": 32,
        "q_lora_rank": 256, "o_lora_rank": 256, "o_groups": 8, "n_routed_experts": 8, "n_shared_experts": 1,
        "num_experts_per_tok": 2, "scoring_func": "sqrtsoftplus", "norm_topk_prob": True, "routed_scaling_factor": 1.5,
        "swiglu_limit": 10.0, "rms_norm_eps": 1e-20, "sliding_window": 128,
        "compress_ratios": [0, 0, 2, 2, 2, 1, 1, 1], "kv_source_layer_ids": [2, 5], "index_source_layer_ids": [2, 5, 7],
        "index_n_heads": 16, "index_head_dim": 64, "index_topk": 64, "candidate_source_layer_id": 5,
        "candidate_topk_blocks": 16, "candidate_block_size": 8, "rope_theta": 10000, "compress_rope_theta": 160000,
        "rope_scaling": {"factor": 16, "beta_fast": 32, "beta_slow": 1, "original_max_position_embeddings": 256},
        "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, "engram_layer_ids": [1, 4], "engram_max_ngram_size": 4,
        "engram_vocab_size": 500, "engram_n_heads": 2, "engram_head_dim": 64, "engram_pad_token_id": 2,
        "engram_compressed_vocab_size": 3000, **kw}
    if "engram_num_embeddings" not in text:
        primes, offsets, _ = engram_layout(DSConfig.from_hf({**text, "engram_num_embeddings": [0, 0]}))
        text["engram_num_embeddings"] = [o[-1] + p[-1][-1] for o, p in zip(offsets, primes, strict=True)]
    return {"architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41", "text_config": text}


def real_shapes_config(**kw: Any) -> dict[str, Any]:
    """The released model's shapes (5120 wide, 64 heads of 512, its ranks, window, indexer and
    candidate sizes) in 6 layers with 8 experts each and small Engram tables: every kind of layer, and
    the kernels at the shapes they serve. Its attention kernel needs an SM 10.0 card."""
    return small_config(
        layers=6, hidden_size=5120, moe_intermediate_size=2304, num_attention_heads=64, head_dim=512,
        qk_rope_head_dim=64, q_lora_rank=1280, o_lora_rank=1024, n_routed_experts=8, num_experts_per_tok=6,
        compress_ratios=[0, 2, 2, 1, 1, 1], kv_source_layer_ids=[1, 3], index_source_layer_ids=[1, 3, 5],
        index_n_heads=32, index_head_dim=128, index_topk=512, candidate_source_layer_id=3, candidate_topk_blocks=2048,
        candidate_block_size=8, engram_layer_ids=[1, 4], engram_n_heads=8, engram_head_dim=256,
        rope_scaling={"factor": 16, "beta_fast": 32, "beta_slow": 1, "original_max_position_embeddings": 65536}, **kw)


def _load(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, REFERENCE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reference_modules() -> tuple[Any, Any]:
    """The reference's ``model`` and ``engram`` modules, imported as it imports them: by bare names,
    its kernels the engine's copy, and stand-ins for the vision files it does not use here."""
    from kernels.tilelang import kernel

    sys.modules["kernel"] = kernel
    images = types.ModuleType("image_processor")
    for i, constant in enumerate(("IMAGE", "IMAGE_END", "IMAGE_NEW_LINE", "IMAGE_START")):
        setattr(images, constant, i)
    vision = types.ModuleType("vision")
    vision.Aligner = vision.ViT = object  # pyright: ignore[reportAttributeAccessIssue]
    sys.modules["image_processor"], sys.modules["vision"] = images, vision
    engram = _load("engram")
    return _load("model"), engram


def reference_model(checkpoint: Path, max_seq_len: int, rows: int = 1, cfg: dict[str, Any] | None = None) -> Any:
    """The reference ``Transformer`` on the checkpoint, loaded as its ``convert.py`` prepares one rank:
    ``wo_a`` dequantized to bf16, the experts viewed as FP4. ``cfg``: the config to build, the
    checkpoint's cut to its first layers (only their tensors are read); default the checkpoint's. The
    compressed token map is the checkpoint's ``engram_token_map.json``, or built from its tokenizer."""
    model, engram = reference_modules()
    from kernels import ds1017_formats as ops
    from models.ds1017_config import compressed_token_map

    t = (cfg or json.loads((checkpoint / "config.json").read_text()))["text_config"]
    rope = t["rope_scaling"]
    args = model.ModelArgs(
        max_batch_size=rows, max_seq_len=max_seq_len, temperature=0, dtype="fp8", expert_dtype="fp4",
        vocab_size=t["vocab_size"], dim=t["hidden_size"], moe_inter_dim=t["moe_intermediate_size"],
        n_layers=t["num_hidden_layers"], n_mtp_layers=0, n_heads=t["num_attention_heads"],
        n_routed_experts=t["n_routed_experts"], n_shared_experts=1, n_activated_experts=t["num_experts_per_tok"],
        score_func=t["scoring_func"], route_scale=t["routed_scaling_factor"], swiglu_limit=t["swiglu_limit"],
        q_lora_rank=t["q_lora_rank"], head_dim=t["head_dim"], rope_head_dim=t["qk_rope_head_dim"],
        norm_eps=t["rms_norm_eps"], o_groups=t["o_groups"], o_lora_rank=t["o_lora_rank"],
        window_size=t["sliding_window"], compress_ratios=tuple(t["compress_ratios"]),
        kv_source_layers=tuple(t["kv_source_layer_ids"]), index_source_layers=tuple(t["index_source_layer_ids"]),
        compress_rope_theta=t["compress_rope_theta"], original_seq_len=rope["original_max_position_embeddings"],
        rope_theta=t["rope_theta"], rope_factor=rope["factor"], beta_fast=rope["beta_fast"], beta_slow=rope["beta_slow"],
        index_n_heads=t["index_n_heads"], index_head_dim=t["index_head_dim"], index_topk=t["index_topk"],
        candidate_source_layer=t["candidate_source_layer_id"], candidate_topk_blocks=t["candidate_topk_blocks"],
        candidate_block_size=t["candidate_block_size"], hc_mult=t["hc_mult"], hc_sinkhorn_iters=t["hc_sinkhorn_iters"],
        hc_eps=t["hc_eps"], engram_layer_ids=tuple(t["engram_layer_ids"]),
        engram_num_embeddings=tuple(t["engram_num_embeddings"]), engram_max_ngram_size=t["engram_max_ngram_size"],
        engram_vocab_size=t["engram_vocab_size"], engram_n_heads=t["engram_n_heads"],
        engram_head_dim=t["engram_head_dim"], engram_pad_id=t["engram_pad_token_id"],
        engram_compressed_vocab_size=t["engram_compressed_vocab_size"], vision_n_layers=0, dspark_block_size=0)
    if (checkpoint / "engram_token_map.json").is_file():
        lookup = json.loads((checkpoint / "engram_token_map.json").read_text())
    else:
        from tokenizers import Tokenizer

        lookup, _ = compressed_token_map(Tokenizer.from_file(str(checkpoint / "tokenizer.json")))
    engram.build_compressed_token_map = lambda tokenizer: (lookup, max(lookup) + 1)  # pyright: ignore[reportUnknownLambdaType]
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)  # as its generate.py: the model's activations and bf16 parameters
    try:
        with torch.device("cuda"):
            transformer = model.Transformer(args, tokenizer=None)
    finally:
        torch.set_default_dtype(previous)
    state: dict[str, torch.Tensor] = {}
    layers = t["num_hidden_layers"]
    for file in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(str(file), framework="pt", device="cpu") as f:  # pyright: ignore[reportUnknownVariableType]
            for name in f.keys():  # noqa: SIM118
                layer = name.split(".")[1] if name.startswith("layers.") else "-1"
                if not name.startswith(("mtp.", "vision.", "aligner.", "image_")) and not name.endswith("bias_vl") \
                        and int(layer) < layers:
                    state[name] = f.get_tensor(name)
    for name in [n for n in state if n.endswith("attn.wo_a.weight")]:
        scale = state.pop(name[:-len("weight")] + "scale")
        state[name] = ops.dequant_fp8(state[name].view(torch.uint8), scale.view(torch.uint8)).to(torch.bfloat16)
    for name, tensor in state.items():
        if tensor.dtype == torch.int8:
            state[name] = tensor.view(torch.float4_e2m1fn_x2)
    missing, unexpected = transformer.load_state_dict(state, strict=False)
    assert not unexpected and not missing, (missing[:8], unexpected[:8])
    return transformer.eval()


def reference_greedy(transformer: Any, prompt: list[int], steps: int) -> tuple[list[int], list[torch.Tensor]]:
    """The reference's generation: the prompt prefilled in one piece, then one token a step; the
    tokens and each step's logits (fp32)."""
    previous = torch.get_default_dtype()
    # As its generate.py: its kernels allocate their outputs in the default dtype, and it makes its
    # index tensors on the default device.
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            ids = torch.tensor([prompt])
            tokens: list[int] = []
            logits: list[torch.Tensor] = []
            output, step, _ = transformer(ids, 0)
            for i in range(steps):
                logits.append(step[0].clone())
                tokens.append(int(output[0]))
                if i + 1 < steps:
                    output, step, _ = transformer(output.view(1, 1), len(prompt) + i)
        return tokens, logits
    finally:
        torch.set_default_dtype(previous)


def worker(cfg: dict[str, Any], checkpoint: Path, platform: Any = None, **config: Any) -> DSWorker:
    """The engine's DS worker units on the GPU, on the checkpoint."""
    return DSWorker(cfg, platform or TorchEagerPlatform(cuda=True), **{"weights.path": str(checkpoint), **config})
