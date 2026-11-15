"""DS 1017 (DeepSeek-V4.1) for tests: a tiny config with every kind of layer, random checkpoints in the
real checkpoint's formats, and the worker units of this model family in one ``UnitRuntime``
(``Worker`` with other units). The step records come from ``Driver`` (``slot_kind="swa.slots"``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from deploy.models import ds1017 as recipe
from kernels import ds1017_formats as formats
from models.ds1017_config import DSConfig, engram_layout
from tests.units.support.worker import Driver, Worker

DS_UNITS = ("core.kernels", "kernels.reference", "kernels.ds1017", "kernels.tilelang", "kernels.deepinfer",
            "core.state.weights",
            "core.state.holder", "state.kv_paged", "state.swa_slots", "worker.batch", "worker.comm", "policies.eplb",
            "state.expert_placement", "layers.embedding",
            "layers.sampler",
            "layers.ds1017", "models.ds1017", "core.plan", "executors.pytorch")
PAGE_SIZE = 4


def expert_stacks(w1: torch.Tensor, s1: torch.Tensor, w2: torch.Tensor, s2: torch.Tensor, w3: torch.Tensor,
                  s3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Experts' matrices and scales, each stacked over the experts as the checkpoint has them, in the
    model's layout: ``w1`` over ``w3``, ``w2``, and their scales packed (``formats.pack_scales``)."""
    return (torch.cat([w1, w3], dim=1), formats.pack_scales(torch.cat([s1, s3], dim=1)), w2, formats.pack_scales(s2))


def tiny_config(**kw: Any) -> dict[str, Any]:
    """Six layers: one with the window only, a source that compresses two tokens to one and a layer
    reading it, a source of one latent per token that also picks candidate blocks, a layer reading
    it and one with its own indexer; Engram before two of them. The window is 8 tokens, so a prompt
    of a few dozen tokens leaves it and has more latents than an indexer keeps."""
    text: dict[str, Any] = {
        "model_type": "deepseek_v41_text", "vocab_size": 128, "hidden_size": 64, "moe_intermediate_size": 64,
        "num_hidden_layers": 6, "num_attention_heads": 2, "head_dim": 32, "qk_rope_head_dim": 16, "q_lora_rank": 32,
        "o_lora_rank": 32, "o_groups": 2, "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
        "scoring_func": "sqrtsoftplus", "norm_topk_prob": True, "routed_scaling_factor": 1.5, "swiglu_limit": 10.0,
        "rms_norm_eps": 1e-20, "sliding_window": 8, "compress_ratios": [0, 2, 2, 1, 1, 1],
        "kv_source_layer_ids": [1, 3], "index_source_layer_ids": [1, 3, 5], "index_n_heads": 2, "index_head_dim": 32,
        "index_topk": 4, "candidate_source_layer_id": 3, "candidate_topk_blocks": 3, "candidate_block_size": 2,
        "rope_theta": 10000, "compress_rope_theta": 160000,
        "rope_scaling": {"factor": 16, "beta_fast": 32, "beta_slow": 1, "original_max_position_embeddings": 32},
        "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, "engram_layer_ids": [1, 4], "engram_max_ngram_size": 4,
        "engram_vocab_size": 50, "engram_n_heads": 2, "engram_head_dim": 32, "engram_pad_token_id": 2,
        "engram_compressed_vocab_size": 100, **kw}
    if "engram_num_embeddings" not in text:  # as many rows as each layer's buckets take
        primes, offsets, _ = engram_layout(DSConfig.from_hf({**text, "engram_num_embeddings": [0] * len(text["engram_layer_ids"])}))
        text["engram_num_embeddings"] = [o[-1] + p[-1][-1] for o, p in zip(offsets, primes, strict=True)]
    return {"architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41", "text_config": text}


def checkpoint(directory: Path, cfg: dict[str, Any], seed: int = 0) -> Path:
    recipe.write(cfg, directory, seed)
    return directory


class DSWorker(Worker):
    """The DS 1017 worker units on a checkpoint: ``DSWorker(cfg, **{"weights.path": str(directory)})``."""

    UNITS = DS_UNITS
    DEFAULTS: dict[str, Any] = {"model.dtype": "bf16", "state.kv.page_size": PAGE_SIZE, "state.kv.pages": 64,
                                "state.swa.slots": 6, "batch.max_rows": 8, "batch.max_context_len": 64,
                                "dsv4.backend": "reference"}


def driver(worker: Worker) -> Driver:
    """The scheduler's records for the worker, with the pages its KV store holds."""
    return Driver(worker, slot_kind="swa.slots", page_size=worker.service("state.kv.paged@1").page_size)
