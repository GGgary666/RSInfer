# Copyright (c) 2023 DeepSeek
# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 (DS 1017): the config fields the text model reads, what its layers store, and the
Engram hash layout. A library shared by ``models.ds1017``, ``layers.ds1017`` and the deploy recipe
(not a unit). After the reference implementation shipped with the checkpoint (``inference/model.py``,
``inference/engram.py`` of deepseek-ai/DeepSeek-V4.1-Flash-1017; LICENSE.deepseek in
``kernels/ds1017``); doc/specs/v1-ds1017.md describes the model.
"""

from __future__ import annotations

from typing import Any

import msgspec

from core.contracts.state import StateDecl
from kernels.ds1017_formats import BLOCK, LATENT_BLOCK

KV_KIND, SLOT_KIND = "kv.paged", "swa.slots"


class DSConfig(msgspec.Struct, frozen=True):
    vocab_size: int
    hidden_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    qk_rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    n_routed_experts: int
    num_experts_per_tok: int
    routed_scaling_factor: float
    swiglu_limit: float
    rms_norm_eps: float
    sliding_window: int
    compress_ratios: tuple[int, ...]
    """One per layer (the draft layers after the backbone's included): 0, the sliding window only; r,
    the window and compressed KV, r tokens to a latent."""
    kv_source_layer_ids: tuple[int, ...]
    """The layers that compress their own KV; each layer after one, up to the next, reads its cache."""
    index_source_layer_ids: tuple[int, ...]
    """The layers that run an indexer; each layer after one, up to the next, attends to what it chose."""
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    candidate_source_layer_id: int
    """Its indexer first keeps ``candidate_topk_blocks`` blocks of ``candidate_block_size`` latents; the
    indexers after it choose inside them. Negative: off."""
    candidate_topk_blocks: int
    candidate_block_size: int
    rope_theta: float
    compress_rope_theta: float
    rope_factor: float
    rope_beta_fast: int
    rope_beta_slow: int
    rope_original_seq_len: int
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    engram_layer_ids: tuple[int, ...]
    engram_num_embeddings: tuple[int, ...]
    engram_max_ngram_size: int
    engram_vocab_size: int
    engram_n_heads: int
    engram_head_dim: int
    engram_pad_token_id: int
    engram_compressed_vocab_size: int

    @staticmethod
    def from_hf(raw: dict[str, Any]) -> DSConfig:
        """A text config, or the ``text_config`` of a ``DeepseekV41ForCausalLM`` one."""
        d = raw.get("text_config", raw)
        rope = d.get("rope_scaling") or {}
        scoring, shared = d.get("scoring_func", "sqrtsoftplus"), d.get("n_shared_experts", 1)
        if scoring != "sqrtsoftplus" or shared != 1 or not d.get("norm_topk_prob", True):
            raise ValueError(f"DS 1017 routes with sqrtsoftplus scores, normalized, and one shared expert; "
                             f"the config has {scoring}, {shared} shared, norm_topk_prob {d.get('norm_topk_prob')}")
        return DSConfig(
            vocab_size=d["vocab_size"], hidden_size=d["hidden_size"], moe_intermediate_size=d["moe_intermediate_size"],
            num_hidden_layers=d["num_hidden_layers"], num_attention_heads=d["num_attention_heads"],
            head_dim=d["head_dim"], qk_rope_head_dim=d["qk_rope_head_dim"], q_lora_rank=d["q_lora_rank"],
            o_lora_rank=d["o_lora_rank"], o_groups=d["o_groups"], n_routed_experts=d["n_routed_experts"],
            num_experts_per_tok=d["num_experts_per_tok"], routed_scaling_factor=d.get("routed_scaling_factor", 1.0),
            swiglu_limit=d.get("swiglu_limit", 0.0), rms_norm_eps=d.get("rms_norm_eps", 1e-20),
            sliding_window=d["sliding_window"], compress_ratios=tuple(d["compress_ratios"]),
            kv_source_layer_ids=tuple(d.get("kv_source_layer_ids", ())),
            index_source_layer_ids=tuple(d.get("index_source_layer_ids", ())),
            index_n_heads=d.get("index_n_heads", 0), index_head_dim=d.get("index_head_dim", 0),
            index_topk=d.get("index_topk", 0), candidate_source_layer_id=d.get("candidate_source_layer_id", -1),
            candidate_topk_blocks=d.get("candidate_topk_blocks", 0), candidate_block_size=d.get("candidate_block_size", 0),
            rope_theta=d.get("rope_theta", 10000), compress_rope_theta=d.get("compress_rope_theta", 10000),
            rope_factor=rope.get("factor", 1), rope_beta_fast=rope.get("beta_fast", 32),
            rope_beta_slow=rope.get("beta_slow", 1),
            rope_original_seq_len=rope.get("original_max_position_embeddings", 0),
            hc_mult=d.get("hc_mult", 4), hc_sinkhorn_iters=d.get("hc_sinkhorn_iters", 20), hc_eps=d.get("hc_eps", 1e-6),
            engram_layer_ids=tuple(d.get("engram_layer_ids", ())),
            engram_num_embeddings=tuple(d.get("engram_num_embeddings", ())),
            engram_max_ngram_size=d.get("engram_max_ngram_size", 1), engram_vocab_size=d.get("engram_vocab_size", 0),
            engram_n_heads=d.get("engram_n_heads", 0), engram_head_dim=d.get("engram_head_dim", 0),
            engram_pad_token_id=d.get("engram_pad_token_id", 2),
            engram_compressed_vocab_size=d.get("engram_compressed_vocab_size", 0))

    def ratio(self, layer: int) -> int:
        return self.compress_ratios[layer]

    def kv_source(self, layer: int) -> int:
        """The layer whose compressed KV ``layer`` attends over; -1 for a window-only layer."""
        if not self.ratio(layer):
            return -1
        sources = [s for s in self.kv_source_layer_ids if s <= layer and s < self.num_hidden_layers]
        if not sources or self.ratio(sources[-1]) != self.ratio(layer):
            raise ValueError(f"layer {layer} compresses {self.ratio(layer)} to 1 but no KV source before it does")
        return sources[-1]

    def index_source(self, layer: int) -> int:
        """The layer whose indexer chose the latents ``layer`` attends to; -1 for a window-only layer."""
        if not self.ratio(layer):
            return -1
        sources = [s for s in self.index_source_layer_ids if s <= layer and s < self.num_hidden_layers]
        if not sources or sources[-1] < self.kv_source(layer):
            raise ValueError(f"layer {layer} has no indexer between its KV source and itself")
        return sources[-1]

    @property
    def engram_hash_cols(self) -> int:
        """Hash ids per token and Engram layer: one per n-gram size above 1 and head."""
        return (self.engram_max_ngram_size - 1) * self.engram_n_heads

    def check(self) -> None:
        """What the layers assume of a config, found at start instead of in a kernel."""
        n = self.num_hidden_layers
        if len(self.compress_ratios) < n:
            raise ValueError("compress_ratios needs one entry per layer")
        for layer in range(n):
            _ = self.kv_source(layer), self.index_source(layer)
        if any(s not in self.index_source_layer_ids for s in self.kv_source_layer_ids if s < n):
            raise ValueError("every KV source layer runs an indexer (its keys come from that layer's latents)")
        for dims in (self.hidden_size, self.head_dim, self.q_lora_rank, self.o_lora_rank, self.moe_intermediate_size,
                     self.index_head_dim or 32, self.engram_head_dim or 32):
            if dims % 32:
                raise ValueError(f"the quantized dimensions are multiples of 32, not {dims}")
        if self.engram_layer_ids and len(self.engram_num_embeddings) != len(self.engram_layer_ids):
            raise ValueError("engram_num_embeddings needs one entry per Engram layer")


def states(cfg: DSConfig, page_size: int) -> list[StateDecl]:
    """What each layer stores, as bytes in DeepInfer's KV cache formats (``formats``' records; its
    kernels read them): per KV source layer the compressed latents ``kv`` (FP4 records) and the
    indexer's keys ``ik`` (FP4, a page's values and then its scales), paged, ``page_size / ratio``
    of them a page; per layer a slot with the window's ring (FP8 records), for a source that
    compresses by more than one the group still filling (``cstate``), and at the first Engram layer
    the last tokens an n-gram reaches back to (``tail``)."""
    out: list[StateDecl] = []
    engram_first = min(cfg.engram_layer_ids) if cfg.engram_layer_ids else -1
    for layer in range(cfg.num_hidden_layers):
        ratio, source = cfg.ratio(layer), layer in cfg.kv_source_layer_ids
        if source:
            latent = cfg.head_dim // 2 + cfg.head_dim // LATENT_BLOCK
            key = cfg.index_head_dim // 2 + cfg.index_head_dim // BLOCK
            per = page_size // ratio  # DeepInfer's block_size // ratio
            out.append(StateDecl(KV_KIND, layer, {"kv": ((per, latent), "u8"), "ik": ((per, key), "u8")},
                                 "request", tokens_per_item=page_size))
        slot: dict[str, tuple[tuple[int, ...], str]] = {
            "ring": ((cfg.sliding_window, cfg.head_dim + cfg.head_dim // BLOCK), "u8")}
        if source and ratio > 1:  # DeepInfer's ca_state: each member's score, then its KV
            slot["cstate"] = ((ratio, 2, cfg.head_dim), "f32")
        if layer == engram_first and cfg.engram_max_ngram_size > 1:
            slot["tail"] = ((cfg.engram_max_ngram_size - 1,), "i64")
        out.append(StateDecl(SLOT_KIND, layer, slot, "slot"))
    return out


def engram_layout(cfg: DSConfig) -> tuple[list[list[list[int]]], list[list[int]], list[list[int]]]:
    """``EngramLayout`` and ``compute_hash_multipliers``: per Engram layer the bucket modulus of every
    (n-gram size, head), each bucket range's offset in the layer's table, and one odd multiplier per
    look-back. The primes are drawn in order and never reused; the multipliers come from a
    per-layer generator and the size of the compressed vocabulary."""
    import numpy as np
    from sympy import isprime

    primes: list[list[list[int]]] = []
    seen: set[int] = set()
    for _ in cfg.engram_layer_ids:
        per_ngram: list[list[int]] = []
        for _ in range(cfg.engram_max_ngram_size - 1):
            sizes: list[int] = []
            current = cfg.engram_vocab_size - 1
            for _ in range(cfg.engram_n_heads):
                current += 1
                while not isprime(current) or current in seen:
                    current += 1
                seen.add(current)
                sizes.append(current)
            per_ngram.append(sizes)
        primes.append(per_ngram)
    offsets: list[list[int]] = []
    for layer in primes:
        flat = [p for per_ngram in layer for p in per_ngram]
        offsets.append([int(v) for v in np.cumsum([0, *flat[:-1]])])
    bound = max(1, (int(np.iinfo(np.int64).max) // cfg.engram_compressed_vocab_size) // 2)
    multipliers: list[list[int]] = []
    for layer_id in cfg.engram_layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(low=0, high=bound, size=(cfg.engram_max_ngram_size,), dtype=np.int64)
        multipliers.append([int(v) * 2 + 1 for v in values])
    return primes, offsets, multipliers


def compressed_token_map(tokenizer: Any) -> tuple[list[int], int]:
    """``build_compressed_token_map``: every token id onto a smaller id space where tokens that
    normalize alike collapse together, and that space's size. ``tokenizer``: a ``tokenizers.Tokenizer``."""
    from tokenizers import Regex, normalizers

    sentinel = ""
    normalizer = normalizers.Sequence([  # pyright: ignore[reportCallIssue]
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " ")])
    key_to_new: dict[str, int] = {}
    size = tokenizer.get_vocab_size(with_added_tokens=True)
    lookup = [0] * size
    for token_id in range(size):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = tokenizer.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        lookup[token_id] = key_to_new.setdefault(key, len(key_to_new))
    return lookup, len(key_to_new)
