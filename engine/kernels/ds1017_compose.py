# Copyright (c) 2023 DeepSeek
# SPDX-License-Identifier: MIT
"""The DS 1017 definitions that are several steps of arithmetic over the states' caches, written once
over the pieces a backend supplies (``Pieces``): a store into a cache (norm, RoPE, quantization, the
record), a load out of one, the indexer's choice, the compressor. ``kernels.ds1017`` binds its
batch-invariant references, ``kernels.tilelang`` the reference implementation's own kernels and torch
calls, so that with them a step computes what that implementation computes (``inference/model.py``
of deepseek-ai/DeepSeek-V4.1-Flash-1017; LICENSE.deepseek in ``kernels/ds1017``). ``kernels.deepinfer``
solves the same definitions with DeepInfer's fused operators, whose interfaces these follow.

A library shared by the kernel units, not a unit.

The caches (``kernels.ds1017_formats``):

- a cache is ``[rows, bytes]``, a row one record; ``entries`` records make a page (block), and row
  ``r`` is entry ``r % entries`` of page ``r // entries``. The window's ring of a request is one
  page of ``window`` entries; a source layer's latents and keys are ``page_size / ratio`` a page.
- ``fmt`` names a record's format: ``"fp8"`` (a window KV), ``"latent"`` (a compressed latent),
  ``"key"`` (an indexer key: a page holds its keys' values, then their scales).
- ``rows`` ``-1`` stores nothing; ``segments`` ``[n, 4]`` int32 are DeepInfer's load segments: a page,
  the first and one past the last entry of it, and where in the output they go.

Rotation takes a ``cos_sin`` table ``[positions, rotary]`` fp32 (cosines, then sines: DeepInfer's
``cos_sin_cache``) and the tokens' positions; the rotation itself is ``apply_rotary_emb``'s.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from kernels import ds1017_formats as formats

Fn = Callable[..., Any]
NONE = -1


@dataclass(frozen=True)
class Pieces:
    """The arithmetic a backend brings to the compositions."""

    rms_norm: Fn
    """``(x, weight, eps)``."""
    rotate: Fn
    """``(x, freqs, rotary, inverse)``: ``apply_rotary_emb`` with complex rows ``freqs``."""
    fake_quant_fp8: Fn
    """``(x, block)``."""
    fake_quant_fp4: Fn
    """``(x, block, e4m3_scale)``."""
    index_score: Fn
    """``(q [b, s, h, d], keys [b, L, d], weights [b, s, h])`` -> ``[b, s, L]``."""
    top: Fn
    """``(scores, k)``: the indices of the ``k`` largest, in no order."""
    pool: Fn
    """``(members [g, ratio, dim], scores [g, ratio, dim])`` -> fp32 ``[g, dim]``."""
    linear_fp8: Fn
    """``(x, weight, scale, packed)``."""
    silu: Fn
    """``(x)``."""


# ---------------------------------------------------------------- rotation, quantization, records

def rope(p: Pieces, x: torch.Tensor, cos_sin: torch.Tensor, positions: torch.Tensor, rotary: int,
         inverse: bool = False) -> torch.Tensor:
    half = rotary // 2
    freqs = torch.complex(cos_sin[positions, :half], cos_sin[positions, half:])
    return p.rotate(x, freqs, rotary, inverse)


def quantize(p: Pieces, values: torch.Tensor, fmt: str) -> torch.Tensor:
    """What a record of ``fmt`` holds of ``values``, as values."""
    if fmt == "fp8":
        return p.fake_quant_fp8(values, formats.BLOCK)
    if fmt == "latent":
        return p.fake_quant_fp4(values, formats.LATENT_BLOCK, True)
    if fmt == "key":
        return p.fake_quant_fp4(values, formats.BLOCK, False)
    raise ValueError(f"no record format {fmt!r}")


def transform(p: Pieces, x: torch.Tensor, norm: torch.Tensor | None, eps: float, cos_sin: torch.Tensor | None,
              positions: torch.Tensor | None, rotary: int, fmt: str) -> torch.Tensor:
    """Norm (with ``norm``), rotation (with ``cos_sin``) and the record's quantization, as values."""
    v = x if norm is None else p.rms_norm(x, norm, eps)
    if cos_sin is not None and positions is not None:
        v = rope(p, v, cos_sin, positions, rotary)
    return quantize(p, v, fmt)


def write(cache: torch.Tensor, rows: torch.Tensor, values: torch.Tensor, fmt: str, entries: int) -> None:
    """``values`` (as the format holds them) into their rows. A row of ``-1`` goes to row 0, the
    scratch entry of the padding page, which nothing reads: no shape depends on the values."""
    safe = rows.clamp_min(0)
    if fmt == "fp8":
        cache[safe] = formats.pack_kv(values)
    elif fmt == "latent":
        cache[safe] = formats.pack_latent(values)
    else:
        data, scales = formats.quantize_fp4(values.float())
        held, held_scales = formats.key_sections(cache, entries, values.shape[-1])
        held[safe // entries, safe % entries] = data
        held_scales[safe // entries, safe % entries] = scales


def store_cache(p: Pieces, x: torch.Tensor, norm: torch.Tensor | None, eps: float, cos_sin: torch.Tensor | None,
                positions: torch.Tensor | None, rotary: int, cache: torch.Tensor, rows: torch.Tensor, fmt: str,
                entries: int, normed: bool = False) -> torch.Tensor | None:
    """``fused_norm_rope_hadamard_cast_store_kvcache`` into a cache: the records of ``x`` normed,
    rotated and quantized, at ``rows``; with ``normed`` it returns ``x`` normed (``out_stage="norm"``)."""
    v = x if norm is None else p.rms_norm(x, norm, eps)
    out = v if normed else None
    if cos_sin is not None and positions is not None:
        v = rope(p, v, cos_sin, positions, rotary)
    write(cache, rows, quantize(p, v, fmt), fmt, entries)
    return out


def expand(segments: torch.Tensor, entries: int, total: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The cache rows ``segments`` name, and where each goes in the output (all ``total`` covered)."""
    seg = segments.long()
    lengths = seg[:, 2] - seg[:, 1]
    which = torch.repeat_interleave(torch.arange(seg.shape[0], device=seg.device), lengths, output_size=total)
    within = torch.arange(total, device=seg.device) - (torch.cumsum(lengths, 0) - lengths)[which]
    return seg[which, 0] * entries + seg[which, 1] + within, seg[which, 3] + within


def load_values(cache: torch.Tensor, entries: int, segments: torch.Tensor, total: int, fmt: str) -> torch.Tensor:
    """``load_paged_kv_cache`` of window KV or latent records: their values ``[total, D]`` bf16."""
    src, dst = expand(segments, entries, total)
    unpack = formats.unpack_kv if fmt == "fp8" else formats.unpack_latent
    out = torch.zeros(total, cache.shape[-1] * (32 if fmt == "fp8" else 16) // (33 if fmt == "fp8" else 9),
                      dtype=torch.bfloat16, device=cache.device)
    out[dst] = unpack(cache[src]).to(torch.bfloat16)
    return out


def load_keys(cache: torch.Tensor, entries: int, segments: torch.Tensor, total: int, dim: int
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """``load_paged_kv_cache`` of key records: their values' bytes ``[total, dim / 2]`` and their scales'
    ``[total, dim / 32]``."""
    src, dst = expand(segments, entries, total)
    held, held_scales = formats.key_sections(cache, entries, dim)
    data = torch.zeros(total, dim // 2, dtype=torch.uint8, device=cache.device)
    scales = torch.zeros(total, dim // formats.BLOCK, dtype=torch.uint8, device=cache.device)
    data[dst] = held.flatten(0, 1)[src]
    scales[dst] = held_scales.flatten(0, 1)[src]
    return data, scales


# ---------------------------------------------------------------- the indexer

def quant_index_q(p: Pieces, q: torch.Tensor, cos_sin: torch.Tensor, positions: torch.Tensor, rotary: int
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """The indexer's queries ``[n, heads, dim]`` rotated and quantized to FP4 per 32: their bytes
    ``[n, heads, dim / 2]`` and scales ``[n, heads, dim / 32]``."""
    v = p.fake_quant_fp4(rope(p, q, cos_sin, positions, rotary), formats.BLOCK, False)
    return formats.quantize_fp4(v.float())


def _values(data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return formats.dequant_fp4(data, scales).to(torch.bfloat16)


def candidate_blocks(p: Pieces, scores: torch.Tensor, lens: torch.Tensor, topk_blocks: int, block: int
                     ) -> torch.Tensor:
    """``select_candidate_blocks`` as block ids: of ``scores`` ``[n, L]`` (what a query cannot reach at
    -inf, ``lens`` ``[n, 1]`` how many it reaches) the best ``topk_blocks`` blocks by their best score,
    the one with the newest latent always in; ``[n, topk_blocks]`` ascending, ``-1`` after them."""
    width = scores.size(-1)
    per_block = F.pad(scores, (0, -width % block), value=-torch.inf).unflatten(-1, (-1, block)).amax(dim=-1)
    count = per_block.size(-1)
    newest = (lens - 1) // block
    per_block = per_block.masked_fill(torch.arange(count, device=scores.device) == newest, torch.inf)
    best = p.top(per_block, min(topk_blocks, count))
    best = torch.where(per_block.gather(-1, best) > -torch.inf, best, count).sort(dim=-1).values
    best = torch.where(best < count, best, NONE)
    return F.pad(best, (0, topk_blocks - best.shape[-1]), value=NONE)


def candidate_mask(candidates: torch.Tensor, width: int, block: int) -> torch.Tensor:
    """The positions ``[n, width]`` inside the candidate blocks ``[n, blocks]``."""
    count = -(-width // block)
    keep = torch.zeros(candidates.shape[0], count + 1, dtype=torch.bool, device=candidates.device)
    keep.scatter_(1, torch.where(candidates >= 0, candidates.clamp_max(count), count), True)
    return keep[:, :count].repeat_interleave(block, dim=-1)[:, :width]


def _choose(p: Pieces, score: torch.Tensor, lens: torch.Tensor, topk: int, mode: str, candidates: torch.Tensor | None,
            topk_blocks: int, block: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``Indexer.forward`` after the scores ``[n, L]``: the causal mask, the candidate blocks, the
    ``topk`` (at most) best positions ascending, those a query does not reach ``-1``."""
    width = score.shape[-1]
    score = score.masked_fill(torch.arange(width, device=score.device) >= lens, -torch.inf)
    chosen: torch.Tensor | None = None
    if mode == "source":  # it chooses the blocks, and its own latents from all of them
        chosen = candidate_blocks(p, score, lens, topk_blocks, block)
    elif mode == "user":
        assert candidates is not None
        score = score.masked_fill(~candidate_mask(candidates, width, block), -torch.inf)
    best = p.top(score, min(topk, width)).sort(dim=-1).values
    best = torch.where(best < lens, best, NONE)
    return F.pad(best, (0, topk - best.shape[-1]), value=NONE), chosen


def index_topk(p: Pieces, q_data: torch.Tensor, q_scales: torch.Tensor, weights: torch.Tensor, k_data: torch.Tensor,
               k_scales: torch.Tensor, ks: torch.Tensor, ke: torch.Tensor, longest: int, topk: int, mode: str,
               candidates: torch.Tensor | None, topk_blocks: int, block: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``MQA._topk_prefill``: each query ``i`` over the keys ``[ks[i], ke[i])`` of the loaded keys (its
    row's latents it reaches), the ``topk`` best as indices into the keys ascending, ``-1`` after them;
    for a candidate source also its candidate blocks (of positions in its row). The queries of a
    row are consecutive and share ``ks``; a row is scored over all its latents at once, as that
    implementation scores a sequence, and masked after. ``longest`` bounds any row's latents (a
    kernel may be compiled for it)."""
    q = _values(q_data, q_scales)
    keys = _values(k_data, k_scales)
    n = q.shape[0]
    out = torch.full((n, topk), NONE, dtype=torch.int64, device=q.device)
    picked = torch.full((n, topk_blocks), NONE, dtype=torch.int64, device=q.device) if mode == "source" else None
    starts, ends = ks.tolist(), ke.tolist()
    a = 0
    while a < n:
        b = a
        while b < n and starts[b] == starts[a]:
            b += 1
        first = starts[a]
        width = max(ends[a:b]) - first
        if width > 0:
            score = p.index_score(q[None, a:b], keys[None, first:first + width], weights[None, a:b])[0]
            lens = (ke[a:b] - first)[:, None].long()
            best, chosen = _choose(p, score, lens, topk, mode, None if candidates is None else candidates[a:b],
                                   topk_blocks, block)
            out[a:b] = torch.where(best >= 0, best + first, NONE)
            if picked is not None and chosen is not None:
                picked[a:b] = chosen
        a = b
    return out, picked


def index_topk_paged(p: Pieces, q_data: torch.Tensor, q_scales: torch.Tensor, weights: torch.Tensor,
                     cache: torch.Tensor, entries: int, block_table: torch.Tensor, lens: torch.Tensor, width: int,
                     topk: int, mode: str, candidates: torch.Tensor | None, topk_blocks: int, block: int
                     ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``MQA._topk_decode``: each query over the first ``width`` latents of its row, read through its
    pages ``block_table`` ``[n, pages]``, ``lens`` of them reached; the ``topk`` best as cache rows
    ascending by position, ``-1`` after them; for a candidate source its candidate blocks. The table
    past a row's own pages is not read: it may hold what another request left, pages the cache no
    longer has."""
    dim = q_data.shape[-1] * 2
    j = torch.arange(width, device=cache.device)
    phys = torch.where(j < lens[:, None], block_table.long()[:, j // entries] * entries + j % entries, 0)
    held, held_scales = formats.key_sections(cache, entries, dim)
    keys = _values(held.flatten(0, 1)[phys], held_scales.flatten(0, 1)[phys])
    score = p.index_score(_values(q_data, q_scales)[:, None], keys, weights[:, None])[:, 0]
    best, chosen = _choose(p, score, lens[:, None].long(), topk, mode, candidates, topk_blocks, block)
    return torch.where(best >= 0, phys.gather(1, best.clamp_min(0)), NONE), chosen


# ---------------------------------------------------------------- the compressor

def compress(p: Pieces, score: torch.Tensor, kv: torch.Tensor, state: torch.Tensor, seq_slot: torch.Tensor,
             cu_q: torch.Tensor, cu_k: torch.Tensor, cu_c: torch.Tensor, c_seq: torch.Tensor, ratio: int
             ) -> torch.Tensor:
    """``compress_and_update_state_prefill``: per sequence ``i`` (its tokens ``[cu_q[i], cu_q[i + 1])``
    of the step, ending at position ``cu_k[i + 1] - cu_k[i]``) the groups its tokens complete, in order
    at ``[cu_c[i], ...)``, each its members pooled; members from earlier steps are in the sequence's
    ring ``state[seq_slot[i]]`` (``[slots, ring, 2, dim]``: score, then KV). Rows a sequence reserves
    beyond its groups are zero. Then each sequence's last tokens go into its ring."""
    total = c_seq.shape[0]
    dim, ring = kv.shape[-1], state.shape[1]
    device = kv.device
    i = c_seq.long()
    qs, qe = cu_q.long()[:-1], cu_q.long()[1:]
    klen = (cu_k[1:] - cu_k[:-1]).long()
    qstart = klen - (qe - qs)
    block = qstart[i] // ratio + torch.arange(total, device=device) - cu_c.long()[i]
    real = block < klen[i] // ratio
    at = block[:, None] * ratio + torch.arange(ratio, device=device)
    earlier = at < qstart[i][:, None]
    token = (at - qstart[i][:, None] + qs[i][:, None]).clamp(0, max(kv.shape[0] - 1, 0))
    slot = seq_slot.long()[i][:, None]
    members = torch.where(earlier[..., None], state[slot, at % ring, 1], kv[token])
    scores = torch.where(earlier[..., None], state[slot, at % ring, 0], score[token])
    out = p.pool(members, scores).to(torch.bfloat16) if total else torch.zeros(0, dim, dtype=torch.bfloat16, device=device)
    out = torch.where(real[:, None], out, 0)
    # The last tokens of each sequence into its ring, after the reads above.
    t = torch.arange(kv.shape[0], device=device)
    seq = torch.searchsorted(qe, t, right=True)
    last = t >= qe[seq] - ring
    pos = qstart[seq] + t - qs[seq]
    keep = torch.nonzero(last).flatten()
    rows, cols = seq_slot.long()[seq[keep]], pos[keep] % ring
    state[rows, cols, 0], state[rows, cols, 1] = score[keep], kv[keep]
    return out


def compress_decode(p: Pieces, score: torch.Tensor, kv: torch.Tensor, state: torch.Tensor, slot: torch.Tensor,
                    context_lens: torch.Tensor, ratio: int) -> torch.Tensor:
    """``compress_and_update_state_decode``: each token (at position ``context_lens - 1``) into its
    request's ring, then the group it completes pooled; ``[n, dim]`` bf16, unspecified for a token
    that completes no group."""
    ring = state.shape[1]
    pos = context_lens.long() - 1
    s = slot.long()
    state[s, pos % ring, 0], state[s, pos % ring, 1] = score, kv
    at = (pos // ratio * ratio)[:, None] + torch.arange(ratio, device=kv.device)
    return p.pool(state[s[:, None], at % ring, 1], state[s[:, None], at % ring, 0]).to(torch.bfloat16)


# ---------------------------------------------------------------- the shared expert, Engram's hash

def shared_expert(p: Pieces, x: torch.Tensor, w13: torch.Tensor, s13: torch.Tensor, p13: torch.Tensor, w2: torch.Tensor,
                  s2: torch.Tensor, p2: torch.Tensor, limit: float) -> torch.Tensor:
    """``Expert.forward`` of the shared expert: ``w13`` is its ``w1`` over its ``w3`` (FP8, block scales
    ``s13`` and as a GEMM reads them ``p13``), the SwiGLU with the clamps, ``w2``. In ``x``'s dtype.
    The two halves are two products here, as that implementation computes them."""
    inter = w13.shape[0] // 2
    blocks = inter // formats.BLOCK

    def half(rows: slice, scales: slice) -> torch.Tensor:
        words = formats.packed_shape(inter, w13.shape[1])
        return p.linear_fp8(x, w13[rows], s13[scales], p13[:, rows][:, :words[1]].contiguous()).float()

    gate = half(slice(0, inter), slice(0, blocks))
    up = half(slice(inter, 2 * inter), slice(blocks, 2 * blocks))
    if limit > 0:
        up = up.clamp(-limit, limit)
        gate = gate.clamp(max=limit)
    return p.linear_fp8((p.silu(gate) * up).to(x.dtype), w2, s2, p2)


def engram_hash(tokens: torch.Tensor, multipliers: torch.Tensor, primes: torch.Tensor, offsets: torch.Tensor
                ) -> torch.Tensor:
    """``NgramHashState.forward`` after the look-backs: ``tokens`` ``[n, ngram]`` (the token and the ones
    before it, compressed ids), per Engram layer the products by its multipliers xor-ed up, each
    n-gram's modulo its tables' sizes ``primes`` ``[layers, ngram - 1, heads]``, at the tables'
    ``offsets``: ``[n, layers, (ngram - 1) * heads]`` rows of the layers' tables."""
    products = tokens.long().unsqueeze(1) * multipliers
    rolling = products[..., 0]
    hashes: list[torch.Tensor] = []
    for i in range(1, tokens.shape[1]):
        rolling = torch.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling.unsqueeze(-1) % primes[:, i - 1])
    return torch.cat(hashes, dim=-1) + offsets


# ---------------------------------------------------------------- a sublayer's input, and the MoE

def hc_step(post: Fn, mixes: Fn, pre_fn: Fn, norm: Fn, x: torch.Tensor | None, residual: torch.Tensor, pre: torch.Tensor,
            post_mix: torch.Tensor | None, comb: torch.Tensor | None, fn: torch.Tensor, scale: torch.Tensor,
            base: torch.Tensor, norm_weight: torch.Tensor, norm_eps: float, iters: int, hc_eps: float
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``ds.hc_step@1``: what stands between two sublayers of the Hyper-Connections stream, in the
    order DeepInfer runs it (``HCLayerMixin._hc_fused_post_pre``; its ``mega_mhc`` is all of it in one
    kernel):

    - the sublayer output ``x`` ``[tokens, dim]`` written back to the stream ``residual`` ``[tokens,
      hc, dim]`` with the coefficients ``post_mix`` and ``comb`` the sublayer before it left (``x`` is
      ``None`` where nothing waits: the first sublayer, or the stream was written already);
    - the next coefficients from the stream as it is then (``fn``, ``scale``, ``base``): the ``pre``
      mix the sublayer after this one takes its input with, and this one's own ``post`` and ``comb``;
    - this sublayer's input: the stream's copies mixed with ``pre`` (the mix handed in: the sublayer
      before computed it), normed.

    Returns the input, the stream, and the three new coefficients. The same kernels in the same order
    as the reference implementation's block: only grouped differently."""
    stream = residual if x is None else post(x, residual, post_mix, comb)
    new_pre, new_post, new_comb = mixes(stream, fn, scale, base, norm_eps, iters, hc_eps)
    return norm(pre_fn(stream, pre), norm_weight, norm_eps), stream, new_pre, new_post, new_comb


def slots(to_physical: torch.Tensor, copies: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """The slot that runs each chosen expert ``[tokens, k]`` (``expert.placement@1``'s maps of a
    layer): of an expert's copies, by the token's place in the step."""
    which = torch.arange(indices.shape[0], device=indices.device).unsqueeze(-1) % copies[indices]
    return to_physical[indices, which].long()


def moe(gate: Fn, experts: Fn, ep_experts: Fn, shared: Fn, group: Any, x: torch.Tensor, gate_weight: torch.Tensor,
        gate_bias: torch.Tensor, stacks: Sequence[torch.Tensor], shared_weights: Sequence[torch.Tensor], topk: int,
        route_scale: float, limit: float, to_physical: torch.Tensor | None, copies: torch.Tensor | None,
        taken: torch.Tensor | None) -> torch.Tensor:
    """``ds.moe@1``, ``MoE.forward``: the gate's choice, the routed experts' weighted sum (among
    ranks that share the layer's experts, from the ranks that hold them), and the shared expert; in
    ``x``'s dtype. ``stacks``: the routed experts' ``w13``, its packed scales, ``w2``, its packed
    scales; ``shared_weights``: the shared expert's six tensors. ``to_physical``, ``copies``,
    ``taken``: the layer's maps and counts of ``expert.placement@1``, ``None`` without it."""
    weights, indices = gate(x, gate_weight, gate_bias, topk, route_scale)
    if group.size > 1:
        where = slots(to_physical, copies, indices) if to_physical is not None and copies is not None else None
        y = ep_experts(group, x, weights, indices, *stacks, limit, where, taken)
    else:
        y = experts(x, weights, indices, *stacks, limit)
    y = y + shared(x, *shared_weights, limit)
    return y.to(x.dtype)

