# Copyright (c) 2023 DeepSeek
# SPDX-License-Identifier: MIT
"""PyTorch reference kernels for DeepSeek-V4.1 (DS 1017), after the reference implementation shipped
with the checkpoint (``inference/kernel.py`` and ``inference/model.py`` of
deepseek-ai/DeepSeek-V4.1-Flash-1017; LICENSE.deepseek). ``kernel.py`` is TileLang; these do the same
arithmetic with torch ops, on any device:

- ``fake_quant_fp8`` / ``fake_quant_fp4``: ``act_quant`` and ``fp4_act_quant`` with ``inplace=True``,
  the fused quantize and dequantize. A quantized value times its scale is exact in bf16 (an e4m3
  value has 4 significant bits, an e2m1 value 2, and a ue8m0 scale is a power of two), so the result
  holds everything the quantized form does.
- ``dequant_fp8`` / ``dequant_fp4``: a quantized weight and its scales, as the checkpoint stores them,
  to fp32. The weights stay in the checkpoint's bytes; a reference GEMM dequantizes what it reads.
- ``linear_fp8`` / ``linear_fp4``: ``linear()`` of ``model.py`` for the two weight formats, the
  activation quantized to FP8 per 32 first: ``fp8_gemm`` and ``fp4_gemm``. Those accumulate one block
  of 32 columns at a time in fp32; these sum a row's products in fp32, another order of the same sum.
- ``sparse_attn``: ``sparse_attn_kernel`` step for step: keys taken by index 64 at a time, a running
  maximum, the weights of a block rounded to bf16 before they multiply the keys, the sink added to
  the denominator. That rounding is why it is not one softmax over all keys.
- ``grouped_linear``: the ``wo_a`` einsum of ``Attention.forward``; ``index_score``: the indexer's
  scores of ``Indexer.forward``.
- ``hc_mixes``: ``Block.hc_mixes`` with ``hc_split_sinkhorn_kernel``; ``hc_pre`` and ``hc_post``.
- ``moe_gate``: ``Gate.forward``; ``moe_experts``: the expert loop of ``MoE.forward`` with
  ``Expert.forward``, over the experts some token chose; ``moe_experts_masked`` the same sum over
  every expert with the others' weights zero, which has fixed shapes (a graph can record it).
- ``rms_norm``, ``rotate``: ``RMSNorm``, ``apply_rotary_emb``; ``rope``: the same at positions of a
  ``cos_sin`` table.
- ``cache_attn``: ``sparse_attn`` over keys still in the states' caches, as their bytes.
- From ``rope`` on, the definitions over the caches (stores, loads, the indexer's choice, the
  compressor): ``kernels.ds1017_compose`` with these references as its pieces.

Batch invariance. Every sum here is ``total``: the elements added in pairs, then the pairs, in an
order the length alone fixes. ``matmul`` multiplies every row of its input on its own, each output
element the ``total`` of its own products. Neither a library GEMM nor a library reduction promises
that: on a GPU, 85% of an fp32 product's elements change in the last bit with the number of rows,
and so does a ``mean`` over the last dimension once there are 16 rows or fewer. This model rounds
to bf16 and to FP8 between its steps and routes each token to the experts with the best scores, so
a last bit that changes in one layer is another token a few layers on. With these references a
token gets the same result in a chunk of a prompt, in a batch with other rows and in a step replayed
from a graph. ``matmul`` costs a product tensor ``[rows, outputs, inputs]`` in pieces of
``MATMUL_BYTES``; the references are not for speed.

On the CPU one more thing follows the size of a tensor: which implementation of ``exp`` or
``sigmoid`` its elements get (a vector one, a scalar one, a math library's), and those round the
last bit differently. ``each`` applies such a function to every row of the first dimension on its
own there, so a row is always computed the same way. On a GPU every element is computed alike.

The number formats and their byte layouts are in ``kernels.ds1017_formats``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from typing import Any

import torch
import torch.nn.functional as F

from kernels import ds1017_compose as compose
from kernels import ds1017_ep as ep
from kernels.ds1017_formats import (BLOCK, FP4_MAX, FP8_MAX, dequant_fp4, dequant_fp8, round_scale, to_fp4,
                                    unpack_kv, unpack_latent, unpack_scales,
                                    to_fp8)

MATMUL_BYTES = 64 << 20
"""``matmul`` works on this many bytes of products at a time."""

def total(v: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """The sum along ``dim`` in an order only its length fixes: neighbours added in pairs, then the
    pairs, the length padded to a power of two with zeros."""
    v = v.movedim(dim, -1)
    size = 1 << max(v.shape[-1] - 1, 0).bit_length()
    if size != v.shape[-1]:
        v = F.pad(v, (0, size - v.shape[-1]))
    while v.shape[-1] > 1:
        v = v[..., 0::2] + v[..., 1::2]
    return v[..., 0]


def each(fn: Callable[[torch.Tensor], torch.Tensor], v: torch.Tensor) -> torch.Tensor:
    """``fn(v)`` for an elementwise function that is not exact (``exp``, ``sigmoid``, ...), each row of
    the first dimension computed as it is on its own (the module docstring's last paragraph)."""
    if v.device.type != "cpu" or not v.shape[0]:
        return fn(v)
    return torch.stack([fn(row.clone()) for row in v])


def fake_quant_fp8(x: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """``act_quant(x, block, "ue8m0", inplace=True)``: FP8 with one power-of-two scale per ``block``
    values along the last dimension, dequantized again, in ``x``'s dtype."""
    v = x.float().unflatten(-1, (-1, block))
    scale = round_scale(v.abs().amax(-1, keepdim=True).clamp_min(1e-4), 1 / FP8_MAX)
    return (to_fp8((v / scale).clamp(-FP8_MAX, FP8_MAX)) * scale).flatten(-2).to(x.dtype)


def fake_quant_fp4(x: torch.Tensor, block: int, e4m3_scale: bool = False) -> torch.Tensor:
    """``fp4_act_quant(x, block, inplace=True)``: FP4 with one scale per ``block`` values, a power of
    two (the indexer's keys and queries) or an e4m3 value (``e4m3_scale``, the compressed KV)."""
    v = x.float().unflatten(-1, (-1, block))
    amax = v.abs().amax(-1, keepdim=True)
    if e4m3_scale:
        scale = to_fp8((amax.clamp_min(6 * 2.0 ** -9) / FP4_MAX).clamp_max(FP8_MAX))
    else:
        scale = round_scale(amax.clamp_min(6 * 2.0 ** -126), 1 / FP4_MAX)
    return (to_fp4((v / scale).clamp(-FP4_MAX, FP4_MAX)) * scale).flatten(-2).to(x.dtype)


def matmul(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` in fp32 for ``x`` ``[..., K]`` and ``weight`` ``[N, K]``, each output element
    summed from its own ``K`` products (the module docstring's batch invariance)."""
    rows = x.reshape(-1, x.shape[-1]).float()
    w = weight.float()
    t, k = rows.shape
    n = w.shape[0]
    out = torch.empty(t, n, dtype=torch.float32, device=x.device)
    cols = max(1, min(n, MATMUL_BYTES // (4 * k)))
    per = max(1, MATMUL_BYTES // (4 * k * cols))
    for a in range(0, t, per):
        for b in range(0, n, cols):
            out[a:a + per, b:b + cols] = total(rows[a:a + per, None, :] * w[None, b:b + cols, :])
    return out.view(*x.shape[:-1], n)


def linear_fp8(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, packed: torch.Tensor | None = None
               ) -> torch.Tensor:
    """``x @ weight.T`` with ``x`` quantized to FP8 per 32 and an FP8 ``weight``; in ``x``'s dtype.
    ``packed`` is the same scales as a GEMM kernel reads them (``pack_scales``), which this does not."""
    return matmul(fake_quant_fp8(x), dequant_fp8(weight, scale)).to(x.dtype)


def linear_fp4(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The same with an FP4 ``weight``."""
    return matmul(fake_quant_fp8(x), dequant_fp4(weight, scale)).to(x.dtype)


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` in the weight's dtype: the model's unquantized projections."""
    return matmul(x.to(weight.dtype), weight).to(weight.dtype)


def grouped_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``einsum("tgd,grd->tgr")``: each group's values ``[tokens, groups, dim]`` through that group's
    own matrix ``[groups, rank, dim]``; in ``x``'s dtype."""
    out = torch.stack([matmul(x[:, g], weight[g]) for g in range(weight.shape[0])], dim=1)
    return out.to(x.dtype)


def index_score(q: torch.Tensor, keys: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """The indexer's scores: ``q`` ``[b, s, heads, dim]`` against ``keys`` ``[b, latents, dim]``, each
    head's products rectified, weighted by ``weights`` ``[b, s, heads]`` and summed over the heads:
    ``[b, s, latents]``, in ``q``'s dtype like the product it comes from."""
    b, s, heads, dim = q.shape
    latents = keys.shape[1]
    qf, kf = q.float(), keys.float()
    score = torch.empty(b, s, heads, latents, dtype=q.dtype, device=q.device)
    per = max(1, MATMUL_BYTES // (4 * dim * heads * max(latents, 1)))
    for a in range(0, s, per):
        score[:, a:a + per] = total(qf[:, a:a + per, :, None, :] * kf[:, None, None, :, :]).to(q.dtype)
    return total(score.relu_() * weights.unsqueeze(-1), dim=2)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    v = x.float()
    return (weight * (v * torch.rsqrt(total(v.square()).unsqueeze(-1) / v.shape[-1] + eps))).to(x.dtype)


def rotate(x: torch.Tensor, freqs: torch.Tensor, rotary: int, inverse: bool = False) -> torch.Tensor:
    """``apply_rotary_emb`` on the last ``rotary`` values of ``x`` ``[tokens, ..., dim]``, ``freqs``
    ``[tokens, rotary / 2]`` the table's rows at the tokens' positions; a new tensor."""
    tail = torch.view_as_complex(x[..., -rotary:].float().unflatten(-1, (-1, 2)))
    f = freqs.conj() if inverse else freqs
    f = f.view(f.shape[0], *([1] * (tail.dim() - 2)), f.shape[-1])
    out = x.clone()
    out[..., -rotary:] = torch.view_as_real(tail * f).flatten(-2).to(x.dtype)
    return out


def sparse_attn(q: torch.Tensor, source: torch.Tensor, idx: torch.Tensor, sink: torch.Tensor, scale: float,
                block: int = 64) -> torch.Tensor:
    """``q`` ``[tokens, heads, dim]`` over the keys ``source[idx]`` (``source`` ``[slots, dim]``, the one
    latent that is both key and value; ``idx`` ``[tokens, keys]``, -1 for none): ``[tokens, heads, dim]``."""
    tokens, heads, dim = q.shape
    qf = q.float()
    out = torch.zeros(tokens, heads, dim, dtype=torch.float32, device=q.device)
    norm = torch.zeros(tokens, heads, dtype=torch.float32, device=q.device)
    high = torch.full((tokens, heads), -1e30, dtype=torch.float32, device=q.device)
    for start in range(0, idx.shape[1], block):
        taken = idx[:, start:start + block]
        valid = taken >= 0
        kv = (source[taken.clamp_min(0)] * valid[..., None]).float()
        scores = (total(qf[:, :, None, :] * kv[:, None, :, :]) * scale).masked_fill(~valid[:, None, :], -torch.inf)
        previous = high
        high = torch.maximum(high, scores.amax(-1))
        decay = each(torch.exp, previous - high)
        weights = each(torch.exp, scores - high[..., None])
        norm = norm * decay + total(weights)
        out = out * decay[..., None] + total(weights.to(q.dtype).float()[..., None] * kv[:, None, :, :], dim=2)
    norm = norm + each(torch.exp, sink - high)
    return (out / norm[..., None]).to(q.dtype)


def cached_keys(window: torch.Tensor, window_idx: torch.Tensor, latents: torch.Tensor | None,
                latent_idx: torch.Tensor | None, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """What ``cache_attn`` attends to, read out of the caches: every token's keys ``[tokens, keys, dim]``
    in ``dtype`` (its window's records, then its latents) and which of them are there."""
    keys = unpack_kv(window.flatten(0, 1)[window_idx.clamp_min(0)])
    valid = window_idx >= 0
    if latents is not None and latent_idx is not None:
        keys = torch.cat([keys, unpack_latent(latents.flatten(0, 1)[latent_idx.clamp_min(0)])], dim=1)
        valid = torch.cat([valid, latent_idx >= 0], dim=1)
    return keys.to(dtype), valid


def cache_attn(q: torch.Tensor, window: torch.Tensor, window_idx: torch.Tensor, latents: torch.Tensor | None,
               latent_idx: torch.Tensor | None, sink: torch.Tensor, scale: float) -> torch.Tensor:
    """``sparse_attn`` on keys still in the states' caches, as their bytes: ``window`` ``[blocks, size,
    D + D / 32]`` holds window KV records (``formats.pack_kv``), ``latents`` ``[blocks, size,
    D / 2 + D / 16]`` compressed latents (``formats.pack_latent``) or is None; ``window_idx`` and
    ``latent_idx`` ``[tokens, keys]`` list each query's records by number (block * size + offset, -1
    for none), as a decode step has them: of the window the missing ones first, of the latents the
    missing ones last. ``[tokens, heads, dim]``."""
    keys, valid = cached_keys(window, window_idx, latents, latent_idx, q.dtype)
    tokens, count = valid.shape
    listed = torch.where(valid, torch.arange(count, device=q.device) + count * torch.arange(tokens, device=q.device)[:, None], -1)
    return sparse_attn(q, keys.reshape(-1, keys.shape[-1]), listed, sink, scale)


def hc_mixes(x: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, norm_eps: float,
             iters: int, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The mixing coefficients of one sublayer from the residual stream ``x`` ``[tokens, hc, dim]``:
    ``pre`` and ``post`` ``[tokens, hc]``, ``comb`` ``[tokens, hc, hc]`` (Sinkhorn), fp32."""
    hc = x.shape[1]
    flat = x.flatten(1).float()
    mixes = matmul(flat, fn) * torch.rsqrt(total(flat.square()).unsqueeze(-1) / flat.shape[-1] + norm_eps)
    pre = each(torch.sigmoid, mixes[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2 * each(torch.sigmoid, mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = (mixes[:, 2 * hc:] * scale[2] + base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = each(torch.exp, comb - comb.amax(-1, keepdim=True))
    comb = comb / total(comb).unsqueeze(-1) + eps
    comb = comb / (total(comb, -2).unsqueeze(-2) + eps)
    for _ in range(iters - 1):
        comb = comb / (total(comb).unsqueeze(-1) + eps)
        comb = comb / (total(comb, -2).unsqueeze(-2) + eps)
    return pre, post, comb


def hc_pre(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """The copies of the stream ``[tokens, hc, dim]`` into one sublayer input ``[tokens, dim]``."""
    return total(pre.unsqueeze(-1) * x.float(), dim=1).to(x.dtype)


def hc_post(y: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """A sublayer's output ``[tokens, dim]`` back to the copies, the residual mixed in through ``comb``."""
    out = post.unsqueeze(-1) * y.unsqueeze(-2) + total(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=1)
    return out.to(y.dtype)


def moe_gate(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, topk: int, route_scale: float
             ) -> tuple[torch.Tensor, torch.Tensor]:
    """``Gate.forward`` with ``sqrtsoftplus`` scores: the bias picks the experts, the weights come from
    the scores without it, normalized over the chosen ones. ``[tokens, topk]`` fp32 weights and indices."""
    scores = each(F.softplus, matmul(x, weight)).sqrt()
    indices = (scores + bias).topk(topk, dim=-1)[1]
    weights = scores.gather(1, indices)
    if topk > 1:
        weights = weights / (total(weights).unsqueeze(-1) + 1e-20)
    return weights * route_scale, indices


def _expert(xq: torch.Tensor, weights: torch.Tensor, w1: torch.Tensor, s1: torch.Tensor, w2: torch.Tensor,
            s2: torch.Tensor, w3: torch.Tensor, s3: torch.Tensor, limit: float, dtype: torch.dtype,
            direct: bool = False) -> torch.Tensor:
    """``Expert.forward`` on activations already quantized (``xq`` fp32): SwiGLU with the clamps.
    ``direct``: the SwiGLU's result is quantized from fp32, not from its rounding to ``dtype``."""
    gate = matmul(xq, dequant_fp4(w1, s1)).to(dtype).float()
    up = matmul(xq, dequant_fp4(w3, s3)).to(dtype).float()
    if limit > 0:
        up = up.clamp(-limit, limit)
        gate = gate.clamp(max=limit)
    hidden = weights * (each(F.silu, gate) * up)
    return matmul(fake_quant_fp8(hidden if direct else hidden.to(dtype)), dequant_fp4(w2, s2)).to(dtype)


def _halves(w13: torch.Tensor, p13: torch.Tensor, w2: torch.Tensor, p2: torch.Tensor, i: int
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expert ``i``'s three matrices and their scales by row, from the stacks."""
    inter = w13.shape[1] // 2
    s13 = unpack_scales(p13[i], w13.shape[1], -(-w13.shape[2] * 2 // BLOCK))
    s2 = unpack_scales(p2[i], w2.shape[1], -(-w2.shape[2] * 2 // BLOCK))
    return w13[i, :inter], s13[:inter], w2[i], s2, w13[i, inter:], s13[inter:]


def moe_experts(x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor, w13: torch.Tensor, p13: torch.Tensor,
                w2: torch.Tensor, p2: torch.Tensor, limit: float, *, direct: bool = False) -> torch.Tensor:
    """The routed experts' sum ``[tokens, dim]`` fp32: each chosen expert on the tokens that chose it, in
    the experts' order. The weights are stacked over the experts: ``w13`` ``[experts, 2 * inter, dim / 2]``
    is each expert's ``w1`` over its ``w3``, ``w2`` ``[experts, dim, inter / 2]``; ``p13`` and ``p2`` are
    their scales as ``pack_scales`` lays them out.

    ``direct`` is not part of the definition: it computes what DeepInfer's kernels compute, which
    quantize the SwiGLU's fp32 result where the reference implementation rounds it to bf16 first
    (about one value in sixteen then lands on the next FP8 value). The tests of those kernels compare
    with it."""
    y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    xq = fake_quant_fp8(x).float()
    for i in torch.unique(indices).tolist():
        rows, top = torch.where(indices == i)
        y[rows] += _expert(xq[rows], weights[rows, top, None], *_halves(w13, p13, w2, p2, i), limit, x.dtype, direct)
    return y


def ep_experts(group: Any, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor, w13: torch.Tensor,
               p13: torch.Tensor, w2: torch.Tensor, p2: torch.Tensor, limit: float, slots: torch.Tensor | None = None,
               taken: torch.Tensor | None = None) -> torch.Tensor:
    """``moe_experts`` on ranks that each hold part of the layer's experts (``kernels.ds1017_ep``)."""
    return ep.ep_experts(moe_experts, group, x, weights, indices, w13, p13, w2, p2, limit, slots, taken)


def engram_fetch(group: Any, ids: torch.Tensor, weights: Sequence[torch.Tensor], scales: Sequence[torch.Tensor]
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows of an Engram table split over the ranks, as bytes and scales (``kernels.ds1017_ep``)."""
    return ep.engram_fetch(group, ids, weights, scales)


def moe_experts_masked(x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor, w13: torch.Tensor,
                       p13: torch.Tensor, w2: torch.Tensor, p2: torch.Tensor, limit: float) -> torch.Tensor:
    """``moe_experts`` with fixed shapes: every expert runs on every token, with weight 0 where the token
    did not choose it, which adds exactly 0."""
    y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    xq = fake_quant_fp8(x).float()
    for i in range(w13.shape[0]):
        chosen = total(weights * (indices == i)).unsqueeze(-1)
        y += _expert(xq, chosen, *_halves(w13, p13, w2, p2, i), limit, x.dtype)
    return y


def top(scores: torch.Tensor, k: int) -> torch.Tensor:
    """The indices of the ``k`` largest along the last dimension, in no order; of equal scores the
    earlier ones. ``torch.topk`` leaves that choice open, and makes it differently for another
    width: the indexer's scores are bf16 sums of rectified products, where equal scores are common."""
    return scores.sort(dim=-1, descending=True, stable=True).indices[..., :k]


def silu(x: torch.Tensor) -> torch.Tensor:
    return each(F.silu, x)


def pool(members: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """``Compressor``'s pooling: each group's members ``[groups, ratio, dim]`` weighted by the softmax
    of their scores over the group; fp32 ``[groups, dim]``."""
    weights = each(torch.exp, scores - scores.amax(dim=1, keepdim=True))
    return total(members * (weights / total(weights, dim=1).unsqueeze(1)), dim=1)


def engram_gate(x: torch.Tensor, kv: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor, eps: float
                ) -> torch.Tensor:
    """``Engram.forward`` after the lookup: ``kv`` ``[tokens, (hc + 1) * dim]`` is one key per copy of
    the stream ``x`` ``[tokens, hc, dim]`` and a shared value; the value is added to each copy through
    a gate, the sigmoid of the signed root of the normalized dot product of copy and key."""
    hc, dim = x.shape[1:]
    key, value = kv.split([hc * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc, dim))
    weight = q_weight.float() * k_weight.float()
    h = x.float()
    rstd = torch.rsqrt(total(h.square()) / dim + eps) * torch.rsqrt(total(key.square()) / dim + eps)
    dot = total(h * weight * key) * rstd * dim ** -0.5
    gate = each(torch.sigmoid, torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)



# ---------------------------------------------------------------- the compositions over the caches

PIECES = compose.Pieces(rms_norm=rms_norm, rotate=rotate, fake_quant_fp8=fake_quant_fp8, fake_quant_fp4=fake_quant_fp4,
                        index_score=index_score, top=top, pool=pool, linear_fp8=linear_fp8, silu=silu)
"""The references above, for ``kernels.ds1017_compose``."""


def rope(x: torch.Tensor, cos_sin: torch.Tensor, positions: torch.Tensor, rotary: int, inverse: bool = False
         ) -> torch.Tensor:
    """``apply_rotary_emb`` on the last ``rotary`` values of ``x`` ``[tokens, ..., dim]`` at the tokens'
    ``positions`` of the table ``cos_sin`` ``[positions, rotary]`` (cosines, then sines); a new tensor."""
    return compose.rope(PIECES, x, cos_sin, positions, rotary, inverse)


def transform(x: torch.Tensor, norm: torch.Tensor | None, eps: float, cos_sin: torch.Tensor | None,
              positions: torch.Tensor | None, rotary: int, fmt: str) -> torch.Tensor:
    return compose.transform(PIECES, x, norm, eps, cos_sin, positions, rotary, fmt)


def store_cache(x: torch.Tensor, norm: torch.Tensor | None, eps: float, cos_sin: torch.Tensor | None,
                positions: torch.Tensor | None, rotary: int, cache: torch.Tensor, rows: torch.Tensor, fmt: str,
                entries: int, normed: bool = False) -> torch.Tensor | None:
    return compose.store_cache(PIECES, x, norm, eps, cos_sin, positions, rotary, cache, rows, fmt, entries, normed)


def load_values(cache: torch.Tensor, entries: int, segments: torch.Tensor, total: int, fmt: str) -> torch.Tensor:
    return compose.load_values(cache, entries, segments, total, fmt)


def load_keys(cache: torch.Tensor, entries: int, segments: torch.Tensor, total: int, dim: int
              ) -> tuple[torch.Tensor, torch.Tensor]:
    return compose.load_keys(cache, entries, segments, total, dim)


def quant_index_q(q: torch.Tensor, cos_sin: torch.Tensor, positions: torch.Tensor, rotary: int
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    return compose.quant_index_q(PIECES, q, cos_sin, positions, rotary)


def index_topk(q_data: torch.Tensor, q_scales: torch.Tensor, weights: torch.Tensor, k_data: torch.Tensor,
               k_scales: torch.Tensor, ks: torch.Tensor, ke: torch.Tensor, longest: int, topk: int, mode: str,
               candidates: torch.Tensor | None, topk_blocks: int, block: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    return compose.index_topk(PIECES, q_data, q_scales, weights, k_data, k_scales, ks, ke, longest, topk, mode,
                              candidates, topk_blocks, block)


def index_topk_paged(q_data: torch.Tensor, q_scales: torch.Tensor, weights: torch.Tensor, cache: torch.Tensor,
                     entries: int, block_table: torch.Tensor, lens: torch.Tensor, width: int, topk: int, mode: str,
                     candidates: torch.Tensor | None, topk_blocks: int, block: int
                     ) -> tuple[torch.Tensor, torch.Tensor | None]:
    return compose.index_topk_paged(PIECES, q_data, q_scales, weights, cache, entries, block_table, lens, width, topk,
                                    mode, candidates, topk_blocks, block)


def compress(score: torch.Tensor, kv: torch.Tensor, state: torch.Tensor, seq_slot: torch.Tensor, cu_q: torch.Tensor,
             cu_k: torch.Tensor, cu_c: torch.Tensor, c_seq: torch.Tensor, ratio: int) -> torch.Tensor:
    return compose.compress(PIECES, score, kv, state, seq_slot, cu_q, cu_k, cu_c, c_seq, ratio)


def compress_decode(score: torch.Tensor, kv: torch.Tensor, state: torch.Tensor, slot: torch.Tensor,
                    context_lens: torch.Tensor, ratio: int) -> torch.Tensor:
    return compose.compress_decode(PIECES, score, kv, state, slot, context_lens, ratio)


def shared_expert(x: torch.Tensor, w13: torch.Tensor, s13: torch.Tensor, p13: torch.Tensor, w2: torch.Tensor,
                  s2: torch.Tensor, p2: torch.Tensor, limit: float) -> torch.Tensor:
    return compose.shared_expert(PIECES, x, w13, s13, p13, w2, s2, p2, limit)


def engram_hash(tokens: torch.Tensor, multipliers: torch.Tensor, primes: torch.Tensor, offsets: torch.Tensor
                ) -> torch.Tensor:
    return compose.engram_hash(tokens, multipliers, primes, offsets)

def hc_step(x: torch.Tensor | None, residual: torch.Tensor, pre: torch.Tensor, post: torch.Tensor | None,
            comb: torch.Tensor | None, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, norm_weight: torch.Tensor,
            norm_eps: float, iters: int, hc_eps: float, stage: Any = None
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """What stands between two sublayers of the stream (``compose.hc_step``); ``stage`` is for a backend
    whose MoE reads its input from where this wrote it."""
    return compose.hc_step(hc_post, hc_mixes, hc_pre, rms_norm, x, residual, pre, post, comb, fn, scale, base,
                           norm_weight, norm_eps, iters, hc_eps)


def moe(group: Any, x: torch.Tensor, gate_weight: torch.Tensor, gate_bias: torch.Tensor, gate_bf16: torch.Tensor,
        w13: torch.Tensor, p13: torch.Tensor, w2: torch.Tensor, p2: torch.Tensor, sw13: torch.Tensor, ss13: torch.Tensor,
        sp13: torch.Tensor, sw2: torch.Tensor, ss2: torch.Tensor, sp2: torch.Tensor, topk: int, route_scale: float,
        limit: float, to_physical: torch.Tensor | None = None, copies: torch.Tensor | None = None,
        taken: torch.Tensor | None = None) -> torch.Tensor:
    """A layer's MoE on its normed input (``compose.moe``); ``gate_bf16`` is the gate's weight as a
    fused gate kernel takes it, not read here."""
    return compose.moe(moe_gate, moe_experts, ep_experts, shared_expert, group, x, gate_weight, gate_bias,
                       (w13, p13, w2, p2), (sw13, ss13, sp13, sw2, ss2, sp2), topk, route_scale, limit, to_physical,
                       copies, taken)


def moe_fixed(group: Any, x: torch.Tensor, gate_weight: torch.Tensor, gate_bias: torch.Tensor, gate_bf16: torch.Tensor,
              w13: torch.Tensor, p13: torch.Tensor, w2: torch.Tensor, p2: torch.Tensor, sw13: torch.Tensor,
              ss13: torch.Tensor, sp13: torch.Tensor, sw2: torch.Tensor, ss2: torch.Tensor, sp2: torch.Tensor, topk: int,
              route_scale: float, limit: float, to_physical: torch.Tensor | None = None,
              copies: torch.Tensor | None = None, taken: torch.Tensor | None = None) -> torch.Tensor:
    """``moe`` with the routed experts' fixed-shape form (``moe_experts_masked``): for a step replayed
    from a graph, on one rank."""
    if group.size > 1:
        raise ValueError("the references' MoE among ranks is not recorded: its experts travel through the host")
    return compose.moe(moe_gate, moe_experts_masked, ep_experts, shared_expert, group, x, gate_weight, gate_bias,
                       (w13, p13, w2, p2), (sw13, ss13, sp13, sw2, ss2, sp2), topk, route_scale, limit, to_physical,
                       copies, taken)

