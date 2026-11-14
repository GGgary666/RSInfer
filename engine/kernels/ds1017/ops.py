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
- ``rms_norm``, ``rope_table``, ``rope``: ``RMSNorm``, ``precompute_freqs_cis``, ``apply_rotary_emb``.

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

from collections.abc import Callable

import torch
import torch.nn.functional as F

from kernels.ds1017_formats import (BLOCK, FP4_MAX, FP8_MAX, dequant_fp4, dequant_fp8, round_scale, to_fp4,
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


def linear_fp8(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` with ``x`` quantized to FP8 per 32 and an FP8 ``weight``; in ``x``'s dtype."""
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


def rope(x: torch.Tensor, freqs: torch.Tensor, rotary: int, inverse: bool = False) -> torch.Tensor:
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
            s2: torch.Tensor, w3: torch.Tensor, s3: torch.Tensor, limit: float, dtype: torch.dtype) -> torch.Tensor:
    """``Expert.forward`` on activations already quantized (``xq`` fp32): SwiGLU with the clamps."""
    gate = matmul(xq, dequant_fp4(w1, s1)).to(dtype).float()
    up = matmul(xq, dequant_fp4(w3, s3)).to(dtype).float()
    if limit > 0:
        up = up.clamp(-limit, limit)
        gate = gate.clamp(max=limit)
    hidden = (weights * (each(F.silu, gate) * up)).to(dtype)
    return matmul(fake_quant_fp8(hidden), dequant_fp4(w2, s2)).to(dtype)


def moe_experts(x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor, w1: torch.Tensor, s1: torch.Tensor,
                w2: torch.Tensor, s2: torch.Tensor, w3: torch.Tensor, s3: torch.Tensor, limit: float) -> torch.Tensor:
    """The routed experts' sum ``[tokens, dim]`` fp32: each chosen expert on the tokens that chose it, in
    the experts' order. The weights are stacked over the experts (``w1`` ``[experts, inter, dim / 2]``...)."""
    y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    xq = fake_quant_fp8(x).float()
    for i in torch.unique(indices).tolist():
        rows, top = torch.where(indices == i)
        y[rows] += _expert(xq[rows], weights[rows, top, None], w1[i], s1[i], w2[i], s2[i], w3[i], s3[i], limit, x.dtype)
    return y


def moe_experts_masked(x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor, w1: torch.Tensor,
                       s1: torch.Tensor, w2: torch.Tensor, s2: torch.Tensor, w3: torch.Tensor, s3: torch.Tensor,
                       limit: float) -> torch.Tensor:
    """``moe_experts`` with fixed shapes: every expert runs on every token, with weight 0 where the token
    did not choose it, which adds exactly 0."""
    y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    xq = fake_quant_fp8(x).float()
    for i in range(w1.shape[0]):
        chosen = total(weights * (indices == i)).unsqueeze(-1)
        y += _expert(xq, chosen, w1[i], s1[i], w2[i], s2[i], w3[i], s3[i], limit, x.dtype)
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

