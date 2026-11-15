# Copyright (c) 2023 DeepSeek
# SPDX-License-Identifier: MIT
"""The DS 1017 definitions (``kernels.ds1017``) on the TileLang kernels of the checkpoint's reference
implementation (``kernel``; LICENSE.deepseek), called the way its ``model.py`` calls them: ``linear()``
for a quantized weight, ``Expert.forward`` and the expert loop of ``MoE.forward``, ``sparse_attn`` with
all the step's queries as one batch row, ``Block.hc_mixes``; and where that implementation calls a
torch product or reduction (``F.linear`` for its unquantized projections and the gate, an einsum for
``wo_a`` and for the indexer's scores, ``mean`` and ``sum`` in its norms, mixes and Engram), the same
call. With these, a step computes what that implementation computes, bit for bit where it runs the
same rows: the last bit of a torch product or reduction follows the number of rows, so a chunk of a
prompt or a batch of rows may differ from a prompt on its own (the references of ``kernels.ds1017``
do not).

The weights arrive as the checkpoint's bytes (``uint8``) and are viewed as FP8, FP4 and ue8m0 here.
The kernels are compiled for bf16 activations, so every activation given must be bf16.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from kernels import ds1017_compose as compose
from kernels import ds1017_ep as ep
from kernels.ds1017_formats import unpack_kv, unpack_latent, unpack_scales
from kernels.tilelang import kernel as K  # pyright: ignore[reportUnknownVariableType]

E8M0 = torch.float8_e8m0fnu
BLOCK = 32


def _bf16(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.bfloat16:
        raise TypeError(f"the TileLang kernels take bf16 activations, not {x.dtype}")
    return x.contiguous()


def fake_quant_fp8(x: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    return K.act_quant(_bf16(x).clone(), block, "ue8m0", E8M0, True)


def fake_quant_fp4(x: torch.Tensor, block: int, e4m3_scale: bool = False) -> torch.Tensor:
    return K.fp4_act_quant(_bf16(x).clone(), block, True, scale_dtype=torch.float8_e4m3fn if e4m3_scale else E8M0)


def _gemm_fp8(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``linear()`` for an FP8 weight: ``act_quant`` and ``fp8_gemm``, the output in bf16 whatever
    torch's default dtype is (the kernel writes bf16)."""
    a, a_s = K.act_quant(_bf16(x), BLOCK, "ue8m0", E8M0)
    n, k = weight.shape
    m = a.numel() // k
    c = torch.empty(*a.shape[:-1], n, dtype=torch.bfloat16, device=a.device)
    K.fp8_gemm_kernel(n, k, block_size=BLOCK, scale_dtype=K.FE8M0)(
        a.view(m, k), weight.contiguous().view(torch.float8_e4m3fn), c.view(m, n), a_s.view(m, -1),
        scale.contiguous().view(E8M0))
    return c


def _gemm_fp4(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``linear()`` for an FP4 weight ``[N, K / 2]`` bytes: ``act_quant`` and ``fp4_gemm``."""
    a, a_s = K.act_quant(_bf16(x), BLOCK, "ue8m0", E8M0)
    n, k = weight.shape[0], weight.shape[1] * 2
    m = a.numel() // k
    c = torch.empty(*a.shape[:-1], n, dtype=torch.bfloat16, device=a.device)
    K.fp4_gemm_kernel(n, k, act_block_size=BLOCK, scale_dtype=K.FE8M0)(
        a.view(m, k), weight.contiguous().view(torch.float4_e2m1fn_x2), c.view(m, n), a_s.view(m, -1),
        scale.contiguous().view(E8M0))
    return c


def linear_fp8(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, packed: torch.Tensor | None = None
               ) -> torch.Tensor:
    return _gemm_fp8(x, weight, scale)


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(x.to(weight.dtype), weight)


def grouped_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bsgd,grd->bsgr", x.unsqueeze(0), weight)[0]


def index_score(q: torch.Tensor, keys: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    score = torch.einsum("bshd,btd->bsht", q, keys)
    return (score.relu_() * weights.unsqueeze(-1)).sum(dim=2)


def top(scores: torch.Tensor, k: int) -> torch.Tensor:
    return scores.topk(k, dim=-1, sorted=False).indices


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    v = x.float()
    return (weight * (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps))).to(x.dtype)


def hc_pre(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    return torch.sum(pre.unsqueeze(-1) * x.float(), dim=1).to(x.dtype)


def hc_post(y: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    out = post.unsqueeze(-1) * y.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=1)
    return out.type_as(y)


def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def pool(members: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    return (members * scores.softmax(dim=1)).sum(dim=1)


def engram_gate(x: torch.Tensor, kv: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor, eps: float
                ) -> torch.Tensor:
    hc, dim = x.shape[1:]
    key, value = kv.split([hc * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc, dim))
    weight = q_weight.float() * k_weight.float()
    h = x.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
    dot = (h * weight * key).sum(-1) * rstd * dim ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


def moe_gate(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, topk: int, route_scale: float
             ) -> tuple[torch.Tensor, torch.Tensor]:
    scores = F.softplus(F.linear(x.float(), weight.float())).sqrt()
    indices = (scores + bias).topk(topk, dim=-1)[1]
    weights = scores.gather(1, indices)
    if topk > 1:
        weights /= weights.sum(dim=-1, keepdim=True) + 1e-20
    weights *= route_scale
    return weights, indices


def sparse_attn(q: torch.Tensor, source: torch.Tensor, idx: torch.Tensor, sink: torch.Tensor, scale: float
                ) -> torch.Tensor:
    out = K.sparse_attn(_bf16(q).unsqueeze(0), _bf16(source).unsqueeze(0), sink, idx.to(torch.int32).contiguous().unsqueeze(0),
                        scale)
    return out[0]


def cache_attn(q: torch.Tensor, window: torch.Tensor, window_idx: torch.Tensor, latents: torch.Tensor | None,
               latent_idx: torch.Tensor | None, sink: torch.Tensor, scale: float) -> torch.Tensor:
    """The reference implementation attends to values: the records are read out of the caches (their
    values are exactly what was quantized into them) and listed in the order they were listed in."""
    keys = unpack_kv(window.flatten(0, 1)[window_idx.clamp_min(0)])
    valid = window_idx >= 0
    if latents is not None and latent_idx is not None:
        keys = torch.cat([keys, unpack_latent(latents.flatten(0, 1)[latent_idx.clamp_min(0)])], dim=1)
        valid = torch.cat([valid, latent_idx >= 0], dim=1)
    tokens, count = valid.shape
    listed = torch.where(valid, torch.arange(count, device=q.device) + count * torch.arange(tokens, device=q.device)[:, None], -1)
    return sparse_attn(q, keys.to(q.dtype).reshape(-1, keys.shape[-1]), listed, sink, scale)


def hc_mixes(x: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, norm_eps: float,
             iters: int, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = x.flatten(1).float()
    mixes = F.linear(flat, fn) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    pre, post, comb = K.hc_split_sinkhorn(mixes.unsqueeze(0).contiguous(), scale, base, x.shape[1], iters, eps)
    return pre[0], post[0], comb[0]


def moe_experts(x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor, w13: torch.Tensor, p13: torch.Tensor,
                w2: torch.Tensor, p2: torch.Tensor, limit: float) -> torch.Tensor:
    y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    inter = w13.shape[1] // 2
    chosen: list[Any] = torch.unique(indices).tolist()
    for i in chosen:
        rows, top = torch.where(indices == i)
        xi = x[rows]
        s13 = unpack_scales(p13[i], 2 * inter, -(-w13.shape[2] * 2 // BLOCK))
        gate = _gemm_fp4(xi, w13[i, :inter], s13[:inter]).float()
        up = _gemm_fp4(xi, w13[i, inter:], s13[inter:]).float()
        if limit > 0:
            up = torch.clamp(up, min=-limit, max=limit)
            gate = torch.clamp(gate, max=limit)
        hidden = weights[rows, top, None] * (F.silu(gate) * up)
        y[rows] += _gemm_fp4(hidden.to(x.dtype), w2[i], unpack_scales(p2[i], w2.shape[1], -(-inter // BLOCK)))
    return y


def ep_experts(group: Any, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor, w13: torch.Tensor,
               p13: torch.Tensor, w2: torch.Tensor, p2: torch.Tensor, limit: float) -> torch.Tensor:
    """``moe_experts`` on ranks that each hold part of the layer's experts (``kernels.ds1017_ep``)."""
    return ep.ep_experts(moe_experts, group, x, weights, indices, w13, p13, w2, p2, limit)


def engram_fetch(group: Any, ids: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    return ep.engram_fetch(group, ids, weight, scale)


# ---------------------------------------------------------------- the compositions over the caches

def rotate(x: torch.Tensor, freqs: torch.Tensor, rotary: int, inverse: bool = False) -> torch.Tensor:
    """``apply_rotary_emb`` on the last ``rotary`` values (the reference implementation rotates in
    place; here a new tensor, the same values)."""
    tail = torch.view_as_complex(x[..., -rotary:].float().unflatten(-1, (-1, 2)))
    f = freqs.conj() if inverse else freqs
    f = f.view(f.shape[0], *([1] * (tail.dim() - 2)), f.shape[-1])
    out = x.clone()
    out[..., -rotary:] = torch.view_as_real(tail * f).flatten(-2).to(x.dtype)
    return out


PIECES = compose.Pieces(rms_norm=rms_norm, rotate=rotate, fake_quant_fp8=fake_quant_fp8, fake_quant_fp4=fake_quant_fp4,
                        index_score=index_score, top=top, pool=pool, linear_fp8=linear_fp8, silu=silu)
"""This implementation's kernels and torch calls, for ``kernels.ds1017_compose``: its norms, its
quantizers, its einsum and sum for the indexer's scores, ``topk``, its pooling."""


def rope(x: torch.Tensor, cos_sin: torch.Tensor, positions: torch.Tensor, rotary: int, inverse: bool = False
         ) -> torch.Tensor:
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
