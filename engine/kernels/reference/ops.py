"""PyTorch reference kernels for Qwen3.8 (stage2-contract.md §6).

The math follows the HF ``qwen3_5`` model, dtype casts included; state layouts follow sglang:
paged KV is ``[slots, kv_heads, head_dim]`` per layer (``MHATokenToKVPool``, NHD), GDN conv
state ``[conv_dim, kernel - 1]`` and recurrent state ``[v_heads, v_dim, k_dim]`` per slot
(``MambaPool``). Rows are computed one at a time and tokens in order, so a row's result does
not depend on the other rows of the batch.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def gemma_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``norm.gemma_rms@1``: RMS norm in fp32, scaled by ``1 + weight`` (HF ``Qwen3_5RMSNorm``)."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (out * (1.0 + weight.float())).type_as(x)


def rms_norm_gated(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``norm.rms_gated@1``: RMS norm, then ``weight``, then ``silu(gate)`` (HF ``Qwen3_5RMSNormGated``)."""
    dtype = x.dtype
    xf = x.float()
    out = weight * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(dtype)
    return (out * F.silu(gate.float())).to(dtype)


def rope_neox(positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor, rotary_dim: int,
              base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``pos.rope_neox@1``: rotate-half RoPE on the first ``rotary_dim`` of each head.
    ``positions`` ``[T]``; ``q`` ``[T, H, D]``, ``k`` ``[T, Hk, D]``. Text positions are the same
    in all three mrope sections, so Qwen3.8's interleaved mrope is this."""
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.int64, device=q.device).float() / rotary_dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos().to(q.dtype)[:, None, :], emb.sin().to(q.dtype)[:, None, :]

    def rotate(x: torch.Tensor) -> torch.Tensor:
        rot, rest = x[..., :rotary_dim], x[..., rotary_dim:]
        half = rotary_dim // 2
        turned = torch.cat((-rot[..., half:], rot[..., :half]), dim=-1)
        return torch.cat((rot * cos + turned * sin, rest), dim=-1)

    return rotate(q), rotate(k)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """``act.silu_and_mul@1``: ``silu(x[..., :d]) * x[..., d:]`` (sglang ``SiluAndMul``)."""
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """``gemm.linear@1``: ``x @ weight.T + bias``."""
    return F.linear(x, weight, bias)


def embedding(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``embed.lookup@1``."""
    return weight[ids]


def kv_store(k_cache: torch.Tensor, v_cache: torch.Tensor, loc: torch.Tensor, k: torch.Tensor,
             v: torch.Tensor) -> None:
    """``kv.store@1``: write the step's keys and values at their token slots ``loc`` ``[T]``."""
    k_cache[loc] = k.to(k_cache.dtype)
    v_cache[loc] = v.to(v_cache.dtype)


def paged_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                    slots: Sequence[torch.Tensor], query_lens: Sequence[int], scaling: float) -> torch.Tensor:
    """``attn.paged@1``: causal attention of each row's new tokens over its whole sequence.

    ``q`` ``[T, H, D]``, the rows' new tokens in row order; ``slots[r]`` holds the token slots
    of row ``r``'s sequence, new tokens last (``req_to_token[row, :seq_len]``). A decode row has
    one new token. Matmuls in the input dtype, softmax in fp32 (HF eager attention)."""
    out = torch.empty_like(q)
    groups = q.shape[1] // k_cache.shape[1]
    start = 0
    for row_slots, n in zip(slots, query_lens, strict=True):
        seq = int(row_slots.shape[0])
        qs = q[start:start + n].transpose(0, 1)
        ks = k_cache[row_slots].to(q.dtype).repeat_interleave(groups, dim=1).transpose(0, 1)
        vs = v_cache[row_slots].to(q.dtype).repeat_interleave(groups, dim=1).transpose(0, 1)
        scores = torch.matmul(qs, ks.transpose(1, 2)) * scaling
        query_pos = torch.arange(seq - n, seq, device=q.device)[:, None]
        scores = scores.masked_fill(torch.arange(seq, device=q.device)[None, :] > query_pos, float("-inf"))
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out[start:start + n] = torch.matmul(probs, vs).transpose(0, 1)
        start += n
    return out


def causal_conv1d(x: torch.Tensor, state: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """``gdn.conv1d@1``: one row's depthwise causal conv with SiLU, continuing from ``state``.

    ``x`` ``[L, C]``; ``state`` ``[C, K - 1]``, the last inputs before ``x`` (zeros for a new
    request), updated in place to the last ``K - 1`` inputs; ``weight`` ``[C, K]``."""
    seq_len, channels = x.shape
    window = torch.cat((state.to(weight.dtype), x.t().to(weight.dtype)), dim=-1)
    out = F.conv1d(window.unsqueeze(0), weight.unsqueeze(1), bias, groups=channels)[0, :, -seq_len:]
    state.copy_(window[:, -state.shape[-1]:])
    return F.silu(out).t().to(x.dtype)


def gdn_gating(A_log: torch.Tensor, a: torch.Tensor, dt_bias: torch.Tensor,
               b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``gdn.gating@1``: the decay ``g = -exp(A_log) * softplus(a + dt_bias)`` (fp32) and
    ``beta = sigmoid(b)`` per token and value head."""
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    return g, b.sigmoid()


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def gated_delta_rule(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
                     state: torch.Tensor) -> torch.Tensor:
    """``gdn.delta_rule@1``: one row's gated delta rule, token by token, continuing from ``state``.

    ``q``, ``k`` ``[L, Hk, K]``; ``v`` ``[L, HV, V]``; ``g``, ``beta`` ``[L, HV]``; ``state``
    ``[HV, V, K]`` fp32, updated in place. q and k are L2-normalized and q scaled by
    ``K ** -0.5`` (HF ``use_qk_l2norm_in_kernel``); key heads are shared by ``HV // Hk`` value heads."""
    groups = v.shape[1] // k.shape[1]
    qf = _l2norm(q.float().repeat_interleave(groups, dim=1)) * (q.shape[-1] ** -0.5)
    kf = _l2norm(k.float().repeat_interleave(groups, dim=1))
    vf, gf, bf = v.float(), g.float(), beta.float()
    s = state.float()
    out = torch.empty_like(vf)
    for t in range(v.shape[0]):
        s = s * gf[t].exp()[:, None, None]
        delta = (vf[t] - torch.einsum("hvk,hk->hv", s, kf[t])) * bf[t][:, None]
        s = s + delta[:, :, None] * kf[t][:, None, :]
        out[t] = torch.einsum("hvk,hk->hv", s, qf[t])
    state.copy_(s)
    return out.to(v.dtype)


def _mix(seed: int, position: int) -> int:
    """A generator seed from (request seed, position): the same draw however rows are batched."""
    return (seed * 0x9E3779B97F4A7C15 + position * 0xBF58476D1CE4E5B9) & ((1 << 63) - 1)


def sample(logits: torch.Tensor, temperature: Sequence[float], top_p: Sequence[float], top_k: Sequence[int],
           seeds: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
    """``sample.seeded@1``: one token per row of ``logits`` ``[N, vocab]``. Temperature 0 is
    greedy; otherwise top-k (``-1``: off), then top-p, then a draw seeded by (seed, position)."""
    out = torch.empty(logits.shape[0], dtype=torch.int64, device=logits.device)
    for r in range(logits.shape[0]):
        row = logits[r].float()
        if temperature[r] <= 0:
            out[r] = torch.argmax(row)
            continue
        row = row / temperature[r]
        if 0 < top_k[r] < row.shape[0]:
            kth = torch.topk(row, top_k[r]).values[-1]
            row = row.masked_fill(row < kth, float("-inf"))
        probs = torch.softmax(row, dim=-1)
        if top_p[r] < 1.0:
            sorted_probs, order = torch.sort(probs, descending=True)
            drop = sorted_probs.cumsum(-1) - sorted_probs > top_p[r]
            probs = probs.scatter(0, order, sorted_probs.masked_fill(drop, 0.0))
        gen = torch.Generator(device=logits.device).manual_seed(_mix(seeds[r], positions[r]))
        out[r] = torch.multinomial(probs, 1, generator=gen)[0]
    return out
