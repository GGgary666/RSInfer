# type: ignore
# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""SGLang's ``fused_sigmoid_gating_delta_rule_update`` (sglang/kernels/ops/attention/linear/
fused_sigmoid_gating_recurrent.py, e838360; LICENSE.sglang) for a GDN verify step (doc/specs/mtp.md):
varlen sequences, ``disable_state_update`` and ``intermediate_states_buffer``. Only the GDN chain path
is kept: no KDA gate or lower bound, no tree mask, no ReplaySSM ring, no CuTe DSL matching, no PDL;
the launch config is its default (``BV = min(next_pow2(V), 32)``, one warp)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log, a, dt_bias, softplus_beta, softplus_threshold, q, k, v, b, o, h0_source, h0_indices, stride_h0_source,
    cu_seqlens, intermediate_states_buffer, intermediate_state_indices, cache_steps, scale, T, stride_a, stride_q,
    stride_k, stride_v, stride_b,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr, DISABLE_STATE_UPDATE: tl.constexpr,
    CACHE_INTERMEDIATE_STATES: tl.constexpr,
):
    i_v, i_n, i_hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_k = 0
    i_h = i_hv // (HV // H)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    all = T
    T = eos - bos

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v
    p_b = b + bos * stride_b + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v
    p_A_log = A_log + i_hv
    p_a = a + bos * stride_a + i_hv
    p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    idx = tl.load(h0_indices + i_n).to(tl.int64)
    if idx >= 0:
        p_h0 = h0_source + idx * stride_h0_source + i_hv * K * V + o_v[None, :] * K + o_k[:, None]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    cache_idx = -1
    if CACHE_INTERMEDIATE_STATES:
        cache_idx = tl.load(intermediate_state_indices + i_n).to(tl.int64)

    step_idx = 0
    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)
        b_A_log = tl.load(p_A_log).to(tl.float32)
        b_a = tl.load(p_a).to(tl.float32)
        b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

        x = b_a + b_dt_bias  # g = -exp(A_log) * softplus(a + dt_bias)
        beta_x = softplus_beta * x
        softplus_x = tl.where(beta_x <= softplus_threshold, (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)), x)
        b_g = -tl.exp(b_A_log) * softplus_x
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        if USE_QK_L2NORM_IN_KERNEL:
            q_sum_sq = tl.sum(b_q * b_q)
            k_sum_sq = tl.sum(b_k * b_k)
            b_q *= 1.0 / tl.sqrt(q_sum_sq + 1e-6)
            b_k *= 1.0 / tl.sqrt(k_sum_sq + 1e-6)
        b_q = b_q * scale
        b_h *= tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        if CACHE_INTERMEDIATE_STATES:
            if cache_idx >= 0:
                cache_ptr = (intermediate_states_buffer + cache_idx * cache_steps * HV * K * V + step_idx * HV * K * V
                             + i_hv * K * V + o_v[None, :] * K + o_k[:, None])
                tl.store(cache_ptr, b_h.to(cache_ptr.dtype.element_ty), mask=mask_h)
        step_idx += 1
        p_q += stride_q
        p_k += stride_k
        p_v += stride_v
        p_b += stride_b
        p_o += HV * V
        p_a += stride_a

    if not DISABLE_STATE_UPDATE:
        if idx >= 0:
            p_h0 = h0_source + idx * stride_h0_source + i_hv * K * V + o_v[None, :] * K + o_k[:, None]
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor, a: torch.Tensor, dt_bias: torch.Tensor, softplus_beta: float, softplus_threshold: float,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b: torch.Tensor, initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor, cu_seqlens: torch.Tensor, scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False, disable_state_update: bool = False,
    intermediate_states_buffer: torch.Tensor | None = None, intermediate_state_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """``q``, ``k`` ``[1, T, H, K]``, ``v`` ``[1, T, HV, V]``, ``a``, ``b`` ``[T, HV]``, sequences by
    ``cu_seqlens``; the pool ``initial_state_source`` ``[slots, HV, V, K]``; ``intermediate_states_buffer``
    ``[lanes, steps, HV, V, K]``. Returns ``[1, T, HV, V]``."""
    _, T, H, K = k.shape
    V, HV = v.shape[-1], v.shape[2]
    N = len(cu_seqlens) - 1
    BV, num_warps = min(triton.next_power_of_2(V), 32), 1
    BK = triton.next_power_of_2(K)
    NV = triton.cdiv(V, BV)
    assert triton.cdiv(K, BK) == 1, "NK > 1 is not supported yet"
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    o = q.new_empty(1, *v.shape)
    cache_steps = intermediate_states_buffer.stride(0) // (HV * K * V) if intermediate_states_buffer is not None else 0
    fused_sigmoid_gating_delta_rule_update_kernel[(NV, N, HV)](
        A_log=A_log, a=a, dt_bias=dt_bias, softplus_beta=softplus_beta, softplus_threshold=softplus_threshold,
        q=q, k=k, v=v, b=b, o=o, h0_source=initial_state_source, h0_indices=initial_state_indices,
        stride_h0_source=initial_state_source.stride(0), cu_seqlens=cu_seqlens,
        intermediate_states_buffer=intermediate_states_buffer, intermediate_state_indices=intermediate_state_indices,
        cache_steps=cache_steps, scale=scale, T=T, stride_a=a.stride()[-2], stride_q=q.stride()[1],
        stride_k=k.stride()[1], stride_v=v.stride()[1], stride_b=b.stride()[-2], H=H, HV=HV, K=K, V=V, BK=BK, BV=BV,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel, DISABLE_STATE_UPDATE=disable_state_update,
        CACHE_INTERMEDIATE_STATES=intermediate_states_buffer is not None, num_warps=num_warps, num_stages=3)
    return o.squeeze(0)
