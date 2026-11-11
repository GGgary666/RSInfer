# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""The GDN definitions, and the fused glue of an attention layer, on SGLang's Triton kernels
(e838360; LICENSE.sglang).

Each function takes its definition's arguments (``kernels.reference``) and makes the call that
SGLang makes: ``GDNAttnBackend`` and ``TritonGDNKernel`` (srt/layers/attention/linear/
gdn_backend.py, kernels/gdn_triton.py) for the GDN, ``Qwen3_5AttentionDecoderLayer``
(srt/models/qwen3_5.py) for the attention layer; the kernels themselves are the modules next to
this one, ported with their launchers. Importing this module imports Triton and asks the GPU what it is,
so only a process with a CUDA device does (``unit``).

State layouts are the reference's (``MambaPool``): the kernels index the pools by slot, and the
padding slot is an ordinary slot to them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from kernels.triton import causal_conv1d, elementwise, fused_gdn_gating as gdn_gating, fused_qk_rmsnorm_rope_gate
from kernels.triton import kv_indices, triton_gdn_fused_proj
from kernels.triton.fla import chunk, fused_recurrent, layernorm_gated

# The ported kernels are untyped: their launchers are called as SGLang calls them.
_causal_conv1d_fn: Any = causal_conv1d.causal_conv1d_fn
_chunk_gated_delta_rule: Any = chunk.chunk_gated_delta_rule
_rms_norm_gated: Any = layernorm_gated.rms_norm_gated
_fused_gdn_gating: Any = gdn_gating.fused_gdn_gating
_fused_qkvzba_causal_conv1d_update_contiguous: Any = triton_gdn_fused_proj.fused_qkvzba_causal_conv1d_update_contiguous
_packed_decode: Any = fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode
_fused_qk_gemma_rmsnorm_rope_gate: Any = fused_qk_rmsnorm_rope_gate.fused_qk_gemma_rmsnorm_rope_gate
_fused_sigmoid_mul: Any = elementwise.fused_sigmoid_mul
_create_flashinfer_kv_indices: Any = kv_indices.create_flashinfer_kv_indices_triton


def causal_conv1d_fn(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, activation: str | None,
                     conv_states: torch.Tensor, has_initial_state: torch.Tensor, cache_indices: torch.Tensor,
                     query_start_loc: torch.Tensor, seq_lens_cpu: Sequence[int] | None = None) -> torch.Tensor:
    """``gdn.conv1d_fn@1``: ``GDNAttnBackend._convolve_prefill``. The launch grid follows the
    longest sequence, which the kernel takes from the host (``seq_lens_cpu``)."""
    if seq_lens_cpu is None:
        seq_lens_cpu = (query_start_loc[1:] - query_start_loc[:-1]).tolist()
    return _causal_conv1d_fn(x, weight, bias, activation=activation, conv_states=conv_states,
                             has_initial_state=has_initial_state, cache_indices=cache_indices,
                             query_start_loc=query_start_loc, seq_lens_cpu=list(seq_lens_cpu))


def fused_qkvzba_causal_conv1d_update(mixed_qkvz: torch.Tensor, mixed_ba: torch.Tensor, conv_state: torch.Tensor,
                                      conv_weight: torch.Tensor, conv_bias: torch.Tensor | None,
                                      conv_state_indices: torch.Tensor, qkv_dim: int, v_dim: int, num_v_heads: int,
                                      head_v_dim: int, activation: str | None
                                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``gdn.qkvzba_conv1d_update@1``: the fused unpack and conv of ``GDNAttnBackend.forward_decode``,
    one kernel. The projections, the conv state and the conv weight have one dtype (the kernel
    refuses others, where SGLang falls back to its unfused kernels)."""
    return _fused_qkvzba_causal_conv1d_update_contiguous(
        mixed_qkvz, mixed_ba, conv_state, conv_weight, conv_bias, conv_state_indices, qkv_dim=qkv_dim, v_dim=v_dim,
        num_v_heads=num_v_heads, head_v_dim=head_v_dim, activation=activation)


def fused_gdn_gating(A_log: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                     dt_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``gdn.gating@1``: ``GDNAttnBackend._prefill_gates``."""
    return _fused_gdn_gating(A_log, a, b, dt_bias)


def chunk_gated_delta_rule(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
                           initial_state: torch.Tensor, initial_state_indices: torch.Tensor,
                           cu_seqlens: torch.Tensor) -> tuple[torch.Tensor, None, Any]:
    """``gdn.chunk_delta_rule@1``: ``TritonGDNKernel.extend``. Not for fp32 activations (the
    kernel refuses them)."""
    return _chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, initial_state=initial_state,
                                   cu_seqlens=cu_seqlens, head_first=False, use_qk_l2norm_in_kernel=True,
                                   initial_state_indices=initial_state_indices, inplace_update=True)


def fused_recurrent_gated_delta_rule_packed_decode(mixed_qkv: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                                                   A_log: torch.Tensor, dt_bias: torch.Tensor, scale: float,
                                                   ssm_states: torch.Tensor, cache_indices: torch.Tensor,
                                                   num_v_heads: int, head_v_dim: int) -> torch.Tensor:
    """``gdn.packed_decode@1``: ``TritonGDNKernel.packed_decode``, one kernel reading q, k and v from
    the packed projection."""
    out = mixed_qkv.new_empty(mixed_qkv.shape[0], 1, num_v_heads, head_v_dim)
    _packed_decode(mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias, scale=scale,
                   initial_state=ssm_states, out=out, ssm_state_indices=cache_indices,
                   use_qk_l2norm_in_kernel=True)
    return out.transpose(0, 1)


def rms_norm_gated(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, z: torch.Tensor | None = None,
                   eps: float = 1e-6, group_size: int | None = None, norm_before_gate: bool = True) -> torch.Tensor:
    """``norm.rms_gated@1``: ``RMSNormGated.forward`` (``layernorm_fn`` with ``is_rms_norm``)."""
    return _rms_norm_gated(x=x, weight=weight, bias=bias, z=z, eps=eps, group_size=group_size,
                           norm_before_gate=norm_before_gate, is_rms_norm=True, activation="swish")


def fused_qk_gemma_rmsnorm_rope_gate(q_gate: torch.Tensor, k: torch.Tensor, q_weight: torch.Tensor,
                                     k_weight: torch.Tensor, cos_sin_cache: torch.Tensor, positions: torch.Tensor,
                                     eps: float, num_q_heads: int, num_kv_heads: int, head_dim: int,
                                     rotary_dim: int, has_gate: bool = True
                                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """``attn.qk_norm_rope_gate@1``: ``forward_prepare_cuda_fused``, one kernel. Its results are
    bf16 or fp16, so it is not for fp32 activations."""
    q, k_out, gate = _fused_qk_gemma_rmsnorm_rope_gate(q_gate, k, q_weight, k_weight, cos_sin_cache, positions, eps,
                                                       num_q_heads, num_kv_heads, head_dim, rotary_dim,
                                                       has_gate=has_gate)
    return q, k_out, gate.view(gate.shape[0], -1) if gate is not None else None


def fused_sigmoid_mul(attn_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``act.sigmoid_mul@1``: ``fused_sigmoid_mul(..., inplace=True)``, as the attention layer calls
    it."""
    return _fused_sigmoid_mul(attn_output, gate, inplace=True)


def create_flashinfer_kv_indices(req_to_token: torch.Tensor, req_pool_indices: torch.Tensor,
                                 page_kernel_lens: torch.Tensor, kv_indptr: torch.Tensor, kv_indices: torch.Tensor,
                                 out_cache_loc: torch.Tensor | None = None) -> None:
    """``attn.kv_indices@1``: ``create_flashinfer_kv_indices_triton[(rows,)]``, as
    ``FlashInferIndicesUpdaterDecode.call_begin_forward`` launches it, one program per row; with
    ``out_cache_loc``, the same programs write each row's last slot."""
    rows = req_pool_indices.numel()
    if rows:
        _create_flashinfer_kv_indices[(rows,)](req_to_token, req_pool_indices, page_kernel_lens, kv_indptr, None,
                                               kv_indices, req_to_token.stride(0), out_cache_loc)
