# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""The GDN definitions on SGLang's Triton kernels (e838360; LICENSE.sglang).

Each function takes its definition's arguments (``kernels.reference``) and makes the call that
SGLang's ``GDNAttnBackend`` and ``TritonGDNKernel`` make (srt/layers/attention/linear/
gdn_backend.py, kernels/gdn_triton.py); the kernels themselves are the modules next to this one,
ported with their launchers. Importing this module imports Triton and asks the GPU what it is,
so only a process with a CUDA device does (``unit``).

State layouts are the reference's (``MambaPool``): the kernels index the pools by slot, and the
padding slot is an ordinary slot to them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from kernels.triton import causal_conv1d, fused_gdn_gating as gdn_gating, fused_sigmoid_gating_recurrent
from kernels.triton.fla import chunk, layernorm_gated

# The ported kernels are untyped: their launchers are called as SGLang calls them.
_causal_conv1d_fn: Any = causal_conv1d.causal_conv1d_fn
_causal_conv1d_update: Any = causal_conv1d.causal_conv1d_update
_chunk_gated_delta_rule: Any = chunk.chunk_gated_delta_rule
_rms_norm_gated: Any = layernorm_gated.rms_norm_gated
_fused_gdn_gating: Any = gdn_gating.fused_gdn_gating
_fused_sigmoid_gating_delta_rule_update: Any = fused_sigmoid_gating_recurrent.fused_sigmoid_gating_delta_rule_update


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


def causal_conv1d_update(x: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
                         activation: str | None, conv_state_indices: torch.Tensor) -> torch.Tensor:
    """``gdn.conv1d_update@1``: the conv of ``GDNAttnBackend.forward_decode``."""
    return _causal_conv1d_update(x, conv_state, weight, bias, activation, conv_state_indices=conv_state_indices)


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


def fused_sigmoid_gating_delta_rule_update(A_log: torch.Tensor, dt_bias: torch.Tensor, q: torch.Tensor,
                                           k: torch.Tensor, v: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                                           initial_state_source: torch.Tensor, initial_state_indices: torch.Tensor,
                                           cu_seqlens: torch.Tensor) -> torch.Tensor:
    """``gdn.recurrent_update@1``: ``TritonGDNKernel.decode``."""
    return _fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b, initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices, cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0, softplus_threshold=20.0)


def rms_norm_gated(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, z: torch.Tensor | None = None,
                   eps: float = 1e-6, group_size: int | None = None, norm_before_gate: bool = True) -> torch.Tensor:
    """``norm.rms_gated@1``: ``RMSNormGated.forward`` (``layernorm_fn`` with ``is_rms_norm``)."""
    return _rms_norm_gated(x=x, weight=weight, bias=bias, z=z, eps=eps, group_size=group_size,
                           norm_before_gate=norm_before_gate, is_rms_norm=True, activation="swish")
