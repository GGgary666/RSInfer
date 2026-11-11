# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""The paged attention definitions on FlashInfer's wrappers, as SGLang's
``FlashInferAttnBackend`` drives them (srt/layers/attention/flashinfer_backend.py, e838360;
LICENSE.sglang): its plain paths, without speculation, sliding windows, cross attention,
multi-item scoring or quantized caches. The operators' interface is the reference's
(``kernels.reference.paged``). And FlashInfer's Gemma RMSNorm, which SGLang's ``GemmaRMSNorm``
runs on CUDA. Importing this module imports FlashInfer, so only a process with a CUDA device
does (``unit``).

FlashInfer plans on the host. Every operator is planned from the host layout the caller gives
it, so a plan waits for nothing on the device: the first plan of a wrapper is FlashInfer's own
(it builds or loads the kernel module, which may compile it), and a decode wrapper's later ones
are ``fast_decode_plan``, as SGLang installs it on the wrappers of its graphs.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Any

import torch

# By name: this unit's own package is called ``flashinfer`` too, which a type checker takes an
# ``import flashinfer`` here for.
_flashinfer: Any = import_module("flashinfer")
BatchDecodeWithPagedKVCacheWrapper: Any = _flashinfer.BatchDecodeWithPagedKVCacheWrapper
BatchPrefillWithPagedKVCacheWrapper: Any = _flashinfer.BatchPrefillWithPagedKVCacheWrapper
fast_decode_plan: Any = import_module("flashinfer.decode").fast_decode_plan
_norm: Any = import_module("flashinfer.norm")
_activation: Any = import_module("flashinfer.activation")

WORKSPACE_BYTES = 384 * 1024 * 1024
"""SGLang's ``SGLANG_FLASHINFER_WORKSPACE_SIZE``."""
LAYOUT = "NHD"
BACKEND = "fa2"
"""``FlashInferAttnBackend.prefill_backend`` and ``decode_backend``."""


def should_use_tensor_core(kv_cache_dtype: torch.dtype, num_attention_heads: int, num_kv_heads: int) -> bool:
    """SGLang's rule of the same name: FlashInfer fuses the head group with the token dimension in
    its MMA, which pays from a GQA group size of 4 on."""
    gqa_group_size = num_attention_heads // num_kv_heads
    if kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return True
    if kv_cache_dtype in (torch.float16, torch.half, torch.bfloat16):
        return gqa_group_size >= 4
    return False


class Workspace:
    """The one workspace buffer every wrapper plans into (SGLang's ``flashinfer_workspace``),
    allocated when the first operator is made."""

    def __init__(self) -> None:
        self.buffers: dict[torch.device, torch.Tensor] = {}

    def on(self, device: torch.device) -> torch.Tensor:
        if device not in self.buffers:
            self.buffers[device] = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        return self.buffers[device]


class PagedDecode:
    """``attn.paged_decode@1``: ``BatchDecodeWithPagedKVCacheWrapper``, planned as
    ``FlashInferIndicesUpdaterDecode.call_begin_forward`` plans it."""

    def __init__(self, workspace: Workspace, num_qo_heads: int, num_kv_heads: int, head_dim: int,
                 q_dtype: torch.dtype, kv_dtype: torch.dtype, device: torch.device,
                 buffers: Mapping[str, torch.Tensor] | None = None) -> None:
        self.shape = (num_qo_heads, num_kv_heads, head_dim)
        self.q_dtype, self.kv_dtype, self.device, self.buffers = q_dtype, kv_dtype, device, buffers
        graph: dict[str, Any] = {} if buffers is None else {
            "use_cuda_graph": True, "paged_kv_indptr_buffer": buffers["kv_indptr"],
            "paged_kv_indices_buffer": buffers["kv_indices"],
            "paged_kv_last_page_len_buffer": buffers["kv_last_page_len"]}
        self.wrapper: Any = BatchDecodeWithPagedKVCacheWrapper(
            workspace.on(device), LAYOUT, backend=BACKEND,
            use_tensor_cores=should_use_tensor_core(kv_dtype, num_qo_heads, num_kv_heads), **graph)
        self.planned = False
        self.last_page_len = torch.ones(0, dtype=torch.int32, device=device)
        """SGLang's ``kv_last_page_len``: one-token pages, grown to the largest eager step."""

    def plan(self, qo_indptr_cpu: torch.Tensor, kv_indptr_cpu: torch.Tensor, kv_indptr: torch.Tensor,
             kv_indices: torch.Tensor) -> None:
        rows = kv_indptr_cpu.shape[0] - 1
        if not self.planned:
            # The first plan builds the module, and FlashInfer's own copies the layout it is
            # given: from the host, without a round trip to the device.
            self.wrapper.begin_forward(kv_indptr_cpu, kv_indices[:int(kv_indptr_cpu[-1])],
                                       torch.ones(rows, dtype=torch.int32), *self.shape, 1, data_type=self.kv_dtype,
                                       q_data_type=self.q_dtype, non_blocking=True)
            self.planned = True
            return
        if self.buffers is not None:
            last_page_len = self.buffers["kv_last_page_len"]
        else:
            if self.last_page_len.shape[0] < rows:
                self.last_page_len = torch.ones(rows, dtype=torch.int32, device=self.device)
            last_page_len = self.last_page_len[:rows]
        fast_decode_plan(self.wrapper, kv_indptr, kv_indices, last_page_len, *self.shape, 1, data_type=self.kv_dtype,
                         q_data_type=self.q_dtype, non_blocking=True, global_override_indptr_cpu=kv_indptr_cpu)

    def run(self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, scaling: float) -> torch.Tensor:
        return self.wrapper.forward(q.contiguous(), (k_cache, v_cache), sm_scale=scaling)


class PagedExtend:
    """``attn.paged_extend@1``: ``BatchPrefillWithPagedKVCacheWrapper`` over each row's whole
    sequence, its new tokens stored first: SGLang's extend without the ragged wrapper
    (``use_ragged`` off, as under ``SGLANG_FLASHINFER_USE_PAGED``)."""

    def __init__(self, workspace: Workspace, num_qo_heads: int, num_kv_heads: int, head_dim: int,
                 q_dtype: torch.dtype, kv_dtype: torch.dtype, device: torch.device,
                 buffers: Mapping[str, torch.Tensor] | None = None) -> None:
        if buffers is not None:
            raise ValueError("an extend step is not replayed from a graph")
        self.shape = (num_qo_heads, num_kv_heads, head_dim)
        self.q_dtype, self.kv_dtype, self.device = q_dtype, kv_dtype, device
        self.wrapper: Any = BatchPrefillWithPagedKVCacheWrapper(workspace.on(device), LAYOUT, backend=BACKEND)

    def plan(self, qo_indptr_cpu: torch.Tensor, kv_indptr_cpu: torch.Tensor, kv_indptr: torch.Tensor,
             kv_indices: torch.Tensor) -> None:
        rows = kv_indptr_cpu.shape[0] - 1
        self.wrapper.begin_forward(qo_indptr_cpu, kv_indptr_cpu, kv_indices, torch.ones(rows, dtype=torch.int32),
                                   *self.shape, 1, q_data_type=self.q_dtype, kv_data_type=self.kv_dtype,
                                   non_blocking=True)

    def run(self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, scaling: float) -> torch.Tensor:
        return self.wrapper.forward(q.contiguous(), (k_cache, v_cache), causal=True, sm_scale=scaling)


def gemma_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float,
                   residual: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """``norm.gemma_rms@1``: ``GemmaRMSNorm._forward_impl`` (srt/layers/layernorm.py), one kernel
    per call where the reference runs about ten. With ``residual`` it works in place, as SGLang's
    does: ``residual`` becomes the sum and ``x`` its norm. Takes fp32 as well as bf16."""
    if residual is not None:
        _norm.gemma_fused_add_rmsnorm(x, residual, weight, eps)
        return x, residual
    return _norm.gemma_rmsnorm(x, weight, eps)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """``act.silu_and_mul@1``: ``SiluAndMul.forward_cuda``, one kernel (SGLang's own is this one,
    built by its JIT). bf16 and fp16 only."""
    return _activation.silu_and_mul(x)
