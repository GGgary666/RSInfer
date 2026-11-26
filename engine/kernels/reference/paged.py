# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""The reference of the paged attention definitions, ``attn.paged_decode@1`` and
``attn.paged_extend@1``: attention planned per step over lists of token slots, the interface of
SGLang's ``FlashInferAttnBackend`` (srt/layers/attention/flashinfer_backend.py, e838360;
LICENSE.sglang) to FlashInfer's paged wrappers, at SGLang's page size of one token.

A solution of these definitions is not a function over tensors but what makes such an operator,
because the libraries behind it split the work in two (stage2-contract.md §8.4):

- ``Solution(num_qo_heads, num_kv_heads, head_dim, q_dtype, kv_dtype, device, buffers)`` makes one
  operator, for the eager steps of a phase (``buffers`` is ``None``) or for one graph bucket
  (``buffers``: its fixed ``kv_indptr`` ``[rows + 1]``, ``kv_indices`` and ``kv_last_page_len``
  ``[rows]``, all int32 on the device, which a captured ``run`` reads at replay);
- ``plan(qo_indptr_cpu, kv_indptr_cpu, kv_indptr, kv_indices)`` runs outside any graph before
  each ``run``. Row ``i`` attends over the token slots ``kv_indices[kv_indptr[i]:kv_indptr[i + 1]]``
  and has ``qo_indptr[i + 1] - qo_indptr[i]`` queries, the row's last tokens: each attends
  causally, the last one over all of the row's slots. The layout is int32, and given on the
  host, where the step's lengths are known, so that planning waits for nothing on the device;
  ``kv_indptr`` and ``kv_indices`` are the device's copy the kernels read (for a graph bucket
  they are its ``buffers``, which the caller has filled);
- ``run(q, k_cache, v_cache, scaling)`` computes ``[tokens, heads, head_dim]`` from ``q`` of that
  shape and the caches ``[slots, kv_heads, head_dim]``, eagerly or under capture.

A decode step is the case of one query per row; the two definitions differ in the kernels behind
them. The references loop over the rows with SGLang's native SDPA (``TorchNativeAttnBackend``:
``_run_sdpa_forward_extend`` and ``_run_sdpa_forward_decode``, whose bits they give), reading the
layout on the host: they are not capturable.
"""

from __future__ import annotations

from collections.abc import Sequence

from collections.abc import Mapping

import torch


def block_paged(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, blocks: torch.Tensor, ends: Sequence[int],
                lengths: Sequence[int], scale: float, causal: bool) -> torch.Tensor:
    """Grouped attention over a table of physical KV pages, with right-aligned causal queries."""
    out: list[torch.Tensor] = []
    start = 0
    page_size = k.shape[1]
    for row, (end, length) in enumerate(zip(ends, lengths, strict=True)):
        ids = blocks[row, : -(-length // page_size)].long()
        key = k.index_select(0, ids).flatten(0, 1)[:length].transpose(0, 1).to(q.dtype)
        val = v.index_select(0, ids).flatten(0, 1)[:length].transpose(0, 1).to(q.dtype)
        query = q[start:end].transpose(0, 1)
        mask = None
        if causal:
            mask = (torch.arange(length, device=q.device)[None, :]
                    <= torch.arange(length - end + start, length, device=q.device)[:, None])
        result = torch.nn.functional.scaled_dot_product_attention(
            query.unsqueeze(0), key.unsqueeze(0), val.unsqueeze(0), attn_mask=mask, scale=scale,
            enable_gqa=query.shape[0] != key.shape[0])
        out.append(result.squeeze(0).transpose(0, 1))
        start = end
    return torch.cat(out, dim=0)
from torch.nn.functional import scaled_dot_product_attention


class PagedExtend:
    """``attn.paged_extend@1``."""

    def __init__(self, num_qo_heads: int, num_kv_heads: int, head_dim: int, q_dtype: torch.dtype,
                 kv_dtype: torch.dtype, device: torch.device,
                 buffers: Mapping[str, torch.Tensor] | None = None) -> None:
        self.num_qo_heads, self.num_kv_heads, self.head_dim = num_qo_heads, num_kv_heads, head_dim
        self.qo_indptr: list[int] = []
        self.kv_indptr: list[int] = []
        self.kv_indices: torch.Tensor | None = None

    def plan(self, qo_indptr_cpu: torch.Tensor, kv_indptr_cpu: torch.Tensor, kv_indptr: torch.Tensor,
             kv_indices: torch.Tensor) -> None:
        self.qo_indptr = [int(i) for i in qo_indptr_cpu.tolist()]
        self.kv_indptr = [int(i) for i in kv_indptr_cpu.tolist()]
        self.kv_indices = kv_indices

    def run(self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, scaling: float) -> torch.Tensor:
        if self.kv_indices is None:
            raise RuntimeError("paged attention ran before plan")
        output = torch.empty_like(q)
        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = q.movedim(0, q.dim() - 2)
        enable_gqa = self.num_qo_heads != self.num_kv_heads
        for row in range(len(self.kv_indptr) - 1):
            start_q, end_q = self.qo_indptr[row], self.qo_indptr[row + 1]
            per_req_tokens = self.kv_indices[self.kv_indptr[row]:self.kv_indptr[row + 1]].long()
            seq_len_kv = per_req_tokens.shape[0]
            prefix = seq_len_kv - (end_q - start_q)
            per_req_query = query[:, start_q:end_q, :]
            per_req_query_redudant = torch.empty((per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
                                                 dtype=per_req_query.dtype, device=per_req_query.device)
            per_req_query_redudant[:, prefix:, :] = per_req_query
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2).to(per_req_query.dtype)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2).to(per_req_query.dtype)
            per_req_out_redudant = (
                scaled_dot_product_attention(
                    per_req_query_redudant.unsqueeze(0), per_req_key.unsqueeze(0), per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa, scale=scaling, is_causal=True)
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out_redudant[prefix:, :, :]
        return output


class PagedDecode(PagedExtend):
    """``attn.paged_decode@1``: one query per row."""

    def run(self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, scaling: float) -> torch.Tensor:
        if self.kv_indices is None:
            raise RuntimeError("paged attention ran before plan")
        output = torch.empty_like(q)
        query = q.movedim(0, q.dim() - 2)
        enable_gqa = self.num_qo_heads != self.num_kv_heads
        for row in range(len(self.kv_indptr) - 1):
            per_req_query = query[:, row:row + 1, :]
            per_req_tokens = self.kv_indices[self.kv_indptr[row]:self.kv_indptr[row + 1]].long()
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2).to(per_req_query.dtype)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2).to(per_req_query.dtype)
            per_req_out = (
                scaled_dot_product_attention(
                    per_req_query.unsqueeze(0), per_req_key.unsqueeze(0), per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa, scale=scaling, is_causal=False)
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[row:row + 1, :, :] = per_req_out
        return output
