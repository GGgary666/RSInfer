# Copyright DeepInfer authors
# SPDX-License-Identifier: Apache-2.0
# type: ignore
"""``convert_kv_index``: a query's chosen latents from positions in its sequence to records of the cache.

From ``deepinfer/model_executor/layers/tilelang_ops/flash_mla_utils.py`` of DeepInfer ``a679bb8`` (LICENSE.deepinfer
in this directory), without its tests and torch reference; the kernel is imported from ``cuda/``."""

import torch

from kernels.deepinfer.tilelang_ops.cuda.flash_mla_utils import convert_kv_index_tl


def convert_kv_index(
    topk_idxs: torch.Tensor,
    block_size: int,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
) -> None:
    """Map topk logical token indices to physical kv-cache slot indices via block_tables.

    Args:
        topk_idxs: [bsz, seqlen_q, topk_tokens], int32, remapped in-place;
            positions >= context_lens[b, s] are set to -1
        context_lens: [bsz, seqlen_q], int32
        block_tables: [bsz, max_num_blocks], int32
    """
    assert topk_idxs.is_contiguous()
    assert context_lens.is_contiguous()
    assert block_tables.is_contiguous()
    assert topk_idxs.ndim == 3
    assert context_lens.ndim == 2
    assert block_tables.ndim == 2
    b, s, k = topk_idxs.shape
    _b, n = block_tables.shape
    assert b == _b
    convert_kv_index_tl(s, k, n, block_size)(
        topk_idxs,
        context_lens,
        block_tables,
    )
