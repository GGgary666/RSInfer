# type: ignore
# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
# Ported from SGLang e838360 (kernels/ops/mamba/mamba_state_scatter_triton.py; LICENSE.sglang): the
# kernel that tracks the GDN states of every layer for the prefix cache (``track_mamba_states_all_layers``)
# and its launcher, unchanged; the other kernels of that module (speculative decoding) are not here.

import functools

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_track_chunk(
    ptr,
    src_off,
    dst_off,
    start,
    numel: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for off in tl.static_range(0, CHUNK, BLOCK):
        offsets = start + off + tl.arange(0, BLOCK)
        mask = offsets < numel
        data = tl.load(ptr + src_off + offsets, mask=mask)
        tl.store(ptr + dst_off + offsets, data, mask=mask)


@triton.jit(do_not_specialize=["batch_size", "num_layers"])
def track_mamba_state_if_needed_kernel(
    conv_states_ptr,
    ssm_states_ptr,
    cache_indices_ptr,
    mamba_track_mask_ptr,
    mamba_track_indices_ptr,
    conv_layer_stride,
    conv_row_stride,
    ssm_layer_stride,
    ssm_row_stride,
    batch_size,
    num_layers,
    conv_state_numel_per_row: tl.constexpr,
    ssm_state_numel_per_row: tl.constexpr,
    CONV_CHUNK: tl.constexpr,
    SSM_CHUNK: tl.constexpr,
    NUM_CONV_CHUNKS: tl.constexpr,
    NUM_SSM_CHUNKS: tl.constexpr,
    BS_POW2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    check_freed_slots: tl.constexpr,  # only the shared/unified KV pool emits -1
):
    """
    For each batch row with mamba_track_mask set, copy its conv and ssm state
    rows from cache_indices[i] to mamba_track_indices[i], in every layer.

    Persistent grid: every program reads the whole mask, so the tracked rows
    are compacted in-kernel and only their (row, layer, chunk) work items are
    spread over the grid. A step with no tracked row reads one mask vector per
    program and exits; one tracked row runs at HBM bandwidth on all SMs.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    rows = tl.arange(0, BS_POW2)
    track = tl.load(mamba_track_mask_ptr + rows, mask=rows < batch_size, other=0).to(
        tl.int32
    )
    num_tracked = tl.sum(track, axis=0)
    track_rank = tl.cumsum(track, axis=0)

    CHUNKS: tl.constexpr = NUM_CONV_CHUNKS + NUM_SSM_CHUNKS
    items_per_row = CHUNKS * num_layers
    for item in range(pid, num_tracked * items_per_row, num_programs):
        rank = item // items_per_row
        rem = item - rank * items_per_row
        layer_idx = (rem // CHUNKS).to(tl.int64)
        chunk = rem % CHUNKS
        batch_idx = tl.sum(
            tl.where((track_rank == rank + 1) & (track != 0), rows, 0), axis=0
        )
        # int64 before the row stride multiply: the page-granularity envelope
        # layout makes the row stride large enough for int32 to overflow.
        src_idx = tl.load(cache_indices_ptr + batch_idx).to(tl.int64)
        dst_idx = tl.load(mamba_track_indices_ptr + batch_idx).to(tl.int64)
        # Skip freed slots (-1): `state_ptr + (-1)*stride` would fault.
        if (not check_freed_slots) or ((src_idx >= 0) & (dst_idx >= 0)):
            if chunk < NUM_CONV_CHUNKS:
                _copy_track_chunk(
                    conv_states_ptr + layer_idx * conv_layer_stride,
                    src_idx * conv_row_stride,
                    dst_idx * conv_row_stride,
                    chunk * CONV_CHUNK,
                    conv_state_numel_per_row,
                    CONV_CHUNK,
                    BLOCK_SIZE,
                )
            else:
                _copy_track_chunk(
                    ssm_states_ptr + layer_idx * ssm_layer_stride,
                    src_idx * ssm_row_stride,
                    dst_idx * ssm_row_stride,
                    (chunk - NUM_CONV_CHUNKS) * SSM_CHUNK,
                    ssm_state_numel_per_row,
                    SSM_CHUNK,
                    BLOCK_SIZE,
                )


_TRACK_BLOCK_SIZE = 8192
_TRACK_CHUNK_BYTES = 32768
_TRACK_PROGRAMS_PER_SM = 4


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count or 132


def _launch_track_mamba_states(
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    mamba_track_mask: torch.Tensor,
    mamba_track_indices: torch.Tensor,
    batch_size: int,
    num_layers: int,
    conv_strides: tuple,
    ssm_strides: tuple,
    conv_state_numel_per_row: int,
    ssm_state_numel_per_row: int,
    check_freed_slots: bool,
):
    if batch_size == 0:
        return
    conv_chunk = max(
        _TRACK_BLOCK_SIZE, _TRACK_CHUNK_BYTES // conv_states.element_size()
    )
    ssm_chunk = max(_TRACK_BLOCK_SIZE, _TRACK_CHUNK_BYTES // ssm_states.element_size())
    num_conv_chunks = triton.cdiv(conv_state_numel_per_row, conv_chunk)
    num_ssm_chunks = triton.cdiv(ssm_state_numel_per_row, ssm_chunk)
    max_items = batch_size * num_layers * (num_conv_chunks + num_ssm_chunks)
    device_index = conv_states.device.index or 0
    grid = (min(max_items, _TRACK_PROGRAMS_PER_SM * _num_sms(device_index)),)
    track_mamba_state_if_needed_kernel[grid](
        conv_states,
        ssm_states,
        cache_indices,
        mamba_track_mask,
        mamba_track_indices,
        *conv_strides,
        *ssm_strides,
        batch_size,
        num_layers,
        conv_state_numel_per_row,
        ssm_state_numel_per_row,
        conv_chunk,
        ssm_chunk,
        num_conv_chunks,
        num_ssm_chunks,
        max(16, triton.next_power_of_2(batch_size)),
        _TRACK_BLOCK_SIZE,
        check_freed_slots,
        num_warps=8,
    )


def track_mamba_states_all_layers(
    conv_states_pool: torch.Tensor,
    ssm_states_pool: torch.Tensor,
    cache_indices: torch.Tensor,
    mamba_track_mask: torch.Tensor,
    mamba_track_indices: torch.Tensor,
    batch_size: int,
    check_freed_slots: bool = False,
):
    """Track conv/ssm states for ALL mamba layers in one launch.

    conv_states_pool / ssm_states_pool are the full [num_layers, pool_size,
    ...] pools; per-row copy semantics are identical to
    track_mamba_states_if_needed applied per layer. The track mask / source /
    destination indices are shared across layers, so a single launch once
    every layer's state is final is equivalent to the per-layer launches.
    """
    _launch_track_mamba_states(
        conv_states_pool,
        ssm_states_pool,
        cache_indices,
        mamba_track_mask,
        mamba_track_indices,
        batch_size,
        conv_states_pool.shape[0],
        (conv_states_pool.stride(0), conv_states_pool.stride(1)),
        (ssm_states_pool.stride(0), ssm_states_pool.stride(1)),
        conv_states_pool[0, 0].numel(),
        ssm_states_pool[0, 0].numel(),
        check_freed_slots,
    )
