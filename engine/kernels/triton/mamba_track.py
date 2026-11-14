# type: ignore
# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
# Ported from SGLang e838360 (kernels/ops/mamba/mamba_state_scatter_triton.py; LICENSE.sglang), unchanged:
# the kernel that tracks the GDN states of every layer for the prefix cache (``track_mamba_states_all_layers``)
# and its launcher; and the masked scatter that commits a verify step's accepted states
# (``fused_mamba_state_scatter_with_mask``, doc/specs/mtp.md). The conv-window variants of that module
# are for SGLang's deduplicated conv windows, which are not ours: ours are dense ``[layers, lanes, D, ...]``.

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


# ---------------------------------------------- speculation: commit a verify step (doc/specs/mtp.md)


def _require_entry_contiguous_dst(
    dst: torch.Tensor, entry_start_dim: int, fn_name: str
) -> None:
    """dst layout contract: the kernels index through the real layer/slot
    strides (int64) plus a FLAT element offset within one (layer, slot)
    entry — layer/slot strides may be arbitrary (envelope-strided unified
    pool views), but the trailing entry dims must be contiguous.
    """
    expected = 1
    for i in range(dst.ndim - 1, entry_start_dim - 1, -1):
        if dst.shape[i] != 1 and dst.stride(i) != expected:
            raise ValueError(
                f"{fn_name}: dst entry dims (dims {entry_start_dim}.."
                f"{dst.ndim - 1}) must be contiguous; got "
                f"shape={tuple(dst.shape)} strides={tuple(dst.stride())}"
            )
        expected *= dst.shape[i]


@triton.jit
def _fused_mamba_state_scatter_with_mask_kernel(
    src_ptr,
    dst_ptr,
    # Raw index arrays (before index_select)
    dst_indices_raw_ptr,  # [total_requests] - state_indices_tensor
    step_indices_raw_ptr,  # [total_requests] - last_correct_step_indices or mamba_steps_to_track
    src_indices_raw_ptr,  # [total_requests] - verify scratch row for each request
    elem_per_entry: tl.constexpr,
    src_layer_stride,
    src_req_stride,
    src_step_stride,
    dst_layer_stride,
    dst_req_stride,
    src_req_size,
    src_step_size,
    dst_req_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused gather-scatter kernel with built-in masking.

    This kernel fuses the index_select operations by:
    1. Iterating over all requests (pid_req from 0 to total_requests-1)
    2. Checking if step_indices_raw[pid_req] >= 0 (valid mask)
    3. If valid, performing the scatter:
       dst[l, dst_indices_raw[pid_req], :] = src[l, pid_req, step_indices_raw[pid_req], :]

    Grid: (total_requests, num_layers, ceil(elem_per_entry / BLOCK_SIZE))
    """
    pid_req = tl.program_id(0)
    pid_layer = tl.program_id(1).to(tl.int64)
    pid_block = tl.program_id(2).to(tl.int64)

    # Load step index to check validity (step >= 0 means valid)
    step_idx = tl.load(step_indices_raw_ptr + pid_req).to(tl.int64)

    # Early exit if this request is not valid (step < 0)
    if step_idx < 0:
        return

    # Load destination index
    dst_idx = tl.load(dst_indices_raw_ptr + pid_req).to(tl.int64)

    src_idx = tl.load(src_indices_raw_ptr + pid_req).to(tl.int64)

    # Bounds check to avoid illegal memory access
    if not (
        (dst_idx >= 0)
        & (dst_idx < dst_req_size)
        & (src_idx >= 0)
        & (src_idx < src_req_size)
        & (step_idx < src_step_size)
    ):
        return

    # Compute base offsets
    src_offset = (
        pid_layer * src_layer_stride
        + src_idx * src_req_stride
        + step_idx * src_step_stride
    )
    dst_offset = pid_layer * dst_layer_stride + dst_idx * dst_req_stride

    # Compute element range for this block
    start = pid_block * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elem_per_entry

    # Load from source and store to destination
    data = tl.load(src_ptr + src_offset + offsets, mask=mask)
    tl.store(dst_ptr + dst_offset + offsets, data, mask=mask)


def _prepare_src_rows(src: torch.Tensor | None, steps: torch.Tensor) -> torch.Tensor:
    if src is None:
        src = torch.arange(steps.shape[0], dtype=torch.int32, device=steps.device)
    if src.ndim != 1 or src.shape != steps.shape:
        raise ValueError(
            f"source indices length mismatch: {src.shape=} vs {steps.shape=}"
        )
    return src.to(torch.int32).contiguous()


def fused_mamba_state_scatter_with_mask(
    dst: torch.Tensor,  # [num_layers, cache_size, *state_shape]
    src: torch.Tensor,  # [num_layers, spec_size, draft_tokens, *state_shape]
    dst_indices_raw: torch.Tensor,  # [total_requests] - raw indices (e.g., state_indices_tensor)
    step_indices_raw: torch.Tensor,  # [total_requests] - raw step indices (step >= 0 means valid)
    src_indices_raw: torch.Tensor | None = None,  # [total_requests] - scratch rows
):
    """
    Fully fused gather-scatter with built-in masking for mamba state updates.

    This function fuses the following operations into a single kernel:
    1. valid_mask = step_indices_raw >= 0
    2. valid_indices = valid_mask.nonzero()
    3. dst_indices = dst_indices_raw[valid_indices]  (index_select)
    4. step_indices = step_indices_raw[valid_indices]  (index_select)
    5. for each valid i: dst[:, dst_indices[i], :] = src[:, i, step_indices[i], :]

    Args:
        dst: Destination tensor [num_layers, cache_size, *state_shape]
        src: Source tensor [num_layers, spec_size, draft_tokens, *state_shape]
        dst_indices_raw: Raw destination indices for all requests [total_requests]
        step_indices_raw: Raw step indices; entry >= 0 means valid [total_requests]
    """
    total_requests = step_indices_raw.shape[0]
    if total_requests == 0:
        return

    if dst.device != src.device:
        raise ValueError(
            f"dst and src must be on the same device. {dst.device=} {src.device=}"
        )
    if not (dst.is_cuda or dst.is_xpu) or dst.device != src.device:
        raise ValueError(
            "fused_mamba_state_scatter_with_mask only supports CUDA/XPU tensors."
        )
    if dst.ndim < 2 or src.ndim < 3:
        raise ValueError(f"Unexpected tensor ranks: {dst.ndim=} {src.ndim=}")
    if dst.shape[0] != src.shape[0]:
        raise ValueError(
            f"Layer dimension mismatch: {dst.shape[0]=} vs {src.shape[0]=}"
        )
    if dst.shape[2:] != src.shape[3:]:
        raise ValueError(
            f"Trailing dims mismatch: {dst.shape[2:]=} vs {src.shape[3:]=}"
        )
    if dst_indices_raw.ndim != 1 or step_indices_raw.ndim != 1:
        raise ValueError(
            f"indices must be 1D: {dst_indices_raw.shape=} {step_indices_raw.shape=}"
        )
    if dst_indices_raw.shape[0] != step_indices_raw.shape[0]:
        raise ValueError(
            f"indices length mismatch: {dst_indices_raw.shape[0]=} vs {step_indices_raw.shape[0]=}"
        )

    num_layers = dst.shape[0]
    src_req_size = src.shape[1]
    src_step_size = src.shape[2]
    dst_req_size = dst.shape[1]

    # Flatten trailing dimensions: number of elements per (layer, cache_line) entry.
    elem_per_entry = dst.numel() // (dst.shape[0] * dst.shape[1])

    # Get strides (in elements, not bytes)
    src_layer_stride = src.stride(0)
    src_req_stride = src.stride(1)
    src_step_stride = src.stride(2)
    dst_layer_stride = dst.stride(0)
    dst_req_stride = dst.stride(1)

    # Ensure indices are int32 and contiguous
    dst_indices_raw = dst_indices_raw.to(torch.int32).contiguous()
    step_indices_raw = step_indices_raw.to(torch.int32).contiguous()
    src_indices_raw = _prepare_src_rows(src_indices_raw, step_indices_raw)

    _require_entry_contiguous_dst(dst, 2, "fused_mamba_state_scatter_with_mask")
    if not src.is_contiguous():
        raise ValueError("src tensor must be contiguous")

    # Block size for copying elements
    BLOCK_SIZE = 1024

    # Grid over all requests - invalid ones will early-exit in the kernel
    grid = (total_requests, num_layers, triton.cdiv(elem_per_entry, BLOCK_SIZE))

    _fused_mamba_state_scatter_with_mask_kernel[grid](
        src,
        dst,
        dst_indices_raw,
        step_indices_raw,
        src_indices_raw,
        elem_per_entry,
        src_layer_stride,
        src_req_stride,
        src_step_stride,
        dst_layer_stride,
        dst_req_stride,
        src_req_size,
        src_step_size,
        dst_req_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
