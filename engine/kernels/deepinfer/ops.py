"""The DS 1017 definitions (``kernels.ds1017``) on the kernel libraries DeepInfer runs: DeepGEMM (the
quantized and bf16 GEMMs), FlashMLA (the sparse attention), TileKernels (casts, norms, the MoE gate and
SwiGLU, Hyper-Connections, the Engram gate) and DeepSelect (top-k). Each is called the way DeepInfer's
own modules call it (``deepinfer/model_executor`` at a679bb8: ``layers/linear.py``,
``models/v4_modules/mlp.py``, ``mhc_ops.py``, ``engram.py``, ``index_attn.py``), on its paths that
fuse nothing across ranks or sublayers: the grouped GEMMs, not ``mega_moe``; the Hyper-Connections
kernels one by one, not ``mega_mhc``.

Where the arithmetic differs from the checkpoint's reference implementation (and so from the
references of ``kernels.ds1017``), beyond the order a sum is accumulated in:

- ``moe_experts``: the SwiGLU's result is quantized to FP8 from fp32. The reference implementation
  rounds it to bf16 first, which moves about one value in sixteen to the next FP8 value; the routed
  experts' sum then differs by a few percent of its largest element.
- ``shared_expert``: the same for the shared expert's SwiGLU.
- ``hc_mixes``: the product with the stream's copies is a TF32 GEMM (``tf32_hc_prenorm_gemm``), so the
  coefficients agree to about 1e-4, not to fp32's last bits.
- ``fake_quant_fp4`` with an e4m3 scale (the compressed KV): TileKernels multiplies by the scale's
  reciprocal where the reference implementation divides by the scale, so a value exactly halfway
  between two FP4 values lands on the other one: about one element in 2000, one FP4 step apart.

Three definitions are torch calls as they stand (``silu``, ``pool``, ``index_score``): the layers do not
call the last two any more; the indexer and the compressor are ``index_topk``, ``index_topk_paged``,
``compress`` and ``compress_decode``, with DeepInfer's own TileLang operators (``tilelang_ops``,
vendored from DeepInfer) and DeepGEMM's indexer kernels, as ``index_attn.py`` and ``compress.py`` of
``v4_modules`` call them.

The weights arrive as the checkpoint's bytes (``uint8``); a weight's scales as
``formats.pack_scales`` lays them out, which is what DeepGEMM reads on SM 10.0. Activations are bf16.
"""
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

from types import MappingProxyType
from typing import Any, cast

import deep_gemm
import flash_mla
import tile_kernels
import torch
import torch.nn.functional as F
from deep_select import interface as deep_select

from kernels.deepinfer.tilelang_ops.cache_common import FP4_INDEX_KV_FORMAT, V41_FORMAT, V41_FP4_FORMAT, KVCacheFormat
from kernels.deepinfer.tilelang_ops.candidate_blocks import (
    candidate_block_max,
    get_candidate_compact_lens,
    restore_candidate_indices_inplace,
)
from kernels.deepinfer.tilelang_ops.compress import compress_and_update_state_decode, compress_and_update_state_prefill
from kernels.deepinfer.tilelang_ops.flash_mla_utils import convert_kv_index
from kernels.deepinfer.tilelang_ops.fused_transform import (
    CastArgs,
    KVCacheOutArgs,
    NormArgs,
    RopeArgs,
    fused_norm_rope_hadamard_cast_store_kvcache,
)
from kernels.deepinfer.tilelang_ops.load_paged_kv_cache import load_paged_kv_cache
from kernels.deepinfer.tilelang_ops.store_kv_cache import store_kv_cache
from kernels import ds1017_ep as ep
from kernels.ds1017_formats import BLOCK, packed_words

E4M3 = torch.float8_e4m3fn
BF16 = torch.bfloat16
ROW_BYTES = 32
"""A GEMM's output rows start at multiples of this (``ReplicatedLinear._allocate_output``)."""
SPLITS = 16
"""How many partial sums the mixes' GEMM leaves to its reduction (``mhc_ops.N_SPLITS_CUDA``)."""
KEY_BLOCK = 128
"""FlashMLA's sparse kernels take a query's keys this many at a time (``topk % B_TOPK == 0``)."""
PADDING_SCALE = 0x7F7F7F7F
"""Four ue8m0 exponents of 1: the scales of a row no token is in."""


def align_experts() -> int:
    """One alignment for an expert's rows in DeepGEMM and TileKernels, as ``DeepseekMoE`` sets it: a
    setting of the process, the same at every call."""
    align: int = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    deep_gemm.set_mk_alignment_for_contiguous_layout(align)
    tile_kernels.config.set_token_alignment(align)
    return align


def _cast(x: torch.Tensor, row_major: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """``get_fp8_cast_params``: FP8 per 32 with power-of-two scales, packed as the GEMMs read them."""
    return tile_kernels.quant.per_token_cast(x, fmt="e4m3", num_per_channels=BLOCK, round_sf=True, use_packed_ue8m0=True,
                                             use_tma_aligned_col_major_sf=not row_major)


def _rows(x: torch.Tensor) -> torch.Tensor:
    """``x`` as contiguous bf16 rows ``[tokens, K]``."""
    if x.dtype != BF16:
        raise TypeError(f"these kernels take bf16 activations, not {x.dtype}")
    return x.reshape(-1, x.shape[-1]).contiguous()


def _out(rows: int, cols: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """An output ``[rows, cols]`` whose rows start at multiples of ``ROW_BYTES``."""
    per = ROW_BYTES // torch.empty((), dtype=dtype).element_size()
    return torch.empty(rows, -(-cols // per) * per, dtype=dtype, device=device)[:, :cols]


def fake_quant_fp8(x: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    out = _rows(x).clone()
    tile_kernels.quant.per_token_cast_and_cast_back(out, "e4m3", block, round_sf=True)
    return out.view(x.shape)


def fake_quant_fp4(x: torch.Tensor, block: int, e4m3_scale: bool = False) -> torch.Tensor:
    out = _rows(x).clone()
    tile_kernels.quant.per_token_cast_and_cast_back(out, "e2m1", block, round_sf=not e4m3_scale, use_e4m3_sf=e4m3_scale)
    return out.view(x.shape)


def linear_fp8(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, packed: torch.Tensor | None = None
               ) -> torch.Tensor:
    """``ReplicatedLinear.apply_weight_fp8``."""
    if packed is None:
        raise ValueError("DeepGEMM reads a weight's scales packed (formats.pack_scales)")
    rows = _rows(x)
    n = weight.shape[0]
    out = _out(rows.shape[0], n, BF16, x.device)
    if rows.shape[0]:
        deep_gemm.fp8_fp4_gemm_nt(_cast(rows), (weight.view(E4M3), packed_words(packed, n)), out, None,
                                  recipe=(1, 1, BLOCK))
    return out.reshape(*x.shape[:-1], n)


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``ReplicatedLinear.apply_weight_bf16`` for a bf16 weight; an fp32 weight (the head, the gate,
    the pooling compressor's projections) is torch's product."""
    if weight.dtype != BF16:
        return F.linear(x.to(weight.dtype), weight)
    rows = _rows(x.to(BF16))
    out = _out(rows.shape[0], weight.shape[0], BF16, x.device)
    if rows.shape[0]:
        deep_gemm.bf16_gemm_nt(rows, weight, out, None)
    return out.reshape(*x.shape[:-1], weight.shape[0])


def grouped_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    out = torch.empty(*x.shape[:-1], weight.shape[1], dtype=BF16, device=x.device)
    if x.shape[0]:
        deep_gemm.einsum("bhr,hdr->bhd", x.contiguous(), weight, out)
    return out


def index_score(q: torch.Tensor, keys: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    score = torch.einsum("bshd,btd->bsht", q, keys)
    return (score.relu_() * weights.unsqueeze(-1)).sum(dim=2)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    rows = x.reshape(-1, x.shape[-1]).contiguous()
    if not rows.shape[0]:
        return x.clone()
    return tile_kernels.quant.norm_forward(rows, weight.to(x.dtype), eps)[0].view(x.shape)


def sparse_attn(q: torch.Tensor, source: torch.Tensor, idx: torch.Tensor, sink: torch.Tensor, scale: float
                ) -> torch.Tensor:
    """FlashMLA's sparse attention over listed keys (its prefill kernel): 64 or 128 heads of 512. It
    takes the keys of a query in blocks, so the list is padded with "none" to a whole number of them."""
    if not q.shape[0]:
        return q.clone()
    listed = F.pad(idx.to(torch.int32), (0, -idx.shape[1] % KEY_BLOCK), value=-1)
    return flash_mla.flash_mla_sparse_fwd(q.contiguous(), source.unsqueeze(1), listed.unsqueeze(1), scale,
                                          source.shape[-1], sink)[0]


def _listed(idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A list of records ``[tokens, keys]`` with the missing ones last, as FlashMLA's decode kernel
    takes it: "none" after them up to a whole number of blocks; and how many are there."""
    listed = F.pad(idx.to(torch.int32), (0, -idx.shape[1] % KEY_BLOCK), value=-1)
    return listed.unsqueeze(1), (idx >= 0).sum(-1).to(torch.int32)


def cache_attn(q: torch.Tensor, window: torch.Tensor, window_idx: torch.Tensor, latents: torch.Tensor | None,
               latent_idx: torch.Tensor | None, sink: torch.Tensor, scale: float) -> torch.Tensor:
    """FlashMLA's decode kernel, which reads the caches' bytes: the window's FP8 records as its main
    cache, the latents' FP4 records as its extra one. It takes the records that are there first: the
    window's list is turned around (its missing ones come first), the latents' is as it is. A query's
    result does not depend on the others in the step (``enable_batch_invariant``)."""
    if not q.shape[0]:
        return q.clone()
    main, count = _listed(window_idx.flip(-1))
    extra = extra_idx = extra_count = None
    if latents is not None and latent_idx is not None:
        extra = latents.unsqueeze(2)
        extra_idx, extra_count = _listed(latent_idx)
    out, _ = flash_mla.flash_mla_with_kvcache(
        q.unsqueeze(1).contiguous(), window.unsqueeze(2), None, None, q.shape[-1], flash_mla.get_mla_metadata()[0],
        softmax_scale=scale, causal=False, is_fp8_kvcache=True, indices=main, attn_sink=sink, extra_k_cache=extra,
        extra_indices_in_kvcache=extra_idx, topk_length=count, extra_topk_length=extra_count,
        enable_batch_invariant=True)
    return out[:, 0]


def hc_mixes(x: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, norm_eps: float,
             iters: int, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``mhc_pre_norm_fn``, ``mhc_pre_split_mixes`` and ``sinkhorn_normalize``."""
    tokens, hc = x.shape[:2]
    width, device = fn.shape[0], x.device
    f32 = torch.float32
    parts = torch.empty(SPLITS, tokens, width, dtype=f32, device=device)
    squares = torch.empty(SPLITS, tokens, dtype=f32, device=device)
    deep_gemm.tf32_hc_prenorm_gemm(_rows(x.flatten(1)), fn, parts, squares, SPLITS)
    product = torch.empty(tokens, 1, width, dtype=f32, device=device)
    square = torch.empty(tokens, 1, dtype=f32, device=device)
    mixes = torch.empty(tokens, width, dtype=f32, device=device)
    tile_kernels.mhc.mhc_reduce_partials_and_rmsnorm_fwd(parts.unsqueeze(2), squares.unsqueeze(2), product, square, mixes,
                                                         rms_group_size=fn.shape[1], rms_eps=norm_eps, n_splits=SPLITS)
    pre = torch.empty(tokens, hc, dtype=f32, device=device)
    post = torch.empty(tokens, hc, dtype=f32, device=device)
    raw = torch.empty(tokens, hc * hc, dtype=f32, device=device)
    tile_kernels.mhc.mhc_pre_split_mixes_fwd(mixes, scale, base, pre, post, raw, 2.0, eps)
    comb = torch.empty(tokens, hc, hc, dtype=f32, device=device)
    tile_kernels.mhc.mhc_sinkhorn_fwd(raw.view(tokens, hc, hc), comb, iters, eps)
    return pre, post, comb


def hc_pre(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.shape[0], x.shape[2], dtype=BF16, device=x.device)
    tile_kernels.mhc.mhc_pre_apply_mix_fwd(x.contiguous(), pre.contiguous(), out)
    return out


def hc_post(y: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    tokens, hc = post.shape
    return tile_kernels.mhc.mhc_post_fwd(y.contiguous().unsqueeze(0), residual.contiguous().unsqueeze(0),
                                         post.contiguous().view(1, tokens, hc, 1), comb.contiguous().unsqueeze(0))[0]


def moe_gate(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, topk: int, route_scale: float
             ) -> tuple[torch.Tensor, torch.Tensor]:
    """``DeepseekMoE.gate_forward`` without its fused gate: the logits, then ``moe_topk_gate``."""
    logits = F.linear(x.float(), weight.float())
    indices, weights = tile_kernels.moe.moe_topk_gate(logits, num_topk=topk, use_shared_as_routed=False,
                                                      num_shared_experts=0, routed_scaling_factor=route_scale,
                                                      ep_rank=0, bias=bias, scoring_func="sqrtsoftplus")
    return weights, indices.long()


def _experts(x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor, w13: torch.Tensor, p13: torch.Tensor,
             w2: torch.Tensor, p2: torch.Tensor, limit: float, fixed: bool) -> torch.Tensor:
    """``DeepseekMoE.routed_experts_forward`` between a dispatch and a combine on this one rank: the
    tokens quantized once, each (token, expert) pair's row laid out by expert in DeepGEMM's contiguous
    layout (an expert's rows start at a multiple of its alignment), two grouped GEMMs with TileKernels'
    SwiGLU between them, and each token's rows added up. ``fixed``: as many rows as the pairs can
    need, so that nothing is read back from the device."""
    tokens, topk = indices.shape
    experts, inter, dim = w13.shape[0], w13.shape[1] // 2, x.shape[-1]
    device = x.device
    y = torch.zeros(tokens, dim, dtype=torch.float32, device=device)
    if not tokens:
        return y
    align = deep_gemm.get_mk_alignment_for_contiguous_layout()
    if tile_kernels.config.get_token_alignment() != align:
        raise RuntimeError("DeepGEMM and TileKernels lay an expert's rows out with different alignments")
    pairs = tokens * topk
    flat = indices.reshape(-1)
    order = torch.argsort(flat, stable=True)
    expert, token = flat[order], order // topk
    counts = torch.zeros(experts, dtype=torch.int64, device=device).scatter_add_(0, flat, torch.ones_like(flat))
    padded = (counts + align - 1) // align * align
    starts = torch.cumsum(padded, 0) - padded
    first = torch.cumsum(counts, 0) - counts
    row = starts[expert] + torch.arange(pairs, device=device) - first[expert]
    ends = (starts + counts).to(torch.int32)
    # Every expert some token chose takes its tokens' rows and less than one alignment of padding.
    total = -(-(pairs + min(experts, pairs) * (align - 1)) // align) * align if fixed else int(padded.sum())

    data, scales = _cast(_rows(x), row_major=True)
    a = torch.zeros(total, dim, dtype=torch.uint8, device=device)
    a[row] = data.view(torch.uint8)[token]
    a_scales = torch.full((scales.shape[1], total), PADDING_SCALE, dtype=torch.int32, device=device)
    a_scales[:, row] = scales[token].T
    gate_up = torch.empty(total, 2 * inter, dtype=BF16, device=device)
    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous((a.view(E4M3), a_scales.T), (w13.view(torch.int8), packed_words(p13, 2 * inter)),
                                                   gate_up, ends, recipe_a=(1, BLOCK), recipe_b=(1, BLOCK),
                                                   use_psum_layout=True, expected_m_for_psum_layout=None)
    weight = torch.zeros(total, dtype=torch.float32, device=device)
    weight[row] = weights.reshape(-1)[order]
    hidden = tile_kernels.quant.swiglu_forward_and_per_token_cast(
        gate_up, fmt="e4m3", num_per_channels=BLOCK, round_sf=True, use_packed_ue8m0=True,
        use_tma_aligned_col_major_sf=True, psum_num_tokens_per_expert=ends, topk_weights=weight,
        clamp_value=limit if limit > 0 else None)
    down = torch.empty(total, dim, dtype=BF16, device=device)
    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(hidden, (w2.view(torch.int8), packed_words(p2, dim)), down, ends,
                                                   recipe_a=(1, BLOCK), recipe_b=(1, BLOCK), use_psum_layout=True,
                                                   expected_m_for_psum_layout=None)
    return y.index_add_(0, token, down[row].float())


def moe_experts(x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor, w13: torch.Tensor, p13: torch.Tensor,
                w2: torch.Tensor, p2: torch.Tensor, limit: float) -> torch.Tensor:
    """The routed experts on exactly the rows the step's tokens need (read back from the device)."""
    return _experts(x, weights, indices, w13, p13, w2, p2, limit, False)


def moe_experts_fixed(x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor, w13: torch.Tensor,
                      p13: torch.Tensor, w2: torch.Tensor, p2: torch.Tensor, limit: float) -> torch.Tensor:
    """The same with a number of rows that only the shapes decide: for a step replayed from a graph."""
    return _experts(x, weights, indices, w13, p13, w2, p2, limit, True)


def top(scores: torch.Tensor, k: int) -> torch.Tensor:
    """DeepSelect's top-k: the indices only, in no order."""
    if not k or not scores.numel():
        return torch.empty(*scores.shape[:-1], k, dtype=torch.int64, device=scores.device)
    rows = scores.reshape(-1, scores.shape[-1])
    per = deep_select.get_stride_requirement()[0] // rows.element_size()
    if rows.stride(0) % per or rows.stride(1) != 1:
        aligned = torch.empty(rows.shape[0], -(-rows.shape[1] // per) * per, dtype=rows.dtype, device=rows.device)
        aligned = aligned[:, :rows.shape[1]]
        aligned.copy_(rows)
        rows = aligned
    indices = deep_select.topk(rows, k, return_value=False, indices_type=torch.int64)[1]
    return indices.reshape(*scores.shape[:-1], k)


def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def pool(members: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    return (members * scores.softmax(dim=1)).sum(dim=1)


def engram_gate(x: torch.Tensor, kv: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor, eps: float
                ) -> torch.Tensor:
    """``EngramGate``: the two norm weights fused, then TileKernels' gate."""
    tokens, hc, dim = x.shape
    fused = tile_kernels.engram.fused_weight(q_weight, k_weight)
    return tile_kernels.engram.engram_gate_fwd(x.contiguous(), kv.contiguous().view(tokens, hc + 1, dim), fused, eps,
                                               1e-6, save_for_backward=False)[0]


# ---------------------------------------------------------------- the definitions over the caches

FORMATS = MappingProxyType({"fp8": V41_FORMAT, "latent": V41_FP4_FORMAT, "key": FP4_INDEX_KV_FORMAT})
"""The record formats (``kernels.ds1017_compose``) as DeepInfer's cache formats."""

INDEX_ROW_BYTES = 2 << 30
"""How many bytes of indexer logits a prefill computes at a time (``DEEPINFER_INDEX_ATTN_MEMORY_LIMIT``)."""


def _layout(fmt: str) -> KVCacheFormat:
    return FORMATS[fmt]


def _paged(cache: torch.Tensor, entries: int) -> torch.Tensor:
    """A cache ``[rows, bytes]`` as DeepInfer's ``[blocks, entries, 1, bytes]``."""
    return cache.view(-1, entries, 1, cache.shape[-1])


def _cast_args(fmt: str, back: bool = False) -> CastArgs:
    layout = _layout(fmt)
    return CastArgs(layout.quantized_dtype, layout.elements_per_sf, round_sf=layout.sf_format != "e4m3",  # pyright: ignore
                    sf_format=layout.sf_format, cast_back=back)  # pyright: ignore


def _rope(cos_sin: torch.Tensor, positions: torch.Tensor, rotary: int, dim: int, inverse: bool = False) -> RopeArgs:
    """DeepInfer's RoPE of the last ``rotary`` values, pairs of neighbours (not NeoX's halves)."""
    return RopeArgs(positions.contiguous(), cos_sin, rotary, False, offset=dim - rotary, conjugate=inverse)


def rope(x: torch.Tensor, cos_sin: torch.Tensor, positions: torch.Tensor, rotary: int, inverse: bool = False
         ) -> torch.Tensor:
    if not x.shape[0]:
        return x.clone()
    rows = x.reshape(x.shape[0], -1, x.shape[-1]).contiguous()
    out = fused_norm_rope_hadamard_cast_store_kvcache(rows, rope_args=_rope(cos_sin, positions, rotary, x.shape[-1], inverse))
    return cast(torch.Tensor, out).view(x.shape)


def transform(x: torch.Tensor, norm: torch.Tensor | None, eps: float, cos_sin: torch.Tensor | None,
              positions: torch.Tensor | None, rotary: int, fmt: str) -> torch.Tensor:
    """The fused operator with ``cast_back``: the prefill's window KV (``_prepare_swa_kv``)."""
    if not x.shape[0]:
        return x.clone()
    return cast(torch.Tensor, fused_norm_rope_hadamard_cast_store_kvcache(
        x.contiguous(), norm_args=None if norm is None else NormArgs(norm, eps),
        rope_args=None if cos_sin is None or positions is None else _rope(cos_sin, positions, rotary, x.shape[-1]),
        cast_args=_cast_args(fmt, back=True)))


def store_cache(x: torch.Tensor, norm: torch.Tensor | None, eps: float, cos_sin: torch.Tensor | None,
                positions: torch.Tensor | None, rotary: int, cache: torch.Tensor, rows: torch.Tensor, fmt: str,
                entries: int, normed: bool = False) -> torch.Tensor | None:
    """The fused operator into a cache (``KVCacheOutArgs``), or ``store_kv_cache`` for rows that need
    neither a norm nor a rotation (a prefill's window KV, transformed already)."""
    out = torch.empty_like(x) if normed else None
    if not x.shape[0]:
        return out
    slots = rows.to(torch.int32).contiguous()
    paged = _paged(cache, entries)
    if norm is None and cos_sin is None:
        store_kv_cache(x.contiguous(), paged, slots, layout=_layout(fmt))
        return out
    fused_norm_rope_hadamard_cast_store_kvcache(
        x.contiguous(), norm_args=None if norm is None else NormArgs(norm, eps),
        rope_args=None if cos_sin is None or positions is None else _rope(cos_sin, positions, rotary, x.shape[-1]),
        cast_args=_cast_args(fmt), out_args=KVCacheOutArgs(paged, slots, out=out, out_stage="norm", layout=_layout(fmt)))
    return out


def load_values(cache: torch.Tensor, entries: int, segments: torch.Tensor, total: int, fmt: str) -> torch.Tensor:
    if not total:
        return torch.empty(0, _layout(fmt).num_elements, dtype=torch.bfloat16, device=cache.device)
    result = load_paged_kv_cache(_paged(cache, entries), segments.to(torch.int32).contiguous(), total, layout=_layout(fmt))
    return result.bf16_data  # pyright: ignore


def load_keys(cache: torch.Tensor, entries: int, segments: torch.Tensor, total: int, dim: int
              ) -> tuple[torch.Tensor, torch.Tensor]:
    if not total:
        return (torch.empty(0, dim // 2, dtype=torch.uint8, device=cache.device),
                torch.empty(0, dim // BLOCK, dtype=torch.uint8, device=cache.device))
    result = load_paged_kv_cache(_paged(cache, entries), segments.to(torch.int32).contiguous(), total,
                                 layout=FP4_INDEX_KV_FORMAT)
    data, scales = result.quantized_data, result.sf_data
    return data.view(torch.uint8), scales.contiguous().view(torch.uint8).view(total, -1)  # pyright: ignore


def quant_index_q(q: torch.Tensor, cos_sin: torch.Tensor, positions: torch.Tensor, rotary: int
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """``MQA._quantize_q``: rotation and FP4 in one operator."""
    n, heads, dim = q.shape
    if not n:
        return (torch.empty(0, heads, dim // 2, dtype=torch.uint8, device=q.device),
                torch.empty(0, heads, dim // BLOCK, dtype=torch.uint8, device=q.device))
    data, scales = cast(tuple[torch.Tensor, torch.Tensor], fused_norm_rope_hadamard_cast_store_kvcache(
        q.contiguous(), rope_args=_rope(cos_sin, positions, rotary, dim), cast_args=_cast_args("key")))
    return data.view(torch.uint8), scales.contiguous().view(torch.uint8).view(n, heads, -1)


def _weights(weights: torch.Tensor) -> torch.Tensor:
    """``_align_mqa_weights``: rows 16-byte aligned for DeepGEMM's loads."""
    per = 16 // weights.element_size()
    if weights.stride(0) % per == 0 and weights.stride(-1) == 1:
        return weights
    out = torch.zeros(weights.shape[0], -(-weights.shape[1] // per) * per, dtype=weights.dtype, device=weights.device)
    return out[:, :weights.shape[1]].copy_(weights)


def _q(data: torch.Tensor, scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The FP4 queries as DeepGEMM takes them: packed values, and a word of four exponents a head."""
    return data.view(torch.int8), scales.contiguous().view(torch.int32).squeeze(-1)


def _top(scores: torch.Tensor, k: int, end: torch.Tensor, offset: torch.Tensor | None = None) -> torch.Tensor:
    """DeepSelect's top-k as the indexer calls it: indices ascending, ``-1`` after the valid ones."""
    return deep_select.topk(scores, k, sorted=False, end=end, indices_type=torch.int32, sorted_index=True,
                            output_idx_offset=offset, idx_oob_fill_value=-1, return_value=False)[1]


def _aligned(idx: torch.Tensor) -> torch.Tensor:
    """``idx`` with rows that start at multiples of 8 entries, as the candidate kernels take them:
    DeepInfer's top-k is always 512 wide, this one as wide as the step's longest row reaches."""
    if idx.stride(0) % 8 == 0 and idx.stride(1) == 1:
        return idx
    out = torch.empty(idx.shape[0], -(-idx.shape[1] // 8) * 8, dtype=idx.dtype, device=idx.device)[:, :idx.shape[1]]
    return out.copy_(idx)


def _padded(idx: torch.Tensor) -> torch.Tensor:
    """``idx`` with none after it up to a multiple of 128 entries, rows aligned: the TileLang kernels
    after the top-k lay a row out over 128 threads and are compiled for its length (DeepInfer's is
    always 512)."""
    return _aligned(F.pad(idx, (0, -idx.shape[1] % 128), value=-1))


def _candidates(logits: torch.Tensor, lens: torch.Tensor, topk_blocks: int, block: int) -> torch.Tensor:
    """``MQA._topk_candidate_block``."""
    scores, valid = candidate_block_max(logits, lens, block)
    return _top(scores, topk_blocks, valid)


def index_topk(q_data: torch.Tensor, q_scales: torch.Tensor, weights: torch.Tensor, k_data: torch.Tensor,
               k_scales: torch.Tensor, ks: torch.Tensor, ke: torch.Tensor, longest: int, topk: int, mode: str,
               candidates: torch.Tensor | None, topk_blocks: int, block: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``MQA._topk_prefill``: DeepGEMM's logits over the loaded keys (``fp8_fp4_mqa_logits``, or over the
    candidate blocks only with ``fp8_fp4_sparse_mqa_logits``), DeepSelect's top-k, query chunk by
    query chunk."""
    n, device = q_data.shape[0], q_data.device
    out = torch.full((n, topk), -1, dtype=torch.int64, device=device)
    picked = torch.full((n, topk_blocks), -1, dtype=torch.int64, device=device) if mode == "source" else None
    if not n or not k_data.shape[0] or not topk:
        return out, picked
    q = _q(q_data, q_scales)
    kv = (k_data.view(torch.int8), k_scales.contiguous().view(torch.int32).squeeze(-1))
    w = _weights(weights)
    ks32, ke32 = ks.to(torch.int32).contiguous(), ke.to(torch.int32).contiguous()
    klen = ke32 - ks32
    width = max(longest, 1)  # the logits' width: one compiled candidate kernel for every step (max_kv_len)
    chunk = max(8, min(8192, INDEX_ROW_BYTES // (width * 2) // 8 * 8))
    for a in range(0, n, chunk):
        c = slice(a, min(a + chunk, n))
        qc = (q[0][c], q[1][c])
        if mode == "user":
            assert candidates is not None
            cand = candidates[c].to(torch.int32).contiguous()
            meta = deep_gemm.get_sparse_mqa_logits_metadata(ks32[c], ke32[c], kv[0].shape[0],
                                                            cand + torch.div(ks32[c], block, rounding_mode="floor")[:, None],
                                                            qc[0].dtype, block, use_unaligned_ks=True)
            logits = deep_gemm.fp8_fp4_sparse_mqa_logits(qc, kv, w[c], meta, cand.shape[1], block, use_unaligned_ks=True)
            idx = _padded(_top(logits, topk, get_candidate_compact_lens(cand, klen[c].contiguous(), block)))
            restore_candidate_indices_inplace(idx, _aligned(cand), block, ks32[c])
            idx = idx[:, :topk]
        else:
            logits = deep_gemm.fp8_fp4_mqa_logits(qc, kv, w[c], ks32[c], ke32[c], max_seqlen_k=width)
            if picked is not None:
                picked[c] = _candidates(logits, klen[c].contiguous(), topk_blocks, block).long()
            idx = _top(logits, topk, klen[c].contiguous(), ks32[c])
        out[c] = idx.long()
    return out, picked


def index_topk_paged(q_data: torch.Tensor, q_scales: torch.Tensor, weights: torch.Tensor, cache: torch.Tensor,
                     entries: int, block_table: torch.Tensor, lens: torch.Tensor, width: int, topk: int, mode: str,
                     candidates: torch.Tensor | None, topk_blocks: int, block: int
                     ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``MQA._topk_decode``: DeepGEMM's paged logits (``fp8_fp4_paged_mqa_logits``, or
    ``fp8_fp4_paged_sparse_mqa_logits`` over the candidate blocks), DeepSelect's top-k, and
    ``convert_kv_index`` to cache rows. Nothing here reads the device back: a graph records it."""
    n, device = q_data.shape[0], q_data.device
    picked: torch.Tensor | None = None
    if not n or not topk:
        return torch.full((n, topk), -1, dtype=torch.int64, device=device), picked
    q = _q(q_data, q_scales)
    q = (q[0][:, None], q[1][:, None])
    paged = _paged(cache, entries)
    lens32 = lens.to(torch.int32).contiguous()
    table = block_table.to(torch.int32).contiguous()
    indices = torch.arange(n, dtype=torch.int32, device=device)
    w = _weights(weights)
    if mode == "user":
        assert candidates is not None
        cand = candidates.to(torch.int32).contiguous()
        meta = deep_gemm.get_paged_sparse_mqa_logits_metadata(lens32, table, indices, entries, cand, q[0].dtype, block)
        logits = deep_gemm.fp8_fp4_paged_sparse_mqa_logits(q, paged, w, meta, cand.shape[1], block)
        valid = get_candidate_compact_lens(cand, lens32, block)
        idx = _padded(_top(logits, topk, valid))
        restore_candidate_indices_inplace(idx, _aligned(cand), block)
        idx = idx[:, :topk]
    else:
        meta = deep_gemm.get_paged_mqa_logits_metadata(lens32[:, None], entries, deep_gemm.get_num_sms(), indices=indices)
        # As wide as the pages given (the batch's longest context): one compiled kernel for every step.
        logits = deep_gemm.fp8_fp4_paged_mqa_logits(q, paged, w, lens32[:, None], table, meta, table.shape[1] * entries,
                                                    indices=indices)
        if mode == "source":
            picked = _candidates(logits, lens32, topk_blocks, block).long()
        valid = lens32
        idx = _top(logits, topk, valid).contiguous()
    # convert_kv_index lays a row out over 128 threads: DeepInfer's lists are 512 long; a shorter one is
    # padded with none (an entry past min(valid, topk) stays none).
    idx = F.pad(idx, (0, -topk % 128), value=-1)[:, None].contiguous()
    convert_kv_index(idx, entries, valid.clamp_max(topk)[:, None].contiguous(), table)
    return idx[:, 0, :topk].long(), picked


def compress(score: torch.Tensor, kv: torch.Tensor, state: torch.Tensor, seq_slot: torch.Tensor, cu_q: torch.Tensor,
             cu_k: torch.Tensor, cu_c: torch.Tensor, c_seq: torch.Tensor, ratio: int) -> torch.Tensor:
    def i32(t: torch.Tensor) -> torch.Tensor:
        return t.to(torch.int32).contiguous()

    return compress_and_update_state_prefill(score.contiguous(), kv.contiguous(), state, i32(seq_slot), i32(c_seq),
                                             i32(cu_q), i32(cu_k), i32(cu_c), ratio)


def compress_decode(score: torch.Tensor, kv: torch.Tensor, state: torch.Tensor, slot: torch.Tensor,
                    context_lens: torch.Tensor, ratio: int) -> torch.Tensor:
    return compress_and_update_state_decode(score.contiguous(), kv.contiguous(), state, slot.to(torch.int32).contiguous(),
                                            context_lens.to(torch.int32).contiguous(), 1, ratio)


def shared_expert(x: torch.Tensor, w13: torch.Tensor, s13: torch.Tensor, p13: torch.Tensor, w2: torch.Tensor,
                  s2: torch.Tensor, p2: torch.Tensor, limit: float) -> torch.Tensor:
    """``DeepseekMoE.shared_forward``: one GEMM over ``w1`` and ``w3`` (``shared_gate_up_proj``), TileKernels'
    SwiGLU with the FP8 cast, the GEMM of ``w2`` on what it cast. The SwiGLU is quantized from fp32 (as
    the routed experts' are here, see the module's docstring)."""
    rows = _rows(x)
    gate_up = linear_fp8(rows, w13, s13, p13)
    if not rows.shape[0]:
        return torch.empty(*x.shape[:-1], w2.shape[0], dtype=BF16, device=x.device)
    hidden = tile_kernels.quant.swiglu_forward_and_per_token_cast(
        gate_up.contiguous(), fmt="e4m3", num_per_channels=BLOCK, round_sf=True, use_packed_ue8m0=True,
        use_tma_aligned_col_major_sf=True, clamp_value=limit if limit > 0 else None)
    out = _out(rows.shape[0], w2.shape[0], BF16, x.device)
    deep_gemm.fp8_fp4_gemm_nt(hidden, (w2.view(E4M3), packed_words(p2, w2.shape[0])), out, None, recipe=(1, 1, BLOCK))
    return out.reshape(*x.shape[:-1], w2.shape[0])


def engram_fetch(group: Any, ids: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows of an Engram table split over the ranks, through the group's collectives
    (``kernels.ds1017_ep``): DeepInfer's own way, ``deep_ep.EngramBuffer``, is not wired yet."""
    return ep.engram_fetch(group, ids, weight, scale)


def engram_hash(tokens: torch.Tensor, multipliers: torch.Tensor, primes: torch.Tensor, offsets: torch.Tensor
                ) -> torch.Tensor:
    """TileKernels' ``engram_hash``."""
    out = tile_kernels.engram.engram_hash(tokens.to(torch.int32).contiguous(), multipliers.contiguous(),
                                          primes.to(torch.int32).contiguous(), offsets.to(torch.int32).contiguous())
    return out.permute(1, 0, 2).long()
