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
- ``hc_mixes``: the product with the stream's copies is a TF32 GEMM (``tf32_hc_prenorm_gemm``), so the
  coefficients agree to about 1e-4, not to fp32's last bits.
- ``fake_quant_fp4`` with an e4m3 scale (the compressed KV): TileKernels multiplies by the scale's
  reciprocal where the reference implementation divides by the scale, so a value exactly halfway
  between two FP4 values lands on the other one: about one element in 2000, one FP4 step apart.

Three definitions have no kernel here yet and are the torch calls a GPU runs as they stand (``silu``,
``pool``, ``index_score``): the libraries' indexer and compressor kernels read the states in
DeepInfer's byte layouts, which the engine does not keep yet.

The weights arrive as the checkpoint's bytes (``uint8``); a weight's scales as
``formats.pack_scales`` lays them out, which is what DeepGEMM reads on SM 10.0. Activations are bf16.
"""
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import deep_gemm
import flash_mla
import tile_kernels
import torch
import torch.nn.functional as F
from deep_select import interface as deep_select

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
