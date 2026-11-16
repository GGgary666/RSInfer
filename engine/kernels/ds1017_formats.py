# Copyright (c) 2023 DeepSeek
# SPDX-License-Identifier: MIT
"""DS 1017's number formats: FP8 (e4m3) and FP4 (e2m1) values with ue8m0 or e4m3 scales, as the
checkpoint stores weights and as the model keeps its KV, and the rotary table. A library shared by
``kernels.ds1017`` (the reference kernels), ``layers.ds1017``, ``models.ds1017`` and the deploy recipe
(not a unit). After ``inference/kernel.py``, ``inference/convert.py`` and ``inference/model.py`` of
deepseek-ai/DeepSeek-V4.1-Flash-1017 (LICENSE.deepseek in ``kernels/ds1017``).

Byte layouts (the checkpoint's): an FP8 weight ``[N, K]`` of e4m3 bytes with ue8m0 scales
``[ceil(N / 32), ceil(K / 32)]``; an FP4 weight ``[N, K / 2]`` bytes, two e2m1 values a byte, the
low nibble first, with ue8m0 scales ``[N, K / 32]``; an Engram table ``[rows, K]`` of e4m3 bytes with
ue8m0 scales ``[rows, K / 32]``.

The records the states hold (DeepInfer's KV cache formats, which its kernels read): a window KV
``[D + D / 32]`` bytes, ``D`` e4m3 values and then their ue8m0 scales, one per 32; a compressed latent
``[D / 2 + D / 16]`` bytes, ``D`` e2m1 values (two a byte, the low nibble first) and then their e4m3
scales, one per 16; an indexer's key as ``D / 2`` bytes of e2m1 values and ``D / 32`` ue8m0 scales,
which a page keeps in two sections, every key's values and then every key's scales (``key_sections``).

The layout a GEMM kernel reads its weight's scales in (``pack_scales``; DeepGEMM's on SM 10.0, which
DeepInfer keeps beside each weight): one scale per row and 32 values, four ue8m0 exponents to an
int32, column-major. The model keeps the FP8 weights' block scales in it beside the checkpoint's, and
the experts' scales in it only.
"""

from __future__ import annotations

import functools
import math

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
BLOCK = 32
"""One FP8 activation scale per 32 values; one weight scale per 32 x 32 (FP8) or per 32 along K (FP4)."""

FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _fp8_values() -> list[float]:
    """Every e4m3fn byte's value: sign, 4 exponent bits (bias 7), 3 mantissa bits; subnormals below 2**-6."""
    out: list[float] = []
    for b in range(256):
        sign, exp, man = -1.0 if b >> 7 else 1.0, (b >> 3) & 0xF, b & 7
        out.append(sign * (man / 8 * 2.0 ** -6 if exp == 0 else (1 + man / 8) * 2.0 ** (exp - 7)))
    return out


@functools.lru_cache(maxsize=None)
def _table(name: str, device: torch.device) -> torch.Tensor:
    """Every byte's (or nibble's) value in one of the formats, on ``device``."""
    values = {"fp8": _fp8_values, "fp4": lambda: list(FP4_VALUES),
              "e8m0": lambda: [2.0 ** (b - 127) for b in range(255)] + [float("nan")]}[name]()
    return torch.tensor(values, dtype=torch.float32, device=device)


def round_scale(amax: torch.Tensor, max_inv: float) -> torch.Tensor:
    """``fast_round_scale``: ``2 ** ceil(log2(amax * max_inv))``, from the float's own exponent."""
    mantissa, exponent = torch.frexp(amax * max_inv)  # amax * max_inv = mantissa * 2 ** exponent, mantissa in [0.5, 1)
    return torch.ldexp(torch.ones_like(mantissa), exponent - (mantissa == 0.5).to(exponent.dtype))


def to_fp8(v: torch.Tensor) -> torch.Tensor:
    """fp32 within +-448 to the nearest e4m3 value, ties to even, as fp32."""
    a = v.abs()
    _, exponent = torch.frexp(a)
    step = torch.ldexp(torch.ones_like(a), (exponent - 1).clamp_min(-6) - 3)
    return torch.copysign(torch.round(a / step) * step, v)


def to_fp4(v: torch.Tensor) -> torch.Tensor:
    """fp32 within +-6 to the nearest e2m1 value (0, 0.5, 1, 1.5, 2, 3, 4, 6), ties to even, as fp32."""
    a = v.abs()
    step = torch.where(a < 2, 0.5, torch.where(a < 4, 1.0, 2.0)).to(a.dtype)
    return torch.copysign(torch.round(a / step) * step, v)


def dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """An FP8 weight ``[..., N, K]`` (e4m3 bytes) with its block scales (ue8m0 bytes) to fp32."""
    n, k = weight.shape[-2:]
    values = _table("fp8", weight.device)[weight.to(torch.int32)]
    scales = _table("e8m0", weight.device)[scale.to(torch.int32)]
    return values * scales.repeat_interleave(BLOCK, -2)[..., :n, :].repeat_interleave(BLOCK, -1)[..., :k]


def dequant_rows_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP8 rows ``[..., K]`` with one ue8m0 scale per 32 values of a row ``[..., K / 32]`` (the Engram
    tables) to fp32."""
    values = _table("fp8", weight.device)[weight.to(torch.int32)].unflatten(-1, (-1, BLOCK))
    return (values * _table("e8m0", weight.device)[scale.to(torch.int32)].unsqueeze(-1)).flatten(-2)


def dequant_fp4(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """An FP4 weight ``[..., N, K / 2]`` (two e2m1 values a byte, low nibble first) with its scales
    ``[..., N, K / 32]`` (ue8m0 bytes) to fp32 ``[..., N, K]``."""
    table = _table("fp4", weight.device)
    w = weight.to(torch.int32)
    values = torch.stack([table[w & 0xF], table[(w >> 4) & 0xF]], dim=-1).flatten(-2)
    return values * _table("e8m0", weight.device)[scale.to(torch.int32)].repeat_interleave(BLOCK, -1)


def packed_shape(n: int, k: int) -> tuple[int, int, int]:
    """The shape of ``pack_scales``' result for a weight ``[n, k]``."""
    return -(-k // (4 * BLOCK)), -(-n // 4) * 4, 4


def pack_scales(rows: torch.Tensor) -> torch.Tensor:
    """A weight's scales by row ``[..., N, K / 32]`` (ue8m0 bytes) to ``[..., K / 128, N, 4]`` bytes:
    as int32 ``[..., K / 128, N]`` one word holds four consecutive exponents of a row (the first in
    the low byte), stored column by column. Rows are padded to a multiple of 4 and exponents to a
    multiple of 4 with the exponent of 1."""
    n, k32 = rows.shape[-2:]
    padded = torch.nn.functional.pad(rows, (0, -k32 % 4, 0, -n % 4), value=127)
    return padded.unflatten(-1, (-1, 4)).transpose(-3, -2).contiguous()


def unpack_scales(packed: torch.Tensor, n: int, k32: int) -> torch.Tensor:
    """``pack_scales`` undone: ``[..., n, k32]`` bytes."""
    return packed.transpose(-3, -2).flatten(-2)[..., :n, :k32]


def packed_words(packed: torch.Tensor, n: int) -> torch.Tensor:
    """``pack_scales``' bytes as the int32 ``[..., n, K / 128]`` a kernel takes: a view, row stride 1."""
    return packed.view(torch.int32).squeeze(-1).transpose(-2, -1)[..., :n, :]


def mega_rows(n: int, halves: bool, scales: bool) -> torch.Tensor:
    """Row ``i`` of a weight ``[n, k]`` (or of its scales by row) in the layout DeepGEMM's fused MoE
    kernel reads is row ``mega_rows(...)[i]`` of the checkpoint's order
    (``deep_gemm.transform_weights_for_mega_moe``, DeepGEMM 1fc9c66, MIT):

    - ``halves`` (``w13``, an expert's ``w1`` over its ``w3``): the two halves interleaved eight rows
      at a time, so that a gate row and its up row are in one tile;
    - ``scales``: within every 128 rows, row ``a * 32 + b`` (``a`` < 4) at ``b * 4 + a``, the order
      the kernel's scale tiles are loaded in."""
    rows = torch.arange(n)
    if halves:
        tile, at = rows // 16, rows % 16
        rows = torch.where(at < 8, tile * 8 + at, n // 2 + tile * 8 + at - 8)
    if scales:
        if n % 128:
            raise ValueError(f"the fused MoE kernel takes scales of weights with a multiple of 128 rows, not {n}")
        chunk, at = torch.arange(n) // 128, torch.arange(n) % 128
        rows = rows[chunk * 128 + (at % 4) * 32 + at // 4]
    return rows


def expert_layout(w13: torch.Tensor, p13: torch.Tensor, w2: torch.Tensor, p2: torch.Tensor, mega: bool) -> None:
    """An expert stack's weights and packed scales, in place, from the checkpoint's row order to the
    fused MoE kernel's (``mega``) or back. Row permutations of each expert: which slot holds which
    expert is not touched."""
    def place(t: torch.Tensor, rows: torch.Tensor, dim: int) -> None:
        rows = (rows if mega else rows.argsort()).to(t.device)
        t.copy_(t.index_select(dim, rows))

    n13, n2 = w13.shape[1], w2.shape[1]
    place(w13, mega_rows(n13, True, False), 1)
    place(p13, mega_rows(n13, True, True), 2)  # packed scales: [experts, K / 128, rows, 4]
    place(p2, mega_rows(n2, False, True), 2)


def shared_layout(w13: torch.Tensor, p13: torch.Tensor, p2: torch.Tensor, mega: bool) -> None:
    """``expert_layout`` for the shared expert's FP8 weights ``w13`` ``[2 * inter, dim]`` and the packed
    scales of both its matrices (its block scales, which only the references read, stay)."""
    def place(t: torch.Tensor, rows: torch.Tensor, dim: int) -> None:
        rows = (rows if mega else rows.argsort()).to(t.device)
        t.copy_(t.index_select(dim, rows))

    place(w13, mega_rows(w13.shape[0], True, False), 0)
    place(p13, mega_rows(w13.shape[0], True, True), 1)  # packed scales: [K / 128, rows, 4]
    place(p2, mega_rows(p2.shape[1], False, True), 1)


def block_rows(scale: torch.Tensor, n: int) -> torch.Tensor:
    """An FP8 weight's 32 x 32 block scales ``[N / 32, K / 32]`` as one row of scales per weight row."""
    return scale.repeat_interleave(BLOCK, dim=-2)[..., :n, :]


def rope_table(dim: int, positions: int, original: int, base: float, factor: float, beta_fast: int,
               beta_slow: int, device: torch.device) -> torch.Tensor:
    """``precompute_freqs_cis``: ``[positions, dim / 2]`` complex; YaRN when ``original > 0``."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    if original > 0:
        def corrected_dim(rotations: int) -> float:
            return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32, device=device) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(positions, device=device), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


# ---------------------------------------------------------------------- quantizing weights
# The inverse of the dequantizers, for random checkpoints in the checkpoint's formats (deploy, tests).


def _fp8_bytes(values: torch.Tensor) -> torch.Tensor:
    """e4m3 values (as ``to_fp8`` gives them) to their bytes."""
    a = values.abs()
    _, exponent = torch.frexp(a)
    e = (exponent - 1).clamp_min(-6)
    normal = a >= 2.0 ** -6
    mantissa = torch.where(normal, (torch.ldexp(a, -e) - 1) * 8, a * 2.0 ** 9)
    field = torch.where(normal, e + 7, torch.zeros_like(e))
    return ((torch.signbit(values).to(torch.int32) << 7) | (field.to(torch.int32) << 3)
            | mantissa.round().to(torch.int32)).to(torch.uint8)


def _e8m0_bytes(scales: torch.Tensor) -> torch.Tensor:
    """Powers of two to ue8m0 bytes."""
    _, exponent = torch.frexp(scales)
    return (exponent - 1 + 127).to(torch.uint8)


def quantize_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """An fp32 weight ``[N, K]`` (multiples of 32) to FP8 bytes and its 32 x 32 block scales' bytes."""
    n, k = weight.shape
    blocks = weight.float().unflatten(0, (-1, BLOCK)).unflatten(-1, (-1, BLOCK)).transpose(1, 2)
    scale = round_scale(blocks.abs().amax((-1, -2), keepdim=True).clamp_min(1e-4), 1 / FP8_MAX)
    q = to_fp8((blocks / scale).clamp(-FP8_MAX, FP8_MAX)).transpose(1, 2).reshape(n, k)
    return _fp8_bytes(q), _e8m0_bytes(scale.flatten(1))


def quantize_rows_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """An fp32 table ``[rows, K]`` to FP8 bytes with one scale per 32 values of a row."""
    blocks = weight.float().unflatten(-1, (-1, BLOCK))
    scale = round_scale(blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4), 1 / FP8_MAX)
    return _fp8_bytes(to_fp8((blocks / scale).clamp(-FP8_MAX, FP8_MAX)).flatten(-2)), _e8m0_bytes(scale.squeeze(-1))


def _fp4_bytes(q: torch.Tensor) -> torch.Tensor:
    """e2m1 values (as ``to_fp4`` gives them) ``[..., K]`` to bytes ``[..., K / 2]``, the low nibble first."""
    a = q.abs()
    code = torch.where(a < 2, a * 2, torch.where(a < 4, a + 2, a / 2 + 4)).to(torch.int32) | (torch.signbit(q).to(torch.int32) << 3)
    return (code[..., 0::2] | (code[..., 1::2] << 4)).to(torch.uint8)


LATENT_BLOCK = 16
"""A compressed latent has one (e4m3) scale per 16 values."""


def pack_kv(values: torch.Tensor) -> torch.Tensor:
    """Window KV ``[..., D]`` to their records ``[..., D + D / 32]``: FP8 per 32 with power-of-two
    scales. Of values ``fake_quant_fp8`` gave, the record holds exactly those values."""
    data, scale = quantize_rows_fp8(values)
    return torch.cat([data, scale], dim=-1)


def unpack_kv(records: torch.Tensor) -> torch.Tensor:
    """``pack_kv``'s records back to fp32 ``[..., D]``."""
    dim = records.shape[-1] * BLOCK // (BLOCK + 1)
    return dequant_rows_fp8(records[..., :dim], records[..., dim:])


def pack_latent(values: torch.Tensor) -> torch.Tensor:
    """Compressed latents ``[..., D]`` to their records ``[..., D / 2 + D / 16]``: FP4 per 16 with e4m3
    scales (``fp4_act_quant`` with an e4m3 scale). Of values that quantizer gave, exactly those values."""
    blocks = values.float().unflatten(-1, (-1, LATENT_BLOCK))
    scale = to_fp8((blocks.abs().amax(-1, keepdim=True).clamp_min(6 * 2.0 ** -9) / FP4_MAX).clamp_max(FP8_MAX))
    q = to_fp4((blocks / scale).clamp(-FP4_MAX, FP4_MAX)).flatten(-2)
    return torch.cat([_fp4_bytes(q), _fp8_bytes(scale.squeeze(-1))], dim=-1)


def unpack_latent(records: torch.Tensor) -> torch.Tensor:
    """``pack_latent``'s records back to fp32 ``[..., D]``."""
    half = records.shape[-1] * LATENT_BLOCK // (LATENT_BLOCK + 2)
    table = _table("fp4", records.device)
    w = records[..., :half].to(torch.int32)
    values = torch.stack([table[w & 0xF], table[(w >> 4) & 0xF]], dim=-1).flatten(-2)
    scales = _table("fp8", records.device)[records[..., half:].to(torch.int32)]
    return values * scales.repeat_interleave(LATENT_BLOCK, -1)


def key_sections(cache: torch.Tensor, page: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The indexer keys' cache ``[slots, dim / 2 + dim / 32]`` bytes as its pages hold it: the values
    ``[pages, page, dim / 2]`` and the scales ``[pages, page, dim / 32]`` of each page's keys (views).
    A key's values and scales are ``quantize_fp4``'s; ``dequant_fp4`` reads them."""
    blocks = cache.view(cache.shape[0] // page, -1)
    return (blocks[:, :page * (dim // 2)].unflatten(1, (page, dim // 2)),
            blocks[:, page * (dim // 2):].unflatten(1, (page, dim // BLOCK)))


def quantize_fp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """An fp32 weight ``[N, K]`` to FP4 bytes ``[N, K / 2]`` (low nibble first) and its scales' bytes
    ``[N, K / 32]``."""
    blocks = weight.float().unflatten(-1, (-1, BLOCK))
    scale = round_scale(blocks.abs().amax(-1, keepdim=True).clamp_min(6 * 2.0 ** -126), 1 / FP4_MAX)
    q = to_fp4((blocks / scale).clamp(-FP4_MAX, FP4_MAX)).flatten(-2)
    return _fp4_bytes(q), _e8m0_bytes(scale.squeeze(-1))
