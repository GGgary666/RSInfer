"""DS 1017's number formats and reference kernels on the CPU (``kernels.ds1017_formats``,
``kernels.ds1017``). What they are held to here: the formats' definitions, plain float64 arithmetic
on what the formats hold, and their own invariants. The reference implementation shipped with the
checkpoint needs a GPU: ``tests/gpu/test_ds1017.py`` holds them to its kernels."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from kernels import ds1017_formats as formats
from kernels.ds1017 import ops

BF16 = torch.bfloat16


def rand(*shape: int, seed: int = 0, scale: float = 1.0) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed)) * scale


def test_every_fp8_byte_is_the_value_torch_gives_it() -> None:
    every = torch.arange(256, dtype=torch.uint8)
    ours, torchs = formats._table("fp8", every.device), every.view(torch.float8_e4m3fn).float()
    finite = torch.isfinite(torchs)
    assert torch.equal(ours[finite], torchs[finite]) and int(finite.sum()) == 254  # e4m3fn: two NaN bytes
    scales = torch.arange(255, dtype=torch.uint8)
    assert torch.equal(formats._table("e8m0", scales.device)[:255], scales.view(torch.float8_e8m0fnu).float())


def test_rounding_to_a_format_takes_the_nearest_value_and_ties_to_even() -> None:
    assert formats.to_fp4(torch.tensor([0.2, 0.25, 0.3, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.1, -5.0, 6.0])).tolist() == \
        [0.0, 0.0, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, -4.0, 6.0]
    # e4m3: three mantissa bits, so steps of 1/8 from 1 to 2, of 32 from 256 to 448, of 2**-9 below 2**-6.
    assert formats.to_fp8(torch.tensor([1.0625, 1.1875, 300.0, 448.0, 2.0 ** -9 * 2.5, -1.0625])).tolist() == \
        [1.0, 1.25, 288.0, 448.0, 2.0 ** -9 * 2, -1.0]
    amax = torch.tensor([1.0, 448.0, 449.0, 896.0, 0.3])
    assert formats.round_scale(amax, 1 / 448).tolist() == [2.0 ** -8, 1.0, 2.0, 2.0, 2.0 ** -10]


@pytest.mark.parametrize(("quantize", "dequant"), [(formats.quantize_fp8, formats.dequant_fp8),
                                                   (formats.quantize_fp4, formats.dequant_fp4),
                                                   (formats.quantize_rows_fp8, formats.dequant_rows_fp8)])
def test_a_quantized_weight_holds_values_of_its_format_and_quantizes_to_itself(
        quantize: Callable[..., Any], dequant: Callable[..., Any]) -> None:
    w = rand(64, 96, scale=0.05)
    data, scale = quantize(w)
    held = dequant(data, scale)
    assert data.dtype == scale.dtype == torch.uint8 and held.shape == w.shape
    worst = 0.25 if quantize is formats.quantize_fp4 else 1 / 16  # half a step of the format, relative to a block's largest
    assert float((held - w).abs().max() / w.abs().max()) < worst
    again, again_scale = quantize(held)
    assert torch.equal(dequant(again, again_scale), held)


def test_fake_quantization_gives_what_the_stored_form_would_hold() -> None:
    x = (rand(9, 64) * torch.exp(rand(9, 1, seed=1) * 3)).to(BF16)
    data, scale = formats.quantize_rows_fp8(x.float())
    assert torch.equal(ops.fake_quant_fp8(x).float(), formats.dequant_rows_fp8(data, scale))
    assert torch.equal(ops.fake_quant_fp8(ops.fake_quant_fp8(x)), ops.fake_quant_fp8(x))
    data, scale = formats.quantize_fp4(x.float())
    assert torch.equal(ops.fake_quant_fp4(x, 32).float(), formats.dequant_fp4(data, scale))
    zero = torch.zeros(2, 32, dtype=BF16)
    assert torch.equal(ops.fake_quant_fp8(zero), zero) and torch.equal(ops.fake_quant_fp4(zero, 16, True), zero)


def test_a_quantized_linear_is_the_exact_product_of_what_the_formats_hold_rounded_once() -> None:
    x, w = rand(7, 96).to(BF16), rand(64, 96, seed=1, scale=0.1)
    for quantize, dequant, linear in ((formats.quantize_fp8, formats.dequant_fp8, ops.linear_fp8),
                                      (formats.quantize_fp4, formats.dequant_fp4, ops.linear_fp4)):
        data, scale = quantize(w)
        exact = ops.fake_quant_fp8(x).double() @ dequant(data, scale).double().T
        got = linear(x, data, scale)
        assert got.dtype == BF16
        unit = torch.ldexp(torch.ones_like(exact), torch.frexp(exact.abs())[1] - 8)  # bf16's last place there
        assert float(((got.double() - exact).abs() / unit).max()) <= 0.51


def test_a_sum_has_one_order_whatever_surrounds_it() -> None:
    v = rand(5, 37)
    assert torch.equal(ops.total(v), torch.stack([ops.total(row) for row in v]))
    assert torch.allclose(ops.total(v), v.double().sum(-1).float(), rtol=1e-5)
    assert torch.equal(ops.total(v, dim=0), ops.total(v.T))
    assert ops.total(torch.zeros(3, 0)).tolist() == [0.0, 0.0, 0.0]


def per_row(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> None:
    """``fn`` gives each row what it gives it alone and beside any other rows, bit for bit."""
    whole = fn(x)
    for size in (1, 3):
        parts = torch.cat([fn(x[a:a + size]) for a in range(0, x.shape[0], size)])
        assert torch.equal(parts, whole), size


def test_a_row_gets_the_same_result_whatever_else_is_in_the_step() -> None:
    x = rand(11, 96).to(BF16)
    data, scale = formats.quantize_fp8(rand(64, 96, seed=1, scale=0.1))
    per_row(lambda t: ops.linear_fp8(t, data, scale), x)
    per_row(lambda t: ops.linear(t, rand(8, 96, seed=2)), x)
    per_row(lambda t: ops.rms_norm(t, rand(96, seed=3).to(BF16), 1e-20), x)
    per_row(lambda t: ops.moe_gate(t, rand(4, 96, seed=4), rand(4, seed=5), 2, 1.5)[0], x)
    stream = rand(11, 4, 24).to(BF16)
    for part in range(3):
        per_row(lambda t, part=part: ops.hc_mixes(t, rand(24, 96, seed=6), rand(3, seed=7), rand(24, seed=8), 1e-20, 20,
                                                  1e-6)[part], stream)
    q, source = rand(11, 2, 32, seed=9).to(BF16), rand(40, 32, seed=10).to(BF16)
    idx = torch.randint(-1, 40, (11, 70), generator=torch.Generator().manual_seed(11))
    sink = rand(2, seed=12)
    per_row(lambda rows: ops.sparse_attn(q[rows], source, idx[rows], sink, 32 ** -0.5), torch.arange(11))


def test_sparse_attention_is_a_softmax_over_the_listed_keys_and_a_sink() -> None:
    q, source, sink = rand(3, 2, 32).to(BF16), rand(50, 32, seed=1).to(BF16), rand(2, seed=2)
    idx = torch.tensor([[4, 9, -1, 30, 7], [-1, -1, -1, -1, -1], [0, 1, 2, 3, 49]])
    got = ops.sparse_attn(q, source, idx, sink, 32 ** -0.5).float()
    for t in range(3):
        keys = source[idx[t][idx[t] >= 0]].double()
        scores = torch.cat([q[t].double() @ keys.T * 32 ** -0.5, sink.double()[:, None]], dim=1)
        want = torch.softmax(scores, dim=1)[:, :-1] @ keys  # the sink takes weight and adds nothing
        assert torch.allclose(got[t], want.float(), atol=2e-2, rtol=2e-2), t
    assert not got[1].any()  # no key: all zero, as the reference's kernel
    # More keys than one block of 64: the running maximum over the blocks gives the same softmax.
    idx = torch.randint(0, 50, (3, 150), generator=torch.Generator().manual_seed(3))
    got = ops.sparse_attn(q, source, idx, sink, 32 ** -0.5).float()
    keys = source[idx].double()
    scores = torch.cat([torch.einsum("thd,tkd->thk", q.double(), keys) * 32 ** -0.5,
                        sink.double()[None, :, None].expand(3, -1, -1)], dim=2)
    want = torch.einsum("thk,tkd->thd", torch.softmax(scores, dim=2)[..., :-1], keys)
    assert torch.allclose(got, want.float(), atol=2e-2, rtol=2e-2)


def test_the_mixing_matrix_is_doubly_stochastic_after_sinkhorn() -> None:
    pre, post, comb = ops.hc_mixes(rand(6, 4, 24).to(BF16), rand(24, 96, seed=1), rand(3, seed=2), rand(24, seed=3),
                                   1e-20, 20, 1e-6)
    assert pre.shape == post.shape == (6, 4) and comb.shape == (6, 4, 4)
    assert bool((pre > 0).all() and (pre < 1.01).all() and (post > 0).all() and (post < 2).all())
    # The last of the 20 rounds scales the columns; the rows are then as close as 20 rounds get them.
    assert torch.allclose(comb.sum(-2), torch.ones(6, 4), atol=1e-4) and torch.allclose(comb.sum(-1), torch.ones(6, 4), atol=3e-2)


def experts(count: int = 4, dim: int = 64, inter: int = 32) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for m, (n, k) in enumerate(((inter, dim), (dim, inter), (inter, dim))):
        pairs = [formats.quantize_fp4(rand(n, k, seed=10 * m + e, scale=k ** -0.5)) for e in range(count)]
        out += [torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs])]
    return out


def test_routing_weights_come_from_the_scores_and_the_bias_only_picks() -> None:
    x, gate = rand(9, 64).to(BF16), rand(4, 64, seed=1, scale=0.2)
    weights, indices = ops.moe_gate(x, gate, torch.zeros(4), 2, 1.5)
    assert torch.allclose(weights.sum(-1), torch.full((9,), 1.5), atol=1e-5)
    biased, picked = ops.moe_gate(x, gate, torch.tensor([0.0, 0.0, 100.0, 100.0]), 2, 1.5)
    assert set(picked.flatten().tolist()) == {2, 3}
    scores = torch.nn.functional.softplus(x.float() @ gate.T).sqrt()
    want = scores.gather(1, picked)
    assert torch.allclose(biased, want / want.sum(-1, keepdim=True) * 1.5, atol=1e-5)
    assert indices.shape == (9, 2)


def test_the_experts_run_on_every_token_with_zero_weight_sum_what_the_chosen_experts_sum() -> None:
    x = rand(9, 64).to(BF16)
    weights, indices = ops.moe_gate(x, rand(4, 64, seed=1, scale=0.2), torch.zeros(4), 2, 1.5)
    w = experts()
    w1, s1, w2, s2, w3, s3 = w
    chosen = ops.moe_experts(x, weights, indices, w1, s1, w2, s2, w3, s3, 10.0)
    assert chosen.dtype == torch.float32 and bool(chosen.any())
    assert torch.equal(ops.moe_experts_masked(x, weights, indices, w1, s1, w2, s2, w3, s3, 10.0), chosen)
    # One token through its two experts by hand: the gate branch clamped from above, the up branch on both sides.
    t = 5
    xq = ops.fake_quant_fp8(x[t:t + 1]).double()
    want = torch.zeros(1, 64, dtype=torch.float64)
    for k in range(2):
        e = int(indices[t, k])
        gate = (xq @ formats.dequant_fp4(w[0][e], w[1][e]).double().T).to(BF16).float().clamp(max=10.0)
        up = (xq @ formats.dequant_fp4(w[4][e], w[5][e]).double().T).to(BF16).float().clamp(-10.0, 10.0)
        hidden = (weights[t, k] * (torch.nn.functional.silu(gate) * up)).to(BF16)
        want += (ops.fake_quant_fp8(hidden).double() @ formats.dequant_fp4(w[2][e], w[3][e]).double().T).to(BF16).double()
    assert torch.allclose(chosen[t:t + 1].double(), want, atol=2e-2, rtol=2e-2)


def test_pooling_and_the_engram_gate_follow_their_formulas() -> None:
    members, scores = rand(5, 2, 32), rand(5, 2, 32, seed=1)
    assert torch.allclose(ops.pool(members, scores), (members * scores.softmax(dim=1)).sum(dim=1), atol=1e-6)
    x, kv = rand(3, 4, 32).to(BF16), rand(3, 5 * 32, seed=1).to(BF16)
    out = ops.engram_gate(x, kv, torch.ones(4, 32, dtype=BF16), torch.ones(4, 32, dtype=BF16), 1e-20).float()
    key, value = kv[:, :128].float().view(3, 4, 32), kv[:, 128:].float()
    h = x.float()
    cos = (h * key).sum(-1) / (h.square().mean(-1).sqrt() * key.square().mean(-1).sqrt()) / 32 ** 0.5
    gate = torch.sigmoid(torch.copysign(cos.abs().clamp_min(1e-6).sqrt(), cos))
    assert torch.allclose(out, h + gate[..., None] * value[:, None, :], atol=2e-2, rtol=2e-2)


def test_the_rotary_table_rotates_and_rotates_back() -> None:
    table = formats.rope_table(16, 40, 0, 10000.0, 16, 32, 1, torch.device("cpu"))
    assert table.shape == (40, 8) and torch.allclose(table.abs(), torch.ones(40, 8))
    yarn = formats.rope_table(16, 40, 8, 160000.0, 16, 32, 1, torch.device("cpu"))
    assert not torch.allclose(yarn, formats.rope_table(16, 40, 0, 160000.0, 16, 32, 1, torch.device("cpu")))
    x = rand(6, 2, 32)
    freqs = table[torch.tensor([0, 1, 5, 9, 20, 39])]
    turned = ops.rope(x, freqs, 16)
    assert torch.equal(turned[..., :16], x[..., :16]) and torch.equal(turned[0], x[0])  # position 0 turns nothing
    assert not torch.allclose(turned[1:, :, 16:], x[1:, :, 16:])
    assert torch.allclose(ops.rope(turned, freqs, 16, True), x, atol=1e-5)
