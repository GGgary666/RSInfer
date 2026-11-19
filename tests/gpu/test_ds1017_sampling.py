"""SM100 checks for the optimized seeded sampler, vocabulary cap and lossless linear cache."""

from __future__ import annotations

import pytest
import torch

from kernels.deepinfer import ops
from kernels.deepinfer.mega import Mega
from kernels.deepinfer.sampling import CachedSampler, force_topk
from kernels.reference import ops as reference


@pytest.mark.covers(edges=("call",), kinds=("logic",))
def test_cached_sampling_keeps_reference_draws_when_requests_and_positions_change() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20261118)
    sampler = CachedSampler()
    for step, rows in enumerate((32, 32, 32, 1, 128, 128, 32)):
        logits = torch.randn(rows, 1024, device="cuda", generator=generator)
        temperatures, top_ps, top_ks = [0.3] * rows, [0.95] * rows, [-1] * rows
        seeds = [7 + (i << 40) for i in range(rows)]
        if step in (2, 5):
            top_ps[0], top_ks[0], seeds[0] = 0.5, 7, 123456
        positions = torch.arange(rows, device="cuda", dtype=torch.int64) + 32768 + step
        expected = reference.sample(logits.clone(), temperatures, top_ps, top_ks, seeds, positions)
        actual = sampler.sample(logits.clone(), temperatures, top_ps, top_ks, seeds, positions)
        assert torch.equal(actual, expected)


@pytest.mark.covers(edges=("call",), kinds=("logic",))
def test_deepselect_vocabulary_cap_returns_the_same_values_and_matching_token_ids() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20261118)
    logits = torch.randn(32, 129280, device="cuda", generator=generator)
    values, indices = force_topk(logits, 1024)
    expected = logits.topk(1024, dim=-1).values
    assert torch.equal(values.sort(dim=-1, descending=True).values, expected)
    assert torch.equal(logits.gather(-1, indices.long()), values)


@pytest.mark.covers(edges=("call", "state"), kinds=("logic",))
def test_bf16_linear_keeps_fp32_output_and_rechecks_changed_weights() -> None:
    ops.align_experts()
    generator = torch.Generator(device="cuda").manual_seed(11)
    x = torch.randn(16, 256, device="cuda", generator=generator).bfloat16()
    weight = (torch.randn(128, 256, device="cuda", generator=generator) * 0.1).bfloat16().float()
    mega = Mega(32)
    try:
        actual = mega.linear(x, weight)
        expected = torch.nn.functional.linear(x.float(), weight)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        weight.add_(0.0007)
        assert torch.equal(mega.linear(x, weight), torch.nn.functional.linear(x.float(), weight))
    finally:
        mega.close()
