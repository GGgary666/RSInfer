"""kernels.reference against the HF ``qwen3_5`` model it follows, and the properties the engine
relies on: a chunked prefill continues where the last chunk stopped, and the references that
claim batch invariance give a row the same bits alone and among other rows."""

from __future__ import annotations

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from kernels.reference import ops
from kernels.reference.unit import DEFINITIONS

torch.manual_seed(0)

ROW_COUNTS = (1, 2, 3, 5, 8, 13)
BATCHED_INVARIANT = {"embed.lookup@1", "kv.store@1", "attn.paged@1", "sample.seeded@1"}
"""Batch invariant and checked below, row alone against the same row among others."""
ONE_ROW = {"gdn.conv1d@1", "gdn.delta_rule@1"}
"""Take one row; the layer calls them row by row."""


def test_only_the_row_by_row_references_claim_batch_invariance() -> None:
    """A new claim needs a check in the test below. Linear, norms, RoPE, activations and gating
    run one op over the whole batch and do not claim it (fp32 linear differs in the last bits)."""
    assert {name for name, *_, invariant in DEFINITIONS if invariant} == BATCHED_INVARIANT | ONE_ROW


def test_the_batch_invariant_references_give_a_row_the_same_bits_alone_and_batched() -> None:
    gen = torch.Generator().manual_seed(3)
    for dtype in (torch.float32, torch.bfloat16):
        table = torch.randn(100, 64, generator=gen).to(dtype)
        for n in ROW_COUNTS:
            ids = torch.randint(0, 100, (n,), generator=gen)
            assert all(torch.equal(ops.embedding(ids, table)[r:r + 1], ops.embedding(ids[r:r + 1], table))
                       for r in range(n))

            heads, kv_heads, dim = 4, 2, 32
            lengths = torch.randint(1, 9, (n,), generator=gen).tolist()
            new = [min(length, q) for length, q in zip(lengths, torch.randint(1, 4, (n,), generator=gen).tolist())]
            free = torch.randperm(sum(lengths), generator=gen) + 1  # scattered slots; slot 0 is padding
            rows = list(torch.split(free, lengths))
            caches = torch.zeros(sum(lengths) + 1, kv_heads, dim, dtype=dtype), \
                torch.zeros(sum(lengths) + 1, kv_heads, dim, dtype=dtype)
            k, v = torch.randn(sum(lengths), kv_heads, dim, generator=gen), torch.randn(sum(lengths), kv_heads, dim,
                                                                                         generator=gen)
            ops.kv_store(*caches, free, k, v)
            for r in range(n):  # a row stored alone lands on the same bits
                alone = torch.zeros_like(caches[0]), torch.zeros_like(caches[1])
                start = sum(lengths[:r])
                ops.kv_store(*alone, rows[r], k[start:start + lengths[r]], v[start:start + lengths[r]])
                assert torch.equal(alone[0][rows[r]], caches[0][rows[r]])
                assert torch.equal(alone[1][rows[r]], caches[1][rows[r]])
            q = torch.randn(sum(new), heads, dim, generator=gen).to(dtype)
            out = ops.paged_attention(q, *caches, rows, new, dim ** -0.5)
            for r in range(n):
                start = sum(new[:r])
                alone_out = ops.paged_attention(q[start:start + new[r]], *caches, [rows[r]], [new[r]], dim ** -0.5)
                assert torch.equal(out[start:start + new[r]], alone_out), (dtype, n, r)

            logits = torch.randn(n, 50, generator=gen).to(dtype)
            temp = torch.rand(n, generator=gen).tolist()
            top_p, top_k = [0.9] * n, [20] * n
            seeds, pos = list(range(n)), torch.randint(0, 100, (n,), generator=gen).tolist()
            drawn = ops.sample(logits, temp, top_p, top_k, seeds, pos)
            assert all(torch.equal(drawn[r:r + 1], ops.sample(logits[r:r + 1], [temp[r]], [top_p[r]], [top_k[r]],
                                                              [seeds[r]], [pos[r]])) for r in range(n))


def test_norms_match_hf() -> None:
    for dtype in (torch.float32, torch.bfloat16):
        x, z = torch.randn(5, 64, dtype=dtype), torch.randn(5, 64, dtype=dtype)
        norm = hf.Qwen3_5RMSNorm(64, eps=1e-6)
        torch.nn.init.normal_(norm.weight)
        assert torch.equal(ops.gemma_rms_norm(x, norm.weight.detach(), 1e-6), norm(x).detach())
        gated = hf.Qwen3_5RMSNormGated(64, eps=1e-6).to(dtype)
        torch.nn.init.normal_(gated.weight)
        assert torch.equal(ops.rms_norm_gated(x, z, gated.weight.detach(), 1e-6), gated(x, z).detach())


def test_partial_rope_matches_hf_on_text_positions() -> None:
    cfg = Qwen3_5TextConfig(hidden_size=256, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                            rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                                             "mrope_section": [3, 3, 2], "mrope_interleaved": True})
    rotary = hf.Qwen3_5TextRotaryEmbedding(cfg)
    q, k = torch.randn(1, 4, 9, 64), torch.randn(1, 2, 9, 64)
    positions = torch.arange(3, 12)
    cos, sin = rotary(q, positions[None])
    want_q, want_k = hf.apply_rotary_pos_emb(q, k, cos, sin)
    got_q, got_k = ops.rope_neox(positions, q[0].transpose(0, 1), k[0].transpose(0, 1), 16, 1e7)
    torch.testing.assert_close(got_q, want_q[0].transpose(0, 1))
    torch.testing.assert_close(got_k, want_k[0].transpose(0, 1))


def test_conv_continues_across_chunks_and_matches_hf() -> None:
    channels, kernel, length = 24, 4, 13
    x, w, b = torch.randn(length, channels), torch.randn(channels, kernel), torch.randn(channels)
    want = hf.causal_conv1d_fn(x.t()[None], w, b, activation="silu")[0].t()
    state = torch.zeros(channels, kernel - 1)
    parts = [ops.causal_conv1d(x[i:j], state, w, b) for i, j in ((0, 5), (5, 6), (6, 13))]
    torch.testing.assert_close(torch.cat(parts), want)
    hf_state = torch.zeros(1, channels, kernel - 1)
    hf.causal_conv1d_fn(x.t()[None], w, b, activation="silu")
    for t in range(length):
        hf.causal_conv1d_update(x[t:t + 1].t()[None], hf_state, w, b, "silu")
    torch.testing.assert_close(state, hf_state[0])


def test_delta_rule_continues_across_chunks_and_matches_hf() -> None:
    length, hk, hv, kd, vd = 11, 2, 4, 8, 16
    q, k, v = torch.randn(length, hk, kd), torch.randn(length, hk, kd), torch.randn(length, hv, vd)
    A_log, dt_bias = torch.randn(hv), torch.randn(hv)
    g, beta = ops.gdn_gating(A_log, torch.randn(length, hv), dt_bias, torch.randn(length, hv))
    rep = hv // hk
    want, want_state = hf.torch_recurrent_gated_delta_rule(
        q.repeat_interleave(rep, 1)[None], k.repeat_interleave(rep, 1)[None], v[None], g[None], beta[None],
        output_final_state=True, use_qk_l2norm_in_kernel=True)
    chunked, _ = hf.torch_chunk_gated_delta_rule(
        q.repeat_interleave(rep, 1)[None], k.repeat_interleave(rep, 1)[None], v[None], g[None], beta[None],
        output_final_state=True, use_qk_l2norm_in_kernel=True)
    state = torch.zeros(hv, vd, kd)  # sglang's layout: [v_heads, v_dim, k_dim]
    out = torch.cat([ops.gated_delta_rule(q[i:j], k[i:j], v[i:j], g[i:j], beta[i:j], state)
                     for i, j in ((0, 4), (4, 5), (5, 11))])
    torch.testing.assert_close(out, want[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out, chunked[0], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state, want_state[0].transpose(-1, -2), rtol=1e-5, atol=1e-5)


def test_paged_attention_is_causal_sdpa_per_row_wherever_the_tokens_sit() -> None:
    heads, kv_heads, dim, slots = 4, 2, 32, 64
    k_cache, v_cache = torch.zeros(slots, kv_heads, dim), torch.zeros(slots, kv_heads, dim)
    lengths, new = [7, 3], [7, 1]  # a prefill row and a decode row
    free = torch.randperm(slots - 1) + 1  # scattered, disjoint slots; slot 0 is padding
    rows = [free[:lengths[0]], free[lengths[0]:sum(lengths)]]
    keys = [torch.randn(n, kv_heads, dim) for n in lengths]
    vals = [torch.randn(n, kv_heads, dim) for n in lengths]
    for r in range(2):
        ops.kv_store(k_cache, v_cache, rows[r], keys[r], vals[r])
    q = torch.randn(sum(new), heads, dim)
    out = ops.paged_attention(q, k_cache, v_cache, rows, new, dim ** -0.5)
    start = 0
    for r in range(2):
        n, seq = new[r], lengths[r]
        qs = q[start:start + n].transpose(0, 1)
        ks = keys[r].repeat_interleave(2, 1).transpose(0, 1)
        vs = vals[r].repeat_interleave(2, 1).transpose(0, 1)
        mask = torch.arange(seq)[None, :] <= torch.arange(seq - n, seq)[:, None]
        want = torch.nn.functional.scaled_dot_product_attention(qs, ks, vs, attn_mask=mask)
        torch.testing.assert_close(out[start:start + n], want.transpose(0, 1), rtol=1e-5, atol=1e-5)
        start += n


def test_sampling_is_greedy_at_zero_and_drawn_from_seed_and_position() -> None:
    logits = torch.randn(3, 50)
    greedy = ops.sample(logits, [0.0] * 3, [1.0] * 3, [-1] * 3, [1, 2, 3], [5, 5, 5])
    assert torch.equal(greedy, logits.argmax(-1))
    temp, top_p, top_k, seeds, pos = [1.0] * 3, [0.9] * 3, [20] * 3, [7, 7, 8], [3, 4, 3]
    a, b = ops.sample(logits, temp, top_p, top_k, seeds, pos), ops.sample(logits, temp, top_p, top_k, seeds, pos)
    assert torch.equal(a, b)
    alone = torch.cat([ops.sample(logits[r:r + 1], [temp[r]], [top_p[r]], [top_k[r]], [seeds[r]], [pos[r]])
                       for r in range(3)])
    assert torch.equal(a, alone)  # the same draw however the rows are batched
    top1 = ops.sample(logits, [1.0] * 3, [1.0] * 3, [1] * 3, [1, 2, 3], [0, 0, 0])
    assert torch.equal(top1, logits.argmax(-1))
