"""kernels.reference (ported from SGLang) against the HF ``qwen3_5`` model, and the properties the
engine relies on: a chunked prefill continues where the last chunk stopped, decode steps continue
a prefill, and the references that claim batch invariance give a row the same bits alone and
among other rows."""

from __future__ import annotations

from typing import cast

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from kernels.reference import ops
from kernels.reference.unit import DEFINITIONS
from layers.rope.unit import rope_cos_sin_cache

torch.manual_seed(0)

INVARIANT = {"embed.lookup@1", "kv.store@1", "attn.extend@1", "attn.decode@1", "gdn.conv1d_fn@1",
             "gdn.chunk_delta_rule@1"}
"""Loop over sequences or only move data; checked bit for bit below."""


def test_only_the_per_sequence_references_claim_batch_invariance() -> None:
    """A new claim needs a check here. One op over the whole batch (linear, norms, RoPE, the
    batched decode steps, sampling) does not claim it: fp32 linear differs in the last bits."""
    assert {name for name, *_, invariant in DEFINITIONS if invariant} == INVARIANT


def test_norms_match_hf() -> None:
    for dtype in (torch.float32, torch.bfloat16):
        x, z, r = (torch.randn(5, 64, dtype=dtype) for _ in range(3))
        norm = hf.Qwen3_5RMSNorm(64, eps=1e-6)
        torch.nn.init.normal_(norm.weight)
        w = norm.weight.detach()
        assert torch.equal(cast(torch.Tensor, ops.gemma_rms_norm(x, w, 1e-6)), norm(x).detach())
        normed, residual = ops.gemma_rms_norm(x, w, 1e-6, r)
        assert torch.equal(residual, x + r) and torch.equal(normed, norm(x + r).detach())
    x, z = torch.randn(5, 64), torch.randn(5, 64)
    gated = hf.Qwen3_5RMSNormGated(64, eps=1e-6)
    torch.nn.init.normal_(gated.weight)
    torch.testing.assert_close(ops.rms_norm_gated(x, gated.weight.detach(), None, z, 1e-6), gated(x, z).detach())


def test_partial_rope_matches_hf_on_text_positions() -> None:
    cfg = Qwen3_5TextConfig(hidden_size=256, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                            rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                                             "mrope_section": [3, 3, 2], "mrope_interleaved": True})
    rotary = hf.Qwen3_5TextRotaryEmbedding(cfg)
    q, k = torch.randn(1, 4, 9, 64), torch.randn(1, 2, 9, 64)
    positions = torch.arange(3, 12)
    want_q, want_k = hf.apply_rotary_pos_emb(q, k, *rotary(q, positions[None]))
    cache = rope_cos_sin_cache(16, 64, 1e7)
    got_q, got_k = ops.rope(positions, q[0].transpose(0, 1).reshape(9, -1), k[0].transpose(0, 1).reshape(9, -1),
                            64, 16, cache)
    torch.testing.assert_close(got_q.view(9, 4, 64), want_q[0].transpose(0, 1))
    torch.testing.assert_close(got_k.view(9, 2, 64), want_k[0].transpose(0, 1))


def test_conv_continues_across_chunks_and_decode_steps_and_matches_hf() -> None:
    channels, kernel, length = 24, 4, 13
    x, w, b = torch.randn(length, channels), torch.randn(channels, kernel), torch.randn(channels)
    want = hf.causal_conv1d_fn(x.t()[None], w, b, activation="silu")[0].t()
    pool = torch.full((4, channels, kernel - 1), float("nan"))  # slot 2; a fresh request has no state
    slot = torch.tensor([2])
    parts = []
    for start, end in ((0, 5), (5, 6), (6, 10)):
        parts.append(ops.causal_conv1d_fn(x[start:end].t(), w, b, "silu", pool, torch.tensor([start > 0]), slot,
                                          torch.tensor([0, end - start])).t())
    parts += [ops.causal_conv1d_update(x[t:t + 1], pool, w, b, "silu", slot) for t in range(10, length)]
    torch.testing.assert_close(torch.cat(parts), want)
    torch.testing.assert_close(pool[2], torch.cat([torch.zeros(channels, kernel - 1), x.t()], -1)[:, -(kernel - 1):])
    assert pool[[0, 1, 3]].isnan().all()  # other slots untouched


def test_delta_rule_chunks_and_decode_steps_continue_and_match_hf() -> None:
    length, hk, hv, kd, vd = 11, 2, 4, 8, 16
    q, k, v = torch.randn(1, length, hk, kd), torch.randn(1, length, hk, kd), torch.randn(1, length, hv, vd)
    A_log, dt_bias, a, b = torch.randn(hv), torch.randn(hv), torch.randn(length, hv), torch.randn(length, hv)
    g, beta = ops.fused_gdn_gating(A_log, a, b, dt_bias)
    rep = hv // hk
    want, want_state = hf.torch_recurrent_gated_delta_rule(
        q.repeat_interleave(rep, 2), k.repeat_interleave(rep, 2), v, g, beta, output_final_state=True,
        use_qk_l2norm_in_kernel=True)
    pool = torch.zeros(3, hv, vd, kd)  # SGLang's layout: [slots, HV, V, K]
    slot = torch.tensor([1])
    parts = []
    for start, end in ((0, 4), (4, 8)):
        out, _, _ = ops.chunk_gated_delta_rule(q[:, start:end], k[:, start:end], v[:, start:end], g[:, start:end],
                                               beta[:, start:end], pool, slot, torch.tensor([0, end - start]))
        parts.append(out)
    for t in range(8, length):
        parts.append(ops.fused_sigmoid_gating_delta_rule_update(
            A_log, dt_bias, q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1], a[t:t + 1], b[t:t + 1], pool, slot,
            torch.tensor([0, 1])))
    torch.testing.assert_close(torch.cat(parts, dim=1), want, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(pool[1], want_state[0].transpose(-1, -2), rtol=1e-4, atol=1e-4)
    assert not pool[[0, 2]].any()


def _paged(n: int, dtype: torch.dtype, gen: torch.Generator) -> dict[str, torch.Tensor]:
    """``n`` sequences in a KV pool at scattered token slots (slot 0 is padding)."""
    heads, kv_heads, dim = 4, 2, 32
    lens = torch.randint(1, 9, (n,), generator=gen)
    new = torch.minimum(lens, torch.randint(1, 4, (n,), generator=gen))
    slots = torch.randperm(int(lens.sum()), generator=gen) + 1
    req_to_token = torch.zeros(n + 1, 16, dtype=torch.int32)
    for r, row in enumerate(torch.split(slots, lens.tolist())):
        req_to_token[r + 1, :len(row)] = row
    k_cache = torch.randn(int(lens.sum()) + 1, kv_heads, dim, generator=gen).to(dtype)
    v_cache = torch.randn(int(lens.sum()) + 1, kv_heads, dim, generator=gen).to(dtype)
    return {"q": torch.randn(int(new.sum()), heads, dim, generator=gen).to(dtype), "k_cache": k_cache,
            "v_cache": v_cache, "req_to_token": req_to_token, "rows": torch.arange(1, n + 1), "lens": lens, "new": new}


def test_attention_is_causal_sdpa_per_sequence_wherever_the_tokens_sit() -> None:
    p = _paged(3, torch.float32, torch.Generator().manual_seed(1))
    out = ops.attention_extend(p["q"], torch.empty_like(p["q"]), p["k_cache"], p["v_cache"], p["req_to_token"],
                               p["rows"], p["lens"], p["lens"] - p["new"], p["new"], 32 ** -0.5, True, True)
    start = 0
    for r in range(3):
        n, seq = int(p["new"][r]), int(p["lens"][r])
        tokens = p["req_to_token"][r + 1, :seq]
        keys = p["k_cache"][tokens].repeat_interleave(2, 1).transpose(0, 1)
        vals = p["v_cache"][tokens].repeat_interleave(2, 1).transpose(0, 1)
        mask = torch.arange(seq)[None, :] <= torch.arange(seq - n, seq)[:, None]
        want = torch.nn.functional.scaled_dot_product_attention(p["q"][start:start + n].transpose(0, 1), keys, vals,
                                                                attn_mask=mask)
        torch.testing.assert_close(out[start:start + n], want.transpose(0, 1), rtol=1e-5, atol=1e-5)
        start += n
    last = torch.cumsum(p["new"], 0) - 1  # a decode step is the extend of one token
    decoded = ops.attention_decode(p["q"][last], torch.empty_like(p["q"][last]), p["k_cache"], p["v_cache"],
                                   p["req_to_token"], p["rows"], p["lens"], 32 ** -0.5, True)
    torch.testing.assert_close(decoded, out[last], rtol=1e-5, atol=1e-5)


def test_the_batch_invariant_references_give_a_row_the_same_bits_alone_and_batched() -> None:
    gen = torch.Generator().manual_seed(3)
    for dtype in (torch.float32, torch.bfloat16):
        table = torch.randn(100, 64, generator=gen).to(dtype)
        ids = torch.randint(0, 100, (13,), generator=gen)
        assert all(torch.equal(ops.embedding(ids, table)[r:r + 1], ops.embedding(ids[r:r + 1], table))
                   for r in range(13))
        for n in (1, 2, 5, 8):
            p = _paged(n, dtype, gen)
            args = (p["k_cache"], p["v_cache"], p["req_to_token"])
            out = ops.attention_extend(p["q"], torch.empty_like(p["q"]), *args, p["rows"], p["lens"],
                                       p["lens"] - p["new"], p["new"], 32 ** -0.5, True, True)
            starts = torch.cumsum(p["new"], 0) - p["new"]
            for r in range(n):
                s, e = int(starts[r]), int(starts[r] + p["new"][r])
                alone = ops.attention_extend(p["q"][s:e], torch.empty_like(p["q"][s:e]), *args, p["rows"][r:r + 1],
                                             p["lens"][r:r + 1], (p["lens"] - p["new"])[r:r + 1], p["new"][r:r + 1],
                                             32 ** -0.5, True, True)
                assert torch.equal(out[s:e], alone), (dtype, n, r)

            dim, kernel = 24, 4
            x, w = torch.randn(dim, int(p["new"].sum()), generator=gen).to(dtype), torch.randn(dim, kernel).to(dtype)
            init = torch.randn(n + 1, dim, kernel - 1, generator=gen).to(dtype)
            cu = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(p["new"], 0)])
            has_init = torch.arange(n) % 2 == 0
            pool = init.clone()
            conv = ops.causal_conv1d_fn(x, w, None, "silu", pool, has_init, p["rows"], cu)
            hk, hv, kd, vd = 1, 2, 8, 8
            T = int(p["new"].sum())
            q, k = torch.randn(1, T, hk, kd, generator=gen).to(dtype), torch.randn(1, T, hk, kd, generator=gen).to(dtype)
            v = torch.randn(1, T, hv, vd, generator=gen).to(dtype)
            g = -torch.rand(1, T, hv, generator=gen)
            beta = torch.rand(1, T, hv, generator=gen).to(dtype)
            states = torch.randn(n + 1, hv, vd, kd, generator=gen)
            rule, _, _ = ops.chunk_gated_delta_rule(q, k, v, g, beta, states.clone(), p["rows"], cu)
            for r in range(n):
                s, e = int(cu[r]), int(cu[r + 1])
                one = init.clone()
                assert torch.equal(conv[:, s:e], ops.causal_conv1d_fn(
                    x[:, s:e], w, None, "silu", one, has_init[r:r + 1], p["rows"][r:r + 1], torch.tensor([0, e - s])))
                assert torch.equal(pool[r + 1], one[r + 1])
                alone, _, _ = ops.chunk_gated_delta_rule(q[:, s:e], k[:, s:e], v[:, s:e], g[:, s:e], beta[:, s:e],
                                                         states.clone(), p["rows"][r:r + 1], torch.tensor([0, e - s]))
                assert torch.equal(rule[:, s:e], alone), (dtype, n, r)

            cache = torch.zeros(p["k_cache"].shape[0], 2, 32, dtype=dtype), torch.zeros(p["k_cache"].shape[0], 2, 32,
                                                                                      dtype=dtype)
            loc = torch.randperm(cache[0].shape[0] - 1, generator=gen)[:T] + 1
            kv = torch.randn(T, 2, 32, generator=gen), torch.randn(T, 2, 32, generator=gen)
            ops.kv_store(*cache, loc, *kv)
            for r in range(n):
                s, e = int(cu[r]), int(cu[r + 1])
                one = torch.zeros_like(cache[0]), torch.zeros_like(cache[1])
                ops.kv_store(*one, loc[s:e], kv[0][s:e], kv[1][s:e])
                assert torch.equal(one[0][loc[s:e]], cache[0][loc[s:e]])


def test_sampling_is_greedy_at_zero_and_drawn_from_seed_and_position() -> None:
    logits = torch.randn(3, 50)
    assert torch.equal(ops.sample(logits, [0.0] * 3, [1.0] * 3, [-1] * 3, [1, 2, 3], [5, 5, 5]),
                       logits.argmax(-1).to(torch.int32))
    for top_p, top_k in ((1.0, -1), (0.9, 20)):
        temps, top_ps, top_ks, seeds, pos = [1.0] * 3, [top_p] * 3, [top_k] * 3, [7, 7, 8], [3, 4, 3]
        a = ops.sample(logits, temps, top_ps, top_ks, seeds, pos)
        assert torch.equal(a, ops.sample(logits, temps, top_ps, top_ks, seeds, pos))
        alone = torch.cat([ops.sample(logits[r:r + 1], temps[r:r + 1], top_ps[r:r + 1], top_ks[r:r + 1], seeds[r:r + 1],
                                      pos[r:r + 1]) for r in range(3)])
        assert torch.equal(a, alone)  # the draw depends on (seed, position), not on the batch
    draws = {int(ops.sample(logits[:1], [1.0], [1.0], [-1], [11], [p])[0]) for p in range(40)}
    assert len(draws) > 5  # another position, another draw
    mixed = ops.sample(logits, [0.0, 1.0, 1.0], [1.0] * 3, [-1, 1, -1], [1, 2, 3], [0, 0, 0])
    assert mixed[:2].tolist() == logits[:2].argmax(-1).tolist()  # temperature 0 and top-k 1 are greedy
    h = ops.murmur_hash32(torch.tensor([1, 1 << 40]), torch.tensor([0, 7]), torch.arange(5))
    assert h.shape == (2, 5) and bool(((h >= 0) & (h < 1 << 32)).all()) and len(set(h.flatten().tolist())) == 10
