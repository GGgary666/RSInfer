"""The references of the speculative step's kernels (``kernels.reference.spec``, doc/specs/mtp.md):
a verify step over D tokens is D decode steps, keeping what each leaves; the commit moves the
accepted step's state; the acceptance takes the drafts the target would, and with temperature
draws what the target's distribution draws."""

from __future__ import annotations

import torch

from kernels.reference import ops, spec

torch.manual_seed(0)
D = 4


def test_a_verify_conv_keeps_the_window_after_each_token_as_decode_steps_leave_it() -> None:
    bs, dim, width = 3, 6, 4
    weight, bias = torch.randn(dim, width), torch.randn(dim)
    x = torch.randn(bs, dim, D)
    pool = torch.randn(5, dim, width - 1)
    slots = torch.tensor([3, 1, 4])
    windows = torch.zeros(bs, D, dim, width - 1)
    stepped = pool.clone()
    want = torch.stack([ops.causal_conv1d_update(x[:, :, j], stepped, weight, bias, "silu", slots) for j in range(D)],
                       dim=-1)
    kept: list[torch.Tensor] = []
    replay = pool.clone()
    for j in range(D):
        ops.causal_conv1d_update(x[:, :, j], replay, weight, bias, "silu", slots)
        kept.append(replay[slots].clone())
    got = spec.verify_conv1d_update(x, pool, weight, bias, "silu", slots, windows, torch.arange(bs))
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(windows, torch.stack(kept, dim=1))
    torch.testing.assert_close(pool, stepped)  # the row's state after all D, as SGLang's kernel leaves it


def test_a_verify_recurrent_keeps_the_state_after_each_token_and_leaves_the_row_s_own() -> None:
    bs, hk, hv, kd, vd = 3, 2, 4, 8, 16
    n = bs * D
    q, k, v = torch.randn(1, n, hk, kd), torch.randn(1, n, hk, kd), torch.randn(1, n, hv, vd)
    A_log, dt_bias, a, b = torch.randn(hv), torch.randn(hv), torch.randn(n, hv), torch.randn(n, hv)
    pool = torch.randn(5, hv, vd, kd)
    slots = torch.tensor([2, 4, 1])
    before = pool.clone()
    kept = torch.zeros(bs, D, hv, vd, kd)
    got = spec.verify_recurrent(A_log, dt_bias, q, k, v, a, b, pool, slots, kept, torch.arange(bs), D)
    assert torch.equal(pool, before)
    stepped = pool.clone()
    rows = torch.arange(bs) * D
    for j in range(D):
        at = rows + j
        out = ops.fused_sigmoid_gating_delta_rule_update(A_log, dt_bias, q[:, at], k[:, at], v[:, at], a[at], b[at],
                                                         stepped, slots, torch.arange(bs + 1))
        torch.testing.assert_close(got[:, at], out, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(kept[:, j], stepped[slots], rtol=1e-5, atol=1e-5)


def test_the_commit_moves_the_accepted_step_s_state_and_the_boundary_s_alone_as_batched() -> None:
    layers, slots, lanes = 2, 7, 3
    conv, temporal = torch.randn(layers, slots, 5, 3), torch.randn(layers, slots, 4, 6, 2)
    iconv, itemp = torch.randn(layers, lanes, D, 5, 3), torch.randn(layers, lanes, D, 4, 6, 2)
    cache, steps = torch.tensor([3, 5, 1]), torch.tensor([0, 3, 2])
    dst, tracked = torch.tensor([6, 0, 2]), torch.tensor([1, -1, 0])
    pools = conv.clone(), temporal.clone()
    spec.commit_states(*pools, iconv, itemp, cache, torch.arange(lanes), steps, dst, tracked)
    for pool, inter in zip(pools, (iconv, itemp), strict=True):
        for i in range(lanes):
            assert torch.equal(pool[:, cache[i]], inter[:, i, steps[i]])
        assert torch.equal(pool[:, 6], inter[:, 0, 1]) and torch.equal(pool[:, 2], inter[:, 2, 0])
    assert torch.equal(pools[0][:, [0, 4]], conv[:, [0, 4]])  # untouched: the padding slot and the rest
    one = conv.clone(), temporal.clone()
    spec.commit_states(*one, iconv[:, 1:2], itemp[:, 1:2], cache[1:2], torch.zeros(1, dtype=torch.long),
                       steps[1:2], dst[1:2], tracked[1:2])
    assert torch.equal(one[0][:, 5], pools[0][:, 5]) and torch.equal(one[1][:, 5], pools[1][:, 5])


def test_greedy_acceptance_takes_the_drafts_while_they_are_the_target_s_argmax() -> None:
    vocab = 10
    predict = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 1, 2, 3]])
    logits = torch.nn.functional.one_hot(predict, vocab).float()
    drafts = torch.tensor([[1, 2, 3], [5, 0, 7], [0, 1, 2]])
    tokens, accept = spec.accept(logits, drafts, [0.0] * 3, [1.0] * 3, [-1] * 3, [1, 2, 3], torch.zeros(3, dtype=torch.long))
    assert accept.tolist() == [4, 2, 1]
    assert [tokens[i, :n].tolist() for i, n in enumerate(accept.tolist())] == [[1, 2, 3, 4], [5, 6], [9]]


def test_sampled_acceptance_draws_the_target_s_tokens_whatever_the_draft() -> None:
    """Speculative sampling keeps the target's distribution: over many seeds, the first token is
    distributed as the target's first distribution, and drafted tokens are taken as often as it gives
    them; the same seed and start give the same tokens alone and batched."""
    vocab, n = 5, 40000
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0, 0.5], [0.0, 1.0, 2.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0],
                           [0.0, 0.0, 0.0, 3.0, 0.0]]).expand(n, D, vocab)
    drafts = torch.tensor([[1, 2, 3]]).expand(n, D - 1)
    seeds = list(range(n))
    tokens, accept = spec.accept(logits, drafts, [1.0] * n, [1.0] * n, [-1] * n, seeds, torch.full((n,), 7))
    p0 = torch.softmax(logits[0, 0], dim=-1)
    first = torch.bincount(tokens[:, 0], minlength=vocab).float() / n
    torch.testing.assert_close(first, p0, atol=0.01, rtol=0)
    assert abs((accept > 1).float().mean().item() - p0[1].item()) < 0.01
    alone, alone_accept = spec.accept(logits[17:18], drafts[17:18], [1.0], [1.0], [-1], [17], torch.full((1,), 7))
    assert torch.equal(alone_accept, accept[17:18]) and torch.equal(alone[0, :int(accept[17])],
                                                                   tokens[17, :int(accept[17])])
