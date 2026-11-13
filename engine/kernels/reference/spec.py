# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""PyTorch references of the speculative step's kernels (doc/specs/mtp.md), ported from SGLang
(e838360; LICENSE.sglang):

- the GDN verify step: ``causal_conv1d_update`` with ``intermediate_conv_window`` and
  ``fused_sigmoid_gating_delta_rule_update`` with ``disable_state_update`` and
  ``intermediate_states_buffer`` (srt/layers/attention/linear/gdn_backend.py, ``forward_extend`` in
  ``TARGET_VERIFY``);
- the commit of the accepted step's state (``scatter_mamba_states_after_mtp_verify``,
  kernels/ops/mamba/mamba_state_scatter_triton.py, with ``_verify_commit_step_indices``);
- the acceptance of a chain of drafts: ``verify_tree_greedy`` and the target-only speculative
  sampling (kernels/jit/include/sgl_kernel/speculative/sampling.cuh,
  ``TreeSpeculativeSamplingTargetOnly``) with its seeded coins (srt/speculative/eagle_utils.py,
  ``_seeded_verify_coins``).

A lane is a row's place in the step; the intermediate buffers are indexed by lane."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from kernels.reference.ops import SAMPLING_EPS, TOP_K_ALL, _l2norm, _on_device, murmur_hash32, softplus


def verify_conv1d_update(x: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
                         activation: str | None, conv_state_indices: torch.Tensor,
                         intermediate_conv_window: torch.Tensor, intermediate_state_indices: torch.Tensor
                         ) -> torch.Tensor:
    """``gdn.verify_conv1d_update@1``: ``x`` ``[bs, dim, D]`` is each row's D tokens of ``[q | k | v]``;
    the conv steps through them from ``conv_state[conv_state_indices[i]]`` (``[dim, width - 1]``).
    The window after token ``j`` goes to ``intermediate_conv_window[intermediate_state_indices[i], j]``
    (``[lanes, D, dim, width - 1]``, this layer's), and the row's state becomes the window after all D
    tokens, as SGLang's kernel leaves it (the commit puts the accepted one there). Returns
    ``[bs, dim, D]``. A negative index is a padded lane: the reference runs it on the padding slot."""
    if activation not in (None, "silu", "swish"):
        raise NotImplementedError("activation must be None, silu, or swish")
    idx = conv_state_indices.to(torch.long).clamp_min(0)
    lanes = intermediate_state_indices.to(torch.long)
    width = weight.shape[1]
    steps = x.shape[-1]
    states = conv_state[idx]
    x_new = torch.cat([states.to(weight.dtype), x.to(weight.dtype)], dim=-1)  # [bs, dim, width - 1 + D]
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=x.shape[1])[:, :, -steps:]
    for j in range(steps):
        intermediate_conv_window[lanes, j] = x_new[:, :, j + 1:j + width].to(intermediate_conv_window.dtype)
    conv_state[idx] = x_new[:, :, -(width - 1):].to(conv_state.dtype)
    return (out if activation is None else F.silu(out)).to(dtype=x.dtype)


def verify_recurrent(A_log: torch.Tensor, dt_bias: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     a: torch.Tensor, b: torch.Tensor, ssm_states: torch.Tensor, cache_indices: torch.Tensor,
                     intermediate_states: torch.Tensor, intermediate_state_indices: torch.Tensor, cache_steps: int,
                     use_qk_l2norm_in_kernel: bool = True) -> torch.Tensor:
    """``gdn.verify_recurrent@1``: the gated delta rule over each row's ``cache_steps`` (D) tokens from
    ``ssm_states[cache_indices[i]]`` (``[HV, V, K]``), which it leaves as it is
    (``disable_state_update``); the state after token ``j`` goes to
    ``intermediate_states[intermediate_state_indices[i], j]`` (``[lanes, D, HV, V, K]``, this layer's).
    ``q``, ``k`` ``[1, bs * D, HK, K]``, ``v`` ``[1, bs * D, HV, V]``, ``a``, ``b`` ``[bs * D, HV]``,
    in row order. Returns ``[1, bs * D, HV, V]``."""
    idx = cache_indices.to(torch.long).clamp_min(0)
    lanes = intermediate_state_indices.to(torch.long)
    bs, steps = idx.shape[0], cache_steps
    num_heads, num_value_heads = q.shape[2], v.shape[2]
    if num_value_heads // num_heads > 1:
        q = q.repeat_interleave(num_value_heads // num_heads, dim=2)
        k = k.repeat_interleave(num_value_heads // num_heads, dim=2)
    if use_qk_l2norm_in_kernel:
        q = _l2norm(q, dim=-1, eps=1e-6)
        k = _l2norm(k, dim=-1, eps=1e-6)
    beta = b.float().sigmoid().view(bs, steps, num_value_heads)
    g = (-A_log.float().exp() * softplus(a.float() + dt_bias)).view(bs, steps, num_value_heads)
    query = q.float().view(bs, steps, num_value_heads, -1) * (q.shape[-1] ** -0.5)
    key = k.float().view(bs, steps, num_value_heads, -1)
    value = v.float().view(bs, steps, num_value_heads, -1)
    state = ssm_states[idx].float().transpose(-1, -2)  # [bs, HV, K, V]
    out = value.new_zeros(value.shape)
    for j in range(steps):
        state = state * g[:, j].exp()[..., None, None]
        kv_mem = (state * key[:, j][..., None]).sum(dim=-2)
        delta = (value[:, j] - kv_mem) * beta[:, j][..., None]
        state = state + key[:, j][..., None] * delta[..., None, :]
        out[:, j] = (state * query[:, j][..., None]).sum(dim=-2)
        intermediate_states[lanes, j] = state.transpose(-1, -2).to(intermediate_states.dtype)
    return out.view(1, bs * steps, num_value_heads, -1).to(v.dtype)


def commit_states(conv: torch.Tensor, temporal: torch.Tensor, intermediate_conv: torch.Tensor,
                  intermediate_temporal: torch.Tensor, cache_indices: torch.Tensor, lanes: torch.Tensor,
                  steps: torch.Tensor, track_dst: torch.Tensor, track_steps: torch.Tensor) -> None:
    """``gdn.commit_states@1`` (``scatter_mamba_states_after_mtp_verify``): every layer's state after
    lane ``lanes[i]``'s step ``steps[i]`` (its accepted length - 1) into its slot ``cache_indices[i]``;
    where ``track_steps[i] >= 0``, the state after that step into slot ``track_dst[i]`` too (the
    prefix cache's boundary, ``mamba_steps_to_track``). Pools ``[layers, slots, ...]``, intermediate
    buffers ``[layers, lanes, D, ...]``. Without a host round trip; a lane of no tracking writes its
    ``track_dst`` with what it holds (the padding slot, 0)."""
    lane = lanes.long()
    slot, step = cache_indices.long().clamp_min(0), steps.long()
    track = track_steps.long()
    keep = track >= 0
    dst = track_dst.long()
    for pool, inter in ((conv, intermediate_conv), (temporal, intermediate_temporal)):
        pool[:, slot] = inter[:, lane, step].to(pool.dtype)
        shape = (1, -1, *([1] * (pool.dim() - 2)))
        pool[:, dst] = torch.where(keep.view(shape), inter[:, lane, track.clamp_min(0)].to(pool.dtype), pool[:, dst])


def track_steps(starts: torch.Tensor, accept_lens: torch.Tensor, grid: int) -> torch.Tensor:
    """``_verify_commit_step_indices`` for a chain: per row, the step whose state is at the tracking
    boundary its accepted tokens cross (``starts`` the length before the step), or -1."""
    end = starts + accept_lens
    crosses = starts // grid != end // grid
    return torch.where(crosses, (end // grid * grid - starts - 1).clamp_min(0), torch.full_like(starts, -1))


def _seeded_coins(seeds: torch.Tensor, starts: torch.Tensor, draft_tokens: int) -> torch.Tensor:
    """``_seeded_verify_coins``: ``[bs, D + 1]`` in ``[0, 1)``; columns ``0 .. D - 1`` decide the drafts,
    column ``D`` the final draw."""
    cols = torch.arange(draft_tokens + 1, device=seeds.device, dtype=torch.int64)
    uniforms = murmur_hash32(seeds, starts, cols).to(torch.float64) / 0xFFFFFFFF
    return uniforms.to(torch.float32).clamp_(max=1.0 - 2 ** -24)


def _renorm(probs: torch.Tensor, top_ks: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    """``top_k_renorm_prob`` then ``top_p_renorm_prob`` per row."""
    sorted_probs, order = probs.sort(dim=-1, descending=True)
    ranks = torch.arange(probs.shape[-1], device=probs.device).view(1, -1)
    sorted_probs = torch.where(ranks < top_ks.view(-1, 1), sorted_probs, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    before = sorted_probs.cumsum(dim=-1) - sorted_probs
    sorted_probs = torch.where(before < top_ps.view(-1, 1), sorted_probs, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    return torch.zeros_like(probs).scatter_(-1, order, sorted_probs)


def accept(logits: torch.Tensor, drafts: torch.Tensor, temperatures: Sequence[float], top_ps: Sequence[float],
           top_ks: Sequence[int], seeds: Sequence[int], starts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``spec.accept@1``: ``logits`` ``[bs, D, vocab]`` the target's at the D tokens each row verified,
    ``drafts`` ``[bs, D - 1]`` the drafts among them (the first token is the one accepted last time),
    ``starts`` ``[bs]`` each row's length before the step. Returns the tokens ``[bs, D]`` (int64; the
    first ``accept_lens[i]`` are row ``i``'s) and ``accept_lens`` ``[bs]`` (1 to D).

    All greedy: the target's argmax at each place, drafts taken while they equal it
    (``verify_tree_greedy``). Else target-only speculative sampling: draft ``j`` is taken with the
    probability the target gives it (coin ``j - 1`` below it); at the first one refused, the token is
    drawn from the target's distribution without it, and after the last draft from the whole
    distribution there (coin ``D``). Sampling parameters are normalized as ``sample.seeded@1``'s:
    temperature 0 is top-k 1."""
    bs, steps, vocab = logits.shape
    device = logits.device
    drafts = drafts.to(torch.int64)
    temps = [1.0 if 0 <= t < SAMPLING_EPS else float(t) for t in temperatures]
    ks = [1 if 0 <= t < SAMPLING_EPS else (TOP_K_ALL if k == -1 else int(k)) for t, k in zip(temperatures, top_ks)]
    if all(k == 1 for k in ks):
        predict = logits.argmax(dim=-1)
        matches = (drafts == predict[:, :-1]).to(torch.int64)
        accept_lens = 1 + matches.cumprod(dim=1).sum(dim=1)
        return predict, accept_lens
    scaled = logits.float() / _on_device(temps, torch.float32, device).view(-1, 1, 1)
    probs = torch.softmax(scaled, dim=-1).view(bs * steps, vocab)
    probs = _renorm(probs, _on_device(ks, torch.int64, device).repeat_interleave(steps).clamp_max(vocab),
                    _on_device(list(top_ps), torch.float32, device).repeat_interleave(steps)).view(bs, steps, vocab)
    coins = _seeded_coins(_on_device(seeds, torch.int64, device), starts.to(torch.int64), steps)
    rows = torch.arange(bs, device=device)
    alive = torch.ones(bs, dtype=torch.bool, device=device)
    accepted = torch.zeros(bs, dtype=torch.int64, device=device)
    refused = torch.full((bs,), -1, dtype=torch.int64, device=device)
    for j in range(steps - 1):
        p = probs[rows, j, drafts[:, j]]
        ok = alive & (p > 0) & (coins[:, j] < p)
        refused = torch.where(alive & ~ok, drafts[:, j], refused)
        accepted += ok.to(torch.int64)
        alive &= ok
    q = probs[rows, accepted]  # the distribution at the place drawn from
    residual = torch.where((torch.arange(vocab, device=device).view(1, -1) == refused.view(-1, 1)), 0.0, q)
    cdf = residual.cumsum(dim=-1)
    u = coins[:, steps] * cdf[:, -1]
    drawn = torch.searchsorted(cdf, u.view(-1, 1), right=True).view(-1).clamp_max(vocab - 1)
    tokens = torch.cat([drafts, drafts[:, :1]], dim=1)
    tokens[rows, accepted] = drawn
    return tokens, accepted + 1
