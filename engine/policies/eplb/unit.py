# Copyright DeepInfer authors
# SPDX-License-Identifier: Apache-2.0
"""``policy.rebalance@1``: where each expert of a layer goes when the experts are placed again (expert
parallel load balancing, doc/specs/v1-ds1017.md stage 3). The decision only: counting the tokens,
moving the weights and the maps the kernels read are ``state.expert_placement``'s.

From ``deepinfer/model_executor/eplb/policy.py`` of DeepInfer ``a679bb8`` (LICENSE.deepinfer in this
directory), the same two greedy passes:

- **Replicas.** A layer has more slots than experts (``physical - logical`` spare ones). They go one by
  one to the expert whose copies carry most, judged over the steps of the window: the copy is given
  where it lowers the sum over steps of the largest load a single copy carries.
- **Ranks.** The copies are then dealt to the ranks, heaviest first, each to the rank where the sum
  over steps of the largest rank's load stays lowest, ``local`` copies a rank.

A logical expert is one the model's gate chooses; a physical slot is a place on a rank that holds one
expert's weights. Every rank computes this from the same counts, so it has to give the same answer
everywhere: it runs on the host, on integers and fp32 sums in a fixed order.
"""

from __future__ import annotations

from typing import Any

import torch


def deal(tokens: torch.Tensor, physical_to_logical: torch.Tensor, copies: torch.Tensor, local: int) -> torch.Tensor:
    """``_batch_greedy_indices``: for each layer, the slots' order such that each run of ``local`` is
    one rank's. ``tokens`` ``[steps, layers, logical]``, ``physical_to_logical`` ``[layers, physical]``,
    ``copies`` ``[layers, logical]``."""
    weights = tokens.sum(dim=0, dtype=torch.float32) / copies
    experts = physical_to_logical.to(torch.int64)
    # Heaviest copy first; of equals, in slot order.
    order = weights.gather(-1, experts).argsort(dim=-1, descending=True, stable=True)
    by_load = experts.gather(-1, order)
    vectors = tokens.permute(1, 2, 0).to(dtype=torch.float32, copy=True).contiguous()
    vectors.div_(copies.unsqueeze(-1))
    layers, _, steps = vectors.shape
    physical = order.size(1)
    ranks = physical // local
    loads = vectors.new_zeros((layers, ranks, steps))
    peaks = vectors.new_zeros((layers, steps))
    held = torch.zeros((layers, ranks), dtype=torch.int64)
    out = torch.empty_like(order)
    each = torch.arange(layers)
    for job in range(physical):
        vector = vectors[each, by_load[:, job]]
        # With this copy on rank j: each step's largest rank load; the rank with the lowest sum of
        # them takes it (the lowest rank of equals), unless it is full.
        scores = torch.maximum(loads + vector.unsqueeze(1), peaks.unsqueeze(1)).sum(dim=-1)
        scores.masked_fill_(held == local, float("inf"))
        rank = scores.argmin(dim=-1)
        place = held[each, rank]
        out[each, rank * local + place] = order[:, job]
        held[each, rank] = place + 1
        updated = loads[each, rank] + vector
        loads[each, rank] = updated
        peaks = torch.maximum(peaks, updated)
    return out


def place(tokens: torch.Tensor, physical: int, local: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``rebalance_experts_global``. ``tokens`` ``[steps, layers, logical]``: the tokens each expert
    took in each step of the window. Returns ``physical_to_logical`` ``[layers, physical]`` (rank ``r``
    holds the slots ``r * local`` onwards), ``logical_to_physical`` ``[layers, logical, spare + 1]``
    (an expert's slots, ``-1`` after them) and each expert's number of slots ``[layers, logical]``."""
    tokens = tokens.to("cpu")
    _, layers, logical = tokens.shape
    spare = physical - logical
    if spare < 0 or physical % local:
        raise ValueError(f"{physical} slots for {logical} experts, {local} a rank")
    p2l = torch.empty(layers, physical, dtype=torch.int32)
    l2p = torch.full((layers, logical, spare + 1), -1, dtype=torch.int32)
    copies = torch.ones(layers, logical, dtype=torch.int32)
    each_layer = torch.arange(layers)
    each_expert = torch.arange(logical, dtype=torch.int32)
    p2l[:, :logical] = each_expert
    l2p[:, :, 0] = each_expert
    # No two experts carry the same: of equals the later one counts as heavier.
    distinct = tokens + each_expert * 1e-4
    for i in range(spare):
        score = distinct / copies
        with_one_more = tokens / (copies + 1)
        values, heaviest = score.max(-1, keepdim=True)
        values = values.expand_as(score).contiguous()
        score.scatter_(-1, heaviest, with_one_more.gather(-1, heaviest))
        values.scatter_(-1, heaviest, score.max(-1, keepdim=True).values)
        chosen = values.sum(0).argmin(-1)  # per layer, the expert copied this round
        slot = logical + i
        p2l[:, slot] = chosen.to(torch.int32)
        l2p[each_layer, chosen, copies[each_layer, chosen].long()] = slot
        copies[each_layer, chosen] += 1
    if local > 1:
        order = deal(tokens, p2l, copies, local)
        p2l = p2l.gather(-1, order)
        # order.argsort: old slot -> new slot.
        missing = l2p == -1
        moved = order.argsort(-1).gather(-1, l2p.masked_fill(missing, 0).view(layers, -1).to(torch.int64))
        l2p = moved.view_as(l2p).to(torch.int32).masked_fill(missing, -1)
    return p2l, l2p, copies


class Greedy:
    """``policy.rebalance@1``."""

    def place(self, tokens: torch.Tensor, physical: int, local: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return place(tokens, physical, local)


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        ctx.provide("policy.rebalance@1", Greedy())
