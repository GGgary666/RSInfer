"""DS 1017 on ranks that each hold part of the model (doc/specs/v1-ds1017.md stage 3): a layer's
routed experts split over the ranks in order (expert parallelism), an Engram table's rows likewise.
Written once here, over a backend's own experts kernel and the two collectives of ``comm.group@1``
(``worker.comm``): the reference and TileLang solutions of ``ds.ep_experts@1`` and
``ds.engram_fetch@1``. DeepInfer's solutions are its own kernels (``deep_gemm.fp8_fp4_mega_moe``,
``deep_ep.EngramBuffer``).

What a rank computes for a token is what one rank holding everything computes, bit for bit:

- an expert's term for a token depends on that token alone (the kernels below it are row invariant),
  so it is the same on whichever rank holds the expert, among whichever other tokens;
- every rank returns each token's terms one by one (not their sum); exactly one rank has a term for
  a token's choice and the others zero, so the sum over ranks is that term;
- the token's own rank then adds its terms in the experts' order, as ``moe_experts`` does.

Every rank calls these once per layer in every step, with no rows when it has none: the other ranks'
tokens may have chosen its experts.
"""

from __future__ import annotations

from typing import Any

import torch

from core.contracts.kernels import KernelFn


def mine(counts: list[int], rank: int) -> slice:
    """This rank's rows among the gathered ones."""
    start = sum(counts[:rank])
    return slice(start, start + counts[rank])


def ep_experts(experts: KernelFn, group: Any, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor,
               w13: torch.Tensor, p13: torch.Tensor, w2: torch.Tensor, p2: torch.Tensor, limit: float) -> torch.Tensor:
    """``ds.ep_experts@1``: ``ds.moe_experts@1`` where ``w13`` and ``w2`` stack this rank's experts
    only, experts ``rank * local`` onwards of the layer's; ``indices`` number the layer's experts."""
    local = w13.shape[0]
    first = group.rank * local
    (xs, ws, idx), counts = group.gather([x, weights, indices])
    held = (idx >= first) & (idx < first + local)
    terms = torch.zeros(*idx.shape, x.shape[-1], dtype=torch.float32, device=x.device)
    for slot in range(idx.shape[1]):  # one choice of a token at a time: each call returns that term alone
        rows = torch.nonzero(held[:, slot]).flatten()
        if rows.numel():
            terms[rows, slot] = experts(xs[rows], ws[rows, slot:slot + 1], idx[rows, slot:slot + 1] - first,
                                        w13, p13, w2, p2, limit)
    own = mine(counts, group.rank)
    terms, idx = group.reduce(terms)[own], idx[own]
    order = idx.argsort(dim=-1)  # a token chooses an expert once: the experts' order
    y = torch.zeros(terms.shape[0], x.shape[-1], dtype=torch.float32, device=x.device)
    for place in range(idx.shape[1]):
        y += torch.gather(terms, 1, order[:, place, None, None].expand(-1, 1, terms.shape[-1])).squeeze(1)
    return y


def engram_fetch(group: Any, ids: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """``ds.engram_fetch@1``: the rows ``ids`` ``[tokens, cols]`` of an Engram table as their bytes
    ``[tokens, cols, width]`` and scales. This rank holds the table's rows ``rank * part`` onwards,
    ``part`` of them (``weight``, ``scale``): the table is split over the ranks in order."""
    part = weight.shape[0]
    (every,), counts = group.gather([ids])
    at = every - group.rank * part
    held = ((at >= 0) & (at < part)).unsqueeze(-1)
    at = at.clamp(0, part - 1)
    # A row is on one rank: the others add zero. The collectives sum integers of 32 bits.
    rows = group.reduce(torch.where(held, weight[at], 0).to(torch.int32))
    scales = group.reduce(torch.where(held, scale[at], 0).to(torch.int32))
    own = mine(counts, group.rank)
    return rows[own].to(torch.uint8), scales[own].to(torch.uint8)
