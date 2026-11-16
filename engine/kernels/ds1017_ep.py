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

from collections.abc import Sequence
from typing import Any

import torch

from core.contracts.kernels import KernelFn


def mine(counts: list[int], rank: int) -> slice:
    """This rank's rows among the gathered ones."""
    start = sum(counts[:rank])
    return slice(start, start + counts[rank])


def ep_experts(experts: KernelFn, group: Any, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor,
               w13: torch.Tensor, p13: torch.Tensor, w2: torch.Tensor, p2: torch.Tensor, limit: float,
               slots: torch.Tensor | None = None, taken: torch.Tensor | None = None) -> torch.Tensor:
    """``ds.ep_experts@1``: ``ds.moe_experts@1`` where ``w13`` and ``w2`` stack the experts in this
    rank's slots only, slots ``rank * local`` onwards of the layer's; ``indices`` number the layer's
    experts. ``slots`` ``[tokens, k]``: the slot that runs each chosen expert (``expert.placement@1``);
    ``None``: slot ``i`` holds expert ``i``. ``taken`` ``[local]`` int32: the tokens each of this
    rank's slots ran are added to it."""
    local = w13.shape[0]
    first = group.rank * local
    (xs, ws, at), counts = group.gather([x, weights, indices if slots is None else slots])
    held = (at >= first) & (at < first + local)
    terms = torch.zeros(*at.shape, x.shape[-1], dtype=torch.float32, device=x.device)
    for slot in range(at.shape[1]):  # one choice of a token at a time: each call returns that term alone
        rows = torch.nonzero(held[:, slot]).flatten()
        if rows.numel():
            terms[rows, slot] = experts(xs[rows], ws[rows, slot:slot + 1], at[rows, slot:slot + 1] - first,
                                        w13, p13, w2, p2, limit)
    if taken is not None:
        taken += torch.bincount((at[held] - first).flatten(), minlength=local).to(taken.dtype)
    own = mine(counts, group.rank)
    terms, idx = group.reduce(terms)[own], indices
    order = idx.argsort(dim=-1)  # a token chooses an expert once: the experts' order, wherever they ran
    y = torch.zeros(terms.shape[0], x.shape[-1], dtype=torch.float32, device=x.device)
    for place in range(idx.shape[1]):
        y += torch.gather(terms, 1, order[:, place, None, None].expand(-1, 1, terms.shape[-1])).squeeze(1)
    return y


def engram_fetch(group: Any, ids: torch.Tensor, weights: Sequence[torch.Tensor], scales: Sequence[torch.Tensor]
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """``ds.engram_fetch@1``: for each Engram layer ``l`` the rows ``ids[:, l]`` (``ids`` ``[tokens,
    layers, cols]``) of its table, as their bytes ``[tokens, layers, cols, width]`` and scales. This
    rank holds each table's rows ``rank * part`` onwards, ``part`` of them (``weights[l]``,
    ``scales[l]``): a table is split over the ranks in order. One exchange for all the layers."""
    (every,), counts = group.gather([ids])
    rows: list[torch.Tensor] = []
    for layer, (weight, scale) in enumerate(zip(weights, scales, strict=True)):
        part = weight.shape[0]
        at = every[:, layer] - group.rank * part
        held = ((at >= 0) & (at < part)).unsqueeze(-1)
        at = at.clamp(0, part - 1)
        # A row is on one rank: the others add zero.
        rows.append(torch.cat([torch.where(held, weight[at], 0), torch.where(held, scale[at], 0)], dim=-1))
    width = weights[0].shape[-1]
    # The collectives sum integers of 32 bits.
    both = group.reduce(torch.stack(rows, dim=1).to(torch.int32))[mine(counts, group.rank)].to(torch.uint8)
    return both[..., :width].contiguous(), both[..., width:].contiguous()
