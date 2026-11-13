"""``layers.sampler``: SGLang's ``Sampler`` with per-request seeds (``sample.seeded@1``). A
request's seed is set by the scheduler at admission (``SamplingParams.seed``); none draws as 0."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from core.contracts.sampling import SamplingParams


class Sampler:
    """``layers.sampler@1``."""

    def __init__(self, catalog: Any) -> None:
        self.catalog = catalog

    def sample(self, logits: torch.Tensor, params: Sequence[SamplingParams], positions: Sequence[int]) -> torch.Tensor:
        """One token per row of ``logits``; ``positions``: of each row's last input token."""
        return self.catalog.call("sample.seeded@1", {"vocab": logits.shape[-1]}, logits,
                                 [p.temperature for p in params], [p.top_p for p in params],
                                 [p.top_k for p in params], [p.seed or 0 for p in params], positions)

    def accept(self, logits: torch.Tensor, drafts: torch.Tensor, params: Sequence[SamplingParams],
               starts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """A verify step's tokens ``[rows, D]`` and accepted lengths ``[rows]`` (``spec.accept@1``,
        doc/specs/mtp.md): ``logits`` ``[rows, D, vocab]``, ``drafts`` ``[rows, D - 1]``, ``starts`` each
        row's length before the step."""
        return self.catalog.call("spec.accept@1", {"vocab": logits.shape[-1]}, logits, drafts,
                                 [p.temperature for p in params], [p.top_p for p in params],
                                 [p.top_k for p in params], [p.seed or 0 for p in params], starts)


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        ctx.provide("layers.sampler@1", Sampler(ctx.use("kernels.catalog@1")))
