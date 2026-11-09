"""``layers.linear``: SGLang's ``LinearBase`` forward at TP1 (``UnquantizedLinearMethod.apply``):
``gemm.linear@1`` on a weight view from the weight store."""

from __future__ import annotations

from typing import Any

import torch


class Linear:
    """``layers.linear@1``."""

    def __init__(self, catalog: Any) -> None:
        self.catalog = catalog

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return self.catalog.call("gemm.linear@1", {"in": weight.shape[1], "out": weight.shape[0]}, x, weight, bias)


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        ctx.provide("layers.linear@1", Linear(ctx.use("kernels.catalog@1")))
