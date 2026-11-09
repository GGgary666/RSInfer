"""``layers.activation``: SGLang's ``SiluAndMul``."""

from __future__ import annotations

from typing import Any

import torch


class Activation:
    """``layers.activation@1``."""

    def __init__(self, catalog: Any) -> None:
        self.catalog = catalog

    def silu_and_mul(self, x: torch.Tensor) -> torch.Tensor:
        return self.catalog.call("act.silu_and_mul@1", {"hidden": x.shape[-1] // 2}, x)


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        ctx.provide("layers.activation@1", Activation(ctx.use("kernels.catalog@1")))
