"""``layers.activation``: SGLang's ``SiluAndMul``, and the output gate of its attention layers
(``fused_sigmoid_mul``)."""

from __future__ import annotations

from typing import Any

import torch


class Activation:
    """``layers.activation@1``."""

    def __init__(self, catalog: Any) -> None:
        self.catalog = catalog

    def silu_and_mul(self, x: torch.Tensor) -> torch.Tensor:
        shape = {"hidden": x.shape[-1] // 2, "fp32": int(x.dtype == torch.float32)}
        return self.catalog.call("act.silu_and_mul@1", shape, x)

    def sigmoid_mul(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """``x * sigmoid(gate)``. The kernel may work in place, as SGLang's does: ``x`` is not to be
        used afterwards."""
        return self.catalog.call("act.sigmoid_mul@1", {"hidden": x.shape[-1]}, x, gate)


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        ctx.provide("layers.activation@1", Activation(ctx.use("kernels.catalog@1")))
