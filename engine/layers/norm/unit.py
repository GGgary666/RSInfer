"""``layers.norm``: SGLang's ``GemmaRMSNorm`` (with the fused residual add) and ``RMSNormGated``
(``norm_before_gate``, no groups)."""

from __future__ import annotations

from typing import Any

import torch


class Norm:
    """``layers.norm@1``."""

    def __init__(self, catalog: Any) -> None:
        self.catalog = catalog

    def gemma_rms(self, x: torch.Tensor, weight: torch.Tensor, eps: float, residual: torch.Tensor | None = None
                  ) -> Any:
        """``normed``, or ``(normed, x + residual)`` with a residual. With a residual the kernel may
        work in place, as SGLang's does: ``x`` and ``residual`` are not to be used afterwards."""
        return self.catalog.call("norm.gemma_rms@1", {"hidden": x.shape[-1]}, x, weight, eps, residual)

    def rms_gated(self, x: torch.Tensor, weight: torch.Tensor, z: torch.Tensor, eps: float) -> torch.Tensor:
        return self.catalog.call("norm.rms_gated@1", {"hidden": x.shape[-1]}, x, weight, None, z, eps, None, True)


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        ctx.provide("layers.norm@1", Norm(ctx.use("kernels.catalog@1")))
