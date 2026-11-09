"""``layers.embedding``: SGLang's ``VocabParallelEmbedding`` and the LM head of its
``LogitsProcessor`` (``_compute_lm_head``) at TP1."""

from __future__ import annotations

from typing import Any

import torch


class Embedding:
    """``layers.embedding@1``."""

    def __init__(self, catalog: Any) -> None:
        self.catalog = catalog

    def embed(self, input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return self.catalog.call("embed.lookup@1", {}, input_ids, weight)

    def logits(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """``hidden_states @ lm_head.weight.T`` in the weight's dtype."""
        return self.catalog.call("gemm.linear@1", {"in": weight.shape[1], "out": weight.shape[0]},
                                 hidden_states.to(weight.dtype), weight)


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        ctx.provide("layers.embedding@1", Embedding(ctx.use("kernels.catalog@1")))
