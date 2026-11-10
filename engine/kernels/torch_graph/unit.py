"""``kernels.torch_graph``: solutions for the steps a plan replays from a graph, where the
reference reads the host (stage2-contract.md §6). They require ``CAPTURE``, so the catalog gives
them to captured steps only, and an eager step keeps the reference."""

from __future__ import annotations

from typing import Any

from core.contracts.kernels import CAPTURE, Implementation, Solution
from kernels.torch_graph import ops


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        fn = ops.attention_decode
        ctx.register("kernels.catalog@1", Implementation(
            Solution("torch_graph.attn.decode@1", "attn.decode@1", f"{fn.__module__}:{fn.__name__}", library="torch",
                     capturable=True, requires=[CAPTURE]), fn), label="solution torch_graph.attn.decode@1")
