"""``kernels.tilelang``: the TileLang kernels of the reference implementation shipped with the DS 1017
checkpoint (``kernel``), as solutions of the DS definitions (``kernels.ds1017``): the backend
``tilelang``. They need a CUDA device and the ``tilelang`` package; without either the unit registers
nothing. They are not capturable: their launches go through TVM's runtime, not the stream a graph
records."""

from __future__ import annotations

import importlib.util
from importlib.metadata import version
from typing import Any

from core.contracts.kernels import Implementation, Solution

BACKEND = "tilelang"

SOLUTIONS = ("ds.fake_quant_fp8@1", "ds.fake_quant_fp4@1", "ds.linear_fp8@1", "ds.sparse_attn@1", "ds.hc_mixes@1",
             "ds.moe_experts@1", "ds.linear@1", "ds.grouped_linear@1", "ds.index_score@1", "ds.moe_gate@1",
             "ds.rms_norm@1", "ds.hc_pre@1", "ds.hc_post@1", "ds.pool@1", "ds.engram_gate@1")
"""The definitions it solves; the function in ``ops`` has the definition's name. From ``ds.linear@1`` on
they are the torch products and reductions that implementation calls, not kernels of its own."""


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        devices = ctx.platform.devices()
        if not devices or "cuda" not in devices[0].capabilities or importlib.util.find_spec("tilelang") is None:
            return
        from kernels.tilelang import ops

        library = f"tilelang=={version('tilelang')}"
        for definition in SOLUTIONS:
            solution = f"{BACKEND}.{definition}"
            name = definition.split(".", 1)[1].split("@")[0]
            ctx.register("kernels.catalog@1", Implementation(
                Solution(solution, definition, f"{ops.__name__}:{name}", library=library, capturable=False,
                         requires=["cuda"]), getattr(ops, name)), label=f"solution {solution}")
