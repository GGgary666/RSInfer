"""``kernels.deepinfer``: the kernel libraries DeepInfer runs DeepSeek-V4.1 on (DeepGEMM, FlashMLA,
TileKernels, DeepSelect), as solutions of the DS definitions (``kernels.ds1017``): the backend
``deepinfer``. They need a GPU of SM 10.0 and the four libraries in the engine's Python environment;
without either the unit registers nothing. Which library call solves which definition, and where its
arithmetic differs from the references, is in ``ops``."""

from __future__ import annotations

import importlib.util
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from core.contracts.kernels import CAPTURE, Implementation, Solution

BACKEND = "deepinfer"
LIBRARIES = ("deep_gemm", "flash_mla", "tile_kernels", "deep_select")

SOLUTIONS: tuple[tuple[str, str, dict[str, list[int]]], ...] = (
    ("ds.fake_quant_fp8@1", "tile_kernels", {}), ("ds.fake_quant_fp4@1", "tile_kernels", {}),
    ("ds.linear_fp8@1", "deep_gemm", {}), ("ds.linear@1", "deep_gemm", {}), ("ds.grouped_linear@1", "deep_gemm", {}),
    ("ds.rms_norm@1", "tile_kernels", {}),
    # FlashMLA's kernels are built for these.
    ("ds.sparse_attn@1", "flash_mla", {"heads": [64, 128], "head_dim": [512]}),
    ("ds.cache_attn@1", "flash_mla", {"heads": [64, 128], "head_dim": [512]}), ("ds.pack_kv@1", "tile_kernels", {}),
    ("ds.hc_mixes@1", "tile_kernels", {}), ("ds.hc_pre@1", "tile_kernels", {}), ("ds.hc_post@1", "tile_kernels", {}),
    ("ds.moe_gate@1", "tile_kernels", {}), ("ds.moe_experts@1", "deep_gemm", {}), ("ds.top@1", "deep_select", {}),
    ("ds.engram_gate@1", "tile_kernels", {}),
    ("ds.silu@1", "torch", {}), ("ds.pool@1", "torch", {}), ("ds.index_score@1", "torch", {}),
)
"""(definition, the library its kernel is from, the shapes it is built for); the function in ``ops``
has the definition's name. A CUDA graph may record every one of them (their launches are on torch's
current stream; tests/gpu/test_ds1017_deepinfer.py replays each) but ``HOST_DEPENDENT``."""

HOST_DEPENDENT = frozenset({"ds.moe_experts@1"})
"""Which read the device back: the routed experts size their rows to the step's tokens. A captured
step runs ``FIXED`` instead."""
FIXED = "deepinfer.fixed.ds.moe_experts@1"


def library(name: str) -> str:
    try:
        return f"{name}=={version(name)}"
    except PackageNotFoundError:
        return name


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        devices = ctx.platform.devices()
        if (not devices or "sm_100" not in devices[0].capabilities
                or any(importlib.util.find_spec(name) is None for name in LIBRARIES)):
            return
        from kernels.deepinfer import ops

        ops.align_experts()
        for definition, source, shapes in SOLUTIONS:
            solution = f"{BACKEND}.{definition}"
            name = definition.split(".", 1)[1].split("@")[0]
            ctx.register("kernels.catalog@1", Implementation(
                Solution(solution, definition, f"{ops.__name__}:{name}", library=library(source), shapes=shapes,
                         capturable=definition not in HOST_DEPENDENT, requires=["cuda", "sm_100"]), getattr(ops, name)),
                label=f"solution {solution}")
        fn = ops.moe_experts_fixed
        ctx.register("kernels.catalog@1", Implementation(
            Solution(FIXED, "ds.moe_experts@1", f"{ops.__name__}:{fn.__name__}", library=library("deep_gemm"),
                     capturable=True, requires=["cuda", "sm_100", CAPTURE]), fn), label=f"solution {FIXED}")
