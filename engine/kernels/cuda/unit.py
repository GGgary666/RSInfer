"""``kernels.cuda``: kernels written in C++/CUDA and compiled when the unit loads (``ops.build``),
as SGLang JIT-compiles its own (python/sglang/kernels/jit/). For now one solution:
``kv.store@1``, the store into the KV pool that SGLang runs as its ``store_kvcache`` under every
attention backend. It needs a CUDA device: in a process without one the unit registers nothing,
and compiles nothing."""

from __future__ import annotations

from functools import partial
from typing import Any

from core.contracts.kernels import Implementation, Solution

BACKEND = "cuda"


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        devices = ctx.platform.devices()
        if not devices or "cuda" not in devices[0].capabilities:
            return
        import torch

        from kernels.cuda import ops

        module = ops.build()
        solution = f"{BACKEND}.kv.store@1"
        ctx.register("kernels.catalog@1", Implementation(
            Solution(solution, "kv.store@1", f"{ops.__name__}:kv_store", library=f"torch=={torch.__version__}",
                     capturable=True, requires=["cuda"]),
            partial(ops.kv_store, module)), label=f"solution {solution}")
