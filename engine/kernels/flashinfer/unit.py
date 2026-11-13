"""``kernels.flashinfer``: FlashInfer's paged attention as solutions of the paged attention
definitions (stage2-contract.md §6), the backend SGLang names ``flashinfer``
(``--attention-backend``), and its Gemma RMSNorm and SiLU-and-mul as solutions of
``norm.gemma_rms@1`` and ``act.silu_and_mul@1``, which SGLang runs under every backend. They need
a CUDA device: in a process without one the unit registers nothing, and does not import
FlashInfer.

FlashInfer's attention takes bf16 and fp16, not fp32, and reads a 16-bit cache as the queries'
type without converting it (a bf16 model over an fp16 cache gives wrong numbers): those
solutions fit a caller whose shape says ``fp32`` 0 and ``kv_as_q`` 1. Its activation takes bf16
and fp16 too (``fp32`` 0); its norm takes fp32 as well. It compiles a kernel module the first time a wrapper of a new shape and dtype is
planned, or an activation first runs, into its cache directory (``FLASHINFER_WORKSPACE_BASE``, by
default under the user's home), with ``ninja`` and the CUDA compiler found on ``PATH``."""

from __future__ import annotations

from functools import partial
from importlib.metadata import version
from typing import Any

from core.contracts.kernels import Implementation, Solution

BACKEND = "flashinfer"


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        devices = ctx.platform.devices()
        if not devices or "cuda" not in devices[0].capabilities:
            return
        from kernels.flashinfer import ops

        self.workspace = ops.Workspace()
        library = f"flashinfer=={version('flashinfer-python')}"
        for definition, operator in (("attn.paged_extend@1", ops.PagedExtend), ("attn.paged_decode@1", ops.PagedDecode)):
            solution = f"{BACKEND}.{definition}"
            ctx.register("kernels.catalog@1", Implementation(
                Solution(solution, definition, f"{ops.__name__}:{operator.__name__}", library=library,
                         shapes={"fp32": [0], "kv_as_q": [1]}, capturable=True,
                         requires=["cuda"]),
                partial(operator, self.workspace)), label=f"solution {solution}")
        for definition, fn, fp32 in (("norm.gemma_rms@1", ops.gemma_rms_norm, True),
                                     ("act.silu_and_mul@1", ops.silu_and_mul, False)):
            solution = f"{BACKEND}.{definition}"
            ctx.register("kernels.catalog@1", Implementation(
                Solution(solution, definition, f"{ops.__name__}:{fn.__name__}", library=library,
                         shapes={} if fp32 else {"fp32": [0]}, capturable=True, requires=["cuda"]), fn),
                label=f"solution {solution}")
