"""``kernels.flashinfer``: FlashInfer's paged attention as solutions of the paged attention
definitions (stage2-contract.md §6), the backend SGLang names ``flashinfer``
(``--attention-backend``). They need a CUDA device: in a process without one the unit registers
nothing, and does not import FlashInfer.

FlashInfer takes bf16 and fp16 (and fp8 caches), not fp32: its solutions fit a caller whose
shape says ``fp32`` 0. It compiles a kernel module the first time a wrapper of a new shape and
dtype is planned, into its cache directory (``FLASHINFER_WORKSPACE_BASE``, by default under the
user's home), with ``ninja`` and the CUDA compiler found on ``PATH``."""

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
                         shapes={"fp32": [0]}, capturable=operator is ops.PagedDecode, requires=["cuda"]),
                partial(operator, self.workspace)), label=f"solution {solution}")
