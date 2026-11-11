"""``kernels.triton``: SGLang's Triton kernels as solutions of the GDN definitions
(stage2-contract.md §6), the backend SGLang names ``triton`` (``--linear-attn-backend``). They
need a CUDA device: in a process without one the unit registers nothing, and does not import
Triton."""

from __future__ import annotations

from importlib.metadata import version
from typing import Any

from core.contracts.kernels import Implementation, Solution

BACKEND = "triton"

SOLUTIONS: tuple[tuple[str, str, bool, bool], ...] = (
    ("gdn.conv1d_fn@1", "causal_conv1d_fn", False, True),
    ("gdn.conv1d_update@1", "causal_conv1d_update", True, True),
    ("gdn.gating@1", "fused_gdn_gating", True, True),
    ("gdn.chunk_delta_rule@1", "chunk_gated_delta_rule", False, False),
    ("gdn.recurrent_update@1", "fused_sigmoid_gating_delta_rule_update", True, True),
    ("norm.rms_gated@1", "rms_norm_gated", True, True),
)
"""(definition, function in ``ops``, capturable, takes fp32 activations). The prefill conv and the
chunked delta rule read sequence lengths on the host, so a graph would freeze them. A kernel that
refuses fp32 activations fits only a caller whose shape says ``fp32`` 0."""


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        devices = ctx.platform.devices()
        if not devices or "cuda" not in devices[0].capabilities:
            return
        from kernels.triton import ops

        library = f"triton=={version('triton')}"
        for definition, name, capturable, fp32 in SOLUTIONS:
            solution = f"{BACKEND}.{definition}"
            ctx.register("kernels.catalog@1", Implementation(
                Solution(solution, definition, f"{ops.__name__}:{name}", library=library,
                         shapes={} if fp32 else {"fp32": [0]}, capturable=capturable, requires=["cuda"]),
                getattr(ops, name)), label=f"solution {solution}")
