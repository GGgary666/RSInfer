"""``kernels.triton``: SGLang's Triton kernels as solutions of the GDN definitions
(stage2-contract.md §6), the backend SGLang names ``triton`` (``--linear-attn-backend``), and of
the fused glue of an attention layer, which SGLang runs on Triton under every attention backend.
They need a CUDA device: in a process without one the unit registers nothing, and does not
import Triton."""

from __future__ import annotations

from importlib.metadata import version
from typing import Any

from core.contracts.kernels import Implementation, Solution

BACKEND = "triton"

SOLUTIONS: tuple[tuple[str, str, bool, tuple[tuple[str, int], ...]], ...] = (
    ("gdn.conv1d_fn@1", "causal_conv1d_fn", False, ()),
    ("gdn.qkvzba_conv1d_update@1", "fused_qkvzba_causal_conv1d_update", True, (("conv_as_x", 1),)),
    ("gdn.gating@1", "fused_gdn_gating", True, ()),
    ("gdn.chunk_delta_rule@1", "chunk_gated_delta_rule", False, (("fp32", 0),)),
    ("gdn.packed_decode@1", "fused_recurrent_gated_delta_rule_packed_decode", True, ()),
    ("norm.rms_gated@1", "rms_norm_gated", True, ()),
    ("attn.qk_norm_rope_gate@1", "fused_qk_gemma_rmsnorm_rope_gate", True, (("fp32", 0),)),
    ("act.sigmoid_mul@1", "fused_sigmoid_mul", True, ()),
    ("attn.kv_indices@1", "create_flashinfer_kv_indices", True, ()),
    ("gdn.track_states@1", "track_states", True, ()),
)
"""(definition, function in ``ops``, capturable, the shape keys it fits one value of). The prefill
conv and the chunked delta rule read sequence lengths on the host, so a graph would freeze them. A
kernel that refuses fp32 activations fits only a caller whose shape says ``fp32`` 0, and the fused
decode conv only one whose conv state has the activations' dtype (``conv_as_x`` 1)."""


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        devices = ctx.platform.devices()
        if not devices or "cuda" not in devices[0].capabilities:
            return
        from kernels.triton import ops

        library = f"triton=={version('triton')}"
        for definition, name, capturable, shapes in SOLUTIONS:
            solution = f"{BACKEND}.{definition}"
            ctx.register("kernels.catalog@1", Implementation(
                Solution(solution, definition, f"{ops.__name__}:{name}", library=library,
                         shapes={key: [value] for key, value in shapes}, capturable=capturable, requires=["cuda"]),
                getattr(ops, name)), label=f"solution {solution}")
