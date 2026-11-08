"""``kernels.reference``: every definition Qwen3.8 needs, with its PyTorch reference solution
(stage2-contract.md §6). Other kernel units add faster solutions of the same definitions."""

from __future__ import annotations

from typing import Any

from core.contracts.kernels import Definition, Implementation, KernelFn, Solution, Tolerance
from kernels.reference import ops

DEFINITIONS: tuple[tuple[str, str, KernelFn, Tolerance], ...] = (
    ("norm.gemma_rms@1", "RMS norm in fp32 scaled by 1 + weight", ops.gemma_rms_norm, Tolerance()),
    ("norm.rms_gated@1", "RMS norm, weight, then silu(gate)", ops.rms_norm_gated, Tolerance()),
    ("pos.rope_neox@1", "rotate-half RoPE on the first rotary_dim of each head", ops.rope_neox, Tolerance()),
    ("act.silu_and_mul@1", "silu(x[..., :d]) * x[..., d:]", ops.silu_and_mul, Tolerance()),
    ("gemm.linear@1", "x @ weight.T + bias", ops.linear, Tolerance(atol=1e-2, rtol=1e-2)),
    ("embed.lookup@1", "embedding rows by token id", ops.embedding, Tolerance()),
    ("kv.store@1", "write keys and values at their token slots", ops.kv_store, Tolerance()),
    ("attn.paged@1", "causal attention of new tokens over a paged sequence", ops.paged_attention,
     Tolerance(atol=1e-2, rtol=1e-2)),
    ("gdn.conv1d@1", "depthwise causal conv with SiLU, continuing from the conv state", ops.causal_conv1d,
     Tolerance(atol=1e-2, rtol=1e-2)),
    ("gdn.gating@1", "GDN decay and beta per token and value head", ops.gdn_gating, Tolerance()),
    ("gdn.delta_rule@1", "gated delta rule, continuing from the recurrent state", ops.gated_delta_rule,
     Tolerance(atol=1e-2, rtol=1e-2)),
    ("sample.seeded@1", "one token per row, drawn from (seed, position)", ops.sample, Tolerance()),
)
"""(definition, what it computes, reference function, tolerance a solution is held to)."""


def reference_id(definition: str) -> str:
    return f"reference.{definition.partition('@')[0]}"


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        for name, doc, fn, tolerance in DEFINITIONS:
            ref = reference_id(name)
            ctx.register("kernels.catalog@1", Definition(name, doc, ref, tolerance), label=f"definition {name}")
            ctx.register("kernels.catalog@1", Implementation(Solution(ref, name, f"{fn.__module__}:{fn.__name__}",
                                                                      library="torch", capturable=False), fn),
                         label=f"solution {ref}")
