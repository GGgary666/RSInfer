"""``kernels.ds1017``: the definitions DeepSeek-V4.1 (DS 1017) needs beyond Qwen3.8's, each with its
PyTorch reference solution (``ops``, after the reference implementation shipped with the checkpoint;
doc/specs/v1-ds1017.md). Other kernel units add faster solutions of the same definitions:
``kernels.tilelang``, that implementation's own TileLang kernels, and ``kernels.deepinfer``, the
kernel libraries DeepInfer runs. Two of the latter round at another place than that implementation
does (the routed experts' SwiGLU; an FP4 value exactly between two): a tolerance here does not cover
that, ``kernels/deepinfer/ops.py`` says where and how much, and its tests compare accordingly."""

from __future__ import annotations

from typing import Any

from core.contracts.kernels import CAPTURE, Definition, Implementation, KernelFn, Solution, Tolerance
from kernels.ds1017 import ops

EXACT = Tolerance()
LOOSE = Tolerance(atol=2e-2, rtol=2e-2)
"""For bf16 results of a GEMM: another accumulation order rounds a sum to the next bf16 value (the
TileLang GEMMs against the references: 1 to 3% of the elements, one unit in the last place)."""
TIGHT = Tolerance(atol=1e-5, rtol=1e-5)
"""For fp32 results: another exp or sigmoid."""
MIXED = Tolerance(atol=1e-3, rtol=2e-3)
"""For the Hyper-Connections coefficients: they come from a product with the whole stream, which
DeepInfer's kernel computes in TF32 (10 bits of each factor's mantissa): 2e-4 to 5e-4 from the fp32
reference on a GPU, on coefficients of order 1."""
CHAINED = Tolerance(atol=8e-2, rtol=8e-2)
"""For the routed experts: two quantized GEMMs with a rounding to FP8 between them, summed over a
token's experts. A unit in the last place of the first product can move the second's input to the
next FP8 value (against the reference implementation's kernels on a GPU: 4 of 71,680 elements beyond
``LOOSE``, the largest 0.03)."""
SUMMED = Tolerance(atol=8e-2, rtol=5e-2)
"""For a sum of bf16 terms kept in bf16 (the indexer's scores, over its heads): every addition
rounds, and another order rounds elsewhere (against the reference implementation's einsum and sum
on a GPU: 3 of 10,500 scores two units in the last place apart). The scores only rank latents."""

DEFINITIONS: tuple[tuple[str, str, KernelFn, Tolerance, bool], ...] = (
    ("ds.fake_quant_fp8@1", "FP8 per block of the last dimension with a power-of-two scale, dequantized again",
     ops.fake_quant_fp8, EXACT, True),
    ("ds.fake_quant_fp4@1", "FP4 per block with a power-of-two or an e4m3 scale, dequantized again",
     ops.fake_quant_fp4, EXACT, True),
    ("ds.linear_fp8@1", "x @ weight.T, x quantized to FP8 per 32, weight FP8 with 32 x 32 block scales",
     ops.linear_fp8, LOOSE, True),
    ("ds.linear@1", "x @ weight.T in the weight's dtype", ops.linear, LOOSE, True),
    ("ds.grouped_linear@1", "each group's values through that group's own matrix", ops.grouped_linear, LOOSE, True),
    ("ds.index_score@1", "the indexer's scores of each query against its row's keys", ops.index_score, SUMMED, True),
    ("ds.rms_norm@1", "RMS norm in fp32, scaled by the weight", ops.rms_norm, LOOSE, True),
    ("ds.rope@1", "rotation of the last rotary values by the tokens' rows of a frequency table", ops.rope, LOOSE,
     True),
    ("ds.sparse_attn@1", "attention of each query over the keys it lists by index, the latent both key and "
     "value, with a sink in the denominator", ops.sparse_attn, LOOSE, True),
    ("ds.hc_mixes@1", "a sublayer's Hyper-Connections coefficients from the residual stream (Sinkhorn)",
     ops.hc_mixes, MIXED, True),
    ("ds.hc_pre@1", "the stream's copies into one sublayer input", ops.hc_pre, LOOSE, True),
    ("ds.hc_post@1", "a sublayer's output back to the stream's copies", ops.hc_post, LOOSE, True),
    ("ds.moe_gate@1", "MoE routing: sqrtsoftplus scores, top-k by biased score, normalized weights",
     ops.moe_gate, TIGHT, True),
    ("ds.moe_experts@1", "the routed experts' sum: FP4 SwiGLU experts on FP8 activations, weighted",
     ops.moe_experts, CHAINED, True),
    ("ds.top@1", "the indices of the k largest scores along the last dimension", ops.top, EXACT, True),
    ("ds.silu@1", "x * sigmoid(x)", ops.silu, TIGHT, True),
    ("ds.pool@1", "a compression group's members pooled by the softmax of their scores", ops.pool, TIGHT, True),
    ("ds.engram_gate@1", "an Engram lookup's value added to the stream's copies through a gate", ops.engram_gate,
     LOOSE, True),
)
"""(definition, doc, reference, tolerance, batch invariant). Every reference sums in a fixed order and
multiplies a row on its own (``ops.total``, ``ops.matmul``), so a token's result does not depend on
the rest of the step."""

HOST_DEPENDENT = frozenset({"ds.moe_experts@1"})
"""References that read the host (which experts the tokens chose): not capturable."""


def reference_id(definition: str) -> str:
    return f"reference.{definition}"


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        for name, doc, fn, tolerance, batch_invariant in DEFINITIONS:
            ref = reference_id(name)
            ctx.register("kernels.catalog@1", Definition(name, doc, ref, tolerance, batch_invariant),
                         label=f"definition {name}")
            ctx.register("kernels.catalog@1", Implementation(Solution(ref, name, f"{fn.__module__}:{fn.__name__}",
                                                                      library="torch",
                                                                      capturable=name not in HOST_DEPENDENT), fn),
                         label=f"solution {ref}")
        # For a step replayed from a graph: every expert on every token, with fixed shapes.
        fn = ops.moe_experts_masked
        ctx.register("kernels.catalog@1", Implementation(
            Solution("torch_graph.ds.moe_experts@1", "ds.moe_experts@1", f"{fn.__module__}:{fn.__name__}",
                     library="torch", capturable=True, requires=[CAPTURE]), fn),
            label="solution torch_graph.ds.moe_experts@1")
