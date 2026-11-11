"""``kernels.reference``: every definition Qwen3.8 needs, with its PyTorch reference solution
(stage2-contract.md §6), ported from SGLang (``ops``). Other kernel units add faster solutions of
the same definitions; a definition's signature is the one of the SGLang kernel it names."""

from __future__ import annotations

from typing import Any

from core.contracts.kernels import Definition, Implementation, KernelFn, Solution, Tolerance
from kernels.reference import ops
from kernels.reference.paged import PagedDecode, PagedExtend

LOOSE = Tolerance(atol=1e-2, rtol=1e-2)
TIGHT = Tolerance(atol=1e-5, rtol=1e-5)
"""For fp32 results: another exp / log / rsqrt, not another accumulation order in bf16."""

DEFINITIONS: tuple[tuple[str, str, KernelFn, Tolerance, bool], ...] = (
    ("norm.gemma_rms@1", "GemmaRMSNorm: fp32 RMS norm scaled by 1 + weight, optional residual add",
     ops.gemma_rms_norm, Tolerance(), False),
    ("norm.rms_gated@1", "RMSNormGated (rms_norm_ref): RMS norm, weight, silu(z)", ops.rms_norm_gated, LOOSE,
     False),
    ("pos.rope@1", "RotaryEmbedding: neox RoPE on the first rotary_dim of each head", ops.rope, Tolerance(), False),
    ("act.silu_and_mul@1", "SiluAndMul: silu(x[..., :d]) * x[..., d:]", ops.silu_and_mul, Tolerance(), False),
    ("gemm.linear@1", "x @ weight.T + bias", ops.linear, LOOSE, False),
    ("embed.lookup@1", "embedding rows by token id", ops.embedding, Tolerance(), True),
    ("kv.store@1", "set_kv_buffer: keys and values at their token slots", ops.kv_store, Tolerance(), True),
    ("attn.extend@1", "causal attention of each sequence's new tokens over req_to_token", ops.attention_extend,
     LOOSE, True),
    ("attn.decode@1", "attention of one token per sequence over req_to_token", ops.attention_decode, LOOSE, True),
    ("attn.paged_extend@1", "an operator planned per step: causal attention of each row's new tokens over its "
     "list of token slots", PagedExtend, LOOSE, True),
    ("attn.paged_decode@1", "an operator planned per step: attention of one token per row over its list of "
     "token slots", PagedDecode, LOOSE, True),
    ("gdn.conv1d_fn@1", "varlen causal conv1d with SiLU, from and into the conv state pool", ops.causal_conv1d_fn,
     LOOSE, True),
    ("gdn.conv1d_update@1", "one-token causal conv1d step on the conv state pool", ops.causal_conv1d_update,
     LOOSE, False),
    ("gdn.gating@1", "fused_gdn_gating: decay g and beta per token and value head", ops.fused_gdn_gating,
     TIGHT, False),
    ("gdn.chunk_delta_rule@1", "chunked gated delta rule per sequence on the temporal state pool",
     ops.chunk_gated_delta_rule, LOOSE, True),
    ("gdn.recurrent_update@1", "sigmoid gating and one delta-rule step on the temporal state pool",
     ops.fused_sigmoid_gating_delta_rule_update, LOOSE, False),
    ("sample.seeded@1", "Sampler: greedy, or top-k/top-p drawn from (seed, position)", ops.sample, Tolerance(),
     False),
)
"""(definition, what it computes, reference function, tolerance a solution is held to, batch
invariant). Batch invariant are the references that loop over sequences or only move data; the
others run one torch op over the whole batch (ops module docstring)."""


HOST_DEPENDENT = frozenset({"attn.extend@1", "attn.decode@1", "attn.paged_extend@1", "attn.paged_decode@1",
                            "gdn.conv1d_fn@1", "gdn.chunk_delta_rule@1", "sample.seeded@1"})
"""The references that read lengths or offsets on the host (a loop over sequences, a seed per row):
a graph would freeze what they read, so these are not capturable; the others are."""


def reference_id(definition: str) -> str:
    """Keeps the version: ``gemm.linear@1`` and ``gemm.linear@2`` can be registered side by side."""
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
