"""``kernels.reference``: every definition Qwen3.8 needs, with its PyTorch reference solution
(stage2-contract.md §6), ported from SGLang (``ops``). Other kernel units add faster solutions of
the same definitions; a definition's signature is the one of the SGLang kernel it names."""

from __future__ import annotations

from typing import Any

from core.contracts.kernels import Definition, Implementation, KernelFn, Solution, Tolerance
from kernels.reference import ops, spec
from kernels.reference.paged import PagedDecode, PagedExtend

LOOSE = Tolerance(atol=2e-2, rtol=2e-2)
"""For bf16 results: two units in the last place, which is up to 1.6% of a value. A reference
rounds to bf16 between its steps (a conv, then its activation; an add, then a norm) where a fused
kernel rounds once, so the two may end two units apart (seen on a SM100 at 1e-2: one element in
2.7 million)."""
TIGHT = Tolerance(atol=1e-5, rtol=1e-5)
"""For fp32 results: another exp / log / rsqrt, not another accumulation order in bf16."""

DEFINITIONS: tuple[tuple[str, str, KernelFn, Tolerance, bool], ...] = (
    ("norm.gemma_rms@1", "GemmaRMSNorm: fp32 RMS norm scaled by 1 + weight, optional residual add (the caller "
     "gives up x and residual: a solution may write its results over them)", ops.gemma_rms_norm, LOOSE, False),
    ("norm.rms_gated@1", "RMSNormGated (rms_norm_ref): RMS norm, weight, silu(z)", ops.rms_norm_gated, LOOSE,
     False),
    ("attn.qk_norm_rope_gate@1", "an attention layer's projection before it attends: GemmaRMSNorm of q and k per "
     "head, neox RoPE on the first rotary_dim of each head, the output gate taken apart from q",
     ops.fused_qk_gemma_rmsnorm_rope_gate, LOOSE, False),
    ("act.silu_and_mul@1", "SiluAndMul: silu(x[..., :d]) * x[..., d:]", ops.silu_and_mul, LOOSE, False),
    ("act.sigmoid_mul@1", "the attention output gate: x * sigmoid(gate) (the caller gives up x: a solution may "
     "write its result over it)", ops.fused_sigmoid_mul, LOOSE, False),
    ("gemm.linear@1", "x @ weight.T + bias", ops.linear, LOOSE, False),
    ("embed.lookup@1", "embedding rows by token id", ops.embedding, Tolerance(), True),
    ("kv.store@1", "set_kv_buffer: keys and values at their token slots", ops.kv_store, Tolerance(), True),
    ("attn.kv_indices@1", "each row's token slots as one list (create_flashinfer_kv_indices_triton), and "
     "optionally each row's last slot", ops.create_flashinfer_kv_indices, Tolerance(), True),
    ("attn.extend@1", "causal attention of each sequence's new tokens over req_to_token", ops.attention_extend,
     LOOSE, True),
    ("attn.decode@1", "attention of one token per sequence over req_to_token", ops.attention_decode, LOOSE, True),
    ("attn.paged_extend@1", "an operator planned per step: causal attention of each row's new tokens over its "
     "list of token slots", PagedExtend, LOOSE, True),
    ("attn.paged_decode@1", "an operator planned per step: attention of one token per row over its list of "
     "token slots", PagedDecode, LOOSE, True),
    ("gdn.conv1d_fn@1", "varlen causal conv1d with SiLU, from and into the conv state pool", ops.causal_conv1d_fn,
     LOOSE, True),
    ("gdn.qkvzba_conv1d_update@1", "a decode step's projections [q|k|v|z] and [b|a] taken apart, and the "
     "one-token causal conv1d step of [q|k|v] on the conv state pool", ops.fused_qkvzba_causal_conv1d_update,
     LOOSE, False),
    ("gdn.gating@1", "fused_gdn_gating: decay g and beta per token and value head", ops.fused_gdn_gating,
     TIGHT, False),
    ("gdn.chunk_delta_rule@1", "chunked gated delta rule per sequence on the temporal state pool",
     ops.chunk_gated_delta_rule, LOOSE, True),
    ("gdn.packed_decode@1", "sigmoid gating and one delta-rule step on the temporal state pool, from the "
     "convolved [q|k|v]", ops.fused_recurrent_gated_delta_rule_packed_decode, LOOSE, False),
    ("gdn.track_states@1", "for the prefix cache: every layer's conv and temporal state copied from each "
     "masked lane's slot to its tracking slot (track_mamba_states_all_layers)", ops.track_states, Tolerance(), True),
    ("sample.topk@1", "Unordered top-k logits and their token indices", ops.sample_topk, Tolerance(), False),
    ("sample.seeded@1", "Sampler: greedy, or top-k/top-p drawn from (seed, position)", ops.sample, Tolerance(),
     False),
    # Speculative decoding (doc/specs/mtp.md)
    ("gdn.verify_conv1d_update@1", "a verify step's causal conv1d over each row's D tokens, the window after each "
     "kept in the intermediate buffer (causal_conv1d_update with intermediate_conv_window)",
     spec.verify_conv1d_update, LOOSE, False),
    ("gdn.verify_recurrent@1", "a verify step's gated delta rule over each row's D tokens, the state after each "
     "kept in the intermediate buffer and the row's own left as it is (fused_sigmoid_gating_delta_rule_update, "
     "disable_state_update)", spec.verify_recurrent, LOOSE, False),
    ("gdn.commit_states@1", "every layer's state after a row's accepted step into its slot, and the one at a "
     "tracking boundary into the tracking slot (scatter_mamba_states_after_mtp_verify)", spec.commit_states,
     Tolerance(), True),
    ("spec.accept@1", "the drafts a chain verify accepts: greedy, or target-only speculative sampling with coins "
     "from (seed, position) (verify_tree_greedy, tree_speculative_sampling_target_only)", spec.accept, Tolerance(),
     False),
)
"""(definition, what it computes, reference function, tolerance a solution is held to, batch
invariant). Batch invariant are the references that loop over sequences or only move data; the
others run one torch op over the whole batch (ops module docstring)."""


HOST_DEPENDENT = frozenset({"attn.extend@1", "attn.decode@1", "attn.paged_extend@1", "attn.paged_decode@1",
                            "gdn.conv1d_fn@1", "gdn.chunk_delta_rule@1", "sample.seeded@1", "spec.accept@1"})
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
