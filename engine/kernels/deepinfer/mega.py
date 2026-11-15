"""DeepGEMM's fused kernels as the backend ``deepinfer``'s sublayer step and MoE (``ds.hc_step@1``,
``ds.moe@1``), as DeepInfer runs them (``deepinfer/model_executor/models/v4_modules/hc.py`` and
``mlp.py``, a679bb8):

- ``mega_mhc``: the sublayer output written back to the Hyper-Connections stream, the next
  coefficients, this sublayer's mixed and normed input, and for the MoE its FP8 form written
  straight into the MoE buffer;
- ``bf16_mega_gate``: the gate's product, the choice and the experts' slots, into the same buffer;
- ``fp8_fp4_mega_moe``: the tokens sent to the ranks that hold their experts, both GEMMs with the
  SwiGLU between them there, the shared expert, and the weighted sum brought back, in one kernel
  launched in the same step on every rank of the group. A rank launches it with as many tokens as it
  has, none too.

The buffer (``deep_gemm.SymmBuffer``: symmetric memory over NVLink; plain device memory for one
rank) is the process's part in a collective object: made at the first call (every rank's first call
is in the same step) and kept by this unit's instance.

The expert stacks and the shared expert's weights are read in the fused kernel's own row order
(``formats.expert_layout``, ``shared_layout``): the model lays them out for the backend whose kernel
reads them.

Unlike the references these add a token's terms in bf16 and with ``fast_math``, quantize the SwiGLU's
result from fp32, and compute the coefficients' product in TF32 (``ops``, the differences listed
there).
"""
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

from typing import Any

import deep_gemm
import torch

from kernels.deepinfer import ops
from kernels.ds1017_formats import packed_words

BF16 = torch.bfloat16
E4M3 = torch.float8_e4m3fn
POST_SCALE = 2.0
"""``hc_post_mult_value``: what the post coefficients are scaled by (``ops.hc_mixes`` passes the same)."""


class Alone:
    """What ``SymmBuffer`` asks of a group, for one rank: its buffer is plain device memory."""

    def size(self) -> int:
        return 1

    def rank(self) -> int:
        return 0

    def barrier(self) -> None:
        return None


class Mega:
    """This rank's part of the ranks' shared MoE buffer, and the two fused solutions over it."""

    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        """Tokens a rank brings to one step at most (a prefill's budget and the decoding rows)."""
        self.buffer: Any = None
        self.made_for: tuple[int, ...] = ()
        self.staged = -1
        """How many tokens' MoE input the last ``hc_step`` wrote into the buffer; -1: none."""

    def symm(self, group: Any, experts: int, topk: int, hidden: int, inter: int) -> Any:
        made_for = (group.size, experts, topk, hidden, inter)
        if self.buffer is None:
            if group.size > 1 and group.device_group is None:
                raise ValueError("the fused MoE kernel shares memory between the ranks' cards: comm.device_backend = nccl")
            # DeepGEMM finds which SMs sit together once, and waits for the device doing it: here, not
            # in a step that is being recorded.
            deep_gemm.locality_domain.get_balanced_sm_locality_domains()
            self.buffer = deep_gemm.SymmBuffer(group.device_group if group.size > 1 else Alone(), num_experts=experts,
                                               num_max_tokens_per_rank=self.max_tokens, num_topk=topk, hidden=hidden,
                                               intermediate_hidden=inter, num_shared_experts=1, mma_type="fp8xfp4",
                                               activation="swiglu")
            self.made_for = made_for
        elif made_for != self.made_for:
            raise ValueError(f"the fused MoE buffer was made for {self.made_for}, not {made_for}")
        return self.buffer

    def hc_step(self, x: torch.Tensor | None, residual: torch.Tensor, pre: torch.Tensor, post: torch.Tensor | None,
                comb: torch.Tensor | None, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor,
                norm_weight: torch.Tensor, norm_eps: float, iters: int, hc_eps: float, stage: Any = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``ds.hc_step@1`` as ``HCLayerMixin._mega_mhc``: one ``deep_gemm.mega_mhc`` writes the sublayer
        output back, computes the next coefficients (Sinkhorn included), mixes and norms this
        sublayer's input, and, with ``stage`` (the sublayer is the MoE: ``(group, slots, topk, inter)``),
        quantizes it to FP8 straight into the MoE buffer, with the scales as the routed and the shared
        experts read them. Where nothing waits to be written back (the first sublayer, after an Engram
        layer) it is DeepInfer's unfused kernels (``_hc_pre``)."""
        self.staged = -1
        if x is None or post is None or comb is None:
            if stage is not None:
                raise ValueError("the MoE's input is staged by the fused kernel, which needs the sublayer before it")
            return ops.hc_step(x, residual, pre, post, comb, fn, scale, base, norm_weight, norm_eps, iters, hc_eps)
        tokens, hidden = x.shape
        hc, device = residual.shape[1], x.device
        stream = torch.empty(tokens, hc, hidden, dtype=BF16, device=device)
        new_pre = torch.empty(tokens, hc, 1, dtype=torch.float32, device=device)
        new_post = torch.empty(tokens, hc, 1, dtype=torch.float32, device=device)
        new_comb = torch.empty(tokens, hc, hc, dtype=torch.float32, device=device)
        y = torch.empty(tokens, hidden, dtype=BF16, device=device)
        y_fp8 = routed = shared = None
        block_m = 0
        if stage is not None:
            group, slots, topk, inter = stage
            if tokens > self.max_tokens:
                raise ValueError(f"{tokens} tokens in a step; the fused MoE buffer was sized for {self.max_tokens} "
                                 "(sched.token_budget + batch.max_rows)")
            symm = self.symm(group, slots, topk, hidden, inter)
            y_fp8, routed, shared = symm.x[:tokens], symm.x_sf[:tokens], symm.shared_l1_acts_sf[:tokens]
            block_m = deep_gemm.get_block_m_for_mega_moe(group.size, slots, symm.num_max_tokens_per_rank, tokens, topk,
                                                         "fp8xfp4")
        deep_gemm.mega_mhc(
            x.contiguous(), residual.contiguous(), pre.contiguous().view(tokens, hc, 1),
            post.contiguous().view(tokens, hc, 1), comb.contiguous(), fn=fn, mix_scales=scale, mix_bases=base, hc_mult=hc,
            hc_norm_eps=norm_eps, hc_pre_eps=hc_eps, hc_post_scale=POST_SCALE, sinkhorn_eps=hc_eps,
            num_sinkhorn_iters=iters, rmsnorm_weight=norm_weight, rmsnorm_eps=norm_eps, rmsnorm_scale=1.0,
            new_residual=stream, new_prev_mix=new_pre, new_post_mix=new_post, new_comb_res_mix=new_comb, y_bf16=y,
            y_fp8=y_fp8, y_gemm_sf=None, y_routed_sf=routed, y_shared_sf=shared, shared_sf_block_m=block_m)
        if stage is not None:
            self.staged = tokens
        return y, stream, new_pre.view(tokens, hc), new_post.view(tokens, hc), new_comb

    def moe(self, group: Any, x: torch.Tensor, gate_weight: torch.Tensor, gate_bias: torch.Tensor, gate_bf16: torch.Tensor,
            w13: torch.Tensor, p13: torch.Tensor, w2: torch.Tensor, p2: torch.Tensor, sw13: torch.Tensor, ss13: torch.Tensor,
            sp13: torch.Tensor, sw2: torch.Tensor, ss2: torch.Tensor, sp2: torch.Tensor, topk: int, route_scale: float,
            limit: float, to_physical: torch.Tensor | None = None, copies: torch.Tensor | None = None,
            taken: torch.Tensor | None = None) -> torch.Tensor:
        """``ds.moe@1`` as ``DeepseekMoE.mega_moe_forward``: ``bf16_mega_gate`` writes each token's
        slots and weights into the buffer (it maps experts to slots itself), and one
        ``fp8_fp4_mega_moe``, launched in the same step on every rank of the group, runs the routed
        experts where they are and the shared expert here, and counts the tokens each slot took. The
        input's FP8 form is in the buffer already: the ``hc_step`` before staged it. A rank with no
        tokens launches the kernel too. The stacks and the shared expert's weights are in the fused
        kernel's row order (``formats.expert_layout``, ``shared_layout``)."""
        tokens, hidden = x.shape
        local, inter = w13.shape[0], w13.shape[1] // 2
        symm = self.symm(group, local * group.size, topk, hidden, inter)
        if tokens:
            if self.staged != tokens:
                raise RuntimeError(f"the fused MoE reads {tokens} tokens' input from its buffer, where the step "
                                   f"before it staged {self.staged}")
            deep_gemm.bf16_mega_gate(
                x, gate_bf16, num_topk=topk, use_shared_as_routed=False, num_shared_experts=0,
                routed_scaling_factor=route_scale, ep_rank=group.rank, mask=None, bias=gate_bias, image_bias=None,
                image_token_mask=None, force_random=None, to_physical_map=to_physical, logical_count=copies,
                scoring_func="sqrtsoftplus", out=(symm.topk_idx[:tokens], symm.topk_weights[:tokens]))
        self.staged = -1
        out = torch.empty(tokens, hidden, dtype=BF16, device=x.device)
        deep_gemm.fp8_fp4_mega_moe(
            out, (w13.view(torch.int8), packed_words(p13, 2 * inter)), (w2.view(torch.int8), packed_words(p2, hidden)), symm,
            shared_l1_weights=(sw13.view(E4M3), packed_words(sp13, 2 * inter)),
            shared_l2_weights=(sw2.view(E4M3), packed_words(sp2, hidden)), cumulative_local_expert_recv_stats=taken,
            activation="swiglu", activation_clamp=limit if limit > 0 else None, fast_math=True)
        return out

    def close(self) -> None:
        if self.buffer is not None:
            self.buffer.destroy()
            self.buffer = None
