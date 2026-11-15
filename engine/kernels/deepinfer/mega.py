"""DeepGEMM's fused MoE kernel as the routed experts of the backend ``deepinfer`` among ranks that
share a layer's experts (``ds.ep_experts@1``), as ``DeepseekMoE.mega_moe_forward`` calls it
(``deepinfer/model_executor/models/v4_modules/mlp.py``, a679bb8): the tokens' FP8 activations, the
experts' slots and weights go into this rank's part of a buffer every rank of the group reads
(``deep_gemm.SymmBuffer``: symmetric memory over NVLink), and one kernel, launched in the same step
on every rank, sends each token to the ranks that hold its experts, runs both GEMMs with the SwiGLU
between them there, and brings the weighted sum back. A rank launches it with as many tokens as it
has, none too.

The buffer is the process's part in a collective object: made at the first call (every rank's first
call is in the same step) and kept by this unit's instance.

The expert stacks are read in the kernel's own row order (``formats.expert_layout``): the model lays
them out for the backend whose kernel reads them.

Unlike the references it adds a token's terms in bf16 and with ``fast_math``, and quantizes the
SwiGLU's result from fp32 (``ops``, the first difference listed there).
"""
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

from typing import Any

import deep_gemm
import tile_kernels
import torch

from kernels.ds1017_formats import BLOCK, packed_words

BF16 = torch.bfloat16


class Alone:
    """What ``SymmBuffer`` asks of a group, for one rank: its buffer is plain device memory."""

    def size(self) -> int:
        return 1

    def rank(self) -> int:
        return 0

    def barrier(self) -> None:
        return None


class Mega:
    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        """Tokens a rank brings to one step at most (a prefill's budget and the decoding rows)."""
        self.buffer: Any = None
        self.made_for: tuple[int, ...] = ()

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
                                               intermediate_hidden=inter, num_shared_experts=0, mma_type="fp8xfp4",
                                               activation="swiglu")
            self.made_for = made_for
        elif made_for != self.made_for:
            raise ValueError(f"the fused MoE buffer was made for {self.made_for}, not {made_for}")
        return self.buffer

    def ep_experts(self, group: Any, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor, w13: torch.Tensor,
                   p13: torch.Tensor, w2: torch.Tensor, p2: torch.Tensor, limit: float,
                   slots: torch.Tensor | None = None, taken: torch.Tensor | None = None) -> torch.Tensor:
        """``ds.ep_experts@1``. ``w13``, ``p13``, ``p2`` in the fused kernel's row order."""
        tokens, topk = indices.shape
        local, inter, hidden = w13.shape[0], w13.shape[1] // 2, x.shape[-1]
        if tokens > self.max_tokens:
            raise ValueError(f"{tokens} tokens in a step; the fused MoE buffer was sized for {self.max_tokens} "
                             "(sched.token_budget + batch.max_rows)")
        symm = self.symm(group, local * group.size, topk, hidden, inter)
        if tokens:
            rows = x.reshape(-1, hidden).contiguous()
            tile_kernels.quant.per_token_cast(rows, fmt="e4m3", num_per_channels=BLOCK, round_sf=True, use_packed_ue8m0=True,
                                              use_tma_aligned_col_major_sf=False, out=(symm.x[:tokens], symm.x_sf[:tokens]))
            symm.topk_idx[:tokens].copy_(indices if slots is None else slots)
            symm.topk_weights[:tokens].copy_(weights)
        out = torch.empty(tokens, hidden, dtype=BF16, device=x.device)
        deep_gemm.fp8_fp4_mega_moe(out, (w13.view(torch.int8), packed_words(p13, 2 * inter)),
                                   (w2.view(torch.int8), packed_words(p2, hidden)), symm,
                                   cumulative_local_expert_recv_stats=taken, activation="swiglu",
                                   activation_clamp=limit if limit > 0 else None, fast_math=True)
        return out.float()

    def close(self) -> None:
        if self.buffer is not None:
            self.buffer.destroy()
            self.buffer = None
