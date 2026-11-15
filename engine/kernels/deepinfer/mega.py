"""DeepInfer's fused kernels as the backend ``deepinfer``'s sublayer step, attention, MoE and Engram
(``ds.hc_step@1``, ``ds.attn_qkv@1``, ``ds.cache_attn_out@1``, ``ds.sparse_attn_out@1``, ``ds.moe@1``,
``ds.engram_fetch@1``, ``ds.engram@1``), as DeepInfer runs them
(``deepinfer/model_executor/models/v4_modules/{hc,attn,mlp,engram}.py``, a679bb8):

- ``mega_mhc``: the sublayer output written back to the Hyper-Connections stream, the next
  coefficients, this sublayer's mixed and normed input, and for the MoE its FP8 form written
  straight into the MoE buffer (or for Attention its ``(y_fp8, y_gemm_sf)`` handed to ``attn_qkv``);
- ``attn_qkv`` + ``{cache,sparse}_attn_out``: merged ``[wq_a; wkv]`` on the staged FP8 input,
  ``norm_forward_and_per_token_cast`` for ``q_norm``, merged ``[permute_q_b_proj(wq_b); indexer.wq_b]``,
  ``flash_mla.fused_norm_rope_attn_rope_cast.{decode,prefill}`` into ``deep_gemm.fp8_einsum`` (``wo_a``)
  and ``wo_b``;
- ``bf16_mega_gate`` and ``fp8_fp4_mega_moe``: the gate's choice and slots into ``SymmBuffer``, then
  dispatch, grouped GEMMs + SwiGLU, shared expert and combine in one kernel across the ranks;
- ``deep_ep.EngramBuffer`` + ``engram``: one-sided RDMA/NVLink FP8 + scale fetch (capturable inside
  CUDA graphs, overlapped with layer 0) fed directly into ``fp8_fp4_gemm_nt`` and ``engram_gate_fwd``.

Both collective buffers (``SymmBuffer`` and ``EngramBuffer``) belong to this unit's instance and are
rebuilt cleanly if a hot-applied config changes expert slot counts or table shapes, and destroyed in
``close()``.
"""
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import os
import sys
import weakref
from collections.abc import Callable, Sequence
from typing import Any

import deep_gemm
import tile_kernels
import torch
import torch.distributed as dist
import torch.nn.functional as F
from flash_mla import fused_norm_rope_attn_rope_cast

from kernels import ds1017_formats as formats
from kernels.deepinfer import ops
from kernels.ds1017_formats import BLOCK, packed_words

BF16 = torch.bfloat16
E4M3 = torch.float8_e4m3fn
POST_SCALE = 2.0
"""``hc_post_mult_value``: what the post coefficients are scaled by (``ops.hc_mixes`` passes the same)."""


def _base(t: torch.Tensor) -> torch.Tensor:
    return t._base if t._base is not None else t


def _listed_2d(idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    listed = F.pad(idx.to(torch.int32), (0, -idx.shape[1] % ops.KEY_BLOCK), value=-1)
    return listed.contiguous(), (idx >= 0).sum(-1).to(torch.int32)


class Alone:
    """What ``SymmBuffer`` asks of a group, for one rank: its buffer is plain device memory."""

    def size(self) -> int:
        return 1

    def rank(self) -> int:
        return 0

    def barrier(self) -> None:
        return None


class Mega:
    """This rank's part of the ranks' shared MoE and Engram buffers, and the fused solutions over them."""

    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        """Tokens a rank brings to one step at most (a prefill's budget and the decoding rows)."""
        self.buffer: Any = None
        self.made_for: tuple[int, ...] = ()
        self.staged = -1
        """How many tokens' MoE input the last ``hc_step`` wrote into the buffer; -1: none."""
        self.attn_input: tuple[torch.Tensor, torch.Tensor] | None = None
        """FP8 input ``(y_fp8, y_gemm_sf)`` staged by ``hc_step(..., 'attn')`` for ``attn_qkv``."""
        self.engram_buf: Any = None
        self.engram_shape: tuple[Any, ...] = ()
        self.engram_versions: tuple[Any, ...] = ()
        self.engram_hooks: list[Any] = []
        self._cache: dict[tuple[Any, ...], tuple[tuple[int, ...], Any]] = {}

    def _cached(self, tag: str, bases: Sequence[torch.Tensor], extra: tuple[Any, ...],
                build: Callable[[], Any]) -> Any:
        roots = [_base(t) for t in bases]
        key = (tag, *(r.data_ptr() for r in roots), *extra)
        versions = tuple(r._version for r in roots)
        hit = self._cache.get(key)
        if hit is not None and hit[0] == versions:
            return hit[1]
        with torch.inference_mode(False):
            value = build()
        self._cache[key] = (versions, value)
        weakref.finalize(roots[0], self._cache.pop, key, None)
        return value

    def symm(self, group: Any, experts: int, topk: int, hidden: int, inter: int) -> Any:
        made_for = (group.size, experts, topk, hidden, inter)
        if self.buffer is not None and made_for != self.made_for:
            torch.cuda.synchronize()
            self.buffer.destroy()
            self.buffer = None
            self.made_for = ()
        if self.buffer is None:
            if group.size > 1 and group.device_group is None:
                raise ValueError("the fused MoE kernel shares memory between the ranks' cards: comm.device_backend = nccl")
            # DeepGEMM finds which SMs sit together once, and waits for the device doing it: here, not
            # in a step that is being recorded.
            deep_gemm.locality_domain.get_balanced_sm_locality_domains()
            with torch.inference_mode(False):
                self.buffer = deep_gemm.SymmBuffer(
                    group.device_group if group.size > 1 else Alone(), num_experts=experts,
                    num_max_tokens_per_rank=self.max_tokens, num_topk=topk, hidden=hidden,
                    intermediate_hidden=inter, num_shared_experts=1, mma_type="fp8xfp4", activation="swiglu")
            self.made_for = made_for
        return self.buffer

    def hc_step(self, x: torch.Tensor | None, residual: torch.Tensor, pre: torch.Tensor, post: torch.Tensor | None,
                comb: torch.Tensor | None, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor,
                norm_weight: torch.Tensor, norm_eps: float, iters: int, hc_eps: float, stage: Any = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``ds.hc_step@1`` as ``HCLayerMixin._mega_mhc``: one ``deep_gemm.mega_mhc`` writes the sublayer
        output back, computes the next coefficients (Sinkhorn included), mixes and norms this
        sublayer's input, and quantizes it straight to FP8: into ``SymmBuffer`` when ``stage`` is the
        MoE tuple ``(group, slots, topk, inter)``, or into ``self.attn_input`` when ``stage == 'attn'``.
        Where nothing waits to be written back (layer 0, or right after an Engram layer), ``ops.hc_step``
        runs and ``stage == 'attn'`` quantizes ``y`` once via ``ops._cast``."""
        self.staged = -1
        self.attn_input = None
        if x is None or post is None or comb is None:
            if stage is not None and stage != "attn":
                raise ValueError("the MoE's input is staged by the fused kernel, which needs the sublayer before it")
            y, stream, new_pre, new_post, new_comb = ops.hc_step(
                x, residual, pre, post, comb, fn, scale, base, norm_weight, norm_eps, iters, hc_eps)
            if stage == "attn" and y.shape[0]:
                self.attn_input = ops._cast(ops._rows(y))
            return y, stream, new_pre, new_post, new_comb
        tokens, hidden = x.shape
        if not tokens:
            return ops.hc_step(x, residual, pre, post, comb, fn, scale, base, norm_weight, norm_eps, iters, hc_eps)
        hc, device = residual.shape[1], x.device
        stream = torch.empty(tokens, hc, hidden, dtype=BF16, device=device)
        new_pre = torch.empty(tokens, hc, 1, dtype=torch.float32, device=device)
        new_post = torch.empty(tokens, hc, 1, dtype=torch.float32, device=device)
        new_comb = torch.empty(tokens, hc, hc, dtype=torch.float32, device=device)
        y = torch.empty(tokens, hidden, dtype=BF16, device=device)
        y_fp8 = gemm_sf = routed = shared = None
        block_m = 0
        if stage == "attn":
            y_fp8 = torch.empty(tokens, hidden, dtype=E4M3, device=device)
            m_pad = deep_gemm.get_tma_aligned_size(tokens, 4)
            gemm_sf = torch.empty(hidden // 128, m_pad, dtype=torch.int32, device=device).mT[:tokens]
        elif stage is not None:
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
            y_fp8=y_fp8, y_gemm_sf=gemm_sf, y_routed_sf=routed, y_shared_sf=shared, shared_sf_block_m=block_m)
        if stage == "attn":
            assert y_fp8 is not None and gemm_sf is not None
            self.attn_input = (y_fp8, gemm_sf)
        elif stage is not None:
            self.staged = tokens
        return y, stream, new_pre.view(tokens, hc), new_post.view(tokens, hc), new_comb

    # ---------------------------------------------------------------- fused attention

    def _qkv_weights(self, wq_a: torch.Tensor, pq_a: torch.Tensor, wkv: torch.Tensor, pkv: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        def build() -> tuple[torch.Tensor, torch.Tensor]:
            w = torch.cat([wq_a.detach().clone(), wkv.detach().clone()], dim=0).contiguous()
            p = torch.cat([pq_a.detach().clone(), pkv.detach().clone()], dim=1).contiguous()
            return w.view(E4M3), packed_words(p, w.shape[0])

        return self._cached("qkv", (wq_a, pq_a, wkv, pkv), (), build)

    def _qb_weights(self, wq_b: torch.Tensor, pq_b: torch.Tensor, iwq_b: torch.Tensor | None,
                    ipq_b: torch.Tensor | None, heads: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        bases: tuple[torch.Tensor, ...] = (wq_b, pq_b) if iwq_b is None or ipq_b is None else (wq_b, pq_b, iwq_b, ipq_b)

        def build() -> tuple[torch.Tensor, torch.Tensor]:
            wb = wq_b.detach().clone().view(E4M3)
            sb = packed_words(pq_b.detach().clone(), wb.shape[0])
            wb_p, sb_p = fused_norm_rope_attn_rope_cast.permute_q_b_proj((wb, sb), heads, head_dim)
            if iwq_b is None or ipq_b is None:
                return wb_p, sb_p
            iw = iwq_b.detach().clone().view(E4M3)
            isf = packed_words(ipq_b.detach().clone(), iw.shape[0])
            w = torch.cat([wb_p, iw], dim=0).contiguous()
            sf = torch.cat([sb_p.mT, isf.mT], dim=1).contiguous().mT
            return w, sf

        return self._cached("qb", bases, (heads, head_dim), build)

    def _wv_weights(self, wo_a: torch.Tensor, groups: int, heads: int, head_dim: int
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        def build() -> tuple[torch.Tensor, torch.Tensor]:
            w2d = wo_a.detach().clone().reshape(wo_a.shape[0], -1)
            o_lora = w2d.shape[0] // groups
            group_in = w2d.shape[1]
            qb, qs = formats.quantize_fp8(w2d)
            wv_w = qb.view(E4M3).view(groups, o_lora, group_in)
            table = formats._table("e8m0", w2d.device).detach().clone()
            sf_f32 = (table[qs.to(torch.int32)]
                      .view(groups, o_lora // BLOCK, group_in // BLOCK)
                      .repeat_interleave(BLOCK, dim=1))
            wv_sf = deep_gemm.transform_sf_into_required_layout(
                sf_f32, o_lora, group_in, num_groups=groups, recipe=(1, 1, BLOCK), is_sfa=False)
            return fused_norm_rope_attn_rope_cast.permute_wv_proj(
                (wv_w, wv_sf), wv_group_size=heads // groups, d_o=head_dim)

        return self._cached("wv", (wo_a,), (groups, heads, head_dim), build)

    def attn_qkv(self, x: torch.Tensor, wq_a: torch.Tensor, sq_a: torch.Tensor, pq_a: torch.Tensor,
                 wkv: torch.Tensor, skv: torch.Tensor, pkv: torch.Tensor, q_norm: torch.Tensor, eps: float,
                 wq_b: torch.Tensor, sq_b: torch.Tensor, pq_b: torch.Tensor, iwq_b: torch.Tensor | None,
                 isq_b: torch.Tensor | None, ipq_b: torch.Tensor | None, heads: int, head_dim: int
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """``ds.attn_qkv@1`` as ``DeepseekV41Attention``: merged ``[wq_a; wkv]`` on the staged FP8 input
        from ``hc_step``, ``norm_forward_and_per_token_cast`` for ``q_norm``, and merged
        ``[permute_q_b_proj(wq_b); indexer.wq_b]``."""
        rows = ops._rows(x)
        tokens, device = rows.shape[0], rows.device
        q_lora, kv_dim = wq_a.shape[0], wkv.shape[0]
        if not tokens:
            self.attn_input = None
            iq_empty = None if iwq_b is None else torch.empty(0, iwq_b.shape[0], dtype=BF16, device=device)
            return (torch.empty(0, heads, head_dim, dtype=BF16, device=device),
                    torch.empty(0, kv_dim, dtype=BF16, device=device), iq_empty)
        staged, self.attn_input = self.attn_input, None
        x_fp8 = staged if staged is not None and staged[0].shape == rows.shape else ops._cast(rows)
        w_qkv = self._qkv_weights(wq_a, pq_a, wkv, pkv)
        qkv = ops._out(tokens, q_lora + kv_dim, BF16, device)
        deep_gemm.fp8_fp4_gemm_nt(x_fp8, w_qkv, qkv, None, recipe=(1, 1, BLOCK))
        qr_fp8 = tile_kernels.quant.norm_forward_and_per_token_cast(
            qkv[:, :q_lora].contiguous(), q_norm.to(BF16), eps, fmt="e4m3", num_per_channels=BLOCK,
            round_sf=True, use_packed_ue8m0=True, use_tma_aligned_col_major_sf=True)[0]
        w_qb = self._qb_weights(wq_b, pq_b, iwq_b, ipq_b, heads, head_dim)
        q_cols = heads * head_dim
        extra_cols = 0 if iwq_b is None else iwq_b.shape[0]
        qb_out = ops._out(tokens, q_cols + extra_cols, BF16, device)
        deep_gemm.fp8_fp4_gemm_nt(qr_fp8, w_qb, qb_out, None, recipe=(1, 1, BLOCK))
        q = qb_out[:, :q_cols].unflatten(-1, (heads, head_dim))
        iq = None if iwq_b is None else qb_out[:, q_cols:]
        return q, qkv[:, q_lora:], iq

    def _wo(self, ctx_fp8: torch.Tensor, ctx_sf: torch.Tensor, groups: int, heads: int, head_dim: int,
            wo_a: torch.Tensor, wo_b: torch.Tensor, po_b: torch.Tensor) -> torch.Tensor:
        tokens, device = ctx_fp8.shape[0], ctx_fp8.device
        o_lora = wo_a.shape[0] // groups
        hidden = wo_b.shape[0]
        wv = self._wv_weights(wo_a, groups, heads, head_dim)
        wv_fp8 = torch.empty((tokens, groups, o_lora), dtype=E4M3, device=device)
        wv_sf = torch.empty_strided((tokens, (groups * o_lora) // 128),
                                    (1, deep_gemm.get_tma_aligned_size(tokens, 4)),
                                    dtype=torch.int32, device=device)
        deep_gemm.fp8_einsum("bhr,hdr->bhd", (ctx_fp8, ctx_sf), wv, (wv_fp8, wv_sf), recipe=(1, 1, BLOCK))
        out = ops._out(tokens, hidden, BF16, device)
        deep_gemm.fp8_fp4_gemm_nt((wv_fp8.flatten(1, 2), wv_sf),
                                  (wo_b.view(E4M3), packed_words(po_b, hidden)), out, None, recipe=(1, 1, BLOCK))
        return out

    def cache_attn_out(self, q: torch.Tensor, cos_sin: torch.Tensor, positions: torch.Tensor, rotary: int,
                       window: torch.Tensor, window_idx: torch.Tensor, latents: torch.Tensor | None,
                       latent_idx: torch.Tensor | None, sink: torch.Tensor, scale: float, groups: int,
                       wo_a: torch.Tensor, wo_b: torch.Tensor, so_b: torch.Tensor, po_b: torch.Tensor
                       ) -> torch.Tensor:
        """``ds.cache_attn_out@1``: ``fused_norm_rope_attn_rope_cast.decode`` + ``fp8_einsum`` + ``wo_b``."""
        tokens, heads, head_dim = q.shape
        if not tokens:
            return torch.empty(0, wo_b.shape[0], dtype=BF16, device=q.device)
        main, count = _listed_2d(window_idx.flip(-1))
        extra = extra_idx = extra_count = None
        if latents is not None and latent_idx is not None:
            extra = latents.unsqueeze(2)
            extra_idx, extra_count = _listed_2d(latent_idx)
        ctx_fp8, ctx_sf, _ = fused_norm_rope_attn_rope_cast.decode(
            q=q.contiguous(), k_cache=window.unsqueeze(2), indices_in_kvcache=main, sm_scale=scale, d_v=head_dim,
            attn_sink=sink, topk_length=count, extra_k_cache=extra, extra_indices_in_kvcache=extra_idx,
            extra_topk_length=extra_count, enable_q_norm=False, rms_norm_eps=0.0,
            token_positions=positions.to(torch.int32).contiguous(), is_rope_neox_style=False, rope_dim=rotary,
            cos_sin_cache=cos_sin, n_wv_group=groups, num_per_channels=BLOCK,
            use_tma_aligned_col_major_sf=True, round_sf=True, use_packed_ue8m0=True)
        return self._wo(ctx_fp8, ctx_sf, groups, heads, head_dim, wo_a, wo_b, po_b)

    def sparse_attn_out(self, q: torch.Tensor, cos_sin: torch.Tensor, positions: torch.Tensor, rotary: int,
                        source: torch.Tensor, idx: torch.Tensor, sink: torch.Tensor, scale: float, groups: int,
                        wo_a: torch.Tensor, wo_b: torch.Tensor, so_b: torch.Tensor, po_b: torch.Tensor
                        ) -> torch.Tensor:
        """``ds.sparse_attn_out@1``: ``fused_norm_rope_attn_rope_cast.prefill`` + ``fp8_einsum`` + ``wo_b``."""
        tokens, heads, head_dim = q.shape
        if not tokens:
            return torch.empty(0, wo_b.shape[0], dtype=BF16, device=q.device)
        listed = F.pad(idx.to(torch.int32), (0, -idx.shape[1] % ops.KEY_BLOCK), value=-1)
        ctx_fp8, ctx_sf, _, _ = fused_norm_rope_attn_rope_cast.prefill(
            q=q.contiguous(), kv=source.unsqueeze(1), indices=listed.unsqueeze(1), sm_scale=scale, attn_sink=sink,
            topk_length=None, enable_q_norm=False, rms_norm_eps=0.0,
            token_positions=positions.to(torch.int32).contiguous(), is_rope_neox_style=False, rope_dim=rotary,
            cos_sin_cache=cos_sin, n_wv_group=groups, num_per_channels=BLOCK,
            use_tma_aligned_col_major_sf=True, round_sf=True, use_packed_ue8m0=True)
        return self._wo(ctx_fp8, ctx_sf, groups, heads, head_dim, wo_a, wo_b, po_b)

    # ---------------------------------------------------------------- fused MoE

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

    # ---------------------------------------------------------------- fused Engram

    @staticmethod
    def _deep_ep() -> Any:
        # Once worker.comm has built an NCCL communicator on InfiniBand/RoCE, /proc/self/maps lists both
        # libnccl.so.2 and NCCL's own RDMA plugin libnccl-net.so, which trips deep_ep 2.5's substring
        # check ('libnccl' in line) when deep_ep is imported later (including on hot-apply). Keep the
        # duplicate-runtime guard on actual libnccl.so* files while ignoring libnccl-net.so*.
        if "deep_ep" not in sys.modules and "EP_SUPPRESS_NCCL_CHECK" not in os.environ:
            try:
                with open("/proc/self/maps") as f:
                    nccl_sos = {
                        p for line in f
                        if os.path.basename(p := line.strip().split(" ")[-1]).startswith("libnccl.so")
                    }
            except OSError:
                nccl_sos = set()
            if len(nccl_sos) <= 1:
                os.environ["EP_SUPPRESS_NCCL_CHECK"] = "1"
        import deep_ep
        return deep_ep

    def _engram_buf(self, group: Any, cols: int, weights: Sequence[torch.Tensor],
                    scales: Sequence[torch.Tensor]) -> Any:
        deep_ep = self._deep_ep()

        if group.size <= 1 or group.device_group is None:
            raise ValueError("EngramBuffer shares tables across ranks: comm.size > 1 and comm.device_backend = nccl")
        entries = tuple(w.shape[0] for w in weights)
        hidden = weights[0].shape[1]
        shape_key = (group.size, entries, hidden, cols, self.max_tokens)
        ver_key = tuple((_base(w).data_ptr(), _base(w)._version, _base(s).data_ptr(), _base(s)._version)
                        for w, s in zip(weights, scales, strict=True))
        if self.engram_buf is not None and shape_key != self.engram_shape:
            torch.cuda.synchronize(weights[0].device)
            self.engram_buf.destroy()
            self.engram_buf = None
            self.engram_shape = ()
            self.engram_versions = ()
        pg = group.device_group
        if self.engram_buf is None:
            torch.cuda.synchronize(weights[0].device)
            sf_packs = (hidden // BLOCK) // 4
            entries_list = list(entries)
            with torch.inference_mode(False):
                gpu_bytes, rdma_bytes = deep_ep.EngramBuffer.get_storage_size_hint(
                    pg, entries_list, hidden, self.max_tokens, cols, E4M3, sf_packs)
                num_qps, qp_depth, restrict_rd = deep_ep.EngramBuffer.get_theoretical_config(
                    pg, len(weights), self.max_tokens, cols)
                buf = deep_ep.EngramBuffer(
                    group=pg, num_gpu_bytes=gpu_bytes, num_rdma_storage_bytes=rdma_bytes,
                    use_cpu_rdma_storage=True, num_allocated_qps=num_qps, qp_depth=qp_depth,
                    restrict_rd_atomic=restrict_rd, allow_hybrid_mode=True, explicitly_destroy=True)
                buf.set_config(entries_list, hidden, self.max_tokens, cols, E4M3, sf_packs)
            self.engram_buf = buf
            self.engram_shape = shape_key
            self.engram_versions = ()
        if ver_key != self.engram_versions:
            torch.cuda.synchronize(weights[0].device)
            with torch.inference_mode(False):
                sfs: list[torch.Tensor] = []
                for s in scales:
                    local_sf = s.detach().contiguous().view(torch.int32)
                    full_sf = local_sf.new_empty(group.size * local_sf.shape[0], local_sf.shape[1])
                    dist.all_gather_into_tensor(full_sf, local_sf, group=pg)
                    sfs.append(full_sf)
                self.engram_buf.write([w.detach().contiguous().view(E4M3) for w in weights], sfs=sfs)
            torch.cuda.synchronize(weights[0].device)
            self.engram_versions = ver_key
        return self.engram_buf

    def engram_fetch(self, group: Any, ids: torch.Tensor, weights: Sequence[torch.Tensor],
                     scales: Sequence[torch.Tensor], stage: bool = False) -> Any:
        """``ds.engram_fetch@1`` via ``deep_ep.EngramBuffer``: one-sided NVLink/RDMA read capturable
        inside CUDA graphs. With ``stage=True``, keeps the layer hooks on ``self.engram_hooks`` so
        layer 0 overlaps with the fetch and ``Mega.engram`` consumes the FP8 rows directly."""
        cols, hidden = ids.shape[2], weights[0].shape[1]
        buf = self._engram_buf(group, cols, weights, scales)
        n = ids.shape[0]
        if n > self.max_tokens:
            raise ValueError(f"{n} tokens in a step; EngramBuffer was sized for {self.max_tokens} "
                             "(sched.token_budget + batch.max_rows)")
        if not n:
            self.engram_hooks = []
            if stage:
                return None, None
            return (torch.empty(0, len(weights), cols, hidden, dtype=torch.uint8, device=ids.device),
                    torch.empty(0, len(weights), cols, hidden // BLOCK, dtype=torch.uint8, device=ids.device))
        hooks = buf.fetch(ids.to(torch.int32).permute(1, 0, 2).contiguous(), use_tma_aligned_col_major_sf=True)
        if stage:
            self.engram_hooks = list(hooks)
            return None, None
        pairs = [h() for h in hooks]
        held = torch.stack([p[0].view(torch.uint8).view(n, cols, hidden) for p in pairs], dim=1)
        sc = torch.stack([p[1].contiguous().view(torch.uint8).view(n, cols, hidden // BLOCK) for p in pairs], dim=1)
        return held, sc

    def _engram_weight(self, q_weight: torch.Tensor, k_weight: torch.Tensor) -> torch.Tensor:
        return self._cached("eg", (q_weight, k_weight), (),
                            lambda: tile_kernels.engram.fused_weight(q_weight.detach().clone(),
                                                                     k_weight.detach().clone()))

    def engram(self, index: int, x: torch.Tensor, held: torch.Tensor | None, scales: torch.Tensor | None,
               ids: torch.Tensor | None, embed_weight: torch.Tensor, embed_scale: torch.Tensor,
               wkv_w: torch.Tensor, wkv_s: torch.Tensor, wkv_p: torch.Tensor,
               q_weight: torch.Tensor, k_weight: torch.Tensor, eps: float) -> torch.Tensor:
        """``ds.engram@1``: direct ``fp8_fp4_gemm_nt`` on the FP8 rows (from ``engram_hooks[index]`` or
        local FP8 tables) + cached ``fused_weight`` + ``engram_gate_fwd``."""
        tokens, hc, dim = x.shape
        if not tokens:
            if index + 1 >= len(self.engram_hooks):
                self.engram_hooks = []
            return x.clone()
        if held is None and scales is None and ids is None:
            act = self.engram_hooks[index]()
            if index + 1 >= len(self.engram_hooks):
                self.engram_hooks = []
        else:
            if held is None or scales is None:
                assert ids is not None
                held, scales = embed_weight[ids], embed_scale[ids]
            a_fp8 = held.flatten(1).contiguous().view(E4M3)
            sf_rm = scales.flatten(1).contiguous().view(torch.int32)
            m_pad = deep_gemm.get_tma_aligned_size(tokens, 4)
            a_sf = torch.empty(sf_rm.shape[1], m_pad, dtype=torch.int32, device=x.device)
            a_sf[:, :tokens].copy_(sf_rm.T)
            act = (a_fp8, a_sf.mT[:tokens])
        out_cols = wkv_w.shape[0]
        kv = ops._out(tokens, out_cols, BF16, x.device)
        deep_gemm.fp8_fp4_gemm_nt(act, (wkv_w.view(E4M3), packed_words(wkv_p, out_cols)), kv, None,
                                  recipe=(1, 1, BLOCK))
        fused = self._engram_weight(q_weight, k_weight)
        return tile_kernels.engram.engram_gate_fwd(x.contiguous(), kv.contiguous().view(tokens, hc + 1, dim), fused,
                                                   eps, 1e-6, save_for_backward=False)[0]

    def close(self) -> None:
        self.attn_input = None
        self.engram_hooks = []
        self._cache.clear()
        if self.engram_buf is not None:
            self.engram_buf.destroy()
            self.engram_buf = None
            self.engram_shape = ()
            self.engram_versions = ()
        if self.buffer is not None:
            self.buffer.destroy()
            self.buffer = None
            self.made_for = ()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
