"""Use torch_npu's CANN operators with the A3 block-table layout used by vLLM Ascend."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import torch

from core.contracts.kernels import Implementation, KernelFn, Solution


class Operators:
    def __init__(self) -> None:
        self.ops = import_module("torch_npu")
        self.mask: torch.Tensor | None = None

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, blocks: torch.Tensor, ends: list[int],
                  lengths: list[int], scale: float, causal: bool) -> torch.Tensor:
        if causal and (self.mask is None or self.mask.device != q.device):
            self.mask = torch.ones((2048, 2048), dtype=torch.bool).triu_(1).to(q.device)
        output, _ = self.ops.npu_fused_infer_attention_score(
            query=q.contiguous(), key=k.flatten(2), value=v.flatten(2), input_layout="TND",
            block_table=blocks, block_size=k.shape[1], actual_seq_lengths=ends, actual_seq_lengths_kv=lengths,
            num_heads=q.shape[1], num_key_value_heads=k.shape[2], scale=scale, sparse_mode=3 if causal else 0,
            atten_mask=self.mask if causal else None)
        return output

    def store(self, k_cache: torch.Tensor, v_cache: torch.Tensor, slots: torch.Tensor,
              k: torch.Tensor, v: torch.Tensor) -> None:
        # The catalog registers this solution for the A3 page size.
        self.ops.npu_scatter_pa_kv_cache(key=k.contiguous(), value=v.contiguous(),
                                        key_cache=k_cache.unflatten(0, (-1, 128)),
                                        value_cache=v_cache.unflatten(0, (-1, 128)),
                                        slot_mapping=slots.contiguous(), cache_mode="Norm")

    def rms(self, x: torch.Tensor, w: torch.Tensor, eps: float, residual: torch.Tensor | None = None
            ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.ops.npu_rms_norm(x, w, eps)[0]
        normed, _, summed = self.ops.npu_add_rms_norm(x, residual, w, eps)
        return normed, summed


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        if not any("ascend" in device.capabilities for device in ctx.platform.devices()):
            return
        ops = Operators()

        def register(name: str, fn: KernelFn, shapes: dict[str, list[int]]) -> None:
            solution = Solution(f"ascend.{name}", name, f"{fn.__module__}:{fn.__name__}", library="torch_npu",
                                requires=["ascend"], shapes=shapes)
            ctx.register("kernels.catalog@1", Implementation(solution, fn), label=f"solution {solution.id}")

        layout = {"fp32": [0], "kv_as_q": [1], "page_size": [128], "head_dim": [64, 128, 256]}
        register("attn.block_paged@1", ops.attention, layout)
        register("kv.store@1", ops.store, layout)
        register("norm.rms@1", ops.rms, {"fp32": [0]})
        register("act.silu_and_mul@1", ops.ops.npu_swiglu, {"fp32": [0]})
