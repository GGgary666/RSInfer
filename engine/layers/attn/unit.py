# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""``layers.attn``: the full-attention shell with its backend (``layers.attn@1``). Adapted from
SGLang's ``RadixAttention`` and ``TorchNativeAttnBackend`` (srt/layers/attention/
torch_native_backend.py, e838360; LICENSE.sglang). ``prepare`` derives the ``ForwardBatch`` fields
the backend reads (``req_pool_indices``, ``seq_lens``, ``extend_prefix_lens``, ``extend_seq_lens``,
``out_cache_loc``) from the persistent batch and the plan; ``forward`` stores K/V and runs
``attn.extend@1`` (prefill and mixed steps, as SGLang's MIXED mode) or ``attn.decode@1``, with the
solutions ``resolve`` fixed at start from ``attn.backend`` and its per-phase overrides.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from core.contracts.plan import AttentionBackend, PersistentBatch, PlanKey
from core.contracts.records import BatchPlan
from core.contracts.state import StateDecl
from layers.solutions import Kernels

KIND = "kv.paged"


def on_device(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Host ``values`` on ``device`` without waiting for the work queued there: from pinned memory,
    ``non_blocking``, as SGLang's ``ForwardBatch`` copies its inputs. A blocking copy would wait
    for the step still running, and so would the next step's preparation."""
    if device.type != "cuda":
        return values.to(device)
    return values.pin_memory().to(device, non_blocking=True)


@dataclass(frozen=True)
class NoBuffers:
    """``Binding``: eager attention keeps no fixed buffers."""

    generation: int = 0

    def buffers(self) -> Mapping[str, torch.Tensor]:
        return {}


@dataclass(frozen=True)
class Metadata:
    decode: bool
    req_to_token: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    extend_prefix_lens: torch.Tensor
    extend_seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor


class TorchNativeAttnBackend:
    """``layers.attn@1``."""

    def __init__(self, kernels: Kernels, kv: Any) -> None:
        self.kernels = kernels
        self.kv = kv
        self.forward_metadata: Metadata | None = None

    @property
    def binding(self) -> NoBuffers:
        return NoBuffers()

    def resolve(self, states: Sequence[StateDecl]) -> Mapping[str, Mapping[str, str]]:
        head_dim = next(d.tensors["k"][0][-1] for d in states if d.kind == KIND)
        shape = {"head_dim": head_dim}
        return self.kernels.resolve({"prefill": {"kv.store@1": {}, "attn.extend@1": shape},
                                     "decode": {"kv.store@1": {}, "attn.decode@1": shape}})

    def prepare(self, batch: PersistentBatch, plan: BatchPlan, key: PlanKey) -> None:
        req_to_token = batch.binding.buffers()["req_to_token"]
        device = req_to_token.device
        rows = torch.tensor(list(plan.rows.values()), dtype=torch.int64)
        prefix = torch.tensor(list(plan.positions.values()), dtype=torch.int64)
        extend = torch.tensor(list(plan.num_tokens.values()), dtype=torch.int64)
        req_pool_indices = rows + 1
        out_cache_loc = torch.cat([req_to_token[int(r), int(p):int(p + n)]
                                   for r, p, n in zip(req_pool_indices, prefix, extend, strict=True)])
        self.forward_metadata = Metadata(plan.phase == "decode", req_to_token, on_device(req_pool_indices, device),
                                         on_device(prefix + extend, device), on_device(prefix, device),
                                         on_device(extend, device), out_cache_loc.to(torch.int64))

    def forward(self, layer: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int,
                num_kv_heads: int, head_dim: int, scaling: float) -> torch.Tensor:
        """``forward_extend`` / ``forward_decode``: ``q`` ``[tokens, heads * head_dim]`` in, the
        same shape out."""
        m = self.forward_metadata
        if m is None:
            raise RuntimeError("layers.attn ran before prepare")
        q = q.reshape(-1, num_heads * head_dim)
        o = torch.empty_like(q)
        cache = self.kv.bind(layer).buffers()
        k_cache, v_cache = cache["k"], cache["v"]
        phase = "decode" if m.decode else "prefill"
        self.kernels(phase, "kv.store@1")(k_cache, v_cache, m.out_cache_loc, k.view(-1, num_kv_heads, head_dim),
                                          v.view(-1, num_kv_heads, head_dim))
        use_gqa = num_heads != num_kv_heads
        q_ = q.view(-1, num_heads, head_dim)
        o_ = o.view(-1, num_heads, head_dim)
        if m.decode:
            self.kernels(phase, "attn.decode@1")(q_, o_, k_cache, v_cache, m.req_to_token, m.req_pool_indices,
                                                 m.seq_lens, scaling, use_gqa, False)
        else:
            self.kernels(phase, "attn.extend@1")(q_, o_, k_cache, v_cache, m.req_to_token, m.req_pool_indices,
                                                 m.seq_lens, m.extend_prefix_lens, m.extend_seq_lens, scaling, use_gqa,
                                                 True)
        return o


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        kernels = Kernels(ctx.use("kernels.catalog@1"), config, "attn")
        ctx.provide("layers.attn@1", TorchNativeAttnBackend(kernels, ctx.use("state.kv.paged@1")))


if TYPE_CHECKING:
    _: type[AttentionBackend] = TorchNativeAttnBackend
