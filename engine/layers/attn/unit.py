# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""``layers.attn``: the full-attention shell with its backends (``layers.attn@1``). Adapted from
SGLang's ``RadixAttention``, ``TorchNativeAttnBackend`` (srt/layers/attention/
torch_native_backend.py, e838360; LICENSE.sglang) and, in ``paged``, ``FlashInferAttnBackend``.

The kernels come in two families, which differ in the metadata a step has to prepare for them,
so each has its backend, as each attention backend of SGLang has its ``init_forward_metadata``:

- ``attn.extend@1`` / ``attn.decode@1`` read ``req_to_token`` themselves (``TorchNativeAttnBackend``
  here): ``prepare`` derives the ``ForwardBatch`` fields they read (``req_pool_indices``,
  ``seq_lens``, ``extend_prefix_lens``, ``extend_seq_lens``, ``out_cache_loc``) from the persistent
  batch and the plan;
- ``attn.paged_extend@1`` / ``attn.paged_decode@1`` are planned per step over lists of token slots
  (``paged.PagedAttnBackend``).

``resolve`` fixes at start, per phase, the family and its solutions from ``attn.backend`` and its
per-phase overrides (SGLang's ``HybridAttnBackend`` over a prefill and a decode backend). ``auto``
is SGLang's default: the paged kernels where a library provides them (FlashInfer), else the
native ones. A mixed step runs as an extend, as SGLang's MIXED mode.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch

from core.contracts.plan import GRAPH, AttentionBackend, PersistentBatch, PlanKey
from core.contracts.records import BatchPlan
from core.contracts.state import StateDecl
from layers.attn.paged import DEFINITIONS as PAGED
from layers.attn.paged import KV_INDICES, PagedAttnBackend
from layers.solutions import PHASES, Kernels

KIND = "kv.paged"
NATIVE = MappingProxyType({"prefill": "attn.extend@1", "decode": "attn.decode@1"})


def on_device(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Host ``values`` on ``device`` without waiting for the work queued there: from pinned memory,
    ``non_blocking``, as SGLang's ``ForwardBatch`` copies its inputs. A blocking copy would wait
    for the step still running, and so would the next step's preparation."""
    if device.type != "cuda":
        return values.to(device)
    return values.pin_memory().to(device, non_blocking=True)


@dataclass(frozen=True)
class GraphBinding:
    """``Binding``: the fixed metadata buffers of the graph buckets; an eager step's are its own."""

    generation: int
    tensors: Mapping[str, torch.Tensor]

    def buffers(self) -> Mapping[str, torch.Tensor]:
        return self.tensors


@dataclass(frozen=True)
class Metadata:
    decode: bool
    req_to_token: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    extend_prefix_lens: torch.Tensor
    extend_seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor
    graph: bool = False
    """The metadata of a graph bucket: fixed buffers, filled before each replay."""


class TorchNativeAttnBackend:
    """One of the backends behind ``layers.attn@1``: the kernels that read ``req_to_token``."""

    def __init__(self, kernels: Kernels, kv: Any) -> None:
        self.kernels = kernels
        self.kv = kv
        self.forward_metadata: Metadata | None = None
        self.graph_metadata: dict[int, Metadata] = {}
        self.generation = 0
        self.graph_buffers: dict[str, torch.Tensor] = {}

    def release_graphs(self) -> None:
        self.graph_metadata.clear()
        self.graph_buffers.clear()
        self.forward_metadata = None
        self.generation += 1

    def prepare(self, batch: PersistentBatch, plan: BatchPlan, key: PlanKey) -> None:
        req_to_token = batch.binding.buffers()["req_to_token"]
        device = req_to_token.device
        if key.bucket:
            # init_forward_metadata_replay_cuda_graph: the bucket's fixed buffers, the rows padded
            # as SGLang pads them (PaddingPolicy.ZERO): a padded lane is the padding row, index 0,
            # with one token, so it reads and writes the padding slot only.
            size, n = key.bucket, len(plan.rows)
            m = self.graph_metadata.get(size)
            if m is None:
                none = torch.empty(0, dtype=torch.int64, device=device)
                m = Metadata(True, req_to_token, torch.zeros(size, dtype=torch.int64, device=device),
                             torch.ones(size, dtype=torch.int64, device=device), none, none,
                             torch.zeros(size, dtype=torch.int64, device=device), graph=True)
                self.graph_metadata[size] = m
                self.graph_buffers.update({f"{size}.req_pool_indices": m.req_pool_indices, f"{size}.seq_lens": m.seq_lens,
                                           f"{size}.out_cache_loc": m.out_cache_loc})
                self.generation += 1
            m.req_pool_indices[n:].zero_()
            m.seq_lens[n:].fill_(1)
            if n:
                m.req_pool_indices[:n].copy_(on_device(torch.tensor(list(plan.rows.values()), dtype=torch.int64) + 1,
                                                       device))
                m.seq_lens[:n].copy_(on_device(torch.tensor(list(plan.positions.values()), dtype=torch.int64) + 1,
                                              device))
            m.out_cache_loc.copy_(req_to_token[m.req_pool_indices, m.seq_lens - 1])
            self.forward_metadata = m
            return
        rows = torch.tensor(list(plan.rows.values()), dtype=torch.int64)
        prefix = torch.tensor(list(plan.positions.values()), dtype=torch.int64)
        extend = torch.tensor(list(plan.num_tokens.values()), dtype=torch.int64)
        req_pool_indices = rows + 1
        out_cache_loc = torch.cat([req_to_token[int(r), int(p):int(p + n)]
                                   for r, p, n in zip(req_pool_indices, prefix, extend, strict=True)])
        self.forward_metadata = Metadata(plan.phase == "decode" and not plan.draft_tokens, req_to_token,
                                         on_device(req_pool_indices, device),
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
        phase = GRAPH if m.graph else "decode" if m.decode else "prefill"
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


class AttnBackend:
    """``layers.attn@1``: each phase's backend."""

    def __init__(self, kernels: Kernels, kv: Any, dtype: str) -> None:
        self.kernels = kernels
        self.dtype = dtype
        """The activations' dtype (``model.dtype``): some kernels take bf16 / fp16 only, and some
        read the cache as if it had the queries' dtype."""
        self.native, self.paged = TorchNativeAttnBackend(kernels, kv), PagedAttnBackend(kernels, kv)
        self.backends: dict[str, TorchNativeAttnBackend | PagedAttnBackend] = {phase: self.native for phase in PHASES}
        self.active: TorchNativeAttnBackend | PagedAttnBackend = self.native
        """The backend of the last ``prepare``: the step's ``forward`` is its."""

    @property
    def binding(self) -> GraphBinding:
        return GraphBinding(self.native.generation + self.paged.generation,
                            {**self.native.graph_buffers, **self.paged.graph_buffers})

    def release_graphs(self) -> None:
        self.native.release_graphs()
        self.paged.release_graphs()

    def resolve(self, states: Sequence[StateDecl]) -> Mapping[str, Mapping[str, str]]:
        shape_k, dtype_k = next(d.tensors["k"] for d in states if d.kind == KIND)
        # kv_as_q: FlashInfer's 16-bit paths load the cache's bits as the queries' type, so a
        # cache of another dtype than the model's is not theirs (the native kernels convert).
        shape = {"head_dim": shape_k[-1], "fp32": int("f32" in (self.dtype, dtype_k)),
                 "kv_as_q": int(dtype_k == self.dtype)}
        calls: dict[str, dict[str, dict[str, int]]] = {}
        for phase in PHASES:
            paged = self._paged(phase, shape)
            self.backends[phase] = self.paged if paged else self.native
            calls[phase] = {"kv.store@1": {}, (PAGED if paged else NATIVE)[phase]: shape}
            if paged:
                calls[phase][KV_INDICES] = {}
        return self.kernels.resolve(calls, shared=frozenset({"kv.store@1", KV_INDICES}))

    def _paged(self, phase: str, shape: Mapping[str, int]) -> bool:
        """Whether ``phase`` runs the paged kernels. A named backend that has neither family fails in
        ``Kernels.resolve``, on the paged definition."""
        catalog, backend = self.kernels.catalog, self.kernels.backends[phase]
        if backend == "auto":
            return not catalog.select(PAGED[phase], shape).fallback
        try:
            catalog.select(NATIVE[phase], shape, backend)
        except LookupError:
            return True
        return False

    def prepare(self, batch: PersistentBatch, plan: BatchPlan, key: PlanKey) -> None:
        # A verify step (a decode plan with draft tokens, doc/specs/mtp.md) attends as an extend of each
        # row's D tokens, as SGLang's TARGET_VERIFY runs the prefill wrappers.
        decode = (key.bucket or plan.phase == "decode") and not plan.draft_tokens
        self.active = self.backends["decode" if decode else "prefill"]
        self.active.prepare(batch, plan, key)

    def forward(self, layer: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int,
                num_kv_heads: int, head_dim: int, scaling: float) -> torch.Tensor:
        return self.active.forward(layer, q, k, v, num_heads, num_kv_heads, head_dim, scaling)


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        kernels = Kernels(ctx.use("kernels.catalog@1"), config, "attn")
        ctx.provide("layers.attn@1", AttnBackend(kernels, ctx.use("state.kv.paged@1"), config["model.dtype"]))


if TYPE_CHECKING:
    _: type[AttentionBackend] = AttnBackend
