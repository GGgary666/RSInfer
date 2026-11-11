# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""The backend of the paged attention kernels (``attn.paged_extend@1``, ``attn.paged_decode@1``):
the plain paths of SGLang's ``FlashInferAttnBackend`` and its indices updaters (srt/layers/
attention/flashinfer_backend.py, e838360; LICENSE.sglang), without speculation, sliding windows,
cross attention, multi-item scoring or quantized caches.

``prepare`` lays each row's token slots out as one list (``kv_indptr``, ``kv_indices``: SGLang's
page size of one token over ``req_to_token``) and has the step's operator plan over it, from the
host: the lengths come with the step's plan, so nothing waits for the device. ``forward`` stores
the step's K/V and runs the operator. An extend runs over each row's whole sequence, its new
tokens stored first (SGLang's extend without the ragged wrapper).

An operator is made by the solution fixed for its phase, one for the eager steps of a phase and
one per graph bucket over the bucket's fixed buffers. It is made at the first ``forward`` that
needs it, which is where the head counts are known; that ``forward`` also plans it. A graph
bucket's first ``forward`` is the eager one before its capture, so a captured ``forward`` only
runs, and every later replay is planned in ``prepare``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from core.contracts.plan import GRAPH, PersistentBatch, PlanKey
from core.contracts.records import BatchPlan
from layers.solutions import Kernels

DEFINITIONS = MappingProxyType({"prefill": "attn.paged_extend@1", "decode": "attn.paged_decode@1"})
KV_INDICES = "attn.kv_indices@1"
"""What ``prepare`` lays the slot lists out with, outside the step's graph (``unit`` resolves it
with the paged definitions)."""


def on_device(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Host ``values`` on ``device`` without waiting for the work queued there (``layers.attn``)."""
    if device.type != "cuda":
        return values.to(device)
    return values.pin_memory().to(device, non_blocking=True)


def indptr_of(lengths: list[int]) -> torch.Tensor:
    """``[0, l0, l0 + l1, ...]``, int32 on the host."""
    out = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    out[1:] = torch.cumsum(torch.tensor(lengths, dtype=torch.int32), dim=0)
    return out


@dataclass
class Metadata:
    table: str
    """``prefill``, ``decode`` or ``GRAPH``: whose solutions ``forward`` calls."""
    bucket: int
    qo_indptr_cpu: torch.Tensor
    kv_indptr_cpu: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    out_cache_loc: torch.Tensor
    planned: bool = False
    # A graph bucket's fixed buffers, filled before each replay.
    req_pool_indices: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    kv_last_page_len: torch.Tensor | None = None


class PagedAttnBackend:
    """One of the backends behind ``layers.attn@1`` (``unit``)."""

    def __init__(self, kernels: Kernels, kv: Any) -> None:
        self.kernels = kernels
        self.kv = kv
        self.forward_metadata: Metadata | None = None
        self.graph_metadata: dict[int, Metadata] = {}
        self.generation = 0
        self.graph_buffers: dict[str, torch.Tensor] = {}
        self.kv_indices_buffer: torch.Tensor | None = None
        """One for all buckets, as SGLang's ``cuda_graph_kv_indices``: a bucket of rows times the
        width of ``req_to_token``."""
        self.operators: dict[tuple[str, int], tuple[Any, tuple[Any, ...]]] = {}
        """Per (table, bucket): the operator, and the head counts and dtypes it was made for."""
        self.operators_version = -1
        """The kernel resolution the operators were made by (``Kernels.version``)."""

    def release_graphs(self) -> None:
        self.graph_metadata.clear()
        self.graph_buffers.clear()
        self.kv_indices_buffer = None
        self.operators = {key: op for key, op in self.operators.items() if key[0] != GRAPH}
        self.forward_metadata = None
        self.generation += 1

    def operator(self, m: Metadata) -> tuple[Any, tuple[Any, ...]] | None:
        """The operator of ``m``'s steps if it was made, by the solutions as they are now."""
        if self.operators and self.kernels.catalog.version != self.operators_version:
            self.operators.clear()  # a kernel unit was reloaded: its operators are the old code's
        return self.operators.get((m.table, m.bucket))

    def prepare(self, batch: PersistentBatch, plan: BatchPlan, key: PlanKey) -> None:
        req_to_token = batch.binding.buffers()["req_to_token"]
        device = req_to_token.device
        rows = [int(r) + 1 for r in plan.rows.values()]
        if key.bucket:
            m = self._bucket(key.bucket, req_to_token)
            assert m.req_pool_indices is not None and m.seq_lens is not None
            # init_forward_metadata_replay_cuda_graph: a padded lane is the padding row, index 0,
            # with one token, so it reads and writes the padding slot only.
            pad = key.bucket - len(rows)
            lengths = [int(p) + 1 for p in plan.positions.values()] + [1] * pad
            m.req_pool_indices.copy_(on_device(torch.tensor(rows + [0] * pad, dtype=torch.int64), device))
            m.kv_indptr_cpu = indptr_of(lengths)
            m.kv_indptr.copy_(on_device(m.kv_indptr_cpu, device))
            m.seq_lens.copy_(on_device(torch.tensor(lengths, dtype=torch.int64), device))
            # The slot lists and each lane's slot to store at, one kernel (SGLang's
            # create_flashinfer_kv_indices_triton, with the last slot of each row added).
            self.kernels(m.table, KV_INDICES)(req_to_token, m.req_pool_indices, m.seq_lens, m.kv_indptr, m.kv_indices,
                                              m.out_cache_loc)
        else:
            decode = plan.phase == "decode"
            prefix = [int(p) for p in plan.positions.values()]
            extend = [int(n) for n in plan.num_tokens.values()]
            lengths = [p + n for p, n in zip(prefix, extend, strict=True)]
            table = "decode" if decode else "prefill"
            req_pool_indices = on_device(torch.tensor(rows, dtype=torch.int64), device)
            kv_indptr_cpu = indptr_of(lengths)
            kv_indptr = on_device(kv_indptr_cpu, device)
            kv_indices = torch.empty(int(kv_indptr_cpu[-1]), dtype=torch.int32, device=device)
            one_each = all(n == 1 for n in extend)
            out_cache_loc = torch.empty(len(rows), dtype=torch.int64, device=device) if one_each else None
            self.kernels(table, KV_INDICES)(req_to_token, req_pool_indices,
                                            on_device(torch.tensor(lengths, dtype=torch.int64), device), kv_indptr,
                                            kv_indices, out_cache_loc)
            if out_cache_loc is None:
                out_cache_loc = torch.cat([req_to_token[r, p:p + n]
                                           for r, p, n in zip(rows, prefix, extend, strict=True)]).to(torch.int64)
            m = Metadata(table, 0, indptr_of(extend), kv_indptr_cpu, kv_indptr, kv_indices, out_cache_loc)
        m.planned = False
        self.forward_metadata = m
        made = self.operator(m)
        if made is not None:
            self._plan(m, made[0])

    def _bucket(self, size: int, req_to_token: torch.Tensor) -> Metadata:
        m = self.graph_metadata.get(size)
        if m is not None:
            return m
        device, width = req_to_token.device, req_to_token.shape[1]
        if self.kv_indices_buffer is None or self.kv_indices_buffer.numel() < size * width:
            self.kv_indices_buffer = torch.zeros(size * width, dtype=torch.int32, device=device)
        m = Metadata(GRAPH, size, torch.arange(size + 1, dtype=torch.int32), indptr_of([1] * size),
                     torch.zeros(size + 1, dtype=torch.int32, device=device), self.kv_indices_buffer[:size * width],
                     torch.zeros(size, dtype=torch.int64, device=device),
                     req_pool_indices=torch.zeros(size, dtype=torch.int64, device=device),
                     seq_lens=torch.ones(size, dtype=torch.int64, device=device),
                     kv_last_page_len=torch.ones(size, dtype=torch.int32, device=device))
        self.graph_metadata[size] = m
        assert m.kv_last_page_len is not None
        self.graph_buffers.update({f"paged.{size}.kv_indptr": m.kv_indptr, f"paged.{size}.kv_indices": m.kv_indices,
                                   f"paged.{size}.kv_last_page_len": m.kv_last_page_len,
                                   f"paged.{size}.out_cache_loc": m.out_cache_loc})
        self.generation += 1
        return m

    @staticmethod
    def _plan(m: Metadata, operator: Any) -> None:
        if not m.planned:
            operator.plan(m.qo_indptr_cpu, m.kv_indptr_cpu, m.kv_indptr, m.kv_indices)
            m.planned = True

    def forward(self, layer: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int,
                num_kv_heads: int, head_dim: int, scaling: float) -> torch.Tensor:
        m = self.forward_metadata
        if m is None:
            raise RuntimeError("layers.attn ran before prepare")
        cache = self.kv.bind(layer).buffers()
        k_cache, v_cache = cache["k"], cache["v"]
        self.kernels(m.table, "kv.store@1")(k_cache, v_cache, m.out_cache_loc, k.view(-1, num_kv_heads, head_dim),
                                           v.view(-1, num_kv_heads, head_dim))
        shape = (num_heads, num_kv_heads, head_dim, q.dtype, k_cache.dtype)
        made = self.operator(m)
        if made is None:
            make = self.kernels(m.table, DEFINITIONS["prefill" if m.table == "prefill" else "decode"])
            buffers = None if m.table != GRAPH else {"kv_indptr": m.kv_indptr, "kv_indices": m.kv_indices,
                                                     "kv_last_page_len": m.kv_last_page_len}
            made = self.operators[m.table, m.bucket] = (make(*shape, q.device, buffers), shape)
            self.operators_version = self.kernels.catalog.version
        operator, planned_for = made
        if planned_for != shape:
            # A plan bakes the head counts in; SGLang plans with the model's one set of them.
            raise ValueError(f"paged attention was planned for {planned_for}, layer {layer} runs {shape}")
        self._plan(m, operator)
        return operator.run(q.view(-1, num_heads, head_dim), k_cache, v_cache, scaling).view(-1, num_heads * head_dim)
