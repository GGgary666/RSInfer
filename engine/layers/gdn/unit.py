# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""``layers.gdn``: the linear-attention (GDN) shell with its backend (``layers.gdn@1``). Adapted
from SGLang's ``RadixLinearAttention`` and ``GDNAttnBackend`` (srt/layers/attention/linear/
gdn_backend.py, hybrid_linear_attn_backend.py, e838360; LICENSE.sglang), the plain paths without
speculation or state tracking. ``prepare`` is ``_forward_metadata`` (``query_start_loc``,
``mamba_cache_indices``) plus ``has_initial_state``; ``forward`` is ``forward_decode`` /
``forward_extend`` (prefill and mixed steps, as SGLang's MIXED mode), with the solutions
``resolve`` fixed at start from ``gdn.backend`` and its per-phase overrides. It takes the layer's
two input projections as they come out of the GEMMs, as SGLang's CUDA path hands them over
(``Qwen3_5GatedDeltaNet.forward``): a decode step unpacks them in the kernel that steps the conv
(``fused_qkvzba_causal_conv1d_update_contiguous``) and reads q, k and v from the packed result
(``packed_decode``); a prefill step works on views of them
(``qwen3_5_gdn_prefill_projection_views``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from core.contracts.plan import GRAPH, GDNBackend, PersistentBatch, PlanKey
from core.contracts.records import BatchPlan
from core.contracts.state import StateDecl
from layers.solutions import Kernels

KIND = "gdn.slots"
PAD_SLOT = -1
"""SGLang's ``pad_slot_id``, the slot of a padded lane: a kernel reads and writes no state for it,
and what it returns for the lane is undefined."""


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
    query_start_loc: torch.Tensor
    mamba_cache_indices: torch.Tensor
    has_initial_state: torch.Tensor
    extend_seq_lens_cpu: Sequence[int] = ()
    graph: bool = False
    """The metadata of a graph bucket: fixed buffers, filled before each replay."""


class GDNAttnBackend:
    """``layers.gdn@1``."""

    def __init__(self, kernels: Kernels, states: Any, dtype: str) -> None:
        self.kernels = kernels
        self.states = states
        self.dtype = dtype
        """The activations' dtype (``model.dtype``): some kernels take bf16 / fp16 only, and some want
        the conv state in the activations' dtype."""
        self.forward_metadata: Metadata | None = None
        self.graph_metadata: dict[int, Metadata] = {}
        self.generation = 0
        self.graph_buffers: dict[str, torch.Tensor] = {}

    @property
    def binding(self) -> GraphBinding:
        return GraphBinding(self.generation, self.graph_buffers)

    def resolve(self, states: Sequence[StateDecl]) -> Mapping[str, Mapping[str, str]]:
        tensors = next(d.tensors for d in states if d.kind == KIND)
        _, head_v_dim, head_k_dim = tensors["temporal"][0]
        shape = {"head_k_dim": head_k_dim, "head_v_dim": head_v_dim, "fp32": int(self.dtype == "f32"),
                 "conv_as_x": int(tensors["conv"][1] == self.dtype)}
        return self.kernels.resolve({
            "prefill": {"gdn.conv1d_fn@1": shape, "gdn.gating@1": shape, "gdn.chunk_delta_rule@1": shape},
            "decode": {"gdn.qkvzba_conv1d_update@1": shape, "gdn.packed_decode@1": shape}})

    def prepare(self, batch: PersistentBatch, plan: BatchPlan, key: PlanKey) -> None:
        mapping = batch.binding.buffers()[f"slots.{KIND}"]
        device = mapping.device
        if key.bucket:
            # The bucket's fixed buffers; a padded lane has no slot (PAD_SLOT), as SGLang's padded
            # rows (MambaAttnBackendBase._replay_metadata): lanes sharing one slot would race on it.
            size, n = key.bucket, len(plan.rows)
            m = self.graph_metadata.get(size)
            if m is None:
                m = Metadata(True, torch.arange(size + 1, dtype=torch.int32, device=device),
                             torch.zeros(size, dtype=mapping.dtype, device=device),
                             torch.zeros(size, dtype=torch.bool, device=device), graph=True)
                self.graph_metadata[size] = m
                self.graph_buffers.update({f"{size}.query_start_loc": m.query_start_loc,
                                           f"{size}.mamba_cache_indices": m.mamba_cache_indices})
                self.generation += 1
            m.mamba_cache_indices[n:].fill_(PAD_SLOT)
            if n:
                rows = on_device(torch.tensor(list(plan.rows.values()), dtype=torch.int64) + 1, device)
                m.mamba_cache_indices[:n].copy_(mapping[rows])
            self.forward_metadata = m
            return
        rows = on_device(torch.tensor(list(plan.rows.values()), dtype=torch.int64), device)
        extend = on_device(torch.tensor(list(plan.num_tokens.values()), dtype=torch.int32), device)
        prefix = on_device(torch.tensor(list(plan.positions.values()), dtype=torch.int64), device)
        bs = rows.shape[0]
        decode = plan.phase == "decode"
        if decode:
            query_start_loc = torch.arange(0, bs + 1, dtype=torch.int32, device=device)
        else:
            query_start_loc = torch.cat([torch.zeros(1, dtype=torch.int32, device=device),
                                         torch.cumsum(extend, dim=0, dtype=torch.int32)])
        # get_mamba_indices: req_index_to_mamba_index_mapping[req_pool_indices]
        self.forward_metadata = Metadata(decode, query_start_loc, mapping[rows + 1], prefix > 0,
                                         [int(n) for n in plan.num_tokens.values()])

    def release_graphs(self) -> None:
        self.graph_metadata.clear()
        self.graph_buffers.clear()
        self.forward_metadata = None
        self.generation += 1

    def forward(self, layer: int, projected_qkvz: torch.Tensor, projected_ba: torch.Tensor,
                conv_weights: torch.Tensor, bias: torch.Tensor | None, activation: str, A_log: torch.Tensor,
                dt_bias: torch.Tensor, num_k_heads: int, num_v_heads: int, head_k_dim: int, head_v_dim: int
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """``projected_qkvz`` ``[tokens, conv_dim + HV * head_v_dim]`` is ``[q | k | v | z]``,
        ``projected_ba`` ``[tokens, 2 * HV]`` is ``[b | a]``. Returns the core output
        ``[1, tokens, HV, head_v_dim]`` and ``z`` ``[tokens, HV, head_v_dim]`` (a view of the
        projection where the kernels left it there)."""
        m = self.forward_metadata
        if m is None:
            raise RuntimeError("layers.gdn ran before prepare")
        cache = self.states.bind(layer).buffers()
        conv_states, ssm_states = cache["conv"], cache["temporal"]
        q_dim = k_dim = num_k_heads * head_k_dim
        v_dim = num_v_heads * head_v_dim
        qkv_dim = q_dim + k_dim + v_dim
        cache_indices = m.mamba_cache_indices
        if m.decode:
            phase = GRAPH if m.graph else "decode"
            mixed_qkv, z, b, a = self.kernels(phase, "gdn.qkvzba_conv1d_update@1")(
                projected_qkvz, projected_ba, conv_states, conv_weights, bias, cache_indices, qkv_dim, v_dim,
                num_v_heads, head_v_dim, activation)
            return self.kernels(phase, "gdn.packed_decode@1")(
                mixed_qkv, a, b, A_log, dt_bias, head_k_dim ** -0.5, ssm_states, cache_indices, num_v_heads,
                head_v_dim), z

        # qwen3_5_gdn_prefill_projection_views: the prefill kernels take strided views
        tokens = projected_qkvz.shape[0]
        mixed_qkv = projected_qkvz[:, :qkv_dim]
        z = projected_qkvz[:, qkv_dim:].view(tokens, num_v_heads, head_v_dim)
        b, a = projected_ba[:, :num_v_heads], projected_ba[:, num_v_heads:]
        mixed_qkv = self.kernels("prefill", "gdn.conv1d_fn@1")(  # _convolve_prefill
            mixed_qkv.transpose(0, 1), conv_weights, bias, activation, conv_states,
            m.has_initial_state, cache_indices, m.query_start_loc,
            m.extend_seq_lens_cpu).transpose(0, 1)[: mixed_qkv.shape[0]]
        actual_seq_len = mixed_qkv.shape[0]
        query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)
        query = query.view(1, actual_seq_len, num_k_heads, head_k_dim)
        key = key.view(1, actual_seq_len, num_k_heads, head_k_dim)
        value = value.view(1, actual_seq_len, num_v_heads, head_v_dim)
        g, beta = self.kernels("prefill", "gdn.gating@1")(A_log, a, b, dt_bias)  # _prefill_gates
        core_attn_out, _, _ = self.kernels("prefill", "gdn.chunk_delta_rule@1")(query, key, value, g, beta,
                                                                              ssm_states, cache_indices,
                                                                              m.query_start_loc)
        return core_attn_out, z


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        kernels = Kernels(ctx.use("kernels.catalog@1"), config, "gdn")
        ctx.provide("layers.gdn@1", GDNAttnBackend(kernels, ctx.use("state.gdn.slots@1"), config["model.dtype"]))


if TYPE_CHECKING:
    _: type[GDNBackend] = GDNAttnBackend
