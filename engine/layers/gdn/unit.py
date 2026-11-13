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

For the prefix cache it keeps a row's state where the step's ``StateTrack`` says, in another slot
(doc/specs/prefix-cache.md; SGLang's ``mamba_track_*`` of the ``extra_buffer`` strategy,
hybrid_linear_attn_backend.py and gdn_backend.py): inside a prefill chunk from the chunk kernel's
state before each chunk of 64 (``h``) and from the input before the conv
(``_track_mamba_state_extend``, ``_init_track_conv_indices``); at the end of a row's input, its own
slot copied over, every layer in one kernel after the last GDN layer when it decodes
(``_track_mamba_state_decode``), in the graph too. At the end of a prefill row we copy its conv state
as well, where SGLang reads the window from the input: the same values once the row has as many
tokens as the window, and right for a shorter one.
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
FLA_CHUNK = 64
"""The chunk of ``gdn.chunk_delta_rule@1``: its ``h`` has the state before each one."""
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
    track: Track | None = None
    """The step's ``StateTrack`` rows (in a graph bucket always there, masked)."""
    verify: bool = False
    """A verify step (doc/specs/mtp.md): each row's ``steps`` tokens, states kept after each in the
    intermediate buffers at the row's lane (``lanes``), committed after the acceptance (``commit``)."""
    steps: int = 0
    lanes: torch.Tensor | None = None


@dataclass(frozen=True)
class Track:
    """Pool slots (an item ``i`` is slot ``i + 1``) and indices, on the device."""

    mask: torch.Tensor | None = None
    """Decode: per lane, whether it keeps its state; ``dst`` per lane (0, padding, where not)."""
    dst: torch.Tensor | None = None
    final_src: torch.Tensor | None = None
    """Prefill: rows that keep their state at the end of their input: from their slot ..."""
    final_dst: torch.Tensor | None = None
    """... to these."""
    h_src: torch.Tensor | None = None
    """Prefill: rows that keep a state inside their input: from these entries of ``h`` ..."""
    h_dst: torch.Tensor | None = None
    """... to these slots, with the conv window of the input before the position."""
    conv_window: torch.Tensor | None = None
    """``[h rows, conv width - 1]``: the input tokens of that window."""


class GDNAttnBackend:
    """``layers.gdn@1``."""

    def __init__(self, kernels: Kernels, states: Any, dtype: str, speculative: bool = False) -> None:
        self.kernels = kernels
        self.states = states
        self.speculative = speculative
        """Speculation is on (``spec.algorithm``): the verify step's kernels are resolved too."""
        self.dtype = dtype
        """The activations' dtype (``model.dtype``): some kernels take bf16 / fp16 only, and some want
        the conv state in the activations' dtype."""
        self.forward_metadata: Metadata | None = None
        self.graph_metadata: dict[int, Metadata] = {}
        self.generation = 0
        self.graph_buffers: dict[str, torch.Tensor] = {}
        self.tracked: set[int] = set()
        """Buckets whose last replay kept states: their mask is cleared before the next one."""

    @property
    def binding(self) -> GraphBinding:
        return GraphBinding(self.generation, self.graph_buffers)

    def resolve(self, states: Sequence[StateDecl]) -> Mapping[str, Mapping[str, str]]:
        tensors = next(d.tensors for d in states if d.kind == KIND)
        _, head_v_dim, head_k_dim = tensors["temporal"][0]
        shape = {"head_k_dim": head_k_dim, "head_v_dim": head_v_dim, "fp32": int(self.dtype == "f32"),
                 "conv_as_x": int(tensors["conv"][1] == self.dtype)}
        decode = {"gdn.qkvzba_conv1d_update@1": shape, "gdn.packed_decode@1": shape, "gdn.track_states@1": shape}
        if self.speculative:  # a verify step runs with the decode backend
            decode |= {"gdn.verify_conv1d_update@1": shape, "gdn.verify_recurrent@1": shape,
                       "gdn.commit_states@1": shape}
        return self.kernels.resolve({
            "prefill": {"gdn.conv1d_fn@1": shape, "gdn.gating@1": shape, "gdn.chunk_delta_rule@1": shape},
            "decode": decode})

    def prepare(self, batch: PersistentBatch, plan: BatchPlan, key: PlanKey) -> None:
        mapping = batch.binding.buffers()[f"slots.{KIND}"]
        device = mapping.device
        if key.bucket:
            # The bucket's fixed buffers; a padded lane has no slot (PAD_SLOT), as SGLang's padded
            # rows (MambaAttnBackendBase._replay_metadata): lanes sharing one slot would race on it.
            size, n = key.bucket, len(plan.rows)
            m = self.graph_metadata.get(size)
            if m is None:
                mask = torch.zeros(size, dtype=torch.bool, device=device)
                dst = torch.zeros(size, dtype=mapping.dtype, device=device)
                m = Metadata(True, torch.arange(size + 1, dtype=torch.int32, device=device),
                             torch.zeros(size, dtype=mapping.dtype, device=device),
                             torch.zeros(size, dtype=torch.bool, device=device), graph=True, track=Track(mask, dst))
                self.graph_metadata[size] = m
                self.graph_buffers.update({f"{size}.query_start_loc": m.query_start_loc,
                                           f"{size}.mamba_cache_indices": m.mamba_cache_indices,
                                           f"{size}.track_mask": mask, f"{size}.track_dst": dst})
                self.generation += 1
            m.mamba_cache_indices[n:].fill_(PAD_SLOT)
            if n:
                rows = on_device(torch.tensor(list(plan.rows.values()), dtype=torch.int64) + 1, device)
                m.mamba_cache_indices[:n].copy_(mapping[rows])
            lanes = self._decode_track(batch, plan, size)
            if lanes is not None or size in self.tracked:  # the last replay of this bucket kept states
                mask, dst = lanes or ([False] * size, [0] * size)
                assert m.track is not None and m.track.mask is not None and m.track.dst is not None
                m.track.mask.copy_(on_device(torch.tensor(mask), device))
                m.track.dst.copy_(on_device(torch.tensor(dst, dtype=mapping.dtype), device))
                self.tracked.discard(size)
                if lanes is not None:
                    self.tracked.add(size)
            self.forward_metadata = m
            return
        rows = on_device(torch.tensor(list(plan.rows.values()), dtype=torch.int64), device)
        extend = on_device(torch.tensor(list(plan.num_tokens.values()), dtype=torch.int32), device)
        prefix = on_device(torch.tensor(list(plan.positions.values()), dtype=torch.int64), device)
        bs = rows.shape[0]
        decode = plan.phase == "decode"
        if decode and plan.draft_tokens:  # a verify step: SGLang's TARGET_VERIFY metadata
            steps = plan.draft_tokens
            lanes = self._decode_track(batch, plan, bs)
            track = None if lanes is None else Track(on_device(torch.tensor(lanes[0]), device),
                                                     on_device(torch.tensor(lanes[1], dtype=mapping.dtype), device))
            self.forward_metadata = Metadata(
                True, torch.arange(bs + 1, dtype=torch.int32, device=device) * steps, mapping[rows + 1], prefix > 0,
                [steps] * bs, track=track, verify=True, steps=steps, lanes=torch.arange(bs, device=device))
            return
        if decode:
            query_start_loc = torch.arange(0, bs + 1, dtype=torch.int32, device=device)
        else:
            query_start_loc = torch.cat([torch.zeros(1, dtype=torch.int32, device=device),
                                         torch.cumsum(extend, dim=0, dtype=torch.int32)])
        # get_mamba_indices: req_index_to_mamba_index_mapping[req_pool_indices]
        cache_indices = mapping[rows + 1]
        if decode:
            lanes = self._decode_track(batch, plan, bs)
            track = None if lanes is None else Track(on_device(torch.tensor(lanes[0]), device),
                                                     on_device(torch.tensor(lanes[1], dtype=mapping.dtype), device))
        else:
            track = self._prefill_track(batch, plan, mapping, device)
        self.forward_metadata = Metadata(decode, query_start_loc, cache_indices, prefix > 0,
                                         [int(n) for n in plan.num_tokens.values()], track=track)

    def _decode_track(self, batch: PersistentBatch, plan: BatchPlan, lanes: int) -> tuple[list[bool], list[int]] | None:
        """Per lane (the plan's rows, then padding), whether it keeps its state and where;
        ``None``: no row does."""
        rows, dst, _ = batch.track(KIND)
        lane = {int(r): i for i, r in enumerate(plan.rows.values())}
        # Only the plan's rows: a graph's warm-up and capture run the padding row, while the batch
        # still has the last step's StateTrack.
        kept = [(lane[r], d) for r, d in zip(rows, dst, strict=True) if r in lane]
        if not kept:
            return None
        mask, slots = [False] * lanes, [0] * lanes
        for i, d in kept:
            mask[i], slots[i] = True, d + 1
        return mask, slots

    def _prefill_track(self, batch: PersistentBatch, plan: BatchPlan, mapping: torch.Tensor,
                       device: torch.device) -> Track | None:
        """``_init_track_ssm_indices`` and ``_init_track_conv_indices``: where each tracked row's
        state comes from."""
        rows, dst, upto = batch.track(KIND)
        if not rows:
            return None
        lane = {int(r): i for i, r in enumerate(plan.rows.values())}
        positions = [int(p) for p in plan.positions.values()]
        lengths = [int(n) for n in plan.num_tokens.values()]
        starts = [0]
        chunks = [0]
        for n in lengths:
            starts.append(starts[-1] + n)
            chunks.append(chunks[-1] + (n - 1) // FLA_CHUNK + 1)
        width = self.states.bind(self.states.pools()[2]).buffers()["conv"].shape[-1]
        final_src: list[int] = []
        final_dst: list[int] = []
        h_src: list[int] = []
        h_dst: list[int] = []
        window: list[list[int]] = []
        for r, d, at in zip(rows, dst, upto, strict=True):
            i = lane[r]
            rel = at - positions[i]
            if rel == lengths[i]:
                final_src.append(int(r) + 1)
                final_dst.append(d + 1)
            elif 0 < rel < lengths[i] and rel % FLA_CHUNK == 0 and rel >= width:
                h_src.append(chunks[i] + rel // FLA_CHUNK)
                h_dst.append(d + 1)
                window.append(list(range(starts[i] + rel - width, starts[i] + rel)))
            else:
                raise ValueError(f"row {r} keeps its state at {at}, {rel} into an input of {lengths[i]} tokens: "
                                 f"not its end nor a multiple of {FLA_CHUNK} inside it")

        def dev(values: list[int]) -> torch.Tensor:
            return on_device(torch.tensor(values, dtype=torch.int64), device)

        src = mapping[dev(final_src)].to(torch.int64) if final_src else None
        return Track(None, None, src, dev(final_dst) if final_dst else None, dev(h_src) if h_src else None,
                     dev(h_dst) if h_dst else None, dev([x for w in window for x in w]).view(len(window), width)
                     if window else None)

    def release_graphs(self) -> None:
        self.graph_metadata.clear()
        self.graph_buffers.clear()
        self.tracked.clear()
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
        if m.verify:
            return self._verify(m, layer, projected_qkvz, projected_ba, conv_states, ssm_states, conv_weights, bias,
                                activation, A_log, dt_bias, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
        if m.decode:
            phase = GRAPH if m.graph else "decode"
            mixed_qkv, z, b, a = self.kernels(phase, "gdn.qkvzba_conv1d_update@1")(
                projected_qkvz, projected_ba, conv_states, conv_weights, bias, cache_indices, qkv_dim, v_dim,
                num_v_heads, head_v_dim, activation)
            out = self.kernels(phase, "gdn.packed_decode@1")(
                mixed_qkv, a, b, A_log, dt_bias, head_k_dim ** -0.5, ssm_states, cache_indices, num_v_heads,
                head_v_dim)
            if m.track is not None:
                conv_pool, temporal_pool, last = self.states.pools()
                if layer == last:  # every layer's state is final now
                    self.kernels(phase, "gdn.track_states@1")(conv_pool, temporal_pool, cache_indices, m.track.mask,
                                                               m.track.dst)
            return out, z

        # qwen3_5_gdn_prefill_projection_views: the prefill kernels take strided views
        tokens = projected_qkvz.shape[0]
        mixed_qkv = projected_qkvz[:, :qkv_dim]
        z = projected_qkvz[:, qkv_dim:].view(tokens, num_v_heads, head_v_dim)
        b, a = projected_ba[:, :num_v_heads], projected_ba[:, num_v_heads:]
        t = m.track
        if t is not None and t.h_dst is not None and t.conv_window is not None:
            # the conv state at a position inside the input: the input before it (forward_extend)
            conv_states[t.h_dst] = mixed_qkv[t.conv_window].transpose(1, 2).to(conv_states.dtype)
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
        core_attn_out, _, h = self.kernels("prefill", "gdn.chunk_delta_rule@1")(query, key, value, g, beta,
                                                                              ssm_states, cache_indices,
                                                                              m.query_start_loc)
        if t is not None:  # _track_mamba_state_extend
            if t.h_src is not None and t.h_dst is not None:
                ssm_states[t.h_dst] = h.squeeze(0)[t.h_src].to(ssm_states.dtype)
            if t.final_src is not None and t.final_dst is not None:
                ssm_states[t.final_dst] = ssm_states[t.final_src]
                conv_states[t.final_dst] = conv_states[t.final_src]
        return core_attn_out, z


    def _verify(self, m: Metadata, layer: int, projected_qkvz: torch.Tensor, projected_ba: torch.Tensor,
                conv_states: torch.Tensor, ssm_states: torch.Tensor, conv_weights: torch.Tensor,
                bias: torch.Tensor | None, activation: str, A_log: torch.Tensor, dt_bias: torch.Tensor,
                num_k_heads: int, num_v_heads: int, head_k_dim: int, head_v_dim: int
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """``forward_extend`` in ``TARGET_VERIFY``: ``causal_conv1d_update`` over each row's tokens with
        the intermediate conv windows, then ``target_verify`` (the gated delta rule keeping the state
        after each token, the row's own left as it is)."""
        assert m.lanes is not None
        tokens, bs, steps = projected_qkvz.shape[0], m.mamba_cache_indices.shape[0], m.steps
        q_dim = k_dim = num_k_heads * head_k_dim
        qkv_dim = q_dim + k_dim + num_v_heads * head_v_dim
        z = projected_qkvz[:, qkv_dim:].view(tokens, num_v_heads, head_v_dim)
        b, a = projected_ba[:, :num_v_heads], projected_ba[:, num_v_heads:]
        conv_window, kept = self.states.verify_buffers(layer)
        x = projected_qkvz[:, :qkv_dim].reshape(bs, steps, qkv_dim).transpose(1, 2)
        mixed_qkv = self.kernels("decode", "gdn.verify_conv1d_update@1")(
            x, conv_states, conv_weights, bias, activation, m.mamba_cache_indices, conv_window, m.lanes)
        mixed_qkv = mixed_qkv.transpose(1, 2).reshape(tokens, qkv_dim)
        query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, num_v_heads * head_v_dim], dim=-1)
        out = self.kernels("decode", "gdn.verify_recurrent@1")(
            A_log, dt_bias, query.reshape(1, tokens, num_k_heads, head_k_dim),
            key.reshape(1, tokens, num_k_heads, head_k_dim), value.reshape(1, tokens, num_v_heads, head_v_dim),
            a, b, ssm_states, m.mamba_cache_indices, kept, m.lanes, steps)
        return out, z

    def commit(self, steps: torch.Tensor, track_steps: torch.Tensor) -> None:
        m = self.forward_metadata
        if m is None or not m.verify or m.lanes is None:
            raise RuntimeError("layers.gdn: commit after no verify step")
        conv_pool, temporal_pool, _ = self.states.pools()
        conv_window, kept = self.states.verify_pools()
        dst = m.track.dst if m.track is not None and m.track.dst is not None else torch.zeros_like(m.mamba_cache_indices)
        if m.track is not None and m.track.mask is not None:  # a lane without a tracking slot keeps nothing
            track_steps = torch.where(m.track.mask, track_steps, torch.full_like(track_steps, -1))
        else:
            track_steps = torch.full_like(track_steps, -1)
        self.kernels("decode", "gdn.commit_states@1")(conv_pool, temporal_pool, conv_window, kept, m.mamba_cache_indices,
                                                       m.lanes, steps, dst, track_steps)


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        kernels = Kernels(ctx.use("kernels.catalog@1"), config, "gdn")
        ctx.provide("layers.gdn@1", GDNAttnBackend(kernels, ctx.use("state.gdn.slots@1"), config["model.dtype"],
                                                   speculative=bool(config["spec.algorithm"])))


if TYPE_CHECKING:
    _: type[GDNBackend] = GDNAttnBackend
