"""GLM-5.3-Flash layers using CANN, FLA-NPU and Ascend's KPool operators."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from core.contracts.plan import GRAPH, PersistentBatch, PlanKey
from core.contracts.records import BatchPlan
from core.contracts.state import StateDecl
from layers.glm53.metadata import IndexShard, Row, Scan, SparseStep, Verify, device_metadata, pad_decode, prepare_scans, rows_of
from models.glm53_config import LINEAR, SPARSE, Config

SERVICE = "layers.glm53@1"


class GlmLayers:
    def __init__(self, group: Any, page: int, cfg: Config, states: Any = None, draft_tokens: int = 0,
                 context: int = 0, *, sequence_prefill: bool = False, index_prefill: bool = False,
                 shared_expert_dp: bool = False, prefill_matmul_reduce_scatter: bool = False) -> None:
        load_ops = import_module("vllm_ascend.utils").enable_custom_op
        if not load_ops():
            raise RuntimeError("GLM-5.3-Flash requires the Ascend custom operator library")
        self.group, self.page, self.cfg, self.context = group, page, cfg, context
        self.states = states
        self.tail_capacity = cfg.tail_capacity(draft_tokens)
        self.npu = import_module("torch_npu")
        self.ops = torch.ops._C_ascend
        self.fla = import_module("fla_npu.ops.ascendc")
        self.kda_ops = import_module("vllm_ascend.ops.kda")
        self.output_norm = import_module("vllm_ascend.ops.triton.kda.kda").rms_norm_gated
        self.compress = import_module(
            "vllm_ascend.ops.triton.glm5_next_kpool_tail_compress").glm5_next_kpool_tail_compress_and_write_cache_triton
        self.index = import_module(
            "vllm_ascend.ops.triton.glm5_next_lightning_indexer").glm5_next_lightning_indexer_triton
        self.scans: tuple[Scan, ...] = ()
        self.sparse: SparseStep | None = None
        self.verify: Verify | None = None
        self.verify_rows = 0
        self.graph_scans: dict[int, Scan] = {}
        self.graph_sparse: dict[tuple[int, int], SparseStep] = {}
        self.graph_verify: dict[tuple[int, int], Verify] = {}
        self.generation = 0
        self.dfc = os.environ.get("RSINFER_GLM_MOE") == "dfc"
        if (sequence_prefill or shared_expert_dp) and not self.dfc:
            raise ValueError("GLM sequence parallel prefill and shared-expert replication require DFC")
        self.sequence_prefill = sequence_prefill
        self.shard_prefill = False
        self.index_prefill = index_prefill and self.group.size > 1
        self.shared_expert_dp = shared_expert_dp
        if prefill_matmul_reduce_scatter and not sequence_prefill:
            raise ValueError("fused prefill projections require GLM sequence parallelism")
        self.prefill_matmul_reduce_scatter = (
            prefill_matmul_reduce_scatter and group.size in (2, 4, 8)
            and os.environ.get("HCCL_OP_EXPANSION_MODE", "").upper() == "AIV"
            and hasattr(self.npu, "npu_mm_reduce_scatter_base"))
        self.index_shard: IndexShard | None = None
        if self.dfc:
            self.npu.npu.config.allow_internal_format = True
            backend = group.device_group._get_backend(torch.device("npu"))
            self.comm_name = backend.get_hccl_comm_name(group.rank)
            self.expert_counts = torch.empty(cfg.experts // group.size, dtype=torch.int32, device="npu")
            self.dfc_bias = [torch.empty(0, dtype=torch.float32, device="npu")]

    def cache_version(self) -> str:
        source = Path(__file__)
        comm = (os.environ.get("HCCL_OP_EXPANSION_MODE", ""), os.environ.get("HCCL_DETERMINISTIC", ""))
        contract = "\0".join(comm).encode()
        return hashlib.sha256(source.read_bytes() + source.with_name("metadata.py").read_bytes()
                              + bytes((self.dfc, self.sequence_prefill, self.index_prefill, self.shared_expert_dp,
                                       self.prefill_matmul_reduce_scatter))
                              + contract).hexdigest()

    @property
    def binding(self) -> GlmLayers:
        return self

    def buffers(self) -> Mapping[str, torch.Tensor]:
        return {}

    def release_graphs(self) -> None:
        self.graph_scans.clear()
        self.graph_sparse.clear()
        self.graph_verify.clear()
        self.scans, self.sparse, self.verify = (), None, None

    def resolve(self, states: Sequence[StateDecl]) -> Mapping[str, Mapping[str, str]]:
        backend = "cann-dfc" if self.dfc else "cann"
        return {"prefill": {"backend": backend}, "decode": {"backend": backend}, GRAPH: {"backend": "acl_graph"}}

    def prepare(self, batch: PersistentBatch, plan: BatchPlan, key: PlanKey) -> None:
        rows = rows_of(plan)
        device = batch.binding.buffers()["last_token"].device
        self.shard_prefill = (self.sequence_prefill and plan.phase == "prefill"
                              and sum(r.count for r in rows) >= self.group.size)
        self.verify = None
        self.index_shard = None
        self.verify_rows = len(rows)
        if key.bucket:
            self.prepare_graph(batch, rows, device, key.bucket, plan.draft_tokens, draft=False)
            return
        if LINEAR in self.cfg.attention:
            if plan.phase == "decode" and plan.draft_tokens:
                self.verify = Verify.prepare(batch, rows, device, plan.draft_tokens)
            else:
                self.scans = prepare_scans(batch, rows, device, plan.phase == "decode")
        if SPARSE in self.cfg.attention:
            self.sparse = SparseStep.prepare(batch, rows, self.page, self.cfg.pool, self.tail_capacity, device)
            self.prepare_index(rows, plan, device)

    def prepare_draft(self, batch: PersistentBatch, plan: BatchPlan, key: PlanKey) -> None:
        device = batch.binding.buffers()["last_token"].device
        rows = rows_of(plan)
        self.index_shard = None
        if key.bucket:
            self.prepare_graph(batch, rows, device, key.bucket, plan.draft_tokens, draft=True)
        else:
            self.sparse = SparseStep.prepare(batch, rows, self.page, self.cfg.pool, self.tail_capacity, device)
            self.prepare_index(rows, plan, device)

    def prepare_index(self, rows: Sequence[Row], plan: BatchPlan, device: torch.device) -> None:
        if self.index_prefill and plan.phase == "prefill" and sum(row.count for row in rows) >= self.group.size:
            self.index_shard = IndexShard.prepare(rows, self.cfg.pool, self.group.rank, self.group.size, device)

    def prepare_graph(self, batch: PersistentBatch, rows: list[Row], device: torch.device,
                      bucket: int, verify: int, *, draft: bool) -> None:
        width = verify or 1
        rows = pad_decode(rows, bucket, width)
        shape = (bucket, width)
        cpu = torch.device("cpu")
        if not draft and LINEAR in self.cfg.attention:
            if verify:
                self.verify = self.graph_verify[shape] = device_metadata(
                    Verify.prepare(batch, rows, cpu, verify), device, self.graph_verify.get(shape))
            else:
                scan, = prepare_scans(batch, rows, cpu, decode=True, graph=True)
                self.graph_scans[bucket] = device_metadata(scan, device, self.graph_scans.get(bucket))
                self.scans = (self.graph_scans[bucket],)
        if draft or SPARSE in self.cfg.attention:
            sparse = SparseStep.prepare(batch, rows, self.page, self.cfg.pool, self.tail_capacity, cpu, self.context)
            self.sparse = self.graph_sparse[shape] = device_metadata(sparse, device, self.graph_sparse.get(shape))

    def rms(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return self.npu.npu_rms_norm(x, weight, epsilon=self.cfg.eps)[0]

    def add_rms(self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        normed, _, summed = self.npu.npu_add_rms_norm(x, residual, weight, epsilon=self.cfg.eps)
        return normed, summed

    def pre(self, x: torch.Tensor, w: Mapping[str, torch.Tensor], name: str, norm: str
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        h, post, mix = self.ops.npu_hc_pre_v2(x, w[name + ".fn"], w[name + ".scale"], w[name + ".base"],
                                              cfg.streams, cfg.sinkhorn_iters, cfg.eps, cfg.hc_eps)
        if cfg.post_mult != 2.0:
            post = post * (cfg.post_mult / 2.0)
        return self.rms(h, w[norm]), post, mix

    def post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
        return self.ops.npu_hc_post(x.unsqueeze(0), residual.unsqueeze(0), post.unsqueeze(0), mix.unsqueeze(0))[0]

    def run_scan(self, m: Scan, qkv: torch.Tensor, raw_gate: torch.Tensor, beta: torch.Tensor,
                 w: Mapping[str, torch.Tensor], conv: torch.Tensor, state: torch.Tensor
                 ) -> torch.Tensor:
        cfg = self.cfg
        heads = cfg.kda_heads // self.group.size
        if m.order is not None:
            qkv = qkv.index_select(0, m.order)
            raw_gate = raw_gate.index_select(0, m.order)
            beta = beta.index_select(0, m.order)
        a_log, dt_bias = w["forget_gate.A_log"], w["forget_gate.dt_bias"]
        if m.decode:
            projected = self.fla.causal_conv1d_update(
                qkv, conv, w["conv.weight"], activation="silu", conv_state_indices=m.slots,
                query_start_loc=m.cu_device, null_block_id=-1)
        else:
            projected = self.fla.causal_conv1d_fn(
                qkv, w["conv.weight"], None, conv_states=conv, query_start_loc=m.cu_device,
                cache_indices=m.slots, has_initial_state=m.initial, activation="silu", pad_slot_id=-1,
                null_block_id=0)
        q, k, v = (t.reshape(1, -1, heads, cfg.kda_dim) for t in projected.chunk(3, dim=-1))
        g = raw_gate.reshape(1, -1, heads, cfg.kda_dim)
        if m.decode:
            out = self.kda_ops.run_recurrent_kda(
                q, k, v, g, beta.unsqueeze(0), state, m.cu_device, m.slots, a_log, dt_bias,
                lower_bound=cfg.gate_bound, beta_is_preprocessed=False)
        else:
            slot_ids = m.slots.to(torch.int64)
            initial = state.index_select(0, slot_ids)
            initial = initial.masked_fill(~m.initial[:, None, None, None], 0)
            out, final = self.kda_ops.run_chunk_kda(
                q, k, v, g, beta.unsqueeze(0).float().sigmoid(), initial, m.cu, m.chunks, a_log, dt_bias,
                lower_bound=cfg.gate_bound)
            state.index_copy_(0, slot_ids, final)
        if m.snapshot.numel():
            conv.index_copy_(0, m.snapshot, conv.index_select(0, m.src))
            state.index_copy_(0, m.snapshot, state.index_select(0, m.src))
        return out[0]

    def run_verify(self, layer: int, m: Verify, qkv: torch.Tensor, raw_gate: torch.Tensor, beta: torch.Tensor,
                   w: Mapping[str, torch.Tensor], conv: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        lanes, steps = m.state_slots.shape
        conv_window, states = self.states.verify_buffers(layer)
        raw = qkv.contiguous().view(lanes, steps, -1)
        history = conv.index_select(0, m.slots)
        # Keep every post-token convolution window before UPDATE overwrites the input projection.
        joined = torch.cat((history, raw), dim=1)
        windows = joined.unfold(1, cfg.conv_width - 1, 1)[:, 1:].transpose(-1, -2)
        conv_window[:lanes].copy_(windows)
        projected = self.fla.causal_conv1d_update(raw, history, w["conv.weight"], activation="silu",
                                                  conv_state_indices=m.conv_slots, null_block_id=-1).flatten(0, 1)
        states[:lanes, 0].copy_(state.index_select(0, m.slots))
        # Reuse the seed slot for the first output state. FLA writes one state into each later slot.
        heads = cfg.kda_heads // self.group.size
        q, k, v = (t.reshape(1, -1, heads, cfg.kda_dim) for t in projected.chunk(3, dim=-1))
        g = raw_gate.reshape(1, -1, heads, cfg.kda_dim)
        out = self.kda_ops.run_recurrent_kda(
            q, k, v, g, beta.unsqueeze(0), states[:lanes].flatten(0, 1), m.cu, m.state_slots,
            w["forget_gate.A_log"], w["forget_gate.dt_bias"], lower_bound=cfg.gate_bound,
            beta_is_preprocessed=False, num_accepted_tokens=m.initial_tokens)
        return out[0]

    def token_slice(self, rows: int) -> tuple[int, int]:
        width = (rows + self.group.size - 1) // self.group.size
        return self.group.rank * width, width

    def gather_tokens(self, local: torch.Tensor, rows: int) -> torch.Tensor:
        full = local.new_empty((len(local) * self.group.size, *local.shape[1:]))
        dist.all_gather_into_tensor(full, local.contiguous(), group=self.group.device_group)
        return full[:rows]

    def reduce_tokens(self, full: torch.Tensor, sharded: bool) -> torch.Tensor:
        if not sharded:
            return self.group.reduce(full)
        _, width = self.token_slice(len(full))
        padding = width * self.group.size - len(full)
        if padding:
            full = F.pad(full, (0, 0, 0, padding))
        local = full.new_empty((width, full.shape[1]))
        dist.reduce_scatter_tensor(local, full, group=self.group.device_group)
        return local

    def out_projection(self, x: torch.Tensor, weight: torch.Tensor, sharded: bool) -> torch.Tensor:
        if not (sharded and self.prefill_matmul_reduce_scatter and 256 <= x.shape[1] < 65535):
            return self.reduce_tokens(F.linear(x, weight), sharded)
        _, width = self.token_slice(len(x))
        pad = width * self.group.size - len(x)
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        return self.npu.npu_mm_reduce_scatter_base(
            x.contiguous(), weight.t(), self.comm_name, self.group.size, comm_mode="aiv")

    def linear_attention(self, layer: int, x: torch.Tensor, w: Mapping[str, torch.Tensor],
                         state: Mapping[str, torch.Tensor], *, sharded: bool = False) -> torch.Tensor:
        cfg = self.cfg
        heads = cfg.kda_heads // self.group.size
        width = heads * cfg.kda_dim
        projected = F.linear(x, w["input.weight"])
        qkv, beta, f, g = projected.split((3 * width, heads, cfg.kda_dim, cfg.kda_dim), dim=-1)
        raw_gate = F.linear(f, w["forget_gate.f_b_proj.weight"])
        output_gate = F.linear(g, w["g_b_proj.weight"]).reshape(-1, heads, cfg.kda_dim)
        conv, temporal = state["conv"], state["temporal"]
        verify = self.verify
        if verify is not None:
            out = self.run_verify(layer, verify, qkv, raw_gate, beta, w, conv, temporal)
        elif len(self.scans) == 1:
            out = self.run_scan(self.scans[0], qkv.contiguous(), raw_gate, beta, w, conv, temporal)
        else:
            out = torch.empty_like(output_gate)
            for scan in self.scans:
                assert scan.order is not None
                out.index_copy_(0, scan.order, self.run_scan(scan, qkv, raw_gate, beta, w, conv, temporal))
        out = self.output_norm(out.contiguous(), output_gate, w["o_norm.weight"], None, activation="sigmoid", eps=1e-5)
        return self.out_projection(out.flatten(1), w["o_proj.weight"], sharded)

    def commit(self, steps: torch.Tensor, track_steps: torch.Tensor) -> None:
        verify = self.verify
        if verify is None:
            raise RuntimeError("KDA commit requires a target verify step")
        conv, temporal, _ = self.states.pools()
        conv_window, states = self.states.verify_pools()
        live, lane, track = (values[:self.verify_rows] for values in (verify.slots, verify.lanes, verify.track))
        conv[:, live] = conv_window[:, lane, steps]
        temporal[:, live] = states[:, lane, steps]
        lanes = ((track > 0) & (track_steps >= 0)).nonzero().flatten()
        dst, kept = track[lanes], track_steps[lanes]
        conv[:, dst] = conv_window[:, lanes, kept]
        temporal[:, dst] = states[:, lanes, kept]

    def sparse_indices(self, q_c: torch.Tensor, positions: torch.Tensor, weights: torch.Tensor,
                       query_weight: torch.Tensor, cache: torch.Tensor, metadata: SparseStep) -> torch.Tensor:
        cfg, shard = self.cfg, self.index_shard
        args = dict(index_topk=cfg.sparse_topk, index_kpool=cfg.pool, pack_tail=True)
        if shard is None:
            query = F.linear(q_c, query_weight).reshape(-1, cfg.index_heads, cfg.index_dim)
            return self.index(query, cache, weights, metadata.ends, metadata.pooled, metadata.table, positions,
                              max_pool_seq_len=metadata.max_pool, **args)
        local = q_c[shard.tokens]
        if len(local):
            query = F.linear(local, query_weight).reshape(-1, cfg.index_heads, cfg.index_dim)
            indices = self.index(query, cache, weights[shard.tokens], shard.ends, shard.pooled,
                                 metadata.table[shard.requests], positions[shard.tokens],
                                 max_pool_seq_len=shard.max_pool, **args)
        else:
            indices = torch.empty((0, 1, cfg.sparse_topk + cfg.pool - 1), dtype=torch.int32, device=q_c.device)
        padded = F.pad(indices, (0, 0, 0, 0, 0, shard.width - len(local)), value=-1)
        return self.gather_tokens(padded, len(q_c))

    def sparse_attention(self, x: torch.Tensor, positions: torch.Tensor, w: Mapping[str, torch.Tensor],
                         state: Mapping[str, torch.Tensor], *, sharded: bool = False) -> torch.Tensor:
        cfg, m = self.cfg, self.sparse
        assert m is not None
        heads = cfg.mla_heads // self.group.size
        q_c = self.rms(F.linear(x, w["q_a_proj.weight"]), w["q_a_layernorm.weight"])
        q = F.linear(q_c, w["q_b_proj.weight"]).view(-1, heads, cfg.q_dim)
        kv = self.rms(F.linear(x, w["kv_a_proj_with_mqa.weight"]), w["kv_a_layernorm.weight"])
        state["kv"].index_copy_(0, m.kv_slots, kv.unsqueeze(1))
        cache = state["kv"].view(-1, m.page, 1, cfg.kv_rank)
        index_cache = state["index"].view(-1, m.page // cfg.pool, 1, cfg.index_dim)
        p = "indexer."
        projected = F.linear(x.float(), w[p + "wk_weights.weight"])
        keys, weights = projected.split((cfg.index_dim, cfg.index_heads), dim=-1)
        keys = F.layer_norm(keys.contiguous(), (cfg.index_dim,), w[p + "k_norm.weight"], w[p + "k_norm.bias"], 1e-6)
        gate = F.linear(x.float(), w[p + "index_kpool_compress_gate"])
        weights = weights.to(q.dtype) * (cfg.index_dim * cfg.index_heads) ** -0.5
        self.compress(state["tail"], index_cache, keys, gate, w[p + "index_kpool_compress_ape"], positions,
                      m.ends, m.seen, m.tail_slots, m.tail_table, m.pool_slots, cfg.pool)
        indices = self.sparse_indices(q_c, positions, weights, w[p + "wq_b.weight"], index_cache, m)
        uk, uv = w["kv_b_proj.weight"].view(heads, cfg.q_dim + cfg.v_dim, cfg.kv_rank).split(
            (cfg.q_dim, cfg.v_dim), dim=1)
        absorbed = torch.bmm(q.transpose(0, 1), uk).transpose(0, 1).contiguous()
        out = self.ops.npu_sparse_flash_attention(
            query=absorbed, key=cache, value=cache, sparse_indices=indices, scale_value=cfg.q_dim ** -0.5,
            sparse_block_size=1, block_table=m.table, actual_seq_lengths_query=m.ends, actual_seq_lengths_kv=m.seen,
            layout_query="TND", layout_kv="PA_BSND", sparse_mode=3, attention_mode=2, return_softmax_lse=False)[0]
        value = torch.bmm(out.transpose(0, 1), uv.transpose(1, 2)).transpose(0, 1).flatten(1)
        return self.out_projection(value, w["o_proj.weight"], sharded)

    def swiglu(self, x: torch.Tensor) -> torch.Tensor:
        return self.npu.npu_clipped_swiglu(x, alpha=1.0, limit=self.cfg.swiglu_limit,
                                          bias=0.0, interleaved=False)

    def w8a8(self, x: torch.Tensor, w: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
        quantized, scale = self.npu.npu_dynamic_quant(x, dst_type=torch.int8)
        return self.npu.npu_quant_matmul(quantized, w[name + ".weight"], w[name + ".scale"], pertoken_scale=scale,
                                        output_dtype=x.dtype)

    def dense(self, x: torch.Tensor, w: Mapping[str, torch.Tensor], prefix: str = "") -> torch.Tensor:
        return self.w8a8(self.swiglu(self.w8a8(x, w, prefix + "gate_up")), w, prefix + "down")

    def moe(self, x: torch.Tensor, w: Mapping[str, torch.Tensor], *, sharded_rows: int = 0) -> torch.Tensor:
        cfg = self.cfg
        logits = F.linear(x.float(), w["gate.weight"])
        bias = w["gate.e_score_correction_bias"]
        if self.dfc:
            prob, top, _ = self.ops.moe_gating_top_k(
                logits, k=cfg.expert_topk, k_group=1, group_count=1, group_select_mode=1, renorm=1, norm_type=1,
                out_flag=False, routed_scaling_factor=cfg.routed_scale, eps=1e-20, bias_opt=bias)
            return self.dispatch_ffn(x, top.to(torch.int32), prob, w, sharded_rows)
        score = logits.sigmoid()
        top = (score + bias).topk(cfg.expert_topk, sorted=False).indices
        prob = score.gather(1, top)
        prob = prob * (cfg.routed_scale / prob.sum(dim=-1, keepdim=True))
        top = top.to(torch.int32)
        count = cfg.experts // self.group.size
        start = self.group.rank * count
        packed, inverse, groups, scale = self.npu.npu_moe_init_routing_v2(
            x, top, active_num=top.numel(), expert_num=cfg.experts, expert_tokens_num_type=1, expert_tokens_num_flag=True,
            active_expert_range=[start, start + count], quant_mode=1, x_dtype=torch.int8)
        groups = groups.to(torch.int64)
        h = self.expert_gemm(packed, scale, groups, w, "experts.gate_up", x.dtype)
        quant, scale = self.npu.npu_dynamic_quant(self.swiglu(h), dst_type=torch.int8)
        h = self.expert_gemm(quant, scale, groups, w, "experts.down", x.dtype)
        prob = prob.masked_fill((top < start) | (top >= start + count), 0).to(x.dtype)
        routed = self.npu.npu_moe_token_unpermute(h, inverse.abs(), probs=prob)
        if cfg.shared_width:
            routed = routed + self.dense(x, w, "shared_experts.")
        return self.group.reduce(routed)

    def dispatch_ffn(self, x: torch.Tensor, expert: torch.Tensor, prob: torch.Tensor,
                     w: Mapping[str, torch.Tensor], sharded_rows: int) -> torch.Tensor:
        cfg = self.cfg
        size = self.group.size
        rows = sharded_rows or len(x)
        start, lane = self.token_slice(rows)
        used = max(0, min(rows - start, lane))
        if sharded_rows:
            vec, idx, router = x, expert, prob
        else:
            end = start + lane
            vec = F.pad(x[start:end], (0, 0, 0, lane - used))
            idx = F.pad(expert[start:end], (0, 0, 0, lane - used))
            router = F.pad(prob[start:end], (0, 0, 0, lane - used))
        mask = None if used == lane else torch.arange(lane, device=x.device) < used
        out = torch.empty_like(vec)
        # Allow every routed assignment to land on the same rank.
        self.ops.dispatch_ffn_combine(
            vec, [w["experts.gate_up.weight"]], [w["experts.down.weight"]], idx,
            [w["experts.gate_up.scale"]], [w["experts.down.scale"]], self.dfc_bias, self.dfc_bias,
            router, self.comm_name, rows * cfg.expert_topk, out, self.expert_counts,
            x_active_mask=mask, swiglu_limit=cfg.swiglu_limit, world_size=size)
        if self.shared_expert_dp:
            if cfg.shared_width:
                out = out + self.dense(vec, w, "shared_experts.")
            return out if sharded_rows else self.gather_tokens(out, rows)
        if sharded_rows:
            if not cfg.shared_width:
                return out
            shared = self.dense(self.gather_tokens(x, rows), w, "shared_experts.")
            return out + self.reduce_tokens(shared, True)
        shared = self.dense(x, w, "shared_experts.") if cfg.shared_width else torch.zeros_like(x)
        if used:
            shared[start:start + used].add_(out[:used])
        return self.group.reduce(shared)

    def expert_gemm(self, x: torch.Tensor, scale: torch.Tensor, groups: torch.Tensor,
                    w: Mapping[str, torch.Tensor], name: str, dtype: torch.dtype) -> torch.Tensor:
        return self.npu.npu_grouped_matmul(
            x=[x], weight=[w[name + ".weight"]], scale=[w[name + ".scale"]], per_token_scale=[scale],
            group_list=groups, group_list_type=1, split_item=2, group_type=0, output_dtype=dtype)[0]


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        cfg = Config.load(config["model.path"], config["model.hf_config"])
        steps = config["spec.num_draft_tokens"] if config["spec.algorithm"] else 0
        ctx.provide(SERVICE, GlmLayers(ctx.use("comm.group@1"), config["state.kv.page_size"], cfg,
                                      ctx.use("state.gdn.slots@1"), steps, config["batch.max_context_len"],
                                      sequence_prefill=config["glm53.sequence_prefill"],
                                      index_prefill=config["glm53.index_prefill"],
                                      shared_expert_dp=config["glm53.shared_expert_dp"],
                                      prefill_matmul_reduce_scatter=config["glm53.prefill_matmul_reduce_scatter"]))
