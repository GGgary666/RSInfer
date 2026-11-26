"""GLM-5.3-Flash text execution in the RSInfer worker."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from core.contracts.model import Assembly
from core.contracts.records import PrefixCopy
from models.glm53.weights import EMBED, HEAD, NORM, ROT, Weights
from models.glm53_config import KDA, KV, LINEAR, Config

SERVICE = "layers.glm53@1"


class Glm53:
    def __init__(self, cfg: Config, page: int, services: Mapping[str, Any], *, draft_tokens: int = 0,
                 rotate: bool = False) -> None:
        self.cfg = cfg
        self.group = services["comm.group@1"]
        self.layers = services[SERVICE]
        self.weights, self.holder = services["state.weights@1"], services["state.holder@1"]
        self.kv, self.kda = services["state.kv.paged@1"], services["state.gdn.slots@1"]
        self.external = services["state.glm53_cache@1"]
        self.draft_tokens, self.rotate = draft_tokens, rotate
        self.draft_layer, self.hidden_size, self.dtype = len(cfg.attention), cfg.hidden, "bf16"
        self.weight_plan = Weights(cfg, self.group.size, self.group.rank, dfc=self.layers.dfc,
                                   draft=bool(draft_tokens), rotate=rotate,
                                   shared_expert_dp=self.layers.shared_expert_dp,
                                   refresh_projections=self.layers.refresh_projections)
        states = cfg.states(self.group.size, page, draft_tokens)
        self.assembly = Assembly([SERVICE] * len(states), states, self.weight_plan.decls)
        self.stores = {KV: self.kv, KDA: self.kda}
        self.backends = {SERVICE: self.layers}

    def start(self) -> None:
        self.weights.declare(self.assembly.weights, allocate=self.weight_plan.allocate)
        tensors: Mapping[str, torch.Tensor] = self.weights.bind("").buffers()
        self.holder.size(self.assembly.states, sum(t.numel() * t.element_size() for t in tensors.values()))
        for kind, store in self.stores.items():
            declarations = [d for d in self.assembly.states if d.kind == kind]
            if declarations:
                store.declare(declarations)
        self.weights.load()
        source = Path(__file__)
        model_code = hashlib.sha256(source.read_bytes() + source.with_name("weights.py").read_bytes()
                                    + bytes((self.draft_tokens, self.rotate))).hexdigest()
        self.external.bind(self.assembly.states, self.cfg, model_code, self.layers.cache_version())
        self.top = {name: tensors[name] for name in (EMBED, NORM, HEAD, ROT) if name in tensors}
        self.parameters: list[tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor],
                                    Mapping[str, torch.Tensor]]] = [
            (self.weights.bind(p).buffers(), self.weights.bind(p + "self_attn.").buffers(),
             self.weights.bind(p + "mlp.").buffers())
            for p in (f"layers.{i}." for i in range(len(self.assembly.layers)))]
        self.cache: list[Mapping[str, torch.Tensor]] = [
            self.stores[decl.kind].bind(decl.layer).buffers() for decl in self.assembly.states]

    def backbone(self, input_ids: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, int]:
        cfg = self.cfg
        sharded = self.layers.shard_prefill
        rows = len(input_ids)
        if sharded:
            start, width = self.layers.token_slice(rows)
            ids = input_ids[start:start + width]
            x = F.pad(F.embedding(ids, self.top[EMBED]), (0, 0, 0, width - len(ids)))
        else:
            x = F.embedding(input_ids, self.top[EMBED])
        x = x[:, None].expand(-1, cfg.streams, -1).contiguous()
        for i, (attention, mlp) in enumerate(zip(cfg.attention, cfg.mlp, strict=True)):
            weights, a, f = self.parameters[i]
            state = self.cache[i]
            h, post, mix = self.layers.pre(x, weights, "attn_hc", "input_layernorm.weight")
            if sharded:
                h = self.layers.gather_tokens(h, rows)
            if attention == LINEAR:
                h = self.layers.linear_attention(i, h, a, state, sharded=sharded)
            else:
                h = self.layers.sparse_attention(h, positions, a, state, sharded=sharded)
            x = self.layers.post(h, x, post, mix)
            h, post, mix = self.layers.pre(x, weights, "ffn_hc", "post_attention_layernorm.weight")
            if mlp == "dense":
                if sharded:
                    h = self.layers.gather_tokens(h, rows)
                h = self.layers.reduce_tokens(self.layers.dense(h, f), sharded)
            else:
                h = self.layers.moe(h, f, sharded_rows=rows if sharded else 0)
            x = self.layers.post(h, x, post, mix)
        return x, rows if sharded else 0

    def copy_prefix(self, command: PrefixCopy, ready: bool) -> bool:
        """Broadcast the cached pages and recurrent seed between equal TP shards."""
        domain = self.group.world_rank // self.group.size
        send, install = domain == command.source_domain, domain == command.destination_domain
        sizes = {kind: store.capacity() for kind, store in self.stores.items()}
        valid = all(kind in sizes and count > 0 and count <= sizes[kind] for kind, count in command.counts.items())
        valid = valid and ({kind: len(ids) for kind, ids in command.items.items()} == command.counts
                           if send or install else not command.items)
        valid = valid and all(len(ids) == len(set(ids)) and all(0 <= i < sizes[kind] for i in ids)
                              for kind, ids in command.items.items())
        if not self.group.prefix_ready(ready and valid):
            return False
        device = self.top[EMBED].device
        indices = {kind: torch.tensor(ids, device=device, dtype=torch.int64).add_(1)
                   for kind, ids in command.items.items()}
        for declaration, state in zip(self.assembly.states, self.cache, strict=True):
            kind = declaration.kind
            if not (count := command.counts.get(kind, 0)):
                continue
            for name, tensor in sorted(state.items()):
                if kind == KV:
                    entries = declaration.tensors[name][0][0]
                    tensor = tensor.view(-1, entries, *tensor.shape[1:])
                data = (tensor.index_select(0, indices[kind]) if send
                        else tensor.new_empty((count, *tensor.shape[1:])))
                self.group.broadcast_prefix(data, command.source_domain)
                if install:
                    tensor.index_copy_(0, indices[kind], data)
        return True

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, last_index: torch.Tensor) -> torch.Tensor:
        x, rows = self.backbone(input_ids, positions)
        if not last_index.numel():
            return x.new_empty((0, self.cfg.vocab))
        h = (self.layers.gather_tokens(x.mean(dim=1), rows).index_select(0, last_index) if rows
             else x.index_select(0, last_index).mean(dim=1))
        return F.linear(self.layers.rms(h, self.top[NORM]), self.top[HEAD])

    def forward_hidden(self, input_ids: torch.Tensor, positions: torch.Tensor, last_index: torch.Tensor
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        x, rows = self.backbone(input_ids, positions)
        h = x.mean(dim=1)
        if rows:
            h = self.layers.gather_tokens(h, rows)
        hidden = self.layers.rms(h, self.top[NORM])
        return F.linear(hidden.index_select(0, last_index), self.top[HEAD]), hidden

    def draft_forward(self, input_ids: torch.Tensor, positions: torch.Tensor, hidden: torch.Tensor,
                      last_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w, a, f = self.parameters[self.draft_layer]
        emb = F.embedding(input_ids, w[EMBED])
        if self.rotate:
            hidden = F.linear(hidden, self.top[ROT])
        if self.layers.fused_eh_norm_fn is not None:
            joined = self.layers.fused_eh_norm_fn(
                positions, emb, hidden, w["enorm.weight"], w["hnorm.weight"], self.cfg.eps)
        else:
            emb = emb.masked_fill(positions[:, None] == 0, 0)
            joined = torch.cat((self.layers.rms(emb, w["enorm.weight"]), self.layers.rms(hidden, w["hnorm.weight"])),
                               dim=-1)
        residual = F.linear(joined, w["eh_proj.weight"])
        h = self.layers.rms(residual, w["input_layernorm.weight"])
        h = self.layers.sparse_attention(h, positions, a, self.cache[self.draft_layer])
        h, residual = self.layers.add_rms(h, residual, w["post_attention_layernorm.weight"])
        h = self.layers.moe(h, f)
        h, _ = self.layers.add_rms(h.index_select(0, last_index), residual.index_select(0, last_index),
                                   w["shared_head.norm.weight"])
        head = w["shared_head.head.weight"]
        if self.layers.local_draft_argmax and self.group.size > 1 and self.cfg.vocab % self.group.size == 0:
            shard = self.cfg.vocab // self.group.size
            v_start = self.group.rank * shard
            vals, idx = F.linear(h, head[v_start:v_start + shard]).max(dim=-1)
            local_pair = torch.stack((vals.float(), (idx + v_start).float()), dim=-1).contiguous()
            gathered = local_pair.new_empty((self.group.size, local_pair.shape[0], 2))
            dist.all_gather_into_tensor(gathered, local_pair, group=self.group.device_group)
            best_rank = gathered[:, :, 0].argmax(dim=0, keepdim=True)
            return gathered[:, :, 1].gather(0, best_rank).squeeze(0).to(torch.int64), h
        return F.linear(h, head), h


class Unit:
    SERVICES = ("comm.group@1", SERVICE, "state.weights@1", "state.holder@1", "state.kv.paged@1",
                "state.gdn.slots@1", "state.glm53_cache@1")

    def __init__(self, ctx: Any, config: Any) -> None:
        if config["model.dtype"] != "bf16" or config["state.kv.dtype"] not in ("auto", "bf16"):
            raise ValueError("GLM-5.3-Flash uses BF16 activations and latent KV")
        if config["comm.domains"] != config["sched.domains"]:
            raise ValueError("GLM-5.3-Flash needs one communication group per scheduling domain")
        cfg = Config.load(config["model.path"], config["model.hf_config"])
        if not 0 < config["batch.max_context_len"] <= cfg.max_context:
            raise ValueError(f"batch.max_context_len must be in 1..{cfg.max_context}")
        steps = config["spec.num_draft_tokens"] if config["spec.algorithm"] else 0
        quant = Path(config["model.path"]) / "quant_model_description.json"
        rotate = bool(steps and json.loads(quant.read_text()).get("is_rot_used"))
        model = Glm53(cfg, config["state.kv.page_size"], {name: ctx.use(name) for name in self.SERVICES},
                      draft_tokens=steps, rotate=rotate)
        ctx.register("state.weights@1", model.weight_plan.load_weights, label="load_weights")
        model.start()
        ctx.provide("model@1", model)
