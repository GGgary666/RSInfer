# Copyright (c) 2023 DeepSeek
# SPDX-License-Identifier: MIT
"""``layers.ds1017``: DeepSeek-V4.1's layers (``layers.ds1017@1``, doc/specs/v1-ds1017.md), after the
reference implementation shipped with the checkpoint (``inference/model.py`` of
deepseek-ai/DeepSeek-V4.1-Flash-1017; LICENSE.deepseek in ``kernels/ds1017``): ``Attention`` with its
``Compressor`` and ``Indexer``, ``Engram`` with ``NgramHashState``, ``MoE``, and the Hyper-Connections
of ``Block``. The kernels are the catalog's (``ds.*``), fixed per phase at start.

What moved. That implementation keeps one cache row per batch row, all rows at the same position,
prefilled in one piece from position 0. Here the rows are the persistent batch's, each at its own
position, a step runs any number of tokens of each, and the caches are the stores':

- The window KV is a ring per request and layer (``state.swa.slots@1``, ``ring``): position ``p`` at
  ``p % window``.
- Compressed latents and the indexer's keys are paged (``state.kv.paged@1``, ``kv`` and ``ik`` of the
  source layer): the latent of tokens ``[j * ratio, (j + 1) * ratio)`` at the token slot of position
  ``j * ratio``.
- A group still filling (``ckv``, ``cscore``) and the tokens an n-gram reaches back to (``tail``) are
  in the request's slot.

A step's keys are taken into one tensor per layer, ``[tokens, keys, dim]``, and the attention kernel
attends over that; the order of a token's keys is that implementation's (the window oldest first,
then the chosen latents by position), which its kernel's result depends on in the last bit.

Two ways to lay a step out (``prepare``):

- A decode step (one token per row) derives everything on the device from the rows' table indices
  and positions, with fixed shapes: a graph records it. A graph bucket scores the indexer over
  ``batch.max_context_len`` positions, an eager step over the longest row's.
- Any other step is laid out on the host (``Layout``): rows of any length from any position, a
  compression group completed by tokens of an earlier step, the indexer row by row.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from core.contracts.kernels import KernelFn
from core.contracts.plan import GRAPH, PersistentBatch, PlanKey
from core.contracts.records import BatchPlan
from core.contracts.state import StateDecl
from kernels import ds1017_formats as formats
from layers.solutions import Kernels
from models.ds1017_config import SLOT_KIND, DSConfig, compressed_token_map, engram_layout

DEFINITIONS = ("ds.fake_quant_fp8@1", "ds.fake_quant_fp4@1", "ds.linear_fp8@1", "ds.linear@1", "ds.grouped_linear@1",
               "ds.index_score@1", "ds.rms_norm@1", "ds.rope@1", "ds.sparse_attn@1", "ds.hc_mixes@1", "ds.hc_pre@1",
               "ds.hc_post@1", "ds.moe_gate@1", "ds.moe_experts@1", "ds.pool@1", "ds.engram_gate@1", "ds.silu@1",
               "ds.top@1")

KEY_BYTES = 192 << 20
"""A step's attention runs this many bytes of keys at a time (``[tokens, keys, dim]`` in bf16)."""

Weights = Mapping[str, torch.Tensor]


def on_device(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Host ``values`` on ``device`` without waiting for the work queued there (pinned, non-blocking)."""
    if device.type != "cuda":
        return values.to(device)
    return values.pin_memory().to(device, non_blocking=True)


@dataclass(frozen=True)
class Group:
    """One compression ratio above 1 in a step laid out on the host."""

    members: torch.Tensor
    """``[groups, ratio]``: each completed group's members, as token indices of the step."""
    from_state: torch.Tensor
    """``[groups, ratio]`` bool: the member is a token of an earlier step, in the request's slot."""
    token: torch.Tensor
    """``[groups]``: a token of the group's row (for its slot and its row)."""
    first: torch.Tensor
    """``[groups]``: the position of the group's first token: where its latent is kept and rotated."""
    left: torch.Tensor
    """Token indices of the tokens whose group this step does not complete: into the slot."""


@dataclass(frozen=True)
class Layout:
    """A step laid out on the host: index tensors on the device, the same for every layer."""

    rows: torch.Tensor
    """``[tokens]``: each token's row in the batch's tables."""
    start: torch.Tensor
    """``[tokens]``: the position its row's tokens of this step begin at."""
    window: torch.Tensor
    """``[tokens, width]``: each token's window key positions, oldest first; ``valid`` says which exist."""
    valid: torch.Tensor
    chunk: torch.Tensor
    """``[tokens, width]``: the token index of a window key computed in this step; ``in_chunk`` says which."""
    in_chunk: torch.Tensor
    keep: torch.Tensor
    """Token indices of the tokens still in their row's window after the step: into the ring."""
    groups: Mapping[int, Group]
    spans: Sequence[tuple[int, int, int, int]]
    """Per row: its first token index and one past its last, its table row, its position after the step."""
    ngram: torch.Tensor
    """``[tokens, n]``: for look-back ``s``, the token index holding position ``p - s`` (``ngram_from``
    0), or the index in the row's tail (1), or nothing before the sequence starts (2)."""
    ngram_from: torch.Tensor
    last: torch.Tensor
    """``[rows]``: each row's last token index."""
    tail: torch.Tensor
    """``[rows, n - 1]``: the same for the tail a row keeps after the step, most recent token first."""
    tail_from: torch.Tensor


def candidate_blocks(scores: torch.Tensor, lens: torch.Tensor, topk_blocks: int, block: int, top: KernelFn
                     ) -> torch.Tensor:
    """``select_candidate_blocks``: level one of the two-level top-k. ``scores`` ``[..., latents]`` with
    what a query cannot reach at -inf, ``lens`` ``[..., 1]`` how many it reaches; a bool mask shaped
    like ``scores`` of the ``topk_blocks`` best blocks (``top`` picks them), the one with the newest
    latent always kept."""
    width = scores.size(-1)
    per_block = F.pad(scores, (0, -width % block), value=-torch.inf).unflatten(-1, (-1, block)).amax(dim=-1)
    count = per_block.size(-1)
    newest = (lens - 1) // block
    per_block = per_block.masked_fill(torch.arange(count, device=scores.device) == newest, torch.inf)
    best = top(per_block, min(topk_blocks, count))
    keep = torch.zeros_like(per_block, dtype=torch.bool).scatter_(-1, best, per_block.gather(-1, best) > -torch.inf)
    return keep.repeat_interleave(block, dim=-1)[..., :width]


class Layers:
    """``layers.ds1017@1``."""

    def __init__(self, catalog: Any, config: Any, cfg: DSConfig, kv: Any, slots: Any, token_map: Sequence[int] | None,
                 max_context_len: int) -> None:
        self.cfg, self.kv, self.slots = cfg, kv, slots
        self.kernels = Kernels(catalog, config, "dsv4")
        self.table = "prefill"
        """Which of the fixed solutions the forward runs: a phase, or ``GRAPH`` for a bucket."""
        self.max_context_len = max_context_len
        self.engram_first = min(cfg.engram_layer_ids) if cfg.engram_layer_ids else -1
        self.token_map_host = list(token_map) if token_map is not None else None
        self.constants: dict[str, torch.Tensor] = {}
        self.fixed: dict[int, torch.Tensor] = {}
        """Bucket -> its rows' table indices, the one buffer a graph of that bucket reads."""
        self.generation = 0
        # The step in progress.
        self.batch_tables: Mapping[str, torch.Tensor] = {}
        self.layout: Layout | None = None
        self.decode_rows: torch.Tensor | None = None
        self.extent: dict[int, int] = {}
        """Ratio -> how many latents a decode step's indexer scores (the longest row's, or the bucket's)."""
        self.step: dict[Any, Any] = {}
        """What ``begin`` derives for the forward, and what a source layer hands to the layers after it."""

    # ------------------------------------------------------------------ Backend

    @property
    def binding(self) -> Layers:
        return self

    def buffers(self) -> Mapping[str, torch.Tensor]:
        return {f"rows.{size}": t for size, t in self.fixed.items()}

    def release_graphs(self) -> None:
        self.generation += bool(self.fixed)
        self.fixed.clear()

    def resolve(self, states: Sequence[StateDecl]) -> Mapping[str, Mapping[str, str]]:
        # A named backend solves what it has a kernel for; the rest runs as ``auto`` would.
        shape = {"heads": self.cfg.num_attention_heads, "head_dim": self.cfg.head_dim}
        return self.kernels.resolve({phase: {name: shape for name in DEFINITIONS} for phase in ("prefill", "decode")},
                                    shared=frozenset(DEFINITIONS))

    def k(self, definition: str) -> KernelFn:
        return self.kernels(self.table, definition)

    def prepare(self, batch: PersistentBatch, plan: BatchPlan, key: PlanKey) -> None:
        if plan.draft_tokens:
            raise ValueError("layers.ds1017 runs no speculative step yet")
        self.table = GRAPH if key.bucket else key.phase
        self.batch_tables = batch.binding.buffers()
        device = self.batch_tables["req_to_token"].device
        rows = [int(r) + 1 for r in plan.rows.values()]
        counts = [int(n) for n in plan.num_tokens.values()]
        starts = [int(p) for p in plan.positions.values()]
        self.step = {}
        if key.phase == "decode":
            self.layout = None
            given = on_device(torch.tensor(rows, dtype=torch.int64), device)
            if key.bucket:
                buf = self.fixed.get(key.bucket)
                if buf is None:
                    buf = self.fixed[key.bucket] = torch.zeros(key.bucket, dtype=torch.int64, device=device)
                    self.generation += 1
                buf[:len(rows)].copy_(given)
                buf[len(rows):].zero_()
                given = buf
            self.decode_rows = given
            ratios = {r for r in self.cfg.compress_ratios[:self.cfg.num_hidden_layers] if r}
            self.extent = {r: self.max_context_len // r if key.bucket else max((p + 1) // r for p in starts)
                           for r in ratios}
        else:
            self.decode_rows = None
            # In a mixed step, a row that goes on from the token it sampled is decoding.
            decoding = [bool(f) and n == 1 for f, n in zip(plan.from_previous.values(), counts, strict=True)]
            self.layout = self.lay_out(rows, counts, starts, decoding, device)

    # ------------------------------------------------------------------ a step laid out on the host

    def lay_out(self, rows: Sequence[int], counts: Sequence[int], starts: Sequence[int], decoding: Sequence[bool],
                device: torch.device) -> Layout:
        cfg, w = self.cfg, self.cfg.sliding_window
        n = torch.tensor(counts)
        p0 = torch.tensor(starts)
        base = torch.cumsum(n, 0) - n
        tok_row, tok_p0, tok_base = (torch.repeat_interleave(torch.tensor(v) if isinstance(v, list) else v, n)
                                     for v in (list(rows), p0, base))
        tok_n = torch.repeat_interleave(n, n)
        pos = torch.cat([torch.arange(p, p + c) for p, c in zip(starts, counts, strict=True)])
        end = tok_p0 + tok_n - 1  # the row's last position of this step

        # The window, ``w`` columns. A prefilling row lists a token's keys as a prefill of the reference
        # does: ascending from max(0, p - w + 1), those that exist first. A decoding row lists them
        # as its decode does: the w positions up to p, those before 0 absent first. From position
        # w - 1 on the two are the same list; before it they hold the same keys at other columns,
        # which the attention kernel's blocks of 64 see. A chunk of a prompt lists them as the whole
        # prompt would, wherever the chunk starts and ends. (For a prompt shorter than the window
        # the reference lists only as many columns as the prompt has tokens, so its compressed
        # keys follow in earlier columns than here: the same keys, another last bit.)
        ahead = ~torch.repeat_interleave(torch.tensor(list(decoding), dtype=torch.bool), n)
        col = torch.arange(w)
        keys = torch.where(ahead[:, None], (pos - w + 1).clamp_min(0)[:, None] + col, pos[:, None] - w + 1 + col)
        valid = (keys >= 0) & (keys <= pos[:, None])
        in_chunk = keys >= tok_p0[:, None]
        chunk = (tok_base[:, None] + keys - tok_p0[:, None]).clamp(0, int(n.sum()) - 1)
        keep = torch.nonzero(pos > end - w).flatten()

        groups: dict[int, Group] = {}
        for ratio in sorted({r for r in cfg.compress_ratios[:cfg.num_hidden_layers] if r > 1}):
            members, from_state, token, first = [], [], [], []
            for b, c, p in zip(base.tolist(), counts, starts, strict=True):
                for g in range(p // ratio, (p + c) // ratio):  # groups whose last member is in this step
                    at = torch.arange(g * ratio, (g + 1) * ratio)
                    earlier = at < p
                    members.append(torch.where(earlier, at % ratio, b + at - p))
                    from_state.append(earlier)
                    token.append(b)
                    first.append(g * ratio)
            left = torch.nonzero(pos >= ((end + 1) // ratio) * ratio).flatten()
            shape = (0, ratio)
            groups[ratio] = Group(
                on_device(torch.stack(members) if members else torch.zeros(shape, dtype=torch.int64), device),
                on_device(torch.stack(from_state) if from_state else torch.zeros(shape, dtype=torch.bool), device),
                on_device(torch.tensor(token, dtype=torch.int64), device),
                on_device(torch.tensor(first, dtype=torch.int64), device), on_device(left, device))

        # Engram: an n-gram ending at p looks back to p - s; the row's tail holds what earlier steps ran.
        ng = max(cfg.engram_max_ngram_size, 1)
        back = pos[:, None] - torch.arange(ng)
        ngram_from = torch.where(back < 0, 2, torch.where(back >= tok_p0[:, None], 0, 1))
        ngram = torch.where(ngram_from == 0, tok_base[:, None] + back - tok_p0[:, None], tok_p0[:, None] - 1 - back)
        row_end = p0 + n - 1
        after = row_end[:, None] - torch.arange(max(ng - 1, 1))
        tail_from = torch.where(after < 0, 2, torch.where(after >= p0[:, None], 0, 1))
        tail = torch.where(tail_from == 0, base[:, None] + after - p0[:, None], p0[:, None] - 1 - after)

        spans = [(int(b), int(b) + c, r, p + c) for b, c, r, p in zip(base.tolist(), counts, rows, starts, strict=True)]

        def dev(t: torch.Tensor) -> torch.Tensor:
            return on_device(t, device)

        return Layout(rows=dev(tok_row), start=dev(tok_p0), window=dev(keys), valid=dev(valid), chunk=dev(chunk),
                      in_chunk=dev(in_chunk), keep=dev(keep), groups=groups, spans=spans, ngram=dev(ngram.clamp_min(0)),
                      ngram_from=dev(ngram_from), last=dev(base + n - 1), tail=dev(tail.clamp_min(0)),
                      tail_from=dev(tail_from))

    # ------------------------------------------------------------------ constants

    def constant(self, name: str, device: torch.device) -> torch.Tensor:
        key = f"{name}@{device}"
        if key not in self.constants:
            cfg = self.cfg
            if name in ("rope.window", "rope.compress"):
                compress = name == "rope.compress"
                value = formats.rope_table(
                    cfg.qk_rope_head_dim, self.max_context_len, cfg.rope_original_seq_len if compress else 0,
                    cfg.compress_rope_theta if compress else cfg.rope_theta, cfg.rope_factor, cfg.rope_beta_fast,
                    cfg.rope_beta_slow, device)
            elif name == "token_map":
                if self.token_map_host is None:
                    raise RuntimeError("Engram needs the compressed token map: a tokenizer.json or an "
                                       "engram_token_map.json in the model directory")
                value = torch.tensor(self.token_map_host, dtype=torch.int64, device=device)
            else:
                primes, offsets, multipliers = engram_layout(cfg)
                value = torch.tensor({"engram.primes": primes, "engram.offsets": offsets,
                                      "engram.multipliers": multipliers}[name], dtype=torch.int64, device=device)
            self.constants[key] = value
        return self.constants[key]

    def freqs(self, layer: int, positions: torch.Tensor) -> torch.Tensor:
        table = self.constant("rope.compress" if self.cfg.ratio(layer) else "rope.window", positions.device)
        return table[positions]

    # ------------------------------------------------------------------ the forward

    def begin(self, input_ids: torch.Tensor, positions: torch.Tensor) -> None:
        """Once per forward, before the layers: what every layer shares, and the Engram hash ids."""
        tables = self.batch_tables
        slot_table = tables[f"slots.{SLOT_KIND}"]
        rows = self.layout.rows if self.layout is not None else self.decode_rows
        assert rows is not None
        self.step = {"rows": rows, "pos": positions, "slot": slot_table[rows].long(),
                     "tok_slot": tables["req_to_token"][rows, positions].long()}
        if self.cfg.engram_layer_ids:
            self.step["hashes"] = self._hashes(input_ids, positions)

    def _hashes(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """``NgramHashState.forward``: ``[tokens, engram layers, hash cols]`` rows of the layers' tables."""
        cfg, device = self.cfg, input_ids.device
        compressed = self.constant("token_map", device)[input_ids]
        pad = self.constant("token_map", device)[cfg.engram_pad_token_id]
        tail = self.slots.bind(self.engram_first).buffers()["tail"]
        slot = self.step["slot"]
        ng = cfg.engram_max_ngram_size
        if self.layout is None:  # one token per row: every look-back is in the tail
            tokens = torch.cat([compressed[:, None], tail[slot]], dim=1)
            tokens = torch.where(positions[:, None] < torch.arange(ng, device=device), pad, tokens)
            tail[slot] = tokens[:, :ng - 1]  # the padding of a short sequence is never read back
        else:
            lay = self.layout
            earlier = tail[slot[:, None], lay.ngram.clamp_max(ng - 2)]
            tokens = torch.where(lay.ngram_from == 0, compressed[lay.ngram.clamp_max(compressed.shape[0] - 1)],
                                 torch.where(lay.ngram_from == 1, earlier, pad))
            row_slot = slot[lay.last]
            kept = tail[row_slot[:, None], lay.tail.clamp_max(ng - 2)]
            tail[row_slot] = torch.where(lay.tail_from == 0, compressed[lay.tail.clamp_max(compressed.shape[0] - 1)],
                                         torch.where(lay.tail_from == 1, kept, pad))
        products = tokens.unsqueeze(1) * self.constant("engram.multipliers", device)
        rolling = products[..., 0]
        primes = self.constant("engram.primes", device)
        hashes: list[torch.Tensor] = []
        for i in range(1, ng):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.constant("engram.offsets", device)

    def linear_fp8(self, x: torch.Tensor, w: Weights, name: str) -> torch.Tensor:
        return self.k("ds.linear_fp8@1")(x, w[name + ".weight"], w[name + ".scale"], w[name + ".packed"])

    def linear(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return self.k("ds.linear@1")(x, weight)

    def norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return self.k("ds.rms_norm@1")(x, weight, self.cfg.rms_norm_eps)

    def hc_mixes(self, x: torch.Tensor, w: Weights, which: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        return self.k("ds.hc_mixes@1")(x, w[f"hc_{which}_fn"], w[f"hc_{which}_scale"], w[f"hc_{which}_base"],
                                        cfg.rms_norm_eps, cfg.hc_sinkhorn_iters, cfg.hc_eps)

    def hc_pre(self, x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        return self.k("ds.hc_pre@1")(x, pre)

    def hc_post(self, y: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        return self.k("ds.hc_post@1")(y, residual, post, comb)

    def engram(self, layer: int, x: torch.Tensor, w: Weights) -> torch.Tensor:
        """``Engram.forward`` on the stream ``[tokens, hc, dim]``: the layer's table rows at the hash ids,
        dequantized, projected to one key per copy and a shared value, added through a gate."""
        cfg = self.cfg
        ids = self.step["hashes"][:, cfg.engram_layer_ids.index(layer)]
        rows = formats.dequant_rows_fp8(w["engram.embed.weight"][ids], w["engram.embed.scale"][ids]).to(x.dtype)
        kv = self.linear_fp8(rows.flatten(-2), w, "engram.wkv")
        return self.k("ds.engram_gate@1")(x, kv, w["engram.q_weight"], w["engram.k_weight"], cfg.rms_norm_eps)

    def moe(self, x: torch.Tensor, w: Weights) -> torch.Tensor:
        """``MoE.forward``: the routed experts' weighted sum and the shared expert."""
        cfg = self.cfg
        weights, indices = self.k("ds.moe_gate@1")(x, w["ffn.gate.weight"], w["ffn.gate.bias"], cfg.num_experts_per_tok,
                                                    cfg.routed_scaling_factor)
        y = self.k("ds.moe_experts@1")(x, weights, indices, w["ffn.experts.w13.weight"], w["ffn.experts.w13.packed"],
                                       w["ffn.experts.w2.weight"], w["ffn.experts.w2.packed"], cfg.swiglu_limit)
        gate = self.linear_fp8(x, w, "ffn.shared_experts.w1").float()
        up = self.linear_fp8(x, w, "ffn.shared_experts.w3").float()
        if cfg.swiglu_limit > 0:
            up = up.clamp(-cfg.swiglu_limit, cfg.swiglu_limit)
            gate = gate.clamp(max=cfg.swiglu_limit)
        y = y + self.linear_fp8((self.k("ds.silu@1")(gate) * up).to(x.dtype), w, "ffn.shared_experts.w2")
        return y.to(x.dtype)

    # ------------------------------------------------------------------ attention

    def attention(self, layer: int, x: torch.Tensor, w: Weights) -> torch.Tensor:
        """``Attention.forward`` on a layer's normed input ``[tokens, dim]``."""
        cfg, step = self.cfg, self.step
        heads, dim, rd = cfg.num_attention_heads, cfg.head_dim, cfg.qk_rope_head_dim
        pos, slot = step["pos"], step["slot"]
        freqs = self.freqs(layer, pos)
        rope, quant8 = self.k("ds.rope@1"), self.k("ds.fake_quant_fp8@1")

        qr = self.norm(self.linear_fp8(x, w, "attn.wq_a"), w["attn.q_norm.weight"])
        q = rope(self.linear_fp8(qr, w, "attn.wq_b").unflatten(-1, (heads, dim)), freqs, rd)
        kv = quant8(rope(self.norm(self.linear_fp8(x, w, "attn.wkv"), w["attn.kv_norm.weight"]), freqs, rd), 32)

        ring = self.slots.bind(layer).buffers()["ring"]
        win = cfg.sliding_window
        if self.layout is None:  # the ring's positions up to p, oldest first; p itself is this token
            at = pos[:, None] - win + 1 + torch.arange(win, device=pos.device)
            keys = ring[slot[:, None], at % win]
            keys[:, -1] = kv
            valid = at >= 0
            ring[slot, pos % win] = kv
        else:
            lay = self.layout
            keys = torch.where(lay.in_chunk[..., None], kv[lay.chunk], ring[slot[:, None], lay.window % win])
            valid = lay.valid
            ring[slot[lay.keep], pos[lay.keep] % win] = kv[lay.keep]

        if cfg.ratio(layer):
            source, taken, reached = self._compressed(layer, x, qr, w)
            keys = torch.cat([keys, source[taken]], dim=1)
            valid = torch.cat([valid, reached], dim=1)

        attend = self.k("ds.sparse_attn@1")
        tokens, count = valid.shape
        idx = torch.where(valid, torch.arange(count, device=valid.device), -1)
        per = max(1, KEY_BYTES // (count * dim * 2))
        parts = [attend(q[a:a + per], keys[a:a + per].reshape(-1, dim),
                        torch.where(idx[a:a + per] >= 0, idx[a:a + per]
                                    + count * torch.arange(min(per, tokens - a), device=idx.device)[:, None], -1),
                        w["attn.attn_sink"], dim ** -0.5) for a in range(0, tokens, per)]
        o = rope(parts[0] if len(parts) == 1 else torch.cat(parts), freqs, rd, True)

        # wo_a projects each group's heads on their own, on o quantized.
        groups = cfg.o_groups
        o = quant8(o.reshape(tokens, groups, -1), 32)
        o = self.k("ds.grouped_linear@1")(o, w["attn.wo_a.weight"].view(groups, cfg.o_lora_rank, -1))
        return self.linear_fp8(o.flatten(1), w, "attn.wo_b")

    def _compressed(self, layer: int, x: torch.Tensor, qr: torch.Tensor, w: Weights
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``Attention._compress_kv``: the source layer's latents ``[slots, dim]``, and per token the
        token slots of the latents it attends to with which of them it reaches. A source layer
        compresses first; an index source chooses, and the layers after it take its choice."""
        cfg, step = self.cfg, self.step
        ratio, source = cfg.ratio(layer), cfg.kv_source(layer)
        pool = self.kv.bind(source).buffers()
        if layer == source:
            self._compress(layer, x, w, pool)
        if layer == cfg.index_source(layer):
            step["chosen", ratio] = self._index(layer, x, qr, w, pool["ik"])
        taken, reached = step["chosen", ratio]
        return pool["kv"], taken, reached

    def _compress(self, layer: int, x: torch.Tensor, w: Weights, pool: Weights) -> None:
        """``Compressor.forward`` and what follows a completed group: its latent rotated and quantized
        into ``kv``, and the indexer's key of it into ``ik``, at the slot of the group's first token."""
        cfg, step = self.cfg, self.step
        ratio, rd = cfg.ratio(layer), cfg.qk_rope_head_dim
        pos, slot, rows = step["pos"], step["slot"], step["rows"]
        table = self.batch_tables["req_to_token"]
        linear = self.k("ds.linear@1")
        if ratio == 1:
            latent = self.norm(linear(x, w["attn.compressor.wkv.weight"]), w["attn.compressor.norm.weight"])
            dest, first = step["tok_slot"], pos
        else:
            state = self.slots.bind(layer).buffers()
            xf = x.float()
            kv, score = linear(xf, w["attn.compressor.wkv.weight"]), linear(xf, w["attn.compressor.wgate.weight"])
            if self.layout is None:
                at = pos % ratio
                state["ckv"][slot, at], state["cscore"][slot, at] = kv, score
                members, scores = state["ckv"][slot], state["cscore"][slot]
                first = pos + 1 - ratio
                # A group still filling yields nothing: its latent goes to the padding slot.
                dest = torch.where((pos + 1) % ratio == 0, table[rows, first.clamp_min(0)].long(), 0)
            else:
                g = self.layout.groups[ratio]
                group_slot = slot[g.token]
                members = torch.where(g.from_state[..., None], state["ckv"][group_slot[:, None], g.members.clamp_max(ratio - 1)],
                                      kv[g.members.clamp_max(kv.shape[0] - 1)])
                scores = torch.where(g.from_state[..., None],
                                     state["cscore"][group_slot[:, None], g.members.clamp_max(ratio - 1)],
                                     score[g.members.clamp_max(kv.shape[0] - 1)])
                state["ckv"][slot[g.left], pos[g.left] % ratio] = kv[g.left]
                state["cscore"][slot[g.left], pos[g.left] % ratio] = score[g.left]
                first = g.first
                if not first.numel():  # no group completed in this step
                    return
                dest = table[rows[g.token], first].long()
            pooled = self.k("ds.pool@1")(members, scores)
            latent = self.norm(pooled.to(x.dtype), w["attn.compressor.norm.weight"])
        freqs = self.freqs(layer, first.clamp_min(0))
        rope, quant4 = self.k("ds.rope@1"), self.k("ds.fake_quant_fp4@1")
        # The indexer's key comes from the latent before it is rotated: before the cache takes it.
        key = self.norm(linear(latent, w["attn.indexer.wk.weight"]), w["attn.indexer.k_norm.weight"])
        pool["ik"][dest] = quant4(rope(key, freqs, rd), 32, False)
        pool["kv"][dest] = quant4(rope(latent, freqs, rd), 16, True)

    def _index(self, layer: int, x: torch.Tensor, qr: torch.Tensor, w: Weights, keys: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """``Indexer.forward``: per token the ``index_topk`` best latents of its row, in position order,
        as token slots, and which of them the token reaches."""
        cfg, step = self.cfg, self.step
        ratio, rd, h, d = cfg.ratio(layer), cfg.qk_rope_head_dim, cfg.index_n_heads, cfg.index_head_dim
        pos, rows = step["pos"], step["rows"]
        table = self.batch_tables["req_to_token"]
        q = self.k("ds.rope@1")(self.linear_fp8(qr, w, "attn.indexer.wq_b").unflatten(-1, (h, d)),
                               self.freqs(layer, pos), rd)
        q = self.k("ds.fake_quant_fp4@1")(q, 32, False)
        weights = self.k("ds.linear@1")(x, w["attn.indexer.weights_proj.weight"]) * (d ** -0.5 * h ** -0.5)
        lens = ((pos + 1) // ratio)[:, None]
        source, uses = layer == cfg.candidate_source_layer_id, 0 <= cfg.candidate_source_layer_id < layer

        def choose(q: torch.Tensor, weights: torch.Tensor, lens: torch.Tensor, slots: torch.Tensor, which: Any
                   ) -> tuple[torch.Tensor, torch.Tensor]:
            """One set of queries against the ``slots`` ``[queries or 1, latents]`` of their row."""
            count = slots.shape[-1]
            score = self.k("ds.index_score@1")(q, keys[slots], weights)
            score = score.masked_fill(torch.arange(count, device=score.device) >= lens, -torch.inf)
            if source:
                step["candidates", which] = candidate_blocks(score, lens, cfg.candidate_topk_blocks,
                                                             cfg.candidate_block_size, self.k("ds.top@1"))
            elif uses:
                score = score.masked_fill(~step["candidates", which], -torch.inf)
            best = self.k("ds.top@1")(score, min(cfg.index_topk, count)).sort(dim=-1).values
            return best, best < lens

        if self.layout is None:  # every row at once, over one extent
            count = self.extent[ratio]
            slots = table[rows][:, 0:count * ratio:ratio].long()
            best, reached = choose(q[:, None], weights[:, None], lens[:, None], slots, None)
            best, reached = best[:, 0], reached[:, 0]
            return slots.gather(1, best), reached
        taken: list[torch.Tensor] = []
        ok: list[torch.Tensor] = []
        for a, b, row, after in self.layout.spans:  # each row over its own latents
            slots = table[row, 0:(after // ratio) * ratio:ratio].long()
            best, reached = choose(q[None, a:b], weights[None, a:b], lens[None, a:b], slots[None], row)
            taken.append(slots[best[0]])
            ok.append(reached[0])
        width = max(t.shape[1] for t in taken)
        return (torch.cat([F.pad(t, (0, width - t.shape[1])) for t in taken]),
                torch.cat([F.pad(t, (0, width - t.shape[1])) for t in ok]))


def token_map(path: str, cfg: DSConfig) -> list[int] | None:
    """The compressed token map Engram hashes over: ``engram_token_map.json`` of the model directory
    (a generated checkpoint has one), else built from its ``tokenizer.json`` as the reference builds it."""
    if not cfg.engram_layer_ids:
        return None
    ready = os.path.join(path, "engram_token_map.json") if path else ""
    if ready and os.path.isfile(ready):
        with open(ready) as f:
            lookup = json.load(f)
        size = max(lookup) + 1
    else:
        tokenizer = os.path.join(path, "tokenizer.json") if path else ""
        if not tokenizer or not os.path.isfile(tokenizer):
            return None
        from tokenizers import Tokenizer

        lookup, size = compressed_token_map(Tokenizer.from_file(tokenizer))
    if size != cfg.engram_compressed_vocab_size:
        raise ValueError(f"the compressed token map has {size} ids, the config says "
                         f"{cfg.engram_compressed_vocab_size}: every hash would land elsewhere")
    return lookup


def hf_config(path: str, inline: dict[str, Any]) -> dict[str, Any]:
    if path:
        with open(os.path.join(path, "config.json")) as f:
            return json.load(f)
    if not inline:
        raise ValueError("layers.ds1017 needs model.path or model.hf_config")
    return inline


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        cfg = DSConfig.from_hf(hf_config(config["model.path"], dict(config["model.hf_config"])))
        cfg.check()
        directory = config["model.path"] or config["weights.path"]
        ctx.provide("layers.ds1017@1", Layers(
            ctx.use("kernels.catalog@1"), config, cfg, ctx.use("state.kv.paged@1"), ctx.use("state.swa.slots@1"),
            token_map(directory, cfg), config["batch.max_context_len"]))

