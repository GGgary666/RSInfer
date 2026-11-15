# Copyright (c) 2023 DeepSeek
# SPDX-License-Identifier: MIT
"""``layers.ds1017``: DeepSeek-V4.1's layers (``layers.ds1017@1``, doc/specs/v1-ds1017.md), after the
reference implementation shipped with the checkpoint (``inference/model.py`` of
deepseek-ai/DeepSeek-V4.1-Flash-1017; LICENSE.deepseek in ``kernels/ds1017``): ``Attention`` with its
``Compressor`` and ``Indexer``, ``Engram`` with ``NgramHashState``, ``MoE``, and the Hyper-Connections
of ``Block``. The kernels are the catalog's (``ds.*``), fixed per phase at start.

What moved. That implementation keeps one cache row per batch row, all rows at the same position,
prefilled in one piece from position 0. Here the rows are the persistent batch's, each at its own
position, a step runs any number of tokens of each, and the caches are the stores', in DeepInfer's
byte formats (``kernels.ds1017_formats``), which its kernels read:

- The window KV is a ring per request and layer (``state.swa.slots@1``, ``ring``, one page of
  ``window`` FP8 records): position ``p`` at entry ``p % window``.
- Compressed latents and the indexer's keys are paged (``state.kv.paged@1``, ``kv`` and ``ik`` of the
  source layer), ``page_size / ratio`` a page: the latent of tokens ``[j * ratio, (j + 1) * ratio)``
  at row ``slot(j * ratio) / ratio``, ``slot`` the token slot.
- A group still filling (``cstate``: score and KV of each member) and the tokens an n-gram reaches
  back to (``tail``) are in the request's slot.

The order of a token's keys is that implementation's (the window oldest first, then the chosen
latents by position), which its attention kernel's result depends on in the last bit.

Two ways to lay a step out (``prepare``), as DeepInfer's decode and prefill:

- A decode step (one token per row) derives everything on the device from the rows' table indices
  and positions, with fixed shapes: a graph records it. It attends in the caches
  (``ds.cache_attn@1``) and the indexer reads the keys through the rows' pages
  (``ds.index_topk_paged@1``). A graph bucket scores the indexer over ``batch.max_context_len``
  positions, an eager step over the longest row's.
- Any other step is laid out on the host (``Layout``): rows of any length from any position, a
  compression group completed by tokens of an earlier step. Its layers load what they attend to out
  of the caches into one source (the window's earlier keys, the latents of its rows: load segments)
  and the indexer scores ranges of the loaded keys (``ds.index_topk@1``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from core.contracts.kernels import KernelFn
from core.contracts.plan import GRAPH, PersistentBatch, PlanKey
from core.contracts.records import BatchPlan
from core.contracts.state import StateDecl
from kernels import ds1017_formats as formats
from layers.solutions import Kernels
from models.ds1017_config import SLOT_KIND, DSConfig, compressed_token_map, engram_layout

PREFILL_ONLY = frozenset({"ds.transform@1", "ds.load_values@1", "ds.load_keys@1", "ds.index_topk@1", "ds.compress@1",
                          "ds.sparse_attn@1"})
"""What only a step laid out on the host runs."""
DECODE_ONLY = frozenset({"ds.cache_attn@1", "ds.index_topk_paged@1", "ds.compress_decode@1"})
"""What only a decode step runs: a graph records these, not the others."""

DEFINITIONS = ("ds.fake_quant_fp8@1", "ds.linear_fp8@1", "ds.linear@1", "ds.grouped_linear@1", "ds.rms_norm@1",
               "ds.rope@1", "ds.sparse_attn@1", "ds.cache_attn@1", "ds.transform@1", "ds.store_cache@1",
               "ds.load_values@1", "ds.load_keys@1", "ds.quant_index_q@1", "ds.index_topk@1", "ds.index_topk_paged@1",
               "ds.compress@1", "ds.compress_decode@1", "ds.hc_step@1", "ds.hc_pre@1", "ds.hc_post@1",
               "ds.moe@1", "ds.engram_gate@1", "ds.engram_hash@1")

MEGA = frozenset({"deepinfer.ds.moe@1"})
"""Solutions that read the expert stacks in the fused MoE kernel's row order (``formats.expert_layout``)."""

Weights = Mapping[str, torch.Tensor]


def on_device(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Host ``values`` on ``device`` without waiting for the work queued there (pinned, non-blocking)."""
    if device.type != "cuda":
        return values.to(device)
    return values.pin_memory().to(device, non_blocking=True)


@dataclass(frozen=True)
class Latents:
    """One compression ratio in a step laid out on the host: where its rows' latents are, and what the
    compressor (above one) needs."""

    segments: torch.Tensor
    """``[n, 4]`` int32 load segments of the rows' latents, in step order (``kernels.ds1017_compose``)."""
    total: int
    width: int
    """The most latents a row has after the step."""
    ks: torch.Tensor
    """``[tokens]``: where its row's latents begin among the loaded ones; ``ke`` how many it reaches past that."""
    ke: torch.Tensor
    seq_token: torch.Tensor
    """``[rows]``: a token of each row (its first). The rest is the compressor's metadata, as DeepInfer's."""
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    cu_c: torch.Tensor
    c_seq: torch.Tensor
    """``[entries]``: the row of each compressed entry; a row reserves at least one."""
    first: torch.Tensor
    """``[entries]``: the position of an entry's first token (0 for a reserved one)."""
    real: torch.Tensor
    """``[entries]``: the entry is a group this step completes."""
    entry_token: torch.Tensor
    """``[entries]``: a token of the entry's row."""


@dataclass(frozen=True)
class Layout:
    """A step laid out on the host: index tensors on the device, the same for every layer."""

    rows: torch.Tensor
    """``[tokens]``: each token's row in the batch's tables."""
    start: torch.Tensor
    """``[tokens]``: the position its row's tokens of this step begin at."""
    ring_segments: torch.Tensor
    """``[rows with earlier tokens, 4]``: each such row's ring, loaded whole (the window's earlier keys)."""
    ring_total: int
    window_index: torch.Tensor
    """``[tokens, window]``: each token's window keys, oldest first, as indices into the layer's source
    (the loaded rings, then this step's tokens); ``-1`` for none."""
    keep: torch.Tensor
    """Token indices of the tokens still in their row's window after the step: into the ring."""
    latents: Mapping[int, Latents]
    ngram: torch.Tensor
    """``[tokens, n]``: for look-back ``s``, the token index holding position ``p - s`` (``ngram_from``
    0), or the index in the row's tail (1), or nothing before the sequence starts (2)."""
    ngram_from: torch.Tensor
    last: torch.Tensor
    """``[rows]``: each row's last token index."""
    tail: torch.Tensor
    """``[rows, n - 1]``: the same for the tail a row keeps after the step, most recent token first."""
    tail_from: torch.Tensor


class Layers:
    """``layers.ds1017@1``."""

    def __init__(self, catalog: Any, config: Any, cfg: DSConfig, kv: Any, slots: Any, token_map: Sequence[int] | None,
                 max_context_len: int, group: Any, placement: Any) -> None:
        self.cfg, self.kv, self.slots = cfg, kv, slots
        self.group = group
        """``comm.group@1``: the ranks that hold this model's experts and Engram tables between them."""
        self.placement = placement
        """``expert.placement@1``: the slot of each expert, when they are placed by the tokens they take."""
        self.chosen: Mapping[str, Mapping[str, str]] = {}
        """``resolve``'s answer: phase -> definition -> solution."""
        self.tables: Callable[[], tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]] = lambda: ((), ())
        """This rank's rows of each Engram layer's table, weights and scales, as loaded now (``set_tables``)."""
        self.bucket = 0
        """The graph bucket of the step being prepared; 0: eager."""
        self.fetched: tuple[torch.Tensor, torch.Tensor] | None = None
        """Among ranks that share the Engram tables: the rows this step's tokens hash to, bytes and
        scales ``[tokens, layers, cols, ...]``, brought in before the forward (``fetch``)."""
        self.fetched_fixed: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        """Bucket -> the buffers a graph of that bucket reads them from."""
        if cfg.n_routed_experts % group.size:
            raise ValueError(f"comm.size = {group.size} does not divide the {cfg.n_routed_experts} routed experts")
        self.kernels = Kernels(catalog, config, "dsv4")
        self.table = "prefill"
        """Which of the fixed solutions the forward runs: a phase, or ``GRAPH`` for a bucket."""
        self.max_context_len = max_context_len
        self.engram_first = min(cfg.engram_layer_ids) if cfg.engram_layer_ids else -1
        if cfg.engram_layer_ids and token_map is None:
            raise ValueError("Engram needs the compressed token map: an engram_token_map.json or a tokenizer.json in "
                             "dsv4.token_map_path, model.path or weights.path")
        self.token_map_host = list(token_map or ())
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
        return {**{f"rows.{size}": t for size, t in self.fixed.items()},
                **{f"engram.{size}.{i}": t for size, pair in self.fetched_fixed.items() for i, t in enumerate(pair)}}

    def release_graphs(self) -> None:
        self.generation += bool(self.fixed)
        self.fixed.clear()
        self.fetched_fixed.clear()

    def set_tables(self, tables: Callable[[], tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]]) -> None:
        """Where this rank's rows of the Engram layers' tables are, in the layers' order: asked at
        every step, the model's weights may have been loaded again."""
        self.tables = tables

    def resolve(self, states: Sequence[StateDecl]) -> Mapping[str, Mapping[str, str]]:
        cfg, page = self.cfg, self.kv.page_size
        ratios = {r for r in cfg.compress_ratios[:cfg.num_hidden_layers] if r}
        # DeepGEMM's paged indexer takes blocks of 32, 64 or 128 keys: a page holds page / ratio.
        blocks = {page // r for r in ratios}
        # DeepGEMM's fused MoE and Hyper-Connections kernels: four copies of the stream, widths their
        # tiles divide (the stream in pieces of 1024 values, the scales of 128 rows together).
        mega = cfg.hc_mult == 4 and cfg.hidden_size % 1024 == 0 and (2 * cfg.moe_intermediate_size) % 128 == 0
        shape = {"heads": cfg.num_attention_heads, "head_dim": cfg.head_dim, "index_head_dim": cfg.index_head_dim,
                 "index_block": min(blocks) if blocks and blocks <= {32, 64, 128} else 0, "mega": int(mega)}
        names = self.definitions()
        calls = {"prefill": {name: shape for name in names if name not in DECODE_ONLY},
                 "decode": {name: shape for name in names if name not in PREFILL_ONLY}}
        # Among ranks the Engram rows are brought in before the forward (``fetch``): no graph records that.
        self.chosen = self.kernels.resolve(calls, outside=frozenset({"ds.engram_fetch@1"}))
        return self.chosen

    def mega_experts(self) -> bool:
        """Whether the routed experts' kernel reads the expert stacks in the fused MoE kernel's row
        order: the model lays them out so (``formats.expert_layout``). One answer for every step."""
        if not self.chosen:
            raise RuntimeError("the layers' kernels are not resolved yet (core.plan does it at start)")
        answers = {solution.split(" ")[0] in MEGA for table in self.chosen.values()
                   for definition, solution in table.items() if definition == "ds.moe@1" and solution}
        if len(answers) > 1:
            raise ValueError("the routed experts' kernels of prefill and decode steps read the experts' weights "
                             "in different row orders: give both the same backend (dsv4.prefill_backend, "
                             "dsv4.decode_backend)")
        return answers == {True}

    def definitions(self) -> tuple[str, ...]:
        """What a step calls: ``DEFINITIONS``, and among ranks that share the Engram tables the
        kernel that brings their rows in (the MoE's kernel reaches the other ranks' experts itself)."""
        if self.group.size == 1 or not self.cfg.engram_layer_ids:
            return DEFINITIONS
        return (*DEFINITIONS, "ds.engram_fetch@1")

    def k(self, definition: str) -> KernelFn:
        return self.kernels(self.table, definition)

    def prepare(self, batch: PersistentBatch, plan: BatchPlan, key: PlanKey) -> None:
        if plan.draft_tokens:
            raise ValueError("layers.ds1017 runs no speculative step yet")
        self.table = GRAPH if key.bucket else key.phase
        self.bucket = key.bucket
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
        # The rings of the rows with earlier tokens are loaded whole, one after the other.
        earlier = [i for i, p in enumerate(starts) if p > 0]
        ring_base = torch.zeros(len(rows), dtype=torch.int64)
        ring_base[earlier] = torch.arange(len(earlier)) * w
        ring_total = len(earlier) * w
        tok_seq = torch.repeat_interleave(torch.arange(len(rows)), n)
        window_index = torch.where(valid, torch.where(in_chunk, ring_total + chunk, ring_base[tok_seq][:, None] + keys % w),
                                   -1)
        slot_table = self.batch_tables[f"slots.{SLOT_KIND}"]
        ring_rows = on_device(torch.tensor([rows[i] for i in earlier], dtype=torch.int64), device)
        ring_segments = torch.stack([slot_table[ring_rows].long(), torch.zeros_like(ring_rows),
                                     torch.full_like(ring_rows, w), on_device(ring_base[earlier], device)], dim=1).int()

        latents: dict[int, Latents] = {}
        table = self.batch_tables["req_to_token"]
        page = self.kv.page_size
        for ratio in sorted({r for r in cfg.compress_ratios[:cfg.num_hidden_layers] if r}):
            entries = page // ratio
            seg_row: list[int] = []
            seg_col: list[int] = []
            seg_rest: list[tuple[int, int, int]] = []
            ks, ke = torch.zeros(len(pos), dtype=torch.int64), torch.zeros(len(pos), dtype=torch.int64)
            c_seq: list[int] = []
            first: list[int] = []
            real: list[bool] = []
            cu_c = [0]
            base_k = 0
            width = 0
            for i, (b, c, p, r) in enumerate(zip(base.tolist(), counts, starts, rows, strict=True)):
                count = (p + c) // ratio
                for j in range(-(-count // entries)):
                    seg_row.append(r)
                    seg_col.append(j * page)
                    seg_rest.append((0, min(entries, count - j * entries), base_k + j * entries))
                ks[b:b + c] = base_k
                ke[b:b + c] = base_k + (pos[b:b + c] + 1) // ratio
                groups = list(range(p // ratio, (p + c) // ratio))
                reserved = max(len(groups), 1)
                c_seq += [i] * reserved
                first += [g * ratio for g in groups] + [0] * (reserved - len(groups))
                real += [True] * len(groups) + [False] * (reserved - len(groups))
                cu_c.append(cu_c[-1] + reserved)
                base_k += count
                width = max(width, count)
            seg_rows = on_device(torch.tensor(seg_row, dtype=torch.int64), device)
            seg_cols = on_device(torch.tensor(seg_col, dtype=torch.int64), device)
            rest = on_device(torch.tensor(seg_rest, dtype=torch.int64).view(-1, 3), device)
            segments = torch.cat([(table[seg_rows, seg_cols].long() // page)[:, None], rest], dim=1).int()
            c_seq_t = torch.tensor(c_seq, dtype=torch.int64)

            def put(values: list[Any] | torch.Tensor, dtype: torch.dtype = torch.int64) -> torch.Tensor:
                return on_device(values if isinstance(values, torch.Tensor) else torch.tensor(values, dtype=dtype), device)

            latents[ratio] = Latents(
                segments=segments, total=base_k, width=width, ks=put(ks), ke=put(ke), seq_token=put(base),
                cu_q=put(torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(n, 0)])),
                cu_k=put(torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(p0 + n, 0)])),
                cu_c=put(cu_c), c_seq=put(c_seq_t), first=put(first), real=put(real, torch.bool),
                entry_token=put(base[c_seq_t]))

        # Engram: an n-gram ending at p looks back to p - s; the row's tail holds what earlier steps ran.
        ng = max(cfg.engram_max_ngram_size, 1)
        back = pos[:, None] - torch.arange(ng)
        ngram_from = torch.where(back < 0, 2, torch.where(back >= tok_p0[:, None], 0, 1))
        ngram = torch.where(ngram_from == 0, tok_base[:, None] + back - tok_p0[:, None], tok_p0[:, None] - 1 - back)
        row_end = p0 + n - 1
        after = row_end[:, None] - torch.arange(max(ng - 1, 1))
        tail_from = torch.where(after < 0, 2, torch.where(after >= p0[:, None], 0, 1))
        tail = torch.where(tail_from == 0, base[:, None] + after - p0[:, None], p0[:, None] - 1 - after)

        def dev(t: torch.Tensor) -> torch.Tensor:
            return on_device(t, device)

        return Layout(rows=dev(tok_row), start=dev(tok_p0), ring_segments=ring_segments, ring_total=ring_total,
                      window_index=dev(window_index), keep=dev(keep), latents=latents, ngram=dev(ngram.clamp_min(0)),
                      ngram_from=dev(ngram_from), last=dev(base + n - 1), tail=dev(tail.clamp_min(0)),
                      tail_from=dev(tail_from))

    # ------------------------------------------------------------------ constants

    def constant(self, name: str, device: torch.device) -> torch.Tensor:
        key = f"{name}@{device}"
        if key not in self.constants:
            cfg = self.cfg
            if name in ("cos_sin.window", "cos_sin.compress"):
                compress = name == "cos_sin.compress"
                freqs = formats.rope_table(
                    cfg.qk_rope_head_dim, self.max_context_len, cfg.rope_original_seq_len if compress else 0,
                    cfg.compress_rope_theta if compress else cfg.rope_theta, cfg.rope_factor, cfg.rope_beta_fast,
                    cfg.rope_beta_slow, device)
                value = torch.cat([freqs.real, freqs.imag], dim=-1).contiguous()  # DeepInfer's cos_sin_cache
            elif name == "token_map":
                value = torch.tensor(self.token_map_host, dtype=torch.int64, device=device)
            else:
                primes, offsets, multipliers = engram_layout(cfg)
                value = torch.tensor({"engram.primes": primes, "engram.offsets": offsets,
                                      "engram.multipliers": multipliers}[name], dtype=torch.int64, device=device)
            self.constants[key] = value
        return self.constants[key]

    # ------------------------------------------------------------------ the forward

    def begin(self, input_ids: torch.Tensor, positions: torch.Tensor) -> None:
        """Once per forward, before the layers: what every layer shares, and the Engram hash ids."""
        tables = self.batch_tables
        slot_table = tables[f"slots.{SLOT_KIND}"]
        rows = self.layout.rows if self.layout is not None else self.decode_rows
        assert rows is not None
        self.step = {"rows": rows, "pos": positions, "slot": slot_table[rows].long(),
                     "tok_slot": tables["req_to_token"][rows, positions].long()}
        if self.cfg.engram_layer_ids and self.group.size == 1:  # among ranks: before the forward (fetch)
            self.step["hashes"] = self._hashes(input_ids, positions, self.step["slot"])

    def fetch(self, input_ids: torch.Tensor, positions: torch.Tensor) -> None:
        """After ``prepare`` and before the forward, outside any graph, with the step's inputs (a
        bucket's padded ones): among ranks that share the Engram tables, the rows this step's tokens
        hash to, from the ranks that hold them, in one exchange with the other ranks' own. A recorded
        step reads them from the bucket's fixed buffers."""
        if self.group.size == 1 or not self.cfg.engram_layer_ids:
            return
        rows = self.layout.rows if self.layout is not None else self.decode_rows
        assert rows is not None
        slot = self.batch_tables[f"slots.{SLOT_KIND}"][rows].long()
        hashes = self._hashes(input_ids, positions, slot)
        held, scales = self.kernels("decode", "ds.engram_fetch@1")(self.group, hashes.long(), *self.tables())
        if self.bucket:
            fixed = self.fetched_fixed.get(self.bucket)
            if fixed is None:
                fixed = self.fetched_fixed[self.bucket] = (held.clone(), scales.clone())
                self.generation += 1
            else:
                fixed[0].copy_(held)
                fixed[1].copy_(scales)
            held, scales = fixed
        self.fetched = (held, scales)

    def _hashes(self, input_ids: torch.Tensor, positions: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        """``NgramHashState.forward``: ``[tokens, engram layers, hash cols]`` rows of the layers' tables."""
        cfg, device = self.cfg, input_ids.device
        compressed = self.constant("token_map", device)[input_ids]
        pad = self.constant("token_map", device)[cfg.engram_pad_token_id]
        tail = self.slots.bind(self.engram_first).buffers()["tail"]
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
        return self.k("ds.engram_hash@1")(tokens, self.constant("engram.multipliers", device),
                                          self.constant("engram.primes", device), self.constant("engram.offsets", device))

    def linear_fp8(self, x: torch.Tensor, w: Weights, name: str) -> torch.Tensor:
        return self.k("ds.linear_fp8@1")(x, w[name + ".weight"], w[name + ".scale"], w[name + ".packed"])

    def linear(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return self.k("ds.linear@1")(x, weight)

    def norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return self.k("ds.rms_norm@1")(x, weight, self.cfg.rms_norm_eps)

    def hc_step(self, x: torch.Tensor | None, stream: torch.Tensor, pre: torch.Tensor, post: torch.Tensor | None,
                comb: torch.Tensor | None, w: Weights, which: str
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Between two sublayers (``ds.hc_step@1``): the output ``x`` of the one before written back to
        the stream, the next coefficients, and the input of the sublayer ``which`` (``attn`` or
        ``ffn``). Returns that input, the stream, and the new ``pre``, ``post`` and ``comb``."""
        cfg = self.cfg
        # A backend whose MoE kernel reads its input from a buffer of its own writes it there now.
        stage = (self.group, self.placement.slots(cfg.n_routed_experts), cfg.num_experts_per_tok,
                 cfg.moe_intermediate_size) if which == "ffn" else None
        return self.k("ds.hc_step@1")(x, stream, pre, post, comb, w[f"hc_{which}_fn"], w[f"hc_{which}_scale"],
                                       w[f"hc_{which}_base"], w[f"{which}_norm.weight"], cfg.rms_norm_eps,
                                       cfg.hc_sinkhorn_iters, cfg.hc_eps, stage)

    def hc_pre(self, x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        return self.k("ds.hc_pre@1")(x, pre)

    def hc_post(self, y: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        return self.k("ds.hc_post@1")(y, residual, post, comb)

    def engram(self, layer: int, x: torch.Tensor, w: Weights) -> torch.Tensor:
        """``Engram.forward`` on the stream ``[tokens, hc, dim]``: the layer's table rows at the hash ids,
        dequantized, projected to one key per copy and a shared value, added through a gate."""
        cfg = self.cfg
        index = cfg.engram_layer_ids.index(layer)
        if self.group.size > 1:  # the table's rows are split over the ranks: brought in before the forward
            assert self.fetched is not None
            held, scales = self.fetched[0][:, index], self.fetched[1][:, index]
        else:
            ids = self.step["hashes"][:, index]
            held, scales = w["engram.embed.weight"][ids], w["engram.embed.scale"][ids]
        rows = formats.dequant_rows_fp8(held, scales).to(x.dtype)
        kv = self.linear_fp8(rows.flatten(-2), w, "engram.wkv")
        return self.k("ds.engram_gate@1")(x, kv, w["engram.q_weight"], w["engram.k_weight"], cfg.rms_norm_eps)

    def moe(self, layer: int, x: torch.Tensor, w: Weights) -> torch.Tensor:
        """``MoE.forward``: the routed experts' weighted sum and the shared expert."""
        cfg = self.cfg
        # Among ranks this rank's stacks hold its part of the layer's experts, where the placement says.
        return self.k("ds.moe@1")(
            self.group, x, w["ffn.gate.weight"], w["ffn.gate.bias"], w["ffn.gate.bf16"],
            *(w[f"ffn.experts.{m}.{part}"] for m in ("w13", "w2") for part in ("weight", "packed")),
            *(w[f"ffn.shared_experts.{m}.{part}"] for m in ("w13", "w2") for part in ("weight", "scale", "packed")),
            cfg.num_experts_per_tok, cfg.routed_scaling_factor, cfg.swiglu_limit, *self.placement.maps(layer))

    def idle(self, weights: Sequence[Weights]) -> None:
        """A step in which this rank has no rows, among ranks that share the experts and the Engram
        tables: its part in what the other ranks compute, layer by layer in the forward's order (their
        tokens may have chosen its experts, or hash to its rows)."""
        if self.group.size == 1:
            return
        cfg = self.cfg
        self.table = "decode"
        device = weights[0]["ffn.gate.weight"].device
        x = torch.empty(0, cfg.hidden_size, dtype=torch.bfloat16, device=device)
        if cfg.engram_layer_ids:  # as ``fetch``, before the layers
            ids = torch.empty(0, len(cfg.engram_layer_ids), cfg.engram_hash_cols, dtype=torch.int64, device=device)
            self.k("ds.engram_fetch@1")(self.group, ids, *self.tables())
        for layer in range(cfg.num_hidden_layers):
            self.moe(layer, x, weights[layer])

    # ------------------------------------------------------------------ attention

    def cos_sin(self, layer: int, device: torch.device) -> torch.Tensor:
        """The layer's rotary table as cosines and sines (``ds.rope@1``)."""
        return self.constant("cos_sin.compress" if self.cfg.ratio(layer) else "cos_sin.window", device)

    def attention(self, layer: int, x: torch.Tensor, w: Weights) -> torch.Tensor:
        """``Attention.forward`` on a layer's normed input ``[tokens, dim]``."""
        cfg, step = self.cfg, self.step
        heads, dim, rd, eps = cfg.num_attention_heads, cfg.head_dim, cfg.qk_rope_head_dim, cfg.rms_norm_eps
        pos, slot = step["pos"], step["slot"]
        table = self.cos_sin(layer, pos.device)
        rope = self.k("ds.rope@1")

        qr = self.norm(self.linear_fp8(x, w, "attn.wq_a"), w["attn.q_norm.weight"])
        q = rope(self.linear_fp8(qr, w, "attn.wq_b").unflatten(-1, (heads, dim)), table, pos, rd)
        kv = self.linear_fp8(x, w, "attn.wkv")
        ring = self.slots.bind(layer).buffers()["ring"]  # [slots, window, record]: a page a request
        win = cfg.sliding_window
        sink = w["attn.attn_sink"]
        store = self.k("ds.store_cache@1")
        if self.layout is None:
            # A decode step attends in the caches: the token's own record goes into the ring first,
            # then its window is the ring's records at the positions up to its own, oldest first.
            store(kv, w["attn.kv_norm.weight"], eps, table, pos, rd, ring.flatten(0, 1), slot * win + pos % win, "fp8", win)
            at = pos[:, None] - win + 1 + torch.arange(win, device=pos.device)
            window = torch.where(at >= 0, slot[:, None] * win + at % win, -1)
            latents = chosen = None
            if cfg.ratio(layer):
                latents, chosen = self._compressed(layer, x, qr, w)
            o = self.k("ds.cache_attn@1")(q, ring, window, latents, chosen, sink, dim ** -0.5)
        else:
            # A step laid out on the host attends over one source of keys, as DeepInfer's prefill does:
            # the window's earlier keys loaded out of the rings, this step's own, the latents loaded.
            # The ring is read before this step's keys go into it (they may take the same entries).
            lay = self.layout
            kv = self.k("ds.transform@1")(kv, w["attn.kv_norm.weight"], eps, table, pos, rd, "fp8")
            held = (self.k("ds.load_values@1")(ring.flatten(0, 1), win, lay.ring_segments, lay.ring_total, "fp8")
                    if lay.ring_total else kv[:0])
            store(kv[lay.keep], None, eps, None, None, 0, ring.flatten(0, 1),
                  slot[lay.keep] * win + pos[lay.keep] % win, "fp8", win)
            parts, idx = [held, kv], lay.window_index
            if cfg.ratio(layer):
                latents, chosen = self._compressed(layer, x, qr, w)
                parts.append(latents)
                idx = torch.cat([idx, torch.where(chosen >= 0, chosen + lay.ring_total + kv.shape[0], -1)], dim=1)
            o = self.k("ds.sparse_attn@1")(q, torch.cat(parts), idx, sink, dim ** -0.5)
        o = rope(o, table, pos, rd, True)

        # wo_a projects each group's heads on their own, on o quantized.
        groups = cfg.o_groups
        o = self.k("ds.fake_quant_fp8@1")(o.reshape(o.shape[0], groups, -1), 32)
        o = self.k("ds.grouped_linear@1")(o, w["attn.wo_a.weight"].view(groups, cfg.o_lora_rank, -1))
        return self.linear_fp8(o.flatten(1), w, "attn.wo_b")

    def _compressed(self, layer: int, x: torch.Tensor, qr: torch.Tensor, w: Weights
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``Attention._compress_kv``. A source layer compresses first; an index source chooses, and the
        layers after it take its choice. In a decode step: the source layer's latent cache by its pages
        ``[pages, entries, record]`` and per token its chosen latents as rows of it; in a step laid out
        on the host: the latents of the step's rows loaded ``[latents, dim]`` and the chosen ones as
        indices into them. ``-1``: none."""
        cfg, step = self.cfg, self.step
        ratio, source = cfg.ratio(layer), cfg.kv_source(layer)
        pool = self.kv.bind(source).buffers()
        entries = self.kv.page_size // ratio
        if layer == source:
            self._compress(layer, x, w, pool, entries)
        if layer == cfg.index_source(layer):
            step["chosen", ratio] = self._index(layer, x, qr, w, pool["ik"], entries)
        chosen = step["chosen", ratio]
        if self.layout is None:
            return pool["kv"].unflatten(0, (-1, entries)), chosen
        lat = self.layout.latents[ratio]
        if not lat.total:
            return pool["kv"].new_zeros(0, cfg.head_dim, dtype=x.dtype), chosen
        return self.k("ds.load_values@1")(pool["kv"], entries, lat.segments, lat.total, "latent"), chosen

    def _compress(self, layer: int, x: torch.Tensor, w: Weights, pool: Weights, entries: int) -> None:
        """``Compressor.forward_store_kv_cache`` and the index keys of ``MQA.store_index_kv``: the
        latents of the groups this step completes, normed, rotated at their first position and
        quantized into ``kv``; the indexer's key of each (from the latent before rotation) into ``ik``.
        A latent's row is its first token's slot over the ratio."""
        cfg, step = self.cfg, self.step
        ratio, rd, eps = cfg.ratio(layer), cfg.qk_rope_head_dim, cfg.rms_norm_eps
        pos, rows = step["pos"], step["rows"]
        table = self.batch_tables["req_to_token"]
        linear = self.k("ds.linear@1")
        if ratio == 1:
            pre = linear(x, w["attn.compressor.wkv.weight"])
            first, dest = pos, step["tok_slot"]
        else:
            state = self.slots.bind(layer).buffers()["cstate"]
            xf = x.float()
            kv, score = linear(xf, w["attn.compressor.wkv.weight"]), linear(xf, w["attn.compressor.wgate.weight"])
            if self.layout is None:
                pre = self.k("ds.compress_decode@1")(score, kv, state, step["slot"], pos + 1, ratio)
                first = pos + 1 - ratio
                # A group still filling yields nothing: its row is -1, stored nowhere.
                dest = torch.where((pos + 1) % ratio == 0, table[rows, first.clamp_min(0)].long() // ratio, -1)
            else:
                lat = self.layout.latents[ratio]
                if not lat.c_seq.numel():
                    return
                pre = self.k("ds.compress@1")(score, kv, state, step["slot"][lat.seq_token], lat.cu_q, lat.cu_k,
                                              lat.cu_c, lat.c_seq, ratio)
                first = lat.first
                dest = torch.where(lat.real, table[rows[lat.entry_token], first].long() // ratio, -1)
        freqs = self.cos_sin(layer, pos.device)
        store = self.k("ds.store_cache@1")
        latent = store(pre, w["attn.compressor.norm.weight"], eps, freqs, first.clamp_min(0), rd, pool["kv"], dest,
                       "latent", entries, True)
        store(linear(latent, w["attn.indexer.wk.weight"]), w["attn.indexer.k_norm.weight"], eps, freqs, first.clamp_min(0),
              rd, pool["ik"], dest, "key", entries)

    def _index(self, layer: int, x: torch.Tensor, qr: torch.Tensor, w: Weights, keys: torch.Tensor, entries: int
               ) -> torch.Tensor:
        """``MQA.forward``: per token the ``index_topk`` best latents of its row it reaches, ascending by
        position (in a decode step as cache rows, else as indices into the step's loaded latents),
        ``-1`` after them. A candidate source also chooses the candidate blocks the layers after it
        score in."""
        cfg, step = self.cfg, self.step
        ratio, rd, h, d = cfg.ratio(layer), cfg.qk_rope_head_dim, cfg.index_n_heads, cfg.index_head_dim
        pos, rows = step["pos"], step["rows"]
        q = self.linear_fp8(qr, w, "attn.indexer.wq_b").unflatten(-1, (h, d))
        q_data, q_scales = self.k("ds.quant_index_q@1")(q, self.cos_sin(layer, pos.device), pos, rd)
        weights = self.k("ds.linear@1")(x, w["attn.indexer.weights_proj.weight"]) * (d ** -0.5 * h ** -0.5)
        source, uses = layer == cfg.candidate_source_layer_id, 0 <= cfg.candidate_source_layer_id < layer
        mode = "source" if source else "user" if uses else "dense"
        candidates = step.get("candidates") if uses else None
        blocks, block = cfg.candidate_topk_blocks, cfg.candidate_block_size
        if self.layout is None:
            # Every row's pages up to the longest context the batch takes: a kernel compiled for that
            # width serves every step (DeepInfer passes its model's longest length); the scores are
            # over the first ``width`` positions.
            width = self.extent[ratio]
            page = self.kv.page_size
            pages = self.batch_tables["req_to_token"][rows][:, ::page] // page
            chosen, picked = self.k("ds.index_topk_paged@1")(q_data, q_scales, weights, keys, entries, pages,
                                                             (pos + 1) // ratio, width, min(cfg.index_topk, width),
                                                             mode, candidates, blocks, block)
        else:
            lat = self.layout.latents[ratio]
            k_data, k_scales = self.k("ds.load_keys@1")(keys, entries, lat.segments, lat.total, d)
            chosen, picked = self.k("ds.index_topk@1")(q_data, q_scales, weights, k_data, k_scales, lat.ks, lat.ke,
                                                       self.max_context_len // ratio, min(cfg.index_topk, lat.width),
                                                       mode, candidates, blocks, block)
        if source:
            step["candidates"] = picked
        return chosen


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
        directory = config["dsv4.token_map_path"] or config["model.path"] or config["weights.path"]
        ctx.provide("layers.ds1017@1", Layers(
            ctx.use("kernels.catalog@1"), config, cfg, ctx.use("state.kv.paged@1"), ctx.use("state.swa.slots@1"),
            token_map(directory, cfg), config["batch.max_context_len"], ctx.use("comm.group@1"),
            ctx.use("expert.placement@1")))

