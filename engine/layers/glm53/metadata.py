"""Prepare CANN metadata once per packed RSInfer step."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from itertools import accumulate, chain

import torch

from core.contracts.plan import PersistentBatch
from core.contracts.records import BatchPlan
from models.glm53_config import CHUNK, KDA, KV


@dataclass(frozen=True)
class Row:
    row: int
    start: int
    count: int
    offset: int


def rows_of(plan: BatchPlan) -> list[Row]:
    count = [int(n) for n in plan.num_tokens.values()]
    offsets = tuple(accumulate(count, initial=0))[:-1]
    return [Row(int(r), int(p), n, off) for r, p, n, off in zip(
        plan.rows.values(), plan.positions.values(), count, offsets, strict=True)]


def pad_decode(rows: list[Row], bucket: int, steps: int) -> list[Row]:
    if any(r.count != steps for r in rows) or not 0 < len(rows) <= bucket:
        raise ValueError(f"decode graph of {bucket} lanes expects {steps} tokens per lane")
    return rows + [Row(-1, 0, steps, i * steps) for i in range(len(rows), bucket)]


def device_metadata[T: (Scan, SparseStep, Verify)](value: T, device: torch.device, current: T | None) -> T:
    """Upload in place on replay. Graphs retain each tensor's original address."""
    new: dict[str, torch.Tensor] = {}
    for field in fields(value):
        data = getattr(value, field.name)
        if isinstance(data, torch.Tensor):
            if current is None:
                new[field.name] = data.to(device)
            else:
                getattr(current, field.name).copy_(data, non_blocking=True)
    return replace(value, **new) if current is None else current


def on_device(values: Sequence[int], device: torch.device, dtype: torch.dtype = torch.int32) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype).to(device=device, non_blocking=True)


@dataclass(frozen=True)
class Scan:
    order: torch.Tensor | None
    cu: tuple[int, ...]
    cu_device: torch.Tensor
    chunks: tuple[int, ...]
    slots: torch.Tensor
    initial: torch.Tensor
    src: torch.Tensor
    snapshot: torch.Tensor
    decode: bool

    @classmethod
    def prepare(cls, rows: Sequence[Row], live: dict[int, int], tracks: dict[int, int], device: torch.device,
                total: int, decode: bool, graph: bool = False) -> Scan:
        cu = tuple(accumulate((r.count for r in rows), initial=0))
        chunks = tuple(chain.from_iterable((i, j) for i, r in enumerate(rows) for j in range(-(-r.count // CHUNK))))
        identity = cu[-1] == total and all(r.offset == cu[i] for i, r in enumerate(rows))
        order = None if identity else on_device([i for r in rows for i in range(r.offset, r.offset + r.count)],
                                                device, torch.int64)
        selected = list(rows) if graph else [r for r in rows if r.row in tracks]
        return cls(order, cu, on_device(cu, device), chunks, on_device([live[r.row] for r in rows], device),
                   on_device([1 if r.start else 0 for r in rows], device, torch.bool),
                   on_device([live[r.row] if r.row in tracks else 0 for r in selected], device, torch.int64),
                   on_device([tracks.get(r.row, 0) for r in selected], device, torch.int64), decode)


def prepare_scans(batch: PersistentBatch, rows: Sequence[Row], device: torch.device, decode: bool,
                  graph: bool = False) -> tuple[Scan, ...]:
    ids = batch.ids(KDA)
    live = {r.row: ids[r.row][-1] + 1 if r.row >= 0 else 0 for r in rows}
    track_rows, track_dst, track_pos = batch.track(KDA)
    tracks = {r: (dst + 1, at) for r, dst, at in zip(track_rows, track_dst, track_pos, strict=True)}
    first: list[Row] = []
    rest: list[Row] = []
    copies: dict[int, int] = {}
    for r in rows:
        if r.row not in tracks:
            first.append(r)
            continue
        dst, pos = tracks[r.row]
        count = pos - r.start
        if not 0 < count <= r.count:
            raise ValueError(f"KDA snapshot at {pos} is outside row {r.row}'s step")
        first.append(Row(r.row, r.start, count, r.offset))
        copies[r.row] = dst
        if count < r.count:
            rest.append(Row(r.row, pos, r.count - count, r.offset + count))
    total = sum(r.count for r in rows)
    first_scan = Scan.prepare(first, live, copies, device, total, decode, graph)
    if not rest:
        return (first_scan,)
    return first_scan, Scan.prepare(rest, live, {}, device, total, False)


@dataclass(frozen=True)
class SparseStep:
    page: int
    ends: torch.Tensor
    seen: torch.Tensor
    pooled: torch.Tensor
    max_pool: int
    table: torch.Tensor
    tail_table: torch.Tensor
    kv_slots: torch.Tensor
    pool_slots: torch.Tensor
    tail_slots: torch.Tensor

    @classmethod
    def prepare(cls, batch: PersistentBatch, rows: Sequence[Row], page: int, pool: int, tail: int,
                device: torch.device, context: int = 0) -> SparseStep:
        all_ids = batch.ids(KV)
        seen = [r.start + r.count for r in rows]
        blocks = -(-(context or max(seen)) // page)
        ids = [[i + 1 for i in all_ids[r.row][:(n + page - 1) // page]] if r.row >= 0 else [0]
               for r, n in zip(rows, seen, strict=True)]
        table = torch.tensor([p + [0] * (blocks - len(p)) for p in ids], dtype=torch.int32)
        tail_table = torch.tensor([[p[r.start // page]] for r, p in zip(rows, ids, strict=True)], dtype=torch.int32)
        # Contiguous positions within each row are already promised by BatchPlan.
        kv_slots = [p[pos // page] * page + pos % page
                    for r, p in zip(rows, ids, strict=True) for pos in range(r.start, r.start + r.count)]
        pooled = [n // pool for n in seen]
        return cls(page, on_device(tuple(accumulate(r.count for r in rows)), device), on_device(seen, device),
                   on_device(pooled, device), context // pool if context else max(pooled),
                   table.to(device, non_blocking=True),
                   tail_table.to(device, non_blocking=True), on_device(kv_slots, device, torch.int64),
                   on_device([i // pool for i in kv_slots], device, torch.int64),
                   on_device([(i // page) * tail + i % tail for i in kv_slots], device, torch.int64))


@dataclass(frozen=True)
class IndexShard:
    width: int
    tokens: slice
    requests: slice
    ends: torch.Tensor
    pooled: torch.Tensor
    max_pool: int

    @classmethod
    def prepare(cls, rows: Sequence[Row], pool: int, rank: int, size: int, device: torch.device) -> IndexShard:
        width = (sum(row.count for row in rows) + size - 1) // size
        begin, end = rank * width, (rank + 1) * width
        selected = []
        for rowno, row in enumerate(rows):
            left, right = max(begin, row.offset), min(end, row.offset + row.count)
            if left < right:
                selected.append((rowno, right - left, (row.start + right - row.offset) // pool))
        requests = slice(selected[0][0], selected[-1][0] + 1) if selected else slice(0, 0)
        ends = tuple(accumulate(item[1] for item in selected))
        available = [item[2] for item in selected]
        return cls(width, slice(begin, end), requests, on_device(ends, device), on_device(available, device),
                   max(available, default=0))


@dataclass(frozen=True)
class Verify:
    slots: torch.Tensor
    track: torch.Tensor
    lanes: torch.Tensor
    conv_slots: torch.Tensor
    state_slots: torch.Tensor
    cu: torch.Tensor
    initial_tokens: torch.Tensor

    @classmethod
    def prepare(cls, batch: PersistentBatch, rows: Sequence[Row], device: torch.device, steps: int) -> Verify:
        if any(r.count != steps for r in rows):
            raise ValueError("every GLM verify lane needs one fixed target window")
        ids = batch.ids(KDA)
        slots = on_device([ids[r.row][-1] + 1 if r.row >= 0 else 0 for r in rows], device, torch.int64)
        track_rows, track_dst, _ = batch.track(KDA)
        tracking = dict(zip(track_rows, track_dst, strict=True))
        track = on_device([tracking.get(r.row, -1) + 1 for r in rows], device, torch.int64)
        count = len(rows)
        lanes = torch.arange(count, device=device, dtype=torch.int64)
        state_slots = torch.arange(count * steps, device=device, dtype=torch.int32).view(count, steps)
        cu = on_device(range(0, (count + 1) * steps, steps), device)
        initial_tokens = torch.ones(count, device=device, dtype=torch.int32)
        return cls(slots, track, lanes, lanes.to(torch.int32), state_slots, cu, initial_tokens)
