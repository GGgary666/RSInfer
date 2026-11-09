# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""``worker.batch``: the persistent batch (``worker.batch@1``, stage2-contract.md §4.3).

Its tables are SGLang's ``HybridReqToTokenPool`` (srt/mem_cache/memory_pool.py, e838360;
LICENSE.sglang): ``req_to_token`` ``[rows + 1, max_context_len]`` int32 token slots, with a padding
row at index 0, and per slot kind the row -> item mapping (``req_index_to_mamba_index_mapping``).
Row ``r`` of the scheduler's records is index ``r + 1`` (SGLang's ``req_pool_idx``), and item ``i``
page or slot ``i + 1`` (the ones SGLang's allocators hand out; 0 is padding). The rows
themselves and their sampling parameters are what the scheduler sent (``RowDelta``); the tokens
sampled last, which ``from_previous`` names, are kept on the device.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from core.contracts.records import BatchPlan, RowDelta, StateOps
from core.contracts.sampling import SamplingParams

PAGED_KIND = "kv.paged"
"""The kind whose items are pages of ``page_size`` token slots; every other kind is one item per row."""


@dataclass
class Row:
    request: int
    sampling: SamplingParams
    ids: dict[str, list[int]] = field(default_factory=dict[str, list[int]])
    has_previous: bool = False
    """A token was sampled for the row since it was added: ``from_previous`` may name it."""


@dataclass(frozen=True)
class Tables:
    """``Binding``."""

    generation: int
    tensors: Mapping[str, torch.Tensor]

    def buffers(self) -> Mapping[str, torch.Tensor]:
        return self.tensors


@dataclass
class PersistentBatch:
    """``worker.batch@1``."""

    platform: Any
    max_rows: int
    max_context_len: int
    page_size: int
    rows: dict[int, Row] = field(default_factory=dict[int, Row])
    tensors: dict[str, torch.Tensor] = field(default_factory=dict[str, torch.Tensor])
    allocated: list[Any] = field(default_factory=list[Any])
    generation: int = 0
    rebuilt: bool = True
    """Built and not yet given a full delta: it refuses every other one (``Resync``)."""

    def __post_init__(self) -> None:
        # +1 padding row at index 0: padded batches read and write there harmlessly.
        self._alloc("req_to_token", (self.max_rows + 1, self.max_pages * self.page_size), "i32")
        self._alloc("last_token", (self.max_rows + 1,), "i64")

    @property
    def max_pages(self) -> int:
        """Pages of one row: whole pages for ``max_context_len`` tokens, so any request of at most
        that many tokens fits."""
        return -(-self.max_context_len // self.page_size)

    def _alloc(self, name: str, shape: tuple[int, ...], dtype: str) -> torch.Tensor:
        buf = self.platform.alloc(0, 0, shape=shape, dtype=dtype)
        self.allocated.append(buf)
        self.tensors[name] = self.platform.tensor(buf)
        return self.tensors[name]

    @property
    def binding(self) -> Tables:
        return Tables(self.generation, dict(self.tensors))

    def check(self, delta: RowDelta, plan: BatchPlan) -> str:
        if self.rebuilt and not delta.full:
            return "the persistent batch was rebuilt and waits for a full delta"
        known = set() if delta.full else set(self.rows)
        for r in (int(x) for x in delta.removed.values()):
            if r not in known:
                return f"row {r} is removed but not in the persistent batch"
            known.discard(r)
        added = {a.row for a in delta.added}
        known |= added
        for r, prev in zip((int(x) for x in plan.rows.values()), plan.from_previous.values(), strict=True):
            if r not in known:
                return f"row {r} is not in the persistent batch"
            if prev and (r in added or not self.rows[r].has_previous):
                return f"row {r} names its previous token, which this batch does not have"
        return ""

    def apply(self, delta: RowDelta, state: StateOps) -> None:
        self._within_limits(delta, state)
        if delta.full:
            self.rows.clear()
            self.rebuilt = False
        for r in (int(x) for x in delta.removed.values()):
            del self.rows[r]
        for a in delta.added:
            self.rows[a.row] = Row(a.request, a.sampling)
        for assign in state.assign:
            offsets, ids = list(assign.offsets.values()), [int(x) for x in assign.ids.values()]
            for i, r in enumerate(int(x) for x in assign.rows.values()):
                self._assign(r, assign.kind, ids[int(offsets[i]):int(offsets[i + 1])])

    def _within_limits(self, delta: RowDelta, state: StateOps) -> None:
        """What ``apply`` cannot do, found before it changes anything: a row outside the batch,
        items for a row it does not have, more pages than one row holds."""
        pages = {} if delta.full else {r: len(row.ids.get(PAGED_KIND, [])) for r, row in self.rows.items()}
        for r in (int(x) for x in delta.removed.values()):
            pages.pop(r, None)
        for a in delta.added:
            if not 0 <= a.row < self.max_rows:
                raise ValueError(f"row {a.row} is outside the batch of {self.max_rows} rows")
            pages[a.row] = 0
        for assign in state.assign:
            offsets = [int(x) for x in assign.offsets.values()]
            for i, r in enumerate(int(x) for x in assign.rows.values()):
                if r not in pages:
                    raise ValueError(f"row {r} is given {assign.kind} items but is not in the batch")
                if assign.kind == PAGED_KIND:
                    pages[r] += offsets[i + 1] - offsets[i]
                    if pages[r] > self.max_pages:
                        raise ValueError(f"row {r} would hold {pages[r]} pages of {self.page_size} tokens, more than "
                                         f"{self.max_context_len} tokens")

    def _assign(self, r: int, kind: str, new: list[int]) -> None:
        held = self.rows[r].ids.setdefault(kind, [])
        if kind == PAGED_KIND:  # req_to_token_pool.write: the token slots of the new pages
            start, end = len(held) * self.page_size, (len(held) + len(new)) * self.page_size
            pages = torch.tensor(new, dtype=torch.int32) + 1
            slots = (pages[:, None] * self.page_size + torch.arange(self.page_size, dtype=torch.int32)).flatten()
            self.tensors["req_to_token"][r + 1, start:end] = slots.to(self.tensors["req_to_token"].device)
        else:
            name = f"slots.{kind}"
            if name not in self.tensors:
                self._alloc(name, (self.max_rows + 1,), "i32")
                self.generation += 1
            self.tensors[name][r + 1] = new[-1] + 1
        held.extend(new)

    def ids(self, kind: str) -> Mapping[int, Sequence[int]]:
        return {r: row.ids.get(kind, []) for r, row in self.rows.items()}

    def sampling(self, rows: Sequence[int]) -> list[SamplingParams]:
        return [self.rows[r].sampling for r in rows]

    def record(self, rows: Sequence[int], tokens: torch.Tensor) -> None:
        index = torch.tensor(list(rows), dtype=torch.int64, device=tokens.device) + 1
        self.tensors["last_token"][index] = tokens.to(torch.int64)
        for r in rows:
            self.rows[r].has_previous = True

    def free(self) -> None:
        for buf in self.allocated:
            self.platform.free(buf)
        self.allocated.clear()


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        self.batch = PersistentBatch(ctx.platform, config["batch.max_rows"], config["batch.max_context_len"],
                                     config["state.kv.page_size"])
        ctx.effect(lambda: self.batch.free, label="batch tables")
        ctx.provide("worker.batch@1", self.batch)
