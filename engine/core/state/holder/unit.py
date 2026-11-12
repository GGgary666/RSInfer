"""``core.state.holder``: the device memory budget of the state kinds (``state.holder@1``).

Sized as SGLang sizes its pools for hybrid GDN models (``mem_fraction_static``,
``mamba_full_memory_ratio``, ``max_mamba_cache_size``; srt/mem_cache/kv_cache_configurator.py):

- The weights and the state pools together take ``holder.mem_fraction`` of the device; the rest is
  left for activations and the runtime. So the states get ``total * mem_fraction - weights``.
- The slot kinds (GDN state) get ``r / (1 + r)`` of that, ``r = holder.state_ratio``, or exactly
  ``holder.max_slots`` slots when set; the paged kinds (KV) get what is left.
- Every request holds one item of each slot kind, so each slot kind gets the same count; the paged
  kinds likewise. Each kind's padding item (index 0, ``StateStore``) is paid for too.
- On a CPU device the states get ``holder.cpu_bytes``: the machine's memory is not the budget.
- Asleep (``holder.sleep``), every kind gets one item, whatever the budget or the stores' own
  counts say (``state.kv.pages``, ``state.gdn.slots``): the engine gives its device memory back
  and serves nothing (the scheduler refuses requests), so one item only keeps the shapes valid.

The weights' size is what the model allocated, not a measurement: memory the runtime or another
library takes is what the remaining ``1 - mem_fraction`` is for.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from core.contracts.state import StateDecl

DTYPE_BYTES = MappingProxyType({"u8": 1, "i8": 1, "bf16": 2, "f16": 2, "f32": 4, "i32": 4, "i64": 8, "f64": 8})


def item_bytes(decls: Sequence[StateDecl]) -> int:
    """Bytes of one item of a kind: every layer's tensors of it."""
    return sum(math.prod(shape) * DTYPE_BYTES[dtype] for d in decls for shape, dtype in d.tensors.values())


class Holder:
    """``state.holder@1``."""

    def __init__(self, device_bytes: int, *, cpu: bool, mem_fraction: float, state_ratio: float,
                 max_slots: int, cpu_bytes: int, asleep: bool = False) -> None:
        if not 0 < mem_fraction <= 1 or state_ratio < 0 or max_slots < 0:
            raise ValueError("holder.mem_fraction must be in (0, 1], state_ratio and max_slots >= 0")
        self.device_bytes, self.cpu = device_bytes, cpu
        self.mem_fraction, self.state_ratio, self.max_slots, self.cpu_bytes = mem_fraction, state_ratio, max_slots, cpu_bytes
        self.asleep = asleep
        self.sizes: dict[str, int] = {}

    def size(self, states: Sequence[StateDecl], weights_bytes: int) -> Mapping[str, int]:
        kinds: dict[str, list[StateDecl]] = {}
        for d in states:
            kinds.setdefault(d.kind, []).append(d)
        per_item = {kind: item_bytes(decls) for kind, decls in kinds.items()}
        if self.asleep:
            self.sizes = dict.fromkeys(kinds, 1)
            print(f"core.state.holder: asleep (holder.sleep): one item of each of {sorted(kinds)}", file=sys.stderr,
                  flush=True)
            return dict(self.sizes)
        slot_kinds = [k for k, ds in kinds.items() if ds[0].lifetime == "slot"]
        paged_kinds = [k for k in kinds if k not in slot_kinds]
        budget = self.cpu_bytes if self.cpu else int(self.device_bytes * self.mem_fraction) - weights_bytes
        budget -= sum(per_item.values())  # the padding item of every kind
        slot_bytes = sum(per_item[k] for k in slot_kinds)
        page_bytes = sum(per_item[k] for k in paged_kinds)
        slots = 0
        if slot_kinds:
            slots = self.max_slots or int(budget * self.state_ratio / (1 + self.state_ratio)) // slot_bytes
        pages = (budget - slots * slot_bytes) // page_bytes if paged_kinds else 0
        if (slot_kinds and slots < 1) or (paged_kinds and pages < 1):
            source = "holder.cpu_bytes" if self.cpu else (
                f"{self.mem_fraction} of {self.device_bytes / 2**30:.1f} GiB less {weights_bytes / 2**30:.2f} GiB of weights")
            raise RuntimeError(f"the states do not fit: {budget / 2**30:.2f} GiB for them ({source}); a slot takes "
                               f"{slot_bytes / 2**20:.1f} MiB, a page {page_bytes / 2**10:.1f} KiB")
        self.sizes = {**{k: slots for k in slot_kinds}, **{k: pages for k in paged_kinds}}
        print(f"core.state.holder: {budget / 2**30:.2f} GiB for states: "
              + ", ".join(f"{k} {n} ({n * per_item[k] / 2**30:.2f} GiB)" for k, n in self.sizes.items()),
              file=sys.stderr, flush=True)
        return dict(self.sizes)

    def items(self, kind: str) -> int:
        return self.sizes.get(kind, 0)


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        device = ctx.platform.devices()[0]
        ctx.provide("state.holder@1", Holder(device.total_bytes, cpu="cpu" in device.capabilities,
                                             mem_fraction=config["holder.mem_fraction"],
                                             state_ratio=config["holder.state_ratio"],
                                             max_slots=config["holder.max_slots"], cpu_bytes=config["holder.cpu_bytes"],
                                             asleep=config["holder.sleep"]))
