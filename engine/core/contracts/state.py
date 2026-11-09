"""The state description a model hands to storage and accounting.

Keep this to current handoff data. Allocator policies and future model-specific
fields belong to their implementation until a concrete consumer needs them.
"""

from __future__ import annotations

from typing import Literal

import msgspec

type Lifetime = Literal["request", "slot"]
"""Paged items per request, or one fixed-size item per request."""


class StateDecl(msgspec.Struct, frozen=True):
    """What one layer stores. The model declares these; the storage unit allocates from them,
    and the ledger's internal per-kind pool accounts by the same declaration."""

    kind: str
    """The state kind shared by storage and ledger pools, e.g. ``kv.paged``, ``gdn.slots``."""
    layer: int
    tensors: dict[str, tuple[tuple[int, ...], str]]
    """Per item: name -> (shape, dtype); e.g. ``{"k": ((16, 4, 256), "bf16"), "v": ...}``. The
    layout inside a shape (page size, NHD or HND) is chosen by the kernel side (§5.5)."""
    lifetime: Lifetime
    tokens_per_item: int = 0
    """Page size for request items; 0 for slots."""
    snapshot_every: int = 0
    """Tokens between snapshots kept for prefix reuse (GDN state at block boundaries); 0 = none."""
