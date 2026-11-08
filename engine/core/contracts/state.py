"""State kinds: the declaration each layer makes, which the storage unit in the worker
(``core.contracts.storage``) and the accounting plugin in the scheduler host
(``core.contracts.ledger``) share (stage2-contract.md §5, v0-plan 5.10).
"""

from __future__ import annotations

from typing import Literal

import msgspec

type Lifetime = Literal["request", "window", "slot"]
"""``request``: items for every token of the request (paged KV); ``window``: only the last
``window`` tokens; ``slot``: one fixed-size item per request (GDN state)."""


class StateDecl(msgspec.Struct, frozen=True):
    """What one layer stores. The model declares these; the storage unit allocates from them,
    and the accounting plugin accounts by the same declaration."""

    kind: str
    """The state kind (a storage unit plus its accounting plugin), e.g. ``kv.paged``, ``gdn.slots``."""
    layer: int
    tensors: dict[str, tuple[tuple[int, ...], str]]
    """Per item: name -> (shape, dtype); e.g. ``{"k": ((16, 4, 256), "bf16"), "v": ...}``. The
    layout inside a shape (page size, NHD or HND) is chosen by the kernel side (§5.5)."""
    lifetime: Lifetime
    tokens_per_item: int = 0
    """Page size for ``request`` and ``window`` kinds; 0 for ``slot``."""
    window: int = 0
    placement: Literal["device", "host"] = "device"
    borrows: int | None = None
    """Read the items of this layer instead of owning any (DS 1017 cross-layer KV)."""
    snapshot_every: int = 0
    """Tokens between snapshots kept for prefix reuse (GDN state at block boundaries); 0 = none."""
    partition: Literal["replicated", "heads"] = "heads"
    """How TP splits an item: by heads, or a full copy on every rank."""
