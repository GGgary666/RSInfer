"""State kinds: the declaration each layer makes, the storage unit in the worker and the
accounting plugin in the scheduler host that share it (stage2-contract.md §5, v0-plan 5.10).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

import msgspec

from core.contracts import Tensor

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


class Binding(Protocol):
    """Buffers a unit hands to others. ``generation`` moves whenever any of them changes
    address, shape or dtype, so a graph checks one integer per provider before replay
    instead of every tensor (GPU probes, proposal 1). Same address, new contents: no change."""

    @property
    def generation(self) -> int: ...
    def buffers(self) -> Mapping[str, Tensor]: ...


class Scratch(msgspec.Struct, frozen=True):
    """Items reserved for capture warm-up, which really runs and would otherwise corrupt the
    state of requests in flight (GPU probes, proposal 6). Never given to a request."""

    ids: list[int]


class StateStore(Protocol):
    """The storage unit of one state kind in a worker (``state.<kind>@1``). Holds the
    memory, never the accounting. Required: ``bind``, ``capacity``, ``scratch``, ``copy``,
    ``invalidate``. ``grow``, ``refit``, ``relocate``, ``export_state`` / ``import_state``
    are optional; the unit declares the ones it has (``capabilities.actions``, ``[state] migrate``)."""

    def declare(self, decls: Sequence[StateDecl]) -> None:
        """At start, from the model; allocates. A kind that cannot hold a declaration raises."""
        ...

    def bind(self, layer: int) -> Binding:
        """The layer's buffers. For VMM-backed kinds the page-indexed view spans the whole
        reservation; only page-indexed kernels may touch it (§10.3)."""
        ...

    def capacity(self) -> int:
        """Usable items, scratch excluded; reported to the plugin as ``Capacity``."""
        ...

    def scratch(self) -> Scratch: ...

    def copy(self, src: Sequence[int], dst: Sequence[int]) -> None:
        """Copy items on the step stream (``StateCopy``): snapshot and restore."""
        ...

    def invalidate(self) -> None:
        """Everything held is void (new weights, a model change); the plugin drops its prefix index."""
        ...


class Growable(Protocol):
    def grow(self, items: int) -> int:
        """Map more memory; addresses and ``generation`` stay. Returns the new capacity."""
        ...


class Migratable(Protocol):
    """state-changes.md §6: version migration, in-process, without copying tensors."""

    def export_state(self) -> object: ...
    def import_state(self, state: object) -> None: ...


class AccountingPlugin(Protocol):
    """The accounting of one state kind in the scheduler host (``ledger.<kind>@1``). Ids
    only, no memory. ``core.state.alloc`` admits a request only if every kind it needs can
    give it items (joint admission), and takes the shortest prefix every kind can reuse."""

    def declare(self, decls: Sequence[StateDecl]) -> None: ...
    def set_capacity(self, items: int) -> None: ...

    def need(self, request: int, new_tokens: int) -> int:
        """Items the request needs to compute ``new_tokens`` more tokens."""
        ...

    def take(self, request: int, n: int) -> list[int]: ...

    def release(self, request: int, last_step: int) -> None:
        """The request is done. Its items return to the free list only once the result of
        ``last_step`` (the last step that used them) has arrived: with overlap, that step may
        still be running when the scheduler learns the request finished (§4.4)."""
        ...

    def retire(self, step: int) -> None:
        """The result of ``step`` arrived: free what was released up to it."""
        ...

    def reusable(self, block_hashes: Sequence[int]) -> int:
        """How many leading blocks of a new request this kind can reuse, after its own
        alignment rules (a GDN snapshot exists only at snapshot boundaries)."""
        ...

    def check(self) -> list[str]:
        """Invariants: free + used == capacity, no item held twice, nothing leaked."""
        ...
