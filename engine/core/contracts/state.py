"""State kinds: the declaration each layer makes, the storage unit in the worker and the
accounting plugin in the scheduler host that share it (stage2-contract.md §5, v0-plan 5.10).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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


class Admitted(msgspec.Struct, frozen=True):
    """What one kind gives a request it admits, for the step's ``StateOps``."""

    items: list[int]
    """Every item the request now holds, in table order: the reused ones (cached KV pages,
    read in place), then the new ones. Goes into ``StateAssign``."""
    copies: list[tuple[int, int]] = []
    """``(src, dst)`` to copy before the step (``StateCopy(when="before")``): a cached GDN
    snapshot into the request's new slot."""


class AccountingPlugin(Protocol):
    """The accounting of one state kind in the scheduler host (``ledger.<kind>@1``). Ids
    only, no memory. Each kind keeps its own prefix index (block hash -> items), since kinds
    align and store prefixes differently; ``core.state.alloc`` only orchestrates (§5.3):

    - admission: ``common_prefix`` over every kind's ``reusable``, then ``admit`` in every
      kind; if one kind returns ``None`` the request is not admitted, and the kinds that did
      admit it get ``release(request, step)``;
    - planning a step: ``take`` for more items, ``publish`` for the blocks the step completes;
    - each result: ``retire(step)``; a step answered with ``Resync``: ``retire(step, ran=False)``.

    Items are free, held (by one or more requests), or cached (published, held by none,
    evictable). A kind evicts cached items, least recently used first, only to give items
    out; never one a request holds or one a pending copy reads."""

    def declare(self, decls: Sequence[StateDecl]) -> None: ...
    def set_capacity(self, items: int) -> None: ...

    def reusable(self, block_hashes: Sequence[int], limit: int) -> int:
        """The longest prefix of a new request, in tokens and at most ``limit``, that this kind
        can resume from: the largest length it supports at or below ``limit``, not the minimum
        of some longest hit. ``block_hashes[i]`` covers tokens ``[0, (i + 1) * B)``, ``B`` being
        the hash block of ``core.state.alloc``; every kind's alignment is a multiple of ``B``.
        Paged KV supports every page boundary up to its longest run of cached blocks; GDN state
        only the snapshots it still holds, so a hit at 96 tokens with snapshots at 64 and 128
        supports 64 when asked for 96. No side effects."""
        ...

    def admit(self, request: int, block_hashes: Sequence[int], length: int, new_tokens: int,
              step: int) -> Admitted | None:
        """Admit the request in ``step``: reuse its first ``length`` tokens (from
        ``common_prefix``, so this kind supports it) and give it items for ``new_tokens`` more.
        All or nothing: ``None`` (nothing taken) if the free and evictable items do not suffice.

        Reused items the request reads in place (KV pages) are held by it until ``release``.
        The source of a copy (a GDN snapshot) is held until the result of ``step`` arrives, so
        it cannot be evicted before the copy has run."""
        ...

    def need(self, request: int, new_tokens: int) -> int:
        """Items an admitted request needs, beyond what it holds, to compute ``new_tokens`` more."""
        ...

    def take(self, request: int, n: int) -> list[int] | None:
        """``n`` more items for an admitted request, evicting cached ones if needed; ``None``
        (nothing taken) if that does not suffice. The scheduler then preempts or waits."""
        ...

    def publish(self, request: int, block_hashes: Sequence[int], upto: int, step: int) -> list[tuple[int, int]]:
        """Step ``step`` computes the request's tokens up to ``upto``: register the blocks it
        completes under their hashes, reusable once the result of ``step`` arrives (before,
        another request could read items the step has not written, or never writes). Returns
        the ``(src, dst)`` copies to make after the step (``StateCopy(when="after")``).

        Paged KV registers the request's own pages, no copy. GDN can snapshot only the state at
        the end of the step: it snapshots when ``upto`` is a multiple of ``snapshot_every``,
        taking a free or evictable slot as ``dst`` (none: no snapshot); boundaries crossed
        inside the step's chunk get no snapshot (§5.4)."""
        ...

    def release(self, request: int, last_step: int) -> None:
        """The request is done. Its items stop being held only once the result of ``last_step``
        (the last step that used them) has arrived: with overlap, that step may still be running
        when the scheduler learns the request finished (§4.4). Then published items no request
        holds become cached, the others free. A ``last_step`` already retired frees at once."""
        ...

    def retire(self, step: int, ran: bool = True) -> None:
        """The result of ``step`` arrived. Free what was released up to it; drop the holds of
        the copies it made. ``ran``: make what it published reusable. Not ``ran`` (answered with
        ``Resync``): drop what it published, its copy destinations become free again. A request
        whose ``admit`` copies were in such a step goes back to the queue: the scheduler
        releases it and admits it again from its token history (§4.3)."""
        ...

    def check(self) -> list[str]:
        """Invariants: free + held + cached == capacity, no item free and held at once, no
        cached item without a hash, nothing leaked."""
        ...


def common_prefix(kinds: Sequence[Callable[[int], int]], limit: int) -> int:
    """The longest prefix, in tokens and at most ``limit``, every kind can resume from
    (stage2-contract.md §5.3). ``kinds[i](n)`` is kind ``i``'s ``reusable(block_hashes, n)``.

    Ask every kind with the current length, take the lowest answer, ask again until no kind
    lowers it. The longest common length is never above an answer, since each kind answers
    the largest length it supports up to what it was asked; and every kind supports the
    length where it stops. ``limit`` leaves the request at least one token to compute."""
    length = limit
    while True:
        answers = [kind(length) for kind in kinds]
        above = [a for a in answers if a > length]
        if above:  # each answer, not the lowest: a valid one would hide one above the limit
            raise ValueError(f"a kind answered {above[0]} tokens when asked for at most {length}")
        lowest = min(answers, default=length)
        if lowest == length:
            return length
        length = lowest
