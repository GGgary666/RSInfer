"""The storage unit's actual handoff to its users in the worker.

Optional state-action hooks follow UnitContext registration; their implementation
does not need another shared Protocol here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from core.contracts.state import StateDecl
from core.contracts.tensor import Binding


class StateStore(Protocol):
    """The storage unit of one state kind in a worker (``state.<kind>@1``). Holds the
    memory, never the accounting. Required: ``bind``, ``capacity``, ``scratch``, ``copy``,
    ``clear``, ``invalidate``. ``grow``, ``refit``, ``relocate``, ``export_state`` / ``import_state``
    are optional; the unit declares the ones it has (``capabilities.actions``, ``[state] migrate``).

    Item ids are ``0 .. capacity() - 1``, as the ledger hands them out; where an item lives in the
    buffers is the store's layout. The SGLang-layout stores keep item ``i`` at index ``i + 1`` (page
    or slot), index 0 being SGLang's padding item, and the persistent batch's tables hold those
    indices (``PersistentBatch``)."""

    def declare(self, decls: Sequence[StateDecl]) -> None:
        """At start, from the model; allocates. A kind that cannot hold a declaration raises."""
        ...

    def bind(self, layer: int) -> Binding:
        """The layer's buffers. For VMM-backed kinds the page-indexed view spans the whole
        reservation; only page-indexed kernels may touch it (§10.3)."""
        ...

    def capacity(self) -> int:
        """Usable items, ids ``0 .. n - 1``; reported to the ledger as ``Capacity``."""
        ...

    def scratch(self) -> Sequence[int]:
        """Ids of the items kept for warm-up and padded rows (stage2-contract.md §5.2), outside
        ``0 .. capacity() - 1`` so the ledger never hands them out; ``copy`` and ``clear`` take them."""
        ...

    def copy(self, src: Sequence[int], dst: Sequence[int]) -> None:
        """Copy items on the step stream (``StateCopy``): snapshot and restore."""
        ...

    def clear(self, items: Sequence[int]) -> None:
        """The items of a request that starts from position 0 and is not restored from a snapshot,
        before that step (SGLang's ``MambaPool.clear_slots`` on ``mamba_clear_indices``). A kind
        whose kernels never read what they did not write (paged KV) does nothing."""
        ...

    def invalidate(self) -> None:
        """Everything held is void (new weights, a model change); the ledger drops its prefix index."""
        ...
