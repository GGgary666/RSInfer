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
        """Usable items, scratch excluded; reported to the ledger as ``Capacity``."""
        ...

    def scratch(self) -> Sequence[int]: ...

    def copy(self, src: Sequence[int], dst: Sequence[int]) -> None:
        """Copy items on the step stream (``StateCopy``): snapshot and restore."""
        ...

    def invalidate(self) -> None:
        """Everything held is void (new weights, a model change); the ledger drops its prefix index."""
        ...
