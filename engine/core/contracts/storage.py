"""The storage unit of a state kind, in the worker (stage2-contract.md §5.2). It shares the
declaration (``core.contracts.state``) with the accounting plugin in the scheduler host.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import msgspec

from core.contracts.state import StateDecl
from core.contracts.tensor import Binding


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
