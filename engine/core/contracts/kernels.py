"""The kernel catalog: what is computed (definition) and how (solution)
(stage2-contract.md §6, v0-plan 5.11). The schema follows flashinfer-bench's
Definition / Solution.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import msgspec

type KernelFn = Callable[..., Any]


class Tolerance(msgspec.Struct, frozen=True):
    atol: float = 0.0
    rtol: float = 0.0
    """0 / 0: bitwise equal to the reference."""


class Definition(msgspec.Struct, frozen=True):
    name: str
    """Versioned, e.g. ``attn.paged_decode@1``."""
    doc: str
    reference: str
    """Id of the PyTorch reference solution (registered by ``kernels.reference``). Every
    definition has one; it runs on ``torch_eager`` in CI and is what a solution is tested against."""
    tolerance: Tolerance = Tolerance()
    batch_invariant: bool = True
    """The result for one row does not depend on the other rows of the batch."""


class Solution(msgspec.Struct, frozen=True):
    id: str
    """E.g. ``flashinfer.paged_decode``; selection results are by-value deps of a plan."""
    definition: str
    entry: str
    """``module:function`` inside the kernel unit."""
    library: str = ""
    """Library and version, e.g. ``flashinfer==0.6.14``."""
    shapes: Mapping[str, list[int]] = {}
    """Supported values per shape key (head dims, page sizes, ...); empty = any."""
    capturable: bool = False
    requires: list[str] = []
    """Platform capabilities, e.g. ``sm_100``."""
    process_init: bool = False
    """Needs once-per-process initialization (a reload cannot redo it: restart the process)."""
    cross_machine: bool = False


class Selected(msgspec.Struct, frozen=True):
    solution: str
    fallback: bool
    """True: no solution fits, the reference runs; reported in the plan and the result."""


@dataclass(frozen=True)
class Implementation:
    """What a kernel unit registers for a solution: its description and the function itself.
    The reference of a definition is the solution whose id is ``Definition.reference``."""

    solution: Solution
    fn: KernelFn


class Catalog(Protocol):
    """``kernels.catalog@1``. Kernel units register into it (``ctx.register``); layers ask
    by definition and shape. A plan fixes its selections when it is built; the eager path
    asks on each call."""

    def register(self, item: Definition | Implementation) -> Callable[[], None]:
        """Add a definition or a solution; the returned undo removes it (a reload re-registers)."""
        ...
    def select(self, definition: str, shape: Mapping[str, int]) -> Selected: ...
    def resolve(self, solution: str) -> KernelFn: ...
