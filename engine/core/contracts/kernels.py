"""The kernel catalog: what is computed (definition) and how (solution)
(stage2-contract.md §6, v0-plan 5.11). The schema follows flashinfer-bench's
Definition / Solution.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
    """``module:function``: the PyTorch reference. Every definition has one; it runs on
    ``torch_cpu`` in CI and is what a solution is tested against."""
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


class Catalog(Protocol):
    """``kernels.catalog@1``. Kernel units register into it (``ctx.register``); layers ask
    by definition and shape. A plan fixes its selections when it is built; the eager path
    asks on each call."""

    def select(self, definition: str, shape: Mapping[str, int]) -> Selected: ...
    def resolve(self, solution: str) -> KernelFn: ...
