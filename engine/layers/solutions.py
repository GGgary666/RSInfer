"""What the backend units ``layers.attn`` and ``layers.gdn`` share (a library, not a unit): the
kernel solutions a backend calls, fixed once at start from the backend configured for each phase
(stage2-contract.md §5.5.3, §8.1), as SGLang fixes its attention and linear-attention backends
from ``--prefill-attention-backend``, ``--linear-attn-decode-backend`` and the like."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.contracts.kernels import KernelFn

PHASES = ("prefill", "decode")


class Kernels:
    """The solutions of one backend, per phase and definition."""

    def __init__(self, catalog: Any, config: Any, prefix: str) -> None:
        """``prefix``: the backend's config keys, ``<prefix>.backend`` and ``<prefix>.<phase>_backend``
        (empty: the former)."""
        self.catalog = catalog
        self.keys = {phase: f"{prefix}.{phase}_backend" if config[f"{prefix}.{phase}_backend"] else f"{prefix}.backend"
                     for phase in PHASES}
        self.backends = {phase: config[key] for phase, key in self.keys.items()}
        self.fns: dict[tuple[str, str], KernelFn] = {}

    def resolve(self, calls: Mapping[str, Mapping[str, Mapping[str, int]]]) -> dict[str, dict[str, str]]:
        """``calls``: phase -> definition -> the shape it is called with. Returns phase -> the backend
        and each definition's solution, a fallback to the reference marked."""
        fns: dict[tuple[str, str], KernelFn] = {}
        out: dict[str, dict[str, str]] = {}
        for phase, definitions in calls.items():
            backend = self.backends[phase]
            chosen = {"backend": backend}
            for definition, shape in definitions.items():
                try:
                    selected = self.catalog.select(definition, shape, backend)
                except LookupError as e:
                    raise ValueError(f"{self.keys[phase]} = {backend!r}: {e}") from None
                fns[phase, definition] = self.catalog.resolve(selected.solution)
                chosen[definition] = selected.solution + (" (fallback)" if selected.fallback else "")
            out[phase] = chosen
        self.fns = fns
        return out

    def __call__(self, phase: str, definition: str) -> KernelFn:
        try:
            return self.fns[phase, definition]
        except KeyError:
            raise RuntimeError(f"{definition} has no solution for {phase}: the backend was not resolved "
                               "(core.plan resolves every backend at start)") from None
