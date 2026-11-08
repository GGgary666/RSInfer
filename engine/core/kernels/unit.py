"""``core.kernels``: the kernel catalog (stage2-contract.md §6, modules.html ``core.kernels``).

Kernel units register definitions and solutions with ``ctx.register("kernels.catalog@1", ...)``;
layers ask for a definition with the shape at hand. Selection takes a registered solution that
fits the shape and the platform's capabilities, the first by id; with none, the reference runs
and the selection says so (``Selected.fallback``). The catalog implements no kernel.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from core.contracts.kernels import Definition, Implementation, KernelFn, Selected
from unit.runtime.registry import Registry


class KernelCatalog:
    """``kernels.catalog@1``."""

    def __init__(self, capabilities: Sequence[str]) -> None:
        self.capabilities = frozenset(capabilities)
        self._definitions: Registry[Definition] = Registry("kernels.definitions")
        self._implementations: Registry[Implementation] = Registry("kernels.implementations")

    def register(self, item: Definition | Implementation) -> Callable[[], None]:
        if isinstance(item, Definition):
            return self._definitions.register(item)
        return self._implementations.register(item)

    def definition(self, name: str) -> Definition:
        for d in self._definitions:
            if d.name == name:
                return d
        raise LookupError(f"no kernel definition {name!r} is registered")

    def definitions(self) -> list[Definition]:
        return sorted(self._definitions, key=lambda d: d.name)

    def solutions(self, definition: str) -> list[str]:
        return sorted(i.solution.id for i in self._implementations if i.solution.definition == definition)

    def _fits(self, impl: Implementation, shape: Mapping[str, int]) -> bool:
        s = impl.solution
        return set(s.requires) <= self.capabilities and all(shape.get(k) in allowed for k, allowed in s.shapes.items())

    def select(self, definition: str, shape: Mapping[str, int]) -> Selected:
        d = self.definition(definition)
        fitting = sorted((i for i in self._implementations if i.solution.definition == definition
                          and i.solution.id != d.reference and self._fits(i, shape)), key=lambda i: i.solution.id)
        if fitting:
            return Selected(fitting[0].solution.id, fallback=False)
        self.resolve(d.reference)
        return Selected(d.reference, fallback=True)

    def resolve(self, solution: str) -> KernelFn:
        for i in self._implementations:
            if i.solution.id == solution:
                return i.fn
        raise LookupError(f"no kernel solution {solution!r} is registered")

    def call(self, definition: str, shape: Mapping[str, int], *args: Any, **kwargs: Any) -> Any:
        """The eager path: select on every call, then run."""
        return self.resolve(self.select(definition, shape).solution)(*args, **kwargs)


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        devices = ctx.platform.devices()
        self.catalog = KernelCatalog(devices[0].capabilities if devices else [])
        ctx.provide("kernels.catalog@1", self.catalog)
