"""``core.kernels``: the kernel catalog (stage2-contract.md §6, modules.html ``core.kernels``).

Kernel units register definitions and solutions with ``ctx.register("kernels.catalog@1", ...)``;
layers ask for a definition with the shape at hand. Selection takes a registered solution that
fits the shape and the platform's capabilities, the first by id; with none, the reference runs
and the selection says so (``Selected.fallback``). Asked for a named backend (the first part of a
solution id), it picks among that backend's solutions only, and fails rather than run the
reference in their place (stage2-contract.md §5.5.3). The catalog implements no kernel.

Every choice is fixed once made: ``call`` keeps each (definition, shape)'s function, and the
backends keep what they resolved at start. ``version`` moves with every registration and undo (a
kernel unit's reload is both), which drops the kept functions, so the next call after a reload
runs the new code.
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
        self.version = 0
        self._kept: dict[tuple[str, tuple[tuple[str, int], ...]], KernelFn] = {}

    def register(self, item: Definition | Implementation) -> Callable[[], None]:
        """Names and ids are unique, so a lookup never depends on the order of registration. A
        reload undoes the old instance's registrations before the new instance registers."""
        if isinstance(item, Definition):
            if any(d.name == item.name for d in self._definitions):
                raise ValueError(f"kernel definition {item.name!r} is already registered")
            undo = self._definitions.register(item)
        else:
            if any(i.solution.id == item.solution.id for i in self._implementations):
                raise ValueError(f"kernel solution {item.solution.id!r} is already registered")
            undo = self._implementations.register(item)
        self._changed()

        def undo_and_note() -> None:
            undo()
            self._changed()
        return undo_and_note

    def _changed(self) -> None:
        self.version += 1
        self._kept.clear()

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

    def select(self, definition: str, shape: Mapping[str, int], backend: str = "auto") -> Selected:
        d = self.definition(definition)
        if backend != "auto":
            named = sorted((i.solution.id for i in self._implementations if i.solution.definition == definition
                            and i.solution.id.split(".", 1)[0] == backend and self._fits(i, shape)))
            if not named:
                raise LookupError(f"backend {backend!r} has no solution of {definition} for {dict(shape)} on "
                                  f"{sorted(self.capabilities)}")
            return Selected(named[0], fallback=False)
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
        """Select once per (definition, shape), then run the kept function."""
        key = (definition, tuple(sorted(shape.items())))
        fn = self._kept.get(key)
        if fn is None:
            fn = self._kept[key] = self.resolve(self.select(definition, shape).solution)
        return fn(*args, **kwargs)


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        devices = ctx.platform.devices()
        self.catalog = KernelCatalog(devices[0].capabilities if devices else [])
        ctx.provide("kernels.catalog@1", self.catalog)
