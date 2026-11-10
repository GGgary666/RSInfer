"""``core.kernels``: the kernel catalog (stage2-contract.md §6, modules.html ``core.kernels``).

Kernel units register definitions and solutions with ``ctx.register("kernels.catalog@1", ...)``;
layers ask for a definition with the shape at hand. Selection takes a registered solution that
fits the shape and the platform's capabilities, the first by id; with none, the reference runs
and the selection says so (``Selected.fallback``). Asked for a named backend (the first part of a
solution id), it picks among that backend's solutions only, and fails rather than run the
reference in their place (stage2-contract.md §5.5.3). For a step replayed from a graph it picks
among the capturable solutions (``select(capture=True)``). The catalog implements no kernel.

Every choice is fixed once made: ``call`` keeps each (definition, shape)'s function, and the
backends keep what they resolved at start. ``version`` moves with every registration and undo (a
kernel unit's reload is both), which drops the kept functions, so the next call after a reload
runs the new code.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from core.contracts.kernels import CAPTURE, Definition, Implementation, KernelFn, Selected, Solution
from unit.runtime.registry import Registry, current_instance


class KernelCatalog:
    """``kernels.catalog@1``."""

    def __init__(self, capabilities: Sequence[str]) -> None:
        self.capabilities = frozenset(capabilities)
        self._definitions: Registry[Definition] = Registry("kernels.definitions")
        self._implementations: Registry[Implementation] = Registry("kernels.implementations")
        self.version = 0
        self._kept: dict[tuple[str, tuple[tuple[str, int], ...]], tuple[str, KernelFn]] = {}
        """What ``call`` chose per (definition, shape): the solution's id and its function."""
        self._sources: dict[str, str] = {}
        self._recording: set[str] | None = None

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
        owner = current_instance()
        name = item.name if isinstance(item, Definition) else item.solution.id
        if owner is not None:
            self._sources[name] = str(owner.unit)

        def undo_and_note() -> None:
            undo()
            self._sources.pop(name, None)
            self._changed()
        return undo_and_note

    def _changed(self) -> None:
        self.version += 1
        self._kept.clear()

    def sources(self) -> list[str]:
        """The units that registered something, alternatives an ``auto`` selection passed over included."""
        return sorted(set(self._sources.values()))

    def describe(self, solution: str) -> Solution:
        for i in self._implementations:
            if i.solution.id == solution:
                return i.solution
        raise LookupError(f"no kernel solution {solution!r} is registered")

    @contextmanager
    def recording(self) -> Generator[set[str]]:
        """The ids of the solutions ``call`` runs inside the scope."""
        used: set[str] = set()
        outer, self._recording = self._recording, used
        try:
            yield used
        finally:
            self._recording = outer

    def definition(self, name: str) -> Definition:
        for d in self._definitions:
            if d.name == name:
                return d
        raise LookupError(f"no kernel definition {name!r} is registered")

    def definitions(self) -> list[Definition]:
        return sorted(self._definitions, key=lambda d: d.name)

    def solutions(self, definition: str) -> list[str]:
        return sorted(i.solution.id for i in self._implementations if i.solution.definition == definition)

    def _fits(self, impl: Implementation, shape: Mapping[str, int], capture: bool) -> bool:
        s = impl.solution
        have = self.capabilities | {CAPTURE} if capture else self.capabilities
        return ((s.capturable or not capture) and set(s.requires) <= have
                and all(shape.get(k) in allowed for k, allowed in s.shapes.items()))

    def select(self, definition: str, shape: Mapping[str, int], backend: str = "auto", *,
               capture: bool = False) -> Selected:
        d = self.definition(definition)
        fitting = sorted(i.solution.id for i in self._implementations
                         if i.solution.definition == definition and self._fits(i, shape, capture))
        where = f"{definition} for {dict(shape)} on {sorted(self.capabilities)}"
        if backend != "auto":
            named = [s for s in fitting if s.split(".", 1)[0] == backend]
            if not named:
                raise LookupError(f"backend {backend!r} has no {'capturable ' if capture else ''}solution of {where}")
            return Selected(named[0], fallback=False)
        others = [s for s in fitting if s != d.reference]
        if others:
            return Selected(others[0], fallback=False)
        if capture and d.reference not in fitting:
            raise LookupError(f"no capturable solution of {where}")
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
        kept = self._kept.get(key)
        if kept is None:
            solution = self.select(definition, shape).solution
            kept = self._kept[key] = (solution, self.resolve(solution))
        if self._recording is not None:
            self._recording.add(kept[0])
        return kept[1](*args, **kwargs)


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        devices = ctx.platform.devices()
        self.catalog = KernelCatalog(devices[0].capabilities if devices else [])
        ctx.provide("kernels.catalog@1", self.catalog)
