"""A revocable registry units can provide as a service (e.g. a kernel catalog).

``register()`` returns the undo function, as ``ctx.register()`` requires. Each
item remembers the instance that was running when it was registered, so the
runtime can find items a unit left behind after it unloaded: an item registered
directly instead of through ``ctx.register()`` is a leak the runtime reports.
"""

from __future__ import annotations

import itertools
import weakref
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from core.common.ids import InstanceId

_current: ContextVar[InstanceId | None] = ContextVar("current_instance", default=None)
_registries: weakref.WeakSet[Registry[object]] = weakref.WeakSet()


def current_instance() -> InstanceId | None:
    """The instance whose code the runtime is executing (init, start, undo, hooks)."""
    return _current.get()


@contextmanager
def running_as(instance: InstanceId | None) -> Generator[None]:
    token = _current.set(instance)
    try:
        yield
    finally:
        _current.reset(token)


class Registry[T]:
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[int, tuple[T, InstanceId | None]] = {}
        self._ids = itertools.count()
        _registries.add(self)  # type: ignore[arg-type]

    def register(self, item: T) -> Callable[[], None]:
        key = next(self._ids)
        self._items[key] = (item, current_instance())

        def remove() -> None:
            self._items.pop(key, None)

        return remove

    def items(self) -> list[T]:
        return [item for item, _ in self._items.values()]

    def owned_by(self, instance: InstanceId) -> list[T]:
        return [item for item, owner in self._items.values() if owner == instance]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        return iter(self.items())


def leftovers(instance: InstanceId) -> list[str]:
    """Items any live registry still holds for ``instance``."""
    out: list[str] = []
    for reg in list(_registries):
        n = len(reg.owned_by(instance))
        if n:
            out.append(f"{n} item(s) in registry {reg.name!r}")
    return out
