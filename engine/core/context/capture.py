"""By-value dependency observation.

While a graph is being recorded, every config key and runtime flag that is read
is a by-value dependency of that graph (modules.html ④, "绑定方式"). Readers
report values here; the graph recorder of the platform observes them and checks
them again before every replay.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ValueDep:
    source: str
    """``"config"``, ``"flag"``, ``"solution"``, ``"scalar"``, ..."""
    name: str
    value: Any
    current: Callable[[], Any] | None = None
    """Re-reads the value at replay time; ``None`` for values fixed by the caller."""


type ValueObserver = Callable[[ValueDep], None]

_observers: ContextVar[tuple[ValueObserver, ...]] = ContextVar("value_observers", default=())


@contextmanager
def observe_values(observer: ValueObserver) -> Generator[None]:
    token = _observers.set((*_observers.get(), observer))
    try:
        yield
    finally:
        _observers.reset(token)


def report_value(dep: ValueDep) -> None:
    for observer in _observers.get():
        observer(dep)


def observing() -> bool:
    return bool(_observers.get())
