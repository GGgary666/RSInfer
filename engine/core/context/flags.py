"""Runtime flags: not config, but change at runtime and affect results
(e.g. the DeepEP mode). Read inside a graph capture, a flag is a by-value
dependency of that graph; the plan unit restores flags before replay.
"""

from __future__ import annotations

from typing import Any

from core.context.capture import ValueDep, observing, report_value


class RuntimeFlags:
    """One per process; units reach it as ``ctx.flags``."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}

    def get(self, name: str, default: Any = None) -> Any:
        value = self._values.get(name, default)
        if observing():
            report_value(ValueDep("flag", name, value, lambda: self._values.get(name, default)))
        return value

    def set(self, name: str, value: Any) -> None:
        self._values[name] = value

    def snapshot(self) -> dict[str, Any]:
        return dict(self._values)

    def restore(self, values: dict[str, Any]) -> None:
        self._values = dict(values)
