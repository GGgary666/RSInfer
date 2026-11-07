"""Declares a native extension; changing that file restarts the process (modules.html ⑨).
The ``.so`` here is a placeholder and is never loaded: what matters is the declaration."""

from __future__ import annotations

from typing import Any


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        pass
