"""A state unit: the request table. Reloading the scheduling logic does not touch it, so
in-flight requests keep their progress (modules.html ⑧)."""

from __future__ import annotations

from typing import Any


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        self.progress: dict[str, int] = {"r1": 0, "r2": 0, "r3": 0}
        ctx.provide("scn.requests@1", self)
