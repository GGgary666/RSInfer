"""A state unit: the request table. Reloading the scheduling logic does not touch it, so what it
holds stays (modules.html ⑧). It declares no ``drain``: a change applies over its counters, which
stand for state, not for requests in flight (doc/specs/empty-apply.md)."""

from __future__ import annotations

from typing import Any


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        self.progress: dict[str, int] = {"r1": 0, "r2": 0, "r3": 0}
        ctx.provide("scn.requests@1", self)
