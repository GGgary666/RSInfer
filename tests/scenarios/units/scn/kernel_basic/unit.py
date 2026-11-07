"""A kernel solution: registers ``(name, unit)`` in the catalog; the registration is undone on unload."""

from __future__ import annotations

from typing import Any

NAME = "basic"


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        ctx.register("scn.kernels@1", (NAME, ctx.id))
