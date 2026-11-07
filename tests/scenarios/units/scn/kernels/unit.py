"""The kernel catalog: kernel units register themselves into it (modules.html "声明 / 注册")."""

from __future__ import annotations

from typing import Any

from unit.runtime import Registry


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        self.registry: Registry[tuple[str, str]] = Registry("scn.kernels")
        ctx.provide("scn.kernels@1", self.registry)
