"""``core.manifest``: check a candidate tree before anything runs (doc/specs/control-plane.md §3)."""

from __future__ import annotations

import asyncio
from typing import Any

from core.change.coordinator import PlanInput
from core.common.framework import FrameworkModules
from core.common.result import Failed, Ok, Result
from core.manifest.checks import check_tree


class Checker:
    """``ctrl.manifest@1``: the coordinator's static check (``core.change`` RECEIVED -> CHECKED)."""

    def __init__(self, host: Any) -> None:
        self._host = host
        self._framework = FrameworkModules.default()

    async def check(self, inp: PlanInput) -> Result[None]:
        with self._host.analysing(inp.new.files) as root:
            # Static AST analysis can exceed the host heartbeat interval.
            errors, warnings = await asyncio.to_thread(check_tree, root, inp.new.topology, self._framework)
        if errors:
            return Failed([*errors, *warnings])
        return Ok(None, warnings)


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        ctx.provide("ctrl.manifest@1", Checker(ctx.use("ctrl.self@1")))
