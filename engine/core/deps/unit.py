"""``core.deps``: what has to move for a change, and how (doc/specs/control-plane.md §4)."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from typing import Any

from core.change.coordinator import PlanInput
from core.change.types import ActionPlan
from core.common.framework import FrameworkModules
from core.common.ids import ProcId
from core.common.result import Diagnostic, Failed, Ok, Result, diag
from core.deps.planner import World, plan
from core.sourcetree import Tree, read_tree
from core.supervisor.protocol import Introspect, ProcIntrospection


class Planner:
    """``ctrl.deps@1``: the coordinator's planner (``core.change`` CHECKED -> PLANNED)."""

    def __init__(self, host: Any, supervisor: Any) -> None:
        self._host = host
        self._sup = supervisor
        self._framework = FrameworkModules.default()

    async def plan(self, inp: PlanInput) -> Result[ActionPlan]:
        with ExitStack() as held:  # the trees are read lazily while planning, across the awaits below
            new_tree = read_tree(held.enter_context(self._host.analysing(inp.new.files)))
            base_tree = read_tree(held.enter_context(self._host.analysing(inp.base.files))) \
                if inp.base.files else None
            return await self._plan(inp, new_tree, base_tree)

    async def _plan(self, inp: PlanInput, new_tree: Tree, base_tree: Tree | None) -> Result[ActionPlan]:
        running = [s.proc for s in inp.procs if s.state == "RUNNING"]
        replies = await asyncio.gather(*(self._sup.call(p, Introspect()) for p in running))
        intro: dict[ProcId, ProcIntrospection] = {}
        missing: list[Diagnostic] = []
        for proc, r in zip(running, replies, strict=True):
            if isinstance(r, Ok) and isinstance(r.value, ProcIntrospection):
                intro[proc] = r.value
            else:  # it would be left on the old identity: plan nothing
                missing.append(diag("core.deps", "plan", f"did not describe itself: {r}", subject=str(proc),
                                    code="deps.introspection_failed", hint="retry, or restart that process"))
        if missing:
            return Failed(missing)
        w = World(cs=inp.cs, base=inp.base, new=inp.new, base_tree=base_tree, new_tree=new_tree, running=intro,
                  control=self._host.introspect_self(), framework=self._framework)
        return plan(w)


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        ctx.provide("ctrl.deps@1", Planner(ctx.use("ctrl.self@1"), ctx.use("ctrl.supervisor@1")))
