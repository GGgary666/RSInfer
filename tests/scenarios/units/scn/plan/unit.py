"""The plan unit: records the decode graph and registers it as a derived artifact.

The graph is built from the kernel the config selects and from ``max_batch``,
both read *by value*, and from the memory unit's buffers, *by address*. When an
upstream change touches it, ``core.deps`` asks for ``scn.rebuild``: a new graph
is recorded next to the old one during prepare, the old one keeps serving, and
the switch happens at the epoch boundary (modules.html ④, ⑦ example 2).
"""

from __future__ import annotations

from functools import partial
from typing import Any

from core.context.capture import ValueDep, report_value

from ..journal import Journal

ARTIFACT = "decode"
KEYS = ("scn.kernel", "scn.max_batch")


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        self.ctx = ctx
        self.config = config
        self.memory = ctx.use("scn.memory@1")
        self.kernels = ctx.use("scn.kernels@1")
        self.journal = Journal(config["scn.log_dir"], ctx)
        self.graph, self.built = self.record(None)
        self.journal.write(f"plan recorded {self.built}")
        ctx.provide("scn.plan@1", self)
        ctx.on_action("scn.rebuild", Rebuild(self))

    def record(self, values: dict[str, Any] | None) -> tuple[Any, dict[str, Any]]:
        """Record the graph. ``values``: the config to build it for (from a change's action
        arguments, not yet in effect); ``None`` reads the config now."""
        ctx = self.ctx
        with ctx.plan_scope(ARTIFACT), ctx.platform.graph().capture() as graph:
            if values is None:
                built = {k: self.config[k] for k in KEYS}  # recorded by value as they are read
            else:
                built = {k: values.get(k, self.config.peek(k)) for k in KEYS}
                for k in KEYS:
                    report_value(ValueDep("config", k, built[k], partial(self.config.peek, k)))
            kernel = self.kernel(built["scn.kernel"])
            ctx.platform.launch("decode", [self.memory.kv, self.memory.placement], solution=kernel,
                                scalars={"max_batch": built["scn.max_batch"]})
        ctx.derived(ARTIFACT, rebuild="scn.rebuild", config=[d.name for d in graph.values if d.source == "config"],
                    units=[kernel])
        return graph, built

    def kernel(self, name: str) -> str:
        for registered, unit in self.kernels.items():
            if registered == name:
                return unit
        raise LookupError(f"no kernel {name!r} in the catalog")

    def replay(self) -> None:
        self.graph.replay()


class Rebuild:
    """``scn.rebuild``: prepare records the new graph in slices; execute switches; abort keeps the old."""

    def __init__(self, plan: Unit) -> None:
        self.plan = plan
        self.next: tuple[Any, dict[str, Any]] | None = None
        self.old: tuple[Any, dict[str, Any]] | None = None

    def prepare(self, args: dict[str, Any]) -> Any:
        def slices() -> Any:
            yield  # a real plan unit records one batch-size bucket per slice
            self.next = self.plan.record(dict(args.get("config", {})))
            yield

        return slices()

    def execute(self, args: dict[str, Any]) -> None:
        assert self.next is not None
        self.old = (self.plan.graph, self.plan.built)
        self.plan.graph, self.plan.built = self.next
        self.plan.journal.write(f"plan switched to {self.plan.built}")

    def commit(self) -> None:
        self.next = self.old = None

    def abort(self) -> None:
        if self.old is not None:
            self.plan.graph, self.plan.built = self.old
        self.next = self.old = None
