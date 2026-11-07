"""A state unit: holds the buffers the decode graph reads and writes by address, and the
expert placement table (modules.html "状态共享"; scenario ⑤ rewrites it in place)."""

from __future__ import annotations

from typing import Any

from ..journal import Journal


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        platform = ctx.platform
        self.kv = ctx.lease_buffer("kv", lambda: platform.alloc(4096, 0))
        self.placement = ctx.lease_buffer("placement", lambda: platform.alloc(64, 0))
        ctx.effect(lambda: lambda: (platform.free(self.kv), platform.free(self.placement)))
        self.table = [0, 1, 2, 3]
        self.journal = Journal(config["scn.log_dir"], ctx)
        ctx.provide("scn.memory@1", self)
        ctx.on_action("relocate", Relocate(self))


class Relocate:
    """EPLB: a new placement, written into the same buffer (same address, new contents)."""

    def __init__(self, memory: Unit) -> None:
        self.memory = memory
        self.next: list[int] = []
        self.old: list[int] | None = None

    def prepare(self, args: dict[str, Any]) -> None:
        self.next = list(args["placement"])

    def execute(self, args: dict[str, Any]) -> None:
        self.old, self.memory.table = self.memory.table, self.next
        self.memory.placement.write()
        self.memory.journal.write(f"memory placement {self.memory.table}")

    def commit(self) -> None:
        self.old = None

    def abort(self) -> None:
        if self.old is not None:
            self.memory.table, self.old = self.old, None
            self.memory.placement.write()
