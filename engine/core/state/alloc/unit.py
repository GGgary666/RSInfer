"""``ledger@1``: joint admission over every state kind, from #22.

Each kind's ``Pool`` keeps its own ids; a request is admitted only when every kind has what it
needs, and what one kind gave is given back at once when another refuses (§5.3). No tensors and
no model execution: items are ids the worker's storage units map to memory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import msgspec

from core.contracts.state import StateDecl

from .pool import Pool


class JointLedger:
    def __init__(self, declarations: Sequence[StateDecl]) -> None:
        grouped: dict[str, list[StateDecl]] = {}
        for declaration in declarations:
            grouped.setdefault(declaration.kind, []).append(declaration)
        if not grouped:
            raise ValueError("the ledger needs at least one state declaration (alloc.states)")
        self.pools = {kind: Pool(decls) for kind, decls in grouped.items()}
        self.ready = False

    def set_capacity(self, items: Mapping[str, int]) -> None:
        if set(items) != set(self.pools):
            raise ValueError(f"capacities for {sorted(items)}, but the declared kinds are {sorted(self.pools)}")
        for kind, count in items.items():
            self.pools[kind].set_capacity(count)
        self.ready = True

    def fits(self, tokens: int) -> bool:
        return self.ready and all(pool.count(tokens) <= pool.capacity for pool in self.pools.values())

    def admit(self, request: int, tokens: int, step: int) -> dict[str, list[int]] | None:
        given: dict[str, list[int]] = {}
        for kind, pool in self.pools.items():
            admitted = pool.admit(request, [], 0, tokens, step)
            if admitted is None:
                self.cancel(request, step)
                return None
            given[kind] = admitted.items
        return given

    def reserve(self, request: int, upto: int, step: int) -> dict[str, list[int]] | None:
        given: dict[str, list[int]] = {}
        for kind, pool in self.pools.items():
            holding = pool.holdings[request]
            items = pool.take(request, pool.need(request, upto - holding.position))
            if items is None:
                self.cancel(request, step)
                return None
            pool.snapshot(request, upto, step)
            given[kind] = items
        return given

    def items(self, request: int) -> dict[str, list[int]]:
        return {kind: list(pool.holdings[request].items) for kind, pool in self.pools.items()}

    def cancel(self, request: int, step: int) -> None:
        for pool in self.pools.values():
            pool.cancel(request, step)

    def release(self, request: int, last_step: int) -> None:
        for pool in self.pools.values():
            pool.release(request, last_step)

    def retire(self, step: int, ran: bool = True) -> None:
        for pool in self.pools.values():
            pool.retire(step, ran)

    def check(self) -> list[str]:
        return [error for pool in self.pools.values() for error in pool.check()]


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        declarations = msgspec.convert(config["alloc.states"], list[StateDecl])
        ctx.provide("ledger@1", JointLedger(declarations))
