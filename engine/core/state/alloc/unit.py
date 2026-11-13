"""``ledger@1``: joint admission over every state kind, from #22, or, with ``alloc.radix_cache``
(the default, as SGLang's), the prefix cache of ``radix.py`` (doc/specs/prefix-cache.md).

Each kind's ``Pool`` keeps its own ids; a request is admitted only when every kind has what it
needs, and what one kind gave is given back at once when another refuses (§5.3). No tensors and
no model execution: items are ids the worker's storage units map to memory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import msgspec

from core.contracts.ledger import Admitted, Reserved
from core.contracts.state import StateDecl

from .pool import Pool
from .radix import RadixLedger


class JointLedger:
    def __init__(self, declarations: Sequence[StateDecl]) -> None:
        grouped: dict[str, list[StateDecl]] = {}
        for declaration in declarations:
            grouped.setdefault(declaration.kind, []).append(declaration)
        if not grouped:
            raise ValueError("the ledger needs at least one state declaration (alloc.states)")
        self.pools = {kind: Pool(decls) for kind, decls in grouped.items()}
        self.ready = False

    @property
    def align(self) -> int:
        return 1

    @property
    def max_requests(self) -> int | None:
        return None

    def set_capacity(self, items: Mapping[str, int]) -> None:
        if set(items) != set(self.pools):
            raise ValueError(f"capacities for {sorted(items)}, but the declared kinds are {sorted(self.pools)}")
        for kind, count in items.items():
            self.pools[kind].set_capacity(count)
        self.ready = True

    def fits(self, tokens: int) -> bool:
        return self.ready and all(pool.count(tokens) <= pool.capacity for pool in self.pools.values())

    def match(self, request: int, prompt: Sequence[int]) -> int:
        return 0

    def admit(self, request: int, prompt: Sequence[int], tokens: int, step: int) -> Admitted | None:
        given: dict[str, list[int]] = {}
        for kind, pool in self.pools.items():
            admitted = pool.admit(request, [], 0, tokens, step)
            if admitted is None:
                self.cancel(request, step)
                return None
            given[kind] = admitted.items
        return Admitted(0, given)

    def reserve(self, request: int, upto: int, step: int, speculative: bool = False) -> Reserved | None:
        """Without a cache, speculation changes nothing here: the pages up to ``upto`` are what a
        speculative step may write, and a position counts pages only."""
        given: dict[str, list[int]] = {}
        for kind, pool in self.pools.items():
            holding = pool.holdings[request]
            items = pool.take(request, pool.need(request, upto - holding.position))
            if items is None:
                self.cancel(request, step)
                return None
            pool.snapshot(request, upto, step)
            given[kind] = items
        return Reserved(given, {}, {})

    def items(self, request: int) -> dict[str, list[int]]:
        return {kind: list(pool.holdings[request].items) for kind, pool in self.pools.items()}

    def cancel(self, request: int, step: int) -> None:
        for pool in self.pools.values():
            pool.cancel(request, step)

    def commit(self, request: int, prompt: Sequence[int]) -> None:
        pass

    def restart(self, request: int) -> bool:
        return False  # no request starts from a cache here

    def flush(self) -> None:
        pass

    def release(self, request: int, last_step: int, tokens: Sequence[int] = ()) -> None:
        for pool in self.pools.values():
            pool.release(request, last_step)

    def retire(self, step: int, ran: bool = True, advanced: Mapping[int, int] | None = None) -> None:
        for pool in self.pools.values():
            pool.retire(step, ran)

    def check(self) -> list[str]:
        return [error for pool in self.pools.values() for error in pool.check()]


def check_page_size(declarations: Sequence[StateDecl], page_size: int) -> None:
    """The ledger counts KV pages of the size the worker's store has (``state.kv.page_size``); a
    different size would make it give rows fewer pages than they fill."""
    for d in declarations:
        if d.kind == "kv.paged" and d.tokens_per_item != page_size:
            raise ValueError(f"alloc.states has kv.paged pages of {d.tokens_per_item} tokens, the workers' "
                             f"state.kv.page_size is {page_size}")


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        declarations = msgspec.convert(config["alloc.states"], list[StateDecl])
        check_page_size(declarations, config["state.kv.page_size"])
        if config["alloc.radix_cache"] and config["spec.algorithm"] and \
                config["alloc.mamba_track_interval"] < config["spec.num_draft_tokens"]:
            # A verify step crosses one tracking boundary at most (SGLang's mamba_hook).
            raise ValueError(f"alloc.mamba_track_interval = {config['alloc.mamba_track_interval']} must be at least "
                             f"spec.num_draft_tokens = {config['spec.num_draft_tokens']} (as SGLang's)")
        if config["alloc.radix_cache"]:
            ctx.provide("ledger@1", RadixLedger(declarations, track_interval=config["alloc.mamba_track_interval"],
                                                max_states_per_path=config["alloc.mamba_max_states_per_path"]))
        else:
            ctx.provide("ledger@1", JointLedger(declarations))
