"""Basic non-caching accounting, adapted from mini-sglang TableManager/CacheManager.

Source: mini-sglang 9a91cfa, MIT (LICENSE.mini-sglang). Retains free-slot allocation
and ceil(page) accounting, with host IDs instead of device index tensors. Epoch
reservations, immediate cancellation and deferred release implement our contract.
Prefix reuse is deliberately disabled in this first ledger, as in NaivePrefixCache.

With one step of overlap a request has reservations in two steps at once, and when the
first is answered with ``Resync`` the second is too (the worker runs nothing until a full
delta). Each puts the position back to where its own step started, at most: retiring the
second must not move the request forward past the first, which did not run either.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from core.contracts.state import StateDecl


@dataclass
class Admitted:
    items: list[int]


@dataclass
class Holding:
    items: list[int]
    position: int = 0
    tentative: list[int] = field(default_factory=list[int])


@dataclass
class Reservation:
    previous: int
    added: list[int]
    admission: bool = False


class Pool:
    def __init__(self, declarations: Sequence[StateDecl]) -> None:
        self.declare(declarations)
        self.capacity = 0
        self.free: list[int] = []
        self.holdings: dict[int, Holding] = {}
        self.reservations: dict[tuple[int, int], Reservation] = {}
        self.releases: dict[int, int] = {}
        self.retired = -1

    def declare(self, decls: Sequence[StateDecl]) -> None:
        if not decls:
            raise ValueError("a pool needs a state declaration")
        first = decls[0]
        if first.lifetime not in ("request", "slot") or any(d.snapshot_every for d in decls):
            raise ValueError("this ledger supports uncached request pages and request slots")
        if any((d.kind, d.lifetime, d.tokens_per_item) !=
               (first.kind, first.lifetime, first.tokens_per_item) for d in decls):
            raise ValueError("layers sharing this pool must use the same item geometry")
        self.kind, self.page = first.kind, first.tokens_per_item
        self.slot = first.lifetime == "slot"
        if not self.slot and self.page <= 0:
            raise ValueError("page size must be positive")

    def count(self, tokens: int) -> int:
        return 1 if self.slot else (tokens + self.page - 1) // self.page

    def set_capacity(self, items: int) -> None:
        """The workers' usable items (``Capacity``). It may shrink only while nothing is held: a
        change applies to an empty engine (doc/specs/empty-apply.md), and rebuilding a store there
        can change its size (a new dtype, another holder budget)."""
        if items < self.capacity:
            if self.holdings or self.reservations or self.releases:
                raise ValueError("a state pool shrinks only while it holds nothing")
            self.free = list(range(items))
        else:
            self.free.extend(range(self.capacity, items))
        self.capacity = items

    def reusable(self, block_hashes: Sequence[int], limit: int) -> int:
        return 0  # prefix caching is off, just as in mini-sglang's NaivePrefixCache

    def _allocate(self, n: int) -> list[int] | None:
        if n > len(self.free):
            return None
        return [self.free.pop() for _ in range(n)]

    def admit(self, request: int, block_hashes: Sequence[int], length: int, new_tokens: int,
              step: int) -> Admitted | None:
        if request in self.holdings or length:
            raise ValueError("request already admitted or prefix caching was requested")
        items = self._allocate(self.count(new_tokens))
        if items is None:
            return None
        self.holdings[request] = Holding(items)
        self.reservations[step, request] = Reservation(0, list(items), True)
        return Admitted(list(items))

    def need(self, request: int, new_tokens: int) -> int:
        holding = self.holdings[request]
        return max(0, self.count(holding.position + new_tokens) - len(holding.items))

    def take(self, request: int, n: int) -> list[int] | None:
        items = self._allocate(n)
        if items is not None:
            holding = self.holdings[request]
            holding.items.extend(items)
            holding.tentative.extend(items)
        return items

    def snapshot(self, request: int, upto: int, step: int) -> list[tuple[int, int]]:
        holding = self.holdings[request]
        reservation = self.reservations.setdefault((step, request), Reservation(holding.position, []))
        reservation.added.extend(holding.tentative)
        holding.tentative.clear()
        holding.position = upto
        return []  # declarations require snapshot_every=0

    def publish(self, request: int, block_hashes: Sequence[int], upto: int) -> None:
        # The non-caching mode retains the request's items only until release.
        if upto > self.holdings[request].position:
            raise ValueError("cannot publish uncomputed input")

    def cancel(self, request: int, step: int) -> None:
        holding = self.holdings.get(request)
        if holding is None:
            return
        reservation = self.reservations.pop((step, request), None)
        added = [*(reservation.added if reservation else []), *holding.tentative]
        holding.tentative.clear()
        remove = set(added)
        holding.items[:] = [item for item in holding.items if item not in remove]
        self.free.extend(added)
        if reservation is not None:
            holding.position = reservation.previous
            if reservation.admission:
                del self.holdings[request]

    def release(self, request: int, last_step: int) -> None:
        if request not in self.holdings:
            return
        self.releases[request] = last_step
        self._free_ready()

    def _free_ready(self) -> None:
        for request, last in list(self.releases.items()):
            if last <= self.retired:
                self.free.extend(self.holdings.pop(request).items)
                del self.releases[request]

    def retire(self, step: int, ran: bool = True) -> None:
        for key, reservation in list(self.reservations.items()):
            epoch, request = key
            if epoch != step:
                continue
            if not ran:
                holding = self.holdings[request]
                holding.position = min(holding.position, reservation.previous)
            del self.reservations[key]
        self.retired = max(self.retired, step)
        self._free_ready()

    def check(self) -> list[str]:
        ids = [*self.free, *(item for holding in self.holdings.values() for item in holding.items)]
        return [] if sorted(ids) == list(range(self.capacity)) else [f"{self.kind}: free/held items disagree"]
