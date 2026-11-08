"""The accounting plugin of a state kind, in the scheduler host (stage2-contract.md §5.3). It
shares the declaration (``core.contracts.state``) with the storage unit in the worker.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import msgspec

from core.contracts.state import StateDecl


class Admitted(msgspec.Struct, frozen=True):
    """What one kind gives a request it admits, for the step's ``StateOps``."""

    items: list[int]
    """Every item the request now holds, in table order: the reused ones (cached KV pages,
    read in place), then the new ones. Goes into ``StateAssign``."""
    copies: list[tuple[int, int]] = []
    """``(src, dst)`` to copy before the step (``StateCopy(when="before")``): a cached GDN
    snapshot into the request's new slot."""


class AccountingPlugin(Protocol):
    """The accounting of one state kind in the scheduler host (``ledger.<kind>@1``). Ids
    only, no memory. Each kind keeps its own prefix index (block hash -> items), since kinds
    align and store prefixes differently; ``core.state.alloc`` only orchestrates (§5.3):

    - admission: ``common_prefix`` over every kind's ``reusable``, then ``admit`` in every
      kind; if one kind returns ``None`` the request is not admitted, and the kinds that did
      admit it get ``cancel(request, step)``;
    - planning a step: ``take`` for more items, ``snapshot`` for each row's chunk end; a row
      left out of the plan after all: ``cancel``;
    - each result: ``retire(step)``, then ``publish`` for the blocks it completed (the
      scheduler knows their tokens only now: with overlap, a decode token is on the worker
      until the result arrives); a step answered with ``Resync``: ``retire(step, ran=False)``.

    Items are free, held (by one or more requests; a reserved snapshot by the request it
    snapshots), or cached (published, held by none, evictable). A kind evicts cached items,
    least recently used first, only to give items out; never one a request holds or one a
    pending copy reads."""

    def declare(self, decls: Sequence[StateDecl]) -> None: ...
    def set_capacity(self, items: int) -> None: ...

    def reusable(self, block_hashes: Sequence[int], limit: int) -> int:
        """The longest prefix of a new request, in tokens and at most ``limit``, that this kind
        can resume from: the largest length it supports at or below ``limit``, not the minimum
        of some longest hit. ``block_hashes[i]`` covers tokens ``[0, (i + 1) * B)``, ``B`` being
        the hash block of ``core.state.alloc``; every kind's alignment is a multiple of ``B``.
        Paged KV supports every page boundary up to its longest run of cached blocks; GDN state
        only the snapshots it still holds, so a hit at 96 tokens with snapshots at 64 and 128
        supports 64 when asked for 96. No side effects."""
        ...

    def admit(self, request: int, block_hashes: Sequence[int], length: int, new_tokens: int,
              step: int) -> Admitted | None:
        """Admit the request in ``step``: reuse its first ``length`` tokens (from
        ``common_prefix``, so this kind supports it) and give it items for ``new_tokens`` more.
        All or nothing: ``None`` (nothing taken) if the free and evictable items do not suffice.

        Reused items the request reads in place (KV pages) are held by it until ``release``.
        The source of a copy (a GDN snapshot) is held until the result of ``step`` arrives, so
        it cannot be evicted before the copy has run."""
        ...

    def need(self, request: int, new_tokens: int) -> int:
        """Items an admitted request needs, beyond what it holds, to compute ``new_tokens`` more."""
        ...

    def take(self, request: int, n: int) -> list[int] | None:
        """``n`` more items for an admitted request, evicting cached ones if needed; ``None``
        (nothing taken) if that does not suffice. The scheduler then preempts or waits."""
        ...

    def snapshot(self, request: int, upto: int, step: int) -> list[tuple[int, int]]:
        """While planning ``step``: the request's chunk in it ends at token ``upto``. Returns the
        ``(src, dst)`` copies to make after the step (``StateCopy(when="after")``); needs no
        hashes, so it works while the previous step's tokens are still on the worker.

        GDN can keep only the state at the end of the step: it snapshots when ``upto`` is a
        multiple of ``snapshot_every``, reserving a free or evictable slot as ``dst``, held for
        the request until ``publish`` (none to reserve: no snapshot). Boundaries crossed inside
        the chunk get none (§5.4). Paged KV keeps every page anyway: no copy."""
        ...

    def publish(self, request: int, block_hashes: Sequence[int], upto: int) -> None:
        """The request's tokens up to ``upto`` are computed: the result of every step that
        computed them has arrived, so the scheduler knows their hashes. Register the blocks
        completed up to there; other requests may hit them at once. Never before the result:
        another request could read items the step has not written, or never writes.

        Paged KV registers the request's own pages. GDN registers the snapshot it reserved at a
        boundary up to ``upto``; it becomes cached, no longer held by the request."""
        ...

    def cancel(self, request: int, step: int) -> None:
        """What the request got while planning ``step`` (``admit``, ``take``, ``snapshot``)
        goes back at once: that part of the plan is not sent, so nothing of it will run. New
        items are free again, reused ones and copy sources no longer held by it. For a
        rejected joint admission: waiting for ``step`` with ``release`` could wait forever,
        since a step nobody is admitted to may never be sent."""
        ...

    def release(self, request: int, last_step: int) -> None:
        """The request is done. Its items stop being held only once the result of ``last_step``
        (the last step that used them) has arrived: with overlap, that step may still be running
        when the scheduler learns the request finished (§4.4). Then published items no request
        holds become cached, the others free (a snapshot never published among them). The
        scheduler publishes what the last result completed before releasing. A ``last_step``
        already retired frees at once."""
        ...

    def retire(self, step: int, ran: bool = True) -> None:
        """The result of ``step`` arrived. Free what was released up to it; drop the holds of
        the copies it made. Not ``ran`` (answered with ``Resync``): it wrote nothing, so the
        snapshots it reserved are free again. A request whose ``admit`` copies were in such a
        step goes back to the queue: the scheduler releases it and admits it again from its
        token history (§4.3)."""
        ...

    def check(self) -> list[str]:
        """Invariants: free + held + cached == capacity, no item free and held at once, no
        cached item without a hash, nothing leaked."""
        ...


def common_prefix(kinds: Sequence[Callable[[int], int]], limit: int) -> int:
    """The longest prefix, in tokens and at most ``limit``, every kind can resume from
    (stage2-contract.md §5.3). ``kinds[i](n)`` is kind ``i``'s ``reusable(block_hashes, n)``.

    Ask every kind with the current length, take the lowest answer, ask again until no kind
    lowers it. The longest common length is never above an answer, since each kind answers
    the largest length it supports up to what it was asked; and every kind supports the
    length where it stops. ``limit`` leaves the request at least one token to compute."""
    length = limit
    while True:
        answers = [kind(length) for kind in kinds]
        above = [a for a in answers if a > length]
        if above:  # each answer, not the lowest: a valid one would hide one above the limit
            raise ValueError(f"a kind answered {above[0]} tokens when asked for at most {length}")
        lowest = min(answers, default=length)
        if lowest == length:
            return length
        length = lowest
