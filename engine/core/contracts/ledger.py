"""The scheduler's calls to the joint ledger, ``ledger@1`` (stage2-contract.md §5.3).

Only the calls ``core.sched`` makes today. The per-kind pools, prefix matching and the pools'
own bookkeeping stay inside ``core.state.alloc``. Items are ids per state kind (``StateDecl.kind``);
the ledger never touches device memory.

Steps are numbered by the scheduler (the record epoch) and retire in order: a step retires once
every rank answered it. With one step of overlap (§4.4) a request can be planned in step
``N + 1`` before step ``N`` retired.
"""

from collections.abc import Mapping
from typing import Protocol


class Ledger(Protocol):
    @property
    def ready(self) -> bool:
        """Whether the usable capacity of every state kind is known."""
        ...

    def set_capacity(self, items: Mapping[str, int]) -> None:
        """Usable items per state kind, scratch excluded: the lowest any rank reported."""
        ...

    def fits(self, tokens: int) -> bool:
        """Whether a request of ``tokens`` tokens fits the whole capacity, i.e. can ever be admitted."""
        ...

    def admit(self, request: int, tokens: int, step: int) -> dict[str, list[int]] | None:
        """Take every kind's items for ``tokens`` tokens, all or nothing, for a request first
        planned in ``step``. Returns the items per kind; ``None``: some kind did not have them, and
        nothing is held."""
        ...

    def reserve(self, request: int, upto: int, step: int) -> dict[str, list[int]] | None:
        """``step`` computes the request's input up to position ``upto`` (exclusive). Returns the
        items it newly needs per kind; ``None``: they are not there, and what ``step`` took for
        the request is given back at once (as ``cancel``)."""
        ...

    def items(self, request: int) -> dict[str, list[int]]:
        """Every item the request holds per kind, in table order: what a full delta assigns."""
        ...

    def cancel(self, request: int, step: int) -> None:
        """Give back what ``step`` took for the request: the step is not sent with it."""
        ...

    def release(self, request: int, last_step: int) -> None:
        """The request is done. Its items return once ``last_step`` retired, not before: a step
        still running may write them (§4.4 item 5)."""
        ...

    def retire(self, step: int, ran: bool = True) -> None:
        """Every rank answered ``step``. ``ran=False`` (``Resync``): its input was not computed, so
        the request's position goes back to before the step; its items stay with it."""
        ...

    def check(self) -> list[str]:
        """Invariant violations; empty when consistent."""
        ...
