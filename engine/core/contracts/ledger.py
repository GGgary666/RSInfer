"""The scheduler's calls to the joint ledger, ``ledger@1`` (stage2-contract.md §5.3, prefix-cache.md).

Only the calls ``core.sched`` makes today. The per-kind pools, the prefix tree and the pools' own
bookkeeping stay inside ``core.state.alloc``. Items are ids per state kind (``StateDecl.kind``);
the ledger never touches device memory.

Steps are numbered by the scheduler (the record epoch) and retire in order: a step retires once
every rank answered it. With one step of overlap (§4.4) a request can be planned in step
``N + 1`` before step ``N`` retired.
"""

from collections.abc import Mapping, Sequence
from typing import NamedTuple, Protocol


class Admitted(NamedTuple):
    """What admitting a request gave it."""

    prefix: int
    """Tokens of its prompt already computed, from the prefix cache: its first step starts there.
    0 without a hit."""
    items: dict[str, list[int]]
    """Items per kind, in table order: the cached prefix's pages first."""


class Reserved(NamedTuple):
    """What one step of a request needs besides the items it holds."""

    items: dict[str, list[int]]
    """Items newly given per kind."""
    restore: dict[str, tuple[int, int]]
    """Per kind, ``(src, dst)``: copy the cached state ``src`` into the request's item ``dst``
    before the step (a prefix hit; ``StateCopy(before)``)."""
    track: dict[str, tuple[int, int]]
    """Per kind, ``(dst, upto)``: the step writes the request's state at position ``upto`` into item
    ``dst`` (``StateTrack``), for the prefix cache."""


class Ledger(Protocol):
    @property
    def ready(self) -> bool:
        """Whether the usable capacity of every state kind is known."""
        ...

    @property
    def align(self) -> int:
        """A prefill chunk that does not end its prompt ends on a multiple of this many tokens
        (the prefix cache keeps whole pages, as SGLang's ``PrefillAdder``); 1: anywhere."""
        ...

    @property
    def max_requests(self) -> int | None:
        """How many requests it holds at most at once (``None``: as many as fit). With the prefix
        cache, the slot kinds' capacity over the slots SGLang counts per request."""
        ...

    def set_capacity(self, items: Mapping[str, int]) -> None:
        """Usable items per state kind, scratch excluded: the lowest any rank reported. Reported again
        while idle: the same numbers change nothing."""
        ...

    def flush(self) -> None:
        """What the items hold may be gone or stale (a change committed, a worker was replaced): drop
        the prefix cache (SGLang's ``flush_cache``). While requests still hold items, nothing is
        matched or kept until the last of them ended, and then it is dropped."""
        ...

    def fits(self, tokens: int) -> bool:
        """Whether a request of ``tokens`` tokens fits the whole capacity, i.e. can ever be admitted."""
        ...

    def match(self, request: int, prompt: Sequence[int]) -> int:
        """How much of ``prompt`` the prefix cache holds now: where the request would start. Takes
        nothing."""
        ...

    def admit(self, request: int, prompt: Sequence[int], tokens: int, step: int) -> Admitted | None:
        """Take every kind's items for ``tokens`` tokens (prompt and output), the cached prefix of
        ``prompt`` included, all or nothing, for a request first planned in ``step``. ``None``:
        some kind did not have them, and nothing is held."""
        ...

    def reserve(self, request: int, upto: int, step: int, speculative: bool = False) -> Reserved | None:
        """``step`` computes the request's input up to position ``upto`` (exclusive). ``None``: what
        it needs is not there, and what ``step`` took for the request is given back at once (as
        ``cancel``). ``speculative``: a decode step that verifies drafts (doc/specs/mtp.md): it
        writes up to ``upto`` at most, advances by what is accepted (``retire``), and a GDN state it
        keeps for the prefix cache is at the boundary it crosses, if any (``StateTrack`` ``upto -grid``)."""
        ...

    def items(self, request: int) -> dict[str, list[int]]:
        """Every item the request holds per kind, in table order: what a full delta assigns."""
        ...

    def cancel(self, request: int, step: int) -> None:
        """Give back what ``step`` took for the request: the step is not sent with it."""
        ...

    def commit(self, request: int, prompt: Sequence[int]) -> None:
        """A step that computed part of the request's prompt retired, and the request goes on: what
        it computed enters the prefix cache (SGLang's ``checkpoint_kv_cache``)."""
        ...

    def restart(self, request: int) -> bool:
        """After ``flush`` (a worker was replaced): the request, admitted from the prefix cache and
        not run yet, computes its prompt again from position 0, on items of its own and with no
        state to restore. ``False``: it cannot (no free items, or no cache to start over from), and
        should end."""
        ...

    def release(self, request: int, last_step: int, tokens: Sequence[int] = ()) -> None:
        """The request is done; ``tokens`` are the ones it computed, prompt and output. Once
        ``last_step`` retired, and not before (a step still running may write its items, §4.4
        item 5), what it computed enters the prefix cache and its other items return."""
        ...

    def retire(self, step: int, ran: bool = True, advanced: Mapping[int, int] | None = None) -> None:
        """Every rank answered ``step``. ``ran=False`` (``Resync``): its input was not computed, so
        the request's position goes back to before the step; its items stay with it. ``advanced``:
        request -> tokens its speculative reservation of the step accepted."""
        ...

    def check(self) -> list[str]:
        """Invariant violations; empty when consistent."""
        ...
