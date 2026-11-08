"""Prefix matching policy owned by the ledger, not a shared interface."""

from collections.abc import Callable, Sequence


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
