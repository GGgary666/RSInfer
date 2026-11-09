"""Ledger policy tests; no shared-schema dependency on the algorithm."""

from collections.abc import Callable

import pytest

from core.state.alloc.prefix import common_prefix


def kv(cached: int, page: int = 16) -> Callable[[int], int]:
    """Paged KV with ``cached`` tokens of leading blocks cached: every page boundary up to that."""
    return lambda n: min(n // page * page, cached)


def gdn(snapshots: list[int]) -> Callable[[int], int]:
    """GDN state: only the snapshots it still holds."""
    return lambda n: max([s for s in snapshots if s <= n], default=0)


def test_the_common_prefix_is_one_every_kind_supports_not_the_shortest_longest_hit() -> None:
    """KV hits 96 tokens, GDN has snapshots at 64 and 128: the shorter of the two longest hits
    is 96, where GDN has no state; the request resumes from 64."""
    kinds = [kv(96), gdn([64, 128])]
    assert min(kind(128) for kind in kinds) == 96 and gdn([64, 128])(96) == 64
    assert common_prefix(kinds, 128) == 64
    assert common_prefix([kv(96), gdn([128])], 128) == 0  # the snapshot at 64 was evicted
    assert common_prefix([kv(256), gdn([64, 128, 192])], 200) == 192
    assert common_prefix([kv(512, page=32), gdn([96, 192, 384])], 400) == 384
    assert common_prefix([kv(1000, page=32), gdn([96, 160]), gdn([64, 128, 192])], 400) == 0
    assert common_prefix([kv(1000), gdn([64, 128, 192]), gdn([128, 256])], 400) == 128
    assert common_prefix([], 48) == 48


def test_a_kind_that_answers_above_its_limit_is_refused() -> None:
    """Every answer is checked, not only the lowest: a valid answer must not hide one above the limit."""
    with pytest.raises(ValueError, match="answered 80 tokens when asked for at most 64"):
        common_prefix([lambda n: n + 16], 64)
    with pytest.raises(ValueError, match="answered 80 tokens when asked for at most 64"):
        common_prefix([lambda n: n + 16, lambda n: n], 64)
    with pytest.raises(ValueError, match="answered 48 tokens when asked for at most 32"):
        common_prefix([kv(32), lambda n: 48], 64)
