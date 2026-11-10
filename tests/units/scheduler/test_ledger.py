"""``core.state.alloc``: joint admission, deferred release, and putting a step back (``Resync``)."""

from __future__ import annotations

import random
import time

import pytest

from core.contracts.state import StateDecl
from core.state.alloc.pool import Holding, Pool
from core.state.alloc.unit import JointLedger

PAGES = StateDecl("kv.paged", 0, {}, "request", tokens_per_item=16)
SLOTS = StateDecl("gdn.slots", 1, {}, "slot")


def test_joint_admission_gives_back_what_one_kind_took_when_another_refuses() -> None:
    ledger = JointLedger([PAGES, SLOTS])
    ledger.set_capacity({"kv.paged": 4, "gdn.slots": 0})
    assert ledger.admit(7, 32, 1) is None
    assert ledger.pools["kv.paged"].holdings == {} and len(ledger.pools["kv.paged"].free) == 4
    ledger.set_capacity({"kv.paged": 4, "gdn.slots": 1})
    assigned = ledger.admit(8, 17, 1)
    assert assigned is not None and len(assigned["kv.paged"]) == 2 and len(assigned["gdn.slots"]) == 1
    assert ledger.check() == []


def test_release_waits_for_the_last_step_but_cancel_undoes_only_the_unsent_one() -> None:
    pool = Pool([PAGES])
    pool.set_capacity(4)
    first = pool.admit(7, [], 0, 16, 1)
    assert first is not None
    pool.snapshot(7, 16, 1)
    pool.retire(1)
    assert pool.take(7, pool.need(7, 16)) is not None
    pool.snapshot(7, 32, 2)
    pool.cancel(7, 2)
    assert pool.holdings[7].items == first.items and pool.holdings[7].position == 16
    assert pool.take(7, pool.need(7, 16)) is not None
    pool.snapshot(7, 32, 3)
    pool.release(7, 3)
    assert len(pool.free) == 2
    pool.retire(2)
    assert len(pool.free) == 2
    pool.retire(3)
    assert len(pool.free) == 4 and pool.check() == []


def test_a_step_not_run_puts_the_position_back_and_keeps_the_items() -> None:
    ledger = JointLedger([PAGES, SLOTS])
    ledger.set_capacity({"kv.paged": 4, "gdn.slots": 2})
    assigned = ledger.admit(7, 48, 1)
    assert assigned is not None and ledger.reserve(7, 8, 1) is not None
    ledger.retire(1, ran=False)
    assert ledger.items(7) == assigned
    assert all(pool.holdings[7].position == 0 for pool in ledger.pools.values())
    ledger.release(7, 1)
    assert ledger.check() == [] and all(not pool.holdings for pool in ledger.pools.values())


def test_with_overlap_the_second_refused_step_does_not_move_the_request_forward() -> None:
    """Steps 2 and 3 are both planned before step 2 is answered; the workers refuse both. Putting
    step 3 back must not leave the request at the end of step 2, which did not run either."""
    ledger = JointLedger([PAGES, SLOTS])
    ledger.set_capacity({"kv.paged": 8, "gdn.slots": 2})
    assert ledger.admit(7, 64, 1) is not None and ledger.reserve(7, 16, 1) is not None
    ledger.retire(1)
    assert ledger.reserve(7, 17, 2) is not None and ledger.reserve(7, 18, 3) is not None
    ledger.retire(2, ran=False)
    ledger.retire(3, ran=False)
    assert all(pool.holdings[7].position == 16 for pool in ledger.pools.values())
    assert ledger.check() == []


def test_this_ledger_does_not_pretend_to_keep_prefix_snapshots() -> None:
    with pytest.raises(ValueError, match="uncached"):
        Pool([StateDecl("gdn.slots", 0, {}, "slot", snapshot_every=64)])
    pool = Pool([PAGES])
    pool.set_capacity(2)
    assert pool.reusable([1, 2], 32) == 0
    assert pool.admit(1, [], 0, 48, 1) is None
    assert len(pool.free) == 2 and pool.check() == []


def test_capacity_must_name_the_declared_kinds() -> None:
    ledger = JointLedger([PAGES, SLOTS])
    with pytest.raises(ValueError, match="declared kinds"):
        ledger.set_capacity({"kv.paged": 4})
    assert not ledger.fits(16)


def test_the_ledger_refuses_pages_of_another_size_than_the_workers() -> None:
    from core.state.alloc.unit import check_page_size

    check_page_size([PAGES, SLOTS], 16)
    with pytest.raises(ValueError, match="pages of 16 tokens, the workers' state.kv.page_size is 32"):
        check_page_size([PAGES, SLOTS], 32)


def test_a_pool_shrinks_only_while_it_holds_nothing() -> None:
    """A change rebuilding a store applies to an empty engine (doc/specs/empty-apply.md), and the
    workers may then report fewer items (a new dtype, another holder budget)."""
    ledger = JointLedger([PAGES, SLOTS])
    ledger.set_capacity({"kv.paged": 8, "gdn.slots": 4})
    assert ledger.admit(1, 20, 1) is not None
    with pytest.raises(ValueError, match="shrinks only while it holds nothing"):
        ledger.set_capacity({"kv.paged": 4, "gdn.slots": 4})
    ledger.cancel(1, 1)
    ledger.set_capacity({"kv.paged": 4, "gdn.slots": 2})
    assert sorted(ledger.pools["kv.paged"].free) == [0, 1, 2, 3] and sorted(ledger.pools["gdn.slots"].free) == [0, 1]
    assert ledger.check() == []


def test_a_pool_hands_out_the_items_a_listed_stack_would() -> None:
    """The free items are a stack, as mini-sglang's free-slot list is: the last given back goes out
    first, and under them the never-used items from the highest down. The pool keeps that bottom
    as a count; here it is held against the list itself, growing while items are out."""
    pool = Pool([PAGES])
    pool.set_capacity(40)
    listed, out = list(range(40)), []
    rng = random.Random(0)
    for _ in range(3000):
        roll = rng.random()
        if roll < 0.5:
            n = rng.randint(0, 5)
            want = [listed.pop() for _ in range(n)] if n <= len(listed) else None
            assert pool._allocate(n) == want
            out += want or []
        elif roll < 0.95 and out:
            back = [out.pop(rng.randrange(len(out))) for _ in range(rng.randint(1, len(out)))]
            pool.free.extend(back)
            listed.extend(back)
        elif out:  # with items out, new capacity goes on top, as extending the list put it
            pool.holdings[0] = Holding(list(out))
            grown = pool.capacity + rng.randint(0, 6)
            listed.extend(range(pool.capacity, grown))
            pool.set_capacity(grown)
            del pool.holdings[0]
        assert list(pool.free) == listed and len(pool.free) == len(listed)


def test_a_pool_of_a_billion_items_lists_none_of_them() -> None:
    """A SM100 gives a small model 159 million KV pages. Listing them took seconds in one call that
    held the GIL, the scheduler host's stream heartbeats stopped, and the worker took it for gone:
    the first request ended with "results of rank 0 come from a new producer"."""
    pool = Pool([PAGES])
    started = time.perf_counter()
    pool.set_capacity(10 ** 9)
    assert time.perf_counter() - started < 0.5 and len(pool.free) == 10 ** 9 and pool.free.returned == []
    admitted = pool.admit(7, [], 0, 40, 1)
    assert admitted is not None and admitted.items == [10 ** 9 - 1, 10 ** 9 - 2, 10 ** 9 - 3]
    assert pool.check() == []
    pool.retire(1)
    pool.release(7, 1)
    assert len(pool.free) == 10 ** 9 and sorted(pool.free.returned) == admitted.items[::-1] and pool.check() == []
    pool.set_capacity(2 * 10 ** 9)  # nothing held: again nothing to list
    assert len(pool.free) == 2 * 10 ** 9 and pool.free.returned == [] and pool.check() == []
    with pytest.raises(ValueError, match="shrinks only while it holds nothing"):
        pool.admit(8, [], 0, 16, 2)
        pool.set_capacity(10)
