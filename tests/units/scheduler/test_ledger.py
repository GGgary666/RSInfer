"""``core.state.alloc``: joint admission, deferred release, and putting a step back (``Resync``)."""

from __future__ import annotations

import pytest

from core.contracts.state import StateDecl
from core.state.alloc.pool import Pool
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
