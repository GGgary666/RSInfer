"""The prefix cache of ``core.state.alloc`` (``radix.py``, doc/specs/prefix-cache.md): SGLang's radix
cache with the mamba ``extra_buffer`` strategy, on ids alone."""

from __future__ import annotations

from array import array

import pytest

from core.contracts.state import StateDecl
from core.state.alloc.radix import GDN, KV, SWA, RadixLedger, RadixTree
from state.prefix_checkpoint import PrefixNode

PAGE = 16
DECLS = [StateDecl(KV, 0, {"k": ((PAGE, 1, 8), "bf16")}, "request", tokens_per_item=PAGE),
         StateDecl(GDN, 1, {"s": ((8, 8), "f32")}, "slot")]


def paged(page: int) -> list[StateDecl]:
    return [StateDecl(KV, 0, {"k": ((page, 1, 8), "bf16")}, "request", tokens_per_item=page), DECLS[1]]


def ledger(pages: int = 64, slots: int = 20, **kw: int) -> RadixLedger:
    led = RadixLedger(DECLS, **kw)
    led.set_capacity({KV: pages, GDN: slots})
    return led


def run(led: RadixLedger, rid: int, prompt: list[int], step: int, upto: int, *, admit: int = 0) -> dict[str, object]:
    """One step of ``rid`` up to ``upto``: admitted first when ``admit`` (its extent) is given; retired."""
    out: dict[str, object] = {}
    if admit:
        got = led.admit(rid, prompt, admit, step)
        assert got is not None
        out["admitted"] = got
    r = led.reserve(rid, upto, step)
    assert r is not None
    out["reserved"] = r
    led.retire(step)
    return out


def test_dense_attention_shares_pages_inside_a_radix_node_without_slot_state() -> None:
    led = RadixLedger(DECLS[:1])
    led.set_capacity({KV: 8})
    prompt = list(range(3 * PAGE + 5))
    run(led, 1, prompt, 1, len(prompt), admit=len(prompt))
    original_pages = led.items(1)[KV]
    led.commit(1, prompt)
    led.release(1, 1, prompt)
    branch = [*prompt[:2 * PAGE], 1000, 1001]
    assert led.match(2, branch) == 2 * PAGE
    assert led.match(3, prompt) == 3 * PAGE
    assert led.match(4, prompt[:2 * PAGE]) == PAGE  # the last prompt token must still run
    admitted = led.admit(2, branch, len(branch), 2)
    assert admitted is not None and admitted.prefix == 2 * PAGE
    assert admitted.items[KV][:2] == original_pages[:2]
    step = led.reserve(2, len(branch), 2)
    assert step is not None and not step.restore and not step.track
    led.retire(2, ran=False)
    led.release(2, 2, branch)
    assert led.check() == [] and led.tree.held()[1] == []


def test_hybrid_prefix_checkpoint_preserves_salted_paths_and_accounts_for_saved_slots() -> None:
    original = ledger()
    prompt = list(range(4 * PAGE + 7))
    other = [*prompt[:PAGE], *range(100, 100 + 3 * PAGE + 7)]
    for rid, tokens, salt in ((1, prompt, "pool"), (2, other, "pool"), (3, prompt, "isolated")):
        entry = original.admit(rid, tokens, len(tokens), rid, salt)
        assert entry is not None and entry.prefix == 0
        assert original.reserve(rid, len(tokens), rid) is not None
        original.retire(rid)
        original.commit(rid, tokens)
        original.release(rid, rid, tokens)
    assert not original.check()

    restored = RadixLedger(DECLS)
    restored.restore_nodes = original.save_prefix()
    restored.set_capacity({KV: 64, GDN: 20})
    assert not restored.check()
    assert restored.match(4, prompt, "pool") == 4 * PAGE
    assert restored.match(5, other, "pool") == 4 * PAGE
    assert restored.match(6, prompt, "isolated") == 4 * PAGE
    assert restored.match(7, prompt, "new salt") == 0
    admitted = restored.admit(8, prompt, len(prompt), 8, "pool")
    assert admitted is not None and admitted.prefix == 4 * PAGE
    reserved = restored.reserve(8, len(prompt), 8)
    assert reserved is not None and GDN in reserved.restore
    source, target = reserved.restore[GDN]
    assert source != target and target == admitted.items[GDN][0]
    assert not restored.check()


@pytest.mark.parametrize("accepted", [False, True])
def test_peer_prefix_publication_requires_copy_acknowledgement(accepted: bool) -> None:
    source, destination = ledger(), ledger()
    prefix = list(range(4 * PAGE))
    prompt = prefix + [1024]
    pages = [8, 3, 9, 0]
    source.load_prefix([PrefixNode(-1, array("q", prefix).tobytes(), pages, 10, b"peer", 1)])
    lease = source.pin_replica(1, prompt, "peer")
    assert lease is not None and lease.items == {KV: pages, GDN: [10]}
    assert destination.stage_replica(1, prompt, "peer", len(prefix), len(prompt) + 16) is not None
    assert destination.match(1, prompt, "peer") == 0
    assert not source.check() and not destination.check()
    destination.finish_replica(prefix, "peer", accepted)
    lease.release()
    assert source.match(1, prompt, "peer") == len(prefix)
    assert destination.match(1, prompt, "peer") == (len(prefix) if accepted else 0)
    assert destination.match(2, prompt, "independent") == 0
    assert not source.check() and not destination.check()


def test_a_prompt_computed_once_is_where_the_next_one_starts_from_its_last_checkpoint() -> None:
    led = ledger()
    prompt = list(range(1000, 1200))  # 200 tokens: the checkpoint grid (64) puts the state at 192
    a = run(led, 1, prompt, 1, 200, admit=210)
    assert a["admitted"].prefix == 0  # type: ignore[attr-defined]
    assert a["reserved"].track[GDN][1] == 192 and not a["reserved"].restore  # type: ignore[attr-defined]
    led.commit(1, prompt)
    assert led.match(2, prompt) == 192 and led.check() == []

    b = led.admit(2, prompt, 210, 2)
    assert b is not None and b.prefix == 192
    assert b.items[KV][:12] == led.items(1)[KV][:12]  # the 12 pages of 192 tokens are shared
    r = led.reserve(2, 200, 2)
    assert r is not None and r.restore[GDN][1] == b.items[GDN][0]  # cached state into its own slot
    assert GDN not in r.track  # 8 tokens: no checkpoint
    led.retire(2)
    assert led.reserve(2, 201, 3) is not None and not led.reserve(2, 201, 3).restore  # type: ignore[union-attr]
    assert led.check() == []


def test_a_request_that_shares_part_of_a_cached_path_stores_the_branching_point() -> None:
    led = ledger()
    a = list(range(1000, 1200))
    run(led, 1, a, 1, 200, admit=210)
    led.commit(1, a)
    c = [*a[:150], *range(5000, 5100)]  # shares 150 tokens: 9 pages of KV, no state below 192
    assert led.match(3, c) == 0
    got = led.admit(3, c, 260, 2)
    assert got is not None and got.prefix == 0
    r = led.reserve(3, 250, 2)
    assert r is not None and r.track[GDN][1] == 128  # the branching point: KV hit 144, on the grid
    led.retire(2)
    led.commit(3, c)
    assert led.match(4, [*a[:150], 7, 7, 7]) == 128  # a third request starts there now
    assert led.match(5, a) == 192  # and the first path is still there
    assert led.check() == []


def test_a_finished_request_leaves_its_decode_checkpoint_and_frees_what_the_tree_did_not_take() -> None:
    led = ledger(track_interval=64)
    prompt = list(range(100, 130))  # 30 tokens: no prefill checkpoint
    run(led, 1, prompt, 1, 30, admit=80)
    for step, upto in enumerate(range(31, 66), start=2):  # decode up to 65 tokens
        r = run(led, 1, prompt, step, upto)["reserved"]
        assert (GDN in r.track) == (upto == 64)  # type: ignore[attr-defined]
    tokens = [*prompt, *range(900, 935)]
    led.release(1, 99, tokens)
    led.retire(99)
    assert not led.holdings and led.check() == []
    assert led.match(2, [*tokens, 1]) == 64
    pages, states = led.tree.held()
    assert len(pages) == 4 and len(states) == 1  # the KV up to the checkpoint and its state
    assert len(led.free[KV]) == 64 - 4 and len(led.free[GDN]) == 20 - 1


def test_a_step_that_did_not_run_tracks_and_restores_nothing() -> None:
    led = ledger()
    prompt = list(range(200))
    run(led, 1, prompt, 1, 200, admit=210)
    led.commit(1, prompt)
    assert led.admit(2, prompt, 210, 2) is not None
    r = led.reserve(2, 200, 2)
    assert r is not None and r.restore
    led.retire(2, ran=False)  # Resync: the copy and the step did not happen
    again = led.reserve(2, 200, 3)
    assert again is not None and again.restore == r.restore  # sent again
    led.retire(3)
    assert led.holdings[2].restored and led.check() == []


def test_with_overlap_two_steps_in_flight_track_into_the_two_slots() -> None:
    led = ledger()
    prompt = list(range(300))
    assert led.admit(1, prompt, 310, 1) is not None
    first = led.reserve(1, 128, 1)
    second = led.reserve(1, 256, 2)  # planned before step 1 retired
    assert first is not None and second is not None
    assert first.track[GDN][0] != second.track[GDN][0]
    led.retire(1)
    led.commit(1, prompt)  # donates step 1's slot; step 2 writes the other one
    assert led.match(9, prompt) == 128
    led.retire(2)
    led.commit(1, prompt)
    assert led.match(9, prompt) == 256 and led.check() == []


def test_eviction_takes_unlocked_leaves_least_recently_used_and_never_a_running_requests_prefix() -> None:
    led = ledger(pages=30, slots=10)  # 2 requests at once (5 slots each)
    a, b = list(range(200)), list(range(1000, 1200))
    run(led, 1, a, 1, 200, admit=210)
    led.commit(1, a)
    led.release(1, 1, a)  # its 192 tokens stay in the tree, 12 pages
    assert led.check() == [] and len(led.free[KV]) == 30 - 12
    got = led.admit(2, a, 210, 2)  # holds a's prefix
    assert got is not None and got.prefix == 192
    assert led.admit(3, b, 290, 2) is None  # 19 pages wanted, 16 free, and a's 12 are held by request 2
    led.cancel(2, 2)
    assert led.admit(3, b, 290, 3) is not None  # now a's cached path is evicted for it
    assert led.match(4, a) == 0 and led.check() == []


def test_at_most_the_slots_over_five_requests_are_admitted() -> None:
    led = ledger(slots=10)
    assert led.max_requests == 2
    assert led.admit(1, [1] * 20, 30, 1) is not None and led.admit(2, [2] * 20, 30, 1) is not None
    assert led.admit(3, [3] * 20, 30, 1) is None


def test_a_bad_configuration_is_refused() -> None:
    with pytest.raises(ValueError, match="alloc.radix_cache = false"):
        RadixLedger([DECLS[1]])
    with pytest.raises(ValueError, match="mamba_max_states_per_path"):
        RadixLedger(DECLS, max_states_per_path=0)
    # As SGLang's (Antigravity, #59): a page of 96 would keep a state at 96, where the GDN
    # kernels have none (the worker refuses it and the engine stops).
    with pytest.raises(ValueError, match="must divide the other"):
        RadixLedger(paged(96), track_interval=192)
    with pytest.raises(ValueError, match="mamba_track_interval = 200 must be a multiple"):
        RadixLedger(paged(16), track_interval=200)
    assert RadixLedger(paged(128)).grid == 128 and RadixLedger(paged(32)).grid == 64


def test_a_flush_drops_the_cache_once_no_request_holds_items_and_a_repeated_capacity_keeps_it() -> None:
    led = ledger()
    a = list(range(200))
    run(led, 1, a, 1, 200, admit=210)
    led.commit(1, a)
    led.set_capacity({KV: 64, GDN: 20})  # an idle worker says it again
    assert led.match(2, a) == 192
    assert led.admit(2, a, 210, 2) is not None
    assert led.match(3, a) == 192
    led.flush()  # a change committed while request 2 still holds a's prefix
    assert led.match(3, a) == 0 and led.admit(3, a, 210, 2) is not None and led.holdings[3].position == 0
    led.retire(2)
    led.release(1, 2, a)
    led.release(2, 2, a)
    led.release(3, 2, a)
    assert not led.holdings and led.tree.held() == ([], []) and led.check() == []
    assert len(led.free[KV]) == 64 and len(led.free[GDN]) == 20


def test_a_match_is_kept_only_while_its_request_waits_and_only_on_its_own_tree() -> None:
    """Requests too short for a checkpoint never change the tree: their matches stayed and grew
    with every request served (Codex, #59). A new tree counts its versions from 0 again: a match
    made on the old one must not pass for one on it."""
    led = ledger()
    for rid in range(1, 1001):
        prompt = [rid % 7 + 1] * 10
        assert led.match(rid, prompt) == 0
        run(led, rid, prompt, rid, 10, admit=12)
        led.release(rid, rid, prompt)
    led.match(5000, [1] * 10)
    led.release(5000, 1000)  # it ended while still waiting
    assert led.matches == {} and not led.holdings and led.check() == []
    a, c = list(range(200)), list(range(1000, 1200))
    run(led, 1001, c, 1001, 200, admit=210)
    led.commit(1001, c)
    led.release(1001, 1001, c)
    assert led.match(9, a) == 0
    version = led.tree.version
    led.flush()  # nothing is held: a new tree at once
    run(led, 1002, a, 1002, 200, admit=210)
    led.commit(1002, a)
    led.release(1002, 1002, a)
    assert led.tree.version == version and led.match(9, a) == 192


def test_a_tree_frees_the_states_past_the_limit_of_a_path_on_its_own() -> None:
    tree = RadixTree(PAGE, max_states_per_path=1)
    a = list(range(64))
    tree.insert(a[:32], [0, 1], 7)
    tree.insert(a, [0, 1, 2, 3], 8)
    assert tree.freed_states == [7] and tree.match(a).prefix == 64


def test_a_request_admitted_from_the_cache_restarts_from_0_on_pages_of_its_own() -> None:
    """A worker was replaced before the request ran (Antigravity, #59): the cached prefix it
    started from is gone with the old worker's stores."""
    led = ledger()
    a = list(range(200))
    run(led, 1, a, 1, 200, admit=210)
    led.commit(1, a)
    led.release(1, 1, a)
    got = led.admit(2, a, 210, 2)
    assert got is not None and got.prefix == 192
    r = led.reserve(2, 200, 2)
    assert r is not None and r.restore  # sent with the step the new worker refuses
    led.flush()
    tree_pages = set(led.tree.held()[0])
    assert led.restart(2) and led.restart(3) is False
    led.retire(2, ran=False)
    h = led.holdings[2]
    assert h.position == 0 and not (set(h.pages) & tree_pages) and set(h.pages) == h.own
    again = led.reserve(2, 200, 3)
    assert again is not None and not again.restore
    led.retire(3)
    led.release(2, 3, a)
    assert not led.holdings and led.check() == [] and led.tree.held() == ([], [])
    assert len(led.free[KV]) == 64 and len(led.free[GDN]) == 20


def test_a_speculative_step_keeps_its_state_only_where_its_accepted_tokens_cross_a_boundary() -> None:
    """doc/specs/mtp.md: every speculative decode step carries the next tracking slot with ``upto -grid``;
    the retired step's accepted length says whether it crossed (SGLang's ``_verify_commit_step_indices``)."""
    led = ledger(track_interval=64)
    prompt = list(range(100, 160))  # 60 tokens: no prefill checkpoint
    run(led, 1, prompt, 1, 60, admit=120)
    r = led.reserve(1, 64, 2, speculative=True)
    assert r is not None and r.track[GDN][1] == -64  # at the multiple of 64 it crosses
    slot = r.track[GDN][0]
    with pytest.raises(ValueError, match="accepted length"):
        led.retire(2)
    led.retire(2, advanced={1: 3})  # 60 -> 63: no boundary
    assert led.holdings[1].last_seqlen is None and led.holdings[1].committed == 63
    r = led.reserve(1, 67, 3, speculative=True)
    assert r is not None and r.track[GDN][0] == slot  # still the next slot
    led.retire(3, advanced={1: 2})  # 63 -> 65: the state at 64 was kept
    assert (led.holdings[1].last_seqlen, led.holdings[1].committed) == (64, 65)
    r = led.reserve(1, 69, 4, speculative=True)
    assert r is not None and r.track[GDN][0] != slot  # the kept one is not written over
    led.retire(4, advanced={1: 1})
    tokens = [*prompt, *range(900, 906)]
    led.release(1, 99, tokens)
    led.retire(99)
    assert led.match(2, [*tokens, 1]) == 64 and led.check() == []

def test_window_slot_publishes_whole_and_mid_chunk_checkpoints_and_restores_privately() -> None:
    led = RadixLedger([DECLS[0], StateDecl(SWA, 1, {"ring": ((8, 8), "u8")}, "slot")])
    led.set_capacity({KV: 20, SWA: 20})
    prompt = list(range(37))
    assert led.prefill_end(0, len(prompt), len(prompt)) == len(prompt)
    parent = led.admit(1, prompt, 48, 1)
    assert parent is not None
    # A chunk ending on a page takes a post-step snapshot; one continuing past the page tracks mid-chunk.
    r_page = led.reserve(1, 16, 1)
    assert r_page is not None and len(r_page.snapshot) == 1 and not r_page.track
    assert r_page.snapshot[0][:2] == (SWA, parent.items[SWA][0])
    led.retire(1)
    r = led.reserve(1, len(prompt), 2)
    assert r is not None and not r.snapshot and r.track[SWA][1] == 32
    snapshot = r.track[SWA][0]
    assert led.match(2, prompt) == 0
    led.retire(2)
    led.commit(1, prompt)
    for rid in (2, 3):
        child = led.admit(rid, prompt[:32] + [rid, 91], 42, rid)
        assert child is not None and child.prefix == 32
        assert child.items[KV][:2] == parent.items[KV][:2]
        reservation = led.reserve(rid, 34, rid)
        assert reservation is not None and reservation.restore[SWA] == (snapshot, child.items[SWA][0])
        assert not reservation.snapshot
        led.retire(rid)
    assert led.check() == []


def test_packed_radix_segments_own_their_keys_and_branch_at_whole_tokens() -> None:
    tree = RadixTree(4)
    source = array("q", [-3, 1 << 42, 3, 4] * 256)
    tree.insert(memoryview(source), list(range(256)), 10)
    original = list(source)
    source[511] += 1 << 33
    assert tree.match(original).prefix == 1024
    assert tree.match(source).kv == 508
    tree.insert(source, list(range(256, 512)), 11)
    assert tree.match(original).node.state == 10
    assert tree.match(source).node.state == 11
    # An integer view with non-unit stride takes the sequence conversion path.
    interleaved = array("q", [v for token in original for v in (token, 0)])
    assert tree.match(memoryview(interleaved)[::2]).prefix == 1024


def test_bounded_replay_cache_leaves_its_retained_suffix_to_compute() -> None:
    led = RadixLedger([DECLS[0], StateDecl(SWA, 1, {"ring": ((8, 8), "u8")}, "slot")],
                      replay_tokens=33)
    led.set_capacity({KV: 64, SWA: 20})
    prompt = list(range(128))
    assert led.prefill_end(0, 128, 128) == 128
    parent = led.admit(1, prompt, 140, 1)
    assert parent is not None
    r = led.reserve(1, 128, 1)
    assert r is not None and not r.snapshot and r.track[SWA][1] == 80
    led.retire(1)
    led.commit(1, prompt)
    assert led.match(2, prompt) == 80
    assert led.match(3, prompt[:20]) == 0
    assert led.check() == []


def test_incremental_waiting_matches_track_splits_extensions_and_evictions() -> None:
    led = RadixLedger([DECLS[0], StateDecl(SWA, 1, {"ring": ((8, 8), "u8")}, "slot")],
                      replay_tokens=4)
    led.set_capacity({KV: 64, SWA: 9})
    base = list(range(1000, 1064))
    suffix_a = list(range(2000, 2036))
    suffix_b = list(range(3000, 3036))

    # Cache the shared 64-token prefix.
    p0 = [*base, 99, 99, 99, 99]
    assert led.admit(1, p0, len(p0), 1) is not None
    assert led.reserve(1, len(p0), 1) is not None
    led.retire(1)
    led.commit(1, p0)
    led.release(1, 1, p0)

    waiting_a = [*base, *suffix_a]
    waiting_b = [*base, *suffix_b]
    branch_mid = [*base[:32], *range(4000, 4052)]
    assert led.match(10, waiting_a) == 64
    assert led.match(11, waiting_b) == 64

    # Splitting an ancestor edge (at 32) and adding a deeper extension on waiting_a's path:
    # waiting_a advances incrementally from 64 -> 96, while waiting_b stays at 64 without rescanning 0..64.
    for rid, prompt, step in ((2, branch_mid, 2), (3, waiting_a, 3)):
        assert led.admit(rid, prompt, len(prompt), step) is not None
        assert led.reserve(rid, len(prompt), step) is not None
        led.retire(step)
        led.commit(rid, prompt)
        led.release(rid, step, prompt)

    assert led.match(10, waiting_a) == 96
    assert led.match(11, waiting_b) == 64

    # Evicting states down to free capacity invalidates only evicted checkpoints and stays exact.
    other = list(range(5000, 5084))
    assert led.admit(4, other, len(other), 4) is not None
    for rid, prompt in ((10, waiting_a), (11, waiting_b)):
        assert led.match(rid, prompt) == led.tree.match(led._limit(prompt)).prefix
    led.cancel(4, 4)
    assert led.check() == []
