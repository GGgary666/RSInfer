# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Hybrid KV + slot prefix caching. For GDN, SGLang's radix cache with the mamba
``extra_buffer`` strategy (e838360, LICENSE.sglang; doc/specs/prefix-cache.md).

From SGLang, the device part only (no HiCache, SWA, sessions, PD, DCP, PP, int8 or host-only
checkpoints, lazy buffers):

- ``RadixTree``: ``UnifiedTreeCore`` with its Full and Mamba components
  (mem_cache/unified_cache/unified_tree_core.py, components/full.py, components/mamba.py). Keys are
  token pages; a node holds the KV pages of its segment and, maybe, the GDN state at its end. A
  match is as deep as the deepest node with a state (``create_match_validator``); the KV beyond it
  gives the branching point (``finalize_match_result_in_tree_core``). Full locks the path, Mamba
  the node (``acquire_component_lock``). Eviction is least recently used, KV by leaves, states also
  from inside the tree (``_evict_device_next_node``), with ``mamba_max_states_per_path``.
- ``RadixLedger``: what ``Req.kv`` and the request's storage do (managers/schedule_batch.py,
  mem_cache/kv/state_storage.py, mem_cache/common.py, mem_cache/unified_radix_cache.py): a running
  slot and two ping-pong tracking slots per request; where a step tracks its state
  (``_mamba_radix_cache_v2_req_prepare_for_extend``, ``prepare_for_decode``); the tracked state
  donated to the tree when a chunk's result is back (``checkpoint_into_tree``,
  ``prepare_for_caching_req``) and at the end (``_select_finished_checkpoint``); a hit restores the
  cached state into the running slot before the first step (``mamba_cow_src_index``).

For DS 1017, a checkpoint is the whole sliding-window slot at the end of a step, as in
DeepInfer's fork_seq. Its compressed KV pages stay shared, while a child gets a private slot.

Ours: the ids of ``ledger@1`` instead of tensors; positions move when a step retires, so a step
that did not run (``Resync``) is undone. Two tracking slots always: ``sched.overlap`` is a live key,
SGLang decides at start. A request keeps its own pages where the tree already had the tokens, and
frees them at its end: the worker's tables are only appended to (SGLang writes the tree's pages
into the request's row instead).
"""

from __future__ import annotations

import heapq
import itertools
import math
import sys
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from core.contracts.ledger import Admitted, Reserved
from core.contracts.state import StateDecl
from state.prefix_checkpoint import PrefixNode

from .pool import Free

TOKEN_BYTES = 8  # ledger token IDs are wire-format signed int64
KV, GDN, SWA = "kv.paged", "gdn.slots", "swa.slots"
FLA_CHUNK = 64
"""The GDN kernels' chunk (``mamba_chunk_size``): a state inside a step's input is there only at its
multiples (the chunk kernel's ``h``)."""
STATES_PER_REQUEST = 3
"""``MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO`` (kv_cache_configurator.py)."""
TRACKING_SLOTS = 2
"""``get_mamba_tracking_slots`` with the overlap schedule."""


class Node:
    __slots__ = ("key", "pages", "state", "children", "parent", "kv_lock", "state_lock", "access", "depth", "salt")

    def __init__(self, key: bytes, pages: list[int], parent: Node | None, depth: int, salt: bytes = b"") -> None:
        self.key, self.pages, self.parent, self.depth = key, pages, parent, depth
        self.salt = salt
        self.state: int | None = None
        self.children: dict[bytes, Node] = {}
        self.kv_lock = 0
        """Requests anchored here or below (Full's lock on the path)."""
        self.state_lock = 0
        """Requests anchored here (Mamba's lock on the node)."""
        self.access = 0

    def __lt__(self, other: Node) -> bool:
        return self.access < other.access


@dataclass
class Match:
    node: Node
    """The deepest node with a state on the path: where a request starts (the root: nowhere)."""
    kv: int
    """Tokens of KV the tree holds for the key, a multiple of the page."""

    @property
    def prefix(self) -> int:
        return self.node.depth


@dataclass
class Inserted:
    node: Node
    dups: set[int]
    """Indexes of the given pages where the tree already had other pages."""
    state_existed: bool


@dataclass(frozen=True)
class PinnedPrefix:
    tree: RadixTree
    node: Node
    items: dict[str, list[int]]

    def release(self) -> None:
        self.tree.unlock(self.node)


class RadixTree:
    def __init__(self, page: int, max_states_per_path: int = -1, *, kv_only: bool = False) -> None:
        self.page, self.max_states_per_path = page, max_states_per_path
        self.kv_only = kv_only
        self.page_bytes = page * TOKEN_BYTES
        self.root = Node(b"", [], None, 0)
        self.root.kv_lock = 1  # never evicted
        self.clock = itertools.count(1)
        self.version = 0
        """Bumped on every change a match could see."""
        self.insert_version = 0
        """Bumped whenever a node or state enters the tree."""
        self.freed_states: list[int] = []
        """States ``mamba_max_states_per_path`` took off a path, for the ledger to free."""

    # ---------------------------------------------------------------- reading

    @staticmethod
    def _packed(tokens: Sequence[int]) -> memoryview:
        # Requests already hold the wire column as a contiguous view of signed int64.
        if isinstance(tokens, memoryview) and tokens.format == "q" and tokens.c_contiguous:
            return tokens.cast("B")
        return memoryview(array("q", tokens)).cast("B")

    def _child_key(self, packed: memoryview, at: int, node: Node, salt: bytes) -> bytes:
        start = at * TOKEN_BYTES
        return (salt if node is self.root else b"") + bytes(packed[start:start + self.page_bytes])

    def _node_key(self, node: Node) -> bytes:
        return node.salt + node.key[:self.page_bytes]

    def _shared(self, key: bytes, packed: memoryview, at: int) -> int:
        """Longest common prefix in whole pages. bytes.startswith compares contiguous buffers in C."""
        start = at * TOKEN_BYTES
        pages = min(len(key), len(packed) - start) // self.page_bytes
        if key.startswith(packed[start:start + pages * self.page_bytes]):
            return pages * self.page
        lo, hi = 0, pages - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            offset = lo * self.page_bytes
            if key.startswith(packed[start + offset:start + mid * self.page_bytes], offset):
                lo = mid
            else:
                hi = mid - 1
        return lo * self.page

    def match(self, tokens: Sequence[int], *, start: Node | None = None, salt: bytes = b"") -> Match:
        """The deepest reusable prefix within ``tokens``, and how far the KV goes."""
        if start is not None and start is not self.root and start.state is not None and start.depth <= len(tokens):
            node, base, best = start, start.depth, start
        else:
            node, base, best = self.root, 0, self.root
        rem = len(tokens) - base
        if rem < self.page or not node.children:
            return Match(best, best.depth)
        tail = tokens[base:] if base else tokens
        packed = self._packed(tail)
        rel, kv = 0, base
        while rel + self.page <= rem:
            child = node.children.get(self._child_key(packed, rel, node, salt))
            if child is None:
                break
            shared = self._shared(child.key, packed, rel)
            kv = base + rel + shared
            if shared * TOKEN_BYTES < len(child.key):
                if self.kv_only:
                    best = self._split(child, shared)
                break
            node, rel = child, rel + shared
            if self.kv_only or node.state is not None:
                best = node
        return Match(best, max(kv, best.depth))

    def path(self, node: Node) -> list[Node]:
        out: list[Node] = []
        while node is not self.root:
            out.append(node)
            assert node.parent is not None
            node = node.parent
        return out[::-1]

    def pages_to(self, node: Node) -> list[int]:
        return [p for n in self.path(node) for p in n.pages]

    def touch(self, node: Node) -> None:
        now = next(self.clock)
        for n in self.path(node):
            n.access = now

    # ---------------------------------------------------------------- locks

    def lock(self, node: Node) -> None:
        for n in self.path(node):
            n.kv_lock += 1
        node.state_lock += node is not self.root

    def unlock(self, node: Node) -> None:
        for n in self.path(node):
            assert n.kv_lock > 0, "unlocking a node not locked"
            n.kv_lock -= 1
        if node is not self.root:
            assert node.state_lock > 0
            node.state_lock -= 1

    # ---------------------------------------------------------------- writing

    def _split(self, child: Node, at: int) -> Node:
        """``_split_node``: ``child`` keeps its end (and its state); a new node takes its first ``at``
        tokens."""
        parent = child.parent
        cut = at * TOKEN_BYTES
        assert parent is not None and 0 < cut < len(child.key) and at % self.page == 0
        depth = child.depth - len(child.pages) * self.page + at
        top = Node(child.key[:cut], child.pages[:at // self.page], parent, depth, child.salt)
        top.kv_lock, top.access = child.kv_lock, child.access
        parent.children[self._node_key(child)] = top
        child.key, child.pages, child.parent = child.key[cut:], child.pages[at // self.page:], top
        child.salt = b""
        top.children[self._node_key(child)] = child
        self.version += 1
        return top

    def insert(self, tokens: Sequence[int], pages: Sequence[int], state: int | None,
               *, anchor: Node | None = None, salt: bytes = b"") -> Inserted:
        """``insert`` of ``tokens`` (whole pages) with the request's ``pages``, and ``state`` at the
        end (``commit_insert_component_data``)."""
        assert len(tokens) % self.page == 0 and len(pages) == len(tokens) // self.page and tokens
        node = self.root if anchor is None else anchor
        base_at = node.depth
        tail_tokens = tokens[base_at:]
        tail_pages = pages[base_at // self.page:]
        packed = self._packed(tail_tokens)
        rel, rel_len = 0, len(tail_tokens)
        dups: set[int] = set()
        now = next(self.clock)
        for ancestor in self.path(node):
            ancestor.access = now
        while rel < rel_len:
            child = node.children.get(self._child_key(packed, rel, node, salt))
            if child is None:
                child = Node(bytes(packed[rel * TOKEN_BYTES:]), list(tail_pages[rel // self.page:]), node, len(tokens),
                             salt if node is self.root else b"")
                node.children[self._node_key(child)] = child
                rel = rel_len
            else:
                shared = self._shared(child.key, packed, rel)
                if shared * TOKEN_BYTES < len(child.key):
                    child = self._split(child, shared)
                offset = rel // self.page
                incoming = tail_pages[offset:offset + len(child.pages)]
                if child.pages != incoming:
                    dups.update((base_at + rel) // self.page + i
                                for i, (p, q) in enumerate(zip(child.pages, incoming, strict=True)) if p != q)
                rel += shared
            child.access = now
            node = child
        existed = node.state is not None
        if not existed and state is not None:
            node.state = state
            self._cap_states(node)
        self.version += 1
        self.insert_version += 1
        return Inserted(node, dups, existed)

    def _cap_states(self, tail: Node) -> None:
        """``_evict_excess_path_states``: past ``mamba_max_states_per_path`` states on the path,
        the shallowest ones go, but not the tail, a fork, a locked node or a leaf."""
        cap = self.max_states_per_path
        if cap < 0:
            return
        holders = [n for n in self.path(tail) if n.state is not None]
        excess = len(holders) - cap
        for n in holders:
            if excess <= 0 or n is tail:
                break
            if n.state_lock or len(n.children) != 1:
                continue
            self.freed_states.append(n.state)  # type: ignore[arg-type]
            n.state = None
            self.version += 1
            excess -= 1

    # ---------------------------------------------------------------- eviction

    def _evictable_leaf(self, n: Node) -> bool:
        return n is not self.root and not n.children and n.kv_lock == 0

    def _drop_leaf(self, n: Node, pages: list[int], states: list[int]) -> None:
        pages.extend(n.pages)
        if n.state is not None:
            states.append(n.state)
            n.state = None
        assert n.parent is not None
        del n.parent.children[self._node_key(n)]
        self.version += 1

    def evict_pages(self, want: int) -> tuple[list[int], list[int]]:
        """Pages (and the states of the leaves they leave) of least recently used unlocked leaves,
        until ``want`` pages are out or none is left."""
        pages: list[int] = []
        states: list[int] = []
        heap = [n for n in self._nodes() if self._evictable_leaf(n)]
        heapq.heapify(heap)
        while heap and len(pages) < want:
            n = heapq.heappop(heap)
            self._drop_leaf(n, pages, states)
            parent = n.parent
            if parent is not None and self._evictable_leaf(parent):
                heapq.heappush(heap, parent)
        return pages, states

    def evict_states(self, want: int) -> tuple[list[int], list[int]]:
        """States of least recently used unlocked nodes: a leaf goes whole, a node inside the tree
        keeps its KV (a tombstone)."""
        pages: list[int] = []
        states: list[int] = []
        heap = [n for n in self._nodes() if n.state is not None and n.state_lock == 0]
        heapq.heapify(heap)
        while heap and len(states) < want:
            n = heapq.heappop(heap)
            if self._evictable_leaf(n):
                self._drop_leaf(n, pages, states)
            else:
                states.append(n.state)  # type: ignore[arg-type]
                n.state = None
                self.version += 1
        return pages, states

    def _nodes(self) -> list[Node]:
        out: list[Node] = []
        stack = list(self.root.children.values())
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(n.children.values())
        return out

    def held(self) -> tuple[list[int], list[int]]:
        """Every page and state the tree holds."""
        nodes = self._nodes()
        return [p for n in nodes for p in n.pages], [n.state for n in nodes if n.state is not None]


@dataclass
class Holding:
    prompt: int
    pages: list[int]
    own: set[int]
    """Pages it gives back at its end: those the tree does not hold."""
    run: int | None
    track: list[int]
    anchor: Node
    """The node it holds locked: its matched prefix, then the end of what it inserted."""
    restore_from: Node | None
    """Locked until its state is in ``run``."""
    position: int
    branching: int | None
    next_idx: int = 0
    last_idx: int = 1
    last_seqlen: int | None = None
    prev_seqlen: int | None = None
    restore_step: int | None = None
    restored: bool = True
    committed: int = 0
    """Tokens computed, as the retired steps say: ``position`` runs ahead of it by the steps in
    flight, and by the drafts a speculative step may not accept (doc/specs/mtp.md)."""
    salt: bytes = b""


@dataclass
class Reservation:
    previous: int
    admission: bool = False
    pages: list[int] = field(default_factory=list[int])
    track: tuple[int, int] | None = None
    decode: bool = False
    restore: bool = False
    upto: int = 0
    speculative: bool = False


class RadixLedger:
    """Share read-only KV pages with a private mutable state slot per admitted request."""

    def __init__(self, declarations: Sequence[StateDecl], *, track_interval: int = 256,
                 max_states_per_path: int = -1, replay_tokens: int = 1) -> None:
        if replay_tokens < 1:
            raise ValueError("alloc.prefix_replay_tokens must be positive")
        self.replay_tokens = replay_tokens
        kinds = {(d.kind, d.lifetime) for d in declarations}
        self.slot_kind = next((kind for kind in (SWA, GDN) if (kind, "slot") in kinds), None)
        self.kinds = [KV] if self.slot_kind is None else [KV, self.slot_kind]
        expected = {(KV, "request")}
        if self.slot_kind is not None:
            expected.add((self.slot_kind, "slot"))
        if kinds != expected or any(d.snapshot_every for d in declarations):
            raise ValueError(f"the prefix cache (alloc.radix_cache) takes {KV} pages with optional {GDN} or {SWA} slots; the model "
                             f"declares {sorted(kinds)}: set alloc.radix_cache = false")
        pages = {d.tokens_per_item for d in declarations if d.kind == KV}
        if len(pages) != 1 or min(pages) <= 0:
            raise ValueError(f"{KV} layers must share one positive page size")
        if track_interval <= 0 or max_states_per_path == 0 or max_states_per_path < -1:
            raise ValueError("alloc.mamba_track_interval must be positive, alloc.mamba_max_states_per_path -1 "
                             "(no limit) or positive")
        self.page = pages.pop()
        if self.slot_kind == GDN and max(FLA_CHUNK, self.page) % min(FLA_CHUNK, self.page):
            # Else a state would be kept where the GDN kernels have none (Antigravity, #59).
            raise ValueError(f"state.kv.page_size = {self.page}: with the prefix cache, it or the GDN kernels' "
                             f"chunk ({FLA_CHUNK}) must divide the other (as SGLang's), or alloc.radix_cache = false")
        if self.slot_kind == GDN and track_interval % self.page:  # SGLang's mamba_hook
            raise ValueError(f"alloc.mamba_track_interval = {track_interval} must be a multiple of "
                             f"state.kv.page_size = {self.page} (as SGLang's)")
        self.chunk = max(FLA_CHUNK, self.page) if self.slot_kind == GDN else self.page
        """``mamba_cache_chunk_size``."""
        self.grid = math.lcm(self.chunk, self.page)
        """``mamba_checkpoint_grid``: where a state may enter the tree."""
        self.decode_grid = math.lcm(self.grid, track_interval)
        """``mamba_track_grid``: where decoding tracks its state."""
        self.max_states_per_path = max_states_per_path
        self.matches: dict[int, tuple[int, int, Match]] = {}
        """``(tree.version, tree.insert_version, Match)`` of the requests still waiting."""
        self.restore_nodes: list[PrefixNode] | None = None
        self.staged_prefix: dict[str, list[int]] | None = None
        self.tree = self._new_tree()
        self.capacity = dict.fromkeys(self.kinds, 0)
        self.free = {kind: Free() for kind in self.kinds}
        self.holdings: dict[int, Holding] = {}
        self.reservations: dict[tuple[int, int], Reservation] = {}
        self.releases: dict[int, tuple[int, Sequence[int]]] = {}
        self.retired = -1
        self.ready = False
        self.stale = False
        """Flushed while requests held items: no match, nothing kept, until the last one ended."""

    def _new_tree(self) -> RadixTree:
        self.matches.clear()  # a new tree counts its versions from 0 again
        return RadixTree(self.page, self.max_states_per_path, kv_only=self.slot_kind is None)

    @property
    def align(self) -> int:
        return self.page

    def prefill_end(self, start: int, end: int, prompt_len: int) -> int:
        return end

    @property
    def max_requests(self) -> int | None:
        if self.slot_kind is None:
            return None
        return self.capacity[self.slot_kind] // (STATES_PER_REQUEST + TRACKING_SLOTS)  # _calculate_mamba_ratio

    def _count(self, tokens: int) -> int:
        return (tokens + self.page - 1) // self.page

    # ---------------------------------------------------------------- capacity

    def set_capacity(self, items: Mapping[str, int]) -> None:
        if set(items) != set(self.kinds):
            raise ValueError(f"capacities for {sorted(items)}, but the declared kinds are {sorted(self.kinds)}")
        if self.ready and dict(items) == self.capacity:
            return  # an idle worker says it again
        if not (self.holdings or self.reservations or self.releases or self.staged_prefix):
            # Every item is free again, so none needs listing; the tree's go with them.
            self.tree = self._new_tree()
            self.free = {kind: Free(n) for kind, n in items.items()}
        else:
            for kind, n in items.items():
                if n < self.capacity[kind]:
                    raise ValueError("a state pool shrinks only while it holds nothing")
                self.free[kind].extend(range(self.capacity[kind], n))
        self.capacity = dict(items)
        if self.restore_nodes is not None:
            self.load_prefix(self.restore_nodes)
            self.restore_nodes = None
        self.ready = True
        if self.slot_kind is not None:
            cap = self.max_requests
            print(f"core.state.alloc: prefix cache on, {items[self.slot_kind]} {self.slot_kind} for at most {cap} requests at once "
                  f"({STATES_PER_REQUEST + TRACKING_SLOTS} per request, as SGLang's)", file=sys.stderr, flush=True)

    def flush(self) -> None:
        self.stale = True
        self.matches.clear()  # nothing is matched from now on (Antigravity, #59)
        self._flush_if_idle()

    def _flush_if_idle(self) -> None:
        if self.stale and not (self.holdings or self.reservations or self.releases or self.staged_prefix):
            self.tree = self._new_tree()
            self.free = {kind: Free(n) for kind, n in self.capacity.items()}
            self.stale = False

    def save_prefix(self) -> list[PrefixNode]:
        if not self.ready or self.stale or self.holdings or self.reservations or self.releases or self.staged_prefix:
            raise RuntimeError("prefix checkpoint requires a ready, empty scheduler domain")
        parents = {self.tree.root: -1}
        nodes: list[PrefixNode] = []
        for n in self.tree._nodes():
            assert n.parent is not None
            parents[n] = len(nodes)
            nodes.append(PrefixNode(parents[n.parent], n.key, n.pages.copy(), n.state, n.salt, n.access))
        return nodes

    def load_prefix(self, nodes: Sequence[PrefixNode]) -> None:
        if self.holdings or self.reservations or self.releases or self.staged_prefix:
            raise RuntimeError("prefix restore requires an empty scheduler domain")
        tree = self._new_tree()
        used: dict[str, set[int]] = {kind: set() for kind in self.kinds}
        installed: list[Node] = []
        for i, n in enumerate(nodes):
            if (not -1 <= n.parent < i or not n.pages or len(n.key) != len(n.pages) * tree.page_bytes
                    or (n.parent != -1 and n.salt)):
                raise ValueError("invalid radix node in prefix checkpoint")
            parent = tree.root if n.parent == -1 else installed[n.parent]
            node = Node(n.key, n.pages.copy(), parent, parent.depth + len(n.pages) * self.page, n.salt)
            key = tree._node_key(node)
            if key in parent.children:
                raise ValueError("duplicate radix edge in prefix checkpoint")
            parent.children[key] = node
            node.access, node.state = n.access, n.slot
            installed.append(node)
            for page in n.pages:
                if not 0 <= page < self.capacity[KV] or page in used[KV]:
                    raise ValueError("invalid or duplicate page in prefix checkpoint")
                used[KV].add(page)
            if n.slot is not None:
                kind = self.slot_kind
                if kind is None or not 0 <= n.slot < self.capacity[kind] or n.slot in used[kind]:
                    raise ValueError("invalid or duplicate recurrent slot in prefix checkpoint")
                used[kind].add(n.slot)
        free: dict[str, Free] = {}
        for kind, slots in used.items():
            start = min(slots, default=self.capacity[kind])
            free[kind] = Free(start)
            free[kind].extend(i for i in range(start, self.capacity[kind]) if i not in slots)
        tree.clock = itertools.count(max((n.access for n in installed), default=0) + 1)
        self.tree, self.free = tree, free
        self.matches.clear()
        self.stale = False

    def fits(self, tokens: int) -> bool:
        return (self.ready and self._count(tokens) <= self.capacity[KV]
                and (self.slot_kind is None
                     or (1 + TRACKING_SLOTS <= self.capacity[self.slot_kind] and bool(self.max_requests))))

    # ---------------------------------------------------------------- items

    def _take(self, kind: str, n: int) -> list[int] | None:
        """``evict_for_alloc``, then ``n`` items, or none at all."""
        free = self.free[kind]
        if len(free) < n:
            if kind == KV:
                pages, states = self.tree.evict_pages(n - len(free))
            else:
                pages, states = self.tree.evict_states(n - len(free))
            self.free[KV].extend(pages)
            if self.slot_kind is not None:
                self.free[self.slot_kind].extend(states)
        if len(free) < n:
            return None
        return [free.pop() for _ in range(n)]

    def _flush_capped(self) -> None:
        if self.slot_kind is not None and self.tree.freed_states:
            self.free[self.slot_kind].extend(self.tree.freed_states)
            self.tree.freed_states.clear()

    def _limit(self, prompt: Sequence[int]) -> Sequence[int]:
        """``_compute_max_prefix_len``: at least replay_tokens remain to compute."""
        return prompt[:max(0, len(prompt) - self.replay_tokens) // self.page * self.page]

    def _resolve_match(self, request: int, prompt: Sequence[int], salt: str) -> Match:
        if self.stale:
            m = Match(self.tree.root, 0)
            self.matches[request] = (self.tree.version, self.tree.insert_version, m)
            return m
        cached = self.matches.get(request)
        start: Node | None = None
        if cached is not None:
            v_tree, v_ins, prev = cached
            if v_tree == self.tree.version:
                return prev
            node_valid = prev.node is self.tree.root or prev.node.state is not None
            if self.slot_kind is not None and node_valid and v_ins == self.tree.insert_version and prev.kv == prev.node.depth:
                self.matches[request] = (self.tree.version, self.tree.insert_version, prev)
                return prev
            if prev.node is not self.tree.root and prev.node.state is not None:
                start = prev.node
        m = self.tree.match(self._limit(prompt), start=start, salt=salt.encode())
        self.matches[request] = (self.tree.version, self.tree.insert_version, m)
        return m

    def match(self, request: int, prompt: Sequence[int], salt: str = "") -> int:
        return self._resolve_match(request, prompt, salt).prefix

    def pin_replica(self, request: int, prompt: Sequence[int], salt: str) -> PinnedPrefix | None:
        if self.slot_kind != GDN or self.stale:
            return None
        match = self._resolve_match(request, prompt, salt)
        if match.node.state is None:
            return None
        self.tree.lock(match.node)
        return PinnedPrefix(self.tree, match.node, {KV: self.tree.pages_to(match.node), GDN: [match.node.state]})

    def stage_replica(self, request: int, prompt: Sequence[int], salt: str, depth: int,
                      extent: int) -> dict[str, list[int]] | None:
        if self.slot_kind != GDN or not self.ready or self.stale or self.staged_prefix is not None:
            return None
        # Keep the KV pages and recurrent seed from the same execution history.
        if self._resolve_match(request, prompt, salt).kv:
            return None
        if len(self.free[KV]) < self._count(extent) or len(self.free[GDN]) < 2 + TRACKING_SLOTS:
            return None
        items = {KV: [self.free[KV].pop() for _ in range(depth // self.page)], GDN: [self.free[GDN].pop()]}
        self.staged_prefix = items
        return items

    def finish_replica(self, tokens: Sequence[int], salt: str, committed: bool) -> None:
        items, self.staged_prefix = self.staged_prefix, None
        assert items is not None
        if committed and not self.stale:
            pages, (slot,) = items[KV], items[GDN]
            result = self.tree.insert(tokens, pages, slot, salt=salt.encode())
            self.free[KV].extend(pages[i] for i in sorted(result.dups))
            if result.state_existed:
                self.free[GDN].extend([slot])
            self._flush_capped()
        else:
            for kind, ids in items.items():
                self.free[kind].extend(ids)
            self._flush_if_idle()

    def admit(self, request: int, prompt: Sequence[int], tokens: int, step: int, salt: str = "") -> Admitted | None:
        if request in self.holdings:
            raise ValueError("request already admitted")
        cap = self.max_requests
        if cap is not None and len(self.holdings) >= cap:
            return None
        m = self._resolve_match(request, prompt, salt)
        anchor = m.node
        self.tree.lock(anchor)  # before anything is taken: eviction must not take the prefix
        prefix_pages = self.tree.pages_to(anchor)
        pages = self._take(KV, max(0, self._count(tokens) - len(prefix_pages)))
        kind = self.slot_kind
        slots = self._take(kind, 1 + TRACKING_SLOTS) if pages is not None and kind is not None else []
        if pages is None or slots is None:
            self.free[KV].extend(pages or [])
            self.tree.unlock(anchor)
            return None
        self.tree.touch(anchor)
        self.matches.pop(request, None)  # no longer waiting
        aligned = m.kv // self.grid * self.grid
        hit = kind is not None and anchor is not self.tree.root
        if hit:
            self.tree.lock(anchor)  # the restore source, until the state is copied
        self.holdings[request] = Holding(
            len(prompt), [*prefix_pages, *pages], set(pages), slots[0] if slots else None, slots[1:], anchor,
            anchor if hit else None, m.prefix, aligned if aligned > m.prefix else None, restored=not hit,
            committed=m.prefix, salt=salt.encode())
        self.reservations[step, request] = Reservation(m.prefix, admission=True)
        return Admitted(m.prefix, self.items(request))

    def _track(self, h: Holding, start: int, upto: int) -> tuple[int, bool] | None:
        """Where this step's state enters the tree, if it does, and whether it decodes."""
        if start >= h.prompt:  # prepare_for_decode
            return (upto, True) if upto % self.decode_grid == 0 else None
        if self.slot_kind == SWA:
            bound = min(upto, max(0, h.prompt - self.replay_tokens))
            at = bound // self.page * self.page
            return (at, False) if at > start else None
        n = upto - start  # _mamba_radix_cache_v2_req_prepare_for_extend, without DCP
        if n < self.grid:
            return None
        at = start + n // self.grid * self.grid
        b = h.branching
        if b is not None and start < b < upto and (b - start) % self.chunk == 0:
            at = b  # the branching point is in this chunk: track it there
        return at, False

    def reserve(self, request: int, upto: int, step: int, speculative: bool = False) -> Reserved | None:
        h = self.holdings[request]
        r = self.reservations.setdefault((step, request), Reservation(h.position))
        r.upto, r.speculative = upto, speculative
        start = h.position
        added = self._take(KV, max(0, self._count(upto) - len(h.pages)))
        if added is None:
            self.cancel(request, step)
            return None
        h.pages.extend(added)
        h.own.update(added)
        r.pages.extend(added)
        kind = self.slot_kind
        if kind is None:
            h.position = upto
            return Reserved({KV: added} if added else {}, {}, {})
        assert h.run is not None
        restore: dict[str, tuple[int, int]] = {}
        if not h.restored and h.restore_step is None:
            assert h.restore_from is not None and h.restore_from.state is not None
            restore[kind] = (h.restore_from.state, h.run)
            h.restore_step, r.restore = step, True
        track: dict[str, tuple[int, int]] = {}
        snapshot: tuple[tuple[str, int, int], ...] = ()
        if speculative:
            # Every speculative decode step carries the next tracking slot (SGLang's
            # mamba_track_indices); whether it crosses a boundary is known once it retired. Two in
            # flight share the slot: one step crosses at most one boundary (mamba_track_interval
            # >= num_draft_tokens), and two in a row do not (a boundary every decode_grid tokens).
            track[kind] = (h.track[h.next_idx], -self.decode_grid)
            r.track, r.decode = (h.next_idx, -self.decode_grid), True
            where = None
        else:
            where = self._track(h, start, upto)
        if where is not None:
            idx = h.next_idx
            if self.slot_kind == SWA and where[0] == upto:
                snapshot = ((SWA, h.run, h.track[idx]),)
            else:
                track[kind] = (h.track[idx], where[0])
            r.track, r.decode = (idx, where[0]), where[1]
            h.next_idx = 1 - idx
        h.position = upto
        return Reserved({KV: added} if added else {}, restore, track, snapshot)

    def items(self, request: int) -> dict[str, list[int]]:
        h = self.holdings[request]
        items = {KV: list(h.pages)}
        if self.slot_kind is not None:
            assert h.run is not None
            items[self.slot_kind] = [h.run]
        return items

    def cancel(self, request: int, step: int) -> None:
        h = self.holdings.get(request)
        r = self.reservations.pop((step, request), None)
        if h is None or r is None:
            return
        if r.admission:
            self._drop(request, h)
            return
        self.free[KV].extend(r.pages)
        h.own.difference_update(r.pages)
        del h.pages[len(h.pages) - len(r.pages):]
        h.position = r.previous
        if r.track is not None:
            h.next_idx = r.track[0]
        if r.restore:
            h.restore_step = None

    def _drop(self, request: int, h: Holding) -> None:
        """Everything the request holds goes back; nothing enters the tree."""
        self.free[KV].extend(h.own)
        if self.slot_kind is not None:
            assert h.run is not None
            self.free[self.slot_kind].extend([h.run, *h.track])
        self.tree.unlock(h.anchor)
        if h.restore_from is not None:
            self.tree.unlock(h.restore_from)
        del self.holdings[request]
        for key in [k for k in self.reservations if k[1] == request]:
            del self.reservations[key]
        self._flush_if_idle()

    def retire(self, step: int, ran: bool = True, advanced: Mapping[int, int] | None = None) -> None:
        missing = [request for (epoch, request), r in self.reservations.items()
                   if epoch == step and ran and r.speculative and (advanced is None or request not in advanced)]
        if missing:  # before anything changes
            raise ValueError(f"speculative step {step} retired without the accepted length of requests {missing}")
        for key, r in list(self.reservations.items()):
            epoch, request = key
            if epoch != step:
                continue
            del self.reservations[key]
            h = self.holdings[request]
            if ran and r.speculative:
                assert advanced is not None
                start, end = h.committed, h.committed + advanced[request]
                if r.track is not None and start // self.decode_grid != end // self.decode_grid:
                    # _verify_commit_step_indices: the worker kept the state at the boundary
                    h.prev_seqlen = h.last_seqlen
                    h.last_seqlen, h.last_idx = end // self.decode_grid * self.decode_grid, r.track[0]
                    h.next_idx = 1 - r.track[0]
                h.committed = end
                if r.restore:
                    h.restored = True
                    self._restored(h)
            elif ran:
                if r.track is not None:  # TrackingAdvance
                    if r.decode:
                        h.prev_seqlen = h.last_seqlen
                    h.last_seqlen, h.last_idx = r.track[1], r.track[0]
                if r.restore:
                    h.restored = True
                    self._restored(h)
                if r.upto:  # an admission's own reservation has none until its first chunk is reserved
                    h.committed = r.upto
            else:
                h.position = min(h.position, r.previous)
                if r.track is not None and not r.speculative:
                    h.next_idx = 1 - h.last_idx
                if r.restore:
                    h.restore_step = None
        self.retired = max(self.retired, step)
        self._finish_ready()

    def _restored(self, h: Holding) -> None:
        if h.restore_from is not None:
            self.tree.unlock(h.restore_from)
            h.restore_from = None

    # ---------------------------------------------------------------- into the tree

    def _insert(self, h: Holding, tokens: Sequence[int], length: int, state: int | None) -> bool:
        """``checkpoint_into_tree`` of ``tokens[:length]`` with ``state`` at its end. Returns whether
        the tree took ``state``."""
        anchor = h.anchor if h.anchor.depth <= length else self.tree.root
        pages = h.pages[:length // self.page]
        got = self.tree.insert(tokens[:length], pages, state, anchor=anchor, salt=h.salt)
        self._flush_capped()
        first_page = anchor.depth // self.page
        if got.dups:
            h.own.difference_update(pages[i] for i in range(first_page, len(pages)) if i not in got.dups)
        else:
            h.own.difference_update(pages[first_page:])
        self.tree.lock(got.node)
        self.tree.unlock(h.anchor)
        h.anchor = got.node
        return not got.state_existed

    def commit(self, request: int, prompt: Sequence[int]) -> None:
        h = self.holdings.get(request)
        if h is None or request in self.releases:
            return
        if self.slot_kind is None:
            self._cache_pages(h, prompt)
            return
        length, h.last_seqlen, h.prev_seqlen = h.last_seqlen, None, None
        if length is None or length > h.position or not h.restored or self.stale:
            return
        assert length % self.page == 0, f"a tracked state at {length}, not on a page"
        new = self._take(self.slot_kind, 1)  # donate_tracking_slot
        if new is None:
            return
        donated, h.track[h.last_idx] = h.track[h.last_idx], new[0]
        if not self._insert(h, prompt, length, donated):
            self.free[self.slot_kind].extend([donated])

    def restart(self, request: int) -> bool:
        """The cache's pages where the request has its own, the root its anchor, nothing to restore:
        as admitted without a hit. Steps still in flight for it retire as not run (the new worker
        refuses them), which leaves its position at 0."""
        h = self.holdings.get(request)
        if h is None:
            return False
        shared = [i for i, p in enumerate(h.pages) if p not in h.own]
        fresh = self._take(KV, len(shared))  # the anchor is still locked: its pages are not evicted
        if fresh is None:
            return False
        for i, p in zip(shared, fresh, strict=True):
            h.pages[i] = p
        h.own.update(fresh)
        self.tree.unlock(h.anchor)
        self._restored(h)
        h.anchor = self.tree.root
        self.tree.lock(h.anchor)
        h.position, h.branching, h.restored, h.restore_step = 0, None, True, None
        h.last_seqlen = h.prev_seqlen = None
        h.committed = 0
        return True

    def release(self, request: int, last_step: int, tokens: Sequence[int] = ()) -> None:
        self.matches.pop(request, None)  # it may end while still waiting
        if request not in self.holdings:
            return
        if last_step <= self.retired and not any(k[1] == request for k in self.reservations):
            self._finish(request, tokens)
            return
        held: Sequence[int] = (tokens if isinstance(tokens, tuple) or (isinstance(tokens, memoryview) and tokens.readonly)
                               else memoryview(array("q", tokens)))
        self.releases[request] = (last_step, held)
        self._finish_ready()

    def _finish_ready(self) -> None:
        for request, (last, tokens) in list(self.releases.items()):
            if last > self.retired or any(k[1] == request for k in self.reservations):
                continue
            del self.releases[request]
            self._finish(request, tokens)

    def _finish(self, request: int, tokens: Sequence[int]) -> None:
        """``release_kv_cache`` with ``checkpoint``: the request's last tracked state, and the KV up to
        it, enter the tree; everything else goes back."""
        h = self.holdings[request]
        if self.slot_kind is None:
            self._cache_pages(h, tokens)
            self._drop(request, h)
            return
        assert h.run is not None
        keep: int | None = None
        checkpoint = self._finished_checkpoint(h, len(tokens)) if h.restored and not self.stale else None
        if checkpoint is not None and checkpoint[0] > 0:
            length, idx = checkpoint
            if self._insert(h, tokens, length, h.track[idx]):
                keep = idx
        self.free[KV].extend(h.own)
        self.free[self.slot_kind].extend([h.run, *(s for i, s in enumerate(h.track) if i != keep)])
        self.tree.unlock(h.anchor)
        self._restored(h)
        del self.holdings[request]
        self._flush_if_idle()

    def _cache_pages(self, h: Holding, tokens: Sequence[int]) -> None:
        length = min(h.committed, len(tokens)) // self.page * self.page
        if not self.stale and length > h.anchor.depth:
            self._insert(h, tokens, length, None)

    def _finished_checkpoint(self, h: Holding, length: int) -> tuple[int, int] | None:
        """``_select_finished_checkpoint``."""
        cache_len = h.last_seqlen or 0
        if cache_len <= length:
            return cache_len, h.last_idx
        if h.prev_seqlen is None or h.prev_seqlen > length:
            return None
        return h.prev_seqlen, 1 - h.last_idx

    # ---------------------------------------------------------------- checks

    def check(self) -> list[str]:
        tree_pages, tree_states = self.tree.held()
        held = {KV: [*tree_pages, *(p for h in self.holdings.values() for p in h.own)]}
        if self.slot_kind is not None:
            held[self.slot_kind] = [
                *tree_states, *(s for h in self.holdings.values() for s in (h.run, *h.track) if s is not None)]
        for kind, items in (self.staged_prefix or {}).items():
            held[kind].extend(items)
        errors = []
        for kind in self.kinds:
            ids = sorted([*self.free[kind].returned, *held[kind]])
            if ids != list(range(self.free[kind].fresh, self.capacity[kind])):
                errors.append(f"{kind}: free, held and cached items disagree")
        return errors
