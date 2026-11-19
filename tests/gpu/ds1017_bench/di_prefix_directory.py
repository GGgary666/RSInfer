"""Online prefix directory for DeepInfer's offline API. Only completed GPU snapshots can match."""
# pyright: reportMissingImports=false

from __future__ import annotations

from collections import OrderedDict
from hashlib import blake2b
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


class Directory:
    def __init__(
        self,
        engine: Any,
        work: Path,
        *,
        max_context: int = 256000,
        reserve_slots: int = 16,
        yoco_clip: int = 140,
    ) -> None:
        ext_dir = str(work / "extension")
        if ext_dir not in sys.path:
            sys.path.insert(0, ext_dir)
        import ds_prefix_pages

        self.pages_of = ds_prefix_pages.pages
        self.engine = engine
        self.max_context = max_context
        self.reserve_slots = reserve_slots
        self.yoco_clip = max(0, int(yoco_clip))
        self.page: int = engine.cache_config.block_size
        self.slots = engine.state_manager
        self.blocks = engine.block_manager
        self.saved: OrderedDict[bytes, tuple[list[int], int]] = OrderedDict()
        self.requests: dict[str, dict[int, bytes]] = {}
        self.empty_pairs = np.empty((0, 2), dtype=np.int32)
        self.empty_ids = np.empty(0, dtype=np.int32)
        engine.pe_prepare_save_kvcache = self.after_prefill

    def clear(self) -> None:
        """Release retained pages and snapshots between fully drained concurrency levels."""
        for pages, slot in self.saved.values():
            self.blocks.unpin_blocks(pages)
            self.slots.unpin_blocks([slot])
        self.saved.clear()
        self.requests.clear()

    def trim(self) -> None:
        # Leave headroom for two new maximum-length prompts and active rows.
        while self.saved and (
            self.slots.get_num_free_blocks() < self.reserve_slots + 3
            or self.blocks.get_num_free_blocks() < 2 * ((self.max_context + self.page - 1) // self.page)
        ):
            _, (pages, slot) = self.saved.popitem(last=False)
            self.blocks.unpin_blocks(pages)
            self.slots.unpin_blocks([slot])

    def add(self, rid: str, ids: Any, params: Any) -> Any:
        self.trim()
        keys: dict[int, bytes] = {}
        digest = blake2b(digest_size=20)
        for stop in range(self.page, len(ids) + 1, self.page):
            digest.update(ids[stop - self.page : stop])
            keys[stop] = digest.digest()
        self.requests[rid] = keys
        # fork_seq's final-token-only path is a decode fork with a supplied sampled token.
        # Use its prefill fork for independent requests with their own prompt suffix,
        # reserving at least yoco_clip tokens (or 2 when YoCo is disabled) for suffix replay.
        replay_guard = max(self.yoco_clip, 2)
        save_guard = max(self.yoco_clip, 1)
        limit = max(0, len(ids) - replay_guard) // self.page * self.page
        hit = next((n for n in reversed(keys) if n <= limit and keys[n] in self.saved), 0)
        end = max(0, len(ids) - save_guard) // self.page * self.page
        checkpoints = [end] if end > hit else []
        seq = self.engine.add_request(
            user_id="0",
            request_id=rid,
            sampling_params=params,
            prompt_token_ids=ids,
            enqueue=not hit,
            chunk_checkpoints=checkpoints,
        )
        if hit:
            key = keys[hit]
            pages, slot = self.saved[key]
            self.saved.move_to_end(key)
            seq.num_cached_tokens = seq.cached_hit_tokens = hit
            seq.state_id = slot
            seq = self.engine.scheduler.fork_seq(
                seq, pages, "_prefix", params, None, 0, -1, next(self.engine.seq_counter), time.monotonic()
            )
            seq.chunk_checkpoints = checkpoints
        return seq

    def after_prefill(self, scheduled: Any, finished: Any) -> None:
        del finished
        for seq in scheduled:
            keys = self.requests.get(seq.request_id)
            if keys is None:
                continue
            stop = seq.get_prompt_len() - seq.chunked_rest_len
            key = keys.get(stop)
            if key is None or key in self.saved:
                continue
            self.trim()
            if self.slots.get_num_free_blocks() < self.reserve_slots + 3:
                continue
            pages = self.pages_of(seq, stop // self.page)
            self.slots.lock()
            try:
                (slot,) = self.slots.allocate_nolock(1)
            finally:
                self.slots.unlock()
            self.blocks.pin_blocks(pages)
            self.engine._run_workers(
                "clone_and_zero_states_and_blocks_on_default_stream",
                clone_states=np.array([[seq.state_id, slot]], dtype=np.int32),
                clone_blocks=self.empty_pairs,
                zero_states=self.empty_ids,
            )
            self.saved[key] = pages, slot
