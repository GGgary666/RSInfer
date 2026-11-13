"""The prefix cache through the real scheduler and real workers (doc/specs/prefix-cache.md): a request
whose prompt begins with a computed one starts where the cache keeps its GDN state, on the shared
KV pages, and generates what HF generates for the whole prompt (fp32: token for token); with one
step of overlap too. This is the comparison "prefix hit against miss" of stage2-contract.md §1.3."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from core.contracts.columns import Column
from core.contracts.requests import NewRequest
from core.contracts.sampling import SamplingParams
from tests.units.support.worker import hf_greedy, hf_model, save_checkpoint, tiny_config
from tests.units.worker.support import stack

PREFIX = [(5 * i + 1) % 97 for i in range(90)]
FIRST = {1: [*PREFIX, 3, 4, 5]}
LATER = {2: [*PREFIX[:80], 9, 8, 7, 6, 5, 4, 3, 2, 1, 11, 12, 13], 3: [*PREFIX, 3, 4, 5]}
STEPS = 6
CONFIG = {"batch.max_context_len": 128}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], Path, Any]:
    cfg = tiny_config()
    model = hf_model(cfg)
    return cfg.to_dict(), save_checkpoint(model, tmp_path_factory.mktemp("ckpt")), model


def greedy(prompts: dict[int, list[int]]) -> Sequence[NewRequest]:
    return [NewRequest(rid, Column.of("i64", p), SamplingParams(temperature=0.0, seed=rid), STEPS)
            for rid, p in prompts.items()]


@pytest.mark.parametrize("overlap", [False, True])
def test_a_request_that_begins_with_a_computed_prompt_starts_from_the_cache_and_follows_hf(
        checkpoint: tuple[dict[str, Any], Path, Any], overlap: bool) -> None:
    want = {rid: hf_greedy(checkpoint[2], p, STEPS)[0] for rid, p in {**FIRST, **LATER}.items()}
    with stack(checkpoint[0], checkpoint[1], overlap=overlap, token_budget=128,
               sched={"alloc.radix_cache": True, **CONFIG},
               **{**CONFIG, "state.kv.pages": 160, "state.gdn.slots": 20}) as (b, workers):
        assert b.engine.ledger.match(0, LATER[2]) == 0
        b.client.send(greedy(FIRST))
        b.run_until(b.finished(*FIRST), timeout=60)
        b.run_until(b.idle, timeout=10)
        until = time.monotonic() + 2.5  # idle: the workers say their capacity again (worker.announce_every_s)
        b.run_until(lambda: time.monotonic() > until, timeout=10)
        # the first prompt's state at 64 (the checkpoint grid) is cached; both later ones start there
        assert [b.engine.ledger.match(0, p) for p in LATER.values()] == [64, 64]
        b.client.send(greedy(LATER))
        b.run_until(b.finished(*LATER), timeout=60)
        assert {rid: b.client.outputs[rid].tokens for rid in want} == want
        assert all(w.loop.failures == 0 for w in workers) and b.engine.loop.report().fault is None
        b.run_until(b.idle, timeout=10)
        assert b.engine.ledger.check() == []
