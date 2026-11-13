"""Speculative decoding through the real scheduler and real workers (doc/specs/mtp.md): with the
model's MTP layer drafting (random weights here), every request generates what HF generates
(greedy, fp32: token for token), with overlap or not, with the prefix cache or not. This is the
comparison "speculation against none" of stage2-contract.md §1.3."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from core.contracts.columns import Column
from core.contracts.requests import NewRequest
from core.contracts.sampling import SamplingParams
from tests.units.support.worker import hf_greedy, hf_model, mtp_weights, save_checkpoint, save_mtp, tiny_config
from tests.units.worker.support import Oracle, stack

PROMPTS = {1: [(5 * i + 1) % 97 for i in range(30)], 2: [3, 4, 5], 3: [(7 * i + 2) % 97 for i in range(70)]}
STEPS = 9
SPEC: dict[str, Any] = {"spec.algorithm": "eagle", "spec.num_steps": 3, "spec.num_draft_tokens": 4}
CONFIG: dict[str, Any] = {"batch.max_context_len": 128}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], Path, Any]:
    cfg = tiny_config()
    model = hf_model(cfg)
    directory = save_checkpoint(model, tmp_path_factory.mktemp("ckpt"))
    return cfg.to_dict(), save_mtp(mtp_weights(cfg), directory), model


def greedy(prompts: dict[int, list[int]]) -> Sequence[NewRequest]:
    return [NewRequest(rid, Column.of("i64", p), SamplingParams(temperature=0.0, seed=rid), STEPS)
            for rid, p in prompts.items()]


@pytest.mark.parametrize(("overlap", "cache"), [(False, False), (True, False), (False, True), (True, True)])
def test_speculative_decoding_generates_what_hf_generates(checkpoint: tuple[dict[str, Any], Path, Any],
                                                         overlap: bool, cache: bool) -> None:
    want = {rid: hf_greedy(checkpoint[2], p, STEPS)[0] for rid, p in PROMPTS.items()}
    with stack(checkpoint[0], checkpoint[1], overlap=overlap, token_budget=32,
               sched={"alloc.radix_cache": cache, **SPEC, **CONFIG},
               **{**SPEC, **CONFIG, "state.kv.pages": 160, "state.gdn.slots": 20}) as (b, workers):
        b.client.send(greedy(PROMPTS))
        b.run_until(b.finished(*PROMPTS), timeout=120)
        assert {rid: b.client.outputs[rid].tokens for rid in want} == want
        assert all(w.loop.failures == 0 for w in workers) and b.engine.loop.report().fault is None
        b.run_until(b.idle, timeout=10)
        assert b.engine.ledger.check() == []


def slow(worker: Any, seconds: float) -> None:
    """The worker's steps take ``seconds`` more, as a GPU's do: the scheduler plans the next step
    (overlap) before the last one's answer is back, so a row's plan position is only where it is
    at least, 0 for its first decode while its prefill is still on its way."""
    import time
    executor = worker.loop.executor
    submit = executor.submit

    def later(*args: Any) -> Any:
        out = submit(*args)
        time.sleep(seconds)
        return out

    executor.submit = later


@pytest.mark.parametrize(("overlap", "cache", "slowly"), [(False, False, False), (True, False, False),
                                                          (False, True, False), (True, True, False),
                                                          (True, False, True), (True, True, True)])
def test_drafts_the_target_accepts_give_its_tokens_whatever_length_each_step_accepts(
        checkpoint: tuple[dict[str, Any], Path, Any], overlap: bool, cache: bool, slowly: bool) -> None:
    """With drafts mostly right, steps accept 1 to D tokens: the GDN states committed at each length,
    the KV written past what was accepted and written over next, the rows moving by more than one
    token while the next step is planned (overlap; ``slowly``: before the step before it answered),
    and the prefix cache tracking inside a step."""
    steps = 24
    want = {rid: hf_greedy(checkpoint[2], p, steps)[0] for rid, p in PROMPTS.items()}
    oracle = Oracle({rid: [*PROMPTS[rid], *want[rid]] for rid in PROMPTS}, checkpoint[0]["vocab_size"])
    counts: dict[int, list[int]] = {rid: [] for rid in PROMPTS}
    with stack(checkpoint[0], checkpoint[1], overlap=overlap, token_budget=32,
               sched={"alloc.radix_cache": cache, "alloc.mamba_track_interval": 64, **SPEC, **CONFIG},
               **{**SPEC, **CONFIG, "state.kv.pages": 160, "state.gdn.slots": 20}) as (b, workers):
        for w in workers:
            w.call(lambda w: setattr(cast(Any, w.runtime).instance("models.qwen38").model, "draft_forward", oracle))
            if slowly:
                w.call(lambda w: slow(w, 0.02))
        table: Any = b.engine.table
        sample = table.sample

        def counting(epoch: int, identity: str, sampled: Any) -> Any:
            for row, n in zip(sampled.rows.values(), sampled.counts.values(), strict=True):
                counts[table.steps[epoch].rows[int(row)]].append(int(n))
            return sample(epoch, identity, sampled)

        table.sample = counting
        new = [NewRequest(rid, Column.of("i64", p), SamplingParams(temperature=0.0, seed=rid), steps)
               for rid, p in PROMPTS.items()]
        for batch in ([[r] for r in new] if slowly else [new]):  # one by one: a decode right after its prefill
            b.client.send(batch)
            b.run_until(b.finished(*(r.request for r in batch)), timeout=120)
        assert {rid: b.client.outputs[rid].tokens for rid in want} == want
        expected = {rid: oracle.counts(len(p), steps, SPEC["spec.num_draft_tokens"]) for rid, p in PROMPTS.items()}
        assert {rid: c[:len(expected[rid])] for rid, c in counts.items()} == expected  # with overlap, a step more
        assert {n for c in expected.values() for n in c} == {1, 2, 3, 4}  # every accepted length happened
        assert all(w.loop.failures == 0 for w in workers) and b.engine.loop.report().fault is None
        b.run_until(b.idle, timeout=10)
        assert b.engine.ledger.check() == []


def test_a_state_kept_inside_a_verify_step_is_where_a_later_prompt_starts_and_follows_hf(
        checkpoint: tuple[dict[str, Any], Path, Any]) -> None:
    """The prefix cache with speculation (SGLang's ``mamba_steps_to_track``): the first request's
    decode crosses 64 inside a step that accepts several tokens; the state there is cached when it
    ends, and a prompt that begins with its tokens starts at 64 and generates what HF does."""
    first, steps = [(3 * i + 5) % 97 for i in range(40)], 40
    out1 = hf_greedy(checkpoint[2], first, steps)[0]
    seq1 = [*first, *out1]
    oracle1 = Oracle({1: seq1}, checkpoint[0]["vocab_size"])
    crossing = oracle1.counts(len(first), steps, SPEC["spec.num_draft_tokens"])
    starts = [len(first) + sum(crossing[1:i]) for i in range(1, len(crossing))]
    at = next(s for s, n in zip(starts, crossing[1:], strict=True) if s < 64 <= s + n)
    assert 64 - at > 1  # the state at 64 is one inside the step, not its start's
    second = seq1[:66]  # 2 tokens after the cached 64: the state fades within a few dozen with these weights
    out2 = hf_greedy(checkpoint[2], second, 6)[0]
    with stack(checkpoint[0], checkpoint[1], overlap=False, token_budget=32,
               sched={"alloc.radix_cache": True, "alloc.mamba_track_interval": 64, **SPEC, **CONFIG},
               **{**SPEC, **CONFIG, "state.kv.pages": 160, "state.gdn.slots": 20}) as (b, workers):
        def draft_with(oracle: Oracle) -> None:
            for w in workers:
                w.call(lambda w: setattr(cast(Any, w.runtime).instance("models.qwen38").model, "draft_forward", oracle))

        draft_with(oracle1)
        b.client.send([NewRequest(1, Column.of("i64", first), SamplingParams(temperature=0.0, seed=1), steps)])
        b.run_until(b.finished(1), timeout=120)
        assert b.client.outputs[1].tokens == out1
        b.run_until(b.idle, timeout=10)
        assert b.engine.ledger.match(2, second) == 64
        draft_with(Oracle({2: [*second, *out2]}, checkpoint[0]["vocab_size"]))
        b.client.send([NewRequest(2, Column.of("i64", second), SamplingParams(temperature=0.0, seed=2), 6)])
        b.run_until(b.finished(2), timeout=120)
        assert b.client.outputs[2].tokens == out2
        assert all(w.loop.failures == 0 for w in workers) and b.engine.loop.report().fault is None


@pytest.mark.parametrize("overlap", [False, True])
def test_with_temperature_the_tokens_are_drawn_from_seed_and_position_the_same_every_time(
        checkpoint: tuple[dict[str, Any], Path, Any], overlap: bool) -> None:
    """Target-only speculative sampling with coins from (seed, position): two engines given the same
    requests give the same tokens, with overlap or not."""
    runs: list[dict[int, list[int]]] = []
    for _ in range(2):
        with stack(checkpoint[0], checkpoint[1], overlap=overlap, token_budget=32, sched={**SPEC, **CONFIG},
                   **{**SPEC, **CONFIG, "state.kv.pages": 160, "state.gdn.slots": 20}) as (b, workers):
            b.client.send([NewRequest(rid, Column.of("i64", p), SamplingParams(temperature=0.8, top_p=0.9, seed=rid),
                                      STEPS) for rid, p in PROMPTS.items()])
            b.run_until(b.finished(*PROMPTS), timeout=120)
            runs.append({rid: b.client.outputs[rid].tokens for rid in PROMPTS})
            assert all(len(t) == STEPS for t in runs[-1].values())
            assert all(w.loop.failures == 0 for w in workers)
    assert runs[0] == runs[1]
