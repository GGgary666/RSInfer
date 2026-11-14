"""Speculative decoding on the GPU (doc/specs/mtp.md): Qwen3.8-27B's real structure cut to 4 layers,
its MTP layer drafting, through the real scheduler and a real worker on CUDA with the real kernels
(Triton for the GDN verify step, FlashInfer for attention), with one step of overlap.

Drafts come from an oracle (``Oracle``): the tokens the engine decodes without speculation, wrong at
every 7th position, so steps accept 1 to D tokens; the random MTP weights of the checkpoint would
accept next to nothing. It takes the draft model's place before ``core.plan`` is reloaded, so the draft
graphs record it. fp32 holds speculation to the tokens of decoding without it; bf16 holds a step
replayed from graphs (the draft's decode forwards, the verify, the draft extend) to the eager one.
A checkpoint is ~8.4 GiB (``QWEN38_MODEL`` reuses one)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import pytest

from core.contracts.columns import Column
from core.contracts.requests import NewRequest
from core.contracts.sampling import SamplingParams
from unit.runtime.types import ReloadPlan
from platforms.cuda_graph import CudaGraphPlatform
from platforms.torch_eager import TorchEagerPlatform
from tests.gpu.support import free_gpu, make_checkpoint
from tests.units.support.worker import spec as unit_spec
from tests.units.worker.support import Oracle, RealWorker, stack

PROMPTS = {1: [(1000 + 37 * i) % 150000 for i in range(40)], 2: [(77 + 13 * i) % 150000 for i in range(9)],
           3: [(5000 + 101 * i) % 150000 for i in range(70)]}
STEPS = 24
WARM = [7, 8, 9, 10, 11]
"""A prompt none of ``PROMPTS`` begins with: what it leaves in the prefix cache matches none of them."""
SPEC: dict[str, Any] = {"spec.algorithm": "eagle", "spec.num_steps": 3, "spec.num_draft_tokens": 4}
ROWS: dict[str, Any] = {"batch.max_context_len": 256, "batch.max_rows": 8}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return Path(os.environ["QWEN38_MODEL"]) if os.environ.get("QWEN38_MODEL") else make_checkpoint(tmp_path_factory, 4)


def draft_with(worker: RealWorker, oracle: Oracle) -> None:
    """The oracle in the draft model's place, and ``core.plan`` reloaded, as a change would reload it
    (the loop quiesced first): its graphs are recorded again, over the oracle."""
    setattr(cast(Any, worker.runtime).instance("models.qwen38").model, "draft_forward", oracle)
    worker.loop.quiesce()
    cast(Any, worker.runtime).reload(ReloadPlan(units_to_reload=[unit_spec("core.plan")])).unwrap()


def generate(checkpoint: Path, dtype: str, *, spec: bool, graphs: bool, oracle: Oracle | None = None
             ) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Each prompt's greedy tokens, and what each of its steps accepted."""
    cfg = json.loads((checkpoint / "config.json").read_text())
    counts: dict[int, list[int]] = {rid: [] for rid in PROMPTS}
    platform = CudaGraphPlatform() if graphs else TorchEagerPlatform(cuda=True)
    worker: dict[str, Any] = {"model.dtype": dtype, "state.gdn.conv_dtype": dtype, "state.kv.pages": 1024,
              "state.gdn.slots": 24, "plan.cuda_graph": "required" if graphs else "off", **ROWS, **(SPEC if spec else {})}
    try:
        with stack(cfg, checkpoint, overlap=True, token_budget=256, platform=platform,
                   sched={**ROWS, **(SPEC if spec else {})}, **worker) as (b, workers):
            if spec:  # a request on what the cold start recorded: with graphs, required fails a stale one
                b.client.send([NewRequest(0, Column.of("i64", WARM), SamplingParams(temperature=0.0, seed=0), 4)])
                b.run_until(b.finished(0), timeout=600)
                b.run_until(b.idle, timeout=60)
            if oracle is not None:
                for w in workers:
                    w.call(lambda w: draft_with(w, oracle))
            table: Any = b.engine.table
            sample = table.sample

            def counting(epoch: int, identity: str, sampled: Any) -> Any:
                for row, n in zip(sampled.rows.values(), sampled.counts.values(), strict=True):
                    counts[table.steps[epoch].rows[int(row)]].append(int(n))
                return sample(epoch, identity, sampled)

            table.sample = counting
            for rid, p in PROMPTS.items():  # one by one: each first decode while its prefill is in flight
                b.client.send([NewRequest(rid, Column.of("i64", p), SamplingParams(temperature=0.0, seed=rid), STEPS)])
                b.run_until(b.finished(rid), timeout=600)
            assert all(w.loop.failures == 0 for w in workers) and b.engine.loop.report().fault is None
            return {rid: b.client.outputs[rid].tokens for rid in PROMPTS}, counts
    finally:
        free_gpu()


def oracle_for(tokens: dict[int, list[int]], vocab: int) -> Oracle:
    return Oracle({rid: [*PROMPTS[rid], *tokens[rid]] for rid in PROMPTS}, vocab, device="cuda")


@pytest.mark.timeout(3600)
def test_speculation_on_the_real_structure_generates_what_decoding_without_it_does(checkpoint: Path) -> None:
    """fp32: the Triton verify kernels and the commit leave the states decoding leaves, at every length."""
    want, _ = generate(checkpoint, "f32", spec=False, graphs=False)
    vocab = json.loads((checkpoint / "config.json").read_text())["text_config"]["vocab_size"]
    oracle = oracle_for(want, vocab)
    got, counts = generate(checkpoint, "f32", spec=True, graphs=False, oracle=oracle)
    assert got == want
    expected = {rid: oracle.counts(len(p), STEPS, SPEC["spec.num_draft_tokens"]) for rid, p in PROMPTS.items()}
    got_counts = {rid: c[:len(expected[rid])] for rid, c in counts.items()}
    assert got_counts == expected, f"{got_counts} != {expected}"  # with overlap, a step more


@pytest.mark.timeout(3600)
def test_a_verify_step_replayed_from_a_graph_gives_what_the_eager_one_gives(checkpoint: Path) -> None:
    """bf16, the same kernels eagerly and replayed (FlashInfer's wrappers over a bucket's fixed buffers;
    the draft extend on the verify step's): the same tokens and the same accepted lengths, which the
    oracle recorded into the draft graphs decides.

    In bf16, speculation can leave the tokens of decoding without it at a near tie: on the SM100
    (2026-11-14) two of the three prompts did, where fp32's logits for the two tokens differ by 0.04 and
    0.002. The oracle drafts the tokens of decoding without it, so past that point it drafts nothing
    right. Its counts are held only for the prompts whose tokens did not leave them; the replayed
    counts are held to the eager ones for every prompt."""
    plain, _ = generate(checkpoint, "bf16", spec=False, graphs=False)
    vocab = json.loads((checkpoint / "config.json").read_text())["text_config"]["vocab_size"]
    oracle = oracle_for(plain, vocab)
    eager = generate(checkpoint, "bf16", spec=True, graphs=False, oracle=oracle)
    replayed = generate(checkpoint, "bf16", spec=True, graphs=True, oracle=oracle)
    assert replayed[0] == eager[0]
    for rid, p in PROMPTS.items():
        want, got = eager[1][rid], replayed[1][rid]
        n = min(len(want), len(got))  # with overlap, a step more where the next was planned before the end was known
        assert got[:n] == want[:n], (rid, want, got)
        if eager[0][rid] == plain[rid]:
            expected = oracle.counts(len(p), STEPS, SPEC["spec.num_draft_tokens"])
            assert want[:len(expected)] == expected, (rid, want, expected)
    assert any(n > 1 for c in eager[1].values() for n in c), "no step took a draft: nothing above is compared"
