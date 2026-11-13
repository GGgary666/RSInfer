"""Speculative decoding on the GPU (doc/specs/mtp.md): Qwen3.8-27B's real structure cut to 4 layers,
its MTP layer drafting, through the real scheduler and a real worker on CUDA with the real kernels
(Triton for the GDN verify step, FlashInfer for attention), with one step of overlap.

Drafts come from an oracle (``Oracle``): the tokens the engine decodes without speculation, wrong at
every 7th position, so steps accept 1 to D tokens; the random MTP weights of the checkpoint would
accept next to nothing. fp32 holds speculation to the tokens of decoding without it; bf16 holds a
verify step replayed from a graph to the eager one. A checkpoint is ~8.4 GiB (``QWEN38_MODEL``
reuses one)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import pytest

from core.contracts.columns import Column
from core.contracts.requests import NewRequest
from core.contracts.sampling import SamplingParams
from platforms.cuda_graph import CudaGraphPlatform
from platforms.torch_eager import TorchEagerPlatform
from tests.gpu.support import free_gpu, make_checkpoint
from tests.units.worker.support import Oracle, stack

PROMPTS = {1: [(1000 + 37 * i) % 150000 for i in range(40)], 2: [(77 + 13 * i) % 150000 for i in range(9)],
           3: [(5000 + 101 * i) % 150000 for i in range(70)]}
STEPS = 24
SPEC: dict[str, Any] = {"spec.algorithm": "eagle", "spec.num_steps": 3, "spec.num_draft_tokens": 4}
ROWS: dict[str, Any] = {"batch.max_context_len": 256, "batch.max_rows": 8}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return Path(os.environ["QWEN38_MODEL"]) if os.environ.get("QWEN38_MODEL") else make_checkpoint(tmp_path_factory, 4)


def generate(checkpoint: Path, dtype: str, *, spec: bool, graphs: bool, oracle: Oracle | None = None
             ) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Each prompt's greedy tokens, and what each of its steps accepted."""
    cfg = json.loads((checkpoint / "config.json").read_text())
    counts: dict[int, list[int]] = {rid: [] for rid in PROMPTS}
    platform = CudaGraphPlatform() if graphs else TorchEagerPlatform(cuda=True)
    worker: dict[str, Any] = {"model.dtype": dtype, "state.gdn.conv_dtype": dtype, "state.kv.pages": 1024,
              "state.gdn.slots": 24, "plan.cuda_graph": "auto" if graphs else "off", **ROWS, **(SPEC if spec else {})}
    try:
        with stack(cfg, checkpoint, overlap=True, token_budget=256, platform=platform,
                   sched={**ROWS, **(SPEC if spec else {})}, **worker) as (b, workers):
            if oracle is not None:
                for w in workers:
                    w.call(lambda w: setattr(cast(Any, w.runtime).instance("models.qwen38").model, "draft_forward",
                                             oracle))
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
    return Oracle({rid: [*PROMPTS[rid], *tokens[rid]] for rid in PROMPTS}, vocab)


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
    """bf16, the same kernels eagerly and replayed (FlashInfer's prefill wrapper over a bucket's fixed
    buffers): the same tokens and the same accepted lengths."""
    plain, _ = generate(checkpoint, "bf16", spec=False, graphs=False)
    vocab = json.loads((checkpoint / "config.json").read_text())["text_config"]["vocab_size"]
    oracle = oracle_for(plain, vocab)
    eager = generate(checkpoint, "bf16", spec=True, graphs=False, oracle=oracle)
    replayed = generate(checkpoint, "bf16", spec=True, graphs=True, oracle=oracle)
    assert replayed == eager
    expected = {rid: oracle.counts(len(p), STEPS, SPEC["spec.num_draft_tokens"]) for rid, p in PROMPTS.items()}
    assert {rid: c[:len(expected[rid])] for rid, c in eager[1].items()} == expected
