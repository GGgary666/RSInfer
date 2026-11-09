"""Qwen3.8-27B's first 4 layers (3 GDN, 1 full attention) at their real sizes, on one CUDA GPU of
24 GB or more: the worker stack against HF's ``Qwen3_5ForCausalLM`` with the same weights.

Manual (``tests/gpu`` is not collected by default; it needs the ``gpu`` extra):

    uv sync --extra gpu --extra frontend
    uv run pytest tests/gpu/test_qwen38_partial.py -s

The weights are random, from HF's initialization, written by ``deploy/qwen38.py checkpoint`` in a
real checkpoint's layout (``model.language_model.*``), so the loader, the vocabulary of 248k, the
head counts and the partial rotary of the real model are all in play. HF runs first and is freed,
then the worker stack, its pools sized by ``core.state.holder``.

- fp32: greedy tokens equal HF's for every prompt, logits close.
- bf16: logits close to HF's in bf16. With random weights the logits of the candidates are close,
  so where greedy tokens first differ is reported, not required to be the end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from platforms.torch_eager import TorchEagerPlatform
from tests.gpu.support import free_gpu, hf_model_from, make_checkpoint
from tests.units.support.worker import Driver, Worker, hf_greedy

LAYERS = 4
STEPS = 12
PROMPTS = {0: [9707, 11, 1879, 0, 3555, 374, 279, 1888], 1: list(range(1000, 1040)), 2: [151, 2]}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_checkpoint(tmp_path_factory, LAYERS)


def hf_reference(checkpoint: Path, dtype: torch.dtype) -> tuple[dict[int, list[int]], dict[int, torch.Tensor]]:
    """HF's greedy tokens and first logits per prompt, then HF is freed."""
    hf = hf_model_from(checkpoint, dtype)
    tokens: dict[int, list[int]] = {}
    first: dict[int, torch.Tensor] = {}
    for row, prompt in PROMPTS.items():
        out, logits = hf_greedy(hf, prompt, STEPS)
        tokens[row], first[row] = out, logits[0].float().cpu()
    del hf
    free_gpu()
    return tokens, first


def ours(checkpoint: Path, dtype: str) -> tuple[dict[int, list[int]], dict[int, torch.Tensor], dict[str, int]]:
    cfg = json.loads((checkpoint / "config.json").read_text())
    w = Worker(cfg, platform=TorchEagerPlatform(cuda=True), **{
        "model.dtype": dtype, "weights.path": str(checkpoint), "state.kv.page_size": 16, "state.kv.pages": 0,
        "state.gdn.slots": 0, "batch.max_rows": 8, "batch.max_context_len": 256})
    try:
        capacity = dict(w.service("executor@1").capacity().items)
        d = Driver(w)
        for row, prompt in PROMPTS.items():
            d.add(row, prompt)
        logits = d.generate(list(PROMPTS), STEPS, chunk=16)
        return ({row: d.requests[row].outputs for row in PROMPTS}, {row: logits[row][0].float().cpu() for row in PROMPTS},
                capacity)
    finally:
        w.close()
        free_gpu()


def report(name: str, got: dict[int, list[int]], want: dict[int, list[int]]) -> dict[int, int]:
    agree = {row: next((i for i, (a, b) in enumerate(zip(got[row], want[row], strict=True)) if a != b), STEPS)
             for row in want}
    print(f"\n{name}: greedy tokens equal HF's for the first {agree} of {STEPS}")
    return agree


@pytest.mark.timeout(1800)
def test_fp32_greedy_tokens_equal_hf(checkpoint: Path) -> None:
    want, want_first = hf_reference(checkpoint, torch.float32)
    got, got_first, capacity = ours(checkpoint, "f32")
    print(f"\npools sized by the holder: {capacity}")
    assert capacity["kv.paged"] > 0 and capacity["gdn.slots"] > 0
    assert report("fp32", got, want) == {row: STEPS for row in PROMPTS}
    for row in PROMPTS:  # prefill logits: chunked in 16 here, whole in HF
        torch.testing.assert_close(got_first[row], want_first[row], rtol=1e-3, atol=1e-3)


@pytest.mark.timeout(1800)
def test_bf16_logits_close_to_hf(checkpoint: Path) -> None:
    want, want_first = hf_reference(checkpoint, torch.bfloat16)
    got, got_first, _ = ours(checkpoint, "bf16")
    report("bf16", got, want)
    for row in PROMPTS:
        a, b = got_first[row], want_first[row]
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        print(f"bf16 row {row}: prefill logits cosine {cos:.5f}, max diff {(a - b).abs().max().item():.4f}, "
              f"top-1 {int(a.argmax())} vs {int(b.argmax())}")
        assert cos > 0.99


