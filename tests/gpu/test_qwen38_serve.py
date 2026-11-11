"""The whole engine on one CUDA GPU, started as an operator starts it (``deploy/agxinfer.py serve``):
text over HTTP equals the text of HF's greedy generation, with and without one step of overlap.

Manual (``tests/gpu`` is not collected by default):

    QWEN38_TOKENIZER=<a Qwen3.8 tokenizer directory> uv run pytest tests/gpu/test_qwen38_serve.py -s

- ``QWEN38_MODEL``: the checkpoint to serve. Default: Qwen3.8-27B's first 4 layers with random
  weights (``make_checkpoint``), which fits a 24 GB GPU. On a SM100, the real model's directory.
- HF runs first and is freed, then the engine starts: the state holder sizes the pools from the
  whole device, so the two would not fit side by side.
- ``QWEN38_DTYPE``: ``f32`` (default) or ``bf16``. In fp32 the text must be HF's exactly. In bf16
  two implementations may round two close candidates differently (stage2-contract.md §6: the
  comparisons use a tolerance), so there the test reports where the texts part and requires the
  engine to agree with itself with and without overlap.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from frontend.tokenizer.unit import HFTokenizer
from tests.gpu.support import free_gpu, hf_model_from, make_checkpoint
from tests.scenarios.deploy import post, served
from tests.units.support.worker import hf_greedy

TOKENIZER = os.environ.get("QWEN38_TOKENIZER", "")
DTYPE = os.environ.get("QWEN38_DTYPE", "f32")
PROMPTS = ["The capital of France is", "def fibonacci(n):", "你好，我是"]
MAX_TOKENS = 8


@pytest.fixture(scope="module")
def model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return Path(os.environ["QWEN38_MODEL"]) if os.environ.get("QWEN38_MODEL") else make_checkpoint(tmp_path_factory, 4)


@pytest.fixture(scope="module")
def tokenizer() -> HFTokenizer:
    assert TOKENIZER, "set QWEN38_TOKENIZER to a Qwen3.8 tokenizer directory"
    return HFTokenizer(TOKENIZER)


@pytest.fixture(scope="module")
def reference(model: Path, tokenizer: HFTokenizer) -> dict[str, str]:
    """HF's greedy text per prompt, encoded, stopped and decoded as the frontend does."""
    hf = hf_model_from(model, torch.float32 if DTYPE == "f32" else torch.bfloat16)
    out: dict[str, str] = {}
    stops = set(tokenizer.stop_token_ids)
    for prompt in PROMPTS:
        tokens, _ = hf_greedy(hf, tokenizer.encode(prompt), MAX_TOKENS)
        cut = next((i for i, t in enumerate(tokens) if t in stops), None)
        out[prompt] = tokenizer.decode(tokens[:cut])
    del hf
    free_gpu()
    return out


SEEN: dict[bool, dict[str, str]] = {}


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("overlap", [False, True])
def test_text_over_http_equals_hf(model: Path, reference: dict[str, str], tmp_path: Path, overlap: bool) -> None:
    args = ["--model", str(model), "--tokenizer", TOKENIZER, "--max-context", "4096", "--dtype", DTYPE,
            "--conv-dtype", "f32" if DTYPE == "f32" else "bf16", *(["--overlap"] if overlap else [])]
    with served(tmp_path / "run", *args, ready_s=3600) as url:
        for prompt in PROMPTS:
            got = post(url + "/v1/completions", {"model": "Qwen3.8-27B", "prompt": prompt, "max_tokens": MAX_TOKENS,
                                                 "temperature": 0.0})
            text = got["choices"][0]["text"]
            print(f"\n{prompt!r} -> {text!r} (HF {reference[prompt]!r})")
            SEEN.setdefault(overlap, {})[prompt] = text
            if DTYPE == "f32":
                assert text == reference[prompt]
    if DTYPE != "f32" and len(SEEN) == 2:
        assert SEEN[True] == SEEN[False]
