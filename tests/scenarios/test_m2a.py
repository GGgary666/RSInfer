"""Stage 2's M2a (stage2-contract.md §1.3): the whole engine on the CPU, started the way an operator
starts it (``deploy/qwen38.py serve``), with a small random-weight Qwen3.8 in a real checkpoint's
layout and a real HF tokenizer. Text through HTTP, out of the frontend, the scheduler host and a
worker process, equals what HF's model computes, greedy, token by token."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from tests.scenarios.deploy import cli, post, served
from tests.units.support.tokenizer import save_tokenizer
from tests.units.support.worker import hf_greedy, tiny_config

VOCAB = 300


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """In a directory of its own (``--tokenizer``): next to a ``qwen3_5`` config.json, transformers
    would load it as Qwen's tokenizer, which this small word tokenizer is not."""
    return save_tokenizer(tmp_path_factory.mktemp("tokenizer"), VOCAB, eos_in_config=False)


@pytest.fixture(scope="module")
def model(tmp_path_factory: pytest.TempPathFactory, tokenizer: Path) -> Path:
    """A checkpoint as ``checkpoint`` writes it: a multimodal Qwen3.8 config cut to the tiny model,
    random weights under ``model.language_model.``."""
    root = tmp_path_factory.mktemp("m2a")
    src = root / "config"
    src.mkdir()
    text = {**tiny_config(vocab_size=VOCAB).to_dict(), "model_type": "qwen3_5_text"}
    src.joinpath("config.json").write_text(json.dumps(
        {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "eos_token_id": 0,
         "text_config": text}))
    cli("checkpoint", "--config", str(src), "--out", str(root / "ckpt"), "--dtype", "f32", "--device", "cpu",
        "--seed", "3")
    return root / "ckpt"


@pytest.fixture(scope="module")
def reference(model: Path) -> Qwen3_5ForCausalLM:
    """HF's model with the checkpoint's weights, read back from it."""
    cfg = json.loads((model / "config.json").read_text())["text_config"]
    hf = Qwen3_5ForCausalLM(tiny_config(vocab_size=VOCAB, **{k: cfg[k] for k in ("num_hidden_layers",)})).eval()
    weights: dict[str, torch.Tensor] = {}
    for file in sorted(model.glob("*.safetensors")):
        for name, t in load_file(str(file)).items():
            if "visual" in name or name.startswith("mtp"):
                continue
            weights[name.replace("model.language_model.", "model.")] = t
    missing, unexpected = hf.load_state_dict(weights, strict=False)
    assert not unexpected and all("rotary" in m for m in missing), (missing, unexpected)
    return hf


def expected(hf: Qwen3_5ForCausalLM, tok: Any, ids: list[int], max_tokens: int) -> tuple[str, str]:
    """HF's greedy text up to the first stop token (the tokenizer's ``<|im_end|>`` or the config's
    ``<|endoftext|>``), and the finish reason."""
    out, _ = hf_greedy(hf, ids, max_tokens)
    stops = {0, int(tok.eos_token_id)}
    cut = next((i for i, t in enumerate(out) if t in stops), None)
    return tok.decode(out[:cut], skip_special_tokens=True), "length" if cut is None else "stop"


@pytest.mark.parametrize("overlap", [False, True])
def test_requests_through_the_whole_engine_follow_hf(model: Path, tokenizer: Path, reference: Qwen3_5ForCausalLM,
                                                    tmp_path: Path, overlap: bool) -> None:
    tok = AutoTokenizer.from_pretrained(str(tokenizer))
    prompts = ["w1 w2 w3 w4 w5", "w9 w8 w7", "w20 w21 w22 w23 w24 w25 w26 w27 w28 w29 w30 w31"]
    texts: list[str] = []
    args = ["--model", str(model), "--tokenizer", str(tokenizer), "--device", "cpu", "--dtype", "f32", "--conv-dtype", "f32",
            "--page-size", "4", "--max-rows", "8", "--max-context", "64", "--token-budget", "8", "--cpu-bytes", str(4 << 20),
            "--timeout", "120", *(["--overlap"] if overlap else [])]
    with served(tmp_path / "run", *args) as url:
        for prompt in prompts:
            got = post(url + "/v1/completions", {"model": "Qwen3.8-27B", "prompt": prompt, "max_tokens": 12,
                                                 "temperature": 0.0})
            text, reason = expected(reference, tok, list(tok.encode(prompt)), 12)
            assert (got["choices"][0]["text"], got["choices"][0]["finish_reason"]) == (text, reason), prompt
            texts.append(text)
        chat = [{"role": "user", "content": "w5 w6 w7"}]
        ids = list(tok.encode(str(tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False))))
        text, _ = expected(reference, tok, ids, 10)
        got = post(url + "/v1/chat/completions", {"model": "Qwen3.8-27B", "messages": chat, "max_tokens": 10,
                                                  "temperature": 0.0})
        assert got["choices"][0]["message"]["content"] == text
        streamed = cli("chat", "w5 w6 w7", "--url", url, "--max-tokens", "10")  # the CLI streams it (SSE)
        assert streamed == text
    assert sum(len(t.split()) for t in texts) >= 12, texts  # the model generated, rather than stopped at once
