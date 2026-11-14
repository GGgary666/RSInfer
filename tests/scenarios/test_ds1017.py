"""DS 1017 (doc/specs/v1-ds1017.md) through the whole engine on the CPU, started the way an operator
starts it (``deploy/agxinfer.py serve``): a tiny random model with every kind of layer in the real
checkpoint's formats, a real HF tokenizer, the reference kernels. Text through HTTP, out of the
frontend, the scheduler host and a worker process, equals what the worker units give in this
process for the same token ids: the prompt there in one piece, here in chunks of the token budget
with a step of overlap, which the reference kernels do not let show.

The deploy tool picks the recipe from the checkpoint's ``config.json`` and gives the worker the
units of this model family alone; the Engram token map comes from the tokenizer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from models.ds1017_config import compressed_token_map
from tests.scenarios.deploy import cli, post, served
from tests.units.support.ds1017 import DSWorker, driver, tiny_config
from tests.units.support.tokenizer import save_tokenizer

VOCAB = 300
MODEL = "DeepSeek-V4.1-Flash"


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return save_tokenizer(tmp_path_factory.mktemp("tokenizer"), VOCAB, eos_in_config=False)


@pytest.fixture(scope="module")
def model(tmp_path_factory: pytest.TempPathFactory, tokenizer: Path) -> tuple[dict[str, Any], Path]:
    """A checkpoint as ``checkpoint`` writes it for a tokenizer: no stand-in token map, so the engine
    derives Engram's from ``tokenizer.json`` as it does for the real checkpoint."""
    root = tmp_path_factory.mktemp("ds1017")
    _, compressed = compressed_token_map(Tokenizer.from_file(str(tokenizer / "tokenizer.json")))
    cfg = {**tiny_config(vocab_size=VOCAB, engram_compressed_vocab_size=compressed), "eos_token_id": 0}
    src = root / "config"
    src.mkdir()
    src.joinpath("config.json").write_text(json.dumps(cfg))
    cli("checkpoint", "--config", str(src), "--out", str(root / "ckpt"), "--tokenizer", str(tokenizer), "--seed", "3")
    assert (root / "ckpt" / "tokenizer.json").is_file() and not (root / "ckpt" / "engram_token_map.json").exists()
    return cfg, root / "ckpt"


def expected(cfg: dict[str, Any], checkpoint: Path, tok: Any, prompts: list[list[int]], max_tokens: int) -> list[str]:
    """The worker units in this process on each prompt alone, the prompt in one piece."""
    texts: list[str] = []
    for ids in prompts:
        w = DSWorker(cfg, **{"weights.path": str(checkpoint), "model.path": str(checkpoint), "model.hf_config": {}})
        try:
            d = driver(w)
            d.add(0, ids)
            d.generate([0], max_tokens)
            out = list(d.requests[0].outputs)
        finally:
            w.close()
        stops = {0, int(tok.eos_token_id)}
        cut = next((i for i, t in enumerate(out) if t in stops), None)
        texts.append(tok.decode(out[:cut], skip_special_tokens=True))
    return texts


def test_requests_through_the_whole_engine_are_what_the_worker_units_compute(
        model: tuple[dict[str, Any], Path], tokenizer: Path, tmp_path: Path) -> None:
    cfg, checkpoint = model
    tok = AutoTokenizer.from_pretrained(str(tokenizer))
    prompts = ["w1 w2 w3 w4 w5", "w9 w8 w7", "w20 w21 w22 w23 w24 w25 w26 w27 w28 w29 w30 w31 w32 w33 w34 w35 w36 w37 w38"]
    chat = [{"role": "user", "content": "w5 w6 w7"}]
    ids = [list(tok.encode(p)) for p in prompts]
    ids.append(list(tok.encode(str(tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)))))
    want = expected(cfg, checkpoint, tok, ids, 10)
    args = ["--model", str(checkpoint), "--tokenizer", str(tokenizer), "--device", "cpu", "--page-size", "4",
            "--max-rows", "4", "--max-context", "64", "--token-budget", "8", "--cpu-bytes", str(16 << 20),
            "--timeout", "120", "--ds-backend", "reference", "--overlap"]
    with served(tmp_path / "run", *args) as url:
        for prompt, text in zip(prompts, want, strict=False):
            got = post(url + "/v1/completions", {"model": MODEL, "prompt": prompt, "max_tokens": 10, "temperature": 0.0})
            assert got["choices"][0]["text"] == text, prompt
        got = post(url + "/v1/chat/completions", {"model": MODEL, "messages": chat, "max_tokens": 10, "temperature": 0.0})
        assert got["choices"][0]["message"]["content"] == want[-1]
        topology = json.loads((tmp_path / "run" / "topology.json").read_text())
    workers = next(r["units"] for r in topology["roles"] if r["name"] == "worker")
    assert "models.ds1017" in workers and "models.qwen38" not in workers and "layers.gdn" not in workers
    assert sum(len(t.split()) for t in want) >= 12, want  # the model generated, rather than stopped at once
