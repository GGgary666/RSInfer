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
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from deploy.models.hf import TOKENIZER_FILES
from models.ds1017_config import compressed_token_map
from tests.scenarios.deploy import CLI, ENV, cli, post, served
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
    derives Engram's from ``tokenizer.json`` as it does for the real checkpoint. The tokenizer is then
    left in its own directory alone (``--tokenizer``): the worker has to be told where it is."""
    root = tmp_path_factory.mktemp("ds1017")
    _, compressed = compressed_token_map(Tokenizer.from_file(str(tokenizer / "tokenizer.json")))
    cfg = {**tiny_config(vocab_size=VOCAB, engram_compressed_vocab_size=compressed), "eos_token_id": 0}
    src = root / "config"
    src.mkdir()
    src.joinpath("config.json").write_text(json.dumps(cfg))
    cli("checkpoint", "--config", str(src), "--out", str(root / "ckpt"), "--tokenizer", str(tokenizer), "--seed", "3")
    assert (root / "ckpt" / "tokenizer.json").is_file() and not (root / "ckpt" / "engram_token_map.json").exists()
    for name in TOKENIZER_FILES:
        (root / "ckpt" / name).unlink(missing_ok=True)
    return cfg, root / "ckpt"


def expected(cfg: dict[str, Any], checkpoint: Path, tokenizer: Path, tok: Any, prompts: list[list[int]],
             max_tokens: int) -> list[str]:
    """The worker units in this process on each prompt alone, the prompt in one piece."""
    texts: list[str] = []
    for ids in prompts:
        w = DSWorker(cfg, **{"weights.path": str(checkpoint), "model.path": str(checkpoint), "model.hf_config": {},
                             "dsv4.token_map_path": str(tokenizer)})
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
    want = expected(cfg, checkpoint, tokenizer, tok, ids, 10)
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


@pytest.mark.parametrize("balanced", [False, True])
def test_two_ranks_that_hold_the_model_between_them_give_what_one_rank_gives(
        model: tuple[dict[str, Any], Path], tokenizer: Path, tmp_path: Path, balanced: bool) -> None:
    """``--ranks 2`` (doc/specs/v1-ds1017.md stage 3): two workers, each with half of every layer's
    routed experts and half of the Engram tables' rows, and with requests of its own (a scheduling
    domain each). A token's experts and table rows are on either rank, so every step is computed by
    both, also a step in which one of them has no request. The text is what one rank holding all of
    it computes: an expert's term for a token is the same wherever it runs, and the token's own rank
    adds its terms in the same order.

    ``balanced``: with expert parallel load balancing. Each layer has two spare slots for copies, and
    every four steps the experts are placed again by the tokens they took, their weights moved between
    the ranks while requests run. The text is still the same."""
    cfg, checkpoint = model
    tok = AutoTokenizer.from_pretrained(str(tokenizer))
    prompts = ["w1 w2 w3 w4 w5", "w9 w8 w7", "w20 w21 w22 w23 w24 w25 w26 w27 w28 w29 w30 w31 w32 w33 w34 w35 w36 w37 w38",
               "w40 w41 w42 w43", "w50 w51", "w60 w61 w62 w63 w64 w65 w66"]
    want = expected(cfg, checkpoint, tokenizer, tok, [list(tok.encode(p)) for p in prompts], 10)
    args = ["--model", str(checkpoint), "--tokenizer", str(tokenizer), "--device", "cpu", "--page-size", "4",
            "--max-rows", "4", "--max-context", "64", "--token-budget", "8", "--cpu-bytes", str(16 << 20),
            "--timeout", "120", "--ds-backend", "reference", "--ranks", "2"]
    if balanced:
        args += ["--set", "eplb.interval=4", "--set", "eplb.window=4", "--set", "eplb.extra_experts=2"]

    def ask(prompt: str) -> str:
        got = post(url + "/v1/completions", {"model": MODEL, "prompt": prompt, "max_tokens": 10, "temperature": 0.0})
        return str(got["choices"][0]["text"])

    with served(tmp_path / "run", *args) as url:
        assert ask(prompts[0]) == want[0]  # alone: the other rank has no request, and runs its experts
        with ThreadPoolExecutor(len(prompts)) as pool:  # together: both ranks have requests
            assert list(pool.map(ask, prompts)) == want
        topology = json.loads((tmp_path / "run" / "topology.json").read_text())
        config = json.loads((tmp_path / "run" / "config.json").read_text())
        logs = sorted((tmp_path / "run").glob("host-a/snapshots/.run/worker-*/log"))
        ran = [log.read_text().count("core.plan:") for log in logs]
        placed = logs[0].read_text().count("experts placed again")
    assert len(topology["placement"]) == 2 and config["sched.domains"] == 2 and config["comm.size"] == 2
    assert len(config["sched.workers"]) == 2 and config["comm.rendezvous"].startswith("file://")
    assert len(logs) == 2 and all(ran), (logs, ran)
    assert (placed >= 3) if balanced else placed == 0, placed


def test_serve_tells_the_worker_where_the_token_map_is_and_refuses_dummy_weights(model: tuple[dict[str, Any], Path],
                                                                               tokenizer: Path, tmp_path: Path) -> None:
    """With ``--layers`` the worker is given the cut config in place of the model's directory, and it is
    never given the tokenizer's: the recipe says where Engram's token map is. Dummy weights (floating-
    point tensors only, for a model stored as bytes) are refused before anything starts."""
    run = tmp_path / "run"
    serve = ["serve", str(model[1]), "--tokenizer", str(tokenizer), "--layers", "3", "--device", "cpu", "--run", str(run),
             "--dry-run"]
    cli(*serve)
    config = json.loads((run / "config.json").read_text())
    assert config["dsv4.token_map_path"] == str(tokenizer.resolve()) and "model.path" not in config
    assert config["model.hf_config"]["text_config"]["num_hidden_layers"] == 3 and config["weights.path"] == str(model[1].resolve())
    # An explicit --tokenizer overrides a stand-in map in the model directory, matching frontend.tokenizer_path.
    with_stand_in = tmp_path / "stand_in_model"
    with_stand_in.mkdir()
    with_stand_in.joinpath("config.json").write_text(json.dumps(model[0]))
    with_stand_in.joinpath("engram_token_map.json").write_text("[0, 0]")
    cli("serve", str(with_stand_in), "--weights", str(model[1]), "--tokenizer", str(tokenizer), "--device", "cpu",
        "--run", str(tmp_path / "run_override"), "--dry-run")
    override = json.loads((tmp_path / "run_override" / "config.json").read_text())
    assert override["dsv4.token_map_path"] == override["frontend.tokenizer_path"] == str(tokenizer.resolve())
    out = subprocess.run([sys.executable, str(CLI), *serve, "--dummy"], env=ENV, capture_output=True, text=True, timeout=60)
    assert out.returncode != 0 and "DS 1017 takes no dummy weights" in out.stderr and "checkpoint" in out.stderr, out.stderr
