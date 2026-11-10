"""Real control-plane apply: kernel changes recapture, failed builds restore serving."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.scenarios.deploy import ENV, REPO, cli, post, served
from tests.units.support.tokenizer import save_tokenizer
from tests.units.support.worker import tiny_config


@pytest.mark.timeout(600)
@pytest.mark.parametrize("overlap", [False, True])
def test_kernel_apply_and_failed_graph_build_restore_serving(tmp_path: Path, overlap: bool) -> None:
    config = tmp_path / "config"
    config.mkdir()
    (config / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "eos_token_id": 0,
        "text_config": {**tiny_config(vocab_size=300).to_dict(), "model_type": "qwen3_5_text"}}))
    checkpoint = tmp_path / "checkpoint"
    cli("checkpoint", "--config", str(config), "--out", str(checkpoint), "--dtype", "f32", "--device", "cpu",
        "--seed", "3")
    tokenizer = save_tokenizer(tmp_path / "tokenizer", 300, eos_in_config=False)
    source = tmp_path / "source"
    shutil.copytree(REPO / "engine", source / "engine", ignore=shutil.ignore_patterns("__pycache__"))
    kernel = source / "engine/kernels/reference/ops.py"
    original = kernel.read_text()
    args = ["--model", str(checkpoint), "--tokenizer", str(tokenizer), "--source", str(source),
            "--device", "cuda", "--cuda-graph", "required", "--max-rows", "4", "--max-context", "64",
            "--dtype", "f32", "--conv-dtype", "f32", "--page-size", "4", "--max-slots", "8",
            "--mem-fraction", "0.1", "--timeout", "120", *(["--overlap"] if overlap else [])]
    body = {"model": "Qwen3.8-27B", "prompt": "w1 w2 w3 w4 w5", "temperature": 0.0,
            "max_tokens": 8, "ignore_eos": True}
    run = tmp_path / "run"
    with served(run, *args, ready_s=180) as url:
        control = json.loads((run / "control/control.json").read_text())["control"]

        def client(*arguments: str) -> tuple[int, dict[str, Any]]:
            p = subprocess.run([sys.executable, "-m", "core.control.client", "--control", control, *arguments],
                               env=ENV, capture_output=True, text=True, timeout=180)
            assert p.stdout.strip(), p.stderr
            return p.returncode, json.loads(p.stdout)

        baseline = post(url + "/v1/completions", body)["choices"][0]["text"]
        assert baseline
        kernel.write_text(original.replace("return F.linear(x, weight, bias)", "return F.linear(x, weight, bias) * 0"))
        code, checked = client("check", "--source", str(source))
        assert code == 0, checked
        code, dry = client("apply", "--source", str(source), "--dry-run")
        assert code == 0 and "core.plan" in json.dumps(dry), dry
        code, applied = client("apply", "--source", str(source))
        assert code == 0 and applied["status"] == "committed", applied
        assert post(url + "/v1/completions", body)["choices"][0]["text"] == ""
        # A second bucket cannot fit this invalid configuration: rollback must recapture the
        # old version and resume only with valid providers. No xfail on recovery failure.
        code, failed = client("apply", "--config", '{"plan.graph_buckets": [0]}')
        assert code != 0 and failed["status"] != "committed", failed
        assert post(url + "/v1/completions", body)["choices"][0]["text"] == ""
        kernel.write_text(original)
        code, restored = client("apply", "--source", str(source))
        assert code == 0 and restored["status"] == "committed", restored
        assert post(url + "/v1/completions", body)["choices"][0]["text"] == baseline
        logs = "\n".join(p.read_text() for p in (run / "host-a/snapshots/.run").rglob("log"))
        assert logs.count('"buckets": [1, 2, 4]') >= 3, logs[-5000:]
    with served(tmp_path / "cold", *args, ready_s=180) as url:
        assert post(url + "/v1/completions", body)["choices"][0]["text"] == baseline
