"""Real apply: config recapture/reuse, kernel changes, failed-build recovery, and what repeated
changes leave on the device."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.scenarios.deploy import ENV, REPO, cli, post, served
from tests.units.support.tokenizer import save_tokenizer
from tests.units.support.worker import tiny_config


def deployment(tmp_path: Path, overlap: bool = False, mem_fraction: str = "0.1") -> tuple[list[str], dict[str, Any]]:
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
    args = ["--model", str(checkpoint), "--tokenizer", str(tokenizer), "--source", str(source),
            "--device", "cuda", "--cuda-graph", "required", "--max-rows", "4", "--max-context", "64",
            "--dtype", "f32", "--conv-dtype", "f32", "--page-size", "4", "--max-slots", "8",
            "--mem-fraction", mem_fraction, "--timeout", "120", *(["--overlap"] if overlap else [])]
    body = {"model": "Qwen3.8-27B", "prompt": "w1 w2 w3 w4 w5", "temperature": 0.0,
            "max_tokens": 8, "ignore_eos": True}
    return args, body


def client(run: Path, *arguments: str) -> tuple[int, dict[str, Any]]:
    control = json.loads((run / "control/control.json").read_text())["control"]
    p = subprocess.run([sys.executable, "-m", "core.control.client", "--control", control, *arguments],
                       env=ENV, capture_output=True, text=True, timeout=180)
    assert p.stdout.strip(), p.stderr
    return p.returncode, json.loads(p.stdout)


def graph_state(run: Path) -> tuple[int, int, list[int]]:
    code, intro = client(run, "introspect")
    assert code == 0, intro
    worker = next(p for p in intro["procs"] if p["proc"]["role"] == "worker")
    plan = next(u for u in worker["units"] if u["unit"] == "core.plan")
    assert plan["state"] == "ACTIVE", plan
    logs = "\n".join(p.read_text() for p in (run / "host-a/snapshots/.run").rglob("log"))
    records = [json.loads(line.split("core.plan: ", 1)[1]) for line in logs.splitlines() if "core.plan: " in line]
    graph = records[-1]["cuda_graph"]
    assert graph["mode"] == "required" and not graph["reason"], graph
    assert set(graph["seconds"]) == {str(b) for b in graph["buckets"]} and graph["kept_bytes"] > 0, graph
    return plan["generation"], logs.count("TEST_GRAPH_CAPTURE_COMPLETED"), graph["buckets"]


@pytest.mark.timeout(600)
def test_config_changes_recapture_only_graph_dependencies(tmp_path: Path) -> None:
    args, body = deployment(tmp_path)
    graph_source = tmp_path / "source/engine/core/plan/graph.py"
    original = graph_source.read_text()
    # Observe completed capture contexts in the test snapshot, without changing production code.
    target = "        self.graph = graph\n"
    assert original.count(target) == 4  # GraphPlan's; and the verify, draft and draft extend plans' (speculation, off here)
    graph_source.write_text(original.replace(target, target + '        print("TEST_GRAPH_CAPTURE_COMPLETED", flush=True)\n'))
    run = tmp_path / "run"
    cases: list[tuple[str, dict[str, Any], bool, list[int]]] = [
        ("live overlap", {"sched.overlap": True}, False, [1, 2, 4]),
        ("scheduler batch limit", {"sched.max_batch": 2}, False, [1, 2, 4]),
        ("graph buckets", {"plan.graph_buckets": [1, 4]}, True, [1, 4]),
        ("unchanged buckets", {"plan.graph_buckets": [1, 4]}, False, [1, 4]),
        ("prefill-only backend", {"attn.prefill_backend": "reference"}, True, [1, 4]),
        ("state slot capacity", {"state.gdn.slots": 12}, True, [1, 4]),
    ]
    with served(run, *args, ready_s=180) as url:
        baseline = post(url + "/v1/completions", body)["choices"][0]["text"]
        assert baseline
        before = graph_state(run)
        assert before[1:] == (3, [1, 2, 4]), before
        for name, values, recapture, buckets in cases:
            code, dry = client(run, "apply", "--config", json.dumps(values), "--dry-run")
            assert code == 0, (name, dry)
            code, applied = client(run, "apply", "--config", json.dumps(values))
            assert code == 0 and applied["status"] == "committed", (name, applied)
            assert not applied.get("restarted") and not applied.get("restart_engine"), (name, applied)
            after = graph_state(run)
            assert after[0] == before[0] + int(recapture), (name, before, after, applied)
            assert after[1] - before[1] == (len(buckets) if recapture else 0), (name, before, after)
            assert after[2] == buckets, (name, after)
            assert post(url + "/v1/completions", body)["choices"][0]["text"] == baseline, name
            # Ordinary decode after apply must not trigger a lazy recapture.
            assert graph_state(run) == after, name
            print("GRAPH_CONFIG_CASE " + json.dumps({"case": name, "config": values,
                  "plan_generation": [before[0], after[0]], "new_captures": after[1] - before[1],
                  "buckets": after[2], "serving_matches": True}), flush=True)
            before = after
        # The plan's own keys are the plan's: the state pools stay where they are.
        for values in ({"plan.cuda_graph": "auto"}, {"plan.graph_memory_bytes": 1 << 31}):
            code, dry = client(run, "apply", "--config", json.dumps(values), "--dry-run")
            assert code == 0 and dry["processes"]["host-a/worker/0"]["reload"] == ["core.plan"], (values, dry)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("overlap", [False, True])
def test_kernel_apply_and_failed_graph_build_restore_serving(tmp_path: Path, overlap: bool) -> None:
    args, body = deployment(tmp_path, overlap)
    source = tmp_path / "source"
    kernel = source / "engine/kernels/reference/ops.py"
    original = kernel.read_text()
    run = tmp_path / "run"
    with served(run, *args, ready_s=180) as url:
        baseline = post(url + "/v1/completions", body)["choices"][0]["text"]
        assert baseline
        kernel.write_text(original.replace("return F.linear(x, weight, bias)", "return F.linear(x, weight, bias) * 0"))
        code, checked = client(run, "check", "--source", str(source))
        assert code == 0, checked
        code, dry = client(run, "apply", "--source", str(source), "--dry-run")
        assert code == 0 and "core.plan" in json.dumps(dry), dry
        code, applied = client(run, "apply", "--source", str(source))
        assert code == 0 and applied["status"] == "committed", applied
        assert post(url + "/v1/completions", body)["choices"][0]["text"] == ""
        # Invalid buckets must roll back to a usable graph set before serving resumes.
        code, failed = client(run, "apply", "--config", '{"plan.graph_buckets": [0]}')
        assert code != 0 and failed["status"] != "committed", failed
        assert post(url + "/v1/completions", body)["choices"][0]["text"] == ""
        kernel.write_text(original)
        code, restored = client(run, "apply", "--source", str(source))
        assert code == 0 and restored["status"] == "committed", restored
        assert post(url + "/v1/completions", body)["choices"][0]["text"] == baseline
        logs = "\n".join(p.read_text() for p in (run / "host-a/snapshots/.run").rglob("log"))
        assert logs.count('"buckets": [1, 2, 4]') >= 3, logs[-5000:]
    with served(tmp_path / "cold", *args, ready_s=180) as url:
        assert post(url + "/v1/completions", body)["choices"][0]["text"] == baseline


@pytest.mark.timeout(900)
def test_repeated_kernel_changes_leave_the_device_as_it_was(tmp_path: Path) -> None:
    """With the state pools at the default share of the device, every kernel change records the
    graphs again in what is left. A closed set of graphs gives its pool back, so the tenth change
    finds the room the first one found, and a change that fails rolls back to graphs that serve."""
    args, body = deployment(tmp_path, mem_fraction="0.85")
    source = tmp_path / "source"
    kernel = source / "engine/kernels/reference/ops.py"
    original = kernel.read_text()
    run = tmp_path / "run"
    free: list[int] = []
    with served(run, *args, ready_s=300) as url:
        baseline = post(url + "/v1/completions", body)["choices"][0]["text"]
        assert baseline
        for i in range(10):
            kernel.write_text(original + f"\n# change {i}\n")
            code, applied = client(run, "apply", "--source", str(source))
            assert code == 0 and applied["status"] == "committed", (i, applied)
            assert post(url + "/v1/completions", body)["choices"][0]["text"] == baseline, i
            free.append(torch.cuda.mem_get_info()[0])
            if i == 4:  # a kernel that breaks the capture: the old graphs come back
                kernel.write_text(original.replace("return F.linear(x, weight, bias)",
                                                   "return F.linear(x, weight, bias) * float(x.sum().item() * 0 + 1)"))
                code, failed = client(run, "apply", "--source", str(source))
                assert code != 0 and failed["status"] != "committed", failed
                code, status = client(run, "status")
                assert code == 0 and status["recovery_required"] is False, status
                assert post(url + "/v1/completions", body)["choices"][0]["text"] == baseline
        assert graph_state(run)[2] == [1, 2, 4]
    assert max(free[1:]) - min(free[1:]) <= (64 << 20), [f >> 20 for f in free]
