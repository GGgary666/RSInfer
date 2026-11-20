"""Stage 2's M2a (stage2-contract.md §1.3): the whole engine on the CPU, started the way an operator
starts it (``deploy/rsinfer.py serve``), with a small random-weight Qwen3.8 in a real checkpoint's
layout and a real HF tokenizer. Text through HTTP, out of the frontend, the scheduler host and a
worker process, equals what HF's model computes, greedy, token by token."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from tests.scenarios.deploy import CLI, ENV, REPO, cli, free_port, post, served
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


def test_a_change_applies_to_an_empty_engine_refused_forced_or_waited_for(model: Path, tokenizer: Path,
                                                                          tmp_path: Path) -> None:
    """doc/specs/empty-apply.md through the whole engine and the operator's client: with a request
    open, ``apply`` is refused and says how to go on; ``--force`` ends the request (its stream says
    why) and applies; on an empty engine ``apply`` just applies; ``--wait-empty`` lets the request
    run to its end, then applies. Serving goes on after each."""
    args = ["--model", str(model), "--tokenizer", str(tokenizer), "--device", "cpu", "--dtype", "f32", "--conv-dtype", "f32",
            "--page-size", "4", "--max-rows", "8", "--max-context", "4096", "--token-budget", "8",
            "--cpu-bytes", str(1 << 28), "--timeout", "120"]
    workspace = tmp_path / "workspace"
    shutil.copytree(REPO / "engine", workspace / "engine", ignore=shutil.ignore_patterns("__pycache__"))
    policy = workspace / "engine" / "policies" / "fcfs" / "unit.py"
    with served(tmp_path / "run", *args) as url:
        control = json.loads((tmp_path / "run" / "control" / "control.json").read_text())["control"]

        def apply(version: int, *flags: str) -> tuple[int, dict[str, Any]]:
            policy.write_text(policy.read_text() + f"\n# version {version}\n")
            out = subprocess.run([sys.executable, "-m", "core.control.client", "--control", control, "apply",
                                  "--source", str(workspace), *flags], env=ENV, capture_output=True, text=True,
                                 timeout=600)
            return out.returncode, json.loads(out.stdout)

        def stream(frames: list[str], max_tokens: int) -> threading.Thread:
            body = {"model": "Qwen3.8-27B", "prompt": "w1 w2 w3", "max_tokens": max_tokens, "temperature": 0.0,
                    "stream": True, "ignore_eos": True}

            def run() -> None:
                req = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=600) as r:
                    for line in r:
                        if line.startswith(b"data: "):
                            frames.append(line[6:].strip().decode())

            t = threading.Thread(target=run, daemon=True)
            t.start()
            deadline = time.monotonic() + 60
            while len(frames) < 3:
                assert time.monotonic() < deadline and t.is_alive(), frames
                time.sleep(0.05)
            return t

        forced_frames: list[str] = []
        long = stream(forced_frames, 4000)
        code, refused = apply(2)
        assert code == 1 and refused["status"] != "committed", refused
        reasons = " ".join(d["reason"] for d in refused["diagnostics"])
        assert "open request" in reasons and "--wait-empty" in reasons and "--force" in reasons, reasons

        code, forced = apply(3, "--force")
        assert code == 0 and forced["status"] == "committed", forced
        long.join(60)
        assert not long.is_alive() and any("apply --force" in f for f in forced_frames), forced_frames[-3:]
        error = next(json.loads(f)["error"] for f in forced_frames if "apply --force" in f)
        assert error["code"] == 503 and error["type"] == "server_error"  # the server's doing: retry later

        code, plain = apply(4)  # nothing in the engine now
        assert code == 0 and plain["status"] == "committed", plain

        waited_frames: list[str] = []
        short = stream(waited_frames, 200)
        code, waited = apply(5, "--wait-empty", "120")
        assert code == 0 and waited["status"] == "committed", waited
        short.join(60)
        last = json.loads(next(f for f in reversed(waited_frames) if f != "[DONE]"))
        assert last["choices"][0]["finish_reason"] == "length" and last["usage"]["completion_tokens"] == 200
        assert waited["identity"]["new"] not in last["engine_identities"]  # it ran to its end before the change
        got = post(url + "/v1/completions", {"model": "Qwen3.8-27B", "prompt": "w1 w2", "max_tokens": 4})
        assert got["choices"][0]["text"] is not None


def test_a_kernel_change_that_cannot_run_rolls_back_and_the_engine_keeps_serving(model: Path, tokenizer: Path,
                                                                                tmp_path: Path) -> None:
    """``core.plan`` is built from the kernel units (``ctx.derived``, no rebuild action), so a kernel
    change reloads it after them, on every platform. A kernel unit that no longer registers a
    definition decode calls fails that reload, at apply rather than at the first request; the reload
    rolls back, nothing is left to recover, and the next change applies."""
    args = ["--model", str(model), "--tokenizer", str(tokenizer), "--device", "cpu", "--dtype", "f32", "--conv-dtype", "f32",
            "--page-size", "4", "--max-rows", "8", "--max-context", "64", "--token-budget", "8", "--cpu-bytes", str(4 << 20),
            "--timeout", "120"]
    workspace = tmp_path / "workspace"
    shutil.copytree(REPO / "engine", workspace / "engine", ignore=shutil.ignore_patterns("__pycache__"))
    ops, unit = (workspace / "engine" / "kernels" / "reference" / name for name in ("ops.py", "unit.py"))
    registrations = unit.read_text()
    body = {"model": "Qwen3.8-27B", "prompt": "w1 w2 w3 w4 w5", "max_tokens": 8, "temperature": 0.0, "ignore_eos": True}
    with served(tmp_path / "run", *args) as url:
        control = json.loads((tmp_path / "run" / "control" / "control.json").read_text())["control"]

        def client(*arguments: str) -> tuple[int, dict[str, Any]]:
            out = subprocess.run([sys.executable, "-m", "core.control.client", "--control", control, *arguments],
                                 env=ENV, capture_output=True, text=True, timeout=600)
            return out.returncode, json.loads(out.stdout)

        def text() -> str:
            return post(url + "/v1/completions", body)["choices"][0]["text"]

        baseline = text()
        assert baseline
        ops.write_text(ops.read_text() + "\n# version 2\n")
        code, applied = client("apply", "--source", str(workspace))
        assert code == 0 and applied["status"] == "committed", applied
        worker = applied["processes"]["host-a/worker/0"]
        assert worker["reload"] == ["core.plan", "kernels.reference"] and "actions" not in worker, worker
        assert text() == baseline

        unit.write_text(registrations.replace('("gdn.packed_decode@1"', '("gdn.packed_decode_gone@1"'))
        assert unit.read_text() != registrations
        code, failed = client("apply", "--source", str(workspace))
        assert code == 1 and failed["status"] == "aborted", failed
        reasons = " ".join(d["reason"] for d in failed["diagnostics"])
        assert "no kernel definition 'gdn.packed_decode@1'" in reasons and "rolled back" in reasons, reasons
        code, status = client("status")
        assert code == 0 and status["recovery_required"] is False, status
        assert text() == baseline

        unit.write_text(registrations)
        ops.write_text(ops.read_text() + "\n# version 3\n")
        code, applied = client("apply", "--source", str(workspace))
        assert code == 0 and applied["status"] == "committed", applied
        assert text() == baseline


def test_a_unit_added_to_the_workspace_loads_at_the_next_apply_and_one_removed_unloads(model: Path, tokenizer: Path,
                                                                                      tmp_path: Path) -> None:
    """``serve`` lists the worker's units in the topology (those of its model's family), and the
    control plane takes a role's listed units as they are. ``control apply`` therefore works the
    list out again from the workspace it applies: a kernel unit added there runs in the worker
    after the apply, and no longer after the apply of the workspace without it."""
    args = ["--model", str(model), "--tokenizer", str(tokenizer), "--device", "cpu", "--dtype", "f32", "--conv-dtype", "f32",
            "--page-size", "4", "--max-rows", "8", "--max-context", "64", "--token-budget", "8", "--cpu-bytes", str(4 << 20),
            "--timeout", "120"]
    run, workspace = tmp_path / "run", tmp_path / "workspace"
    shutil.copytree(REPO / "engine", workspace / "engine", ignore=shutil.ignore_patterns("__pycache__"))
    added = workspace / "engine" / "kernels" / "added"
    body = {"model": "Qwen3.8-27B", "prompt": "w1 w2 w3 w4 w5", "max_tokens": 8, "temperature": 0.0, "ignore_eos": True}

    def apply(*flags: str) -> dict[str, Any]:
        return json.loads(cli("control", "apply", "--source", str(workspace), "--run", str(run), *flags))

    def running() -> set[str]:
        procs = json.loads(cli("control", "introspect", "--run", str(run)))["procs"]
        return {u["unit"] for p in procs if p["proc"]["role"] == "worker" for u in p["units"] if u["state"] == "ACTIVE"}

    def worker_role() -> dict[str, Any]:
        roles = json.loads((run / "topology.json").read_text())["roles"]
        return next(r for r in roles if r["name"] == "worker")

    def listed() -> list[str]:
        return list(worker_role()["units"])

    with served(run, *args) as url:
        baseline = post(url + "/v1/completions", body)["choices"][0]["text"]
        assert baseline and "kernels.reference" in running() and "kernels.added" not in running()
        # An explicit --topology change first (worker env -> one worker restart): topology.json is
        # synced, and a subsequent hot-add builds on the running topology instead of reverting env
        # and restarting the worker a second time.
        custom = json.loads((run / "topology.json").read_text())
        next(r for r in custom["roles"] if r["name"] == "worker")["env"] = {"RSINFER_TEST_TOPO_ENV": "kept"}
        topo_file = tmp_path / "custom-topology.json"
        topo_file.write_text(json.dumps(custom))
        restarted = json.loads(cli("control", "apply", "--topology", str(topo_file), "--run", str(run)))
        assert restarted["status"] == "committed" and restarted["restarted"] == ["host-a/worker/0"], restarted
        assert worker_role()["env"] == {"RSINFER_TEST_TOPO_ENV": "kept"}

        # Direct core.control.client --topology (bypassing rsinfer.py) is also picked up from the
        # control plane's committed change log rather than overwritten by a stale topology.json.
        control = json.loads((run / "control" / "control.json").read_text())["control"]
        next(r for r in custom["roles"] if r["name"] == "worker")["env"] = {"RSINFER_TEST_TOPO_ENV": "kept-direct"}
        topo_file.write_text(json.dumps(custom))
        direct = subprocess.run([sys.executable, "-m", "core.control.client", "--control", control, "apply",
                                 "--topology", str(topo_file)], env=ENV, capture_output=True, text=True, timeout=300)
        assert direct.returncode == 0 and json.loads(direct.stdout)["restarted"] == ["host-a/worker/0"], direct.stdout
        assert worker_role()["env"] == {"RSINFER_TEST_TOPO_ENV": "kept"}  # local file not touched by direct client
        assert post(url + "/v1/completions", body)["choices"][0]["text"] == baseline

        added.mkdir()
        (added / "__init__.py").write_text("")
        (added / "unit.toml").write_text('[unit]\nid = "kernels.added"\nkind = "logic"\nprocess = ["worker"]\n'
                                         'entry = "kernels.added.unit:Unit"\n\n[provides]\nservices = ["kernels.added@1"]\n')
        (added / "unit.py").write_text("import os\nfrom typing import Any\n\n\nclass Unit:\n"
                                       "    def __init__(self, ctx: Any, config: Any) -> None:\n"
                                       '        assert os.environ.get("RSINFER_TEST_TOPO_ENV") == "kept-direct"\n'
                                       '        ctx.provide("kernels.added@1", self)\n')
        assert apply("--dry-run")["processes"]["host-a/worker/0"] == {"stage": ["kernels.added"]}
        assert "kernels.added" not in listed()  # a dry run leaves the run directory's topology
        applied = apply()
        assert applied["status"] == "committed" and "restarted" not in applied, applied
        assert applied["processes"]["host-a/worker/0"] == {"stage": ["kernels.added"]}
        assert "kernels.added" in running() and "kernels.added" in listed()
        assert worker_role()["env"] == {"RSINFER_TEST_TOPO_ENV": "kept-direct"}
        assert "models.ds1017" not in running()  # still the units of this model's family alone
        assert post(url + "/v1/completions", body)["choices"][0]["text"] == baseline

        shutil.rmtree(added)
        applied = apply()
        assert applied["status"] == "committed" and "restarted" not in applied, applied
        assert applied["processes"]["host-a/worker/0"]["remove"] == ["kernels.added"]
        assert "kernels.added" not in running() and "kernels.added" not in listed()
        assert worker_role()["env"] == {"RSINFER_TEST_TOPO_ENV": "kept-direct"}
        assert post(url + "/v1/completions", body)["choices"][0]["text"] == baseline


def test_the_cli_runs_an_engine_in_the_background_puts_it_to_sleep_and_stops_it(model: Path, tokenizer: Path,
                                                                              tmp_path: Path) -> None:
    """``rsinfer serve --detach`` returns once the engine is up. ``status``, ``control`` and the
    rest find it through its run directory. ``sleep`` gives its memory back and requests are refused;
    ``wake`` loads the weights again and serves what it served before. ``stop`` leaves no process of it."""
    run, port = tmp_path / "run", free_port()
    serve = ["serve", str(model), "--tokenizer", str(tokenizer), "--device", "cpu", "--dtype", "f32", "--conv-dtype", "f32",
             "--page-size", "4", "--max-rows", "8", "--max-context", "64", "--token-budget", "8",
             "--cpu-bytes", str(4 << 20), "--timeout", "120", "--port", str(port), "--run", str(run)]
    body = {"model": "Qwen3.8-27B", "prompt": "w1 w2 w3 w4 w5", "max_tokens": 8, "temperature": 0.0, "ignore_eos": True}

    def status() -> tuple[int, dict[str, Any]]:
        out = subprocess.run([sys.executable, str(CLI), "status", "--run", str(run)], env=ENV, capture_output=True,
                             text=True, timeout=60)
        return out.returncode, json.loads(out.stdout)

    try:
        cli(*serve, "--detach", timeout=180)
        code, now = status()
        assert code == 0 and now["running"] and now["health"] == "ok", now
        assert now["url"] == f"http://127.0.0.1:{port}" and set(now["procs"].values()) == {"RUNNING"}, now
        again = subprocess.run([sys.executable, str(CLI), *serve, "--detach"], env=ENV, capture_output=True, text=True,
                               timeout=60)
        assert again.returncode != 0 and "already runs" in again.stderr, again.stderr  # one engine per run directory
        url = now["url"]
        baseline = post(url + "/v1/completions", body)["choices"][0]["text"]
        assert baseline

        assert json.loads(cli("sleep", "--run", str(run)))["status"] == "committed"
        with pytest.raises(AssertionError, match="asleep"):
            post(url + "/v1/completions", body)
        assert json.loads(cli("wake", "--run", str(run)))["status"] == "committed"
        assert post(url + "/v1/completions", body)["choices"][0]["text"] == baseline
        # The client's options, with the engine's after them.
        checked = json.loads(cli("control", "check", "--source", str(REPO), "--run", str(run)))
        assert checked["diagnostics"] == [], checked
        assert cli("complete", "w1 w2 w3 w4 w5", "--run", str(run), "--max-tokens", "8")
    finally:
        cli("stop", "--run", str(run), timeout=120)
    code, now = status()
    assert code == 1 and not now["running"] and now["processes"] == 0, now
    assert "no engine runs" in subprocess.run([sys.executable, str(CLI), "stop", "--run", str(run)], env=ENV,
                                              capture_output=True, text=True, timeout=60).stderr


def test_stop_leaves_an_engine_run_from_inside_the_run_directory_alone(model: Path, tokenizer: Path,
                                                                      tmp_path: Path) -> None:
    """``stop`` took every engine process whose arguments named a path under the run directory, so
    also those of an engine run from a directory inside it (Codex's review of #54). And ``--host``
    is the frontend's alone: with ``0.0.0.0`` the control plane still listens on this machine only,
    and the URL a client is given is the loopback one."""
    from deploy.rsinfer import engine_processes

    outer, inner = tmp_path / "run", tmp_path / "run" / "nested"
    ports = {outer: free_port(), inner: free_port()}
    body = {"model": "Qwen3.8-27B", "prompt": "w1 w2 w3", "max_tokens": 4, "temperature": 0.0, "ignore_eos": True}

    def serve(run: Path, *more: str) -> None:
        cli("serve", str(model), "--tokenizer", str(tokenizer), "--device", "cpu", "--dtype", "f32", "--conv-dtype",
            "f32", "--page-size", "4", "--max-rows", "8", "--max-context", "64", "--token-budget", "8", "--cpu-bytes",
            str(4 << 20), "--timeout", "120", "--port", str(ports[run]), "--run", str(run), "--detach", *more,
            timeout=180)

    try:
        serve(outer)
        serve(inner, "--host", "0.0.0.0")
        (_, mine), (_, theirs) = engine_processes(outer), engine_processes(inner)
        assert len(mine) == len(theirs) == 5 and not set(mine) & set(theirs), (mine, theirs)  # control, agent, 3 hosts
        record = json.loads((inner / "engine.json").read_text())
        assert record["url"] == f"http://127.0.0.1:{ports[inner]}", record
        assert record["control"].startswith("tcp://127.0.0.1:"), record
        assert post(record["url"] + "/v1/completions", body)["choices"][0]["text"]

        cli("stop", "--run", str(outer), timeout=120)
        assert engine_processes(outer) == (None, [])
        assert engine_processes(inner)[1] == theirs
        assert post(record["url"] + "/v1/completions", body)["choices"][0]["text"]
    finally:
        for run in (outer, inner):
            cli("stop", "--run", str(run), timeout=120)


def test_serve_refuses_a_run_directory_whose_last_engine_left_processes(monkeypatch: pytest.MonkeyPatch,
                                                                         tmp_path: Path) -> None:
    """Its control plane is gone, a worker it started is not: that worker keeps its GPU memory, and
    another engine would start beside it."""
    import deploy.rsinfer as rsinfer

    monkeypatch.setattr(rsinfer, "engine_processes", lambda run: (None, [4242]))
    with pytest.raises(SystemExit, match=r"processes \[4242\] of an earlier engine .* rsinfer stop"):
        rsinfer.cmd_serve(rsinfer.parser().parse_args(["serve", str(tmp_path), "--run", str(tmp_path / "run")]))
    assert [rsinfer.reachable(h) for h in ("0.0.0.0", "::", "127.0.0.1", "fe80::1")] == \
        ["127.0.0.1", "[::1]", "127.0.0.1", "[fe80::1]"]


def test_stop_without_a_control_plane_asks_every_process_left_to_stop(monkeypatch: pytest.MonkeyPatch,
                                                                     tmp_path: Path) -> None:
    """The control plane is gone and so is the node agent: nothing stops the process hosts for one
    another. Before, only the first one got SIGTERM; the others stayed until ``--timeout`` passed,
    were killed, and ``stop`` failed (Gemini's review of #54)."""
    import deploy.rsinfer as rsinfer

    left = [subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]) for _ in range(3)]
    try:
        monkeypatch.setattr(rsinfer, "engine_processes", lambda run: (None, [p.pid for p in left if p.poll() is None]))
        t0 = time.monotonic()
        assert rsinfer.cmd_stop(rsinfer.parser().parse_args(["stop", "--run", str(tmp_path), "--timeout", "20"])) == 0
        assert time.monotonic() - t0 < 10
        assert [p.returncode for p in left] == [-15, -15, -15]  # each one by SIGTERM, none killed
    finally:
        for p in left:
            p.kill()
            p.wait()


def test_a_change_s_execution_gets_as_long_as_start_up_and_not_less_than_the_engine_s_limit(
        model: Path, tokenizer: Path, tmp_path: Path) -> None:
    """A wake reads the weights again while it executes, as start-up does before the check-in: with
    ``--timeout`` above the engine's limit (3600 s), start-up had longer than the wake after it
    (Gemini's second review of #55)."""
    import deploy.rsinfer as rsinfer

    def limits(timeout: str) -> tuple[float, float]:
        args = rsinfer.parser().parse_args(["serve", str(model), "--tokenizer", str(tokenizer), "--device", "cpu",
                                        "--run", str(tmp_path / timeout), "--timeout", timeout])
        tunables = rsinfer.plan(args)[2]
        return tunables["supervisor.checkin_timeout_s"], tunables["change.execute_timeout_s"]

    assert limits("7200") == (7200.0, 7200.0)
    assert limits("120") == (120.0, 3600.0)


def test_the_worker_gpu_follows_the_callers_cuda_visible_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """The node agent sets an engine process's ``CUDA_VISIBLE_DEVICES`` from the topology alone, so
    the topology has to place the worker on the GPU the caller was given: before, every engine of a
    machine landed on GPU 0."""
    from deploy.rsinfer import worker_gpu

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert worker_gpu(None) == 0
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    assert (worker_gpu(None), worker_gpu(1)) == (3, 1)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-5f0e")
    with pytest.raises(SystemExit, match="--gpu"):
        worker_gpu(None)
