"""② Starting the engine from the command line, as an operator or an agent would."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import msgspec
import pytest

from tests.scenarios.harness import AGENT_TUNABLES, ENGINE, PROC_TUNABLES, build_workspace, topology

ENV = {**os.environ, "PYTHONPATH": str(ENGINE)}


def client(control: str, *args: str, exit_code: int = 0) -> Any:
    out = subprocess.run([sys.executable, "-m", "core.control.client", "--control", control, *args],
                         env=ENV, capture_output=True, text=True, timeout=60)
    assert out.returncode == exit_code, out.stdout + out.stderr
    return json.loads(out.stdout)


@pytest.mark.covers(edges=("ctrl",), kinds=("framework", "logic", "state"))
def test_2_serve_starts_the_engine_and_the_client_drives_it(tmp_path: Path) -> None:
    """``python -m core.control serve --spawn-agents``: the startup change runs the static checks,
    computes the identity, and the supervisor starts every process of the topology through node
    agents; ``python -m core.control.client`` then drives it; its exit code says whether the change committed.
    SIGTERM stops everything."""
    workspace = build_workspace(tmp_path / "workspace")
    (tmp_path / "logs").mkdir()
    topo = tmp_path / "topology.json"
    topo.write_bytes(msgspec.json.encode(topology(tmp_path)))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"scn.log_dir": str(tmp_path / "logs")}))
    state = tmp_path / "control"
    log = open(tmp_path / "serve.log", "wb")  # noqa: SIM115 - the child writes to it
    serve = subprocess.Popen(
        [sys.executable, "-m", "core.control", "serve", "--source", str(workspace), "--topology", str(topo),
         "--state-dir", str(state), "--host", "host-a", "--bind", "127.0.0.1", "--config", str(config),
         "--spawn-agents", "--platform", json.dumps({"name": "fake", "versions": {"driver": "0"}}),
         "--tunables", json.dumps({"supervisor.heartbeat_timeout_s": 2.0, "supervisor.stop_timeout_s": 5.0}),
         "--proc-tunables", json.dumps(PROC_TUNABLES), "--agent-tunables", json.dumps(AGENT_TUNABLES)],
        env=ENV, stdout=log, stderr=log, start_new_session=True)
    try:
        deadline = time.monotonic() + 60
        while not (state / "control.json").exists():
            assert serve.poll() is None, (tmp_path / "serve.log").read_text()
            assert time.monotonic() < deadline, "serve did not become ready"
            time.sleep(0.1)
        ready = json.loads((state / "control.json").read_text())
        assert ready["result"]["status"] == "committed", ready
        control = ready["control"]

        status = client(control, "status")
        assert status["identity"] == ready["result"]["new_identity"]
        procs = sorted(p["proc"] for p in status["procs"])
        assert procs == ["host-a/sched/0", "host-a/worker/0", "host-b/worker/1"]
        assert {p["state"] for p in status["procs"]} == {"RUNNING"}
        assert sorted(u["unit"] for u in status["control_units"]) == \
            ["core.api", "core.deps", "core.identity", "core.manifest"]

        policy = workspace / "engine/scn/policy_fcfs/unit.py"
        policy.write_text(policy.read_text().replace("VERSION = 1", "VERSION = 2"))
        dry = client(control, "apply", "--source", str(workspace), "--dry-run")["result"]
        assert dry["status"] == "planned" and dry["new_identity"] != status["identity"]
        applied = client(control, "apply", "--source", str(workspace), "--intent", "a faster policy")["result"]
        assert applied["status"] == "committed" and applied["new_identity"] == dry["new_identity"]
        assert client(control, "status")["identity"] == applied["new_identity"]
        logs = client(control, "logs")
        assert [c["source"] for c in logs["changes"]] == ["startup", "apply", "apply"]

        # the control plane answers, but the change failed: scripts must see it in the exit code
        queue = workspace / "engine/core/change/queue.py"
        queue.write_text(queue.read_text().replace('"""Change queue', '"""The change queue'))
        failed = client(control, "apply", "--source", str(workspace), exit_code=1)["result"]
        assert failed["status"] == "failed" and failed["plan"]["restart_engine"]
        assert client(control, "status")["identity"] == applied["new_identity"]
    finally:
        serve.send_signal(signal.SIGTERM)
        try:
            serve.wait(timeout=30)
        finally:
            if serve.poll() is None:
                os.killpg(serve.pid, signal.SIGKILL)
            log.close()
    assert serve.returncode == 0
