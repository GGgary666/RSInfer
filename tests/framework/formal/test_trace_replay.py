"""Replay TLC-generated events through real Participant handlers.

Witnesses must match observable states. Mutant traces must diverge while the
implementation continues to satisfy the independently stated safety oracles.
"""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from typing import Any

import msgspec
import pytest

from core.change.protocol import Abort, Commit, Execute, Pause, Prepare, Resume
from core.change.types import ProcActions, UnitAction
from core.common.ids import Identity
from core.ipc.rpc import Deferred
from core.supervisor.protocol import ProcSpec
from tests.framework.change.harness import Capture
from tests.framework.support import PROC, spec, write_tree
from unit.runtime import ProcessPlan
from unit.runtime.host import ProcessHost

ROOT = Path(__file__).resolve().parent
RUNS = Path(os.environ.get("FORMAL_RUNS", ROOT / ".runs"))
UNIT = '''
class Unit:
    def __init__(self, ctx, config):
        self.value = self.before = 0
        self.execute_fails = self.abort_fails = False
        self.execs = {}
        self.aborts = {}
        ctx.on_action("install", self)
    def prepare(self, args):
        self.before = self.value
        self.tx = args["value"]
    def execute(self, args):
        self.execs[self.tx] = self.execs.get(self.tx, 0) + 1
        self.value = args["value"]
        if self.execute_fails:
            raise RuntimeError("injected execute failure")
    def abort(self):
        self.aborts[self.tx] = self.aborts.get(self.tx, 0) + 1
        if self.abort_fails:
            raise RuntimeError("injected abort failure")
        self.value = self.before
    def commit(self):
        pass
'''

CASES = ["witness-NeverCommitted", "witness-NeverExecutedAbort", "witness-NeverTwoCommits",
         "witness-NeverCrossAbort", "witness-NeverUndoFailure",
         "mutant-ForgetAbort", "mutant-RearmUndone", "mutant-ResumeFailedUndo"]


@pytest.mark.parametrize("case", CASES)
def test_model_trace_against_real_participant(case: str, tmp_path: Path, pkg: str,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    folder = RUNS / case
    assert (folder / "result.json").is_file(), "run tests/framework/formal/run.py first; missing traces are not skips"
    info = json.loads((folder / "result.json").read_text())
    assert info["outcome"] == "counterexample" and info["module"] == "Participant"
    assert info["model_sha256"] == hashlib.sha256((ROOT / "models" / "Participant.tla").read_bytes()).hexdigest(), (
        "model changed; regenerate traces")
    assert info["trace_sha256"] == hashlib.sha256((folder / "trace.json").read_bytes()).hexdigest()
    states = json.loads((folder / "trace.json").read_text())["states"]
    assert len(states) > 1
    unit = spec(f"{pkg}.formal_unit", f"{pkg}.formal_unit:Unit")
    root = write_tree(tmp_path / "code", {f"{pkg}/formal_unit.py": UNIT})
    plan = ProcessPlan(PROC, Identity("v0"), [unit], code_root=str(root))
    host = ProcessHost(ProcSpec(PROC, Identity("v0"), plan, "unused", "127.0.0.1"))
    monkeypatch.setattr(host.rpc, "defer", Capture)
    divergences = []
    try:
        host.runtime.load_process_plan(plan).unwrap()
        instance: Any = host.runtime.instance(unit.id)
        instance.execute_fails = info["constants"]["ExecuteFails"] == "TRUE"
        instance.abort_fails = info["constants"]["AbortFails"] == "TRUE"
        for index, state in enumerate(states):
            event = state["event"]
            assert event is not None, "unsupported or missing TLC event encoding"
            kind, tx = event["op"], event["tx"]
            if kind != "Init":
                messages: dict[str, msgspec.Struct] = {
                    "Prepare": Prepare(tx, Identity(f"v{tx}"), ProcActions(
                        PROC, actions=[UnitAction(unit.id, "install", {"value": tx})]), 60.0),
                    "Pause": Pause(tx, None), "Execute": Execute(tx), "Commit": Commit(tx),
                    "Resume": Resume(tx, Identity(f"v{tx}")), "Abort": Abort(tx),
                }
                message = messages[kind]
                reply = host.handlers[type(message)](host.registry.wrap(message, host.me), message)
                if isinstance(reply, Capture):
                    assert reply.result is not None
                else:
                    assert not isinstance(reply, Deferred)
            c = host.participant.change
            actual = {"current": c.id if c else 0, "phase": c.phase if c else "idle",
                      "value": instance.value, "identity": int(host.identity[1:]), "paused": host.paused,
                      "undoFailed": bool(c and c.undo_diags)}
            # These oracles use runtime effects, not just returned identity labels.
            assert not actual["undoFailed"] or host.paused
            assert host.paused or actual["value"] == actual["identity"]
            assert c is None or c.id not in host.participant.finished
            assert all(n <= 1 for n in instance.execs.values())
            assert all(n <= 1 for n in instance.aborts.values())
            expected = {key: state["values"][key] for key in actual}
            if actual != expected:
                divergences.append((index, kind, expected, actual))
        if case.startswith("mutant-"):
            assert divergences, "real code followed the known unsafe model trace"
        else:
            assert not divergences, divergences
    finally:
        host.shutdown()
