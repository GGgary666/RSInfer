"""A single real ProcessHost whose participant handlers a test drives one message at a time."""
from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Protocol, cast

import msgspec
import pytest

from core.change.protocol import Prepare
from core.change.types import ProcActions, UnitAction
from core.common.ids import Identity
from core.common.result import Result
from core.ipc.rpc import Deferred
from core.supervisor.protocol import ProcSpec
from tests.framework.change.harness import Capture
from tests.framework.support import PROC, spec, write_tree
from unit.runtime import ProcessPlan
from unit.runtime.host import ProcessHost

UNIT = '''
class Unit:
    def __init__(self, ctx, config):
        self.prepares = self.executes = self.aborts = self.commits = 0
        self.resources = 0
        self.value = 1
        ctx.on_action("install", self)
    def prepare(self, args):
        self.prepares += 1
        self.resources += 1
        if args.get("fail") == "prepare":
            raise RuntimeError("prepare failed after acquiring a resource")
    def execute(self, args):
        self.executes += 1
        self.value = 2
        if args.get("fail") == "execute":
            raise RuntimeError("execute failed after changing state")
    def abort(self):
        self.aborts += 1
        self.resources -= 1
        self.value = 1
    def commit(self):
        self.commits += 1
        self.resources -= 1
'''


class ActionState(Protocol):
    prepares: int
    executes: int
    aborts: int
    commits: int
    resources: int
    value: int


class Peer:
    def __init__(self, tmp: Path, pkg: str, monkeypatch: pytest.MonkeyPatch) -> None:
        self.pkg = pkg
        self.root = tmp / "snapshots"
        self.unit = spec(f"{pkg}.main", f"{pkg}.main:Unit")
        code = write_tree(self.root / "v1" / "engine", {f"{pkg}/main.py": UNIT})
        plan = ProcessPlan(PROC, Identity("v1"), [self.unit], code_root=str(code))
        ps = ProcSpec(PROC, Identity("v1"), plan, "unused", "127.0.0.1")
        self.host = ProcessHost(ps)
        monkeypatch.setattr(self.host.rpc, "defer", Capture)
        self.host.runtime.load_process_plan(plan).unwrap()

    @property
    def state(self) -> ActionState:
        return cast(ActionState, self.host.runtime.instance(self.unit.id))

    def send(self, msg: msgspec.Struct) -> Result[msgspec.Struct]:
        env = self.host.registry.wrap(msg, self.host.me)
        result = self.host.handlers[type(msg)](env, msg)
        if isinstance(result, Capture):
            assert result.result is not None, "this test expects immediate safe-point completion"
            return result.result
        assert not isinstance(result, Deferred)
        return result

    def prepare(self, cid: int, fail: str = "") -> Prepare:
        return Prepare(cid, Identity("v1"), ProcActions(
            PROC, actions=[UnitAction(self.unit.id, "install", {"fail": fail})]), 60.0)


@pytest.fixture
def peer(tmp_path: Path, pkg: str, monkeypatch: pytest.MonkeyPatch) -> Generator[Peer]:
    p = Peer(tmp_path, pkg, monkeypatch)
    try:
        yield p
    finally:
        p.host.shutdown()
