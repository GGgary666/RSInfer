"""context.md §6 tests 1, 2, 4 and the graph half of 3; leases and flags."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from core.common.ids import Identity
from core.common.manifest import parse_manifest
from core.common.result import Failed
from core.context import ConfigStore, UndeclaredConfigKey
from platforms.base import StaleGraph
from platforms.fake import FakePlatform
from tests.support import PROC, manifest_toml, spec
from unit.runtime import InstanceState, ProcessPlan, UnitRuntime, UnitSpec

Snapshot = Callable[[str, dict[str, str]], Path]

UNITS = '''
class Owner:
    """Owns the keys; accepts chunk size in place."""
    def __init__(self, ctx, config):
        self.config = config
        self.seen = []
        self.depth = config["{pkg}.depth"]          # startup read
        ctx.on_config("{pkg}.chunk", self.seen.append)

    def chunk(self):
        return self.config["{pkg}.chunk"]           # runtime read, every step


class Reader:
    """Reads the owner's chunk key every step, no hook."""
    def __init__(self, ctx, config):
        self.config = config

    def chunk(self):
        return self.config["{pkg}.chunk"]


class Sneaky:
    def __init__(self, ctx, config):
        config["{pkg}.depth"]


class Plan:
    def __init__(self, ctx, config):
        self.ctx = ctx
        self.config = config
        self.stream = ctx.lease_stream("alt")
        self.scratch = ctx.lease_buffer("scratch", lambda: bytearray(16))

    def capture(self):
        p = self.ctx.platform
        with self.ctx.plan_scope("decode/bs8"), p.graph().capture() as g:
            p.launch("attn", scalars={{"max_bs": self.config["{pkg}.max_bs"]}})
            self.config["{pkg}.max_bs"]
        return g
'''


def owner_spec(pkg: str) -> UnitSpec:
    return spec(f"{pkg}.owner", f"{pkg}.units:Owner", capabilities="accept_config = true", config={
        f"{pkg}.chunk": {"type": "int", "default": 8192, "binding": "live"},
        f"{pkg}.depth": {"type": "int", "default": 2, "binding": "startup"},
        f"{pkg}.max_bs": {"type": "int", "default": 128, "binding": "plan"},
    })


@pytest.fixture
def root(pkg: str, snapshot: Snapshot) -> Path:
    return snapshot("v1", {f"{pkg}/units.py": UNITS.format(pkg=pkg)})


def start(root: Path, units: list[UnitSpec], **kw: object) -> UnitRuntime:
    rt = UnitRuntime(PROC, **kw)  # type: ignore[arg-type]
    rt.load_process_plan(ProcessPlan(PROC, Identity("x"), units, code_root=str(root))).unwrap()
    return rt


def test_reading_a_key_outside_the_manifest_fails(pkg: str, root: Path) -> None:
    sneaky = spec(f"{pkg}.sneaky", f"{pkg}.units:Sneaky")
    rt = UnitRuntime(PROC)
    r = rt.load_process_plan(ProcessPlan(PROC, Identity("x"), [owner_spec(pkg), sneaky], code_root=str(root)))
    assert isinstance(r, Failed)
    assert UndeclaredConfigKey.__name__ in (rt.snapshot().unit(sneaky.id).error or "")
    with pytest.raises(UndeclaredConfigKey):
        rt.config.read("no.such.key", rt.snapshot().unit(f"{pkg}.owner").instance)  # type: ignore[arg-type]


def test_wrong_types_are_rejected_not_defaulted(pkg: str, root: Path) -> None:
    rt = start(root, [owner_spec(pkg)])
    r = rt.config.set_now({f"{pkg}.chunk": "4096"})
    assert isinstance(r, Failed) and r.has_code("config.type")
    assert isinstance(rt.config.set_now({f"{pkg}.chunk": True}), Failed)
    bad = manifest_toml("x.y", "x.y:Z", config={"x.k": {"type": "int", "default": 1.5}})
    assert isinstance(parse_manifest(bad), Failed)


def test_live_key_takes_effect_at_commit_without_reloading(pkg: str, root: Path) -> None:
    reader = spec(f"{pkg}.reader", f"{pkg}.units:Reader", reads=[f"{pkg}.chunk"])
    rt = start(root, [owner_spec(pkg), reader])
    owner, rd = rt.instance(f"{pkg}.owner"), rt.instance(reader.id)
    assert owner.chunk() == rd.chunk() == 8192  # type: ignore[union-attr]
    gens = {u.unit: u.generation for u in rt.snapshot().units}

    assert rt.config.stage({f"{pkg}.chunk": 4096}).unwrap() == {f"{pkg}.chunk": 4096}
    assert owner.chunk() == 8192  # type: ignore[union-attr]  # staged, not yet at the epoch boundary
    commit = rt.config.commit()

    assert owner.seen == [4096]  # type: ignore[union-attr]
    assert owner.chunk() == rd.chunk() == 4096  # type: ignore[union-attr]
    assert [inst.unit for _, inst in commit.hooks_called] == [f"{pkg}.owner"]
    assert {u.unit: u.generation for u in rt.snapshot().units} == gens
    readers = {i.unit for i in rt.config.readers_of(f"{pkg}.chunk")}
    hooked = {i.unit for i in rt.config.hooked(f"{pkg}.chunk")}
    assert readers - hooked == {reader.id}  # what core.deps reloads if the read was not per step
    phases = {(r.key, r.reader.unit, r.phase) for r in rt.config.reads().runtime}
    assert (f"{pkg}.chunk", reader.id, "runtime") in phases


def test_startup_reads_are_recorded_against_the_instance(pkg: str, root: Path) -> None:
    rt = start(root, [owner_spec(pkg)])
    startup = rt.config.reads().startup
    assert [(r.key, r.reader.unit) for r in startup] == [(f"{pkg}.depth", f"{pkg}.owner")]
    assert {i.unit for i in rt.config.readers_of(f"{pkg}.depth")} == {f"{pkg}.owner"}


def test_plan_key_is_a_by_value_graph_dependency(pkg: str, root: Path) -> None:
    plan = spec(f"{pkg}.plan", f"{pkg}.units:Plan", reads=[f"{pkg}.max_bs"],
                capabilities="captures_graph = true")
    p = FakePlatform()
    rt = start(root, [owner_spec(pkg), plan], platform=p)
    g = rt.instance(plan.id).capture()  # type: ignore[union-attr]
    g.replay()
    plan_reads = rt.config.reads().plan_build["decode/bs8"]
    assert {r.key for r in plan_reads} == {f"{pkg}.max_bs"}

    rt.config.set_now({f"{pkg}.max_bs": 256}).unwrap()
    with pytest.raises(StaleGraph, match="max_bs"):
        g.replay()


def test_leases_are_returned_on_unload(pkg: str, root: Path) -> None:
    plan = spec(f"{pkg}.plan", f"{pkg}.units:Plan", reads=[f"{pkg}.max_bs"],
                capabilities="captures_graph = true")
    rt = start(root, [owner_spec(pkg), plan], platform=FakePlatform())
    inst = rt.snapshot().unit(plan.id).instance
    assert inst is not None
    assert rt.resources.held_by(inst) == ["buffer:scratch", "stream:alt@0"]
    rt.unload_all().unwrap()
    assert rt.resources.held_by(inst) == []
    assert not [d for d in rt.diagnostics if d.code == "unit.leak"]


def test_one_owner_per_key(pkg: str, root: Path) -> None:
    twin = spec(f"{pkg}.twin", f"{pkg}.units:Reader",
                config={f"{pkg}.chunk": {"type": "int", "default": 1}})
    rt = UnitRuntime(PROC)
    r = rt.load_process_plan(ProcessPlan(PROC, Identity("x"), [owner_spec(pkg), twin], code_root=str(root)))
    assert isinstance(r, Failed) and r.has_code("config.owner_conflict")


def test_reading_a_key_nobody_owns_is_rejected(pkg: str, root: Path) -> None:
    reader = spec(f"{pkg}.reader", f"{pkg}.units:Reader", reads=[f"{pkg}.chunk"])
    rt = UnitRuntime(PROC)
    r = rt.load_process_plan(ProcessPlan(PROC, Identity("x"), [reader], code_root=str(root)))
    assert isinstance(r, Failed) and r.has_code("config.undeclared")


def test_hook_requires_capability(pkg: str, root: Path) -> None:
    no_cap = spec(f"{pkg}.owner", f"{pkg}.units:Owner", config={
        f"{pkg}.chunk": {"type": "int", "default": 1},
        f"{pkg}.depth": {"type": "int", "default": 2, "binding": "startup"},
    })
    rt = UnitRuntime(PROC)
    rt.load_process_plan(ProcessPlan(PROC, Identity("x"), [no_cap], code_root=str(root)))
    info = rt.snapshot().unit(no_cap.id)
    assert info.state is InstanceState.FAILED and "accept_config" in (info.error or "")


def test_store_alone_rejects_unknown_keys() -> None:
    store = ConfigStore()
    r = store.stage({"nope": 1})
    assert isinstance(r, Failed) and r.has_code("config.undeclared")
