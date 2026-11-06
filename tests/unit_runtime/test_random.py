"""Randomized checks of unit-runtime.md §8 tests 2, 3 and 4 on random dependency graphs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from core.common.ids import Identity
from tests.support import PROC, events, spec, write_tree
from unit.runtime import InstanceState, ProcessPlan, ReloadPlan, UnitRuntime, UnitSpec

PKG = "toyrand"

RANDOM_UNIT = '''
from tests.support import events


class Service:
    pass


class Catalog:
    def __init__(self, ctx, config):
        from unit.runtime import Registry
        self.reg = Registry("random-catalog")
        ctx.provide("cat@1", self.reg)


class Unit:
    def __init__(self, ctx, config):
        for name in ctx.manifest.requires.services:
            ctx.use(name)
        for name in ctx.manifest.provides.services:
            ctx.provide(name, Service())
        key = ctx.id + ".nreg"
        for i in range(config[key] if key in config else 0):
            ctx.register("cat@1", (ctx.id, i))
        ctx.effect(lambda: (lambda: events.append(("unload", ctx.id))), "log")
        events.append(("load", ctx.id))
'''

Graph = list[tuple[frozenset[int], int]]


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_tree(tmp_path_factory.mktemp("rand") / "engine", {f"{PKG}/units.py": RANDOM_UNIT})


@st.composite
def graphs(draw: st.DrawFn) -> Graph:
    n = draw(st.integers(1, 7))
    out: Graph = []
    for i in range(n):
        deps = draw(st.frozensets(st.integers(0, i - 1), max_size=3)) if i else frozenset[int]()
        out.append((deps, draw(st.integers(0, 3))))
    return out


def uid(i: int) -> str:
    return f"{PKG}.u{i}"


def build(graph: Graph) -> tuple[list[UnitSpec], UnitSpec]:
    cat = spec(f"{PKG}.cat", f"{PKG}.units:Catalog", provides=["cat@1"])
    units = [
        spec(uid(i), f"{PKG}.units:Unit", provides=[f"s{i}@1"],
             requires=["cat@1", *(f"s{d}@1" for d in sorted(deps))],
             config={f"{uid(i)}.nreg": {"type": "int", "default": nreg}})
        for i, (deps, nreg) in enumerate(graph)
    ]
    return units, cat


def dependents(graph: Graph, k: int) -> set[int]:
    out = {k}
    changed = True
    while changed:
        changed = False
        for i, (deps, _) in enumerate(graph):
            if i not in out and deps & out:
                out.add(i)
                changed = True
    return out


def start(root: Path, graph: Graph) -> tuple[UnitRuntime, list[UnitSpec], UnitSpec]:
    units, cat = build(graph)
    rt = UnitRuntime(PROC)
    rt.load_process_plan(ProcessPlan(PROC, Identity("x"), [cat, *units], code_root=str(root))).unwrap()
    assert all(s is InstanceState.ACTIVE for s in rt.snapshot().states().values())
    return rt, units, cat


def catalog_items(rt: UnitRuntime) -> Counter[object]:
    cat = rt.instance(f"{PKG}.cat")
    return Counter(cat.reg.items())  # type: ignore[union-attr]


def assert_users_unloaded_first(graph: Graph, unloaded: Sequence[str]) -> None:
    pos = {u: i for i, u in enumerate(unloaded)}
    for i, (deps, _) in enumerate(graph):
        for d in deps:
            if uid(i) in pos and uid(d) in pos:
                assert pos[uid(i)] < pos[uid(d)], f"{uid(i)} uses {uid(d)} but was unloaded after it"


settings.register_profile("ci", deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
settings.load_profile("ci")


@settings(max_examples=60)
@given(graph=graphs(), data=st.data())
def test_reload_propagates_exactly_to_transitive_users(root: Path, graph: Graph, data: st.DataObject) -> None:
    events.clear()
    rt, units, _ = start(root, graph)
    k = data.draw(st.integers(0, len(graph) - 1))
    before_gen = {u.unit: u.generation for u in rt.snapshot().units}
    before_items = catalog_items(rt)
    events.clear()

    report = rt.reload(ReloadPlan(units_to_reload=[units[k]])).unwrap()

    expected = {uid(i) for i in dependents(graph, k)}
    assert set(report.unloaded) == expected
    assert set(report.loaded) == expected
    assert_users_unloaded_first(graph, report.unloaded)
    unload_events = [u for kind, u in events if kind == "unload"]
    assert unload_events == report.unloaded
    after_gen = {u.unit: u.generation for u in rt.snapshot().units}
    for unit, g in before_gen.items():
        assert after_gen[unit] == g + (1 if unit in expected else 0)
    assert catalog_items(rt) == before_items
    rt.unload_all().unwrap()


@settings(max_examples=60)
@given(graph=graphs())
def test_removing_units_restores_registries(root: Path, graph: Graph) -> None:
    events.clear()
    rt, units, _ = start(root, graph)
    assert sum(catalog_items(rt).values()) == sum(n for _, n in graph)

    report = rt.reload(ReloadPlan(units_to_remove=[u.id for u in units])).unwrap()

    assert_users_unloaded_first(graph, report.unloaded)
    assert catalog_items(rt) == Counter()
    snap = rt.snapshot()
    assert [u.unit for u in snap.units] == [f"{PKG}.cat"]
    assert set(snap.services) == {"cat@1"}
    assert not [d for d in snap.diagnostics if d.code == "unit.leak"]
    rt.unload_all().unwrap()


@settings(max_examples=40)
@given(graph=graphs())
def test_unload_all_orders_users_first(root: Path, graph: Graph) -> None:
    events.clear()
    rt, _, _ = start(root, graph)
    events.clear()
    rt.unload_all().unwrap()
    unloaded = [u for kind, u in events if kind == "unload"]
    assert set(unloaded) == {uid(i) for i in range(len(graph))}
    assert_users_unloaded_first(graph, unloaded)
