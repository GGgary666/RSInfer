"""unit-runtime.md §8 tests 6–8: code reload, rollback, native extensions, staging."""

from __future__ import annotations

import importlib.machinery
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

from core.common.ids import Identity
from core.common.result import Failed, Ok
from tests.framework.support import PROC, events, spec
from unit.runtime import (
    ESCALATE_ENGINE,
    ESCALATE_PROCESS,
    InstanceState,
    ProcessPlan,
    ReloadPlan,
    UnitRuntime,
    UnitSpec,
)

Snapshot = Callable[[str, dict[str, str]], Path]

CATALOG = '''
from unit.runtime import Registry

class Catalog:
    def __init__(self, ctx, config):
        self.reg = Registry("catalog")
        ctx.provide("cat@1", self.reg)
'''

POLICY = '''
from {pkg} import helper
from tests.framework.support import events

class Policy:
    def __init__(self, ctx, config):
        ctx.provide("pol@1", self)
        ctx.register("cat@1", ("policy", helper.VALUE))
        events.append(("load", ctx.id))

    def value(self):
        return helper.VALUE
'''

USER = '''
class User:
    def __init__(self, ctx, config):
        self.pol = ctx.use("pol@1")

    def ask(self):
        return self.pol.value()
'''

OTHER = '''
class Other:
    def __init__(self, ctx, config):
        ctx.provide("other@1", self)


class Native:
    def __init__(self, ctx, config):
        pass
'''


@dataclass
class World:
    pkg: str
    v1: Path
    v2: Path
    catalog: UnitSpec
    policy: UnitSpec
    user: UnitSpec
    other: UnitSpec


@pytest.fixture
def world(pkg: str, snapshot: Snapshot) -> World:
    files = {
        f"{pkg}/catalog.py": CATALOG,
        f"{pkg}/helper.py": "VALUE = 1\n",
        f"{pkg}/policy.py": POLICY.format(pkg=pkg),
        f"{pkg}/user.py": USER,
        f"{pkg}/other.py": OTHER,
    }
    v1 = snapshot("v1", files)
    v2 = snapshot("v2", {**files, f"{pkg}/helper.py": "VALUE = 2\n"})
    return World(
        pkg, v1, v2,
        catalog=spec(f"{pkg}.catalog", f"{pkg}.catalog:Catalog", provides=["cat@1"]),
        policy=spec(f"{pkg}.policy", f"{pkg}.policy:Policy", provides=["pol@1"], requires=["cat@1"]),
        user=spec(f"{pkg}.user", f"{pkg}.user:User", requires=["pol@1"]),
        other=spec(f"{pkg}.other", f"{pkg}.other:Other", provides=["other@1"]),
    )


def start(w: World, **kw: object) -> UnitRuntime:
    rt = UnitRuntime(PROC, **kw)  # type: ignore[arg-type]
    units = [w.catalog, w.policy, w.user, w.other]
    rt.load_process_plan(ProcessPlan(PROC, Identity("v1"), units, code_root=str(w.v1))).unwrap()
    return rt


def ask(rt: UnitRuntime, w: World) -> int:
    return rt.instance(w.user.id).ask()  # type: ignore[union-attr]


def catalog(rt: UnitRuntime, w: World) -> list[object]:
    return rt.instance(w.catalog.id).reg.items()  # type: ignore[union-attr]


def helper_change(w: World, **kw: object) -> ReloadPlan:
    return ReloadPlan(changed_modules=[f"{w.pkg}.helper"], new_code_root=str(w.v2), **kw)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ImportState:
    modules: dict[str, int]
    sys_path: list[str]
    package_paths: dict[str, list[str]]


def import_state(pkg: str) -> ImportState:
    mods: dict[str, ModuleType] = {n: m for n, m in sys.modules.items() if n == pkg or n.startswith(pkg + ".")}
    return ImportState(
        {n: id(m) for n, m in mods.items()},
        list(sys.path),
        {n: list(m.__path__) for n, m in mods.items() if hasattr(m, "__path__")},
    )


def unit_state(rt: UnitRuntime) -> tuple[dict[str, InstanceState], list[str]]:
    snap = rt.snapshot()
    return snap.states(), sorted(snap.services)


def test_changed_module_reloads_its_importers_from_the_new_snapshot(world: World) -> None:
    rt = start(world)
    assert ask(rt, world) == 1
    other_gen = rt.snapshot().unit(world.other.id).generation

    report = rt.reload(helper_change(world, units_to_reload=[world.policy])).unwrap()

    assert ask(rt, world) == 2
    assert set(report.reimported) == {f"{world.pkg}.helper", f"{world.pkg}.policy"}
    assert report.unloaded == [world.user.id, world.policy.id]
    assert report.unplanned == []
    assert sys.modules[f"{world.pkg}.helper"].__file__.startswith(str(world.v2))  # type: ignore[union-attr]
    assert sys.modules[world.pkg].__path__ == [str(world.v2 / world.pkg)]
    assert str(world.v1) not in sys.path and str(world.v2) in sys.path
    assert rt.snapshot().unit(world.other.id).generation == other_gen
    assert catalog(rt, world) == [("policy", 2)]


def test_units_missing_from_the_plan_are_reloaded_and_reported(world: World) -> None:
    rt = start(world)
    report = rt.reload(helper_change(world)).unwrap()
    assert report.unplanned == [world.policy.id]
    assert [d.code for d in report.diagnostics] == ["unit.unplanned_reload"]
    assert ask(rt, world) == 2


FAULT_POINTS = [
    "before_unload", "unload:{user}", "unload:{policy}", "after_unload",
    "after_purge", "after_switch", "load:{policy}", "load:{user}", "after_load",
]


@pytest.mark.parametrize("point", FAULT_POINTS)
def test_rollback_from_any_point_restores_everything(world: World, point: str) -> None:
    rt = start(world)
    imports_before = import_state(world.pkg)
    units_before = unit_state(rt)
    catalog_before = catalog(rt, world)

    rt.faults.add(point.format(user=world.user.id, policy=world.policy.id))
    r = rt.reload(helper_change(world, units_to_reload=[world.policy]))

    assert isinstance(r, Failed) and r.has_code("unit.rolled_back")
    assert not r.has_code(ESCALATE_PROCESS)
    assert import_state(world.pkg) == imports_before
    assert unit_state(rt) == units_before
    assert catalog(rt, world) == catalog_before
    assert ask(rt, world) == 1
    rt.faults.clear()
    rt.reload(helper_change(world, units_to_reload=[world.policy])).unwrap()
    assert ask(rt, world) == 2


def test_broken_new_code_rolls_back(world: World, snapshot: Snapshot) -> None:
    broken = snapshot("broken", {
        f"{world.pkg}/catalog.py": CATALOG,
        f"{world.pkg}/helper.py": "VALUE = (\n",
        f"{world.pkg}/policy.py": POLICY.format(pkg=world.pkg),
        f"{world.pkg}/user.py": USER,
        f"{world.pkg}/other.py": OTHER,
    })
    rt = start(world)
    before = import_state(world.pkg)
    r = rt.reload(ReloadPlan(changed_modules=[f"{world.pkg}.helper"], units_to_reload=[world.policy],
                             new_code_root=str(broken)))
    assert isinstance(r, Failed) and r.has_code("unit.not_active")
    assert any("SyntaxError" in d.reason for d in r.diagnostics)
    assert import_state(world.pkg) == before
    assert ask(rt, world) == 1


def test_declared_native_extension_change_restarts_process_without_unloading(world: World) -> None:
    native = spec(f"{world.pkg}.native", f"{world.pkg}.other:Native",
                  capabilities=f'native_extensions = ["{world.pkg}._fast"]')
    rt = start(world)
    rt.reload(ReloadPlan(units_to_add=[native])).unwrap()
    gens = {u.unit: u.generation for u in rt.snapshot().units}
    events.clear()

    r = rt.reload(ReloadPlan(changed_modules=[f"{world.pkg}._fast"], units_to_reload=[native]))

    assert isinstance(r, Failed) and r.has_code(ESCALATE_PROCESS)
    assert {u.unit: u.generation for u in rt.snapshot().units} == gens
    assert events == []


def test_undeclared_native_module_restarts_process_and_reports_manifest(world: World) -> None:
    ext = next((n for n, m in sys.modules.items()
                if isinstance(getattr(m.__spec__, "loader", None), importlib.machinery.ExtensionFileLoader)),
               None)
    if ext is None:
        pytest.skip("no extension module loaded")
    rt = start(world)
    r = rt.reload(ReloadPlan(changed_modules=[ext]))
    assert isinstance(r, Failed)
    assert r.has_code(ESCALATE_PROCESS) and r.has_code("manifest.undeclared_native")


def test_framework_change_restarts_engine(world: World) -> None:
    rt = start(world)
    r = rt.reload(ReloadPlan(changed_modules=["unit.runtime.runtime"]))
    assert isinstance(r, Failed) and r.has_code(ESCALATE_ENGINE)


# -- staging (§6.3) ----------------------------------------------------------

KERNEL = '''
class Kernel:
    def __init__(self, ctx, config):
        ctx.register("cat@1", ("kernel", "triton"))
'''


def with_kernel(w: World, snapshot: Snapshot, name: str, kernel: str, helper: str = "VALUE = 1\n") -> Path:
    return snapshot(name, {
        f"{w.pkg}/catalog.py": CATALOG,
        f"{w.pkg}/helper.py": helper,
        f"{w.pkg}/policy.py": POLICY.format(pkg=w.pkg),
        f"{w.pkg}/user.py": USER,
        f"{w.pkg}/other.py": OTHER,
        f"{w.pkg}/kernels/triton_attn.py": kernel,
    })


def kernel_spec(w: World) -> UnitSpec:
    return spec(f"{w.pkg}.kernels.triton_attn", f"{w.pkg}.kernels.triton_attn:Kernel", requires=["cat@1"])


def test_new_unit_is_staged_next_to_old_ones_then_committed(world: World, snapshot: Snapshot) -> None:
    v3 = with_kernel(world, snapshot, "v3", KERNEL)
    rt = start(world)
    gens = {u.unit: u.generation for u in rt.snapshot().units}
    k = kernel_spec(world)

    assert isinstance(rt.stage_new_units([k], code_root=str(v3)), Ok)
    assert rt.snapshot().unit(k.id).staged
    assert ("kernel", "triton") in catalog(rt, world)
    assert {u.unit: u.generation for u in rt.snapshot().units if u.unit != k.id} == gens
    assert sys.path.index(str(world.v1)) < sys.path.index(str(v3))

    report = rt.reload(ReloadPlan(units_to_add=[k], new_code_root=str(v3))).unwrap()
    assert report.unloaded == [] and report.loaded == []
    snap = rt.snapshot()
    assert not snap.unit(k.id).staged and snap.unit(k.id).state is InstanceState.ACTIVE
    assert str(world.v1) not in sys.path and str(v3) in sys.path
    assert sys.modules[world.pkg].__path__ == [str(v3 / world.pkg)]


def test_discarding_staged_units_leaves_no_trace(world: World, snapshot: Snapshot) -> None:
    v3 = with_kernel(world, snapshot, "v3", KERNEL)
    rt = start(world)
    imports_before = import_state(world.pkg)
    units_before = unit_state(rt)
    catalog_before = catalog(rt, world)

    rt.stage_new_units([kernel_spec(world)], code_root=str(v3)).unwrap()
    rt.discard_staged().unwrap()

    assert import_state(world.pkg) == imports_before
    assert unit_state(rt) == units_before
    assert catalog(rt, world) == catalog_before


def test_staging_refuses_units_that_import_changed_modules(world: World, snapshot: Snapshot) -> None:
    kernel = f"from {world.pkg} import helper\n" + KERNEL
    v3 = with_kernel(world, snapshot, "v3", kernel, helper="VALUE = 3\n")
    rt = start(world)
    r = rt.stage_new_units([kernel_spec(world)], code_root=str(v3), changed_modules=[f"{world.pkg}.helper"])
    assert isinstance(r, Failed) and r.has_code("unit.stage_needs_safe_point")
    assert str(v3) not in sys.path


HELPER_KERNEL = '''
from {pkg} import helper

class Kernel:
    def __init__(self, ctx, config):
        self.value = helper.VALUE
        ctx.register("cat@1", ("kernel", "triton"))
'''


def test_staging_refuses_units_that_import_changed_modules_not_loaded_here(world: World, snapshot: Snapshot) -> None:
    """unit-runtime.md §8 test 9: the changed module would be found in the old snapshot first."""
    v3 = with_kernel(world, snapshot, "v3", HELPER_KERNEL.format(pkg=world.pkg), helper="VALUE = 3\n")
    rt = UnitRuntime(PROC)
    rt.load_process_plan(ProcessPlan(PROC, Identity("v1"), [world.catalog, world.other],
                                     code_root=str(world.v1))).unwrap()
    assert f"{world.pkg}.helper" not in sys.modules
    k = kernel_spec(world)
    r = rt.stage_new_units([k], code_root=str(v3), changed_modules=[f"{world.pkg}.helper"])
    assert isinstance(r, Failed) and r.has_code("unit.stage_needs_safe_point")
    assert f"{world.pkg}.helper" not in sys.modules and str(v3) not in sys.path

    rt.reload(ReloadPlan(changed_modules=[f"{world.pkg}.helper"], units_to_add=[k], new_code_root=str(v3))).unwrap()
    assert rt.instance(k.id).value == 3  # type: ignore[union-attr]


def test_a_staged_unit_that_imports_a_changed_module_fails_the_reload(world: World, snapshot: Snapshot) -> None:
    v3 = with_kernel(world, snapshot, "v3", HELPER_KERNEL.format(pkg=world.pkg), helper="VALUE = 3\n")
    rt = start(world)
    k = kernel_spec(world)
    rt.stage_new_units([k], code_root=str(v3)).unwrap()  # the plan forgot that helper changed
    before = unit_state(rt)
    r = rt.reload(ReloadPlan(changed_modules=[f"{world.pkg}.helper"], units_to_add=[k], new_code_root=str(v3)))
    assert isinstance(r, Failed) and r.has_code("unit.staged_imports_changed")
    assert unit_state(rt) == before and ask(rt, world) == 1


def test_modified_modules_cannot_be_staged(world: World) -> None:
    rt = start(world)
    again = spec(f"{world.pkg}.policy2", f"{world.pkg}.policy:Policy", provides=["pol2@1"], requires=["cat@1"])
    r = rt.stage_new_units([again], code_root=str(world.v2))
    assert isinstance(r, Failed) and r.has_code("unit.stage_existing_module")
