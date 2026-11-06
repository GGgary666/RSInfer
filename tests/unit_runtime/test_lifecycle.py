"""unit-runtime.md §8 tests 1, 3, 5 and the registration rules of §4–§5."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from core.common.ids import Identity
from core.common.result import Failed, Ok
from tests.support import GENERIC_UNIT, PROC, events, spec
from unit.runtime import (
    InactiveContextError,
    InstanceState,
    ProcessPlan,
    ReloadPlan,
    StaleServiceError,
    UndeclaredServiceError,
    UnitRuntime,
    UnitSpec,
)

Snapshot = Callable[[str, dict[str, str]], Path]


def plan(units: list[UnitSpec], root: Path | None = None) -> ProcessPlan:
    return ProcessPlan(PROC, Identity("id0"), units, code_root=str(root) if root else None)


@pytest.fixture
def generic(pkg: str, snapshot: Snapshot) -> Callable[..., UnitSpec]:
    """Specs for the generic toy unit, all in one snapshot."""
    root = snapshot("v1", {f"{pkg}/generic.py": GENERIC_UNIT})

    def make(uid: str, **kw: object) -> UnitSpec:
        return spec(f"{pkg}.{uid}", f"{pkg}.generic:Unit", **kw)  # type: ignore[arg-type]

    make.root = root  # type: ignore[attr-defined]
    return make


def test_pending_until_dependency_appears(generic: Callable[..., UnitSpec]) -> None:
    rt = UnitRuntime(PROC)
    user = generic("user", requires=["svc.a@1"])
    r = rt.load_process_plan(plan([user], generic.root))  # type: ignore[attr-defined]
    assert isinstance(r, Ok)
    assert [d.code for d in r.diagnostics] == ["unit.pending"]
    assert rt.snapshot().unit(user.id).state is InstanceState.PENDING

    provider = generic("provider", provides=["svc.a@1"])
    report = rt.reload(ReloadPlan(units_to_add=[provider])).unwrap()
    states = rt.snapshot().states()
    assert states[user.id] is InstanceState.ACTIVE
    assert states[provider.id] is InstanceState.ACTIVE
    assert report.loaded == [provider.id, user.id]


def test_failed_init_leaves_nothing(pkg: str, snapshot: Snapshot) -> None:
    root = snapshot("v1", {f"{pkg}/bad.py": '''
        from unit.runtime import Registry

        class Catalog:
            def __init__(self, ctx, config):
                self.reg = Registry("catalog")
                ctx.provide("cat@1", self.reg)

        class Bad:
            def __init__(self, ctx, config):
                ctx.register("cat@1", "item")
                ctx.effect(lambda: (lambda: None), "noop")
                ctx.provide("bad@1", object())
                raise RuntimeError("boom")
    '''})
    cat = spec(f"{pkg}.cat", f"{pkg}.bad:Catalog", provides=["cat@1"])
    bad = spec(f"{pkg}.bad", f"{pkg}.bad:Bad", provides=["bad@1"], requires=["cat@1"])
    rt = UnitRuntime(PROC)
    r = rt.load_process_plan(plan([cat, bad], root))
    assert isinstance(r, Failed) and r.has_code("unit.load_failed")
    snap = rt.snapshot()
    info = snap.unit(bad.id)
    assert info.state is InstanceState.FAILED
    assert info.error is not None and "boom" in info.error
    assert info.registrations == [] and info.provides == []
    assert "bad@1" not in snap.services
    catalog = rt.instance(cat.id)
    assert catalog is not None and len(catalog.reg) == 0  # type: ignore[attr-defined]
    assert not any(d.code == "unit.leak" for d in snap.diagnostics)


def test_provider_reload_propagates_to_dependents_only(generic: Callable[..., UnitSpec]) -> None:
    a = generic("a", provides=["svc.a@1"])
    b = generic("b", provides=["svc.b@1"], requires=["svc.a@1"])
    c = generic("c", requires=["svc.b@1"])
    other = generic("other", provides=["svc.o@1"])
    rt = UnitRuntime(PROC)
    rt.load_process_plan(plan([a, b, c, other], generic.root)).unwrap()  # type: ignore[attr-defined]
    gen = {u.unit: u.generation for u in rt.snapshot().units}

    report = rt.reload(ReloadPlan(units_to_reload=[a])).unwrap()
    assert report.unloaded == [c.id, b.id, a.id]
    assert report.loaded == [a.id, b.id, c.id]
    after = {u.unit: u.generation for u in rt.snapshot().units}
    assert after[other.id] == gen[other.id]
    for u in (a, b, c):
        assert after[u.id] == gen[u.id] + 1
    snap = rt.snapshot()
    assert snap.unit(b.id).fingerprint == [snap.services["svc.a@1"]]


def test_old_proxy_raises_after_reload(generic: Callable[..., UnitSpec]) -> None:
    a = generic("a", provides=["svc.a@1"])
    b = generic("b", requires=["svc.a@1"])
    rt = UnitRuntime(PROC)
    rt.load_process_plan(plan([a, b], generic.root)).unwrap()  # type: ignore[attr-defined]
    old_proxy = rt.instance(b.id).used["svc.a@1"]  # type: ignore[union-attr]
    assert old_proxy.ping()[0] == "svc.a@1"

    rt.reload(ReloadPlan(units_to_reload=[a])).unwrap()
    with pytest.raises(StaleServiceError):
        old_proxy.ping()
    assert rt.instance(b.id).used["svc.a@1"].ping()[0] == "svc.a@1"  # type: ignore[union-attr]


def test_undo_runs_in_reverse_and_survives_failures(pkg: str, snapshot: Snapshot) -> None:
    root = snapshot("v1", {f"{pkg}/u.py": '''
        from tests.support import events

        class Unit:
            def __init__(self, ctx, config):
                ctx.effect(lambda: (lambda: events.append(("undo", "first"))), "first")
                def broken():
                    raise RuntimeError("undo failed")
                ctx.effect(lambda: broken, "broken")
                ctx.effect(lambda: (lambda: events.append(("undo", "third"))), "third")
    '''})
    u = spec(f"{pkg}.u", f"{pkg}.u:Unit")
    rt = UnitRuntime(PROC)
    rt.load_process_plan(plan([u], root)).unwrap()
    rt.unload_all().unwrap()
    assert events == [("undo", "third"), ("undo", "first")]
    assert [d.code for d in rt.diagnostics] == ["unit.undo_failed"]


def test_early_dispose_runs_once(pkg: str, snapshot: Snapshot) -> None:
    root = snapshot("v1", {f"{pkg}/u.py": '''
        from tests.support import events

        class Unit:
            def __init__(self, ctx, config):
                self.stop = ctx.effect(lambda: (lambda: events.append(("undo", "x"))), "x")
    '''})
    u = spec(f"{pkg}.u", f"{pkg}.u:Unit")
    rt = UnitRuntime(PROC)
    rt.load_process_plan(plan([u], root)).unwrap()
    obj = rt.instance(u.id)
    obj.stop()  # type: ignore[union-attr]
    obj.stop()  # type: ignore[union-attr]
    assert rt.snapshot().unit(u.id).registrations == []
    rt.unload_all().unwrap()
    assert events == [("undo", "x")]


def test_manifest_limits_what_a_unit_can_touch(pkg: str, snapshot: Snapshot) -> None:
    root = snapshot("v1", {f"{pkg}/u.py": '''
        class Provider:
            def __init__(self, ctx, config):
                ctx.provide("svc.a@1", object())

        class Sneaky:
            def __init__(self, ctx, config):
                self.ctx = ctx
                ctx.use("svc.a@1")
    '''})
    p = spec(f"{pkg}.p", f"{pkg}.u:Provider", provides=["svc.a@1"])
    s = spec(f"{pkg}.s", f"{pkg}.u:Sneaky")
    rt = UnitRuntime(PROC)
    r = rt.load_process_plan(plan([p, s], root))
    assert isinstance(r, Failed)
    assert "UndeclaredServiceError" in (rt.snapshot().unit(s.id).error or "")


def test_context_is_dead_after_unload(pkg: str, snapshot: Snapshot) -> None:
    root = snapshot("v1", {f"{pkg}/u.py": '''
        class Unit:
            def __init__(self, ctx, config):
                self.ctx = ctx
    '''})
    u = spec(f"{pkg}.u", f"{pkg}.u:Unit")
    rt = UnitRuntime(PROC)
    rt.load_process_plan(plan([u], root)).unwrap()
    ctx = rt.instance(u.id).ctx  # type: ignore[union-attr]
    rt.reload(ReloadPlan(units_to_reload=[u])).unwrap()
    with pytest.raises(InactiveContextError):
        ctx.effect(lambda: (lambda: None))


def test_direct_registration_is_reported_as_leak(pkg: str, snapshot: Snapshot) -> None:
    root = snapshot("v1", {f"{pkg}/u.py": '''
        from unit.runtime import Registry

        class Catalog:
            def __init__(self, ctx, config):
                self.reg = Registry("catalog")
                ctx.provide("cat@1", self.reg)

        class Leaky:
            def __init__(self, ctx, config):
                ctx.use("cat@1").register("kernel")   # bypasses ctx.register: no undo
    '''})
    cat = spec(f"{pkg}.cat", f"{pkg}.u:Catalog", provides=["cat@1"])
    leaky = spec(f"{pkg}.leaky", f"{pkg}.u:Leaky", requires=["cat@1"])
    rt = UnitRuntime(PROC)
    rt.load_process_plan(plan([cat, leaky], root)).unwrap()
    rt.reload(ReloadPlan(units_to_remove=[leaky.id])).unwrap()
    leaks = [d for d in rt.diagnostics if d.code == "unit.leak"]
    assert len(leaks) == 1 and leaks[0].subject == leaky.id


def test_undeclared_service_name_is_rejected(pkg: str, snapshot: Snapshot) -> None:
    root = snapshot("v1", {f"{pkg}/u.py": '''
        class Unit:
            def __init__(self, ctx, config):
                ctx.provide("svc.other@1", object())
    '''})
    u = spec(f"{pkg}.u", f"{pkg}.u:Unit", provides=["svc.a@1"])
    rt = UnitRuntime(PROC)
    assert isinstance(rt.load_process_plan(plan([u], root)), Failed)
    assert UndeclaredServiceError.__name__ in (rt.snapshot().unit(u.id).error or "")


def test_optional_service_waits_for_pending_provider(generic: Callable[..., UnitSpec]) -> None:
    # "a.user" sorts before "b.provider" but waits for the optional service it provides.
    user = generic("a_user", optional=["svc.x@1"])
    provider = generic("b_provider", provides=["svc.x@1"])
    rt = UnitRuntime(PROC)
    rt.load_process_plan(plan([user, provider], generic.root)).unwrap()  # type: ignore[attr-defined]
    assert events[:2] == [("load", provider.id), ("load", user.id)]
