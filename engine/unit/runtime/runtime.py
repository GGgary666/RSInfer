"""The unit runtime (doc/specs/unit-runtime.md).

Modelled on Cordis' Fiber and HMR, with deliberate differences:

* synchronous and single-threaded: it only runs at a safe point the caller
  picks, and undo functions run one at a time in reverse order;
* unload order is computed (users before providers) instead of providers
  waiting for their dependents to finish;
* one flat level of units per process, no isolation scopes;
* the import graph comes from source (``imports.py``), and reloads are only
  triggered by an explicit ``ReloadPlan``.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import os
import sys
import traceback
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from core.common.framework import FrameworkModules
from core.common.ids import InstanceId, ProcId, UnitId
from core.common.manifest import UnitManifest
from core.common.result import Diagnostic, Failed, Ok, Result, Severity, diag
from core.context.config import ConfigEntry, ConfigStore, ConfigView, read_phase
from core.context.flags import RuntimeFlags
from core.context.resources import Resources
from platforms.base import Platform
from unit.runtime.context import UnitContext
from unit.runtime.errors import InjectedFault, ServiceConflictError
from unit.runtime.imports import ImportGraph, scan_root
from unit.runtime.proxy import Provision, ServiceProxy
from unit.runtime.registry import leftovers, running_as
from unit.runtime.types import (
    InstanceInfo,
    InstanceState,
    ProcessPlan,
    ReloadPlan,
    ReloadReport,
    RuntimeSnapshot,
    UnitSpec,
)

COMPONENT = "unit.runtime"

ESCALATE_PROCESS = "escalate.restart_process"
ESCALATE_ENGINE = "escalate.restart_engine"


@dataclass
class _Registration:
    label: str
    undo: Callable[[], object]
    done: bool = False


@dataclass(eq=False)
class _Slot:
    """One unit id in this process, across all of its loads."""

    spec: UnitSpec
    state: InstanceState = InstanceState.PENDING
    generation: int = 0
    instance: InstanceId | None = None
    obj: object | None = None
    regs: list[_Registration] = field(default_factory=list[_Registration])
    staged_provisions: dict[str, Provision] = field(default_factory=dict[str, Provision])
    provisions: dict[str, Provision] = field(default_factory=dict[str, Provision])
    uses: dict[str, Provision] = field(default_factory=dict[str, Provision])
    fingerprint: list[InstanceId] = field(default_factory=list[InstanceId])
    error: str | None = None
    staged: bool = False

    @property
    def id(self) -> UnitId:
        return self.spec.id

    @property
    def manifest(self) -> UnitManifest:
        return self.spec.manifest


@dataclass(frozen=True)
class _Before:
    spec: UnitSpec
    state: InstanceState
    generation: int
    error: str | None
    staged: bool


@dataclass
class _ModuleBackup:
    entries: dict[str, ModuleType]
    parent_attrs: dict[tuple[str, str], ModuleType]
    loaded_before: set[str]


@dataclass
class _PathBackup:
    sys_path: list[str]
    package_paths: dict[str, list[str]]
    code_root: str | None
    staged_root: str | None


class _Abort(Exception):
    def __init__(self, diags: list[Diagnostic]) -> None:
        super().__init__("; ".join(str(d) for d in diags))
        self.diags = diags


def _resolve(path: str) -> str:
    return str(Path(path).resolve())


def _is_extension(mod: ModuleType) -> bool:
    spec = getattr(mod, "__spec__", None)
    if spec is not None and isinstance(spec.loader, importlib.machinery.ExtensionFileLoader):
        return True
    file = getattr(mod, "__file__", None) or ""
    return file.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))


class UnitRuntime:
    def __init__(
        self,
        proc: ProcId,
        *,
        config: ConfigStore | None = None,
        flags: RuntimeFlags | None = None,
        platform: Platform | None = None,
        framework: FrameworkModules | None = None,
    ) -> None:
        self.proc = proc
        self.config = config if config is not None else ConfigStore()
        self.flags = flags if flags is not None else RuntimeFlags()
        self.platform = platform
        self.resources = Resources(platform)
        self.framework = framework if framework is not None else FrameworkModules.default()
        self.faults: set[str] = set()
        """One-shot fault points for tests: ``"before_unload"``, ``"unload:<unit>"``,
        ``"after_unload"``, ``"after_purge"``, ``"after_switch"``,
        ``"load:<unit>"``, ``"after_load"``."""
        self._slots: dict[UnitId, _Slot] = {}
        self._services: dict[str, Provision] = {}
        self._code_root: str | None = None
        self._staged_root: str | None = None
        self._roots: list[str] = []
        self._diags: list[Diagnostic] = []
        self._busy = False
        self._load_log: list[UnitId] = []
        self._unload_log: list[UnitId] = []
        self._actions: dict[tuple[UnitId, str], tuple[InstanceId, object]] = {}

    # ------------------------------------------------------------------ API

    def load_process_plan(self, plan: ProcessPlan) -> Result[None]:
        with self._operation():
            if plan.code_root is not None:
                root = _resolve(plan.code_root)
                if root not in sys.path:
                    sys.path.insert(0, root)
                self._code_root = root
                self._add_root(root)
            r = self.config.declare_entries(plan.config)
            if isinstance(r, Failed):
                return r
            errors: list[Diagnostic] = []
            for spec in plan.units:
                if spec.id in self._slots:
                    errors.append(self._diag("load", f"unit {spec.id} listed twice", spec.id, "unit.duplicate"))
                    continue
                errors += self._declare_unit(spec)
                self._slots[spec.id] = _Slot(spec)
            errors += self._check_reads(plan.units)
            if errors:
                for spec in plan.units:
                    self._slots.pop(spec.id, None)
                return Failed(errors)
            self._settle()
            return self._settled_result()

    def reload(self, plan: ReloadPlan, *, values: dict[str, Any] | None = None) -> Result[ReloadReport]:
        """Execute a reload instruction. Call only at a safe point.

        ``values``: config values that change together with the reload. Units it builds read
        them; if it fails, the old units are built again with the old values. ``on_config``
        hooks of the other units are the caller's to run once it succeeded.
        """
        with self._operation():
            return self._reload(plan, values or {})

    def stage_new_units(
        self, units: list[UnitSpec], *, code_root: str | None = None,
        changed_modules: Iterable[str] = (), config: dict[str, ConfigEntry] | None = None,
    ) -> Result[None]:
        """Prepare phase: load *new* units next to the running ones (§6.3).

        Units whose code imports a changed module cannot be staged, whether or
        not that module is loaded here: a loaded one is the old version, and an
        unloaded one would be found in the current snapshot first. They wait
        for the safe point.
        """
        with self._operation():
            return self._stage(units, code_root, set(changed_modules), config or {})

    def discard_staged(self) -> Result[None]:
        """Abort of a prepared change: remove staged units and their code."""
        with self._operation():
            self._discard_staged()
            return Ok(None)

    def unload_all(self) -> Result[None]:
        """Process stop: unload every unit, users first."""
        with self._operation():
            active = {s for s in self._slots.values() if s.state is InstanceState.ACTIVE}
            for slot in self._unload_order(active):
                self._unload(slot, final=True)
            for slot in self._slots.values():
                slot.state = InstanceState.DISPOSED
            return Ok(None)

    def settle(self) -> None:
        """Load units whose dependencies appeared since the last safe point."""
        with self._operation():
            self._settle()

    def lookup(self, service: str) -> Any | None:
        """For the process host: a proxy to the current provider, without a dependency edge."""
        prov = self._services.get(service)
        return ServiceProxy(prov) if prov is not None and not prov.revoked else None

    def action_handler(self, unit: str, name: str) -> object | None:
        """The handler a unit registered with ``ctx.on_action(name, ...)``."""
        entry = self._actions.get((UnitId(unit), name))
        return entry[1] if entry is not None else None

    def units_touched_by(self, plan: ReloadPlan) -> set[UnitId]:
        """Units whose instance ``plan`` replaces or creates: listed for reload or removal,
        importing a changed module, using any of those (they are unloaded with their
        providers), or added by it."""
        accepted = self._affected_modules(plan.changed_modules)
        direct = {*(s.id for s in plan.units_to_reload), *plan.units_to_remove,
                  *(s.id for s in self._slots.values() if s.manifest.unit.entry_module in accepted)}
        unloaded = {s.id for s in self._unload_order(self._slots[u] for u in direct if u in self._slots)}
        added = {s.id for s in plan.units_to_add if s.id not in self._slots or not self._slots[s.id].staged}
        return direct | unloaded | added

    def spec_of(self, unit: str) -> UnitSpec | None:
        slot = self._slots.get(UnitId(unit))
        return slot.spec if slot is not None else None

    def instance(self, unit: str) -> object | None:
        """The unit object itself; for tests and introspection only."""
        slot = self._slots.get(UnitId(unit))
        return slot.obj if slot is not None else None

    def snapshot(self) -> RuntimeSnapshot:
        units = [
            InstanceInfo(
                unit=s.id,
                kind=s.manifest.unit.kind,
                state=s.state,
                generation=s.generation,
                instance=s.instance if s.state is not InstanceState.PENDING else None,
                fingerprint=list(s.fingerprint),
                provides=sorted(s.provisions),
                uses={name: p.owner for name, p in sorted(s.uses.items())},
                registrations=[r.label for r in s.regs if not r.done],
                staged=s.staged,
                error=s.error,
            )
            for s in sorted(self._slots.values(), key=lambda s: s.id)
        ]
        services = {name: p.owner for name, p in sorted(self._services.items())}
        return RuntimeSnapshot(self.proc, self._code_root, units, services, list(self._diags))

    @property
    def code_root(self) -> str | None:
        return self._code_root

    @property
    def diagnostics(self) -> list[Diagnostic]:
        return list(self._diags)

    # ------------------------------------------------------- helpers: misc

    @contextmanager
    def _operation(self) -> Generator[None]:
        if self._busy:
            raise RuntimeError("unit runtime operations are not reentrant; call them at a safe point")
        self._busy = True
        try:
            yield
        finally:
            self._busy = False

    def _fault(self, point: str) -> None:
        if point in self.faults:
            self.faults.discard(point)  # one-shot: it models a failure of this attempt
            raise InjectedFault(point)

    def _diag(self, phase: str, reason: str, subject: str = "", code: str = "",
              hint: str = "", severity: Severity = "error") -> Diagnostic:
        d = diag(COMPONENT, phase, reason, subject=subject, code=code, hint=hint, severity=severity)
        self._diags.append(d)
        return d

    def _add_root(self, root: str) -> None:
        if root not in self._roots:
            self._roots.append(root)

    def _is_engine_module(self, mod: ModuleType | None) -> bool:
        if mod is None:
            return False
        file = getattr(mod, "__file__", None)
        if not file:
            return False
        return any(file.startswith(root + os.sep) for root in self._roots)

    def _engine_modules(self) -> dict[str, ModuleType]:
        return {name: mod for name, mod in list(sys.modules.items()) if self._is_engine_module(mod)}

    def _declare_unit(self, spec: UnitSpec) -> list[Diagnostic]:
        """Declare keys the unit owns (unless already shipped). Reads are checked by
        ``_check_reads`` once every unit of the operation has declared its keys."""
        m = spec.manifest
        out: list[Diagnostic] = []
        if self.framework.contains(m.unit.entry_module):
            out.append(self._diag("load", f"entry {m.unit.entry} is a framework module", m.id,
                                  "unit.entry_in_framework"))
        if self.proc.role not in m.unit.process:
            out.append(self._diag("load", f"unit is for roles {m.unit.process}, not {self.proc.role!r}",
                                  m.id, "unit.wrong_process"))
        for key, decl in m.config.items():
            if self.config.owner_of(key) is None:
                r = self.config.declare(key, m.id, decl)
                if isinstance(r, Failed):
                    out += r.diagnostics
            elif self.config.owner_of(key) != m.id:
                out.append(self._diag("load", f"owns {key!r}, already owned by {self.config.owner_of(key)}",
                                      m.id, "config.owner_conflict"))
        return out

    def _check_reads(self, specs: Iterable[UnitSpec]) -> list[Diagnostic]:
        return [d for spec in specs for d in self.config.check_readable(spec.manifest)]

    def _settled_result(self) -> Result[None]:
        failed = [s for s in self._slots.values() if s.state is InstanceState.FAILED]
        pending = [s for s in self._slots.values() if s.state is InstanceState.PENDING]
        warnings = [
            diag(COMPONENT, "load", f"waiting for {self._missing(s)}", subject=s.id,
                 code="unit.pending", severity="warning")
            for s in pending
        ]
        if failed:
            return Failed([
                diag(COMPONENT, "load", s.error or "failed", subject=s.id, code="unit.load_failed")
                for s in failed
            ] + warnings)
        return Ok(None, warnings)

    def _missing(self, slot: _Slot) -> list[str]:
        return [n for n in slot.manifest.requires.services if n not in self._services]

    # ------------------------------------------------- services & regs

    def _provide(self, slot: _Slot, name: str, obj: object) -> Provision:
        assert slot.instance is not None
        current = self._services.get(name)
        if current is not None and not current.revoked:
            raise ServiceConflictError(f"{slot.id} provides {name!r}, already provided by {current.owner}")
        for other in self._slots.values():
            if other is not slot and name in other.staged_provisions:
                raise ServiceConflictError(f"{slot.id} provides {name!r}, being provided by {other.id}")
        if name in slot.staged_provisions or name in slot.provisions:
            raise ServiceConflictError(f"{slot.id} provides {name!r} twice")
        prov = Provision(name, slot.instance, obj)
        if slot.state is InstanceState.ACTIVE:
            slot.provisions[name] = prov
            self._services[name] = prov
        else:
            slot.staged_provisions[name] = prov
        return prov

    def _revoke(self, prov: Provision) -> None:
        prov.revoked = True
        if self._services.get(prov.name) is prov:
            del self._services[prov.name]
        for slot in self._slots.values():
            if slot.instance == prov.owner:
                slot.provisions.pop(prov.name, None)
                slot.staged_provisions.pop(prov.name, None)

    def _add_action(self, slot: _Slot, name: str, handler: object) -> Callable[[], None]:
        assert slot.instance is not None
        key = (slot.id, name)
        if key in self._actions:
            raise ServiceConflictError(f"{slot.id} registers action {name!r} twice")
        self._actions[key] = (slot.instance, handler)

        def remove() -> None:
            if self._actions.get(key, (None,))[0] == slot.instance:
                del self._actions[key]

        return remove

    def _add_registration(self, slot: _Slot, label: str, undo: Callable[[], object]) -> Callable[[], None]:
        reg = _Registration(label, undo)
        slot.regs.append(reg)
        instance = slot.instance

        def dispose() -> None:
            if reg.done:
                return
            reg.done = True
            with running_as(instance):
                undo()
            if reg in slot.regs:
                slot.regs.remove(reg)

        return dispose

    # ---------------------------------------------------- load & unload

    def _settle(self) -> None:
        while (slot := self._next_ready()) is not None:
            self._load(slot)

    def _next_ready(self) -> _Slot | None:
        pending = sorted((s for s in self._slots.values() if s.state is InstanceState.PENDING),
                         key=lambda s: s.id)
        fallback: _Slot | None = None
        for s in pending:
            if any(n not in self._services for n in s.manifest.requires.services):
                continue
            # Prefer to wait for optional services that another pending unit will provide.
            waits = [
                n for n in s.manifest.requires.optional
                if n not in self._services
                and any(o is not s and n in o.manifest.provides.services for o in pending)
            ]
            if not waits:
                return s
            fallback = fallback or s
        return fallback

    def _load(self, slot: _Slot) -> bool:
        m = slot.manifest
        slot.generation += 1
        inst = InstanceId(self.proc, slot.id, slot.generation)
        slot.instance = inst
        slot.state = InstanceState.LOADING
        slot.error = None
        slot.uses = {name: self._services[name] for name in m.requires.services}
        slot.fingerprint = [p.owner for p in slot.uses.values()]
        view = ConfigView(self.config, inst, frozenset(m.readable_keys()))
        ctx = UnitContext(self, slot, inst, view)
        try:
            with running_as(inst), read_phase("startup"):
                self._fault(f"load:{slot.id}")
                module = importlib.import_module(m.unit.entry_module)
                cls = getattr(module, m.unit.entry_class)
                obj = cls(ctx, view)
                start = getattr(obj, "start", None)
                if callable(start):
                    start()
        except Exception as e:  # noqa: BLE001 - a unit's failure must not take the runtime down
            frames = traceback.extract_tb(e.__traceback__)
            where = f" at {frames[-1].filename}:{frames[-1].lineno}" if frames else ""
            slot.error = f"{type(e).__name__}: {e}{where}"
            self._diag("load", slot.error, slot.id, "unit.load_failed")
            self._teardown(slot)
            self._leak_check(slot, inst)
            slot.state = InstanceState.FAILED
            slot.uses = {}
            slot.fingerprint = []
            return False
        slot.obj = obj
        for name, prov in slot.staged_provisions.items():
            self._services[name] = prov
        slot.provisions = slot.staged_provisions
        slot.staged_provisions = {}
        slot.state = InstanceState.ACTIVE
        missing = sorted(set(m.provides.services) - set(slot.provisions))
        if missing:
            self._diag("load", f"declares but does not provide {missing}", slot.id,
                       "manifest.unprovided_service", severity="warning")
        self._load_log.append(slot.id)
        return True

    def _teardown(self, slot: _Slot) -> None:
        """Run undo functions in reverse registration order; keep going on failure."""
        with running_as(slot.instance):
            for reg in reversed(slot.regs):
                if reg.done:
                    continue
                reg.done = True
                try:
                    reg.undo()
                except Exception as e:  # noqa: BLE001
                    self._diag("unload", f"undo of {reg.label} raised {type(e).__name__}: {e}",
                               slot.id, "unit.undo_failed")
        slot.regs = []

    def _leak_check(self, slot: _Slot, inst: InstanceId) -> None:
        left: list[str] = []
        for prov in [*slot.provisions.values(), *slot.staged_provisions.values()]:
            left.append(f"service {prov.name}")
            self._revoke(prov)
        for name, prov in list(self._services.items()):
            if prov.owner == inst:
                left.append(f"service {name}")
                self._revoke(prov)
        left += [f"config hook {k}" for k in self.config.hooks_of(inst)]
        left += [f"action {n}" for (_, n), (owner, _) in self._actions.items() if owner == inst]
        left += self.resources.held_by(inst)
        left += leftovers(inst)
        if left:
            self._diag("unload", f"left registrations behind: {sorted(left)}", slot.id, "unit.leak",
                       hint="register through ctx so the runtime can undo it")
        slot.provisions = {}
        slot.staged_provisions = {}
        self.config.forget_reader(inst)

    def _unload(self, slot: _Slot, *, final: bool) -> None:
        inst = slot.instance
        assert inst is not None
        slot.state = InstanceState.UNLOADING
        self._teardown(slot)
        self._leak_check(slot, inst)
        slot.uses = {}
        slot.fingerprint = []
        slot.obj = None
        slot.state = InstanceState.DISPOSED if final else InstanceState.PENDING
        self._unload_log.append(slot.id)

    def _dependents(self, slot: _Slot) -> set[_Slot]:
        if slot.instance is None:
            return set()
        live = (InstanceState.ACTIVE, InstanceState.LOADING)
        return {
            s for s in self._slots.values()
            if s is not slot and s.state in live
            and any(p.owner == slot.instance for p in s.uses.values())
        }

    def _unload_order(self, targets: Iterable[_Slot]) -> list[_Slot]:
        """Targets plus their transitive users, users before providers (§6.2 steps 2–3)."""
        closure: set[_Slot] = set()
        stack = [t for t in targets if t.state is InstanceState.ACTIVE]
        while stack:
            s = stack.pop()
            if s not in closure:
                closure.add(s)
                stack.extend(self._dependents(s))
        order: list[_Slot] = []
        remaining = set(closure)
        while remaining:
            free = [s for s in remaining if not (self._dependents(s) & remaining)]
            if not free:
                self._diag("unload", "dependency cycle among active units", code="unit.cycle")
                free = list(remaining)
            pick = min(free, key=lambda s: s.id)
            order.append(pick)
            remaining.remove(pick)
        return order

    # ----------------------------------------------------------- reload

    def _escalation(self, changed: Iterable[str]) -> list[Diagnostic]:
        out: list[Diagnostic] = []
        declared = {
            ext: s.id for s in self._slots.values() for ext in s.manifest.capabilities.native_extensions
        }
        for m in sorted(set(changed)):
            if self.framework.contains(m):
                out.append(diag(COMPONENT, "check", f"framework module {m} changed", subject=m,
                                code=ESCALATE_ENGINE, hint="the engine must restart"))
            elif m in declared:
                out.append(diag(COMPONENT, "check", f"native extension {m} of {declared[m]} changed",
                                subject=m, code=ESCALATE_PROCESS, hint="native code cannot be unloaded"))
            elif (mod := sys.modules.get(m)) is not None and _is_extension(mod):
                out.append(diag(COMPONENT, "check", f"native module {m} changed", subject=m,
                                code=ESCALATE_PROCESS, hint="native code cannot be unloaded"))
                out.append(self._diag("check", f"native module {m} is loaded but no manifest declares it",
                                      m, "manifest.undeclared_native",
                                      hint="list it in capabilities.native_extensions"))
        return out

    def _affected_modules(self, changed: Iterable[str]) -> set[str]:
        loaded: dict[str, Path] = {}
        for name, mod in self._engine_modules().items():
            file = getattr(mod, "__file__", "") or ""
            if file.endswith(".py"):
                loaded[name] = Path(file)
        return ImportGraph(loaded).importers_closure(c for c in changed if c in loaded)

    def _reload(self, plan: ReloadPlan, values: dict[str, Any]) -> Result[ReloadReport]:
        errors: list[Diagnostic] = []
        reload_specs = {s.id: s for s in plan.units_to_reload}
        removed = set(plan.units_to_remove)
        for uid in [*reload_specs, *removed]:
            if uid not in self._slots:
                errors.append(self._diag("check", "no such unit in this process", uid, "unit.unknown"))
        for spec in plan.units_to_add:
            slot = self._slots.get(spec.id)
            if slot is not None and not slot.staged:
                errors.append(self._diag("check", "unit already exists", spec.id, "unit.duplicate"))
        if errors:
            return Failed(errors)
        escalation = self._escalation(plan.changed_modules)
        if escalation:
            return Failed(escalation)

        accepted = self._affected_modules(plan.changed_modules)
        framework_hit = sorted(m for m in accepted if self.framework.contains(m))
        if framework_hit:
            return Failed([diag(COMPONENT, "check", f"framework modules import changed code: {framework_hit}",
                                code=ESCALATE_ENGINE, hint="framework code must not import units")])
        bypassed = sorted(
            s.id for s in self._slots.values() if s.staged and s.manifest.unit.entry_module in accepted)
        if bypassed:
            return Failed([self._diag("check", "was staged during prepare but imports a changed module", uid,
                                      "unit.staged_imports_changed",
                                      hint="units that import changed modules are loaded at the safe point")
                           for uid in bypassed])
        code_hit = {s.id for s in self._slots.values() if s.manifest.unit.entry_module in accepted}
        unplanned = sorted(code_hit - set(reload_specs) - removed)
        warnings = [
            self._diag("check", "imports a changed module but was not in the plan; reloading it too",
                       uid, "unit.unplanned_reload", severity="warning")
            for uid in unplanned
        ]
        to_add = [s for s in plan.units_to_add if s.id not in self._slots]
        targets = [self._slots[u] for u in sorted({*reload_specs, *unplanned, *removed})]

        before = {
            s.id: _Before(s.spec, s.state, s.generation, s.error, s.staged) for s in self._slots.values()
        }
        config_before = self.config.entries()
        self._load_log = []
        self._unload_log = []
        mods: _ModuleBackup | None = None
        paths: _PathBackup | None = None
        try:
            self._fault("before_unload")
            for slot in self._unload_order(targets):
                self._unload(slot, final=slot.id in removed)
                self._fault(f"unload:{slot.id}")
            self._fault("after_unload")
            mods = self._purge_modules(accepted)
            self._fault("after_purge")
            paths = self._switch_root(plan.new_code_root)
            self._fault("after_switch")

            for uid in removed:
                del self._slots[uid]
                self.config.retract_owner(uid)
            # after the removals: a key may move from a removed unit to an added one
            r = self.config.declare_entries(plan.config)
            if isinstance(r, Failed):
                raise _Abort(r.diagnostics)
            applied = self.config.apply(values)
            if isinstance(applied, Failed):
                raise _Abort(applied.diagnostics)
            decl_errors: list[Diagnostic] = []
            for uid, spec in reload_specs.items():
                slot = self._slots[uid]
                slot.spec = spec
                if slot.state is InstanceState.FAILED:
                    slot.state = InstanceState.PENDING
                decl_errors += self._declare_unit(spec)
            for spec in to_add:
                decl_errors += self._declare_unit(spec)
                self._slots[spec.id] = _Slot(spec)
            decl_errors += self._check_reads([*reload_specs.values(), *to_add])
            if decl_errors:
                raise _Abort(decl_errors)
            self._settle()
            self._fault("after_load")
            expected = {u for u, b in before.items() if b.state is InstanceState.ACTIVE} - removed
            expected |= {s.id for s in plan.units_to_add}
            bad = sorted(u for u in expected if self._slots[u].state is not InstanceState.ACTIVE)
            if bad:
                raise _Abort([
                    diag(COMPONENT, "load",
                         f"is {self._slots[u].state} after reload"
                         + (f": {self._slots[u].error}" if self._slots[u].error else
                            f", waiting for {self._missing(self._slots[u])}"),
                         subject=u, code="unit.not_active")
                    for u in bad
                ])
        except (_Abort, InjectedFault) as e:
            cause = e.diags if isinstance(e, _Abort) else [
                diag(COMPONENT, "reload", f"fault injected at {e}", code="unit.injected_fault")]
            rolled = self._rollback(before, config_before, mods, paths)
            return Failed([*cause, *rolled])

        for slot in self._slots.values():
            slot.staged = False
        report = ReloadReport(
            unloaded=list(self._unload_log),
            loaded=list(self._load_log),
            reimported=sorted(accepted),
            unplanned=unplanned,
            diagnostics=warnings,
        )
        return Ok(report)

    def _rollback(
        self, before: dict[UnitId, _Before], config_before: dict[str, ConfigEntry],
        mods: _ModuleBackup | None, paths: _PathBackup | None,
    ) -> list[Diagnostic]:
        """Restore module cache, code source and units as they were (§6.2 step 7)."""
        fresh = [
            s for s in self._slots.values()
            if s.state is InstanceState.ACTIVE
            and (s.id not in before or before[s.id].generation != s.generation)
        ]
        for slot in self._unload_order(fresh):
            self._unload(slot, final=slot.id not in before)
        if mods is not None:
            self._restore_modules(mods)
        if paths is not None:
            self._restore_paths(paths)
        importlib.invalidate_caches()
        self.config.restore_entries(config_before)
        for uid in [u for u in self._slots if u not in before]:
            del self._slots[uid]
        for uid, b in before.items():
            slot = self._slots.get(uid)
            if slot is None:
                slot = self._slots[uid] = _Slot(b.spec, generation=b.generation)
            slot.spec = b.spec
            slot.staged = b.staged
            if slot.state is InstanceState.ACTIVE:
                continue
            if b.state is InstanceState.FAILED:
                slot.state, slot.error = InstanceState.FAILED, b.error
            else:
                slot.state, slot.error = InstanceState.PENDING, None
        self._settle()
        out = [diag(COMPONENT, "rollback", "reload rolled back; the previous units are serving",
                    code="unit.rolled_back", severity="warning")]
        lost = sorted(u for u, b in before.items()
                      if b.state is InstanceState.ACTIVE and self._slots[u].state is not InstanceState.ACTIVE)
        if lost:
            out.append(self._diag("rollback", f"units did not come back: {lost}", code=ESCALATE_PROCESS,
                                  hint="restart the process"))
        return out

    def _purge_modules(self, accepted: set[str]) -> _ModuleBackup:
        """Drop changed modules and their importers from the module cache (§6.2 step 4)."""
        backup = _ModuleBackup({}, {}, set(self._engine_modules()))
        for name in sorted(accepted, key=lambda n: -n.count(".")):
            mod = sys.modules.pop(name, None)
            if mod is None:
                continue
            backup.entries[name] = mod
            parent, _, child = name.rpartition(".")
            parent_mod = sys.modules.get(parent) if parent else None
            if parent_mod is not None and getattr(parent_mod, child, None) is mod:
                backup.parent_attrs[(parent, child)] = mod
                delattr(parent_mod, child)
        importlib.invalidate_caches()
        return backup

    def _restore_modules(self, backup: _ModuleBackup) -> None:
        for name, mod in list(sys.modules.items()):
            if not self._is_engine_module(mod):
                continue
            if name in backup.loaded_before and name not in backup.entries:
                continue
            del sys.modules[name]
            parent, _, child = name.rpartition(".")
            parent_mod = sys.modules.get(parent) if parent else None
            if parent_mod is not None and getattr(parent_mod, child, None) is mod:
                delattr(parent_mod, child)
        sys.modules.update(backup.entries)
        for (parent, child), mod in backup.parent_attrs.items():
            parent_mod = sys.modules.get(parent)
            if parent_mod is not None:
                setattr(parent_mod, child, mod)

    def _engine_packages(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for name, mod in self._engine_modules().items():
            path = getattr(mod, "__path__", None)
            if isinstance(path, list):
                out[name] = path  # type: ignore[assignment]
        return out

    def _backup_paths(self) -> _PathBackup:
        return _PathBackup(
            list(sys.path),
            {name: list(p) for name, p in self._engine_packages().items()},
            self._code_root,
            self._staged_root,
        )

    def _restore_paths(self, backup: _PathBackup) -> None:
        sys.path[:] = backup.sys_path
        for name, entries in backup.package_paths.items():
            mod = sys.modules.get(name)
            path = getattr(mod, "__path__", None) if mod is not None else None
            if isinstance(path, list):
                path[:] = entries
        self._code_root = backup.code_root
        self._staged_root = backup.staged_root

    def _map_entry(self, entry: str, new_root: str) -> str | None:
        for root in self._roots:
            if entry == root or entry.startswith(root + os.sep):
                candidate = new_root + entry[len(root):]
                return candidate if os.path.isdir(candidate) else None
        return None

    def _switch_root(self, new_root: str | None) -> _PathBackup | None:
        """Point sys.path and every engine package at the new snapshot (§6.2 step 5).

        Unchanged modules stay loaded from the old snapshot; snapshots are
        immutable, so their content is the same.
        """
        if new_root is None:
            return None
        new = _resolve(new_root)
        backup = self._backup_paths()
        old_roots = {r for r in (self._code_root, self._staged_root) if r is not None}
        replaced = False
        new_path: list[str] = []
        for entry in sys.path:
            if entry in old_roots:
                if not replaced:
                    new_path.append(new)
                    replaced = True
                continue
            new_path.append(entry)
        if not replaced:
            new_path.insert(0, new)
        sys.path[:] = new_path
        self._add_root(new)
        for path in self._engine_packages().values():
            mapped: list[str] = []
            for entry in path:
                target = self._map_entry(entry, new)
                choice = target if target is not None else entry
                if choice not in mapped:
                    mapped.append(choice)
            path[:] = mapped
        self._code_root = new
        self._staged_root = None
        importlib.invalidate_caches()
        return backup

    # ------------------------------------------------------------ staging

    def _stage(self, units: list[UnitSpec], code_root: str | None, changed: set[str],
               config: dict[str, ConfigEntry]) -> Result[None]:
        errors: list[Diagnostic] = []
        for spec in units:
            if spec.id in self._slots:
                errors.append(self._diag("stage", "unit already exists", spec.id, "unit.duplicate"))
            if spec.manifest.unit.entry_module in sys.modules:
                errors.append(self._diag("stage", f"module {spec.manifest.unit.entry_module} is already loaded",
                                         spec.id, "unit.stage_existing_module",
                                         hint="changed modules are reloaded at the safe point"))
        if errors:
            return Failed(errors)
        root = _resolve(code_root) if code_root is not None else None
        if root is not None and root != self._code_root:
            if self._staged_root not in (None, root):
                return Failed([self._diag("stage", "another snapshot is already staged", root,
                                          "unit.stage_conflict")])
            graph = ImportGraph(scan_root(Path(root)))
            closure = graph.imports_closure(s.manifest.unit.entry_module for s in units)
            conflict = sorted(closure & changed)
            if conflict:
                return Failed([self._diag("stage", f"imports changed modules {conflict}", root,
                                          "unit.stage_needs_safe_point",
                                          hint="load it at the safe point instead")])
            self._stage_root(root)
        r = self.config.declare_entries(config)
        if isinstance(r, Failed):
            return r
        for spec in units:
            errors += self._declare_unit(spec)
            self._slots[spec.id] = _Slot(spec, staged=True)
        errors += self._check_reads(units)
        if errors:
            self._discard_staged()
            return Failed(errors)
        self._settle()
        bad = [s for s in self._slots.values() if s.staged and s.state is not InstanceState.ACTIVE]
        if bad:
            failure = Failed([
                diag(COMPONENT, "stage", s.error or f"is {s.state}", subject=s.id, code="unit.not_active")
                for s in bad
            ])
            self._discard_staged()
            return failure
        return Ok(None)

    def _stage_root(self, root: str) -> None:
        """Make a new snapshot importable *after* the current one: only new modules come from it."""
        if root not in sys.path:
            idx = sys.path.index(self._code_root) + 1 if self._code_root in sys.path else len(sys.path)
            sys.path.insert(idx, root)
        self._add_root(root)
        for path in self._engine_packages().values():
            extra = [t for e in path if (t := self._map_entry(e, root)) is not None and t not in path]
            path.extend(dict.fromkeys(extra))
        self._staged_root = root
        importlib.invalidate_caches()

    def _discard_staged(self) -> None:
        staged = [s for s in self._slots.values() if s.staged]
        staged_ids = {s.id for s in staged}
        for slot in self._unload_order(staged):
            self._unload(slot, final=slot.id in staged_ids)
        for slot in staged:
            del self._slots[slot.id]
            self.config.retract_owner(slot.id)
        root = self._staged_root
        if root is not None:
            for name, mod in list(sys.modules.items()):
                file = getattr(mod, "__file__", None) or ""
                if file.startswith(root + os.sep):
                    del sys.modules[name]
                    parent, _, child = name.rpartition(".")
                    parent_mod = sys.modules.get(parent) if parent else None
                    if parent_mod is not None and getattr(parent_mod, child, None) is mod:
                        delattr(parent_mod, child)
            sys.path[:] = [e for e in sys.path if e != root]
            for path in self._engine_packages().values():
                path[:] = [e for e in path if not (e == root or e.startswith(root + os.sep))]
            self._staged_root = None
            importlib.invalidate_caches()
        self._settle()
