"""From a change-set to an action plan (doc/specs/control-plane.md §4).

Answers "what must move because of this change, and how": the cheapest rung of
the action ladder (modules.html ③ §3) that is still correct, for every process.
It never executes anything; ``apply --dry-run`` returns exactly what it builds.
When it cannot tell, it picks the more expensive action and says so in ``notes``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.change.state import IdentityManifest
from core.change.types import EMPTY, ActionPlan, ChangeSet, ProcActions, UnitAction
from core.common.framework import FrameworkModules
from core.common.ids import ProcId, UnitId
from core.common.manifest import ConfigKeyDecl
from core.common.result import Diagnostic, Failed, Ok, Result, diag
from core.context.config import config_entries
from core.sourcetree import Tree, module_of
from core.supervisor.protocol import ProcIntrospection
from core.supervisor.topology import DesiredProc, Topology, desired_procs
from unit.runtime.imports import ImportGraph
from unit.runtime.types import ProcessPlan, ReloadPlan, UnitSpec

COMPONENT = "core.deps"
CONTROL_ROLE = "control"
NATIVE_SUFFIXES = (".so", ".pyd")


@dataclass
class World:
    """Everything a plan is derived from."""

    cs: ChangeSet
    base: IdentityManifest
    new: IdentityManifest
    base_tree: Tree | None
    new_tree: Tree
    running: dict[ProcId, ProcIntrospection]
    """Processes that take part in place (RUNNING), as they described themselves."""
    control: ProcIntrospection | None
    framework: FrameworkModules
    notes: list[str] = field(default_factory=list[str])


def plan(w: World) -> Result[ActionPlan]:
    topo = w.new.topology
    if topo is None:
        return Failed([diag(COMPONENT, "plan", "the system has no topology", code="deps.no_topology")])
    if w.cs.source == "startup" or w.base_tree is None:
        return Ok(ActionPlan(spawn=[_process_plan(d.proc, d.units, w.new_tree, w.new.config)
                                    for d in _desired(w.new_tree, topo).values()], topology=topo))
    changed_paths = sorted(p for p in {*w.base.files, *w.new.files} if w.base.files.get(p) != w.new.files.get(p))
    # modified or deleted; a new file changes nothing that runs (new units are added, not reloaded)
    changed_modules = sorted({m for p in changed_paths if p in w.base.files and (m := module_of(p)) is not None})
    framework = [m for m in changed_modules if w.framework.contains(m)]
    if framework or "engine/framework.toml" in changed_paths:
        changed = framework or ["framework.toml"]
        return Ok(ActionPlan(restart_engine=True, notes=[f"framework modules changed: {changed}"]))
    bad = _check_state_requests(w)
    if bad:
        return Failed(bad)

    p = _Planner(w, changed_paths, changed_modules)
    old_topo = w.base.topology or topo
    want = _desired(w.new_tree, topo)
    had = _desired(w.base_tree, old_topo)
    procs: list[ProcActions] = []
    spawn: list[ProcessPlan] = []
    stop: list[ProcId] = []
    leads = {r.name for r in topo.roles if r.leads_epochs}
    for proc in sorted(w.running):
        if proc not in want:
            stop.append(proc)
            continue
        d = want[proc]
        after = _process_plan(proc, d.units, w.new_tree, w.new.config)
        why = p.restart_reason(proc, d, had.get(proc), old_topo, topo)
        if why:
            w.notes.append(f"{proc}: restart ({why})")
            procs.append(ProcActions(proc, restart=after, leads_epochs=proc.role in leads))
        else:
            procs.append(p.in_place(w.running[proc], d.units, after, leads_epochs=proc.role in leads))
    for proc in sorted(want):
        if proc not in w.running:
            spawn.append(_process_plan(proc, want[proc].units, w.new_tree, w.new.config))
    control: ProcActions | None = None
    if w.control is not None:
        units = w.new_tree.role_units(topo, CONTROL_ROLE)
        if p.natives_touch(units) or p.natives_touch([UnitId(u.unit) for u in w.control.units]):
            return Ok(ActionPlan(restart_engine=True, notes=["a native extension of a control-plane unit changed"]))
        c = p.in_place(w.control, units, _process_plan(w.control.proc, units, w.new_tree, w.new.config))
        if c.reload is not None or c.config or c.stage or c.actions:
            control = c
    if p.state_lost:
        return Ok(ActionPlan(restart_engine=True, notes=p.state_lost))
    return Ok(ActionPlan(procs=procs, control=control, spawn=spawn, stop=stop,
                         topology=topo if w.cs.topology is not None else None, notes=w.notes))


def _desired(tree: Tree, topo: Topology) -> dict[ProcId, DesiredProc]:
    """The topology's processes, with role units resolved against ``tree`` (manifests decide when a role
    lists none)."""
    out: dict[ProcId, DesiredProc] = {}
    for proc, d in desired_procs(topo).items():
        out[proc] = DesiredProc(proc, tree.role_units(topo, proc.role), d.gpus, d.env, d.rank,
                                d.platform, d.platform_options)
    return out


def _process_plan(proc: ProcId, units: list[UnitId], tree: Tree, values: dict[str, Any]) -> ProcessPlan:
    """What a process is: its units and the config keys they own or read, with this system's values."""
    config = config_entries([tree.units[u].manifest for u in units], [s.manifest for s in tree.units.values()],
                            values)
    return ProcessPlan(proc, EMPTY, [UnitSpec(tree.units[u].manifest) for u in sorted(units)], config)


def _owners(tree: Tree) -> dict[str, tuple[UnitId, ConfigKeyDecl]]:
    return {key: (uid, decl) for uid, s in tree.units.items() for key, decl in s.manifest.config.items()}


def _check_state_requests(w: World) -> list[Diagnostic]:
    out: list[Diagnostic] = []
    for r in w.cs.state:
        s = w.new_tree.units.get(r.unit)
        if s is None or s.manifest.unit.kind != "state":
            out.append(diag(COMPONENT, "plan", f"{r.unit} is not a state unit", subject=r.unit,
                            code="deps.bad_state_request"))
        elif r.action not in s.manifest.capabilities.actions:
            out.append(diag(COMPONENT, "plan", f"{r.unit} does not offer {r.action!r} "
                            f"(capabilities.actions: {s.manifest.capabilities.actions})", subject=r.unit,
                            code="deps.bad_state_request"))
    return out


class _Planner:
    def __init__(self, w: World, changed_paths: list[str], changed_modules: list[str]) -> None:
        assert w.base_tree is not None
        self.w = w
        self.base_tree = w.base_tree
        self.changed_paths = changed_paths
        self.changed = set(changed_modules)
        self.new_graph = ImportGraph(w.new_tree.modules)
        self.base_graph = ImportGraph(w.base_tree.modules)
        self.owners = _owners(w.new_tree)
        self.state_lost: list[str] = []
        """Why the change restarts the engine: a config change would rebuild state in flight."""

    def _binding(self, key: str) -> str:
        owner = self.owners.get(key)
        return owner[1].binding if owner is not None else "startup"

    # ------------------------------------------------------------- code

    def code_changed(self, uid: UnitId) -> bool:
        """Its manifest changed, or its code imports (transitively) a changed module."""
        new, old = self.w.new_tree.units.get(uid), self.base_tree.units.get(uid)
        if new is None or old is None or new.manifest != old.manifest:
            return True
        closure = self.new_graph.imports_closure([new.manifest.unit.entry_module]) | \
            self.base_graph.imports_closure([old.manifest.unit.entry_module])
        return bool(closure & self.changed)

    def stageable(self, uid: UnitId) -> bool:
        """New units load during prepare unless they import a changed module (unit-runtime.md §6.3)."""
        s = self.w.new_tree.units[uid]
        return not (self.new_graph.imports_closure([s.manifest.unit.entry_module]) & self.changed)

    def staging(self, intro: ProcIntrospection, added: list[UnitId], target: set[UnitId],
                changed_keys: dict[str, Any]) -> tuple[list[UnitId], dict[UnitId, str]]:
        """Which new units load during prepare (rung 0); the others are added inside the reload at the
        safe point, which declares new keys and applies new values before it builds them. A unit loads
        early only if what it needs is ready then: its code imports no changed module, every key it
        owns or reads is declared with its final value, and every service it uses is provided: by the
        units running now, as they are now, or by new units loaded with it. Returns the early units and
        why each other one waits."""
        declared = set(intro.config)
        here = [UnitId(u.unit) for u in intro.units]
        provided_now = {name for u in here for name in self._provides_now(u)}
        staying = {u for u in here if u in target}  # their candidate versions serve once the reload ran
        early = {u for u in added if self.stageable(u)}
        why = {u: "it imports changed code" for u in added if u not in early}
        while True:  # deferring one unit can defer those that need it
            late = {u: r for u in sorted(early) if (r := self._not_ready(u, early, declared, changed_keys,
                                                                          provided_now, set(why) | staying))}
            if not late:
                return sorted(early), why
            early -= set(late)
            why.update(late)

    def _not_ready(self, uid: UnitId, early: set[UnitId], declared: set[str], changed_keys: dict[str, Any],
                   provided_now: set[str], after_reload: set[UnitId]) -> str:
        """Why ``uid`` cannot be built during prepare, or ``""``. ``after_reload``: the units whose
        services exist only once the reload ran: new units added in it, and the units running now,
        whose candidate versions may provide services their running versions do not."""
        m = self.w.new_tree.units[uid].manifest
        taken = sorted(k for k in m.config if k in declared)
        if taken:
            return f"it owns {taken}, declared here by another unit until the reload"
        setting = sorted(k for k in m.reads.config if k in changed_keys)
        if setting:
            return f"it reads {setting}, which this change sets"
        undeclared = sorted(k for k in m.reads.config if k not in declared
                            and (k not in self.owners or self.owners[k][0] not in early))
        if undeclared:
            return f"it reads {undeclared}, which only the reload declares here"
        needs = sorted(name for name in [*m.requires.services, *m.requires.optional]
                       if name not in provided_now and not any(name in self._provides_new(u) for u in early)
                       and any(name in self._provides_new(u) for u in after_reload))
        if needs:
            return f"it uses {needs}, provided only once the reload ran"
        return ""

    def _provides_now(self, uid: UnitId) -> list[str]:
        """Services a running unit provides now: its version in the running identity's tree."""
        s = self.base_tree.units.get(uid)
        return list(s.manifest.provides.services) if s is not None else []

    def _provides_new(self, uid: UnitId) -> list[str]:
        """Services a unit provides after the change: a new unit, or the candidate version of a running one."""
        s = self.w.new_tree.units.get(uid)
        return list(s.manifest.provides.services) if s is not None else []

    def reloaded_with(self, units: set[UnitId], present: set[UnitId]) -> set[UnitId]:
        """``units`` and, transitively, the units of ``present`` that use a service one of them
        provides: the runtime reloads a provider's users with it."""
        out, todo = set(units), list(units)
        while todo:
            provided = set(self._provides_new(todo.pop()))
            for u in sorted(present - out):
                m = self.w.new_tree.units[u].manifest
                if provided & {*m.requires.services, *m.requires.optional}:
                    out.add(u)
                    todo.append(u)
        return out

    def drains(self, uid: UnitId) -> bool:
        m = self.w.new_tree.units[uid].manifest
        return m.unit.kind == "state" and (m.state is None or m.state.drain_on_reload)

    def natives_touch(self, units: list[UnitId]) -> bool:
        changed = {p.removeprefix("engine/") for p in self.changed_paths}
        for uid in units:
            for tree in (self.w.new_tree, self.base_tree):
                s = tree.units.get(uid)
                if s is None:
                    continue
                declared = set(s.manifest.capabilities.native_extensions)
                own = {p for p in changed if p.endswith(NATIVE_SUFFIXES)
                       and (tree.root / p).resolve().is_relative_to(s.directory.resolve())}
                if declared & changed or own:
                    return True
        return False

    def restart_reason(self, proc: ProcId, want: DesiredProc, had: DesiredProc | None, old: Topology,
                       new: Topology) -> str:
        if had is None:
            return "it is not part of the topology it was started for"
        if want.rank is not None and old.mesh != new.mesh:
            return f"the mesh changed {old.mesh.dims} -> {new.mesh.dims}"
        if (want.gpus, want.env, want.rank) != (had.gpus, had.env, had.rank):
            return "its placement changed"
        if (want.platform, want.platform_options) != (had.platform, had.platform_options):
            return f"its platform changed {had.platform} {had.platform_options} -> " \
                   f"{want.platform} {want.platform_options}"
        if self.natives_touch(sorted({*want.units, *(UnitId(u.unit) for u in self.w.running[proc].units)})):
            return "a native extension it loads changed; it cannot be unloaded"
        return ""

    # ---------------------------------------------------------- in place

    def in_place(self, intro: ProcIntrospection, units_after: list[UnitId], after: ProcessPlan, *,
                 leads_epochs: bool = False) -> ProcActions:
        w = self.w
        before = {UnitId(u.unit) for u in intro.units}
        target = set(units_after)
        removed = before - target
        added = sorted(target - before)
        reload = {u for u in before & target if self.code_changed(u)}
        new_values = {k: e.value for k, e in after.config.items()}
        changed_keys = {k: v for k, v in new_values.items() if k in intro.config and intro.config[k] != v}
        new_keys = {k: v for k, v in new_values.items() if k not in intro.config}

        actions: list[UnitAction] = []
        rebuild: dict[tuple[UnitId, str], tuple[str, set[str]]] = {}
        by_config: dict[UnitId, set[str]] = {}
        for r in intro.reads:
            if r.key not in changed_keys or r.unit in reload or r.unit in removed:
                continue
            if r.phase == "plan_build":
                art = next((a for a in intro.artifacts if a.owner == r.unit and a.name == r.artifact), None)
                if art is None:
                    w.notes.append(f"{intro.proc}: {r.unit} built {r.artifact!r} from {r.key!r} but did not "
                                   "register it; reloading it")
                    reload.add(r.unit)
                    by_config.setdefault(r.unit, set()).add(r.key)
                else:
                    rebuild.setdefault((art.owner, art.name), (art.rebuild, set()))[1].add(r.key)
            elif r.hooked and self._binding(r.key) != "startup":
                pass  # rung 1: it takes the new value in place (ctx.on_config)
            else:
                reload.add(r.unit)  # rung 2: a startup key, or read without a hook
                by_config.setdefault(r.unit, set()).add(r.key)
        lost = sorted(u for u in self.reloaded_with(set(by_config), before & target) if self.drains(u))
        if lost:
            keys = sorted({k for ks in by_config.values() for k in ks})
            self.state_lost.append(
                f"{intro.proc}: changing {keys} reloads {lost}, which hold state in flight ([state] drain_on_reload); "
                "draining and migrating it are not implemented yet (stage2-contract.md §5.5.5): restart the engine")
        for a in intro.artifacts:
            if a.owner in reload or a.owner in removed:
                continue
            keys = set(a.config) & set(changed_keys)
            built_from = set(a.units) & (reload | removed)
            if keys or built_from:
                rebuild.setdefault((a.owner, a.name), (a.rebuild, set()))[1].update(keys)
        for (owner, name), (action, keys) in sorted(rebuild.items()):
            actions.append(UnitAction(owner, action, {"artifact": name,
                                                      "config": {k: changed_keys[k] for k in sorted(keys)}}))
        for r in w.cs.state:
            if r.unit in target:
                actions.append(UnitAction(r.unit, r.action, dict(r.args)))

        early, why = self.staging(intro, added, target, changed_keys)
        for u, reason in sorted(why.items()):
            w.notes.append(f"{intro.proc}: {u} is added at the safe point: {reason}")
        stage = [UnitSpec(w.new_tree.units[u].manifest) for u in early]
        add_at_safe_point = [UnitSpec(w.new_tree.units[u].manifest) for u in added if u not in early]
        # keys new to this process: those of added units come with the units (participant), the others
        # (declared anew by a reloaded unit, or read from another role) are declared by the reload
        added_set = set(added)
        reload_plan: ReloadPlan | None = None
        if reload or removed or add_at_safe_point or self.changed:
            reload_plan = ReloadPlan(
                changed_modules=sorted(self.changed),
                units_to_reload=[UnitSpec(w.new_tree.units[u].manifest) for u in sorted(reload)],
                units_to_add=add_at_safe_point, units_to_remove=sorted(removed),
                config={k: e for k, e in after.config.items() if k in new_keys and e.owner not in added_set},
            )
        config = {**changed_keys, **{k: v for k, v in new_keys.items()
                                      if after.config[k].owner in added_set and k in w.new.config}}
        return ProcActions(intro.proc, config=config, stage=stage, reload=reload_plan, actions=actions,
                           leads_epochs=leads_epochs, plan_after=after)
