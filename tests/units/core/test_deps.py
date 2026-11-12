"""core.deps (doc/specs/control-plane.md §4): one rule of the action ladder at a time, on small trees.

Also context.md §6 tests 2 (live key: in place with a hook, reload without), 3 (plan key: rebuild
what was recorded from it) and 4 (startup key: reload only its readers).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from msgspec.structs import replace

from core.change.blobs import content_hash
from core.change.state import IdentityManifest
from core.change.types import ActionPlan, ChangeMeta, ChangeSet, ProcActions, StateRequest, UnitAction, WhenBusy
from core.common.framework import FrameworkModules
from core.common.ids import HostId, Identity, ProcId, UnitId
from core.common.manifest import parse_manifest
from core.common.result import Failed
from core.deps.planner import FORCE_DRAIN_S, World, plan
from core.sourcetree import read_tree
from core.supervisor import HostSpec, Mesh, RankPlacement, RoleSpec, Topology
from core.supervisor.topology import desired_procs
from core.supervisor.protocol import ProcIntrospection, ReadRecord, UnitBrief
from tests.framework.support import manifest_toml, write_tree
from unit.runtime import DerivedArtifact, InstanceState

H = HostId("h")
W0 = ProcId(H, "worker", 0)
W1 = ProcId(H, "worker", 1)


def topology(ranks: int = 1) -> Topology:
    return Topology([HostSpec(H, "addr", "/snapshots", gpus=list(range(ranks)))], Mesh({"tp": ranks}),
                    [RankPlacement(i, H, i) for i in range(ranks)], [RoleSpec("worker", [], per_rank=True)])


A = manifest_toml("pkg.a", "pkg.a.unit:Unit", provides=["pkg.a@1"], config={
    "pkg.live": {"type": "int", "default": 1, "binding": "live"},
    "pkg.plan": {"type": "int", "default": 8, "binding": "plan"},
    "pkg.start": {"type": "int", "default": 0, "binding": "startup"},
})
B = manifest_toml("pkg.b", "pkg.b.unit:Unit", requires=["pkg.a@1"], reads=["pkg.live"])
BASE = {
    "pkg/a/unit.toml": A, "pkg/a/unit.py": "class Unit:\n    pass\n",
    "pkg/b/unit.toml": B, "pkg/b/unit.py": "from pkg import lib\n\nclass Unit:\n    pass\n",
    "pkg/lib.py": "def helper():\n    return 1\n",
}
CURRENT = {"pkg.live": 1, "pkg.plan": 8, "pkg.start": 0}


@dataclass
class Case:
    tmp: Path
    new: dict[str, str] = field(default_factory=lambda: dict(BASE))
    base: dict[str, str] = field(default_factory=lambda: dict(BASE))
    config: dict[str, Any] = field(default_factory=dict[str, Any])
    hooked: bool = True
    topo: Topology = field(default_factory=topology)
    state: list[StateRequest] = field(default_factory=list[StateRequest])
    units: tuple[str, ...] = ("pkg.a", "pkg.b")
    meta: ChangeMeta = field(default_factory=ChangeMeta)
    artifacts: list[DerivedArtifact] | None = None

    def plan(self) -> ActionPlan | Failed:
        base_root = write_tree(self.tmp / "base" / "engine", self.base)
        new_root = write_tree(self.tmp / "new" / "engine", self.new)
        base = IdentityManifest(files=_hashes(base_root), topology=topology())
        new = IdentityManifest(files=_hashes(new_root), config=self.config, topology=self.topo)
        cs = ChangeSet(base=Identity("b"), config=self.config, topology=self.topo if self.topo != topology() else None,
                       state=self.state, meta=self.meta)
        intro = ProcIntrospection(
            W0, Identity("b"), [UnitBrief(u, InstanceState.ACTIVE, None, 1) for u in self.units],
            reads=[ReadRecord("pkg.live", UnitId("pkg.b"), "runtime", hooked=self.hooked),
                   ReadRecord("pkg.start", UnitId("pkg.a"), "startup"),
                   ReadRecord("pkg.plan", UnitId("pkg.a"), "plan_build", artifact="g")],
            artifacts=self.artifacts if self.artifacts is not None else
                      [DerivedArtifact(UnitId("pkg.a"), "g", "pkg.rebuild", ["pkg.plan"], [])],
            config=dict(CURRENT))
        r = plan(World(cs, base, new, read_tree(base_root), read_tree(new_root), {W0: intro}, None,
                       FrameworkModules.default()))
        return r if isinstance(r, Failed) else r.value


def _hashes(root: Path) -> dict[str, str]:
    return {"engine/" + p.relative_to(root).as_posix(): content_hash(p.read_bytes())
            for p in sorted(root.rglob("*")) if p.is_file()}


def only(p: ActionPlan | Failed) -> ProcActions:
    assert not isinstance(p, Failed), p
    assert [pa.proc for pa in p.procs] == [W0]
    return p.procs[0]


def reloaded(pa: ProcActions) -> list[str]:
    return [u.id for u in pa.reload.units_to_reload] if pa.reload else []


def test_a_live_key_with_a_hook_is_taken_in_place(tmp_path: Path) -> None:
    pa = only(Case(tmp_path, config={"pkg.live": 5}).plan())
    assert pa.config == {"pkg.live": 5} and reloaded(pa) == [] and pa.actions == []


def test_a_live_key_read_without_a_hook_reloads_the_reader(tmp_path: Path) -> None:
    pa = only(Case(tmp_path, config={"pkg.live": 5}, hooked=False).plan())
    assert pa.config == {"pkg.live": 5} and reloaded(pa) == ["pkg.b"]


def test_a_key_recorded_into_an_artifact_rebuilds_that_artifact(tmp_path: Path) -> None:
    pa = only(Case(tmp_path, config={"pkg.plan": 16}).plan())
    assert reloaded(pa) == []
    assert pa.actions == [UnitAction(UnitId("pkg.a"), "pkg.rebuild", {"artifact": "g", "config": {"pkg.plan": 16}})]


def test_a_startup_key_reloads_only_its_readers(tmp_path: Path) -> None:
    pa = only(Case(tmp_path, config={"pkg.start": 3}).plan())
    assert reloaded(pa) == ["pkg.a"] and pa.actions == []


def test_an_artifact_without_a_rebuild_action_reloads_its_owner(tmp_path: Path) -> None:
    """``ctx.derived(rebuild="")``: nothing rebuilds the artifact in place, so a change to a unit it
    was built from reloads its owner too, after that unit; and so does a unit added that registers
    with one of them (a new kernel unit may change what the artifact is built from). A rollback
    then reloads the old owner over the restored units, which an action's abort could not wait for."""
    tree = {**BASE, "pkg/b/unit.toml": manifest_toml("pkg.b", "pkg.b.unit:Unit", requires=["pkg.a@1"])}
    artifact = DerivedArtifact(UnitId("pkg.b"), "g", "", [], [UnitId("pkg.a")])
    changed = {**tree, "pkg/a/unit.py": "class Unit:\n    changed = True\n"}
    pa = only(Case(tmp_path / "changed", base=tree, new=changed, artifacts=[artifact]).plan())
    assert reloaded(pa) == ["pkg.a", "pkg.b"] and pa.actions == []
    added = {**tree, "pkg/c/unit.toml": manifest_toml("pkg.c", "pkg.c.unit:Unit", requires=["pkg.a@1"]),
             "pkg/c/unit.py": "class Unit:\n    pass\n"}
    pa = only(Case(tmp_path / "added", base=tree, new=added, artifacts=[artifact]).plan())
    assert reloaded(pa) == ["pkg.b"] and pa.actions == []
    with_action = DerivedArtifact(UnitId("pkg.b"), "g", "pkg.rebuild", [], [UnitId("pkg.a")])
    pa = only(Case(tmp_path / "action", base=tree, new=changed, artifacts=[with_action]).plan())
    assert reloaded(pa) == ["pkg.a"] and [(a.unit, a.action) for a in pa.actions] == [("pkg.b", "pkg.rebuild")]


def with_units(**manifests: str) -> dict[str, str]:
    return {**BASE, **{f"pkg/{name}/unit.toml": m for name, m in manifests.items()}}


def state_a(drain: bool) -> str:
    return manifest_toml("pkg.a", "pkg.a.unit:Unit", provides=["pkg.a@1"], kind="state", config={
        "pkg.live": {"type": "int", "default": 1, "binding": "live"},
        "pkg.plan": {"type": "int", "default": 8, "binding": "plan"},
        "pkg.start": {"type": "int", "default": 0, "binding": "startup"},
    }) + ("" if drain else "\n[state]\ndrain_on_reload = false\n")


def test_a_config_change_that_rebuilds_a_state_unit_reloads_it_in_place(tmp_path: Path) -> None:
    """A change applies to an empty engine (doc/specs/empty-apply.md): a state unit rebuilt then
    holds nothing in flight, so the change reloads it rather than restart the engine (as it did
    until the engine was emptied first)."""
    tree = with_units(a=state_a(drain=True))
    p = Case(tmp_path, new=tree, base=tree, config={"pkg.start": 3}).plan()
    assert not isinstance(p, Failed) and not p.restart_engine
    assert reloaded(only(p)) == ["pkg.a"]


DRAINS = {"pkg/d/unit.toml": manifest_toml("pkg.d", "pkg.d.unit:Unit", capabilities='actions = ["drain"]'),
          "pkg/d/unit.py": "class Unit:\n    pass\n"}
"""A unit that holds requests (a scheduler, a frontend): it declares the action ``drain``."""


def test_the_units_that_hold_requests_empty_the_engine_before_a_change(tmp_path: Path) -> None:
    """Each gets ``drain`` with the change's mode; the change waits as long as it asked to (or,
    forcing, for the steps in flight), and the plan says the engine is emptied first."""
    tree = {**BASE, **DRAINS}
    cases: list[tuple[WhenBusy, float | None]] = [("refuse", None), ("wait", 120.0), ("force", FORCE_DRAIN_S)]
    for when, timeout in cases:
        p = Case(tmp_path / when, new=tree, base=tree, config={"pkg.start": 3}, units=("pkg.a", "pkg.b", "pkg.d"),
                 meta=ChangeMeta(when_busy=when, wait_s=120.0)).plan()
        pa = only(p)
        # the only process draining: nothing in front of it closes a door, so it is the gate
        assert pa.drain == [UnitAction(UnitId("pkg.d"), "drain", {"mode": when, "gate": True})]
        assert not isinstance(p, Failed) and p.drain_timeout_s == timeout
        what = {"refuse": "refuse the change", "wait": "wait until they end", "force": "end them"}[when]
        assert any(f"must be empty first; if requests are in it: {what}" in n for n in p.notes)


def test_a_code_change_empties_the_engine_too(tmp_path: Path) -> None:
    tree = {**BASE, **DRAINS}
    pa = only(Case(tmp_path, new={**tree, "pkg/lib.py": "def helper():\n    return 2\n"}, base=tree,
                   units=("pkg.a", "pkg.b", "pkg.d")).plan())
    assert reloaded(pa) == ["pkg.b"] and [a.unit for a in pa.drain] == ["pkg.d"]


def test_a_change_of_live_keys_only_waits_for_nothing(tmp_path: Path) -> None:
    tree = {**BASE, **DRAINS}
    p = Case(tmp_path, new=tree, base=tree, config={"pkg.live": 5}, units=("pkg.a", "pkg.b", "pkg.d")).plan()
    pa = only(p)
    assert pa.config == {"pkg.live": 5} and pa.drain == []
    assert not isinstance(p, Failed) and p.drain_timeout_s is None


def test_a_state_unit_that_need_not_drain_is_reloaded_for_a_config_change(tmp_path: Path) -> None:
    tree = with_units(a=state_a(drain=False))
    pa = only(Case(tmp_path, new=tree, base=tree, config={"pkg.start": 3}).plan())
    assert reloaded(pa) == ["pkg.a"]


def test_a_library_change_reloads_the_units_that_import_it(tmp_path: Path) -> None:
    pa = only(Case(tmp_path, new={**BASE, "pkg/lib.py": "def helper():\n    return 2\n"}).plan())
    assert reloaded(pa) == ["pkg.b"] and pa.reload is not None and pa.reload.changed_modules == ["pkg.lib"]


def test_a_change_to_what_only_a_build_step_imports_reloads_its_unit(tmp_path: Path) -> None:
    """``[unit] build`` may import what the entry does not (the source of what it compiles): a change
    there reloads the unit, and its build step runs (Gemini's review of #55)."""
    built = {"pkg/b/unit.toml": manifest_toml("pkg.b", "pkg.b.unit:Unit", requires=["pkg.a@1"], reads=["pkg.live"],
                                              build="pkg.b.make:build"),
             "pkg/b/make.py": "from pkg.b import source\n\ndef build():\n    pass\n",
             "pkg/b/source.py": "CODE = 'v1'\n"}
    base = {**BASE, **built}
    pa = only(Case(tmp_path, base=base, new={**base, "pkg/b/source.py": "CODE = 'v2'\n"}).plan())
    assert reloaded(pa) == ["pkg.b"] and pa.reload is not None and pa.reload.changed_modules == ["pkg.b.source"]


def test_a_new_unit_is_staged_unless_it_imports_changed_code(tmp_path: Path) -> None:
    c = {"pkg/c/unit.toml": manifest_toml("pkg.c", "pkg.c.unit:Unit"), "pkg/c/unit.py": "class Unit:\n    pass\n"}
    pa = only(Case(tmp_path / "1", new={**BASE, **c}).plan())
    assert [u.id for u in pa.stage] == ["pkg.c"] and pa.reload is None
    uses_lib = {**c, "pkg/c/unit.py": "from pkg import lib\n\nclass Unit:\n    pass\n"}
    pa = only(Case(tmp_path / "2", new={**BASE, **uses_lib, "pkg/lib.py": "def helper():\n    return 2\n"}).plan())
    assert pa.stage == [] and pa.reload is not None and [u.id for u in pa.reload.units_to_add] == ["pkg.c"]


def test_a_new_unit_that_reads_a_key_the_change_sets_is_added_inside_the_reload(tmp_path: Path) -> None:
    """Staged during prepare it would be built under the old value; inside the reload it gets the new one."""
    c = {"pkg/c/unit.toml": manifest_toml("pkg.c", "pkg.c.unit:Unit", requires=["pkg.a@1"], reads=["pkg.live"]),
         "pkg/c/unit.py": "class Unit:\n    pass\n"}
    pa = only(Case(tmp_path / "1", new={**BASE, **c}).plan())
    assert [u.id for u in pa.stage] == ["pkg.c"]
    p = Case(tmp_path / "2", new={**BASE, **c}, config={"pkg.live": 5}).plan()
    pa = only(p)
    assert pa.stage == [] and pa.reload is not None and [u.id for u in pa.reload.units_to_add] == ["pkg.c"]
    assert pa.config == {"pkg.live": 5} and reloaded(pa) == []
    assert not isinstance(p, Failed) and any("pkg.c is added at the safe point: it reads ['pkg.live']" in n for n in p.notes)


SCHED_UNIT = {"pkg/s/unit.toml": manifest_toml("pkg.s", "pkg.s.unit:Unit", process=["sched"], config={
    "pkg.other": {"type": "int", "default": 2, "binding": "live"}}), "pkg/s/unit.py": "class Unit:\n    pass\n"}


def test_a_new_unit_reading_a_key_of_another_role_is_added_once_the_reload_declares_it(tmp_path: Path) -> None:
    """The key was never declared in this process: staged during prepare, the unit would read an
    undeclared key. The reload declares it first, at its owner's default or at the value set."""
    c = {"pkg/c/unit.toml": manifest_toml("pkg.c", "pkg.c.unit:Unit", reads=["pkg.other"]),
         "pkg/c/unit.py": "class Unit:\n    pass\n"}
    for sub, config, value in (("default", {}, 2), ("set", {"pkg.other": 3}, 3)):
        p = Case(tmp_path / sub, base={**BASE, **SCHED_UNIT}, new={**BASE, **SCHED_UNIT, **c}, config=config).plan()
        pa = only(p)
        assert pa.stage == [] and pa.reload is not None and [u.id for u in pa.reload.units_to_add] == ["pkg.c"]
        entry = pa.reload.config["pkg.other"]
        assert (entry.owner, entry.value) == ("pkg.s", value) and "pkg.other" not in pa.config
        assert not isinstance(p, Failed) and any("only the reload declares here" in n for n in p.notes)


LATE = {"pkg/d/unit.toml": manifest_toml("pkg.d", "pkg.d.unit:Unit", provides=["pkg.d@1"], config={
            "pkg.dkey": {"type": "int", "default": 4, "binding": "startup"}}),
        "pkg/d/unit.py": "from pkg import lib\n\nclass Unit:\n    pass\n"}
NEEDS_D = {
    "pkg/c/unit.toml": manifest_toml("pkg.c", "pkg.c.unit:Unit", requires=["pkg.d@1"]),
    "pkg/c/unit.py": "class Unit:\n    pass\n",
    "pkg/e/unit.toml": manifest_toml("pkg.e", "pkg.e.unit:Unit", reads=["pkg.dkey"]),
    "pkg/e/unit.py": "class Unit:\n    pass\n",
    "pkg/f/unit.toml": manifest_toml("pkg.f", "pkg.f.unit:Unit", optional=["pkg.d@1"]),
    "pkg/f/unit.py": "class Unit:\n    pass\n",
    "pkg/g/unit.toml": manifest_toml("pkg.g", "pkg.g.unit:Unit", requires=["pkg.c@1"]),
    "pkg/g/unit.py": "class Unit:\n    pass\n",
}


def test_a_new_unit_that_needs_a_unit_added_at_the_safe_point_waits_for_it(tmp_path: Path) -> None:
    """``pkg.d`` imports changed code, so it is added at the safe point; the new units that use its
    service (required or optional) or read its key wait with it, and so does one that needs those."""
    needs = {**NEEDS_D, "pkg/c/unit.toml": manifest_toml("pkg.c", "pkg.c.unit:Unit", provides=["pkg.c@1"],
                                                          requires=["pkg.d@1"])}
    p = Case(tmp_path, new={**BASE, **LATE, **needs, "pkg/lib.py": "def helper():\n    return 2\n"}).plan()
    pa = only(p)
    assert pa.stage == [] and pa.reload is not None
    assert sorted(u.id for u in pa.reload.units_to_add) == ["pkg.c", "pkg.d", "pkg.e", "pkg.f", "pkg.g"]
    assert not isinstance(p, Failed)
    notes = "\n".join(p.notes)
    for expected in ("pkg.c is added at the safe point: it uses ['pkg.d@1']",
                     "pkg.e is added at the safe point: it reads ['pkg.dkey'], which only the reload declares",
                     "pkg.f is added at the safe point: it uses ['pkg.d@1']",
                     "pkg.g is added at the safe point: it uses ['pkg.c@1']",
                     "pkg.d is added at the safe point: it imports changed code"):
        assert expected in notes, notes


def test_a_new_unit_waits_for_a_service_a_running_unit_provides_only_once_reloaded(tmp_path: Path) -> None:
    """``pkg.a`` runs now without ``pkg.extra@1``; its new manifest adds it, so only the reloaded
    ``pkg.a`` provides it. The new units that use it (required or optional) wait for the reload,
    and so does one that needs them; a new unit using a service ``pkg.a`` provides now is staged."""
    a_extra = manifest_toml("pkg.a", "pkg.a.unit:Unit", provides=["pkg.a@1", "pkg.extra@1"], config={
        "pkg.live": {"type": "int", "default": 1, "binding": "live"},
        "pkg.plan": {"type": "int", "default": 8, "binding": "plan"},
        "pkg.start": {"type": "int", "default": 0, "binding": "startup"},
    })
    users = {
        "pkg/c/unit.toml": manifest_toml("pkg.c", "pkg.c.unit:Unit", provides=["pkg.c@1"], requires=["pkg.extra@1"]),
        "pkg/c/unit.py": "class Unit:\n    pass\n",
        "pkg/f/unit.toml": manifest_toml("pkg.f", "pkg.f.unit:Unit", optional=["pkg.extra@1"]),
        "pkg/f/unit.py": "class Unit:\n    pass\n",
        "pkg/g/unit.toml": manifest_toml("pkg.g", "pkg.g.unit:Unit", requires=["pkg.c@1"]),
        "pkg/g/unit.py": "class Unit:\n    pass\n",
        "pkg/h/unit.toml": manifest_toml("pkg.h", "pkg.h.unit:Unit", requires=["pkg.a@1"]),
        "pkg/h/unit.py": "class Unit:\n    pass\n",
    }
    p = Case(tmp_path, new={**BASE, "pkg/a/unit.toml": a_extra, **users}).plan()
    pa = only(p)
    assert [u.id for u in pa.stage] == ["pkg.h"]
    assert pa.reload is not None and reloaded(pa) == ["pkg.a"]
    assert sorted(u.id for u in pa.reload.units_to_add) == ["pkg.c", "pkg.f", "pkg.g"]
    assert not isinstance(p, Failed)
    notes = "\n".join(p.notes)
    for expected in ("pkg.c is added at the safe point: it uses ['pkg.extra@1'], provided only once the reload ran",
                     "pkg.f is added at the safe point: it uses ['pkg.extra@1']",
                     "pkg.g is added at the safe point: it uses ['pkg.c@1']"):
        assert expected in notes, notes


def test_new_units_that_need_each_other_are_staged_together(tmp_path: Path) -> None:
    """When the unit they need is staged too, its keys and services are ready during prepare."""
    early = {**LATE, "pkg/d/unit.py": "class Unit:\n    pass\n"}
    pa = only(Case(tmp_path, new={**BASE, **early, **{k: v for k, v in NEEDS_D.items() if "/g/" not in k}}).plan())
    assert sorted(u.id for u in pa.stage) == ["pkg.c", "pkg.d", "pkg.e", "pkg.f"] and pa.reload is None


def test_a_removed_unit_is_unloaded(tmp_path: Path) -> None:
    new = {k: v for k, v in BASE.items() if not k.startswith("pkg/b/")}
    pa = only(Case(tmp_path, new=new).plan())
    assert pa.reload is not None and pa.reload.units_to_remove == ["pkg.b"]


def test_a_framework_change_restarts_the_engine(tmp_path: Path) -> None:
    p = Case(tmp_path, base={**BASE, "core/change/queue.py": "# v1\n"},
             new={**BASE, "core/change/queue.py": "# v2\n"}).plan()
    assert not isinstance(p, Failed) and p.restart_engine and p.procs == []


def test_a_native_extension_change_restarts_the_process(tmp_path: Path) -> None:
    b = manifest_toml("pkg.b", "pkg.b.unit:Unit", requires=["pkg.a@1"], reads=["pkg.live"],
                      capabilities='native_extensions = ["pkg/b/_ext.so"]')
    base = {**BASE, "pkg/b/unit.toml": b, "pkg/b/_ext.so": "v1"}
    pa = only(Case(tmp_path, base=base, new={**base, "pkg/b/_ext.so": "v2"}).plan())
    assert pa.restart is not None and sorted(u.id for u in pa.restart.units) == ["pkg.a", "pkg.b"]


def test_a_mesh_change_restarts_the_ranks_and_starts_new_ones(tmp_path: Path) -> None:
    p = Case(tmp_path, topo=topology(2)).plan()
    assert not isinstance(p, Failed)
    assert [pa.proc for pa in p.procs if pa.restart is not None] == [W0]
    assert [s.proc for s in p.spawn] == [W1] and p.topology == topology(2)


def test_a_platform_change_restarts_the_role_processes(tmp_path: Path) -> None:
    """A platform is chosen when the process starts (``make_platform``): a role moving to another
    platform restarts its processes, and a topology naming no known platform is refused."""
    base = topology()
    eager = replace(base, roles=[replace(r, platform="torch_eager") for r in base.roles])
    p = Case(tmp_path / "1", topo=eager).plan()
    assert not isinstance(p, Failed)
    assert [pa.proc for pa in p.procs if pa.restart is not None] == [W0]
    assert any("its platform changed fake {} -> torch_eager {}" in n for n in p.notes)
    assert [d.platform for d in desired_procs(eager).values()] == ["torch_eager"]
    cuda = replace(eager, roles=[replace(r, platform_options={"cuda": 1}) for r in eager.roles])
    assert [pa.proc for pa in only_restarts(Case(tmp_path / "2", topo=cuda).plan())] == [W0]
    bad = replace(base, roles=[replace(r, platform="tpu") for r in base.roles])
    assert bad.check() == ["role worker names an unknown platform 'tpu'; known: ['fake', 'torch_eager', 'cuda_graph']"]


def only_restarts(p: ActionPlan | Failed) -> list[ProcActions]:
    assert not isinstance(p, Failed), p
    return [pa for pa in p.procs if pa.restart is not None]


def test_state_requests_become_the_state_units_own_actions(tmp_path: Path) -> None:
    s = {**BASE, "pkg/s/unit.toml": manifest_toml("pkg.s", "pkg.s.unit:Unit", kind="state",
                                                  capabilities='actions = ["relocate"]'),
         "pkg/s/unit.py": "class Unit:\n    pass\n"}
    units = ("pkg.a", "pkg.b", "pkg.s")
    pa = only(Case(tmp_path / "1", base=s, new=s, units=units,
                   state=[StateRequest(UnitId("pkg.s"), "relocate", {"to": [1]})]).plan())
    assert pa.actions == [UnitAction(UnitId("pkg.s"), "relocate", {"to": [1]})] and pa.reload is None
    bad = Case(tmp_path / "2", base=s, new=s, units=units, state=[StateRequest(UnitId("pkg.s"), "refit")]).plan()
    assert isinstance(bad, Failed) and bad.diagnostics[0].code == "deps.bad_state_request"


def test_a_unit_names_its_own_actions(tmp_path: Path) -> None:
    """The framework keeps no list of action names (modules.html P4): a unit may offer one no other
    unit has, without a change to the framework core. Only the form of the name is checked.

    A logic unit may declare actions too: the scheduler logic holds the requests and declares
    ``drain`` (state-changes.md §5.1). A ``StateRequest`` still reaches only state units."""
    s = {**BASE, "pkg/s/unit.toml": manifest_toml("pkg.s", "pkg.s.unit:Unit", kind="state",
                                                  capabilities='actions = ["compact", "grow"]'),
         "pkg/s/unit.py": "class Unit:\n    pass\n",
         "pkg/q/unit.toml": manifest_toml("pkg.q", "pkg.q.unit:Unit", capabilities='actions = ["drain"]'),
         "pkg/q/unit.py": "class Unit:\n    pass\n"}
    units = ("pkg.a", "pkg.b", "pkg.s", "pkg.q")
    pa = only(Case(tmp_path / "1", base=s, new=s, units=units,
                   state=[StateRequest(UnitId("pkg.s"), "compact", {"below": 0.5})]).plan())
    assert pa.actions == [UnitAction(UnitId("pkg.s"), "compact", {"below": 0.5})] and pa.reload is None
    logic = Case(tmp_path / "2", base=s, new=s, units=units, state=[StateRequest(UnitId("pkg.q"), "drain")]).plan()
    assert isinstance(logic, Failed) and logic.diagnostics[0].code == "deps.bad_state_request"
    assert "not a state unit" in logic.diagnostics[0].reason
    for name in ["Compact", "", "compact now", "kv.compact", "compact\n"]:
        bad = manifest_toml("pkg.s", "pkg.s.unit:Unit", kind="state", capabilities=f"actions = [{json.dumps(name)}]")
        r = parse_manifest(bad)
        assert isinstance(r, Failed) and r.diagnostics[0].reason.startswith(f"bad action name {name!r}"), name


def test_names_in_a_manifest_are_checked_in_full(tmp_path: Path) -> None:
    """``$`` also matches before a trailing newline: a name that ends in one is refused, not trimmed."""
    # "\\n" in TOML text is the escape for a newline in the value.
    cases = {"unit id": manifest_toml("pkg.s\\n", "pkg.s.unit:Unit"),
             "entry": manifest_toml("pkg.s", "pkg.s.unit:Unit\\n"),
             "service": manifest_toml("pkg.s", "pkg.s.unit:Unit", provides=["pkg.s@1\\n"])}
    for what, text in cases.items():
        r = parse_manifest(text)
        assert isinstance(r, Failed) and r.has_code("manifest.invalid") and "\\n" in r.diagnostics[0].reason, what
    assert not isinstance(parse_manifest(manifest_toml("pkg.s", "pkg.s.unit:Unit", provides=["pkg.s@1"])), Failed)


def test_the_same_inputs_give_the_same_plan(tmp_path: Path) -> None:
    """Why ``--dry-run`` can promise the plan that runs: the planner is a function of its inputs."""
    a = Case(tmp_path / "1", config={"pkg.plan": 16, "pkg.live": 2}).plan()
    b = Case(tmp_path / "2", config={"pkg.plan": 16, "pkg.live": 2}).plan()
    assert a == b
