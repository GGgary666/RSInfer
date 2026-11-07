"""core.deps (doc/specs/control-plane.md §4): one rule of the action ladder at a time, on small trees.

Also context.md §6 tests 2 (live key: in place with a hook, reload without), 3 (plan key: rebuild
what was recorded from it) and 4 (startup key: reload only its readers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.change.blobs import content_hash
from core.change.state import IdentityManifest
from core.change.types import ActionPlan, ChangeSet, ProcActions, StateRequest, UnitAction
from core.common.framework import FrameworkModules
from core.common.ids import HostId, Identity, ProcId, UnitId
from core.common.result import Failed
from core.deps.planner import World, plan
from core.sourcetree import read_tree
from core.supervisor import HostSpec, Mesh, RankPlacement, RoleSpec, Topology
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

    def plan(self) -> ActionPlan | Failed:
        base_root = write_tree(self.tmp / "base" / "engine", self.base)
        new_root = write_tree(self.tmp / "new" / "engine", self.new)
        base = IdentityManifest(files=_hashes(base_root), topology=topology())
        new = IdentityManifest(files=_hashes(new_root), config=self.config, topology=self.topo)
        cs = ChangeSet(base=Identity("b"), config=self.config, topology=self.topo if self.topo != topology() else None,
                       state=self.state)
        intro = ProcIntrospection(
            W0, Identity("b"), [UnitBrief(u, InstanceState.ACTIVE, None, 1) for u in self.units],
            reads=[ReadRecord("pkg.live", UnitId("pkg.b"), "runtime", hooked=self.hooked),
                   ReadRecord("pkg.start", UnitId("pkg.a"), "startup"),
                   ReadRecord("pkg.plan", UnitId("pkg.a"), "plan_build", artifact="g")],
            artifacts=[DerivedArtifact(UnitId("pkg.a"), "g", "pkg.rebuild", ["pkg.plan"], [])],
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


def test_a_library_change_reloads_the_units_that_import_it(tmp_path: Path) -> None:
    pa = only(Case(tmp_path, new={**BASE, "pkg/lib.py": "def helper():\n    return 2\n"}).plan())
    assert reloaded(pa) == ["pkg.b"] and pa.reload is not None and pa.reload.changed_modules == ["pkg.lib"]


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
    assert not isinstance(p, Failed) and any("pkg.c reads ['pkg.live']" in n for n in p.notes)


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


def test_state_requests_become_the_state_units_own_actions(tmp_path: Path) -> None:
    s = {**BASE, "pkg/s/unit.toml": manifest_toml("pkg.s", "pkg.s.unit:Unit", kind="state",
                                                  capabilities='actions = ["migrate"]'),
         "pkg/s/unit.py": "class Unit:\n    pass\n"}
    units = ("pkg.a", "pkg.b", "pkg.s")
    pa = only(Case(tmp_path / "1", base=s, new=s, units=units,
                   state=[StateRequest(UnitId("pkg.s"), "migrate", {"to": [1]})]).plan())
    assert pa.actions == [UnitAction(UnitId("pkg.s"), "migrate", {"to": [1]})] and pa.reload is None
    bad = Case(tmp_path / "2", base=s, new=s, units=units, state=[StateRequest(UnitId("pkg.s"), "refit")]).plan()
    assert isinstance(bad, Failed) and bad.diagnostics[0].code == "deps.bad_state_request"


def test_the_same_inputs_give_the_same_plan(tmp_path: Path) -> None:
    """Why ``--dry-run`` can promise the plan that runs: the planner is a function of its inputs."""
    a = Case(tmp_path / "1", config={"pkg.plan": 16, "pkg.live": 2}).plan()
    b = Case(tmp_path / "2", config={"pkg.plan": 16, "pkg.live": 2}).plan()
    assert a == b
