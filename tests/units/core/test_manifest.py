"""core.manifest (doc/specs/control-plane.md §3): static checks of a source tree, one rule at a time.

Also platform.md §4 test 4 (only plan units record graphs) and context.md §6 test 5 (reads in the
code against the manifest), which needed this unit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core.common.framework import FrameworkModules
from core.common.ids import HostId, UnitId
from core.manifest.checks import check_tree
from core.supervisor import HostSpec, Mesh, RankPlacement, RoleSpec, Topology
from tests.framework.support import manifest_toml, write_tree

TOPO = Topology([HostSpec(HostId("h"), "addr", "/snapshots")], Mesh({"tp": 1}), [RankPlacement(0, HostId("h"), 0)],
                [RoleSpec("worker", [], per_rank=True)])

PROVIDER = '''
class Unit:
    def __init__(self, ctx, config):
        self.k = config["pkg.k"]
        ctx.provide("pkg.a@1", self)
'''
USER = '''
class Unit:
    def __init__(self, ctx, config):
        self.k = config["pkg.k"]
        self.a = ctx.use("pkg.a@1")
'''


def good() -> dict[str, str]:
    return {
        "pkg/a/unit.toml": manifest_toml("pkg.a", "pkg.a.unit:Unit", provides=["pkg.a@1"],
                                         config={"pkg.k": {"type": "int", "default": 1}}),
        "pkg/a/unit.py": PROVIDER,
        "pkg/b/unit.toml": manifest_toml("pkg.b", "pkg.b.unit:Unit", requires=["pkg.a@1"], reads=["pkg.k"]),
        "pkg/b/unit.py": USER,
        "pkg/lib.py": "def helper():\n    return 1\n",
    }


def check(tmp: Path, files: dict[str, str], topology: Topology | None = TOPO) -> tuple[set[str], set[str]]:
    root = write_tree(tmp / "engine", files)
    errors, warnings = check_tree(root, topology, FrameworkModules.default())
    return {d.code for d in errors}, {d.code for d in warnings}


def test_a_well_formed_tree_passes(tmp_path: Path) -> None:
    assert check(tmp_path, good()) == (set(), set())


def change(**edits: Any) -> dict[str, str]:
    return {**good(), **edits}


CASES: list[tuple[str, dict[str, str], str]] = [
    ("entry module missing", change(**{"pkg/b/unit.toml": manifest_toml(
        "pkg.b", "pkg.b.gone:Unit", requires=["pkg.a@1"], reads=["pkg.k"])}), "manifest.missing_entry"),
    ("unit declared twice", change(**{"pkg/c/unit.toml": manifest_toml("pkg.a", "pkg.c.unit:Unit"),
                                      "pkg/c/unit.py": "class Unit:\n    pass\n"}), "manifest.duplicate_unit"),
    ("two owners of a key", change(**{"pkg/b/unit.toml": manifest_toml(
        "pkg.b", "pkg.b.unit:Unit", requires=["pkg.a@1"], config={"pkg.k": {"type": "int", "default": 2}})}),
     "manifest.config_owner"),
    ("reads a key nobody declares", change(**{"pkg/b/unit.toml": manifest_toml(
        "pkg.b", "pkg.b.unit:Unit", requires=["pkg.a@1"], reads=["pkg.k", "pkg.nobody"])}), "manifest.undeclared_read"),
    ("code reads an undeclared key", change(**{"pkg/b/unit.py": USER + "        self.x = config['pkg.other']\n"}),
     "manifest.read_not_declared"),
    ("required service nobody provides", change(**{"pkg/b/unit.toml": manifest_toml(
        "pkg.b", "pkg.b.unit:Unit", requires=["pkg.a@1", "pkg.z@1"], reads=["pkg.k"])}), "manifest.unprovided_service"),
    ("imports another unit's code", change(**{"pkg/b/unit.py": "from pkg.a.unit import Unit as A\n" + USER}),
     "manifest.cross_unit_import"),
    ("module-level mutable", change(**{"pkg/b/unit.py": "CACHE = {}\n" + USER}), "manifest.mutable_global"),
    ("module-level mutable in a library", change(**{"pkg/lib.py": "seen = list()\n"}), "manifest.mutable_global"),
    ("global statement", change(**{"pkg/lib.py": "N = 0\ndef bump():\n    global N\n    N += 1\n"}),
     "manifest.mutable_global"),
    ("graph outside a plan unit", change(**{"pkg/b/unit.py": USER + "        ctx.platform.graph()\n"}),
     "manifest.graph_outside_plan"),
    ("undeclared native extension", change(**{"pkg/b/_ext.so": "binary"}), "manifest.undeclared_native"),
    ("code peeks at an undeclared key", change(**{"pkg/b/unit.py": USER + "        config.peek('unknown.value')\n"}),
     "manifest.read_not_declared"),
    ("code that does not parse", change(**{"pkg/b/unit.py": USER + "        self.x = (\n"}), "manifest.syntax_error"),
]


@pytest.mark.parametrize(("name", "files", "code"), CASES, ids=[c[0] for c in CASES])
def test_each_rule(tmp_path: Path, name: str, files: dict[str, str], code: str) -> None:
    errors, _ = check(tmp_path, files)
    assert code in errors, (name, errors)


def test_declared_reads_the_code_never_does_are_reported(tmp_path: Path) -> None:
    """context.md §6 test 5, the other half: declared but never read is a warning (the read may be dynamic)."""
    no_read = USER.replace('        self.k = config["pkg.k"]\n', "")
    errors, warnings = check(tmp_path, change(**{"pkg/b/unit.py": no_read}))
    assert errors == set() and warnings == {"manifest.unused_read"}


def test_a_plan_unit_may_record_graphs(tmp_path: Path) -> None:
    files = change(**{
        "pkg/b/unit.toml": manifest_toml("pkg.b", "pkg.b.unit:Unit", requires=["pkg.a@1"], reads=["pkg.k"],
                                         capabilities="captures_graph = true"),
        "pkg/b/unit.py": USER + "        ctx.platform.graph()\n",
    })
    assert check(tmp_path, files) == (set(), set())


def test_a_role_listing_a_unit_for_another_role_is_refused(tmp_path: Path) -> None:
    topo = Topology(TOPO.hosts, TOPO.mesh, TOPO.placement, [RoleSpec("sched", [UnitId("pkg.a")], hosts=[HostId("h")])])
    errors, _ = check(tmp_path, good(), topo)
    assert "manifest.wrong_role" in errors


def test_a_declared_key_read_with_peek_is_used(tmp_path: Path) -> None:
    """``config.peek`` reads a key without recording the read: the key still counts as read."""
    peeks = USER.replace('config["pkg.k"]', 'config.peek("pkg.k")')
    assert check(tmp_path, change(**{"pkg/b/unit.py": peeks})) == (set(), set())


def test_code_that_does_not_parse_is_reported_with_its_line_and_checked_no_further(tmp_path: Path) -> None:
    """A syntax error is a static check failure, not a crash of the checks. The checks that read code
    skip that module: its imports and reads are unknown, so it gets no ``unused_read`` warning."""
    broken = 'class Unit:\n    def __init__(self, ctx, config):\n        self.a = ctx.use("pkg.a@1"\n'
    root = write_tree(tmp_path / "engine", change(**{"pkg/b/unit.py": broken, "pkg/lib.py": "x = [\n"}))
    errors, warnings = check_tree(root, TOPO, FrameworkModules.default())
    assert [(d.code, d.subject) for d in errors] == [("manifest.syntax_error", "pkg.b"),
                                                     ("manifest.syntax_error", "pkg.lib")]
    assert "pkg/b/unit.py, line 3" in errors[0].reason and "pkg/lib.py, line 1" in errors[1].reason
    assert warnings == []
