"""An engine source tree as the control plane sees it: manifests and modules, read
from files without importing anything. A library shared by ``core.manifest`` and
``core.deps`` (it is not a unit: no ``unit.toml``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.common.ids import UnitId
from core.common.manifest import MANIFEST_FILE, UnitManifest, load_manifest
from core.common.result import Diagnostic, Failed, diag
from core.supervisor.topology import Topology
from unit.runtime.imports import scan_root

COMPONENT = "core.sourcetree"


@dataclass
class UnitSource:
    manifest: UnitManifest
    directory: Path
    """Where its ``unit.toml`` is; the modules under it are the unit's code."""
    modules: list[str] = field(default_factory=list[str])


@dataclass
class Tree:
    root: Path
    units: dict[UnitId, UnitSource]
    modules: dict[str, Path]
    owner: dict[str, UnitId]
    """Module -> the unit whose directory holds it (modules outside unit directories are libraries)."""
    errors: list[Diagnostic]

    def role_units(self, topology: Topology | None, role: str) -> list[UnitId]:
        """A role's units: listed in the topology, or else every unit whose ``process`` names the role."""
        if topology is not None:
            for r in topology.roles:
                if r.name == role and r.units:
                    return list(r.units)
        return sorted(u for u, s in self.units.items() if role in s.manifest.unit.process)


def read_tree(root: Path) -> Tree:
    """Manifests and modules of a code root (the ``engine/`` directory of a snapshot)."""
    errors: list[Diagnostic] = []
    units: dict[UnitId, UnitSource] = {}
    for path in sorted(root.rglob(MANIFEST_FILE)):
        r = load_manifest(path)
        if isinstance(r, Failed):
            errors += r.diagnostics
            continue
        m = r.value
        if m.id in units:
            errors.append(diag("core.manifest", "manifest", f"declared again in {path.parent.relative_to(root)}",
                               subject=m.id, code="manifest.duplicate_unit"))
            continue
        units[m.id] = UnitSource(m, path.parent)
    modules = scan_root(root)
    owner: dict[str, UnitId] = {}
    by_depth = sorted(units.values(), key=lambda s: len(s.directory.parts), reverse=True)
    for name, file in modules.items():
        for s in by_depth:
            if file.is_relative_to(s.directory):
                owner[name] = s.manifest.id
                s.modules.append(name)
                break
    return Tree(root, units, modules, owner, errors)


def module_of(path: str) -> str | None:
    """``"engine/a/b.py"`` -> ``"a.b"``; ``"engine/a/__init__.py"`` -> ``"a"``; other files -> ``None``."""
    if not path.startswith("engine/") or not path.endswith(".py"):
        return None
    parts = path[len("engine/"):-3].split("/")
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) if parts else None
