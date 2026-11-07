"""Static import graph of engine modules, read from source with ``ast``.

Python has no equivalent of Node's ``ModuleJob.linked`` that Cordis HMR walks,
so the graph comes from the source files themselves. Imports anywhere in a file
count (including inside functions), and a module depends on its parent
packages, whose ``__init__`` runs first: both err on the side of reloading more.
"""

from __future__ import annotations

import ast
import functools
from collections import deque
from collections.abc import Iterable, Mapping
from pathlib import Path


def resolve_relative(module: str, is_package: bool, level: int, target: str | None) -> str:
    package = module if is_package else module.rpartition(".")[0]
    parts = package.split(".") if package else []
    if level > 1:
        parts = parts[: len(parts) - (level - 1)]
    base = ".".join(parts)
    if target:
        base = f"{base}.{target}" if base else target
    return base


@functools.lru_cache(maxsize=4096)
def _imports_cached(path: str, mtime_ns: int, size: int, module: str, is_package: bool) -> frozenset[str]:
    del mtime_ns, size  # cache key only
    tree = ast.parse(Path(path).read_bytes(), filename=path)
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = resolve_relative(module, is_package, node.level, node.module)
            else:
                base = node.module or ""
            if base:
                out.add(base)
            for alias in node.names:
                if alias.name != "*":
                    out.add(f"{base}.{alias.name}" if base else alias.name)
    return frozenset(out)


def imports_of(path: Path, module: str, is_package: bool) -> frozenset[str]:
    """Every module name ``path`` may import (some may be attributes, not modules). Raises
    ``SyntaxError`` if the file does not parse: callers reading candidate code check syntax
    first (``core.manifest`` reports it as ``manifest.syntax_error``)."""
    st = path.stat()
    return _imports_cached(str(path), st.st_mtime_ns, st.st_size, module, is_package)


def parents(module: str) -> list[str]:
    parts = module.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts))]


def scan_root(root: Path) -> dict[str, Path]:
    """All importable ``.py`` modules under ``root`` (packages need ``__init__.py``)."""
    out: dict[str, Path] = {}

    def walk(directory: Path, prefix: str) -> None:
        for child in sorted(directory.iterdir()):
            if child.name.startswith((".", "__pycache__")):
                continue
            if child.is_dir():
                init = child / "__init__.py"
                if init.is_file() and child.name.isidentifier():
                    name = f"{prefix}{child.name}"
                    out[name] = init
                    walk(child, f"{name}.")
            elif child.suffix == ".py" and child.stem != "__init__" and child.stem.isidentifier():
                out[f"{prefix}{child.stem}"] = child

    walk(root, "")
    return out


class ImportGraph:
    """Edges from a module to the modules (within the given set) it imports."""

    def __init__(self, modules: Mapping[str, Path]) -> None:
        self.modules = dict(modules)
        self.edges: dict[str, set[str]] = {}
        for name, path in self.modules.items():
            is_package = path.name == "__init__.py"
            targets = set(imports_of(path, name, is_package)) | set(parents(name))
            self.edges[name] = {t for t in targets if t in self.modules and t != name}

    def importers_closure(self, changed: Iterable[str]) -> set[str]:
        """``changed`` plus every module that imports one of them, transitively."""
        reverse: dict[str, set[str]] = {}
        for src, targets in self.edges.items():
            for t in targets:
                reverse.setdefault(t, set()).add(src)
        seen = {c for c in changed}
        queue = deque(seen)
        while queue:
            for importer in reverse.get(queue.popleft(), ()):
                if importer not in seen:
                    seen.add(importer)
                    queue.append(importer)
        return seen

    def imports_closure(self, start: Iterable[str]) -> set[str]:
        """``start`` plus everything it imports, transitively."""
        seen = {s for s in start if s in self.modules}
        queue = deque(seen)
        while queue:
            for target in self.edges.get(queue.popleft(), ()):
                if target not in seen:
                    seen.add(target)
                    queue.append(target)
        return seen
