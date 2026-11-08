"""Whether a change touches the framework core, so CI runs ``tests/framework`` (AGENTS.md).

Usage: ``PYTHONPATH=engine python .github/scripts/framework_changed.py BASE`` prints ``true`` or
``false`` for the change from the merge base of BASE to HEAD. Without BASE (a push), ``true``.

The framework is what ``engine/framework.toml`` lists. Besides its modules, the framework's
tests, the shared test setup, the dependencies and CI itself count: each can break the
framework tests without touching a unit.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from core.common.framework import FrameworkModules
from core.sourcetree import module_of

ALWAYS = ("tests/framework/", "tests/conftest.py", "tests/__init__.py", "engine/framework.toml",
          "pyproject.toml", "uv.lock", ".github/")


def touches_framework(paths: Iterable[str], framework: FrameworkModules) -> bool:
    for path in paths:
        if path.startswith(ALWAYS):
            return True
        if path.startswith("engine/"):
            # A module, or for another file (a README) the package whose directory holds it.
            module = module_of(path) or ".".join(Path(path[len("engine/"):]).parent.parts)
            if module and framework.contains(module):
                return True
    return False


def main(argv: list[str]) -> int:
    if len(argv) < 2 or not argv[1]:
        print("true")
        return 0
    root = Path(__file__).resolve().parents[2]
    changed = subprocess.run(["git", "diff", "--name-only", f"{argv[1]}...HEAD"], cwd=root, check=True,
                             capture_output=True, text=True).stdout.split()
    print("true" if touches_framework(changed, FrameworkModules.load(root / "engine")) else "false")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
