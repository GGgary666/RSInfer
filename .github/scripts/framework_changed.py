"""Whether a change touches the framework core, so CI runs ``tests/framework`` (AGENTS.md).

Usage: ``PYTHONPATH=engine python .github/scripts/framework_changed.py BASE --event EVENT``
prints ``true`` or ``false``. For a pull_request (the default), compare its merge base to HEAD;
for a push, compare the before commit to HEAD, including reversions in a force push.
A missing or all-zero base means the whole tree needs checking.

The framework is what ``engine/framework.toml`` lists. Besides its modules, the framework's
tests, the shared test setup, the dependencies and CI itself count: each can break the
framework tests without touching a unit.
"""

from __future__ import annotations

import argparse
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


def changed(base: str, root: Path, *, merge_base: bool = True) -> list[str]:
    """Compare the PR merge base, or a push's exact before commit, to HEAD. A move is its old path and its new
    one (``--no-renames``): moving a file out of the framework changes the framework too."""
    span = f"{base}...HEAD" if merge_base else f"{base}..HEAD"
    out = subprocess.run(["git", "diff", "--name-only", "--no-renames", span], cwd=root, check=True,
                         capture_output=True, text=True).stdout
    return out.split()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", nargs="?", default="")
    parser.add_argument("--event", choices=("pull_request", "push"), default="pull_request")
    args = parser.parse_args(argv[1:])
    if not args.base or args.base == "0" * 40:
        print("true")
        return 0
    root = Path(__file__).resolve().parents[2]
    paths = changed(args.base, root, merge_base=args.event == "pull_request")
    print("true" if touches_framework(paths, FrameworkModules.load(root / "engine")) else "false")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
