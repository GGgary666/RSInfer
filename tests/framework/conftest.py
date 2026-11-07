from __future__ import annotations

import itertools
import sys
from collections.abc import Callable, Generator
from pathlib import Path

import pytest

from tests.framework.support import events, write_tree

_counter = itertools.count()


@pytest.fixture(autouse=True)
def _isolate_imports() -> Generator[None]:
    """Each test gets back the sys.path and sys.modules it started with."""
    path = list(sys.path)
    modules = set(sys.modules)
    events.clear()
    yield
    sys.path[:] = path
    for name in set(sys.modules) - modules:
        del sys.modules[name]


@pytest.fixture
def pkg() -> str:
    """A top-level package name no other test uses (sys.modules is process-wide)."""
    return f"toy{next(_counter)}"


@pytest.fixture
def snapshot(tmp_path: Path) -> Callable[[str, dict[str, str]], Path]:
    """``snapshot("v1", files)`` creates an immutable-by-convention code root."""

    def make(name: str, files: dict[str, str]) -> Path:
        return write_tree(tmp_path / name / "engine", files)

    return make
