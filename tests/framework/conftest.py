from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.framework.support import write_tree

_counter = itertools.count()


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
