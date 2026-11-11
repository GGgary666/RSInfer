"""For every GPU test: a third-party module a test imported stays imported for the tests after
it. The tests' import isolation (tests/conftest.py) forgets every module a test imported, so the
next test's kernel units would import their libraries again, and those do not survive it:
FlashInfer's CuTe DSL dependency is a native extension that aborts the process when it is
imported twice, and a submodule imported again brings new classes that the modules still loaded
do not recognize (the DSL then fails to compile its next kernel). An engine process imports
them once: a reload drops the units' own modules only."""

from __future__ import annotations

import sys
from collections.abc import Generator
from pathlib import Path
from types import ModuleType

import pytest

OURS = frozenset({p.name for p in (Path(__file__).resolve().parents[2] / "engine").iterdir() if p.is_dir()}
                 | {"tests", "deploy"})
"""The top-level packages of this repository: the modules the import isolation is there for."""
_LIBRARIES: dict[str, ModuleType] = {}
"""Every other module the tests so far imported (native submodules have no file to tell them by)."""


@pytest.fixture(autouse=True)
def _keep_libraries() -> Generator[None]:
    """Set up after the import isolation and torn down before it: what it is about to forget is
    put back for the next test."""
    for name, module in _LIBRARIES.items():
        sys.modules.setdefault(name, module)
    yield
    _LIBRARIES.update({name: module for name, module in sys.modules.items() if name.split(".")[0] not in OURS})
