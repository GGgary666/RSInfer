from __future__ import annotations

import sys
from collections.abc import Generator

import pytest

from tests.framework.support import events


@pytest.fixture(autouse=True)
def _isolate_imports() -> Generator[None]:
    """Each test gets back the sys.path, sys.modules and package search paths it started with:
    units are imported into the test process, and a control plane run in it points packages at
    its snapshots."""
    path = list(sys.path)
    modules = set(sys.modules)
    package_paths = {name: list(p) for name, m in sys.modules.items()
                     if isinstance(p := getattr(m, "__path__", None), list)}
    events.clear()
    yield
    sys.path[:] = path
    for name in set(sys.modules) - modules:
        del sys.modules[name]
    for name, p in package_paths.items():
        m = sys.modules.get(name)
        if m is not None and isinstance(getattr(m, "__path__", None), list):
            m.__path__[:] = p  # type: ignore[index]
