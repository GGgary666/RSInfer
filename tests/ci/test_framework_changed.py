"""The CI's scope check (.github/scripts/framework_changed.py): which changes run tests/framework."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from core.common.framework import FrameworkModules

REPO = Path(__file__).resolve().parents[2]
FRAMEWORK = FrameworkModules.load(REPO / "engine")


def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("framework_changed", REPO / ".github/scripts/framework_changed.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("paths", "framework"), [
    (["engine/frontend/http/api.py", "tests/units/frontend/test_serving.py"], False),
    (["engine/core/contracts/requests.py", "doc/specs/ipc.md", "AGENTS.md"], False),
    (["engine/core/deps/planner.py"], False),  # a control-plane unit, not the framework
    (["engine/core/ipc/channels.py"], True),  # a comment there is a framework change too
    (["engine/unit/runtime/host.py"], True),
    (["engine/core/__init__.py"], True),
    (["engine/core/ipc/README.md"], True),
    (["engine/framework.toml"], True),
    (["tests/framework/ipc/test_channels.py"], True),
    (["tests/conftest.py"], True),
    (["uv.lock"], True),
    ([".github/workflows/ci.yml"], True),
])
def test_what_counts_as_touching_the_framework(paths: list[str], framework: bool) -> None:
    assert script().touches_framework(paths, FRAMEWORK) is framework


def test_moving_a_file_out_of_the_framework_touches_it(tmp_path: Path) -> None:
    """Codex on #28: git reports a move as its new path only, unless told otherwise."""
    def git(*args: str) -> None:
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=tmp_path, check=True,
                       capture_output=True)

    (tmp_path / "engine/core/ipc").mkdir(parents=True)
    (tmp_path / "engine/core/ipc/helper.py").write_text("X = 1\n")
    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True,
                          text=True).stdout.strip()
    (tmp_path / "engine/helpers").mkdir()
    git("mv", "engine/core/ipc/helper.py", "engine/helpers/helper.py")
    git("commit", "-q", "-m", "move")
    changed = script().changed(base, tmp_path)
    assert sorted(changed) == ["engine/core/ipc/helper.py", "engine/helpers/helper.py"]
    assert script().touches_framework(changed, FRAMEWORK)
