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


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()


def commit_file(root: Path, name: str) -> str:
    file = root / name
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("X = 1\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", name)
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, ModuleType]:
    """Run the actual CLI entry point against a small Git history and the real framework list."""
    git(tmp_path, "init", "-q")
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine/framework.toml").write_bytes((REPO / "engine/framework.toml").read_bytes())
    commit_file(tmp_path, "engine/frontend/initial.py")
    module = script()
    monkeypatch.setattr(module, "__file__", str(tmp_path / ".github/scripts/framework_changed.py"))
    return tmp_path, module


@pytest.mark.parametrize(("path", "expected"), [("engine/frontend/http/api.py", "false"),
                                                ("engine/core/ipc/channels.py", "true")])
def test_master_push_checks_changes_since_before(repo: tuple[Path, ModuleType], path: str, expected: str,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    root, module = repo
    # A framework change from an earlier push must not force a later unit-only push to run it again.
    before = commit_file(root, "engine/core/context/earlier.py")
    commit_file(root, path)
    assert module.main(["framework_changed.py", before, "--event", "push"]) == 0
    assert capsys.readouterr().out.strip() == expected


def test_force_push_reverting_a_framework_change_runs_framework_tests(repo: tuple[Path, ModuleType],
                                                                    capsys: pytest.CaptureFixture[str]) -> None:
    root, module = repo
    target = git(root, "rev-parse", "HEAD")
    before = commit_file(root, "engine/core/ipc/channels.py")
    git(root, "checkout", "--detach", target)
    assert module.main(["framework_changed.py", before, "--event", "push"]) == 0
    assert capsys.readouterr().out.strip() == "true"


def test_pr_excludes_changes_only_on_its_target_branch(repo: tuple[Path, ModuleType],
                                                      capsys: pytest.CaptureFixture[str]) -> None:
    root, module = repo
    fork = git(root, "rev-parse", "HEAD")
    target = commit_file(root, "engine/core/ipc/channels.py")
    git(root, "checkout", "-b", "feature", fork)
    commit_file(root, "engine/frontend/http/api.py")
    assert module.main(["framework_changed.py", target, "--event", "pull_request"]) == 0
    assert capsys.readouterr().out.strip() == "false"


@pytest.mark.parametrize("before", ["", "0" * 40])
def test_a_push_without_a_previous_commit_runs_every_test(before: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert script().main(["framework_changed.py", before, "--event", "push"]) == 0
    assert capsys.readouterr().out.strip() == "true"
