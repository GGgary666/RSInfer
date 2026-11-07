"""A broken checker invocation must never count as a killed mutant."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parent
module_spec = importlib.util.spec_from_file_location("formal_runner", ROOT / "run.py")
assert module_spec is not None and module_spec.loader is not None
runner = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(runner)


@pytest.mark.parametrize("failure", ["syntax", "other-invariant", "no-violation", "timeout"])
def test_bad_tlc_run_cannot_pass_as_an_expected_counterexample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    case = next(c for c in runner.cases() if c["name"] == "mutant-ForgetAbort")
    dest = tmp_path / case["name"]
    dest.mkdir()
    (dest / "trace.json").write_text('{"stale": true}')
    (dest / "result.json").write_text('{"outcome": "counterexample"}')

    def invoke(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired("java", 1, output=b"incomplete")
        if failure == "syntax":
            return subprocess.CompletedProcess([], 150, "Parsing failed")
        if failure == "no-violation":
            return subprocess.CompletedProcess([], 0, "Model checking completed. No error has been found.")
        return subprocess.CompletedProcess([], 12, 'Invariant WrongProperty is violated.\n'
                                           'State 1: initial\n/\\ value = 0\nState 2: next\n/\\ value = 1\n')

    monkeypatch.setattr(runner.subprocess, "run", invoke)
    with pytest.raises(RuntimeError):
        runner.check(case, Path("unused.jar"), "unused-java", 1, tmp_path)
    assert not (dest / "result.json").exists()


def test_tool_checksum_is_enforced(tmp_path: Path) -> None:
    jar = tmp_path / "untrusted.jar"
    jar.write_bytes(b"wrong tool")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        runner.tool(jar, False)


def test_unknown_case_fails_before_running_any_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner.sys, "argv", ["run.py", "--case", "misspelled-case"])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
