"""How ``kernels.cuda`` builds: under a lock of its own (``ops.only_builder``), which a builder
killed mid-way does not leave behind, unlike PyTorch's (2026-11-12, Codex's review of #55: an
undone build step left PyTorch's ``lock`` and every later build of the source waited on it)."""

from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from kernels.cuda.ops import only_builder

REPO = Path(__file__).resolve().parents[3]

KILLED_BUILDER = """
import sys, time
from pathlib import Path
from torch.utils.file_baton import FileBaton
from kernels.cuda.ops import only_builder

with only_builder(Path(sys.argv[1])):
    assert FileBaton(str(Path(sys.argv[1]) / "lock")).try_acquire()  # as cpp_extension.load does
    print("building", flush=True)
    time.sleep(60)
"""


def test_a_builder_killed_mid_way_leaves_no_lock_the_next_one_waits_on(tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": f"{REPO / 'engine'}{os.pathsep}{REPO}"}
    builder = subprocess.Popen([sys.executable, "-c", KILLED_BUILDER, str(tmp_path)], env=env, stdout=subprocess.PIPE,
                               text=True)
    try:
        assert builder.stdout is not None and builder.stdout.readline() == "building\n"
        with open(tmp_path / ".rsinfer-build.lock", "a") as other, pytest.raises(BlockingIOError):
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)  # another builder waits meanwhile
    finally:
        builder.send_signal(signal.SIGKILL)
        builder.wait()
    assert (tmp_path / "lock").exists()  # PyTorch's, left behind
    with only_builder(tmp_path):
        assert not (tmp_path / "lock").exists()  # the next builder goes on
