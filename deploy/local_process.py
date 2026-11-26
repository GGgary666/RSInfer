"""Process groups owned by one local deployment run."""

from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

TOKEN = "RSINFER_PROCESS_TOKEN"


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class Service:
    def __init__(self, directory: Path, name: str) -> None:
        self.record = directory / f"{name}.process.json"
        self.log = directory / f"{name}.log"

    def _members(self) -> tuple[int, list[int]]:
        if not self.record.is_file():
            return 0, []
        record = json.loads(self.record.read_text())
        group = int(record["pid"])
        needle = f"{TOKEN}={record['token']}".encode()
        members: list[int] = []
        for path in Path("/proc").iterdir():
            if not path.name.isdecimal():
                continue
            try:
                state, _, pgid, sid, *_ = (path / "stat").read_text().rsplit(")", 1)[1].split()
                if state != "Z" and int(pgid) == int(sid) == group:
                    if needle in (path / "environ").read_bytes().split(b"\0"):
                        members.append(int(path.name))
            except (OSError, ValueError):
                continue
        return group, members

    @property
    def running(self) -> bool:
        return bool(self._members()[1])

    def start(self, args: Sequence[str], env: Mapping[str, str], *, cwd: Path) -> subprocess.Popen[bytes]:
        if self.running:
            raise RuntimeError(f"service already running, see {self.record}")
        self.record.parent.mkdir(parents=True, exist_ok=True)
        token = secrets.token_hex(16)
        with self.log.open("ab") as log:
            process = subprocess.Popen(args, cwd=cwd, env={**env, TOKEN: token}, start_new_session=True,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        try:
            write_json(self.record, {"pid": process.pid, "token": token})
        except OSError:
            process.terminate()
            process.wait()
            raise
        return process

    def wait_ready(self, process: subprocess.Popen[bytes], ready: Callable[[], bool], timeout: float) -> None:
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            if (code := process.poll()) is not None:
                raise RuntimeError(f"process {process.pid} exited with {code}; see {self.log}")
            if ready():
                return
            time.sleep(0.25)
        raise TimeoutError(f"process {process.pid} did not become ready in {timeout:g}s; see {self.log}")

    def stop(self, timeout: float = 60) -> None:
        for signum, seconds in ((signal.SIGTERM, timeout), (signal.SIGKILL, 5)):
            group, members = self._members()
            if not members:
                self.record.unlink(missing_ok=True)
                return
            try:
                os.killpg(group, signum)
            except ProcessLookupError:
                pass
            until = time.monotonic() + seconds
            while time.monotonic() < until:
                if not self.running:
                    self.record.unlink(missing_ok=True)
                    return
                time.sleep(0.25)
        raise RuntimeError(f"process group {group} is still running; see {self.record}")
