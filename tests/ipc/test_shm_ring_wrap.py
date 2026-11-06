"""SHM wrap regression with a real reader and a separate writer process.

The control pipe only drives writes and reports their outcome. Payloads always
travel through the actual shared-memory ring, independently of the control pipe.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from core.ipc.config import IpcTunables
from core.ipc.stream import ShmReader, ShmWriter


def writer_main() -> None:
    """A standalone child has its own resource tracker, as engine processes do."""
    writer = ShmWriter(sys.argv[1], IpcTunables(heartbeat_interval_s=0.02, peer_timeout_s=5.0))
    print(json.dumps({"ready": True}), flush=True)
    try:
        for line in sys.stdin:
            command = json.loads(line)
            if command.get("close"):
                break
            try:
                writer.write(bytes.fromhex(command["data"]), timeout=command["timeout"])
            except (TimeoutError, ValueError) as error:
                print(json.dumps({"error": type(error).__name__}), flush=True)
            else:
                print(json.dumps({"written": True}), flush=True)
    finally:
        writer.close()


class RingPair:
    def __init__(self, capacity: int) -> None:
        name = f"agx-wrap-{uuid.uuid4().hex[:12]}"
        self.reader = ShmReader(name, IpcTunables(shm_capacity=capacity, heartbeat_interval_s=0.02,
                                                peer_timeout_s=5.0))
        root = Path(__file__).resolve().parents[2]
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root / "engine"), str(root)]))
        try:
            self.child = subprocess.Popen(
                [sys.executable, "-c", "from tests.ipc.test_shm_ring_wrap import writer_main; writer_main()", name],
                cwd=root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            assert self.reply(timeout=5) == {"ready": True}
        except BaseException:
            self.close()
            raise

    def start_write(self, data: bytes, timeout: float = 0.5) -> None:
        assert self.child.stdin is not None
        self.child.stdin.write((json.dumps({"data": data.hex(), "timeout": timeout}) + "\n").encode())
        self.child.stdin.flush()

    def reply(self, timeout: float = 2.0) -> dict[str, Any]:
        assert self.child.stdout is not None
        ready, _, _ = select.select([self.child.stdout], [], [], timeout)
        assert ready, "writer did not return within the test deadline"
        line = self.child.stdout.readline()
        assert line, "writer exited without reporting the write outcome"
        return json.loads(line)

    def write(self, data: bytes) -> None:
        self.start_write(data)
        assert self.reply() == {"written": True}

    def close(self) -> None:
        child = getattr(self, "child", None)
        if child is not None:
            try:
                _, errors = child.communicate(b'{"close": true}\n', timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                _, errors = child.communicate()
            finally:
                self.reader.close()
            assert child.returncode == 0, errors.decode()
        else:
            self.reader.close()


@pytest.fixture
def ring() -> Generator[RingPair]:
    pair = RingPair(64)
    try:
        yield pair
    finally:
        pair.close()


@pytest.mark.parametrize(("warmup", "large"), [(20, 40), (4, 60)])
def test_large_record_progresses_after_drained_ring_wrap(ring: RingPair, warmup: int, large: int) -> None:
    first, second = b"a" * warmup, b"b" * large
    ring.write(first)
    assert ring.reader.read(timeout=1) == first
    # Both records fit individually. Padding plus the second record exceeds 64.
    ring.start_write(second)
    assert ring.reader.read(timeout=1) == second
    assert ring.reply() == {"written": True}
