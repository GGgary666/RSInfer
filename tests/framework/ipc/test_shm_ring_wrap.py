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
from core.ipc.stream import ShmReader, ShmWriter, StreamClosed


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
        root = Path(__file__).resolve().parents[3]
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root / "engine"), str(root)]))
        try:
            self.child = subprocess.Popen(
                [sys.executable, "-c", "from tests.framework.ipc.test_shm_ring_wrap import writer_main; writer_main()", name],
                cwd=root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
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


@pytest.mark.parametrize(("warmup", "large"), [(20, 40), (4, 60)])
def test_timed_out_wrap_preserves_unread_record_and_can_retry(ring: RingPair, warmup: int, large: int) -> None:
    first, unread, next_record = b"a" * warmup, b"unread" * 3 + b"!!", b"z" * large
    ring.write(first)
    assert ring.reader.read(timeout=1) == first
    ring.write(unread)
    # Stop consuming. The unread 24-byte frame plus the new frame exceeds 64.
    ring.start_write(next_record, timeout=0.05)
    outcome = ring.reply()
    received = ring.reader.read(timeout=1)
    assert received == unread, f"wrap discarded or overwrote an unread record: {received!r}"
    assert outcome == {"error": "TimeoutError"}, "a full ring must apply backpressure"
    assert ring.reader.read(timeout=0.02) is None  # Padding is not a record; the timed-out payload was not published.
    ring.start_write(next_record)
    assert ring.reader.read(timeout=1) == next_record
    assert ring.reply() == {"written": True}
    assert ring.reader.read(timeout=0.02) is None  # Retrying must publish exactly once.


def test_closing_after_timed_out_wrap_drains_unread_data_before_eof(ring: RingPair) -> None:
    ring.write(b"a" * 20)
    assert ring.reader.read(timeout=1) == b"a" * 20
    ring.write(b"b" * 20)
    ring.start_write(b"c" * 40, timeout=0.05)
    assert ring.reply() == {"error": "TimeoutError"}
    assert ring.child.stdin is not None
    ring.child.stdin.write(b'{"close": true}\n')
    ring.child.stdin.flush()
    assert ring.child.wait(timeout=2) == 0
    assert ring.reader.read(timeout=1) == b"b" * 20
    with pytest.raises(StreamClosed):
        ring.reader.read(timeout=1)


@pytest.mark.parametrize("capacity", [64, 65, 67, 128])
def test_variable_records_are_ordered_and_lossless_across_repeated_wraps(capacity: int) -> None:
    pair = RingPair(capacity)
    # 65/67 also exercise wrap padding too short to contain a 4-byte marker.
    sizes = [0, 20, 40, 60, 1, 12, 4, 52] * 3
    records = [bytes([i + 1]) * size for i, size in enumerate(sizes)]
    try:
        for offset in range(0, len(records), 3):
            batch = records[offset:offset + 3]
            for data in batch:
                pair.start_write(data)
            for expected in batch:
                assert pair.reader.read(timeout=1) == expected
            for _ in batch:
                assert pair.reply() == {"written": True}
        assert pair.reader.read(timeout=0.02) is None
    finally:
        pair.close()


def test_oversized_record_is_rejected_without_losing_queued_data(ring: RingPair) -> None:
    ring.write(b"queued")
    ring.start_write(b"x" * 61)  # Header and alignment make this exceed the 64-byte capacity.
    assert ring.reply() == {"error": "ValueError"}
    assert ring.reader.read(timeout=1) == b"queued"
    assert ring.reader.read(timeout=0.02) is None
