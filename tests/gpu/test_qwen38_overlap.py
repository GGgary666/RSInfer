"""One step of overlap through the worker stack on a CUDA GPU (stage2-contract.md §4.4): the next
step is submitted before the previous one has finished, as ``core.worker`` does under
``sched.overlap``, its input tokens taken on the device (``from_previous``). The tokens must equal
those of steps run one at a time, and the device must sit idle less between steps.

Manual (``tests/gpu`` is not collected by default):
    uv run pytest tests/gpu/test_qwen38_overlap.py -s
Qwen3.8-27B's first 4 layers with random weights (``make_checkpoint``), which fits a 24 GB GPU.
"""

from __future__ import annotations

import statistics
from collections import deque
from pathlib import Path

import pytest

from core.contracts.records import Resync, StepStats
from platforms.torch_eager import TorchEagerPlatform
from tests.gpu.support import free_gpu, make_checkpoint
from tests.units.support.worker import Driver, Worker

LAYERS, ROWS, PROMPT, STEPS = 4, 8, 24, 48
WARMUP = 4
"""Decode steps left out of the timings."""


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_checkpoint(tmp_path_factory, LAYERS)


def decode(checkpoint: Path, depth: int) -> tuple[dict[int, list[int]], list[StepStats]]:
    """Prefill every row, then ``STEPS`` decode steps with ``depth`` of them submitted before the
    oldest is finished (1: one at a time)."""
    import json

    cfg = json.loads((checkpoint / "config.json").read_text())
    w = Worker(cfg, platform=TorchEagerPlatform(cuda=True), **{
        "weights.path": str(checkpoint), "model.dtype": "bf16", "state.gdn.conv_dtype": "bf16",
        "state.kv.pages": 2048, "state.gdn.slots": ROWS, "batch.max_rows": ROWS, "batch.max_context_len": 1024})
    try:
        d = Driver(w)
        executor = w.service("executor@1")
        rows = list(range(ROWS))
        for r in rows:
            d.add(r, [(1000 + 37 * r + 11 * i) % 248000 for i in range(PROMPT)])
        d.step({r: PROMPT for r in rows})
        pending: deque[object] = deque()
        out: dict[int, list[int]] = {r: [] for r in rows}
        stats: list[StepStats] = []

        def finish() -> None:
            step = pending.popleft()
            sampled = step.finish()  # type: ignore[attr-defined]
            for r, t in zip(sampled.rows.values(), sampled.tokens.values(), strict=True):
                out[int(r)].append(int(t))
            stats.append(step.stats())  # type: ignore[attr-defined]

        for _ in range(STEPS):
            delta, plan, state = d.records({r: 1 for r in rows})
            d.epoch += 1
            step = executor.submit(d.epoch, delta, plan, state)
            assert not isinstance(step, Resync), step
            d.full, d.removed = False, []
            for r in rows:  # its token is still on the device: the next step takes it there
                q = d.requests[r]
                q.in_batch, q.computed = True, q.computed + 1
                q.known.append(-1)
            pending.append(step)
            while len(pending) >= depth:
                finish()
        while pending:
            finish()
        return out, stats[WARMUP:]
    finally:
        w.close()
        free_gpu()


def test_overlapped_steps_sample_the_tokens_of_steps_one_at_a_time(checkpoint: Path) -> None:
    serial, serial_stats = decode(checkpoint, 1)
    overlapped, overlapped_stats = decode(checkpoint, 2)
    for name, s in (("one at a time", serial_stats), ("overlapped", overlapped_stats)):
        print(f"\n{name}: median host {statistics.median(x.host_us for x in s) / 1e3:.2f} ms, device "
              f"{statistics.median(x.device_us for x in s) / 1e3:.2f} ms, idle before "
              f"{statistics.median(x.idle_before_us for x in s) / 1e3:.3f} ms")
    assert overlapped == serial
    assert (statistics.median(x.idle_before_us for x in overlapped_stats)
            < statistics.median(x.idle_before_us for x in serial_stats))
