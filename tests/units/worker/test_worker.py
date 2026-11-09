"""``core.worker``: the loop alone against fakes, then the whole worker stack (#30's units and the
loop) under the real scheduler (#31): tokens equal HF's greedy generation token by token."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from core.contracts.columns import Column
from core.contracts.records import (
    STEP_PLAN,
    STEP_RESULT,
    BatchPlan,
    Capacity,
    Resync,
    RowDelta,
    Sampled,
    StateOps,
    StepFailure,
    StepStats,
)
from core.contracts.requests import NewRequest
from core.contracts.sampling import SamplingParams
from core.ipc.records import RecordBuilder, decode_record, encode_record, read_sections
from core.worker.unit import WorkerLoop
from tests.units.support.worker import hf_greedy, hf_model, save_checkpoint, spec, tiny_config
from tests.units.worker.support import RealWorker, stack
from unit.runtime import ReloadPlan

# ------------------------------------------------------------------ the loop alone


@dataclass
class Pending:
    epoch: int
    done: bool = True

    def ready(self) -> bool:
        return self.done

    def finish(self) -> Sampled:
        return Sampled(Column.of("i32", [0]), Column.of("i32", [1]), Column.of("i64", [self.epoch]))

    def stats(self) -> StepStats:
        return StepStats(self.epoch, 1.0, 0.0, 0.0)


@dataclass
class FakeExecutor:
    answers: dict[int, Any] = field(default_factory=dict[int, Any])
    """epoch -> a ``Pending``, a ``Resync`` or an exception to raise."""
    synced: int = 0

    def capacity(self) -> Capacity:
        return Capacity({"kv.paged": 4})

    def submit(self, epoch: int, delta: RowDelta, plan: BatchPlan, state: StateOps) -> Any:
        out = self.answers.get(epoch, Pending(epoch))
        if isinstance(out, Exception):
            raise out
        return out

    def synchronize(self) -> None:
        self.synced += 1


@dataclass
class Pipe:
    """Plans in, answers out; ``generation`` grows when the reader of the answers is replaced."""

    plans: list[bytes] = field(default_factory=list[bytes])
    written: list[bytes] = field(default_factory=list[bytes])
    generation: int = 1
    taking: bool = True

    def read(self, timeout: float | None = None) -> bytes | None:
        return self.plans.pop(0) if self.plans else None

    def write(self, data: bytes, timeout: float | None = None) -> bool:
        if not self.taking:
            raise TimeoutError
        self.written.append(data)
        return True

    def answers(self) -> list[tuple[int, str]]:
        out: list[tuple[int, str]] = []
        for raw in self.written:
            record = decode_record(raw)
            assert record.schema == STEP_RESULT
            out += [(record.epoch, name) for name in sorted(record.sections)]
        return out


def plan(epoch: int, rank: int = 0) -> bytes:
    empty = BatchPlan(Column.of("i32"), Column.of("i32"), Column.of("i32"), Column.of("i64"), Column.of("u8"),
                      "decode", 0)
    record = (RecordBuilder(STEP_PLAN, epoch, rank).add(RowDelta.SECTION, RowDelta())
              .add(BatchPlan.SECTION, empty).add(StateOps.SECTION, StateOps()).build())
    return encode_record(record)


def loop(executor: FakeExecutor, pipe: Pipe, **kw: Any) -> WorkerLoop:
    return WorkerLoop(executor, pipe, pipe, rank=0, idle_wait_s=0.0, **kw)


def test_capacity_first_then_answers_in_plan_order() -> None:
    """Step 2 runs on while step 3 is refused: the refusal waits for step 2's answer."""
    slow = Pending(2, done=False)
    ex = FakeExecutor({2: slow, 3: Resync("needs a full delta")})
    pipe = Pipe([plan(1), plan(2), plan(3)])
    w = loop(ex, pipe)
    for _ in range(4):
        w.step()
    assert pipe.answers() == [(0, Capacity.SECTION), (1, Sampled.SECTION), (1, StepStats.SECTION)]
    assert w.epoch == 1 and len(w.taken) == 2  # step 2 runs, step 3 taken alongside it
    slow.done = True
    assert w.step() == 3
    assert pipe.answers()[3:] == [(2, Sampled.SECTION), (2, StepStats.SECTION), (3, Resync.SECTION)]


def test_a_failure_is_answered_and_the_loop_goes_on() -> None:
    ex = FakeExecutor({2: RuntimeError("kernel fault")})
    pipe = Pipe([plan(1), plan(2), plan(3), plan(4, rank=1)])
    w = loop(ex, pipe)
    for _ in range(5):
        w.step()
    failures = [r for r in pipe.written if StepFailure.SECTION in decode_record(r).sections]
    reasons = [cast(StepFailure, read_sections(decode_record(r), {StepFailure.SECTION: StepFailure}).unwrap()
                    [StepFailure.SECTION]).reason for r in failures]
    assert reasons == ["RuntimeError: kernel fault", "ValueError: sched.step_plan@1 for rank 1 reached rank 0"]
    assert (3, Sampled.SECTION) in pipe.answers() and w.epoch == 4 and w.failures == 2


def test_a_device_error_reported_while_asking_whether_a_step_is_ready_is_answered_as_a_failure() -> None:
    """A CUDA event query may raise the error of the device work, before ``finish`` is called
    (Codex, #41): the step is answered ``StepFailure`` and the loop goes on, instead of asking
    again until the scheduler's step timeout."""

    @dataclass
    class Faulted(Pending):
        def ready(self) -> bool:
            raise RuntimeError("CUDA error: device-side assert triggered")

    ex = FakeExecutor({1: Faulted(1)})
    pipe = Pipe([plan(1), plan(2)])
    w = loop(ex, pipe)
    for _ in range(3):
        w.step()
    assert pipe.answers() == [(0, Capacity.SECTION), (1, StepFailure.SECTION), (2, Sampled.SECTION),
                              (2, StepStats.SECTION)]
    assert w.failures == 1 and not w.taken and w.epoch == 2


def test_a_new_reader_of_the_answers_gets_capacity_again() -> None:
    """A new scheduler process starts with an empty ledger: it learns the capacity from the worker."""
    pipe = Pipe()
    w = loop(FakeExecutor(), pipe)
    w.step()
    w.step()
    pipe.generation = 2
    w.step()
    assert [a for a in pipe.answers() if a[1] == Capacity.SECTION] == [(0, Capacity.SECTION)] * 2


def test_an_idle_worker_finds_a_scheduler_restarted_while_nothing_ran() -> None:
    """Over a real ``ChannelWriter``: the connection is found replaced only when a write is tried,
    and an idle worker has nothing to write. The new scheduler waits for ``Capacity`` before it
    sends a plan, so without the idle re-announce both would wait for ever (Codex, #37)."""
    import time
    import uuid

    import msgspec
    import zmq

    from core.common.ids import HostId
    from core.ipc import IpcTunables, open_reader
    from core.ipc.channels import ChannelWriter

    host, zctx = HostId("host-a"), zmq.Context()
    tun = msgspec.structs.replace(IpcTunables(), lookup_retry_s=0.02, peer_timeout_s=0.5)

    def reader() -> tuple[Any, Any]:
        return open_reader(zctx, local=host, peer=host, bind_host="127.0.0.1", shm_name=f"agx-w-{uuid.uuid4().hex[:8]}",
                           tun=tun)

    def capacity_within(r: Any, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            w.step()
            raw = r.read(timeout=0.01)
            if raw is not None and Capacity.SECTION in decode_record(raw).sections:
                return True
        return False

    first, address = reader()
    current = {"address": address}
    results = ChannelWriter(zctx, host, tun, lambda: current["address"])
    w = WorkerLoop(FakeExecutor(), Pipe(), results, rank=0, idle_wait_s=0.005, announce_every_s=0.2)
    try:
        assert capacity_within(first, 5)
        first.close()  # the scheduler host is restarted while nothing runs
        second, current["address"] = reader()
        assert capacity_within(second, 10)
        second.close()
    finally:
        results.close()
        zctx.term()


def test_quiesce_finishes_every_step_and_says_when_answers_cannot_go_out() -> None:
    slow = Pending(1, done=False)
    ex = FakeExecutor({1: slow})
    pipe = Pipe([plan(1)])
    w = loop(ex, pipe, quiesce_timeout_s=0.05)
    w.step()
    slow.done = True
    assert w.quiesce() == 1 and ex.synced == 1 and not w.taken
    pipe.plans.append(plan(2))
    pipe.taking = False
    w.step()
    with pytest.raises(TimeoutError, match=r"answers to steps \[2\]"):
        w.quiesce()


# ------------------------------------------------------------------ the whole stack

PROMPTS = {1: [5, 17, 3, 44, 9, 61, 2], 2: [8, 30, 30, 1, 77, 12, 40, 3, 19, 6, 55], 3: [4, 4, 9]}
STEPS = 6


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], Path, Any]:
    cfg = tiny_config()
    model = hf_model(cfg)
    return cfg.to_dict(), save_checkpoint(model, tmp_path_factory.mktemp("ckpt")), model


@pytest.fixture(scope="module")
def reference(checkpoint: tuple[dict[str, Any], Path, Any]) -> dict[int, list[int]]:
    return {rid: hf_greedy(checkpoint[2], prompt, STEPS)[0] for rid, prompt in PROMPTS.items()}


def greedy(prompts: dict[int, list[int]], steps: int = STEPS) -> Sequence[NewRequest]:
    return [NewRequest(rid, Column.of("i64", p), SamplingParams(temperature=0.0, seed=rid), steps)
            for rid, p in prompts.items()]


@pytest.mark.parametrize(("ranks", "overlap"), [(1, False), (1, True), (2, True)])
def test_requests_through_the_worker_loop_follow_hf(checkpoint: tuple[dict[str, Any], Path, Any],
                                                    reference: dict[int, list[int]], ranks: int, overlap: bool) -> None:
    """Chunked prefill (8 tokens a step), prefill first, then decode; with two ranks every rank runs
    the whole model, and the scheduler checks they sampled the same."""
    with stack(checkpoint[0], checkpoint[1], ranks=ranks, overlap=overlap) as (b, workers):
        b.client.send(greedy(PROMPTS))
        b.run_until(b.finished(*PROMPTS), timeout=60)
        assert {rid: b.client.outputs[rid].tokens for rid in PROMPTS} == reference
        assert all(w.loop.failures == 0 for w in workers) and b.engine.loop.report().fault is None
        b.run_until(b.idle, timeout=10)


def test_a_rebuilt_persistent_batch_is_resynced_and_generation_goes_on(
        checkpoint: tuple[dict[str, Any], Path, Any], reference: dict[int, list[int]]) -> None:
    """``worker.batch`` reloaded mid-generation, as a change would do it (the loop quiesced first):
    the next plan is refused, the full delta rebuilds it, and the KV and GDN state, which live in the
    stores, carry on."""
    def rebuild(w: RealWorker) -> None:
        w.loop.quiesce()
        assert w.runtime is not None
        w.runtime.reload(ReloadPlan(units_to_reload=[spec("worker.batch")])).unwrap()

    with stack(checkpoint[0], checkpoint[1], overlap=True) as (b, workers):
        b.client.send(greedy(PROMPTS))
        b.run_until(lambda: all(len(b.client.outputs[rid].tokens) >= 2 for rid in PROMPTS), timeout=60)
        workers[0].call(rebuild)
        b.run_until(b.finished(*PROMPTS), timeout=60)
        assert {rid: b.client.outputs[rid].tokens for rid in PROMPTS} == reference
        assert any("not run" in d for d in b.engine.loop.report().diagnostics)


def test_an_executor_failure_reaches_the_scheduler_as_step_failure(checkpoint: tuple[dict[str, Any], Path, Any]) -> None:
    def inject(w: RealWorker) -> None:
        real = w.loop.executor

        class Failing:
            def __getattr__(self, name: str) -> Any:
                return getattr(real, name)

            def submit(self, *args: Any) -> Any:
                w.loop.executor = real
                raise RuntimeError("injected device fault")

        w.loop.executor = cast(Any, Failing())

    with stack(checkpoint[0], checkpoint[1]) as (b, workers):
        b.client.send(greedy({1: PROMPTS[1]}, steps=20))
        b.run_until(lambda: len(b.client.outputs[1].tokens) >= 2, timeout=60)
        workers[0].call(inject)
        b.run_until(lambda: b.engine.loop.fault is not None, timeout=30)
        fault = b.engine.loop.fault
        assert fault is not None and "injected device fault" in fault.reason and fault.rank == 0
        assert workers[0].loop.failures == 1
