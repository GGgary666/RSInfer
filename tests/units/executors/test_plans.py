"""executors.pytorch picks each step's plan from ``plan@1`` (stage2-contract.md §9.1): a decode step
takes the smallest ready graph bucket that holds its rows, any other step the eager plan, and a
graph found stale runs eagerly instead. A stand-in graph plan pads the rows to its bucket, as a
graph would, and runs them eagerly: the routing shows, and so does what padding may write (only
the padding items, §9.1). ``core.plan``'s own graph plans are in ``test_cuda_graph.py``."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

from core.contracts.columns import Column
from core.contracts.plan import PersistentBatch, Phase, Plan, PlanKey
from core.contracts.records import BatchPlan
from executors.pytorch.unit import PyTorchExecutor
from platforms.base import StaleGraph
from platforms.torch_eager import TorchEagerPlatform
from tests.units.support.worker import Driver, Worker, hf_model, save_checkpoint, tiny_config

PROMPTS = {0: [5, 17, 3, 44, 9, 61, 2], 1: [8, 30, 30, 1, 77, 12, 40, 3, 19, 6, 55], 2: [70, 4, 4, 23],
           3: [11, 90, 2, 2, 2, 36], 4: [13, 57, 81]}
STEPS = 5


class GraphStandIn:
    """A decode graph of ``bucket`` rows: pads the step's rows to its bucket with the padding row
    (row -1, index 0 of the batch tables, whose zeros point at the padding page and slot), runs
    them, and returns the real rows' logits."""

    def __init__(self, eager: Plan, bucket: int) -> None:
        self.eager, self.bucket = eager, bucket
        self.stale = False
        self.padded: list[int] = []

    def run(self, batch: PersistentBatch, plan: BatchPlan, input_ids: torch.Tensor, positions: torch.Tensor,
            last_index: torch.Tensor) -> torch.Tensor:
        if self.stale:
            raise StaleGraph(f"decode graph of bucket {self.bucket}")
        rows = len(plan.rows)
        pad = self.bucket - rows
        padded = BatchPlan(Column.of("i32", [*plan.rows.values(), *[-1] * pad]),
                           Column.of("i32", [*plan.num_tokens.values(), *[1] * pad]),
                           Column.of("i32", [*plan.positions.values(), *[0] * pad]),
                           Column.of("i64", [*plan.tokens.values(), *[0] * pad]),
                           Column.of("u8", [*plan.from_previous.values(), *[0] * pad]), plan.phase, self.bucket)
        tokens = input_ids.shape[0]
        self.padded.append(self.bucket)
        logits = self.eager.run(batch, padded, torch.cat([input_ids, input_ids.new_zeros(pad)]),
                                torch.cat([positions, positions.new_zeros(pad)]),
                                torch.cat([last_index, torch.arange(tokens, tokens + pad, dtype=last_index.dtype,
                                                                    device=last_index.device)]))
        return logits[:rows]


class StandInPlans:
    """``plan@1``: the real eager plan, and the stand-in decode graph."""

    required = False

    def __init__(self, real: Any, bucket: int) -> None:
        self.real = real
        self.graph = GraphStandIn(real.get(PlanKey("decode", 0)), bucket)

    def buckets(self, phase: Phase) -> list[int]:
        return [self.graph.bucket] if phase == "decode" else []

    def get(self, key: PlanKey) -> Plan:
        return self.graph if key == PlanKey("decode", self.graph.bucket) else self.real.get(key)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, object], Path]:
    cfg = tiny_config()
    return cfg.to_dict(), save_checkpoint(hf_model(cfg), tmp_path_factory.mktemp("ckpt"))


@pytest.fixture
def worker(checkpoint: tuple[dict[str, object], Path]) -> Iterator[Worker]:
    w = Worker(checkpoint[0], **{"weights.path": str(checkpoint[1])})
    yield w
    w.close()


def executor(worker: Worker, plans: Any) -> PyTorchExecutor:
    stores = {"kv.paged": worker.service("state.kv.paged@1"), "gdn.slots": worker.service("state.gdn.slots@1")}
    return PyTorchExecutor(TorchEagerPlatform(), worker.service("worker.batch@1"), plans,
                           worker.service("layers.sampler@1"), stores)


def run(worker: Worker, rows: list[int], plans: Any = None) -> tuple[dict[int, list[int]], dict[int, torch.Tensor],
                                                                     dict[int, list[torch.Tensor]]]:
    """Every row generates ``STEPS`` tokens: their tokens, logits and the state the rows hold."""
    d = Driver(worker, executor(worker, plans) if plans is not None else None)
    for row in rows:
        d.add(row, PROMPTS[row])
    logits = d.generate(rows, STEPS)
    kv, gdn = worker.service("state.kv.paged@1"), worker.service("state.gdn.slots@1")
    held: dict[int, list[torch.Tensor]] = {}
    for row in rows:
        r = d.requests[row]
        slots = kv.slots(r.pages)[:r.computed]
        held[row] = [*(kv.k_buffer[layer][slots].clone() for layer in kv.k_buffer),
                     *(kv.v_buffer[layer][slots].clone() for layer in kv.v_buffer),
                     gdn.conv[:, r.slot + 1].clone(), gdn.temporal[:, r.slot + 1].clone()]
    return {row: d.requests[row].outputs for row in rows}, {row: torch.stack(logits[row]) for row in rows}, held


def same(a: tuple[dict[int, list[int]], dict[int, torch.Tensor], dict[int, list[torch.Tensor]]],
         b: tuple[dict[int, list[int]], dict[int, torch.Tensor], dict[int, list[torch.Tensor]]]) -> None:
    assert a[0] == b[0]
    for row in a[0]:
        torch.testing.assert_close(a[1][row], b[1][row], rtol=1e-4, atol=1e-4)
        for x, y in zip(a[2][row], b[2][row], strict=True):
            torch.testing.assert_close(x, y, rtol=1e-4, atol=1e-4)


def test_decode_steps_take_the_smallest_ready_graph_and_padded_rows_write_only_padding(worker: Worker) -> None:
    """Three rows decode in the bucket of four: one padding row each step. The real rows' tokens,
    logits, KV and GDN state are those of the eager plan; the prefill step stays eager."""
    eager = run(worker, [0, 1, 2])
    plans = StandInPlans(worker.service("plan@1"), 4)
    padded = run(worker, [0, 1, 2], plans)
    assert plans.graph.padded == [4] * (STEPS - 1)
    same(padded, eager)


def test_rows_no_graph_holds_run_eagerly(worker: Worker) -> None:
    eager = run(worker, [0, 1, 2, 3, 4])
    plans = StandInPlans(worker.service("plan@1"), 4)
    assert run(worker, [0, 1, 2, 3, 4], plans)[0] == eager[0]
    assert plans.graph.padded == []


def test_a_stale_graph_runs_eagerly_instead(worker: Worker) -> None:
    eager = run(worker, [0, 1, 2])
    plans = StandInPlans(worker.service("plan@1"), 4)
    plans.graph.stale = True
    same(run(worker, [0, 1, 2], plans), eager)
    assert plans.graph.padded == []


def test_core_plan_has_only_the_eager_plan_where_the_platform_records_no_graphs(worker: Worker) -> None:
    plans = worker.service("plan@1")
    assert plans.buckets("decode") == [] and plans.buckets("prefill") == []
    with pytest.raises(LookupError):
        plans.get(PlanKey("decode", 8))
