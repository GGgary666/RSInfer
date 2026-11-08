"""core.contracts (doc/specs/stage2-contract.md): the stage-2 interface library. Columns and
records survive the wire, and units on both sides of an interface may import it."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from core.common.framework import FrameworkModules
from core.common.ids import HostId
from core.contracts.columns import Column, DType, ragged
from core.contracts.records import (
    STEP_PLAN,
    AddedRow,
    BatchPlan,
    RowDelta,
    SamplingParams,
    StateAssign,
    StateCopy,
    StateOps,
)
from core.contracts.state import common_prefix
from core.ipc.records import RecordBuilder, decode_record, encode_record, read_sections
from core.manifest.checks import check_tree
from core.supervisor import HostSpec, Mesh, RankPlacement, RoleSpec, Topology
from tests.framework.support import manifest_toml, write_tree

ENGINE = Path(__file__).resolve().parents[3] / "engine"
TOPO = Topology([HostSpec(HostId("h"), "addr", "/snapshots")], Mesh({"tp": 1}), [RankPlacement(0, HostId("h"), 0)],
                [RoleSpec("worker", [], per_rank=True)])


def test_columns_round_trip_every_dtype() -> None:
    cases: list[tuple[DType, list[int] | list[float]]] = [
        ("u8", [0, 1, 255]), ("i32", [-(2**31), 0, 2**31 - 1]), ("i64", [-(2**63), 2**63 - 1]), ("f32", [0.5, -2.0])]
    for dtype, values in cases:
        col = Column.of(dtype, values)
        assert len(col) == len(values)
        assert list(col.values()) == values
    offsets, flat = ragged("i32", [[1, 2], [], [3]])
    assert list(offsets.values()) == [0, 2, 2, 3] and list(flat.values()) == [1, 2, 3]


def test_a_step_plan_round_trips_through_an_open_record() -> None:
    offsets, ids = ragged("i32", [[7, 8], [9]])
    batch = BatchPlan(rows=Column.of("i32", [0, 1]), num_tokens=Column.of("i32", [2, 1]),
                      positions=Column.of("i32", [0, 5]), tokens=Column.of("i64", [11, 12]),
                      from_previous=Column.of("u8", [0, 1]), phase="mixed", bucket=0)
    delta = RowDelta(added=[AddedRow(0, 42, SamplingParams(temperature=0.0, seed=3))])
    ops = StateOps(assign=[StateAssign("kv.paged", Column.of("i32", [0, 1]), offsets, ids)],
                   copy=[StateCopy("gdn.slots", "after", Column.of("i32", [4]), Column.of("i32", [30]))])
    record = (RecordBuilder(STEP_PLAN, 17, rank=0).add(BatchPlan.SECTION, batch).add(RowDelta.SECTION, delta)
              .add(StateOps.SECTION, ops).build())
    got = read_sections(decode_record(encode_record(record)),
                        {BatchPlan.SECTION: BatchPlan, RowDelta.SECTION: RowDelta},
                        {StateOps.SECTION: StateOps}).unwrap()
    assert got == {BatchPlan.SECTION: batch, RowDelta.SECTION: delta, StateOps.SECTION: ops}
    assert list(got[BatchPlan.SECTION].from_previous.values()) == [0, 1]
    assert got[BatchPlan.SECTION].check() == []


def batch_plan(num_tokens: list[int], from_previous: list[int], tokens: list[int], rows: int | None = None) -> BatchPlan:
    n = len(num_tokens) if rows is None else rows
    return BatchPlan(rows=Column.of("i32", list(range(n))), num_tokens=Column.of("i32", num_tokens),
                     positions=Column.of("i32", [0] * len(num_tokens)), tokens=Column.of("i64", tokens),
                     from_previous=Column.of("u8", from_previous), phase="mixed", bucket=0)


def test_a_batch_plan_gives_each_row_its_tokens_and_a_previous_token_row_none() -> None:
    """A prefill row of 3, a decode row on the previous token, a decode row with its token
    given: ``tokens`` holds 3 + 0 + 1 values."""
    assert batch_plan([3, 1, 1], [0, 1, 0], [5, 6, 7, 8]).check() == []
    assert batch_plan([2, 1], [1, 0], [9]).check() == ["row 0 takes the previous token but computes 2 tokens"]
    assert batch_plan([3, 1], [0, 1], [5, 6, 7, 8]).check() == [
        "tokens has 4 values, the rows without from_previous take 3"]
    assert batch_plan([1, 1], [1, 1], [], rows=3).check() == [
        "num_tokens has 2 values for 3 rows", "positions has 2 values for 3 rows", "from_previous has 2 values for 3 rows"]


def kv(cached: int, page: int = 16) -> Callable[[int], int]:
    """Paged KV with ``cached`` tokens of leading blocks cached: every page boundary up to that."""
    return lambda n: min(n // page * page, cached)


def gdn(snapshots: list[int]) -> Callable[[int], int]:
    """GDN state: only the snapshots it still holds."""
    return lambda n: max([s for s in snapshots if s <= n], default=0)


def test_the_common_prefix_is_one_every_kind_supports_not_the_shortest_longest_hit() -> None:
    """KV hits 96 tokens, GDN has snapshots at 64 and 128: the shorter of the two longest hits
    is 96, where GDN has no state; the request resumes from 64."""
    kinds = [kv(96), gdn([64, 128])]
    assert min(kind(128) for kind in kinds) == 96 and gdn([64, 128])(96) == 64
    assert common_prefix(kinds, 128) == 64
    assert common_prefix([kv(96), gdn([128])], 128) == 0  # the snapshot at 64 was evicted
    assert common_prefix([kv(256), gdn([64, 128, 192])], 200) == 192
    assert common_prefix([kv(512, page=32), gdn([96, 192, 384])], 400) == 384
    assert common_prefix([kv(1000, page=32), gdn([96, 160]), gdn([64, 128, 192])], 400) == 0
    assert common_prefix([kv(1000), gdn([64, 128, 192]), gdn([128, 256])], 400) == 128
    assert common_prefix([], 48) == 48


def test_a_kind_that_answers_above_its_limit_is_refused() -> None:
    """Every answer is checked, not only the lowest: a valid answer must not hide one above the limit."""
    with pytest.raises(ValueError, match="answered 80 tokens when asked for at most 64"):
        common_prefix([lambda n: n + 16], 64)
    with pytest.raises(ValueError, match="answered 80 tokens when asked for at most 64"):
        common_prefix([lambda n: n + 16, lambda n: n], 64)
    with pytest.raises(ValueError, match="answered 48 tokens when asked for at most 32"):
        common_prefix([kv(32), lambda n: 48], 64)


def test_units_may_import_the_contracts_and_the_contracts_pass_the_rules(tmp_path: Path) -> None:
    """The contracts are a library outside every unit directory: importing them is not a
    cross-unit import, and they keep the rules reloadable code keeps (no mutable globals)."""
    root = tmp_path / "engine"
    shutil.copytree(ENGINE / "core" / "contracts", root / "core" / "contracts",
                    ignore=shutil.ignore_patterns("__pycache__"))
    write_tree(root, {
        "core/__init__.py": "",
        "pkg/store/unit.toml": manifest_toml("pkg.store", "pkg.store.unit:Unit", provides=["state.kv@1"],
                                             kind="state"),
        "pkg/store/unit.py": "from core.contracts.state import StateDecl\n\nclass Unit:\n"
                             "    def __init__(self, ctx, config):\n        ctx.provide('state.kv@1', self)\n",
        "pkg/backend/unit.toml": manifest_toml("pkg.backend", "pkg.backend.unit:Unit", requires=["state.kv@1"]),
        "pkg/backend/unit.py": "from core.contracts.plan import PlanKey\nfrom core.contracts.records import BatchPlan\n"
                               "\nclass Unit:\n    def __init__(self, ctx, config):\n"
                               "        self.kv = ctx.use('state.kv@1')\n",
    })
    errors, _ = check_tree(root, TOPO, FrameworkModules.default())
    assert errors == []
