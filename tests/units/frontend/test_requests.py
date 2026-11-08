from __future__ import annotations

import pytest

from core.contracts.columns import Column
from core.contracts.records import Sampled
from core.contracts.requests import NewRequest
from sched.reqs.unit import RequestTable


def result(tokens: list[int]) -> Sampled:
    return Sampled(Column.of("i32", [0]), Column.of("i32", [len(tokens)]), Column.of("i64", tokens))


def test_cancel_drops_late_tokens_but_waits_for_every_submitted_step() -> None:
    table = RequestTable()
    entry = table.add(NewRequest(7, Column.of("i64", [10]))).unwrap()
    table.start_step(1, {0: 7})
    table.start_step(2, {0: 7})
    assert table.cancel(7, "v1").finished == {7: "abort"}
    assert table.cancel(7, "v1").finished == {}
    with pytest.raises(ValueError, match="in flight"):
        table.reap(7)
    assert len(table.sample(1, "v1", result([20])).tokens) == 0
    assert table.retire(1) == []
    assert len(table.sample(2, "v1", result([21])).tokens) == 0
    assert table.retire(2) == [7]
    assert table.reap(7) is entry and entry.output_ids == []


@pytest.mark.parametrize("tokens,limit,reason,accepted", [
    ([11, 99, 12], 3, "stop", [11, 99]),
    ([11, 12, 99], 2, "length", [11, 12]),
    ([11, 99], 2, "stop", [11, 99]),
])
def test_finish_priority_and_accepted_length_follow_sglang(
    tokens: list[int], limit: int, reason: str, accepted: list[int],
) -> None:
    table = RequestTable()
    entry = table.add(NewRequest(7, Column.of("i64", [10]), max_tokens=limit, stop_token_ids=[99])).unwrap()
    table.start_step(1, {0: 7})
    events = table.sample(1, "v2", result(tokens))
    assert events.finished == {7: reason} and list(events.tokens.values()) == accepted
    assert entry.output_ids == accepted and entry.identities == ["v2"]
    assert not table.sample(1, "v2", result(tokens)).finished
    assert table.retire(1) == [7]


def test_request_ids_and_step_row_mappings_do_not_alias() -> None:
    table = RequestTable()
    first = NewRequest(7, Column.of("i64", [10]))
    assert table.add(first).ok
    assert not table.add(first).ok
    table.add(NewRequest(8, Column.of("i64", [12]))).unwrap()
    rows = {0: 7}
    table.start_step(1, rows)
    rows[0] = 8
    assert list(table.sample(1, "v1", result([44])).requests.values()) == [7]
    assert table.requests[8].output_ids == []
    assert table.requests[7].input.sampling.seed is not None
