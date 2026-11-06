"""change.md §7 tests 8 and 9, on the queue itself."""

from __future__ import annotations

from core.change.queue import ChangeQueue, Queued
from core.change.types import ChangeMeta, ChangeSet
from core.common.ids import Identity

CUR = Identity("cur")


def apply(name: str) -> Queued:
    return Queued(ChangeSet(base=CUR, meta=ChangeMeta(intent=name)))


def internal(name: str, under: Identity = CUR) -> Queued:
    return Queued(ChangeSet(base=CUR, source="internal", computed_under=under, meta=ChangeMeta(intent=name)))


def fault(name: str) -> Queued:
    return Queued(ChangeSet(base=CUR, source="operator", meta=ChangeMeta(intent=name, handles_fault=True)))


def drain(q: ChangeQueue, current: Identity = CUR) -> tuple[list[str], list[str]]:
    order: list[str] = []
    dropped: list[str] = []
    while True:
        item, gone = q.pop(current)
        dropped += [g.cs.meta.intent for g in gone]
        if item is None:
            return order, dropped
        order.append(item.cs.meta.intent)


def test_priority_and_at_most_one_internal_ahead_of_a_waiting_apply() -> None:
    q = ChangeQueue()
    for item in [apply("A1"), internal("I1"), internal("I2"), fault("F"), apply("A2"), internal("I3")]:
        q.push(item)
    order, dropped = drain(q)
    assert order == ["F", "I1", "A1", "I2", "A2", "I3"]
    assert dropped == []


def test_internal_changes_run_back_to_back_when_no_apply_waits() -> None:
    q = ChangeQueue()
    for item in [internal("I1"), internal("I2"), internal("I3")]:
        q.push(item)
    assert drain(q)[0] == ["I1", "I2", "I3"]


def test_stale_internal_changes_are_dropped_applies_never() -> None:
    q = ChangeQueue()
    for item in [internal("old", Identity("before")), apply("A"), internal("fresh")]:
        q.push(item)
    order, dropped = drain(q)
    assert order == ["fresh", "A"] and dropped == ["old"]
    q.push(Queued(ChangeSet(base=Identity("before"), meta=ChangeMeta(intent="stale apply"))))
    assert drain(q) == (["stale apply"], [])  # rejected later by the coordinator, never silently dropped
