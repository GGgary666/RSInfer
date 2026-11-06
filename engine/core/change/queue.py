"""Change queue (change.md §4.5).

1. No preemption: the running change is never interrupted.
2. Priority: fault handling > internal changes > everything else.
3. No starvation: while an apply waits, at most one internal change in a row goes first.
4. Internal changes computed under an identity that is no longer current are
   dropped (and logged); applies are never dropped.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from core.change.types import ChangeSet
from core.common.ids import Identity

Lane = Literal["fault", "internal", "normal"]


def lane_of(cs: ChangeSet) -> Lane:
    if cs.meta.handles_fault:
        return "fault"
    if cs.source == "internal":
        return "internal"
    return "normal"


@dataclass
class Queued:
    cs: ChangeSet
    dry_run: bool = False
    seq: int = 0
    lane: Lane = "normal"
    data: dict[str, object] = field(default_factory=dict[str, object])


class ChangeQueue:
    def __init__(self) -> None:
        self._lanes: dict[Lane, deque[Queued]] = {"fault": deque(), "internal": deque(), "normal": deque()}
        self._seq = itertools.count()
        self._internal_streak = 0

    def push(self, item: Queued) -> Queued:
        item.seq = next(self._seq)
        item.lane = lane_of(item.cs)
        self._lanes[item.lane].append(item)
        return item

    def __len__(self) -> int:
        return sum(len(q) for q in self._lanes.values())

    def pop(self, current: Identity) -> tuple[Queued | None, list[Queued]]:
        """The next change to run, and the stale internal changes dropped on the way."""
        dropped: list[Queued] = []
        while True:
            if self._lanes["fault"]:
                self._internal_streak = 0
                return self._lanes["fault"].popleft(), dropped
            normal_waiting = bool(self._lanes["normal"])
            if self._lanes["internal"] and (not normal_waiting or self._internal_streak < 1):
                item = self._lanes["internal"].popleft()
                if item.cs.computed_under is not None and item.cs.computed_under != current:
                    dropped.append(item)
                    continue
                if normal_waiting:
                    self._internal_streak += 1
                return item, dropped
            if normal_waiting:
                self._internal_streak = 0
                return self._lanes["normal"].popleft(), dropped
            return None, dropped
