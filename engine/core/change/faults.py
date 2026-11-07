"""Fault injection for the coordinator's tests (doc/specs/change.md §7).

Nothing here does anything unless a test arms a point (``coord.faults.add("after_execute")``).
The coordinator's main flow only asks one-line questions (``take``, ``armed``, ``silent``,
``quiet``); the state a fault needs (which change went silent, whether keepalives pause)
lives here, not in the coordinator.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterable
from types import MappingProxyType
from typing import Any

POINTS = MappingProxyType({
    # one-shot: the next change that reaches the point takes it
    "after_check": "abort right after the checks and the plan",
    "after_prepare": "abort once every participant prepared",
    "after_pause": "abort once every participant paused",
    "after_execute": "abort once every participant executed",
    "vanish_during_prepare": "send Prepare, then go silent: no keepalives, no Abort, queries unanswered",
    "vanish_after_prepare": "go silent once every participant prepared",
    "skip_abort": "decide abort, but send no Abort",
    "delay_commit": "decide commit, then stay quiet for twice the participants' lease before Commit",
    # while armed
    "no_decisions": "never answer QueryDecision",
})
"""Every fault point, and what it does."""

DRAIN_S = 0.1
"""How long ``vanish_during_prepare`` lets its Prepares travel before it stops sending."""


class Faults:
    def __init__(self) -> None:
        self._armed: set[str] = set()
        self._silent: set[int] = set()
        self.quiet = False
        """The coordinator sends no keepalives (``delay_commit``)."""

    def add(self, point: str) -> None:
        if point not in POINTS:
            raise ValueError(f"unknown fault point {point!r}; known: {sorted(POINTS)}")
        self._armed.add(point)

    def __ior__(self, points: Iterable[str]) -> Faults:
        for p in points:
            self.add(p)
        return self

    def take(self, point: str) -> bool:
        """A one-shot point: ``True`` the first time it is reached after it was armed."""
        assert point in POINTS, point
        if point in self._armed:
            self._armed.discard(point)
            return True
        return False

    def armed(self, point: str) -> bool:
        """A point that holds while it is armed (``no_decisions``)."""
        assert point in POINTS, point
        return point in self._armed

    def silence(self, change_id: int) -> None:
        """The coordinator vanished during this change: it keeps silent about it from now on."""
        self._silent.add(change_id)

    def silent(self, change_id: int) -> bool:
        return change_id in self._silent

    async def send_then_vanish(self, calls: Iterable[Awaitable[Any]]) -> None:
        """Let the calls leave, then drop them: nothing else follows."""
        sent = [asyncio.ensure_future(c) for c in calls]
        await asyncio.sleep(DRAIN_S)
        for t in sent:
            t.cancel()

    async def hold(self, seconds: float) -> None:
        """Stay quiet (no keepalives) for ``seconds``."""
        self.quiet = True
        try:
            await asyncio.sleep(seconds)
        finally:
            self.quiet = False
