"""``core.sched``'s ``drain`` action: a change applies to an empty engine (doc/specs/empty-apply.md).

``core.deps`` gives every change but one of live config keys only a ``drain`` of the units that
hold requests, with the change's ``mode`` (``ChangeMeta.when_busy``). The frontend is the gate: it
takes no new requests while a change runs. Here the scheduler:

- ``refuse``: with requests or steps in it, fails the change and says how to wait or force;
- ``wait``: serves what is in it (and what the frontend took before it closed) until all ended;
- ``force``: ends every request at once with an error, refuses what arrives, and waits only for
  the steps in flight.

Once empty it takes no requests (``Scheduler.held``) until the change committed or was undone:
they wait in the stream and run on what the change made.
"""

from __future__ import annotations

import sys
from typing import Any

from .loop import FORCED, Scheduler

HINT = ("A change applies to an empty engine. Run the same apply with --wait-empty [SECONDS] to take no new "
        "requests and apply once these ended (3600 s at most by default), or with --force to end them now with "
        "an error and apply.")


def holds(running: int, waiting: int, steps: int) -> str:
    out = f"the scheduler holds {running} running and {waiting} waiting request(s)"
    return out + (f" and {steps} step(s) in flight" if steps else "")


class Drain:
    def __init__(self, loop: Scheduler) -> None:
        self.loop = loop

    def start_drain(self, args: dict[str, Any]) -> None:
        mode = str(args.get("mode", "refuse"))
        if mode not in ("refuse", "wait", "force"):
            raise ValueError(f"drain mode {mode!r}: refuse, wait or force")
        occupied = self.loop.occupancy()
        if mode == "refuse" and any(occupied):
            raise RuntimeError(f"{holds(*occupied)}. {HINT}")
        self.loop.drain_mode = mode
        if mode == "force":
            ended = self.loop.end_all(FORCED)
            print(f"core.sched: a change (apply --force) ended {ended} request(s)", file=sys.stderr, flush=True)
        self.drained()

    def drained(self) -> bool:
        if self.loop.drain_mode is None:
            return True
        if any(self.loop.occupancy()):
            return False
        self.loop.held = True
        return True

    def drain_deadline(self) -> dict[str, int]:
        occupied = self.loop.occupancy()
        what = "the steps in flight did not end" if self.loop.drain_mode == "force" else "the engine did not empty"
        raise TimeoutError(f"{what} in time: {holds(*occupied)}; nothing was ended for this change, and "
                           "requests come in again. Retry with a longer --wait-empty, or with --force.")

    def commit(self) -> None:
        self._open()

    def abort(self) -> None:
        self._open()

    def _open(self) -> None:
        self.loop.drain_mode, self.loop.held = None, False
