"""``core.sched``'s ``drain`` action: a change applies to an empty engine (doc/specs/empty-apply.md).

``core.deps`` gives every change but one of live config keys only a ``drain`` of the units that
hold requests, with the change's ``mode`` (``ChangeMeta.when_busy``). The frontend is the gate: it
takes no new requests while a change runs. Here the scheduler:

- ``refuse``: with requests or steps in it, fails the change and says how to wait or force;
- ``wait``: serves what is in it (and what the frontend took before it closed) until all ended;
- ``force``: ends every request at once with an error, refuses what arrives, and waits only for
  the steps in flight.

With a frontend in front of it (``gate`` false), the scheduler goes on reading: a request the
frontend took before it closed may still be on its way, and the change waits for the frontend to
see it end (served, or refused when forcing). Without one (``gate`` true) the scheduler is the
gate: once empty it takes no requests (``Scheduler.held``) until the change committed or was
undone; they wait in the stream and run on what the change made.
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
        self.gate = False
        """No frontend in front of this scheduler drains: it stops taking requests itself once empty."""

    def start_drain(self, args: dict[str, Any]) -> None:
        mode = str(args.get("mode", "refuse"))
        if mode not in ("refuse", "wait", "force"):
            raise ValueError(f"drain mode {mode!r}: refuse, wait or force")
        occupied = self.loop.occupancy()
        self._fail_if_stuck(mode)
        if mode == "refuse" and any(occupied):
            raise RuntimeError(f"{holds(*occupied)}. {HINT}")
        self.loop.drain_mode = mode
        self.gate = bool(args.get("gate", False))
        if mode == "force":
            ended = self.loop.end_all(FORCED)
            print(f"core.sched: a change (apply --force) ended {ended} request(s)", file=sys.stderr, flush=True)
        self.drained()

    def drained(self) -> bool:
        if self.loop.drain_mode is None:
            return True
        self._fail_if_stuck(self.loop.drain_mode)  # a worker may fail while the change waits (Devin, #42)
        if any(self.loop.occupancy()):
            return False
        self.loop.held = self.gate
        return True

    def _fail_if_stuck(self, mode: str) -> None:
        """After a worker failure no step goes out any more: what the requests hold never ends by
        itself, so only ``force`` can empty the engine (Devin, #42)."""
        fault, occupied = self.loop.fault, self.loop.occupancy()
        if fault is not None and mode != "force" and any(occupied):
            raise RuntimeError(f"the engine stopped serving at step {fault.step} ({fault.reason}) and "
                               f"{holds(*occupied)}, which can no longer end: waiting cannot empty it. Run the "
                               "same apply with --force to end them and apply.")

    def drain_deadline(self) -> dict[str, int]:
        occupied = self.loop.occupancy()
        what = "the steps in flight did not end" if self.loop.drain_mode == "force" else "the engine did not empty"
        raise TimeoutError(f"{what} in time: {holds(*occupied)}; nothing was ended for this change, and "
                           "requests come in again. Retry with a longer --wait-empty, or with --force.")

    def commit(self) -> None:
        """The change may have rebuilt the workers' stores with another size: plan again once each
        rank sent its ``Capacity`` under the new identity (Codex, Antigravity, #42)."""
        self._open()
        self.loop.capacity_wanted = set(range(len(self.loop.results)))

    def abort(self) -> None:
        self._open()

    def _open(self) -> None:
        self.loop.drain_mode, self.loop.held = None, False
