"""The worker loop: every step replays the decode graph (modules.html ①, worker side).

Steps are numbered from the clock every fake host shares, as workers report the
scheduler's step numbers (change.md §4.2).
"""

from __future__ import annotations

import time
from typing import Any

from ..journal import Journal

STEP_S = 0.005


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        self.plan = ctx.use("scn.plan@1")
        self.journal = Journal(config["scn.log_dir"], ctx)
        self.failures = 0
        ctx.provide("proc.loop@1", self)

    def step(self) -> int:
        time.sleep(STEP_S)
        try:
            self.plan.replay()
        except Exception as e:  # a stale graph must never go unnoticed by a test
            self.failures += 1
            if self.failures == 1:
                self.journal.write(f"worker replay failed: {type(e).__name__}: {e}")
        return int(time.time() / STEP_S)
