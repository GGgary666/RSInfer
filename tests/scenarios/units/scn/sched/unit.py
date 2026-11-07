"""The scheduling logic: every step it asks a policy which requests to advance, and advances
them by ``scn.chunk``, read every step and taken in place when it changes (modules.html ⑦ example 1).
Policies register into the registry it provides (the decision point)."""

from __future__ import annotations

import time
from typing import Any

from unit.runtime import Registry

from ..journal import Journal

STEP_S = 0.005


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        self.ctx = ctx
        self.config = config
        self.requests = ctx.use("scn.requests@1")
        self.policies: Registry[Any] = Registry("scn.policies")
        self.journal = Journal(config["scn.log_dir"], ctx)
        self.journal.write(f"sched gen{ctx.instance.generation} progress {sum(self.requests.progress.values())}")
        ctx.on_config("scn.chunk", lambda v: self.journal.write(f"sched chunk {v}"))
        ctx.provide("scn.policies@1", self.policies)
        ctx.provide("proc.loop@1", self)

    def step(self) -> int:
        time.sleep(STEP_S)
        chunk = self.config["scn.chunk"]
        for policy in self.policies.items()[:1]:
            for rid in policy.pick(self.requests.progress):
                self.requests.progress[rid] += chunk
        balancer = self.ctx.get("scn.balancer@1")
        if balancer is not None:
            balancer.tick()
        return int(time.time() / STEP_S)
