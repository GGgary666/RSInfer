"""An EPLB-like strategy (modules.html ⑤). The scheduler asks it every step; when a new round is
requested (``scn.rebalance`` grows) it submits an internal, state-only change that rewrites every
worker's placement table. The new placement is state contents: the identity stays, the state
version grows by one. It runs from the step, never during a change, so the change it submits is
computed under the identity that is current."""

from __future__ import annotations

from typing import Any

from core.change.types import ChangeSet, StateRequest
from core.common.ids import UnitId

from ..journal import Journal


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        self.config = config
        self.control = ctx.use("proc.control@1")
        self.journal = Journal(config["scn.log_dir"], ctx)
        self.done = config["scn.rebalance"]
        ctx.on_config("scn.rebalance", lambda v: None)  # read every step: a new value needs no reload
        ctx.provide("scn.balancer@1", self)

    def tick(self) -> None:
        wanted = self.config["scn.rebalance"]
        if wanted <= self.done:
            return
        self.done = wanted
        placement = [(i + wanted) % 4 for i in range(4)]
        identity = self.control.identity
        cs = ChangeSet(base=identity, source="internal", computed_under=identity, state_only=True,
                       state=[StateRequest(UnitId("scn.memory"), "relocate", {"placement": placement})])
        self.journal.write(f"balancer submitted {placement}: {self.control.submit(cs)}")
