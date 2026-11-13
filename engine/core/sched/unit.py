"""``core.sched``: the scheduler host's loop unit (``proc.loop@1``), wired to its host's channels.

Readers it owns: ``requests`` (from the frontend host) and ``results.<rank>`` (from each worker).
Writers: to each worker's reader ``plans``, and to the token reader of each frontend's
``ReturnRoute``. The rank of a worker is its place in ``sched.workers``.
"""

from __future__ import annotations

from typing import Any, cast

import msgspec

from core.common.ids import HostId, ProcId
from core.contracts.ledger import Ledger
from core.contracts.policy import COMPOSE, ORDER, ComposePolicy, OrderPolicy
from core.ipc.channels import Channels

from .drain import Drain
from .loop import Scheduler, Settings, Table


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        workers = msgspec.convert(config["sched.workers"], list[ProcId])
        if not workers:
            raise ValueError("sched.workers lists no worker")
        channels = cast(Channels, ctx.use("proc.streams@1"))
        control = ctx.use("proc.control@1")
        settings = Settings(max_batch=min(config["sched.max_batch"], config["batch.max_rows"]),
                            max_context=config["batch.max_context_len"], token_budget=config["sched.token_budget"],
                            step_timeout_s=config["sched.step_timeout_s"],
                            quiesce_timeout_s=config["sched.quiesce_timeout_s"],
                            frontend_timeout_s=config["sched.frontend_timeout_s"],
                            idle_wait_s=config["sched.idle_wait_s"], asleep=config["holder.sleep"],
                            draft_tokens=config["spec.num_draft_tokens"] if config["spec.algorithm"] else 0)
        self.loop = Scheduler(
            table=cast(Table, ctx.use("sched.reqs@1")), ledger=cast(Ledger, ctx.use("ledger@1")),
            order=cast(OrderPolicy, ctx.use(ORDER)), compose=cast(ComposePolicy, ctx.use(COMPOSE)),
            requests=channels.reader("requests", HostId(config["sched.frontend_host"])),
            results=[channels.reader(f"results.{rank}", w.host) for rank, w in enumerate(workers)],
            plans=[channels.writer(w, "plans") for w in workers],
            tokens_to=channels.writer_to, discard=channels.discard, identity=lambda: str(control.identity),
            depth=lambda: 2 if config["sched.overlap"] else 1, settings=settings)
        ctx.on_config("sched.overlap", lambda value: None)  # read when planning: no reload needed
        ctx.on_action("drain", Drain(self.loop))  # a change applies to an empty engine
        ctx.provide("proc.loop@1", self.loop)
        ctx.provide("sched.status@1", self.loop)
