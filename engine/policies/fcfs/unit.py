"""``policy.order@1``: first come, first served.

sglang's default ``schedule_policy = "fcfs"``, and the only order of mini-sglang (9a91cfa,
``PrefillManager.pending_list``): requests are tried in the order they arrived, and the scheduler
stops at the first that does not fit, so no later request overtakes it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from core.contracts.policy import ORDER, Candidate


class Fcfs:
    def order(self, waiting: Sequence[Candidate]) -> list[int]:
        return [c.request for c in sorted(waiting, key=lambda c: c.arrival)]


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        ctx.provide(ORDER, Fcfs())
