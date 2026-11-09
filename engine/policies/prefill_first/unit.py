"""``policy.compose@1``: prefill first, no mixed steps.

As sglang by default (``get_next_batch_to_run``: "Run prefill first if possible";
``enable_mixed_chunk = False``) and mini-sglang (9a91cfa, ``Scheduler._schedule_next_batch``:
the prefill batch, else the decode batch):

- If any prompt is left to compute, the step is a prefill step: first the requests already
  admitted whose prompt is partly computed (the chunked request goes first in both), then the
  waiting requests in the order policy's order, while the token budget lasts. The last one
  takes what is left of the budget, so its prompt is chunked (``PrefillAdder``).
- Otherwise every decoding request decodes one token.
"""

from __future__ import annotations

from typing import Any

from core.contracts.policy import COMPOSE, StepView, Take


class PrefillFirst:
    def compose(self, view: StepView) -> list[Take]:
        takes: list[Take] = []
        budget, rows = view.token_budget, view.free_rows
        for c in view.prefilling:
            if budget <= 0:
                break
            n = min(budget, c.prompt - c.planned)
            takes.append(Take(c.request, n))
            budget -= n
        for c in view.waiting:
            if budget <= 0 or rows <= 0:
                break
            n = min(budget, c.prompt - c.planned)
            takes.append(Take(c.request, n))
            budget -= n
            rows -= 1
        if takes:
            return takes
        return [Take(c.request, 1) for c in view.decoding]


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        ctx.provide(COMPOSE, PrefillFirst())
