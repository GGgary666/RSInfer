"""The scheduler's decision points, ``policy.order@1`` and ``policy.compose@1`` (stage2-contract.md §2).

A policy unit gets a read-only snapshot and returns a decision; ``core.sched`` checks the decision
and carries it out. Whether a request fits the state capacity is ``core.sched``'s question to the
ledger, never the policy's. The defaults (``policies.fcfs``, ``policies.prefill_first``) follow
sglang's: first come first served, prefill first, no mixed steps.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import msgspec

ORDER = "policy.order@1"
COMPOSE = "policy.compose@1"


class Candidate(msgspec.Struct, frozen=True):
    """One live request, as of the steps already planned (some may still be running)."""

    request: int
    arrival: int
    """When the scheduler took it, as a sequence number: smaller is earlier."""
    prompt: int
    """Prompt length in tokens."""
    planned: int
    """Prompt tokens planned so far, computed or in flight."""
    output: int
    """Output tokens sampled so far, counting the ones steps in flight will sample."""
    max_tokens: int


class OrderPolicy(Protocol):
    def order(self, waiting: Sequence[Candidate]) -> Sequence[int]:
        """Which waiting requests to try admitting, in order. ``waiting`` is in arrival order.
        ``core.sched`` admits in this order and stops at the first that does not fit (sglang's and
        mini-sglang's ``break``); a request left out waits for a later step."""
        ...


class Take(msgspec.Struct, frozen=True):
    request: int
    tokens: int


class StepView(msgspec.Struct, frozen=True):
    waiting: list[Candidate]
    """Not admitted yet, in the order policy's order."""
    prefilling: list[Candidate]
    """Admitted; part of the prompt is not planned yet. In arrival order."""
    decoding: list[Candidate]
    """The whole prompt is planned and the request may sample more. In arrival order."""
    token_budget: int
    """Prompt tokens one step may compute (sglang's ``chunked_prefill_size``); decodes do not
    count against it, as in sglang's steps that do not mix."""
    free_rows: int
    """Rows of the persistent batch a waiting request can be admitted to."""


class ComposePolicy(Protocol):
    def compose(self, view: StepView) -> Sequence[Take]:
        """What the next step computes: which requests, and how many tokens each. A waiting or
        prefilling request takes 1 to its unplanned prompt tokens, a decoding one exactly 1; at
        most ``token_budget`` prompt tokens in all and ``free_rows`` waiting requests. Empty:
        nothing.

        If no waiting request in the decision could be admitted and nothing else is in it,
        ``core.sched`` asks again with ``waiting`` empty (sglang runs decode when no prefill
        batch could be formed)."""
        ...
