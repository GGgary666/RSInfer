"""The worker loop's one seam to what runs a step (``executor@1``, stage2-contract.md §4).

The executor hides the persistent batch, the backends, the model and the sampler behind it, as
SGLang's ``ModelRunner`` does behind ``forward`` and ``sample``. A step is submitted and its result
taken later, so one step of overlap needs no other interface (§4.4).
"""

from __future__ import annotations

from typing import Protocol

from core.contracts.records import BatchPlan, Capacity, Resync, RowDelta, Sampled, StateOps, StepStats


class PendingStep(Protocol):
    def ready(self) -> bool:
        """The device work of the step has finished."""
        ...
    def finish(self) -> Sampled:
        """Waits for the step; its sampled tokens."""
        ...
    def stats(self) -> StepStats: ...


class Executor(Protocol):
    def capacity(self) -> Capacity:
        """Items per state kind; the worker sends it at start and whenever its answers go to a new
        reader."""
        ...
    def submit(self, epoch: int, delta: RowDelta, plan: BatchPlan, state: StateOps) -> PendingStep | Resync:
        """Run step ``epoch``, or answer ``Resync`` having changed nothing and queued no device work
        (the persistent batch cannot run the plan, ``PersistentBatch.check``). Raises on a malformed
        plan or a failure after device work was queued: the loop then cannot know what ran."""
        ...
    def synchronize(self) -> None:
        """All submitted work has finished (the worker's ``quiesce()``)."""
        ...
