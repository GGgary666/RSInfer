# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""The speculative step of ``executors.pytorch`` (doc/specs/mtp.md): SGLang's ``EAGLEWorkerV2`` for a
chain (srt/speculative/eagle_worker_v2.py, eagle_worker_common.py, spec_utils.py, e838360;
LICENSE.sglang), eager.

A decode step of D tokens per row, from the length ``L`` the row is at, its last accepted token ``b``
(``last_token``) and its first draft ``d1`` with the draft model's hidden state there (``spec.drafts``,
``spec.hidden``, left by its last step):

1. ``draft``: the draft model's D - 2 decode forwards, ``d1`` at ``L``, then each argmax at the next;
2. ``verify``: the target over ``[b, d1 .. d_{D-1}]`` at ``L .. L + D - 1``;
3. ``accept`` (``spec.accept@1``), and the GDN states of the accepted step committed (``commit``,
   with the prefix cache's boundary: ``StateTrack`` ``upto = -grid``);
4. ``draft_extend``: the draft model over the same D positions on the target's hidden states and
   predictions, its row at the accepted end giving the next ``d1`` and hidden state.

A prefill step runs the draft model's extend over its chunk after the target's, the inputs shifted by
one (``_draft_extend_for_prefill``): the last one is the next prompt token (``BatchPlan.next_tokens``)
or, ending the prompt, the token just sampled.

Where a row is: a row with ``from_previous`` is where its last step left it (``committed``), which
the scheduler may not know yet; that step's accepted lengths are copied to the host once its verify
finished, and the next step waits for that (``settle``), as SGLang's next step waits for the
published ``new_seq_lens``. A row without drafts of its own (added again by a full delta) verifies
placeholder drafts and accepts one token: a plain decode step, then its draft extend.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from core.contracts.columns import Column
from core.contracts.records import BatchPlan

KIND = "gdn.slots"


def on_device(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type != "cuda":
        return values.to(device)
    return values.pin_memory().to(device, non_blocking=True)


@dataclass
class Accepted:
    """A decode step's accepted lengths on their way to the host."""

    rows: list[int]
    starts: list[int]
    lengths: torch.Tensor
    """``[rows]`` on the host (pinned on CUDA); filled once ``event`` passed."""
    event: Any


def view(rows: Sequence[int], num_tokens: Sequence[int], positions: Sequence[int], phase: str,
         draft_tokens: int = 0) -> BatchPlan:
    """A step as the backends see one forward of it."""
    return BatchPlan(Column.of("i32", rows), Column.of("i32", num_tokens), Column.of("i32", positions),
                     Column.of("i64", []), Column.of("u8", [0] * len(rows)), phase, 0, draft_tokens)  # type: ignore[arg-type]


def track_steps(starts: torch.Tensor, accept_lens: torch.Tensor, grid: int) -> torch.Tensor:
    """``_verify_commit_step_indices``: the step whose state is at the boundary of ``grid`` the accepted
    tokens cross, or -1."""
    end = starts + accept_lens
    crosses = starts // grid != end // grid
    return torch.where(crosses, (end // grid * grid - starts - 1).clamp_min(0), torch.full_like(starts, -1))


def spec_draft_tokens(config: Any) -> int:
    """``spec.num_draft_tokens`` when speculation is on, else 0; the keys checked as SGLang checks them
    (doc/specs/mtp.md §2.1)."""
    algorithm = config["spec.algorithm"]
    if not algorithm:
        return 0
    if algorithm != "eagle":
        raise ValueError(f"spec.algorithm = {algorithm!r}: only \"\" (off) and \"eagle\" (the MTP layer drafts)")
    steps, topk, tokens = config["spec.num_steps"], config["spec.eagle_topk"], config["spec.num_draft_tokens"]
    if topk != 1:
        raise ValueError(f"spec.eagle_topk = {topk}: only a chain (1) is supported")
    if steps < 1 or tokens != steps + 1:
        raise ValueError(f"spec.num_draft_tokens = {tokens} must be spec.num_steps + 1 = {steps + 1} for a chain")
    return tokens


class Speculative:
    def __init__(self, plans: Any, batch: Any, sampler: Any, draft_tokens: int, hidden_size: int, dtype: str) -> None:
        self.plans, self.batch, self.sampler = plans, batch, sampler
        self.draft_tokens = draft_tokens
        batch.prepare_spec(draft_tokens, hidden_size, dtype)
        self.pending: Accepted | None = None

    def settle(self) -> None:
        """The last decode step's rows are where its accepted lengths left them (waits for its verify)."""
        p, self.pending = self.pending, None
        if p is None:
            return
        if p.event is not None:
            p.event.synchronize()
        rows = [r for r in p.rows if r in self.batch.rows]  # a row removed since keeps nothing
        lengths = dict(zip(p.rows, (int(x) for x in p.lengths.tolist()), strict=True))
        starts = dict(zip(p.rows, p.starts, strict=True))
        self.batch.speculated(rows, [starts[r] + lengths[r] for r in rows])

    def prefill(self, plan: BatchPlan, rows: list[int], num_tokens: list[int], positions: list[int],
                input_ids: torch.Tensor, pos: torch.Tensor, last_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The target's prefill and the draft model's extend; the sampled token of each row, and the logits."""
        logits, hidden = self.plans.prefill_hidden(self.batch, plan, input_ids, pos, last_index)
        sampled = self.sampler.sample(logits, self.batch.sampling(rows),
                                      [p + n - 1 for p, n in zip(positions, num_tokens, strict=True)]).to(torch.int64)
        nexts = [int(x) for x in plan.next_tokens.values()]
        tails = torch.where(on_device(torch.tensor(nexts, dtype=torch.int64), sampled.device) >= 0,
                            on_device(torch.tensor(nexts, dtype=torch.int64), sampled.device), sampled)
        shifted = torch.roll(input_ids, -1)  # each row's inputs one on; the last of each row is its tail
        shifted[last_index] = tails
        draft_logits, draft_hidden = self.plans.draft(self.batch, plan, shifted, pos, hidden, last_index)
        self._keep(rows, draft_logits.argmax(dim=-1), draft_hidden)
        self.batch.record(rows, sampled)
        self.batch.speculated(rows, [p + n for p, n in zip(positions, num_tokens, strict=True)])
        return sampled, logits

    def decode(self, plan: BatchPlan, rows: list[int], positions: list[int], bonus: torch.Tensor,
               state_track: tuple[list[int], list[int], list[int]]) -> tuple[torch.Tensor, torch.Tensor, Accepted]:
        """One verify step; the tokens ``[rows, D]`` and accepted lengths ``[rows]`` on the device."""
        d, bs, batch = self.draft_tokens, len(rows), self.batch
        prev = [int(x) for x in plan.from_previous.values()]
        starts = [c if f and (c := batch.committed(r)) is not None else p for r, p, f in zip(rows, positions, prev, strict=True)]
        # Drafts are the row's own once it ran since it was added (a full delta adds every row again),
        # whether the scheduler names its token (from_previous, a step in flight) or gives it.
        own = [batch.has_drafts(r) for r in rows]
        tables = batch.binding.buffers()
        device = bonus.device
        idx = on_device(torch.tensor(rows, dtype=torch.int64) + 1, device)
        start = on_device(torch.tensor(starts, dtype=torch.int64), device)
        lanes = torch.arange(bs, device=device)
        drafts = tables["spec.drafts"][idx].clone()
        hidden = tables["spec.hidden"][idx].clone()
        token = drafts[:, 0]
        for i in range(d - 2):  # draft_forward: topk 1, each argmax the next one's input
            logits, hidden = self.plans.draft(batch, view(rows, [1] * bs, [s + i for s in starts], "decode"),
                                              token, start + i, hidden, lanes)
            token = logits.argmax(dim=-1)
            drafts[:, i + 1] = token
        steps = torch.arange(d, device=device)
        positions_all = (start[:, None] + steps).flatten()
        input_ids = torch.cat([bonus.view(-1, 1), drafts], dim=1).flatten()
        logits, target_hidden = self.plans.verify(batch, view(rows, [d] * bs, starts, "decode", d), input_ids,
                                                  positions_all)
        logits = logits.view(bs, d, -1)
        params = batch.sampling(rows)
        tokens, accept = self.sampler.accept(logits, drafts, params, start)
        if not all(own):  # placeholder drafts: one token, drawn as a plain decode step draws it
            plain = self.sampler.sample(logits[:, 0], params, starts).to(tokens.dtype)
            mask = on_device(torch.tensor(own), device)
            tokens[:, 0] = torch.where(mask, tokens[:, 0], plain)
            accept = torch.where(mask, accept, torch.ones_like(accept))
        accept = accept.to(torch.int64)
        _, _, uptos = state_track
        grids = {-u for u in uptos if u < 0}
        if len(grids) > 1:
            raise ValueError(f"a speculative step's rows track at different grids {sorted(grids)}")
        kept = track_steps(start, accept, grids.pop()) if grids else torch.full_like(start, -1)
        self.plans.commit(accept - 1, kept)
        batch.record(rows, tokens.gather(1, (accept - 1).view(-1, 1)).view(-1))
        draft_logits, draft_hidden = self.plans.draft(batch, view(rows, [d] * bs, starts, "prefill"), tokens.flatten(),
                                                      positions_all, target_hidden, lanes * d + accept - 1)
        self._keep(rows, draft_logits.argmax(dim=-1), draft_hidden)
        lengths = accept
        event = None
        if device.type == "cuda":
            lengths = torch.empty(accept.shape, dtype=accept.dtype, pin_memory=True)
            lengths.copy_(accept, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
        done = Accepted(rows, starts, lengths, event)
        self.pending = done
        return tokens, accept, done

    def _keep(self, rows: list[int], first: torch.Tensor, hidden: torch.Tensor) -> None:
        tables = self.batch.binding.buffers()
        idx = on_device(torch.tensor(rows, dtype=torch.int64) + 1, first.device)
        tables["spec.drafts"][idx, 0] = first.to(torch.int64)
        tables["spec.hidden"][idx] = hidden.to(tables["spec.hidden"].dtype)
