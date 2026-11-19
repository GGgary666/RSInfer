# Copyright 2023-2024 SGLang Team
"""Request finishing adapted from SGLang Req (e838360, Apache-2.0).

See LICENSE.sglang. Retains output_ids, finished_reason/finished_len, and the stop
token before length-cap ordering from schedule_batch.py. RequestTable is our
integration: wire ids, scheduler-owned facts and deferred epoch retirement.

The scheduler records a step's row mapping when it submits the step. Sampling and
retirement are separate: only retire() after all ranks complete permits reclamation.
An aborted request can therefore finish for its client while GPU work still holds it.
"""

from __future__ import annotations

import secrets
from array import array
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import msgspec

from core.common.result import Result, Ok, fail
from core.contracts.columns import Column, ragged
from core.contracts.records import Sampled
from core.contracts.requests import FinishReason, NewRequest, TokenEvents
from core.ipc.stream import StreamAddress


@dataclass
class Request:
    input: NewRequest
    status: Literal["queued", "running", "finished"] = "queued"
    output_ids: list[int] = field(default_factory=list[int])
    pending: set[int] = field(default_factory=set[int])
    identities: list[str] = field(default_factory=list[str])
    finished_reason: FinishReason | None = None
    finished_len: int | None = None
    computed: int = 0
    row: int | None = None
    route: StreamAddress | None = None
    prompt_tokens: Sequence[int] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Keep the immutable wire buffer instead of boxing every long-context token.
        self.prompt_tokens = memoryview(self.input.prompt.data).cast("q")

    def finished(self) -> bool:
        return self.finished_reason is not None

    def finish_outside_step(self, reason: FinishReason) -> None:
        self.finished_reason = reason
        self.status = "finished"

    def update_finish_state(self, new_accepted_len: int = 1) -> None:
        if self.finished():
            return
        new_accepted_tokens = self.output_ids[-new_accepted_len:]
        for i, token_id in enumerate(new_accepted_tokens):
            if token_id in self.input.stop_token_ids:
                self.finished_reason = "stop"
                self.finished_len = len(self.output_ids) - new_accepted_len + i + 1
                break
        if self.finished_len is not None and self.finished_len > self.input.max_tokens:
            self.finished_reason, self.finished_len = "length", self.input.max_tokens
        if self.finished_reason is None and len(self.output_ids) >= self.input.max_tokens:
            self.finished_reason, self.finished_len = "length", self.input.max_tokens
        if self.finished():
            self.status = "finished"

    @property
    def reclaimable(self) -> bool:
        return self.status == "finished" and not self.pending


@dataclass
class Step:
    rows: dict[int, int]
    counts: dict[int, int] = field(default_factory=dict[int, int])
    sampled: bool = False


def token_events(identity: str, outputs: Mapping[int, list[int]],
                 finished: dict[int, FinishReason] | None = None) -> TokenEvents:
    offsets, tokens = ragged("i64", outputs.values())
    return TokenEvents(identity, Column.of("i64", outputs), offsets, tokens, finished or {})


class RequestTable:
    """``sched.reqs@1``. Called only by the scheduler's thread; holds no GPU resources."""

    def __init__(self) -> None:
        self.requests: dict[int, Request] = {}
        self.steps: dict[int, Step] = {}
        self.last_submitted = -1
        self.last_retired = -1

    def entries(self) -> Iterable[Request]:
        return self.requests.values()

    def get(self, request: int) -> Request | None:
        return self.requests.get(request)

    def add(self, request: NewRequest, *, route: StreamAddress | None = None) -> Result[Request]:
        if request.request in self.requests:
            return fail("sched.reqs", "add", "request id is already live", code="request.duplicate")
        if request.max_tokens < 1 or request.prompt.dtype != "i64" or not len(request.prompt):
            return fail("sched.reqs", "add", "a nonempty prompt and positive max_tokens are required",
                        code="request.invalid")
        if request.sampling.seed is None:
            request = msgspec.structs.replace(request, sampling=msgspec.structs.replace(
                request.sampling, seed=secrets.randbits(63)))
        entry = Request(request, route=route)
        self.requests[request.request] = entry
        return Ok(entry)

    def start_step(self, epoch: int, rows: Mapping[int, int], counts: Mapping[int, int] | None = None) -> None:
        """Snapshot the submitted row mapping; a later row reuse cannot misroute an old result."""
        if epoch <= self.last_submitted:
            raise ValueError("step epochs must increase")
        seen: set[int] = set()
        entries: list[Request] = []
        reqs = self.requests
        for rid in rows.values():
            if rid in seen:
                raise ValueError("a request occurs twice in the step")
            seen.add(rid)
            entry = reqs[rid]
            if entry.status == "finished":
                raise ValueError("a finished request cannot start another step")
            entries.append(entry)
        self.steps[epoch] = Step(dict(rows), dict(counts) if counts else {})
        self.last_submitted = epoch
        for request in entries:
            request.pending.add(epoch)
            request.status = "running"

    def sample(self, epoch: int, identity: str, sampled: Sampled) -> TokenEvents:
        """Accept one sampling result; suppress late output after stop, length or cancellation.

        This does not retire the step. The scheduler must collect completion from every
        participating rank before calling retire(), including for cancelled requests.
        """
        step = self.steps[epoch]
        if step.sampled:
            return token_events(identity, {})
        rows = cast(Sequence[int], sampled.rows.values())
        counts = cast(Sequence[int], sampled.counts.values())
        tokens = cast(Sequence[int], sampled.tokens.values())
        n_rows = len(rows)
        step_rows = step.rows
        if n_rows != len(counts):
            raise ValueError("malformed sampling result")
        if n_rows and len(tokens) == n_rows and all(c == 1 for c in counts):
            seen_rows: set[int] = set()
            for row in rows:
                if row in seen_rows or row not in step_rows:
                    raise ValueError("malformed sampling result")
                seen_rows.add(row)
            reqs = self.requests
            out_reqs = array("q")
            out_offs = array("i", [0])
            out_toks = array("q")
            finished: dict[int, FinishReason] = {}
            for row, tok in zip(rows, tokens):
                rid = step_rows[row]
                request = reqs[rid]
                if request.status == "finished":
                    continue
                if identity not in request.identities:
                    request.identities.append(identity)
                out_ids = request.output_ids
                inp = request.input
                if not inp.stop_token_ids and request.finished_len is None:
                    out_ids.append(tok)
                    if len(out_ids) >= inp.max_tokens:
                        request.finished_reason = "length"
                        request.finished_len = inp.max_tokens
                        request.status = "finished"
                        finished[rid] = "length"
                    out_reqs.append(rid)
                    out_toks.append(tok)
                    out_offs.append(len(out_toks))
                else:
                    start = len(out_ids)
                    out_ids.append(tok)
                    request.update_finish_state(1)
                    end = request.finished_len or len(out_ids)
                    emitted = out_ids[start:end]
                    del out_ids[end:]
                    if request.finished_reason is not None:
                        finished[rid] = request.finished_reason
                    if emitted or rid in finished:
                        out_reqs.append(rid)
                        out_toks.extend(emitted)
                        out_offs.append(len(out_toks))
            step.sampled = True
            return TokenEvents(identity, Column("i64", out_reqs.tobytes()),
                               Column("i32", out_offs.tobytes()),
                               Column("i64", out_toks.tobytes()), finished)
        if len(set(rows)) != n_rows or any(n < 0 for n in counts) \
                or sum(counts) != len(tokens) or any(row not in step_rows for row in rows):
            raise ValueError("malformed sampling result")
        outputs: dict[int, list[int]] = {}
        finished = {}
        offset = 0
        for row, count in zip(rows, counts, strict=True):
            rid = step_rows[row]
            request = self.requests[rid]
            incoming = tokens[offset:offset + count]
            offset += count
            if request.status == "finished":
                continue
            if identity not in request.identities:
                request.identities.append(identity)
            start = len(request.output_ids)
            request.output_ids.extend(incoming)
            if incoming:
                request.update_finish_state(len(incoming))
            end = request.finished_len or len(request.output_ids)
            # As in mini-sglang's DetokenizeMsg, include the final stop token on
            # the wire; the detokenizer trims it and usage still counts it.
            emitted = request.output_ids[start:end]
            del request.output_ids[end:]
            if request.finished_reason is not None:
                finished[rid] = request.finished_reason
            if emitted or rid in finished:
                outputs[rid] = emitted
        step.sampled = True
        return token_events(identity, outputs, finished)

    def cancel(self, request: int, identity: str, *, reason: FinishReason = "abort") -> TokenEvents:
        """End client-visible output immediately; keep the entry until its submitted steps retire."""
        entry = self.requests.get(request)
        if entry is None or entry.status == "finished":
            return token_events(identity, {})
        entry.finish_outside_step(reason)
        if identity not in entry.identities:
            entry.identities.append(identity)
        return token_events(identity, {request: []}, {request: reason})

    def retire(self, epoch: int, ran: bool = True, advanced: Mapping[int, int] | None = None) -> list[int]:
        """All ranks finished this step. Return terminal requests whose resources may now go.
        ``advanced``: row -> tokens a speculative decode row accepted (doc/specs/mtp.md), instead of
        the count the step was planned with."""
        step = self.steps.pop(epoch)
        ready: list[int] = []
        for row, rid in step.rows.items():
            request = self.requests[rid]
            if ran:
                request.computed += advanced[row] if advanced is not None and row in advanced else \
                    step.counts.get(row, 0)
            request.pending.remove(epoch)
            if request.reclaimable:
                ready.append(rid)
        self.last_retired = max(self.last_retired, epoch)
        return ready

    def reap(self, request: int) -> Request:
        entry = self.requests[request]
        if not entry.reclaimable:
            raise ValueError("request still active or has steps in flight")
        return self.requests.pop(request)


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        ctx.provide("sched.reqs@1", RequestTable())
