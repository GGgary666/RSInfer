"""Late requests actually sent by transaction 1 must not alter transaction 2."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

import msgspec
import pytest

from core.change.types import ChangeSet
from core.change.protocol import Abort
from core.common.ids import ProcId
from core.common.result import Result
from core.supervisor.protocol import Exited, Heartbeat, ProcBeat
from core.supervisor.supervisor import _Host
from platforms.base import PlatformInfo
from tests.framework.change.harness import Capture, Harness, harness as harness
from unit.runtime.host import ProcessHost


@pytest.mark.parametrize("first_outcome,late_kind", [
    ("aborted", "Prepare"), ("aborted", "Pause"), ("aborted", "Execute"), ("aborted", "Abort"),
    ("committed", "Commit"), ("committed", "Resume"),
])
@pytest.mark.parametrize("checkpoint", ["Prepare", "Execute", "Commit", "Resume"])
def test_previous_transaction_cannot_change_the_next_transaction(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, first_outcome: str, late_kind: str, checkpoint: str,
) -> None:
    h = harness
    old_messages: dict[str, msgspec.Struct] = {}
    counters: dict[str, int] = {}
    unit = h.unit_of(h.target)

    def counted(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def invoke(*args: Any, **kwargs: Any) -> Any:
            counters[name] = counters.get(name, 0) + 1
            return original(*args, **kwargs)
        return invoke

    for name in ["prepare", "execute", "abort", "commit"]:
        monkeypatch.setattr(unit, name, counted(name, getattr(unit, name)))
    original = h.sup.call
    second = False
    probed = False

    def snapshot() -> tuple[object, ...]:
        host = h.hosts[h.target]
        c = host.participant.change
        return (host.identity, host.paused, c.id if c else None, c.phase if c else None,
                unit.version, unit.commits, tuple(sorted(counters.items())),
                tuple(sorted(host.participant.finished)))

    async def call(proc: ProcId, msg: msgspec.Struct, *, timeout: float | None = None) -> Result[msgspec.Struct]:
        nonlocal probed
        kind = type(msg).__name__
        if not second and proc == h.target:
            old_messages[kind] = msg
        reply = await original(proc, msg, timeout=timeout)
        if second and proc == h.target and kind == checkpoint and not probed:
            probed = True
            before = snapshot()
            for _ in range(2):
                await original(proc, old_messages[late_kind], timeout=timeout)
                assert snapshot() == before, f"late {late_kind} changed transaction 2 at {checkpoint}"
        return reply

    monkeypatch.setattr(h.sup, "call", call)

    async def main() -> None:
        nonlocal second
        await h.coord.start()
        if first_outcome == "aborted":
            h.coord.faults.add("after_execute")
        first = await h.submit()
        assert first.status == first_outcome
        assert late_kind in old_messages, "must replay a message the coordinator really sent"
        second = True
        following = await h.coord.submit(ChangeSet(base=h.coord.identity, config={"revision": 3}))
        assert following.change_id != first.change_id
        assert probed and following.status == "committed", following.diagnostics
        assert not h.coord.recovery_required
        assert all(host.identity == following.new_identity and not host.paused for host in h.hosts.values())
        assert counters == {"prepare": 2, "execute": 2, "commit": 1 if first_outcome == "aborted" else 2,
                            **({"abort": 1} if first_outcome == "aborted" else {})}

    asyncio.run(main())


def test_abort_replace_one_peer_then_replay_old_messages_during_next_execute(
    harness: Harness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace the actual ProcessHost after confirmed rollback; preserve the other peer.

    Old transaction requests go to the surviving endpoint. Old incarnation
    reports go to the supervisor; they must not mark the replacement LOST.
    This does not assert that change messages carry an incarnation token.
    """
    h = harness
    original = h.sup.call
    first_id = 0
    probed = False

    async def call(proc: ProcId, msg: msgspec.Struct, *, timeout: float | None = None) -> Result[msgspec.Struct]:
        nonlocal probed
        reply = await original(proc, msg, timeout=timeout)
        if first_id and type(msg).__name__ == "Execute" and proc == h.target and not probed:
            probed = True
            survivor = h.hosts[h.first]
            c = survivor.participant.change
            before = (c.id if c else None, c.phase if c else None, survivor.paused, h.unit_of(h.first).version)
            await original(h.first, Abort(first_id), timeout=timeout)
            c = survivor.participant.change
            assert (c.id if c else None, c.phase if c else None, survivor.paused, h.unit_of(h.first).version) == before
            h.sup._exited(Exited(h.target, 1, 9))
            h.sup._heartbeat(Heartbeat(h.target.host, [ProcBeat(h.target, 1, "exited", 9)]))
            assert h.sup.procs[h.target].state == "RUNNING"
        return reply

    monkeypatch.setattr(h.sup, "call", call)

    async def main() -> None:
        nonlocal first_id
        await h.coord.start()
        h.coord.faults.add("after_execute")
        first = await h.submit()
        assert first.status == "aborted" and not h.coord.recovery_required
        old = h.hosts[h.target]
        plan = old.spec.plan
        spec = msgspec.structs.replace(old.spec, incarnation=2)
        old.shutdown()
        replacement = h.hosts[h.target] = ProcessHost(spec)
        monkeypatch.setattr(replacement.rpc, "defer", Capture)
        monkeypatch.setattr(replacement.client, "call", h.query)
        replacement.runtime.load_process_plan(plan).unwrap()
        assert not replacement.participant.finished, "the new process has no old in-memory tombstones"
        h.sup.procs[h.target].spec = spec
        h.sup.hosts[h.target.host] = _Host(h.target.host, "unused", PlatformInfo("fake"), time.monotonic())
        first_id = first.change_id
        following = await h.coord.submit(ChangeSet(base=h.coord.identity, config={"revision": 3}))
        assert probed and following.status == "committed", following.diagnostics
        assert all(host.identity == following.new_identity and not host.paused for host in h.hosts.values())
        assert all(h.unit_of(p).version == 2 for p in h.hosts)

    asyncio.run(main())
