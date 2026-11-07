"""Check transaction-model obligations at the real transport boundary.

These are implementation contract tests, not a claim of full trace refinement.
"""
from __future__ import annotations

import asyncio
import time

import msgspec
import pytest

from core.change.protocol import Abort, Commit, Committed, Resume
from core.change.types import ActionPlan
from core.common.ids import ProcId
from core.common.result import Failed, Ok, Result
from core.supervisor.supervisor import _Host
from platforms.base import PlatformInfo
from tests.framework.change.harness import Harness, OLD, harness as harness


@pytest.mark.parametrize("kind", ["Commit", "Resume"])
@pytest.mark.parametrize("drop_reply", [False, True], ids=["request", "reply"])
@pytest.mark.parametrize("permanent", [False, True], ids=["once", "forever"])
def test_durable_commit_barrier_and_failed_confirmation_freeze(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, kind: str, drop_reply: bool, permanent: bool,
) -> None:
    h = harness
    h.drop_kind, h.drop_reply, h.drop_count = kind, drop_reply, -1 if permanent else 1
    original = h.sup.call
    confirmed: set[ProcId] = set()

    async def call(proc: ProcId, msg: msgspec.Struct, *, timeout: float | None = None) -> Result[msgspec.Struct]:
        if isinstance(msg, Commit):
            assert h.coord.log.decision(msg.change_id) == "commit", "send must follow durable decision"
        if isinstance(msg, Resume):
            assert confirmed == set(h.hosts), "Resume needs received acknowledgements, not just changed labels"
        reply = await original(proc, msg, timeout=timeout)
        if isinstance(msg, Commit) and isinstance(reply, Ok) and isinstance(reply.value, Committed):
            assert reply.value.change_id == msg.change_id
            confirmed.add(proc)
        return reply

    def observe() -> None:
        new_serving = [host for host in h.hosts.values() if not host.paused and host.identity != OLD]
        if new_serving:
            assert all(host.identity == new_serving[0].identity for host in h.hosts.values())
            assert all(h.unit_of(p).version == 2 for p in h.hosts)

    monkeypatch.setattr(h.sup, "call", call)
    h.observe = observe

    async def main() -> None:
        await h.coord.start()
        r = await h.submit()
        assert r.status == ("failed" if permanent else "committed")
        assert h.coord.log.decision(r.change_id) == "commit"
        assert not any(message == "Abort" for _, message in h.messages)
        assert h.coord.recovery_required == permanent
        if permanent:
            before = list(h.messages)
            assert (await h.submit()).status == "failed"
            assert h.messages == before
        else:
            assert all(not host.paused for host in h.hosts.values())

    asyncio.run(main())


def test_lost_is_not_stopped(harness: Harness) -> None:
    h = harness
    rec = h.sup.procs[h.target]
    h.sup.hosts[h.target.host] = _Host(h.target.host, "unreachable", PlatformInfo("fake"), time.monotonic(), lost=True)
    rec.state = "LOST"
    h.action_plan = ActionPlan(procs=[h.action_plan.procs[0]], stop=[h.target])

    async def main() -> None:
        await h.coord.start()
        r = await h.submit()
        assert r.status == "failed" and h.coord.recovery_required
        assert h.sup.procs[h.target] is rec and rec.state == "LOST"
        assert not h.hosts[h.target].paused and h.unit_of(h.target).version == 1
        assert (await h.submit()).status == "failed"

    asyncio.run(main())


def test_failed_undo_cannot_be_hidden_by_a_label_or_duplicate(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    h = harness

    def abort() -> None:
        raise RuntimeError("resource rollback failed")

    monkeypatch.setattr(h.unit_of(h.target), "abort", abort)

    async def main() -> None:
        await h.coord.start()
        h.coord.faults.add("after_execute")
        r = await h.submit()
        host = h.hosts[h.target]
        assert host.identity == OLD and h.unit_of(h.target).version == 2
        assert r.status == "failed" and h.coord.recovery_required and host.paused
        for _ in range(2):
            assert isinstance(await h.call(h.target, Abort(r.change_id)), Failed)
            assert host.paused
        assert (await h.submit()).status == "failed"

    asyncio.run(main())
