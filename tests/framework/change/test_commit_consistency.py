"""Fault injection at the transport boundary, with real coordinator/participants.

The original implementation must run these tests too: do not import new protocol
messages at collection time. A timeout is simulated by advancing the participant's
deadline, not by sleeping. Unit behavior is checked independently of identity labels.
"""

from __future__ import annotations

import asyncio
import time
from types import MethodType

import msgspec
import pytest

from core.change.protocol import Commit
from core.change.types import ActionPlan, ProcActions
from core.common.ids import HostId, ProcId
from core.common.result import Failed, Ok, Result, fail
from core.ipc.rpc import SyncRpcServer
from core.supervisor.protocol import ProcSpec
from core.supervisor.supervisor import _Proc
from core.supervisor.topology import DesiredProc, HostSpec, Mesh, RankPlacement, RoleSpec, Topology
from tests.framework.change.harness import Capture, Harness, OLD, harness as harness
from unit.runtime.host import ProcessHost


@pytest.mark.parametrize("drop_reply", [False, True], ids=["lost-request", "lost-reply"])
def test_lost_commit_is_retried_until_every_participant_serves_new_version(harness: Harness, drop_reply: bool) -> None:
    h = harness
    h.drop_count = 1
    h.drop_reply = drop_reply

    async def main() -> None:
        await h.coord.start()
        result = await h.submit()
        h.expire()
        assert result.status == "committed", result.diagnostics
        for p, host in h.hosts.items():
            assert host.identity == result.new_identity != OLD
            assert not host.paused
            assert h.unit_of(p).version == 2
            assert h.unit_of(p).commits == 1
            assert h.sup.procs[p].spec.identity == host.identity
        assert h.messages.count((h.target, "Commit")) == 2

    asyncio.run(main())


def test_lost_detector_cannot_turn_unacknowledged_commit_into_success(
    harness: Harness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = harness
    h.drop_count = -1
    monkeypatch.setattr(h.hosts[h.target].client, "call", lambda *args, **kw:
                        fail("test", "query", "partitioned control plane", code="ipc.timeout"))

    async def call(proc: ProcId, msg: msgspec.Struct, *, timeout: float | None = None) -> Result[msgspec.Struct]:
        if proc == h.target and isinstance(msg, Commit):
            h.sup._lose_proc(h.sup.procs[proc], "agent heartbeat timed out")
        return await h.call(proc, msg, timeout=timeout)

    monkeypatch.setattr(h.sup, "call", call)

    async def main() -> None:
        await h.coord.start()
        result = await h.submit()
        assert result.status == "failed", result
        assert all(host.paused for host in h.hosts.values())
        assert all(h.unit_of(p).version == 2 for p in h.hosts)
        assert not any(kind == "Resume" for _, kind in h.messages)

    asyncio.run(main())


def test_query_timeout_cannot_resume_old_version_alongside_new_version(
    harness: Harness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = harness
    h.drop_count = 1
    monkeypatch.setattr(h.hosts[h.target].client, "call", lambda *args, **kw:
                        fail("test", "query", "reply lost while coordinator lives", code="ipc.timeout"))
    split: list[dict[ProcId, int]] = []

    def observe() -> None:
        serving = {p: h.unit_of(p).version for p, host in h.hosts.items() if not host.paused}
        if len(set(serving.values())) > 1:
            split.append(serving)

    h.observe = observe

    async def main() -> None:
        await h.coord.start()
        result = await h.submit()
        assert not split, split
        assert result.status == "committed", result.diagnostics
        assert all(host.identity == result.new_identity for host in h.hosts.values())

    asyncio.run(main())


def test_confirmation_deadline_bounds_a_stuck_rpc_and_blocks_next_change(
    harness: Harness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = harness
    h.coord.tun = msgspec.structs.replace(h.coord.tun, commit_ack_timeout_s=0.05)
    cancelled: list[ProcId] = []

    async def call(proc: ProcId, msg: msgspec.Struct, *, timeout: float | None = None) -> Result[msgspec.Struct]:
        if proc == h.target and isinstance(msg, Commit):
            try:
                await asyncio.Future[None]()  # Simulate a transport that never returns.
            finally:
                cancelled.append(proc)
        return await h.call(proc, msg, timeout=timeout)

    monkeypatch.setattr(h.sup, "call", call)

    async def main() -> None:
        await h.coord.start()
        started = time.monotonic()
        result = await asyncio.wait_for(h.submit(), timeout=1.0)
        assert 0.04 <= time.monotonic() - started < 1.0
        assert result.status == "failed" and cancelled == [h.target]
        assert all(host.paused for host in h.hosts.values())
        assert h.coord.decision_of(result.change_id) == "commit"
        count = len(h.messages)
        assert (await h.submit()).status == "failed"
        assert len(h.messages) == count

    asyncio.run(main())


@pytest.mark.parametrize("field", ["change_id", "identity", "rolled_back"])
def test_invalid_confirmation_is_rejected(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, field: str,
) -> None:
    h = harness

    async def call(proc: ProcId, msg: msgspec.Struct, *, timeout: float | None = None) -> Result[msgspec.Struct]:
        reply = await h.call(proc, msg, timeout=timeout)
        phase = "Resume" if field == "rolled_back" else "Commit"
        if proc == h.target and type(msg).__name__ == phase and isinstance(reply, Ok):
            wrong: object = {"change_id": -1, "identity": OLD, "rolled_back": True}[field]
            return Ok(msgspec.structs.replace(reply.value, **{field: wrong}))
        return reply

    monkeypatch.setattr(h.sup, "call", call)

    async def main() -> None:
        await h.coord.start()
        result = await h.submit()
        assert result.status == "failed"
        assert any(d.code == "change.recovery_required" for d in result.diagnostics)
        if field != "rolled_back":
            assert all(host.paused for host in h.hosts.values())

    asyncio.run(main())


def test_commit_and_resume_cross_real_rpc_with_lost_commit_reply(
    harness: Harness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = harness
    for host in h.hosts.values():
        monkeypatch.setattr(host.rpc, "defer", MethodType(SyncRpcServer.defer, host.rpc))
    dropped = False

    async def call(proc: ProcId, msg: msgspec.Struct, *, timeout: float | None = None) -> Result[msgspec.Struct]:
        nonlocal dropped
        r = await h.sup.client.call(h.hosts[proc].rpc.address, msg, timeout=timeout or 1.0)
        if proc == h.target and isinstance(msg, Commit) and not dropped:
            assert not isinstance(r, Failed), r
            dropped = True
            return fail("test", "rpc", "reply lost", code="ipc.timeout")
        return r

    monkeypatch.setattr(h.sup, "call", call)

    async def serve() -> None:
        while True:
            for host in h.hosts.values():
                host.safe_point()
            await asyncio.sleep(0.001)

    async def main() -> None:
        serving = asyncio.create_task(serve())
        try:
            await h.coord.start()
            result = await asyncio.wait_for(h.submit(), timeout=3.0)
            assert dropped and result.status == "committed", result.diagnostics
            assert all(host.identity == result.new_identity and not host.paused for host in h.hosts.values())
            assert all(h.unit_of(p).commits == 1 for p in h.hosts)
        finally:
            serving.cancel()
            await asyncio.gather(serving, return_exceptions=True)

    asyncio.run(main())


@pytest.mark.parametrize("lost_target", [0, 1])
@pytest.mark.parametrize("drop_reply", [False, True], ids=["lost-request", "lost-reply"])
def test_no_participant_serves_before_global_commit_confirmation(
    harness: Harness, lost_target: int, drop_reply: bool,
) -> None:
    """Observe the interval between deliveries, not just the final state.

    The early-resume mutant passes the initial tests by fencing again at timeout.
    Serving during that interval is already a violation that a later pause cannot undo.
    """
    h = harness
    h.target = list(h.hosts)[lost_target]
    h.drop_count = 1
    h.drop_reply = drop_reply
    premature_service: list[tuple[ProcId, int]] = []

    def observe() -> None:
        if any(host.identity == OLD for host in h.hosts.values()):
            premature_service.extend((p, h.unit_of(p).version) for p, host in h.hosts.items()
                                     if host.identity != OLD and not host.paused)

    h.observe = observe

    async def main() -> None:
        await h.coord.start()
        result = await h.submit()
        assert not premature_service, f"served before all participants committed: {premature_service}"
        assert result.status == "committed", result.diagnostics
        assert all(not host.paused and host.identity == result.new_identity for host in h.hosts.values())

    asyncio.run(main())


def test_persistent_commit_loss_is_not_reported_as_success(harness: Harness) -> None:
    h = harness
    h.drop_count = -1

    async def main() -> None:
        await h.coord.start()
        result = await h.submit()
        assert result.status == "failed", result
        assert any(d.code == "change.recovery_required" for d in result.diagnostics)
        assert h.coord.identity == result.new_identity != OLD
        assert all(host.paused for host in h.hosts.values())
        assert all(h.unit_of(p).version == 2 for p in h.hosts)
        assert not any(kind in ("Resume", "Abort") for _, kind in h.messages)
        count = len(h.messages)
        blocked = await h.submit()
        assert blocked.status == "failed"
        assert len(h.messages) == count  # No subsequent transaction enters an unresolved state.
        assert h.coord.log.events("commit_decided")

    asyncio.run(main())


@pytest.mark.parametrize("drop_reply", [False, True], ids=["lost-request", "lost-reply"])
def test_resume_loss_retries_without_committing_twice(harness: Harness, drop_reply: bool) -> None:
    h = harness
    h.drop_kind = "Resume"
    h.drop_count = 1
    h.drop_reply = drop_reply

    async def main() -> None:
        await h.coord.start()
        result = await h.submit()
        assert result.status == "committed", result.diagnostics
        assert h.messages.count((h.target, "Resume")) == 2
        assert all(not host.paused and host.identity == result.new_identity for host in h.hosts.values())
        assert all(h.unit_of(p).commits == 1 for p in h.hosts)

    asyncio.run(main())


def test_persistent_resume_loss_blocks_following_changes(harness: Harness) -> None:
    h = harness
    h.drop_kind = "Resume"
    h.drop_count = -1

    async def main() -> None:
        await h.coord.start()
        result = await h.submit()
        assert result.status == "failed"
        assert any(d.code == "change.recovery_required" for d in result.diagnostics)
        assert h.hosts[h.target].paused
        assert all(host.identity == result.new_identity for host in h.hosts.values())
        assert all(h.unit_of(p).version == 2 for p in h.hosts)
        assert not any(kind == "Abort" for _, kind in h.messages)
        count = len(h.messages)
        assert (await h.submit()).status == "failed"
        assert len(h.messages) == count

    asyncio.run(main())


@pytest.mark.parametrize("restart", [False, True], ids=["spawn", "restart"])
def test_candidate_process_starts_paused_until_commit_barrier(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, restart: bool,
) -> None:
    h = harness
    candidate = h.target if restart else ProcId(HostId("host-new"), "worker", 2)
    existing = list(h.hosts)
    desired_procs = existing if restart else [*existing, candidate]
    topo = Topology(
        hosts=[HostSpec(p.host, "127.0.0.1", "/unused") for p in desired_procs],
        mesh=Mesh({"tp": len(desired_procs)}),
        placement=[RankPlacement(i, p.host, 0) for i, p in enumerate(desired_procs)],
        roles=[RoleSpec("worker", [h.unit.id], per_rank=True)],
    )
    plan = msgspec.structs.replace(h.hosts[h.first].spec.plan, proc=candidate)
    participants = [pa for pa in h.action_plan.procs if pa.proc != candidate]
    h.action_plan = ActionPlan(
        procs=participants + ([ProcActions(candidate, restart=plan)] if restart else []),
        spawn=[] if restart else [plan], topology=topo,
    )
    born_paused: list[bool] = []
    before_barrier: list[bool] = []

    async def spawn(desired: DesiredProc, ps: ProcSpec) -> Result[None]:
        old = h.hosts.pop(desired.proc, None)
        if old is not None:
            old.shutdown()
        host = h.hosts[desired.proc] = ProcessHost(ps)
        host.runtime.load_process_plan(ps.plan).unwrap()
        # A candidate snapshot starts with the new behavior already installed.
        h.unit_of(desired.proc).version = 2
        monkeypatch.setattr(host.rpc, "defer", Capture)
        h.sup.procs[desired.proc] = _Proc(desired, ps, "RUNNING")
        born_paused.append(host.paused)
        observe()
        return Ok(None)

    async def stop(proc: ProcId, timeout_s: float | None = None) -> Result[None]:
        return Ok(None)

    def observe() -> None:
        if born_paused and any(h.hosts[pa.proc].identity == OLD for pa in participants):
            before_barrier.append(h.hosts[candidate].paused)

    monkeypatch.setattr(h.sup, "spawn", spawn)
    monkeypatch.setattr(h.sup, "stop", stop)
    h.observe = observe

    async def main() -> None:
        await h.coord.start()
        result = await h.submit()
        assert born_paused == [True], result.diagnostics
        assert before_barrier and all(before_barrier)
        assert result.status == "committed", result.diagnostics
        assert not h.hosts[candidate].paused
        assert h.hosts[candidate].identity == result.new_identity
        assert h.unit_of(candidate).version == 2
        assert (candidate, "Resume") in h.messages

    asyncio.run(main())
