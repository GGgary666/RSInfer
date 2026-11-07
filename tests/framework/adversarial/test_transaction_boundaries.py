"""Exercise actual participant handlers with late messages and rejected prepares.

Only Deferred delivery is replaced: these tests drive one safe point explicitly.
There are no platform shims, production patches, xfails, or expected-bug assertions.
"""
from __future__ import annotations

import asyncio
from types import MethodType

import msgspec
import pytest
import zmq.asyncio

from core.change.protocol import Abort, Commit, Execute, Pause, Prepare, Resume
from core.change.types import ProcActions
from core.common.ids import Identity
from core.common.result import Failed, Result
from core.ipc.rpc import RpcClient, SyncRpcServer
from tests.framework.adversarial.peer import Peer, peer as peer
from tests.framework.support import PROC, spec, write_tree


@pytest.mark.parametrize("first_prepare_failed", [False, True], ids=["abort-overtakes-prepare", "prepare-raised"])
def test_acknowledged_abort_rejects_late_prepare(peer: Peer, first_prepare_failed: bool) -> None:
    request = peer.prepare(1, "prepare" if first_prepare_failed else "")
    if first_prepare_failed:
        assert isinstance(peer.send(request), Failed)
    peer.send(Abort(1)).unwrap()
    before = (peer.state.prepares, peer.state.aborts, peer.state.resources)

    late = peer.send(request)

    assert (peer.state.prepares, peer.state.aborts, peer.state.resources) == before, (
        "a Prepare arriving after an acknowledged Abort must not acquire resources or rerun hooks", late)
    assert isinstance(late, Failed)
    peer.send(peer.prepare(2)).unwrap()
    peer.send(Abort(2)).unwrap()
    assert peer.state.resources == 0


def test_late_pause_cannot_rearm_an_execute_that_was_already_undone(peer: Peer) -> None:
    peer.send(peer.prepare(1, "execute")).unwrap()
    peer.send(Pause(1, None)).unwrap()
    assert isinstance(peer.send(Execute(1)), Failed)
    assert (peer.state.executes, peer.state.aborts, peer.state.resources, peer.state.value) == (1, 1, 0, 1)

    peer.send(Pause(1, None))  # duplicate Pause can arrive after the failed Execute reply
    repeated = peer.send(Execute(1))
    peer.send(Abort(1)).unwrap()
    assert not peer.host.paused and peer.host.identity == Identity("v1")

    assert (peer.state.executes, peer.state.aborts, peer.state.resources, peer.state.value) == (1, 1, 0, 1), (
        "late messages must not execute an action after its preparation was cleaned up", repeated)
    assert isinstance(repeated, Failed)


@pytest.mark.parametrize("finish", ["abort", "commit"])
def test_completed_change_replays_do_not_repeat_actions(peer: Peer, finish: str) -> None:
    """Positive control: the ordinary finished-change deduplication already works."""
    request = peer.prepare(1)
    peer.send(request).unwrap()
    peer.send(Pause(1, None)).unwrap()
    peer.send(Execute(1)).unwrap()
    if finish == "commit":
        peer.send(Commit(1)).unwrap()
        peer.send(Resume(1, Identity("v1"))).unwrap()
    else:
        peer.send(Abort(1)).unwrap()
    before = (peer.state.prepares, peer.state.executes, peer.state.aborts, peer.state.commits, peer.state.value)
    for msg in [request, Pause(1, None), Execute(1), Commit(1), Resume(1, Identity("v1")), Abort(1)]:
        peer.send(msg)
    assert (peer.state.prepares, peer.state.executes, peer.state.aborts, peer.state.commits, peer.state.value) == before
    assert peer.state.resources == 0


@pytest.mark.parametrize("trace", ["abort-before-prepare", "pause-after-failed-execute", "completed-control"])
def test_adversarial_sequences_cross_real_rpc(peer: Peer, monkeypatch: pytest.MonkeyPatch, trace: str) -> None:
    """Replay serialized control messages through real ZMQ sockets and safe points."""
    monkeypatch.setattr(peer.host.rpc, "defer", MethodType(SyncRpcServer.defer, peer.host.rpc))
    context = zmq.asyncio.Context()
    client = RpcClient(context, peer.host.registry, peer.host.me)

    async def serve() -> None:
        while True:
            peer.host.safe_point()
            await asyncio.sleep(0.001)

    async def send(msg: msgspec.Struct) -> Result[msgspec.Struct]:
        return await client.call(peer.host.rpc.address, msg, timeout=2.0)

    async def main() -> None:
        serving = asyncio.create_task(serve())
        try:
            if trace == "abort-before-prepare":
                (await send(Abort(1))).unwrap()
                late = await send(peer.prepare(1))
                assert peer.state.resources == 0 and peer.state.prepares == 0, late
                assert isinstance(late, Failed)
            elif trace == "pause-after-failed-execute":
                (await send(peer.prepare(1, "execute"))).unwrap()
                (await send(Pause(1, None))).unwrap()
                assert isinstance(await send(Execute(1)), Failed)
                await send(Pause(1, None))
                await send(Execute(1))
                (await send(Abort(1))).unwrap()
                assert not peer.host.paused and peer.host.identity == Identity("v1")
                assert (peer.state.executes, peer.state.aborts, peer.state.value) == (1, 1, 1)
            else:
                (await send(peer.prepare(1))).unwrap()
                (await send(Pause(1, None))).unwrap()
                (await send(Execute(1))).unwrap()
                (await send(Abort(1))).unwrap()
                assert isinstance(await send(peer.prepare(1)), Failed)
                assert (peer.state.prepares, peer.state.executes, peer.state.aborts, peer.state.resources) == (1, 1, 1, 0)
        finally:
            serving.cancel()
            await asyncio.gather(serving, return_exceptions=True)
            client.close()
            context.term()

    asyncio.run(main())


@pytest.mark.parametrize("invalid", [False, True], ids=["valid-config-control", "invalid-second-value"])
@pytest.mark.parametrize("observation", ["config", "next-change"])
def test_rejected_staging_restores_config_and_allows_a_new_candidate(
    peer: Peer, invalid: bool, observation: str,
) -> None:
    """Partial declaration failure must not leave keys or reserve the staged root."""
    good, bad = f"{peer.pkg}.good", f"{peer.pkg}.bad"
    addon = spec(f"{peer.pkg}.addon", f"{peer.pkg}.addon:Unit", config={
        good: {"type": "int", "default": 1}, bad: {"type": "int", "default": 2}})
    files = {f"{peer.pkg}/addon.py": "class Unit:\n    def __init__(self, ctx, config): pass\n"}
    write_tree(peer.root / "v2" / "engine", files)
    write_tree(peer.root / "v3" / "engine", files)
    before = peer.host.runtime.config.entries()
    request = Prepare(1, Identity("v2"), ProcActions(
        PROC, stage=[addon], config={good: 42, bad: "wrong-type" if invalid else 43}), 60.0)
    result = peer.send(request)
    assert isinstance(result, Failed) == invalid
    peer.send(Abort(1)).unwrap()
    if observation == "config":
        assert peer.host.runtime.config.entries() == before, "rejected staging leaked config declarations"
    else:
        fresh = peer.send(Prepare(2, Identity("v3"), ProcActions(PROC, stage=[addon]), 60.0))
        assert not isinstance(fresh, Failed), "a rejected prepare blocked the next candidate: " + str(fresh)
        peer.send(Abort(2)).unwrap()
