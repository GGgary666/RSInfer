"""An Abort tombstone belongs to its own change, not whichever change is active."""
import pytest

from core.change.protocol import Abort, Commit, Execute, Pause, Resume
from core.common.ids import Identity
from core.common.result import Failed
from tests.adversarial.test_transaction_boundaries import Peer, peer as peer


@pytest.mark.parametrize("active", ["idle", "prepared", "paused"])
def test_unknown_abort_does_not_cancel_or_resume_another_change(peer: Peer, active: str) -> None:
    if active != "idle":
        peer.send(peer.prepare(2)).unwrap()
        if active == "paused":
            peer.send(Pause(2, None)).unwrap()
    before = (peer.host.paused, peer.state.prepares, peer.state.aborts, peer.state.resources)
    first = peer.send(Abort(1)).unwrap()
    assert peer.send(Abort(1)).unwrap() == first
    assert (peer.host.paused, peer.state.prepares, peer.state.aborts, peer.state.resources) == before
    if active == "idle":
        peer.send(peer.prepare(2)).unwrap()
    peer.send(Pause(2, None)).unwrap()
    peer.send(Execute(2)).unwrap()
    peer.send(Commit(2)).unwrap()
    peer.send(Resume(2, Identity("v1"))).unwrap()
    assert (peer.state.value, peer.state.commits, peer.state.resources) == (2, 1, 0)
    assert isinstance(peer.send(peer.prepare(1)), Failed)
    assert (peer.state.prepares, peer.state.executes) == (1, 1)
