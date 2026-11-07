"""A failed Execute stays paused and cannot be prepared or executed again."""
import pytest

from core.change.protocol import Abort, Commit, Execute, Pause, Resume
from core.common.ids import Identity
from core.common.result import Failed
from tests.framework.adversarial.peer import Peer, peer as peer


@pytest.mark.parametrize("replay", ["prepare", "pause", "prepare-then-pause"])
def test_undone_change_waits_for_abort_before_serving_or_accepting_new_work(peer: Peer, replay: str) -> None:
    request = peer.prepare(1, "execute")
    peer.send(request).unwrap()
    peer.send(Pause(1, None)).unwrap()
    assert isinstance(peer.send(Execute(1)), Failed)
    if replay != "pause":
        assert isinstance(peer.send(request), Failed)
    if replay != "prepare":
        assert isinstance(peer.send(Pause(1, None)), Failed)
    assert peer.host.paused, "an unconfirmed rollback must not resume service early"
    assert isinstance(peer.send(Execute(1)), Failed)
    assert (peer.state.executes, peer.state.aborts, peer.state.value) == (1, 1, 1)
    peer.send(Abort(1)).unwrap()
    assert not peer.host.paused and peer.state.resources == 0
    peer.send(peer.prepare(2)).unwrap()
    peer.send(Pause(2, None)).unwrap()
    peer.send(Execute(2)).unwrap()
    peer.send(Commit(2)).unwrap()
    peer.send(Resume(2, Identity("v1"))).unwrap()
    assert (peer.state.value, peer.state.commits, peer.state.resources) == (2, 1, 0)
