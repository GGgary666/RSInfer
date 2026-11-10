"""``frontend.http``'s ``drain`` action, the engine's gate for changes (doc/specs/empty-apply.md):
while a change runs, new requests are turned away with 503 and ``Retry-After``; the change hears
how many are open."""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

import httpx
import pytest

from tests.units.frontend.support import Frontend, ScriptScheduler, wait_empty


@pytest.fixture
def front() -> Generator[tuple[Frontend, ScriptScheduler]]:
    scheduler = ScriptScheduler()
    f = Frontend(scheduler)
    try:
        yield f, scheduler
    finally:
        f.close()
        scheduler.close()


def gate(f: Frontend) -> Any:
    handler = f.runtime.action_handler("frontend.http", "drain")
    assert handler is not None
    return handler


def complete(client: httpx.Client) -> httpx.Response:
    return client.post("/v1/completions", json={"model": "toy", "prompt": "hi", "max_tokens": 3})


def test_a_closed_gate_turns_new_requests_away_until_the_change_ends(front: tuple[Frontend, ScriptScheduler]) -> None:
    f, _ = front
    g = gate(f)
    with httpx.Client(base_url=f.url, timeout=10) as client:
        assert complete(client).status_code == 200
        g.start_drain({"mode": "wait"})
        assert g.drained()
        r = complete(client)
        assert r.status_code == 503 and r.headers["retry-after"] == "1"
        assert "being changed" in r.json()["error"]["message"]
        g.commit()
        assert complete(client).status_code == 200


def test_the_change_sees_the_open_requests_refuse_fails_and_wait_waits(front: tuple[Frontend, ScriptScheduler]) -> None:
    f, scheduler = front
    scheduler.hold = True  # requests stay open: the scheduler sends nothing
    g = gate(f)
    with httpx.Client(base_url=f.url, timeout=10) as client:
        with client.stream("POST", "/v1/completions", json={"model": "toy", "prompt": "wait", "stream": True}) as s:
            uid = int(s.headers["x-request-id"])
            scheduler.new.get(timeout=2)
            with pytest.raises(RuntimeError, match=r"the frontend has 1 open request.*--wait-empty.*--force"):
                g.start_drain({"mode": "refuse"})
            g.abort()
            g.start_drain({"mode": "wait"})
            assert not g.drained()
            with pytest.raises(TimeoutError, match=r"still has 1 open request.*nothing was ended"):
                g.drain_deadline()
            assert client.delete(f"/v1/requests/{uid}").json()["cancelled"]
            frames = [line[6:] for line in s.iter_lines() if line.startswith("data: ")]
            assert json.loads(frames[0])["engine_finish_reason"] == "abort"
        wait_empty(f.server.manager)
        assert g.drained()
        g.abort()
        scheduler.hold = False
        assert complete(client).status_code == 200


def test_a_frontend_built_while_its_process_is_paused_stays_closed_until_it_resumes() -> None:
    """A change's Execute that reloads ``frontend.http`` builds a new instance: the drain closed the
    old one, and the new one must not take requests before the change committed or was undone
    (Codex, #42). Its HTTP thread serves even though the process is paused."""
    scheduler = ScriptScheduler()
    f = Frontend(scheduler, paused=True)
    try:
        with httpx.Client(base_url=f.url, timeout=10) as client:
            r = complete(client)
            assert r.status_code == 503 and r.headers["retry-after"] == "1"
            f.control.paused = False  # the change ended: the process resumes
            assert complete(client).status_code == 200
            f.control.paused = True  # a later pause, of a change that does not reload it, leaves it open
            assert complete(client).status_code == 200
    finally:
        f.close()
        scheduler.close()
