"""The real frontend units over real HTTP and record streams, against a scripted scheduler."""

from __future__ import annotations

import json
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from frontend.detok.unit import Detokenizer
from tests.units.frontend.support import ByteTokenizer, FakeSupervisor, Frontend, ScriptScheduler


@pytest.fixture
def serving() -> Generator[tuple[str, ScriptScheduler]]:
    scheduler = ScriptScheduler()
    front = Frontend(scheduler)
    try:
        yield front.url, scheduler
    finally:
        front.close()
        scheduler.close()


def test_completions_and_chat_use_the_real_units_and_record_streams(serving: tuple[str, ScriptScheduler]) -> None:
    url, scheduler = serving
    with httpx.Client(base_url=url, timeout=10) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/v1/models").json()["data"][0]["id"] == "toy"
        for path, body, chat in [
            ("/v1/completions", {"prompt": "hello"}, False),
            ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hello"}]}, True),
        ]:
            response = client.post(path, json={"model": "toy", "max_tokens": 64, **body})
            assert response.status_code == 200, response.text
            data = response.json()
            choice = data["choices"][0]
            assert (choice["message"]["content"] if chat else choice["text"]) == scheduler.text
            assert choice["finish_reason"] == "stop"
            assert response.headers["x-engine-identity"] == "v1"
            assert data["engine_identities"] == ["v1"]
            assert data["usage"]["completion_tokens"] == len(scheduler.text.encode()) + 1
            sent = scheduler.new.get(timeout=2)
            assert len(sent.prompt) == data["usage"]["prompt_tokens"]


def test_chat_sse_and_stop_across_record_boundaries(serving: tuple[str, ScriptScheduler]) -> None:
    url, scheduler = serving
    scheduler.text = "你好END不能泄漏"
    with httpx.Client(base_url=url, timeout=10) as client:
        with client.stream("POST", "/v1/chat/completions", json={
            "model": "toy", "messages": [{"role": "user", "content": "hi"}],
            "stream": True, "max_tokens": 128, "stop": ["END"],
        }) as response:
            assert response.status_code == 200
            frames = [line[6:] for line in response.iter_lines() if line.startswith("data: ")]
        assert frames[-1] == "[DONE]"
        chunks = [json.loads(line) for line in frames[:-1]]
        assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        text = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
        assert text == "你好"
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert scheduler.aborts.get(timeout=2).reason == "stop string matched"


def test_disconnect_before_first_token_sends_abort(serving: tuple[str, ScriptScheduler]) -> None:
    url, scheduler = serving
    scheduler.hold = True
    with httpx.Client(base_url=url, timeout=10) as client:
        with client.stream("POST", "/v1/completions", json={"model": "toy", "prompt": "wait", "stream": True}):
            sent = scheduler.new.get(timeout=2)
        assert scheduler.aborts.get(timeout=3).request == sent.request


def test_explicit_cancel_finishes_an_idle_stream(serving: tuple[str, ScriptScheduler]) -> None:
    url, scheduler = serving
    scheduler.hold = True
    with httpx.Client(base_url=url, timeout=10) as client:
        with client.stream("POST", "/v1/completions", json={"model": "toy", "prompt": "wait", "stream": True}) as stream:
            uid = int(stream.headers["x-request-id"])
            assert client.delete(f"/v1/requests/{uid}").json()["cancelled"]
            frames = [line[6:] for line in stream.iter_lines() if line.startswith("data: ")]
            assert json.loads(frames[0])["engine_finish_reason"] == "abort"
            assert frames[-1] == "[DONE]"
        assert scheduler.aborts.get(timeout=2).request == uid
        assert not client.delete(f"/v1/requests/{uid}").json()["cancelled"]


def test_length_and_unsupported_parameters_are_reported(serving: tuple[str, ScriptScheduler]) -> None:
    url, scheduler = serving
    scheduler.text = "abcdef"
    with httpx.Client(base_url=url, timeout=10) as client:
        data = client.post("/v1/completions", json={"model": "toy", "prompt": "x", "max_tokens": 3}).json()
        assert data["choices"][0] == {"index": 0, "finish_reason": "length", "text": "abc"}
        assert data["usage"]["completion_tokens"] == 3
        assert client.post("/v1/completions", json={"model": "toy", "prompt": "x", "n": 2}).status_code == 400
        assert client.post("/v1/completions", json={"model": "missing", "prompt": "x"}).status_code == 404


def test_a_refusal_by_the_scheduler_is_an_error_in_http_and_sse() -> None:
    scheduler = ScriptScheduler(refuse="prompt plus max_tokens exceeds state capacity")
    front = Frontend(scheduler)
    try:
        with httpx.Client(base_url=front.url, timeout=10) as client:
            body = {"model": "toy", "prompt": "x", "max_tokens": 2}
            response = client.post("/v1/completions", json=body)
            assert response.status_code == 400
            assert "exceeds state capacity" in response.json()["error"]["message"]
            with client.stream("POST", "/v1/completions", json={**body, "stream": True}) as stream:
                frames = [line[6:] for line in stream.iter_lines() if line.startswith("data: ")]
            assert json.loads(frames[0])["error"]["code"] == 400 and frames[-1] == "[DONE]"
    finally:
        front.close()
        scheduler.close()


def test_a_silent_scheduler_times_out_the_http_request_and_sends_cancel() -> None:
    scheduler = ScriptScheduler(hold=True)
    front = Frontend(scheduler, response_timeout_s=0.1)
    try:
        with httpx.Client(base_url=front.url, timeout=10) as client:
            response = client.post("/v1/completions", json={"model": "toy", "prompt": "hello"})
        assert response.status_code == 503
        assert "timed out" in response.json()["error"]["message"]
        assert scheduler.aborts.get(timeout=2).request == scheduler.new.get(timeout=2).request
    finally:
        front.close()
        scheduler.close()


def test_without_a_scheduler_a_request_fails_fast_and_the_frontend_stays_up() -> None:
    """Not a permanent failure: the frontend answers 503 for this request and keeps serving."""
    scheduler = ScriptScheduler()
    front = Frontend(scheduler, io_timeout_s=0.3)
    scheduler.close()
    try:
        with httpx.Client(base_url=front.url, timeout=10) as client:
            response = client.post("/v1/completions", json={"model": "toy", "prompt": "x"})
            assert response.status_code == 503
            assert "no scheduler" in response.json()["error"]["message"]
            assert client.get("/health").json() == {"status": "ok"}
    finally:
        front.close()


def test_a_replaced_scheduler_ends_what_it_lost_and_the_next_one_serves() -> None:
    """The scheduler's process is replaced: the new one has a new reader address, which the
    supervisor names. The request in flight at the old one is lost, and ends with an error;
    the next request goes to the new scheduler."""
    supervisor = FakeSupervisor()
    old = ScriptScheduler(hold=True)
    supervisor.target = old.address
    front = Frontend(supervisor=supervisor)
    new: ScriptScheduler | None = None
    try:
        with httpx.Client(base_url=front.url, timeout=10) as client, ThreadPoolExecutor(max_workers=1) as pool:
            lost = pool.submit(client.post, "/v1/completions", json={"model": "toy", "prompt": "wait"})
            old.new.get(timeout=2)
            new = ScriptScheduler()
            supervisor.target = new.address
            old.close()
            response = client.post("/v1/completions", json={"model": "toy", "prompt": "x", "max_tokens": 64})
            assert response.status_code == 200, response.text
            assert response.json()["choices"][0]["text"] == new.text
            failed = lost.result(timeout=5)
            assert failed.status_code == 503
            assert "scheduler was replaced" in failed.json()["error"]["message"]
    finally:
        front.close()
        if new is not None:
            new.close()
        supervisor.close()


def test_concurrent_requests_keep_their_own_limits_and_return_identity_history(
    serving: tuple[str, ScriptScheduler],
) -> None:
    url, scheduler = serving
    scheduler.text, scheduler.identity = "abcdefghij", "v2"
    with httpx.Client(base_url=url, timeout=10) as client, ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(client.post, "/v1/completions", json={
            "model": "toy", "prompt": "x", "max_tokens": length}) for length in (3, 7)]
        responses = [f.result(timeout=5) for f in futures]
        assert [r.json()["choices"][0]["text"] for r in responses] == ["abc", "abcdefg"]
        assert all(r.json()["engine_identities"] == ["v1", "v2"] for r in responses)
        assert len({r.headers["x-request-id"] for r in responses}) == 2


def test_incremental_decoder_preserves_unicode_and_partial_stop_at_finish() -> None:
    detok = Detokenizer(ByteTokenizer())
    decoder = detok.open(["END"])
    parts = [decoder.push([token]).text for token in "中文EN".encode()]
    parts.append(decoder.push([], final=True).text)
    assert "".join(parts) == "中文EN"
    decoder = detok.open(["END"])
    assert decoder.push(list(b"aE")).text == "a"
    assert decoder.push(list(b"NDignored")).stopped
    assert decoder.push(list(b"late")).text == ""
