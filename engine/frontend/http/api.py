# Copyright (c) 2026 sgl-project
"""Adapted from mini-sglang/server/api_server.py (9a91cfa, MIT).

See ../LICENSE.mini-sglang. Retains FrontendManager's per-request ack/event maps,
listener, wait_for_ack and StreamingResponse cancellation flow. Changes are the
agxinfer record channel, instance-owned app/lifespan, identity metadata, actual
finish reasons and token counts, and disconnect/stop cleanup.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Any, Literal

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from core.contracts.columns import Column
from core.contracts.requests import AbortRequest, FinishReason, NewRequest, Requests, TokenEvents
from core.contracts.sampling import SamplingParams
from frontend.interfaces import Detokenizer, Message, TextStream, Tokenizer
from frontend.transport import Channel


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAICompletionRequest(BaseModel):
    model: str
    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    max_tokens: int = Field(default=16, ge=1)
    temperature: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    top_k: int = -1
    top_p: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    n: int = 1
    stream: bool = False
    stop: str | list[str] = Field(default_factory=list)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    ignore_eos: bool = False
    seed: int | None = None


@dataclass
class UserReply:
    uid: int
    incremental_output: str
    finished: bool
    reason: FinishReason | None = None
    error: str | None = None
    error_status: int = 503


@dataclass
class RequestState:
    decoder: TextStream
    prompt_tokens: int
    identity: str
    completion_tokens: int = 0
    identities: list[str] = field(default_factory=list[str])
    last_epoch: int = -1
    finished: bool = False
    abort_sent: bool = False
    incarnation: int = 0
    """The scheduler instance it was sent to."""


class FrontendManager:
    def __init__(self, model: str, tokenizer: Tokenizer, detokenizer: Detokenizer, channel: Channel,
                 identity: Callable[[], str], *, max_tokens: int = 4096, max_requests: int = 1024,
                 response_timeout_s: float = 120.0) -> None:
        self.model, self.tokenizer, self.detokenizer, self.channel = model, tokenizer, detokenizer, channel
        self.identity, self.max_tokens, self.max_requests = identity, max_tokens, max_requests
        self.response_timeout_s = response_timeout_s
        self.uid_counter = secrets.randbits(30) << 32
        self.ack_map: dict[int, list[UserReply]] = {}
        self.event_map: dict[int, asyncio.Event] = {}
        self.states: dict[int, RequestState] = {}
        self.listener: asyncio.Task[None] | None = None
        self.failure: str | None = None
        """The token stream broke for good (two schedulers writing at once); a reload fixes it."""
        self.incarnation = 0

    async def start(self) -> None:
        await self.channel.start()
        self.listener = asyncio.create_task(self.listen())

    async def shutdown(self) -> None:
        for uid in list(self.states):
            with suppress(Exception):
                await self.abort_user(uid, "frontend unloaded")
        if self.listener is not None:
            self.listener.cancel()
            with suppress(asyncio.CancelledError):
                await self.listener
        await self.channel.close()

    def _ack(self, reply: UserReply) -> None:
        if reply.uid in self.ack_map:
            self.ack_map[reply.uid].append(reply)
            self.event_map[reply.uid].set()

    def _end_replaced(self) -> None:
        """A new scheduler instance knows nothing of the requests sent to an earlier one."""
        if self.channel.incarnation == self.incarnation:
            return
        self.incarnation = self.channel.incarnation
        for uid, state in self.states.items():
            if not state.finished and state.incarnation < self.incarnation:
                state.finished = True
                self._ack(UserReply(uid, "", True, error="the scheduler was replaced; this request was lost"))

    async def listen(self) -> None:
        try:
            while True:
                epoch, events = await self.channel.receive()
                self._end_replaced()
                await self.deliver(epoch, events)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.failure = str(exc)
            for uid, state in self.states.items():
                if not state.finished:
                    state.finished = True
                    self._ack(UserReply(uid, "", True, error=self.failure))

    async def deliver(self, epoch: int, events: TokenEvents) -> None:
        ids = [int(x) for x in events.requests.values()]
        offsets = [int(x) for x in events.offsets.values()]
        tokens = [int(x) for x in events.tokens.values()]
        if len(offsets) != len(ids) + 1 or offsets[0] != 0 or offsets[-1] != len(tokens) \
                or any(a > b for a, b in zip(offsets, offsets[1:])):
            raise ValueError("malformed TokenEvents offsets")
        pieces = {uid: tokens[offsets[i]:offsets[i + 1]] for i, uid in enumerate(ids)}
        for uid in events.finished:
            pieces.setdefault(uid, [])
        for uid in events.errors:
            pieces.setdefault(uid, [])
        for uid, new_tokens in pieces.items():
            state = self.states.get(uid)
            if state is None or state.finished:
                continue
            if uid in events.errors:
                state.finished = True
                self._ack(UserReply(uid, "", True, error=events.errors[uid], error_status=400))
                continue
            state.last_epoch = epoch
            if events.identity not in state.identities:
                state.identities.append(events.identity)
            state.completion_tokens += len(new_tokens)
            reason = events.finished.get(uid)
            piece = state.decoder.push(new_tokens, final=reason is not None)
            if piece.stopped:
                reason = "stop"
                await self._send_abort(uid, state, "stop string matched")
            state.finished = reason is not None
            self._ack(UserReply(uid, piece.text, state.finished, reason))

    async def submit(self, request: OpenAICompletionRequest, *, chat: bool) -> int:
        if self.failure is not None:
            raise HTTPException(503, self.failure)
        if request.model != self.model:
            raise HTTPException(404, "unknown model")
        if request.n != 1 or request.presence_penalty or request.frequency_penalty:
            raise HTTPException(400, "this frontend supports n=1 and zero repetition penalties")
        if request.max_tokens > self.max_tokens or request.top_k < -1:
            raise HTTPException(400, "generation limit or top_k is out of range")
        stops = [request.stop] if isinstance(request.stop, str) else request.stop
        if any(not s for s in stops):
            raise HTTPException(400, "stop strings must not be empty")
        if chat:
            if not request.messages:
                raise HTTPException(400, "messages must not be empty")
            prompt = await asyncio.to_thread(self.tokenizer.chat,
                                            [Message(m.role, m.content) for m in request.messages])
        else:
            if request.prompt is None:
                raise HTTPException(400, "prompt is required")
            prompt = await asyncio.to_thread(self.tokenizer.encode, request.prompt)
        if not prompt:
            raise HTTPException(400, "tokenized prompt must not be empty")
        if len(self.states) >= self.max_requests:
            raise HTTPException(503, "frontend request capacity reached")
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid], self.event_map[uid] = [], asyncio.Event()
        identity = self.identity()
        self.states[uid] = RequestState(self.detokenizer.open(stops), len(prompt), identity, identities=[identity])
        new = NewRequest(uid, Column.of("i64", prompt),
                         SamplingParams(request.temperature, request.top_p, request.top_k, request.seed),
                         request.max_tokens, [] if request.ignore_eos else list(self.tokenizer.stop_token_ids))
        sent = await self.channel.send(Requests(new=[new]))
        if sent is None:
            self.forget(uid)
            raise HTTPException(503, "no scheduler is reachable right now")
        self.states[uid].incarnation = sent
        self._end_replaced()
        return uid

    async def wait_for_ack(self, uid: int) -> AsyncGenerator[UserReply]:
        event = self.event_map[uid]
        while True:
            try:
                await asyncio.wait_for(event.wait(), timeout=self.response_timeout_s)
            except TimeoutError:
                yield UserReply(uid, "", True, error="scheduler response timed out")
                return
            event.clear()
            pending, self.ack_map[uid] = self.ack_map[uid], []
            for ack in pending:
                yield ack
                if ack.finished:
                    return

    async def _send_abort(self, uid: int, state: RequestState, reason: str) -> None:
        """Best effort: a scheduler that cannot be reached holds nothing to abort for long, and a
        replaced one never had the request."""
        if not state.abort_sent:
            state.abort_sent = True
            await self.channel.send(Requests(abort=[AbortRequest(uid, reason)]))
            self._end_replaced()

    async def abort_user(self, uid: int, reason: str = "client disconnected") -> bool:
        state = self.states.get(uid)
        if state is None or state.finished:
            return False
        await self._send_abort(uid, state, reason)
        state.finished = True
        self._ack(UserReply(uid, "", True, "abort"))
        return True

    def forget(self, uid: int) -> None:
        self.states.pop(uid, None)
        self.ack_map.pop(uid, None)
        self.event_map.pop(uid, None)

    def usage(self, uid: int) -> dict[str, int]:
        state = self.states[uid]
        return dict(prompt_tokens=state.prompt_tokens, completion_tokens=state.completion_tokens,
                    total_tokens=state.prompt_tokens + state.completion_tokens)

    async def stream_completions(self, uid: int, chat: bool) -> AsyncGenerator[bytes]:
        first_chunk = True
        try:
            async for ack in self.wait_for_ack(uid):
                if ack.error:
                    kind = "invalid_request_error" if ack.error_status < 500 else "server_error"
                    yield self.sse({"error": {"message": ack.error, "type": kind, "code": ack.error_status}})
                    break
                delta: dict[str, str] = {}
                if first_chunk and chat:
                    delta["role"] = "assistant"
                first_chunk = False
                if ack.incremental_output:
                    delta["content"] = ack.incremental_output
                reason = ack.reason if ack.reason in ("stop", "length") else ("stop" if ack.finished else None)
                choice: dict[str, Any] = {"index": 0, "finish_reason": reason}
                choice.update({"delta": delta} if chat else {"text": ack.incremental_output})
                chunk: dict[str, Any] = {
                    "id": f"{'chatcmpl' if chat else 'cmpl'}-{uid}",
                    "object": "chat.completion.chunk" if chat else "text_completion",
                    "created": int(time.time()), "model": self.model, "choices": [choice],
                }
                if ack.finished:
                    chunk.update(usage=self.usage(uid), engine_finish_reason=ack.reason,
                                 engine_identities=self.states[uid].identities)
                yield self.sse(chunk)
            yield b"data: [DONE]\n\n"
        finally:
            try:
                with anyio.CancelScope(shield=True):
                    await self.abort_user(uid)
            finally:
                self.forget(uid)

    @staticmethod
    def sse(value: dict[str, Any]) -> bytes:
        return f"data: {json.dumps(value, ensure_ascii=False)}\n\n".encode()


def create_app(manager: FrontendManager) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        try:
            await manager.start()
            yield
        finally:
            await manager.shutdown()

    app = FastAPI(title="agxinfer", lifespan=lifespan)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        kind = "invalid_request_error" if exc.status_code < 500 else "server_error"
        return JSONResponse({"error": {"message": str(exc.detail), "type": kind}},
                            status_code=exc.status_code)

    @app.get("/health")
    async def health() -> dict[str, str]:
        if manager.failure:
            raise HTTPException(503, manager.failure)
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": manager.model, "object": "model",
                "created": int(time.time()), "owned_by": "agxinfer"}]}

    @app.delete("/v1/requests/{uid}")
    async def cancel(uid: int) -> dict[str, Any]:
        return {"id": uid, "cancelled": await manager.abort_user(uid, "explicit cancellation")}

    async def completion(req: OpenAICompletionRequest, request: Request, chat: bool) -> Any:
        uid = await manager.submit(req, chat=chat)
        headers = {"x-engine-identity": manager.states[uid].identity, "x-request-id": str(uid)}
        if req.stream:
            return StreamingResponse(manager.stream_completions(uid, chat), media_type="text/event-stream",
                                     headers=headers)

        async def collect() -> tuple[str, UserReply]:
            chunks: list[str] = []
            async for ack in manager.wait_for_ack(uid):
                if ack.error:
                    raise HTTPException(ack.error_status, ack.error)
                chunks.append(ack.incremental_output)
                if ack.finished:
                    return "".join(chunks), ack
            raise HTTPException(503, "scheduler ended without a final result")

        task = asyncio.create_task(collect())
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=0.05)
                if await request.is_disconnected():
                    await manager.abort_user(uid)
                    raise HTTPException(499, "client disconnected")
            content, last = task.result()
            reason = last.reason if last.reason in ("stop", "length") else "stop"
            choice: dict[str, Any] = {"index": 0, "finish_reason": reason}
            choice.update({"message": {"role": "assistant", "content": content}} if chat else {"text": content})
            return JSONResponse({"id": f"{'chatcmpl' if chat else 'cmpl'}-{uid}",
                                 "object": "chat.completion" if chat else "text_completion",
                                 "created": int(time.time()), "model": manager.model, "choices": [choice],
                                 "usage": manager.usage(uid), "engine_finish_reason": last.reason,
                                 "engine_identities": manager.states[uid].identities}, headers=headers)
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError, HTTPException):
                await task
            try:
                with anyio.CancelScope(shield=True):
                    await manager.abort_user(uid)
            finally:
                manager.forget(uid)

    @app.post("/v1/chat/completions")
    async def chat(req: OpenAICompletionRequest, request: Request) -> Any:
        return await completion(req, request, True)

    @app.post("/v1/completions")
    async def text(req: OpenAICompletionRequest, request: Request) -> Any:
        return await completion(req, request, False)

    return app
