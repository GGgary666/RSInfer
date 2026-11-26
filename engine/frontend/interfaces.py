"""Local interfaces between the frontend units; independent of GPU execution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import msgspec


class Message(msgspec.Struct, frozen=True):
    role: str
    content: str


class Tokenizer(Protocol):
    @property
    def stop_token_ids(self) -> Sequence[int]: ...
    def encode(self, text: str) -> list[int]: ...
    def chat(self, messages: Sequence[Message], *, reasoning_effort: str | None = None) -> list[int]: ...
    def decode(self, tokens: Sequence[int]) -> str: ...


class TextPiece(msgspec.Struct, frozen=True):
    text: str
    stopped: bool = False


class TextStream(Protocol):
    def push(self, tokens: Sequence[int], *, final: bool = False) -> TextPiece: ...


class Detokenizer(Protocol):
    def open(self, stops: Sequence[str] = ()) -> TextStream: ...
