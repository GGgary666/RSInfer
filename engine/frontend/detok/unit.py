# Copyright (c) 2026 sgl-project
"""Adapted from mini-sglang/tokenizer/detokenize.py (9a91cfa, MIT).

See ../LICENSE.mini-sglang. Retains the upstream read/surrounding offsets and UTF-8
replacement-character check. A stop-prefix holdback is added for our frontend's
stop-string cancellation; upstream's minimal API accepts but does not enforce stop.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from frontend.interfaces import TextPiece, TextStream, Tokenizer


@dataclass
class DecodeStatus:
    decoded_ids: list[int] = field(default_factory=list[int])
    read_offset: int = 0
    surr_offset: int = 0

    def decode(self, tokenizer: Tokenizer, tokens: Sequence[int], final: bool) -> str:
        self.decoded_ids.extend(tokens)
        read_str = tokenizer.decode(self.decoded_ids[self.surr_offset:])
        surr_str = tokenizer.decode(self.decoded_ids[self.surr_offset:self.read_offset])
        new_text = read_str[len(surr_str):]
        if final or (new_text and not new_text.endswith("\ufffd")):
            self.surr_offset = self.read_offset
            self.read_offset = len(self.decoded_ids)
            return new_text
        return ""


class Stream:
    def __init__(self, tokenizer: Tokenizer, stops: Sequence[str]) -> None:
        if any(not stop for stop in stops):
            raise ValueError("stop strings must not be empty")
        self.tokenizer, self.stops = tokenizer, tuple(stops)
        self.status = DecodeStatus()
        self.pending = ""
        self.stopped = False

    def push(self, tokens: Sequence[int], *, final: bool = False) -> TextPiece:
        if self.stopped:
            return TextPiece("", True)
        if final and tokens and tokens[-1] in self.tokenizer.stop_token_ids:
            tokens = tokens[:-1]
        self.pending += self.status.decode(self.tokenizer, tokens, final)
        hits = [at for stop in self.stops if (at := self.pending.find(stop)) >= 0]
        if hits:
            text = self.pending[:min(hits)]
            self.pending, self.stopped = "", True
            return TextPiece(text, True)
        held = 0
        if not final:
            for stop in self.stops:
                for n in range(1, min(len(stop), len(self.pending) + 1)):
                    if self.pending.endswith(stop[:n]):
                        held = max(held, n)
        end = len(self.pending) - held
        text, self.pending = self.pending[:end], self.pending[end:]
        if final:
            self.stopped = True
        return TextPiece(text)


class Detokenizer:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    def open(self, stops: Sequence[str] = ()) -> TextStream:
        return Stream(self.tokenizer, stops)


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        ctx.provide("frontend.detok@1", Detokenizer(cast(Tokenizer, ctx.use("frontend.tokenizer@1"))))
