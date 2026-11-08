# Copyright (c) 2026 sgl-project
"""Adapted from mini-sglang/tokenizer/tokenize.py (9a91cfa, MIT).

See ../LICENSE.mini-sglang. The upstream chat-template -> encode flow is retained;
our wire carries integer columns, so tokenization does not construct torch tensors.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

from frontend.interfaces import Message


class HFTokenizer:
    def __init__(self, path: str) -> None:
        if not path:
            raise ValueError("frontend.tokenizer_path must name a local tokenizer directory")
        module = importlib.import_module("transformers")
        self.tokenizer: Any = module.AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
        if not self.tokenizer.is_fast:
            raise ValueError("the frontend requires a fast tokenizer for incremental decoding")

    @property
    def stop_token_ids(self) -> Sequence[int]:
        eos = self.tokenizer.eos_token_id
        return [] if eos is None else [int(eos)]

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=True))

    def chat(self, messages: Sequence[Message]) -> list[int]:
        prompt = self.tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in messages],
            tokenize=False, add_generation_prompt=True)
        return self.encode(str(prompt))

    def decode(self, tokens: Sequence[int]) -> str:
        return str(self.tokenizer.decode(list(tokens), skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False))


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        ctx.provide("frontend.tokenizer@1", HFTokenizer(config["frontend.tokenizer_path"]))
