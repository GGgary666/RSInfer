# Copyright (c) 2026 sgl-project
"""Adapted from mini-sglang/tokenizer/tokenize.py (9a91cfa, MIT).

See ../LICENSE.mini-sglang. The upstream chat-template -> encode flow is retained;
our wire carries integer columns, so tokenization does not construct torch tensors.

The stop tokens are SGLang's (``ModelConfig._get_hf_eos_token_id``, srt/configs/model_config.py;
Apache-2.0, LICENSE.sglang): the ``eos_token_id`` of the HF config and of its generation config,
each an int or a list, with the tokenizer's own (2026-11-09). A Qwen3.8 checkpoint stops chat on
the tokenizer's ``<|im_end|>`` and plain completions on the generation config's ``<|endoftext|>``.
The configs are the model's, read from ``frontend.model_path`` as SGLang reads them from the model
path; empty: the tokenizer directory, for a checkpoint that holds both (2026-11-09, #37 review).
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from frontend.interfaces import Message


def hf_eos_token_ids(path: str) -> set[int]:
    """``_get_hf_eos_token_id``: the ``eos_token_id`` of ``config.json`` and ``generation_config.json``."""
    out: set[int] = set()
    for name in ("config.json", "generation_config.json"):
        file = Path(path, name)
        eos = json.loads(file.read_text()).get("eos_token_id") if file.is_file() else None
        if eos is not None:
            out |= {int(eos)} if isinstance(eos, int) else {int(e) for e in eos}
    return out


class HFTokenizer:
    def __init__(self, path: str, model_path: str = "") -> None:
        if not path:
            raise ValueError("frontend.tokenizer_path must name a local tokenizer directory")
        module = importlib.import_module("transformers")
        self.tokenizer: Any = module.AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
        if not self.tokenizer.is_fast:
            raise ValueError("the frontend requires a fast tokenizer for incremental decoding")
        eos = self.tokenizer.eos_token_id
        self._stop = sorted(hf_eos_token_ids(model_path or path) | ({int(eos)} if eos is not None else set()))

    @property
    def stop_token_ids(self) -> Sequence[int]:
        return self._stop

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
        ctx.provide("frontend.tokenizer@1", HFTokenizer(config["frontend.tokenizer_path"],
                                                        config["frontend.model_path"]))
