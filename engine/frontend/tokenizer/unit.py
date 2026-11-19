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

DS 1017 (2026-11-15): its checkpoint keeps the chat template in ``chat_template.jinja`` beside the
tokenizer (read here where the tokenizer loaded none), the template's own arguments come from
``frontend.chat_template_kwargs`` (``{"thinking_mode": "chat"}`` answers without the reasoning
part), and a plain prompt starts with the BOS token as DeepInfer makes it (``frontend.add_bos``;
its ``hugging_face/tokenizer.py`` sets ``add_bos_token``): the chat template writes its own.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
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
    def __init__(self, path: str, model_path: str = "", add_bos: bool = False,
                 template_kwargs: Mapping[str, Any] | None = None) -> None:
        if not path:
            raise ValueError("frontend.tokenizer_path must name a local tokenizer directory")
        module = importlib.import_module("transformers")
        self.tokenizer: Any = module.AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
        if not self.tokenizer.is_fast:
            raise ValueError("the frontend requires a fast tokenizer for incremental decoding")
        self._backend: Any = self.tokenizer.backend_tokenizer
        template = Path(path, "chat_template.jinja")
        if not getattr(self.tokenizer, "chat_template", None) and template.is_file():
            self.tokenizer.chat_template = template.read_text()
        eos = self.tokenizer.eos_token_id
        self._stop = sorted(hf_eos_token_ids(model_path or path) | ({int(eos)} if eos is not None else set()))
        bos = self.tokenizer.bos_token_id
        if add_bos and bos is None:
            raise ValueError("frontend.add_bos: the tokenizer has no BOS token")
        self.bos = int(bos) if add_bos else None
        """The token a prompt starts with, when the tokenizer does not put it there itself."""
        self.template_kwargs = dict(template_kwargs or {})

    @property
    def stop_token_ids(self) -> Sequence[int]:
        return self._stop

    def encode(self, text: str) -> list[int]:
        ids = list(self._backend.encode(text, add_special_tokens=True).ids)
        if self.bos is not None and ids[:1] != [self.bos]:
            ids.insert(0, self.bos)
        return ids

    def chat(self, messages: Sequence[Message]) -> list[int]:
        prompt = self.tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in messages],
            tokenize=False, add_generation_prompt=True, **self.template_kwargs)
        return self.encode(str(prompt))

    def decode(self, tokens: Sequence[int]) -> str:
        if not tokens:
            return ""
        ids = tokens if isinstance(tokens, list) else list(tokens)
        return str(self._backend.decode(ids, skip_special_tokens=True))


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        ctx.provide("frontend.tokenizer@1", HFTokenizer(config["frontend.tokenizer_path"],
                                                        config["frontend.model_path"], config["frontend.add_bos"],
                                                        dict(config["frontend.chat_template_kwargs"])))
