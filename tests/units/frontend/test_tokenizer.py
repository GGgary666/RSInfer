"""``frontend.tokenizer`` on a real HF tokenizer directory."""

from __future__ import annotations

import json
from pathlib import Path

from frontend.interfaces import Message
from frontend.tokenizer.unit import HFTokenizer
from tests.units.support.tokenizer import save_tokenizer


def test_the_stop_tokens_are_the_tokenizers_and_the_configs(tmp_path: Path) -> None:
    """As SGLang: the tokenizer's eos (``<|im_end|>``, chat) and the configs' (``<|endoftext|>``)."""
    path = save_tokenizer(tmp_path / "tok", 64)
    (path / "generation_config.json").write_text(json.dumps({"eos_token_id": [0, 5]}))
    tok = HFTokenizer(str(path))
    assert list(tok.stop_token_ids) == [0, 2, 5]


def test_chat_and_completion_encode_and_decode_whole_words(tmp_path: Path) -> None:
    tok = HFTokenizer(str(save_tokenizer(tmp_path / "tok", 64, eos_in_config=False)))
    assert list(tok.stop_token_ids) == [2]
    ids = tok.chat([Message("user", "w1 w2")])
    assert tok.tokenizer.convert_ids_to_tokens(ids) == ["<|im_start|>", "user", "w1", "w2", "<|im_end|>",
                                                        "<|im_start|>", "assistant"]
    assert tok.decode(tok.encode("w3 w4 w5")) == "w3 w4 w5"
