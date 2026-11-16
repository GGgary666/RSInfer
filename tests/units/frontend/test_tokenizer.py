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


def test_the_stop_tokens_come_from_the_model_directory_when_the_tokenizer_is_elsewhere(tmp_path: Path) -> None:
    """``--model A --tokenizer B``: A's configs name the model's stop tokens, as SGLang reads them from
    the model path; B's own eos still counts (Codex, #37: the model's [0, 5] were lost)."""
    tokenizer = save_tokenizer(tmp_path / "tok", 64, eos_in_config=False)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"eos_token_id": 0}))
    (model / "generation_config.json").write_text(json.dumps({"eos_token_id": [0, 5]}))
    assert list(HFTokenizer(str(tokenizer), str(model)).stop_token_ids) == [0, 2, 5]
    assert list(HFTokenizer(str(tokenizer)).stop_token_ids) == [2]


def test_chat_and_completion_encode_and_decode_whole_words(tmp_path: Path) -> None:
    tok = HFTokenizer(str(save_tokenizer(tmp_path / "tok", 64, eos_in_config=False)))
    assert list(tok.stop_token_ids) == [2]
    ids = tok.chat([Message("user", "w1 w2")])
    assert tok.tokenizer.convert_ids_to_tokens(ids) == ["<|im_start|>", "user", "w1", "w2", "<|im_end|>",
                                                        "<|im_start|>", "assistant"]
    assert tok.decode(tok.encode("w3 w4 w5")) == "w3 w4 w5"


def test_a_template_file_its_arguments_and_a_bos_the_tokenizer_does_not_add(tmp_path: Path) -> None:
    """DS 1017's checkpoint: the chat template in ``chat_template.jinja`` beside the tokenizer, the
    template's own arguments from the config, and a plain prompt started with BOS (its tokenizer
    config says not to add one; the chat template writes its own)."""
    directory = save_tokenizer(tmp_path / "tok", 64)
    settings = json.loads((directory / "tokenizer_config.json").read_text())
    settings.pop("chat_template", None)
    settings["bos_token"] = "<|im_start|>"
    (directory / "tokenizer_config.json").write_text(json.dumps(settings))
    for name in ("chat_template.jinja", "chat_template.json"):  # however this transformers saved it
        (directory / name).unlink(missing_ok=True)
    (directory / "chat_template.jinja").write_text(
        "<|im_start|> {% for m in messages %}{{ m['content'] }} {% endfor %}"
        "{% if mode is defined and mode == 'short' %}w1{% else %}w2{% endif %}")
    plain = HFTokenizer(str(directory))
    bos = plain.tokenizer.bos_token_id
    assert plain.encode("w3 w4")[0] != bos  # the tokenizer itself adds none
    tok = HFTokenizer(str(directory), add_bos=True, template_kwargs={"mode": "short"})
    assert tok.encode("w3 w4") == [bos, *plain.encode("w3 w4")]
    chat = tok.chat([Message("user", "w5")])
    assert chat == [bos, *plain.encode("w5 w1")]  # the template's BOS, not a second one
    assert HFTokenizer(str(directory), add_bos=True).chat([Message("user", "w5")]) == [bos, *plain.encode("w5 w2")]
