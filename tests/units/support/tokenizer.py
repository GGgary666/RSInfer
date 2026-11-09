"""A small real HF tokenizer for tests: whole words from a fixed vocabulary, Qwen's chat markers,
and a chat template, saved as a tokenizer directory the frontend loads (``frontend.tokenizer_path``).
Decoding joins words with spaces, so a decoded text is exactly the words generated."""

from __future__ import annotations

import json
from pathlib import Path

# At import, not in a test: tests/conftest.py drops modules a test imported, and a native
# extension (tokenizers) cannot load twice in one process.
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

SPECIAL = ("<|endoftext|>", "<|im_start|>", "<|im_end|>")
TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m['role'] }} {{ m['content'] }} <|im_end|> {% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant {% endif %}")


def words(n: int) -> list[str]:
    return ["user", "assistant", "system", *(f"w{i}" for i in range(n - 3))]


def save_tokenizer(directory: Path, vocab_size: int, *, eos_in_config: bool = True) -> Path:
    """``vocab_size`` tokens: the specials, ``[UNK]``, then words. ``config.json`` names
    ``<|endoftext|>`` as eos and the tokenizer ``<|im_end|>``, as a Qwen3.8 checkpoint does."""
    vocab = {t: i for i, t in enumerate([*SPECIAL, "[UNK]", *words(vocab_size - len(SPECIAL) - 1)])}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    raw.decoder = decoders.WordPiece(prefix="##", cleanup=False)
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", eos_token="<|im_end|>",
                                  pad_token="<|endoftext|>", additional_special_tokens=list(SPECIAL))
    tok.chat_template = TEMPLATE
    directory.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(directory))
    if eos_in_config:
        config = directory / "config.json"
        data = json.loads(config.read_text()) if config.is_file() else {}
        config.write_text(json.dumps({**data, "eos_token_id": vocab["<|endoftext|>"]}))
    return directory
