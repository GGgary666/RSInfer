"""What every model recipe reads from a Hugging Face model directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt",
                   "special_tokens_map.json", "generation_config.json")


def read_config(model: Path) -> dict[str, Any]:
    path = model / "config.json"
    if not path.is_file():
        raise SystemExit(f"{model}: no config.json (a Hugging Face model directory is expected)")
    return json.loads(path.read_text())


def text_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """The language model's part of a multimodal config, or the config itself."""
    return cfg.get("text_config", cfg)


def cut(cfg: dict[str, Any], layers: int) -> dict[str, Any]:
    """The config with only its first ``layers`` decoder layers (``layer_types`` cut along)."""
    out = json.loads(json.dumps(cfg))
    t = text_config(out)
    if not 0 < layers <= t["num_hidden_layers"]:
        raise SystemExit(f"--layers must be in 1..{t['num_hidden_layers']}")
    t["num_hidden_layers"] = layers
    if "layer_types" in t:
        t["layer_types"] = t["layer_types"][:layers]
    return out
