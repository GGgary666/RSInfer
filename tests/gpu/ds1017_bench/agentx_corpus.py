"""Prompts for the DS 1017 AgentX replay, the same for every engine
under test: real text of given token lengths, cut out of the Python standard library's sources (code
and its English documentation, what every Python environment has). The model routes real text to its
experts as it would in service; random token ids would spread them evenly.

    python corpus.py --tokenizer <model dir> --out <dir> --spec 1024:600 4096:300 32000:160

writes ``<dir>/<length>.json`` for each ``length:count``: ``{"texts": [...], "ids": [[...], ...]}``.
``ids`` are the tokens a plain prompt becomes in both engines, the BOS token first (DeepInfer's
tokenizer and this engine's frontend put it there): an engine without a text API is given the ids, one
with an HTTP API the text, and both run the same tokens. A prompt has at most ``length`` tokens, a few
less where a cut does not fall between two tokens. No two prompts start at the same place.
"""

from __future__ import annotations

import argparse
import json
import sysconfig
from pathlib import Path
from typing import Any

BOS = 0


def corpus(limit: int) -> str:
    root = Path(sysconfig.get_paths()["stdlib"])
    parts: list[str] = []
    size = 0
    for path in sorted(root.rglob("*.py")):
        if "site-packages" in path.parts or "test" in path.parts or "tests" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if len(text) < 4000:
            continue
        parts.append(f"# ---- {path.relative_to(root)} ----\n{text}\n")
        size += len(text)
        if size >= limit:
            break
    return "".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", required=True, help="the model directory (its tokenizer)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--spec", nargs="+", required=True, help="length:count ...")
    p.add_argument("--corpus-bytes", type=int, default=16_000_000)
    args = p.parse_args()
    from transformers import AutoTokenizer

    tok: Any = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    text = corpus(args.corpus_bytes)
    # In pieces: one call on megabytes of text is slow in some tokenizers' pre-tokenization.
    ids: list[int] = []
    for start in range(0, len(text), 200_000):
        ids += tok.encode(text[start : start + 200_000], add_special_tokens=False)
    print(f"corpus: {len(text)} characters, {len(ids)} tokens", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    for item in args.spec:
        length, count = (int(x) for x in item.split(":"))
        if length + 64 >= len(ids):
            raise SystemExit(f"the corpus has {len(ids)} tokens, too few for prompts of {length}")
        stride = max(1, (len(ids) - length - 64) // count)
        texts: list[str] = []
        tokens: list[list[int]] = []
        for i in range(count):
            start = (i * stride + 7919 * (length % 1009)) % (len(ids) - length - 64)
            take = length - 1  # and the BOS token
            while True:
                piece = tok.decode(
                    ids[start : start + take], skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                again = tok.encode(piece, add_special_tokens=False)
                if len(again) + 1 <= length:
                    break
                take -= len(again) + 1 - length
            texts.append(piece)
            tokens.append([BOS, *again])
        lengths = sorted(len(t) for t in tokens)
        (args.out / f"{length}.json").write_text(json.dumps({"texts": texts, "ids": tokens}))
        print(f"{length}: {count} prompts, {lengths[0]}..{lengths[-1]} tokens", flush=True)


if __name__ == "__main__":
    main()
