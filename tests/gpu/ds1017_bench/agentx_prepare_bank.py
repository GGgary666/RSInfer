"""Materialize binary ``.i32`` prompt banks with shared 64-token prefix blocks from session hashes."""
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np


def iter_turns(session: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for step in session["steps"]:
        if step["kind"] == "turn":
            yield step["turn"]
        else:
            for chain in step["chains"]:
                yield from chain


def materialize_session_bank(
    suite: dict[str, Any],
    corpus: Any,
    out_prefix: Path,
    hashes_manifest: list[dict[str, Any]] | None = None,
) -> int:
    """Write ``<out_prefix>.i32`` and ``<out_prefix>.json`` with exact shared 64-token blocks."""
    offset = 0
    corpus_arr = np.asarray(corpus, dtype="<i4")
    span = np.uint64(len(corpus_arr) - 63)
    with out_prefix.with_suffix(".i32").open("wb") as f:
        for s_idx, session in enumerate(suite["sessions"]):
            salt = np.uint64(
                int.from_bytes(hashlib.blake2b(session["trace_id"].encode(), digest_size=8).digest(), "little")
            )
            if hashes_manifest is not None:
                expected = hashes_manifest[s_idx]
                assert session["trace_id"] == expected["trace_id"]
                pairs = zip(iter_turns(session), expected["requests"], strict=True)
            else:
                pairs = ((turn, None) for turn in iter_turns(session))
            for turn, source in pairs:
                if source is not None:
                    assert [turn[k] for k in ("req_id", "in", "out", "is_subagent")] == source[:4]
                    raw_hashes = source[4]
                else:
                    raw_hashes = turn["hash_ids"]
                in_len = int(turn["in"])
                need_blocks = (in_len + 63) // 64
                if len(raw_hashes) < need_blocks:
                    pad_base = int(turn["req_id"]) * 1_000_003
                    raw_hashes = list(raw_hashes) + [pad_base + i for i in range(len(raw_hashes), need_blocks)]
                keys = np.asarray(raw_hashes[:need_blocks], dtype=np.uint64) ^ salt
                keys = (keys ^ (keys >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
                keys = (keys ^ (keys >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
                starts = ((keys ^ (keys >> np.uint64(31))) % span).astype(np.int64)
                ids = corpus_arr[starts[:, None] + np.arange(64)].reshape(-1)[:in_len].copy()
                assert len(ids) == in_len
                ids[0] = 0
                ids.tofile(f)
                turn["offset"] = offset
                turn.pop("hash_ids", None)
                offset += len(ids)
    out_prefix.with_suffix(".json").write_text(json.dumps(suite, separators=(",", ":")), encoding="utf-8")
    return offset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True, type=Path)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="Output prefix (writes <out>.i32 and <out>.json)")
    ap.add_argument("--hashes", type=Path, default=None,
                    help="Optional separate .json.gz manifest of per-request block hashes")
    args = ap.parse_args()
    suite: dict[str, Any] = json.loads(args.suite.read_text(encoding="utf-8"))
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    manifest: list[dict[str, Any]] | None = None
    if args.hashes is not None:
        with gzip.open(args.hashes, "rt", encoding="utf-8") as f:
            manifest = json.load(f)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = materialize_session_bank(suite, corpus, args.out, manifest)
    print(f"prepared {len(suite['sessions'])} sessions, {total} input tokens -> {args.out.with_suffix('.i32')}")


if __name__ == "__main__":
    main()
