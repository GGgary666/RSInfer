"""Content-addressed blob store on the control plane's local disk (change.md §5.1).

File contents, change-sets and identity file lists are stored once per content
hash. Node agents materialize snapshots by pulling blobs from here.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path

HASH = "sha256"


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class BlobStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def _path(self, h: str) -> Path:
        if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
            raise ValueError(f"not a {HASH} hex digest: {h!r}")
        return self.root / h[:2] / h[2:]

    def put(self, data: bytes) -> str:
        h = content_hash(data)
        path = self._path(h)
        if path.exists():
            os.utime(path)  # uploaded again: its grace period starts again (change.md §5.1)
        else:
            path.parent.mkdir(exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent)
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        return h

    def age(self, h: str) -> float:
        """Seconds since the blob was last stored."""
        return time.time() - self._path(h).stat().st_mtime

    def get(self, h: str) -> bytes | None:
        try:
            return self._path(h).read_bytes()
        except FileNotFoundError:
            return None

    def has(self, h: str) -> bool:
        return self._path(h).exists()

    def missing(self, hashes: Iterable[str]) -> list[str]:
        return sorted({h for h in hashes if not self.has(h)})

    def delete(self, h: str) -> None:
        self._path(h).unlink(missing_ok=True)

    def hashes(self) -> set[str]:
        return {p.parent.name + p.name for p in self.root.glob("??/*") if p.is_file()}

    def ingest_tree(self, tree: Path, prefix: str = "") -> dict[str, str]:
        """Store every file under ``tree``; returns ``{prefix + relative path: hash}``."""
        files: dict[str, str] = {}
        for path in sorted(tree.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(".pyc"):
                files[prefix + path.relative_to(tree).as_posix()] = self.put(path.read_bytes())
        return files
