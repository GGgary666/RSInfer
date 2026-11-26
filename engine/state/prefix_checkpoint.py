"""On-disk radix index for state pages held by an external MemCache pool."""

from __future__ import annotations

from pathlib import Path

import msgspec


class PrefixNode(msgspec.Struct, frozen=True):
    parent: int
    key: bytes
    pages: list[int]
    slot: int | None
    salt: bytes
    access: int


class PrefixCheckpoint(msgspec.Struct, frozen=True):
    nonce: str
    world: int
    page: int
    slot_kind: str
    domains: list[list[PrefixNode]]
    version: int = 1

    @classmethod
    def load(cls, path: Path) -> PrefixCheckpoint:
        index = msgspec.json.decode(path.read_bytes(), type=cls)
        if index.version != 1 or not index.domains or index.world < 1 or index.world % len(index.domains):
            raise ValueError(f"incompatible prefix checkpoint {path}")
        return index
