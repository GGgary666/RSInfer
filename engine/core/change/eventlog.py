"""The event log (change.md §5.1): sqlite holds indexes and summaries only.

File contents and change-sets live in the blob store, referenced by hash;
diffs are computed on demand. Records are kept forever; when retention removes
the blobs a record points to, the record says so.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import msgspec

from core.change.blobs import BlobStore
from core.change.state import IdentityManifest, decode_manifest, encode_manifest
from core.change.types import ChangeResult, ChangeSet
from core.common.ids import Identity

_SCHEMA = """
CREATE TABLE IF NOT EXISTS changes (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    status TEXT NOT NULL,
    base TEXT NOT NULL,
    new_identity TEXT,
    state_version INTEGER NOT NULL DEFAULT 0,
    changeset_hash TEXT NOT NULL,
    config_json TEXT NOT NULL,
    plan_json TEXT,
    timings_json TEXT,
    diagnostics_json TEXT,
    aborted_requests INTEGER NOT NULL DEFAULT 0,
    content_purged INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    finished_at REAL
);
CREATE TABLE IF NOT EXISTS identities (
    identity TEXT PRIMARY KEY,
    manifest_hash TEXT NOT NULL,
    parent TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    change_id INTEGER,
    kind TEXT NOT NULL,
    detail_json TEXT NOT NULL,
    at REAL NOT NULL
);
"""

_json = msgspec.json.Encoder()


class ChangeRecord(msgspec.Struct, frozen=True):
    id: int
    source: str
    status: str
    base: str
    new_identity: str | None
    state_version: int
    changeset_hash: str
    config: dict[str, Any]
    plan: dict[str, Any] | None
    timings: dict[str, float] | None
    diagnostics: list[dict[str, Any]] | None
    aborted_requests: int
    content_purged: bool


class EventLog:
    def __init__(self, path: Path, blobs: BlobStore) -> None:
        self.blobs = blobs
        self.db = sqlite3.connect(path)
        self.db.executescript(_SCHEMA)

    # -- changes ----------------------------------------------------------

    def begin(self, cs: ChangeSet) -> int:
        h = self.blobs.put(msgspec.msgpack.encode(cs))
        cur = self.db.execute(
            "INSERT INTO changes (source, status, base, changeset_hash, config_json, created_at)"
            " VALUES (?, 'received', ?, ?, ?, ?)",
            (cs.source, cs.base, h, _json.encode(cs.config).decode(), time.time()),
        )
        self.db.commit()
        assert cur.lastrowid is not None
        return cur.lastrowid

    def finish(self, result: ChangeResult) -> None:
        self.db.execute(
            "UPDATE changes SET status = ?, new_identity = ?, state_version = ?, plan_json = ?,"
            " timings_json = ?, diagnostics_json = ?, aborted_requests = ?, finished_at = ? WHERE id = ?",
            (result.status, result.new_identity, result.state_version,
             _json.encode(result.plan).decode() if result.plan is not None else None,
             _json.encode(result.timings).decode(), _json.encode(result.diagnostics).decode(),
             result.aborted_requests, time.time(), result.change_id),
        )
        self.db.commit()

    def event(self, change_id: int | None, kind: str, detail: dict[str, Any]) -> None:
        self.db.execute("INSERT INTO events (change_id, kind, detail_json, at) VALUES (?, ?, ?, ?)",
                        (change_id, kind, _json.encode(detail).decode(), time.time()))
        self.db.commit()

    def events(self, kind: str | None = None) -> list[tuple[int | None, str, dict[str, Any]]]:
        rows = self.db.execute("SELECT change_id, kind, detail_json FROM events ORDER BY id").fetchall()
        return [(c, k, json.loads(d)) for c, k, d in rows if kind is None or k == kind]

    def changes(self) -> list[ChangeRecord]:
        rows = self.db.execute(
            "SELECT id, source, status, base, new_identity, state_version, changeset_hash, config_json,"
            " plan_json, timings_json, diagnostics_json, aborted_requests, content_purged FROM changes ORDER BY id"
        ).fetchall()
        return [
            ChangeRecord(r[0], r[1], r[2], r[3], r[4], r[5], r[6], json.loads(r[7]),
                         json.loads(r[8]) if r[8] else None, json.loads(r[9]) if r[9] else None,
                         json.loads(r[10]) if r[10] else None, r[11], bool(r[12]))
            for r in rows
        ]

    def changeset(self, change_id: int) -> ChangeSet | None:
        row = self.db.execute("SELECT changeset_hash FROM changes WHERE id = ?", (change_id,)).fetchone()
        data = self.blobs.get(row[0]) if row else None
        return msgspec.msgpack.decode(data, type=ChangeSet) if data is not None else None

    # -- identities ---------------------------------------------------------

    def record_identity(self, identity: Identity, manifest: IdentityManifest, parent: Identity | None) -> None:
        h = self.blobs.put(encode_manifest(manifest))
        self.db.execute("INSERT OR IGNORE INTO identities (identity, manifest_hash, parent, created_at)"
                        " VALUES (?, ?, ?, ?)", (identity, h, parent, time.time()))
        self.db.commit()

    def manifest(self, identity: Identity) -> IdentityManifest | None:
        row = self.db.execute("SELECT manifest_hash FROM identities WHERE identity = ?", (identity,)).fetchone()
        data = self.blobs.get(row[0]) if row else None
        return decode_manifest(data) if data is not None else None

    def diff(self, a: Identity, b: Identity) -> dict[str, tuple[str | None, str | None]]:
        """Paths whose content differs between two identities (old hash, new hash)."""
        ma, mb = self.manifest(a), self.manifest(b)
        if ma is None or mb is None:
            raise KeyError("content of one of the identities was purged")
        paths = set(ma.files) | set(mb.files)
        return {p: (ma.files.get(p), mb.files.get(p)) for p in sorted(paths) if ma.files.get(p) != mb.files.get(p)}

    # -- retention ----------------------------------------------------------

    def prune(self, keep: list[Identity], retention: int) -> int:
        """Delete blobs not referenced by ``keep`` or the ``retention`` most recent identities."""
        recent = [r[0] for r in self.db.execute(
            "SELECT identity FROM identities ORDER BY created_at DESC, rowid DESC LIMIT ?", (retention,))]
        kept = sorted({*keep, *recent})
        live: set[str] = set()
        for identity in kept:
            row = self.db.execute("SELECT manifest_hash FROM identities WHERE identity = ?", (identity,)).fetchone()
            if row is None:
                continue
            live.add(row[0])
            m = self.manifest(Identity(identity))
            if m is not None:
                live |= set(m.files.values()) | set(m.artifacts.values())
        for h, base, new in self.db.execute("SELECT changeset_hash, base, new_identity FROM changes"):
            if base in kept or new in kept:
                live.add(h)
        removed = 0
        for h in self.blobs.hashes() - live:
            self.blobs.delete(h)
            removed += 1
        for cid, h in self.db.execute("SELECT id, changeset_hash FROM changes").fetchall():
            if not self.blobs.has(h):
                self.db.execute("UPDATE changes SET content_purged = 1 WHERE id = ?", (cid,))
        self.db.commit()
        return removed
