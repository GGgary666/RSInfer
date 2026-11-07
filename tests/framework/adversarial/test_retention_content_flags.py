"""Content availability includes artifacts, grace periods, and file deletions."""
import os
import time
from pathlib import Path

import pytest

from core.change.blobs import BlobStore
from core.change.eventlog import EventLog
from core.change.state import IdentityManifest, identity_of
from core.change.types import ChangeResult, ChangeSet


@pytest.mark.parametrize("kind", ["file", "artifact"])
@pytest.mark.parametrize("mode", ["prune", "grace", "keep"])
def test_purge_flags_track_referenced_files_and_artifacts(tmp_path: Path, kind: str, mode: str) -> None:
    blobs = BlobStore(tmp_path / "blobs")
    log = EventLog(tmp_path / "events.sqlite", blobs)
    try:
        base = IdentityManifest(files={"engine/old.py": blobs.put(b"old")})
        identity = identity_of(base)
        log.record_identity(identity, base, None)
        payload = blobs.put(b"candidate")
        orphan = blobs.put(b"unreferenced")
        for h in [payload, orphan]:
            old = time.time() - 7200
            os.utime(blobs.root / h[:2] / h[2:], (old, old))
        if mode == "grace":
            blobs.put(b"candidate")
        cs = ChangeSet(base=identity, files={"engine/old.py": None, **({"engine/new.py": payload} if kind == "file" else {})},
                       artifacts={"weights": payload} if kind == "artifact" else {})
        cid = log.begin(cs)
        log.finish(ChangeResult("aborted", cid, identity, None, None, {}))
        log.prune([identity], 0, extra=[payload] if mode == "keep" else [], grace_s=3600)
        assert not blobs.has(orphan)
        assert blobs.has(payload) == (mode != "prune")
        assert log.changes()[0].content_purged == (mode == "prune")
        assert log.changeset(cid) == cs
    finally:
        log.db.close()
