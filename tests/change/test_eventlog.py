"""change.md §7 test 11: sqlite holds no file contents; blobs rebuild change-sets and diffs."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.change.blobs import BlobStore
from core.change.eventlog import EventLog
from core.change.state import IdentityManifest, apply_changeset, identity_of
from core.change.types import ChangeResult, ChangeSet
from core.common.ids import Identity

SECRET = b"def kernel():  # SECRET-CONTENT-7f3a\n"


def test_contents_live_in_blobs_and_purges_are_recorded(tmp_path: Path) -> None:
    blobs = BlobStore(tmp_path / "blobs")
    log = EventLog(tmp_path / "events.sqlite", blobs)
    base = IdentityManifest(files={"engine/k.py": blobs.put(b"old\n")})
    b0 = identity_of(base)
    log.record_identity(b0, base, None)

    cs = ChangeSet(base=b0, files={"engine/k.py": blobs.put(SECRET)}, config={"sched.chunk": 4096})
    new = apply_changeset(base, cs)
    b1 = identity_of(new)
    assert identity_of(apply_changeset(base, cs)) == b1  # deterministic
    log.record_identity(b1, new, b0)
    cid = log.begin(cs)
    log.finish(ChangeResult("committed", cid, b0, b1, None, {"prepare": 0.1}))

    db = (tmp_path / "events.sqlite").read_bytes()
    assert b"SECRET-CONTENT" not in db
    assert log.changeset(cid) == cs
    assert log.diff(b0, b1) == {"engine/k.py": (base.files["engine/k.py"], new.files["engine/k.py"])}
    rec = log.changes()[0]
    assert (rec.status, rec.new_identity, rec.config, rec.content_purged) == ("committed", b1, {"sched.chunk": 4096}, False)

    later = IdentityManifest(files={"engine/k.py": blobs.put(b"newest\n")})
    b2 = identity_of(later)
    log.record_identity(b2, later, b1)
    removed = log.prune(keep=[b2], retention=1)
    assert removed > 0
    assert log.changes()[0].content_purged
    assert log.changeset(cid) is None
    with pytest.raises(KeyError):
        log.diff(b0, b1)
    assert log.manifest(Identity(b2)) == later
