"""Retained metadata must not claim that already deleted content remains available."""
from pathlib import Path

import pytest

from core.change.blobs import BlobStore
from core.change.eventlog import EventLog
from core.change.state import IdentityManifest, identity_of
from core.change.types import ChangeResult, ChangeSet


@pytest.mark.parametrize("keep_candidate", [False, True], ids=["pruned-candidate", "retained-control"])
def test_aborted_change_reports_when_referenced_content_was_pruned(tmp_path: Path, keep_candidate: bool) -> None:
    blobs = BlobStore(tmp_path / "blobs")
    log = EventLog(tmp_path / "events.sqlite", blobs)
    try:
        base = IdentityManifest(files={"engine/k.py": blobs.put(b"old\n")})
        base_id = identity_of(base)
        log.record_identity(base_id, base, None)
        candidate = blobs.put(b"candidate that was rejected\n")
        cs = ChangeSet(base=base_id, files={"engine/k.py": candidate})
        cid = log.begin(cs)
        log.finish(ChangeResult("aborted", cid, base_id, None, None, {}))

        log.prune([base_id], retention=0, extra=[candidate] if keep_candidate else [], grace_s=0)

        assert log.changeset(cid) == cs  # metadata stays because its base is current
        record = log.changes()[0]
        assert record.content_purged == (not blobs.has(candidate)), (
            "content_purged must reflect missing referenced content, not only the changeset blob")
    finally:
        log.db.close()
