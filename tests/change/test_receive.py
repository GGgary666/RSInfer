"""change.md §6, §5.1 and §7 test 17 without processes: refusals on receipt, blob retention."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from core.change.blobs import BlobStore, content_hash
from core.change.config import ChangeTunables
from core.change.coordinator import AcceptAll, Coordinator, PlanInput
from core.change.eventlog import EventLog
from core.change.queue import ChangeQueue
from core.change.state import IdentityManifest, identity_of
from core.change.types import ActionPlan, ChangeSet
from core.common.ids import Identity
from core.common.result import Ok, Result
from core.supervisor import Supervisor

BASE = Identity("base")


class Nothing:
    async def plan(self, inp: PlanInput) -> Result[ActionPlan]:
        return Ok(ActionPlan())


def coordinator(tmp: Path, **tun: Any) -> Coordinator:
    log = EventLog(tmp / "events.sqlite", BlobStore(tmp / "blobs"))
    sup = cast(Supervisor, object())  # never reached: nothing here runs a change
    c = Coordinator(sup, log, planner=Nothing(), checker=AcceptAll(), tun=ChangeTunables(**tun))
    c.identity = BASE
    return c


@pytest.mark.parametrize(("cs", "code"), [
    (ChangeSet(base=BASE, source="internal"), "change.missing_computed_under"),
    (ChangeSet(base=BASE, source="internal", computed_under=BASE, state_only=True,
               files={"engine/x.py": "0" * 64}), "change.state_only_with_content"),
    (ChangeSet(base=BASE, source="internal", computed_under=BASE, state_only=True, config={"k": 1}),
     "change.state_only_with_content"),
], ids=["internal-without-computed_under", "state_only-with-files", "state_only-with-config"])
def test_invalid_changes_are_refused_on_receipt(tmp_path: Path, cs: ChangeSet, code: str) -> None:
    async def main() -> None:
        c = coordinator(tmp_path)
        r = await c.enqueue(cs)
        assert r.status == "failed" and [d.code for d in r.diagnostics] == [code]
        assert len(c.queue) == 0 and c.identity == BASE
        assert c.log.changes()[-1].status == "failed"

    asyncio.run(main())


def test_blob_retention_keeps_what_queued_changes_still_need(tmp_path: Path) -> None:
    async def main() -> None:
        c = coordinator(tmp_path, blob_retention=1, blob_grace_s=0.0)
        blobs = c.log.blobs
        for n in range(3):
            m = IdentityManifest(files={"engine/k.py": blobs.put(f"v{n}\n".encode())})
            c.log.record_identity(identity_of(m), m, None)
        current = identity_of(IdentityManifest(files={"engine/k.py": blobs.put(b"v0\n")}))
        c.identity = current
        queued = blobs.put(b"uploaded for a change still in the queue\n")
        c.enqueue(ChangeSet(base=current, files={"engine/k.py": queued}))

        v0, v1, v2 = (content_hash(f"v{n}\n".encode()) for n in range(3))
        assert c.prune_blobs() == 2  # v1's file and file list: neither current, nor recent, nor queued
        assert not blobs.has(v1)
        assert blobs.has(v0) and blobs.has(v2) and blobs.has(queued)
        assert c.log.manifest(current) is not None

    asyncio.run(main())


def test_young_blobs_are_never_pruned(tmp_path: Path) -> None:
    """§7 test 17: an agent uploads first and submits later."""
    c = coordinator(tmp_path, blob_grace_s=3600.0)
    blobs = c.log.blobs
    fresh = blobs.put(b"uploaded a moment ago, not referenced yet\n")
    stale = blobs.put(b"uploaded two hours ago, never used\n")
    two_hours_ago = time.time() - 7200
    os.utime(blobs.root / stale[:2] / stale[2:], (two_hours_ago, two_hours_ago))
    assert c.prune_blobs() == 1
    assert blobs.has(fresh) and not blobs.has(stale)

    reused = blobs.put(b"old content, uploaded again\n")
    os.utime(blobs.root / reused[:2] / reused[2:], (two_hours_ago, two_hours_ago))
    blobs.put(b"old content, uploaded again\n")  # uploading it again restarts its grace period
    assert c.prune_blobs() == 0 and blobs.has(reused)


def test_queued_artifacts_survive_pruning_and_missing_ones_are_refused(tmp_path: Path) -> None:
    async def main() -> None:
        c = coordinator(tmp_path, blob_grace_s=3600.0)
        blobs = c.log.blobs
        artifact = blobs.put(b"weights of a refit, uploaded long ago\n")
        two_hours_ago = time.time() - 7200
        os.utime(blobs.root / artifact[:2] / artifact[2:], (two_hours_ago, two_hours_ago))
        c.enqueue(ChangeSet(base=BASE, artifacts={"model.weights": artifact}))
        c.prune_blobs()
        assert blobs.has(artifact)

        # a change that names an artifact nobody uploaded is refused before anything runs
        c.sup = cast(Supervisor, SimpleNamespace(blobs=blobs, handlers={}))
        c.queue = ChangeQueue()
        await c.start()
        try:
            r = await c.submit(ChangeSet(base=BASE, artifacts={"model.weights": content_hash(b"never uploaded")}))
            assert r.status == "failed" and [d.code for d in r.diagnostics] == ["change.missing_blobs"]
        finally:
            c.close()

    asyncio.run(main())
