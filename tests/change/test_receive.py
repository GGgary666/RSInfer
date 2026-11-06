"""change.md §6 and §5.1 without processes: what the coordinator refuses on receipt, blob retention."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest

from core.change.blobs import BlobStore, content_hash
from core.change.config import ChangeTunables
from core.change.coordinator import AcceptAll, Coordinator, PlanInput
from core.change.eventlog import EventLog
from core.change.state import IdentityManifest, identity_of
from core.change.types import ActionPlan, ChangeSet
from core.common.ids import Identity
from core.common.result import Ok, Result
from core.supervisor import Supervisor

BASE = Identity("base")


class Nothing:
    async def plan(self, inp: PlanInput) -> Result[ActionPlan]:
        return Ok(ActionPlan())


def coordinator(tmp: Path, **tun: int) -> Coordinator:
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
        c = coordinator(tmp_path, blob_retention=1)
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
