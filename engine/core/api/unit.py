"""``core.api``: the engine's remote control (doc/specs/control-plane.md §5).

Every outside operation comes in here: status, static checks, apply (with
``--dry-run``), introspection, logs, and the internal changes units submit. It
turns a request into a change-set and hands it to ``core.change``; it never
executes a change itself.
"""

from __future__ import annotations

import asyncio
from typing import Any

import msgspec

from core.change.blobs import content_hash
from core.change.coordinator import PlanInput
from core.change.state import apply_changeset
from core.change.types import ChangeSet
from core.common.result import Diagnostic, Failed, Ok, Result, fail
from core.control.protocol import (
    ApiApplied,
    ApiApply,
    ApiChange,
    ApiCheck,
    ApiChecked,
    ApiIntrospect,
    ApiIntrospection,
    ApiLogRecords,
    ApiLogs,
    ApiStatus,
    ApiStatusReport,
    BlobsStored,
    ChangeQueued,
    HaveBlobs,
    MissingBlobs,
    ProcRow,
    PutBlobs,
    SubmitChange,
)
from core.supervisor.protocol import Introspect, ProcIntrospection

COMPONENT = "core.api"


class Api:
    """``ctrl.api@1``."""

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx
        self._coord = ctx.use("ctrl.change@1")
        self._sup = ctx.use("ctrl.supervisor@1")
        self._blobs = ctx.use("ctrl.blobs@1")
        self._log = ctx.use("ctrl.events@1")
        self._host = ctx.use("ctrl.self@1")

    async def handle(self, msg: msgspec.Struct) -> Result[msgspec.Struct]:
        match msg:
            case ApiStatus():
                return Ok(self._status())
            case HaveBlobs():
                return Ok(MissingBlobs(self._blobs.missing(msg.hashes)))
            case PutBlobs():
                return self._put(msg)
            case ApiCheck():
                return await self._check(msg.change)
            case ApiApply():
                result = await self._coord.submit(self._changeset(msg.change), dry_run=msg.dry_run)
                return Ok(ApiApplied(result))
            case SubmitChange():
                self._coord.enqueue(msg.cs)  # internal: queued, not waited for
                return Ok(ChangeQueued(len(self._coord.queue)))
            case ApiIntrospect():
                return Ok(await self._introspect())
            case ApiLogs():
                changes = [c for c in self._log.changes() if c.id > msg.since]
                events = [e for e in self._log.events() if e[0] is None or e[0] > msg.since]
                return Ok(ApiLogRecords(changes, events))
            case _:
                return fail(COMPONENT, "dispatch", f"cannot handle {type(msg).__name__}", code="api.unhandled")

    def _status(self) -> ApiStatusReport:
        coord, sup = self._coord, self._sup
        rows = [ProcRow(str(s.proc), s.state, s.identity, s.units, sup.local_pid(s.proc)) for s in sup.status()]
        return ApiStatusReport(coord.identity, coord.state_version, coord.recovery_required, coord.busy,
                               len(coord.queue), rows, self._host.introspect_self().units)

    def _put(self, msg: PutBlobs) -> Result[msgspec.Struct]:
        bad = sorted(h for h, data in msg.blobs.items() if content_hash(data) != h)
        if bad:
            return fail(COMPONENT, "upload", f"{len(bad)} blobs do not match their hash", subject=bad[0],
                        code="api.blob_corrupt")
        for data in msg.blobs.values():
            self._blobs.put(data)
        return Ok(BlobsStored(len(msg.blobs)))

    def _changeset(self, change: ApiChange) -> ChangeSet:
        """The difference between what is running and what the client wants."""
        coord = self._coord
        files: dict[str, str | None] = {}
        if change.files is not None:
            current = coord.manifest.files
            for path in sorted({*current, *change.files}):
                if current.get(path) != change.files.get(path):
                    files[path] = change.files.get(path)
        source = "config" if change.config and not files and change.topology is None else "apply"
        return ChangeSet(base=change.base or coord.identity, files=files, config=dict(change.config),
                         deps=dict(change.deps), topology=change.topology, artifacts=dict(change.artifacts),
                         source=source, meta=change.meta, state=change.state,
                         state_only=bool(change.state) and not (files or change.config or change.topology
                                                               or change.deps or change.artifacts))

    async def _check(self, change: ApiChange) -> Result[msgspec.Struct]:
        cs = self._changeset(change)
        missing = self._blobs.missing(h for h in cs.files.values() if h is not None)
        if missing:
            return fail(COMPONENT, "check", f"{len(missing)} file contents are not uploaded", code="api.missing_blobs")
        coord = self._coord
        new = apply_changeset(coord.manifest, cs)
        checker, identity = self._ctx.get("ctrl.manifest@1"), self._ctx.get("ctrl.identity@1")
        if checker is None or identity is None:
            return fail(COMPONENT, "check", "core.manifest or core.identity is not loaded", code="control.unit_missing")
        r = await checker.check(PlanInput(cs, coord.identity, coord.manifest, new, self._sup.status()))
        if isinstance(r, Failed):
            return r
        warnings: list[Diagnostic] = list(r.diagnostics)
        return Ok(ApiChecked(identity.identity(new, self._sup.reference_platform), warnings))

    async def _introspect(self) -> ApiIntrospection:
        running = [s.proc for s in self._sup.status() if s.state == "RUNNING"]
        replies = await asyncio.gather(*(self._sup.call(p, Introspect()) for p in running))
        procs = [r.value for r in replies if isinstance(r, Ok) and isinstance(r.value, ProcIntrospection)]
        return ApiIntrospection(procs, self._host.introspect_self())


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        ctx.provide("ctrl.api@1", Api(ctx))
