"""A client of the control interface, for agents, operators and tests.

``python -m core.control.client --control <address> status|check|apply|introspect|logs``

``apply`` exits 1 unless the change committed (or, with ``--dry-run``, was planned).
``apply`` and ``check`` take a workspace (``--source``): the client hashes its
``engine/`` tree, asks the control plane which contents it lacks, uploads only
those, and sends the whole file list; ``core.api`` works out the diff.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import msgspec
import zmq.asyncio

from core.change.blobs import content_hash
from core.change.types import ChangeMeta
from core.common.ids import Endpoint, HostId, Identity, ProcId
from core.common.result import Failed, Ok, Result
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
    HaveBlobs,
    MissingBlobs,
    PutBlobs,
)
from core.ipc.rpc import RpcClient
from core.supervisor.protocol import control_registry
from core.supervisor.topology import Topology


def workspace_files(source: Path) -> dict[str, bytes]:
    """``{"engine/...": content}`` for every file of a workspace's ``engine/`` tree."""
    root = source / "engine"
    return {
        "engine/" + p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc")
    }


class ControlClient:
    def __init__(self, zctx: zmq.asyncio.Context, control: str, *, timeout_s: float = 600.0) -> None:
        self.control = control
        self.timeout_s = timeout_s
        self._rpc = RpcClient(zctx, control_registry(), Endpoint(ProcId(HostId("client"), "client", 0), "rpc"))

    def close(self) -> None:
        self._rpc.close()

    async def _call[T: msgspec.Struct](self, msg: msgspec.Struct, reply: type[T]) -> Result[T]:
        r = await self._rpc.call(self.control, msg, timeout=self.timeout_s)
        if isinstance(r, Failed):
            return r
        assert isinstance(r.value, reply), r.value
        return Ok(r.value)

    async def status(self) -> Result[ApiStatusReport]:
        return await self._call(ApiStatus(), ApiStatusReport)

    async def upload(self, contents: dict[str, bytes]) -> Result[dict[str, str]]:
        """Store what the control plane lacks; returns ``{path: hash}``."""
        files = {path: content_hash(data) for path, data in contents.items()}
        missing = await self._call(HaveBlobs(sorted(set(files.values()))), MissingBlobs)
        if isinstance(missing, Failed):
            return missing
        by_hash = {files[p]: data for p, data in contents.items()}
        if missing.value.hashes:
            r = await self._rpc.call(self.control, PutBlobs({h: by_hash[h] for h in missing.value.hashes}),
                                     timeout=self.timeout_s)
            if isinstance(r, Failed):
                return r
        return Ok(files)

    async def change(self, source: Path | None = None, *, config: dict[str, Any] | None = None,
                     topology: Topology | None = None, base: Identity | None = None,
                     **meta: Any) -> Result[ApiChange]:
        files: dict[str, str] | None = None
        if source is not None:
            up = await self.upload(workspace_files(source))
            if isinstance(up, Failed):
                return up
            files = up.value
        return Ok(ApiChange(files=files, config=dict(config or {}), topology=topology, base=base,
                            meta=msgspec.convert(meta, type=ChangeMeta)))

    async def check(self, change: ApiChange) -> Result[ApiChecked]:
        return await self._call(ApiCheck(change), ApiChecked)

    async def apply(self, change: ApiChange, *, dry_run: bool = False) -> Result[ApiApplied]:
        return await self._call(ApiApply(change, dry_run), ApiApplied)

    async def introspect(self) -> Result[ApiIntrospection]:
        return await self._call(ApiIntrospect(), ApiIntrospection)

    async def logs(self, since: int = 0) -> Result[ApiLogRecords]:
        return await self._call(ApiLogs(since), ApiLogRecords)


def _print(r: Result[msgspec.Struct]) -> int:
    """Print the reply; the exit code is 0 only if the request did what it asked for: a change
    that was refused, failed or rolled back exits 1 although the control plane answered."""
    if isinstance(r, Failed):
        print(json.dumps({"failed": msgspec.to_builtins(r.diagnostics)}, indent=2))
        return 1
    print(json.dumps(msgspec.to_builtins(r.value), indent=2, default=str))
    if isinstance(r.value, ApiApplied) and r.value.result.status not in ("committed", "planned"):
        return 1
    return 0


async def _main(args: argparse.Namespace) -> int:
    zctx = zmq.asyncio.Context()
    client = ControlClient(zctx, args.control)
    try:
        match args.command:
            case "status":
                return _print(await client.status())
            case "introspect":
                return _print(await client.introspect())
            case "logs":
                return _print(await client.logs(args.since))
            case "check" | "apply":
                config = json.loads(args.config) if args.config else {}
                topology = msgspec.json.decode(Path(args.topology).read_bytes(), type=Topology) \
                    if args.topology else None
                change = await client.change(args.source, config=config, topology=topology,
                                             submitter=args.submitter, intent=args.intent)
                if isinstance(change, Failed):
                    return _print(change)
                if args.command == "check":
                    return _print(await client.check(change.value))
                return _print(await client.apply(change.value, dry_run=args.dry_run))
            case _:
                raise AssertionError(args.command)
    finally:
        client.close()
        zctx.destroy(linger=0)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="core.control.client")
    ap.add_argument("--control", required=True, help="the control endpoint printed by serve")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("introspect")
    logs = sub.add_parser("logs")
    logs.add_argument("--since", type=int, default=0)
    for name in ("check", "apply"):
        p = sub.add_parser(name)
        p.add_argument("--source", type=Path, help="workspace whose engine/ tree to apply")
        p.add_argument("--config", help="config values as JSON")
        p.add_argument("--topology", help="topology as a JSON file")
        p.add_argument("--submitter", default="")
        p.add_argument("--intent", default="")
        if name == "apply":
            p.add_argument("--dry-run", action="store_true")
    return asyncio.run(_main(ap.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
