"""The node agent (doc/specs/nodeagent.md): the supervisor's hands on one machine.

It decides nothing. It materializes snapshots by content hash, spawns and
stops processes with the snapshot as their only engine code, hands out
shared-memory segment names, and reports heartbeats and exits.
"""

from __future__ import annotations

import asyncio
import ctypes
import itertools
import json
import os
import shutil
import signal
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import msgspec
import zmq.asyncio

from core.change.blobs import BlobStore, content_hash
from core.common.ids import Endpoint, HostId, Identity, ProcId
from core.common.messages import Envelope
from core.common.result import Failed, Ok, Result, fail
from core.ipc.rpc import RpcClient, RpcServer
from core.supervisor.protocol import (
    AllocShm,
    Blobs,
    Exited,
    FetchBlobs,
    Heartbeat,
    Materialize,
    Materialized,
    ProcBeat,
    ProcSpec,
    Register,
    Registered,
    ShmAllocated,
    Spawn,
    Spawned,
    Stop,
    Stopped,
    control_registry,
)
from platforms.base import PlatformInfo

COMPONENT = "nodeagent"


class AgentTunables(msgspec.Struct, frozen=True):
    heartbeat_interval_s: float = 1.0
    snapshot_retention: int = 5
    """Keep the identities in use plus this many most recent others."""
    rpc_timeout_s: float = 10.0


@dataclass
class _Proc:
    spec: ProcSpec
    process: asyncio.subprocess.Process
    run_dir: Path
    stopping: bool = False
    exit_code: int | None = None
    exited: asyncio.Event = field(default_factory=asyncio.Event)
    shm: list[str] = field(default_factory=list[str])


def _die_with_parent() -> None:
    """Spawned processes die with their agent: a process nobody manages is a leak."""
    if sys.platform == "linux":
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG


CODE_SOURCE_VARS = frozenset({"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE"})
"""Environment variables that decide where Python finds code: the snapshot decides that."""


def child_env(environ: Mapping[str, str], spec: ProcSpec, code_root: Path) -> dict[str, str]:
    """The environment of an engine process: engine code comes from ``code_root`` only.

    Other ``PYTHON*`` variables (hash seed, unbuffered output, fault handler) are
    not about where code comes from and are kept.
    """
    env = {k: v for k, v in {**environ, **spec.env}.items()
           if k not in CODE_SOURCE_VARS and k != "CUDA_VISIBLE_DEVICES"}
    env["PYTHONPATH"] = str(code_root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # snapshots are read-only
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in spec.gpus)
    return env


class NodeAgent:
    def __init__(
        self, *, host: HostId, address: str, supervisor: str, snapshot_root: Path, run_root: Path,
        platform: PlatformInfo, tun: AgentTunables, python: str = sys.executable,
    ) -> None:
        self.host = host
        self.address = address
        self.supervisor = supervisor
        self.snapshot_root = snapshot_root
        self.run_root = run_root
        self.platform = platform
        self.tun = tun
        self.python = python
        self.cache = BlobStore(snapshot_root / ".blobs")
        self.registry = control_registry()
        self.me = Endpoint(ProcId(host, "nodeagent", 0), "rpc")
        self.procs: dict[ProcId, _Proc] = {}
        self._materialized: dict[Identity, float] = {}
        self._materializing: set[Identity] = set()
        self._runs = itertools.count()
        self._shm = itertools.count()
        self._lifecycle: dict[ProcId, asyncio.Lock] = {}
        """Spawn and Stop of one ProcId run one at a time: a repeated request sees the first one's result."""

    # ------------------------------------------------------------ main

    async def run(self) -> int:
        zctx = zmq.asyncio.Context()
        self.server = RpcServer(zctx, self.registry, self.me, self.address)
        self.client = RpcClient(zctx, self.registry, self.me)
        serving = asyncio.create_task(self.server.serve(self._dispatch))
        r = await self.client.call(self.supervisor, Register(self.host, self.server.address, self.platform),
                                   timeout=self.tun.rpc_timeout_s)
        if isinstance(r, Failed) or not isinstance(r.value, Registered) or not r.value.accepted:
            print(f"registration refused: {r}", file=sys.stderr)
            serving.cancel()
            return 2
        try:
            while True:
                await asyncio.sleep(self.tun.heartbeat_interval_s)
                beats = [ProcBeat(p, rec.spec.incarnation, "exited" if rec.exit_code is not None else "running",
                                  rec.exit_code)
                         for p, rec in sorted(self.procs.items())]
                await self.client.call(self.supervisor, Heartbeat(self.host, beats),
                                       timeout=self.tun.rpc_timeout_s)
        finally:
            serving.cancel()
            for rec in self.procs.values():
                if rec.exit_code is None:
                    rec.process.kill()

    async def _dispatch(self, env: Envelope, msg: msgspec.Struct) -> Result[msgspec.Struct]:
        match msg:
            case Materialize():
                return await self.materialize(msg)
            case Spawn():
                async with self._lifecycle.setdefault(msg.spec.proc, asyncio.Lock()):
                    return await self.spawn(msg.spec)
            case Stop():
                async with self._lifecycle.setdefault(msg.proc, asyncio.Lock()):
                    return await self.stop(msg.proc, msg.timeout_s, msg.incarnation)
            case AllocShm():
                return self.alloc_shm(msg)
            case _:
                return fail(COMPONENT, "dispatch", f"unexpected {env.schema}", code="agent.unhandled")

    # ---------------------------------------------------------- snapshots

    def snapshot_dir(self, identity: Identity) -> Path:
        return self.snapshot_root / identity

    async def materialize(self, msg: Materialize) -> Result[msgspec.Struct]:
        target = self.snapshot_dir(msg.identity)
        if target.is_dir():
            self._materialized[msg.identity] = time.time()
            return Ok(Materialized(msg.identity, str(target), 0, len(msg.files)))
        self._materializing.add(msg.identity)
        try:
            return await self._materialize(msg, target)
        finally:
            self._materializing.discard(msg.identity)

    async def _materialize(self, msg: Materialize, target: Path) -> Result[msgspec.Struct]:
        missing = self.cache.missing(msg.files.values())
        if missing:
            r = await self.client.call(self.supervisor, FetchBlobs(missing), timeout=self.tun.rpc_timeout_s)
            if isinstance(r, Failed):
                return r
            got = r.value
            assert isinstance(got, Blobs)
            if got.missing:
                return fail(COMPONENT, "materialize", f"control plane lacks blobs {got.missing[:3]}...",
                            subject=msg.identity, code="agent.blob_missing")
            for h, data in got.blobs.items():
                if content_hash(data) != h:
                    return fail(COMPONENT, "materialize", f"blob {h[:12]} does not match its hash",
                                subject=msg.identity, code="agent.blob_corrupt")
                self.cache.put(data)
        tmp = Path(tempfile.mkdtemp(prefix=f".tmp-{msg.identity}-", dir=self.snapshot_root))
        try:
            for rel, h in sorted(msg.files.items()):
                if rel.startswith("/") or ".." in Path(rel).parts:
                    raise ValueError(f"bad snapshot path {rel!r}")
                dest = tmp / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                data = self.cache.get(h)
                if data is None or content_hash(data) != h:
                    raise ValueError(f"cached blob {h[:12]} for {rel} is missing or corrupt")
                dest.write_bytes(data)
            (tmp / "manifest.json").write_text(json.dumps(msg.files, indent=1, sort_keys=True))
            for rel, h in msg.files.items():  # verify what is on disk, not what we meant to write
                if content_hash((tmp / rel).read_bytes()) != h:
                    raise ValueError(f"{rel} does not match its hash after writing")
            for d in sorted({p for p in tmp.rglob("*") if p.is_dir()}, reverse=True):
                d.chmod(0o555)
            tmp.chmod(0o555)
            os.rename(tmp, target)
        except Exception as e:  # noqa: BLE001 - never leave a half-built snapshot
            _rmtree(tmp)
            return fail(COMPONENT, "materialize", f"{type(e).__name__}: {e}", subject=msg.identity,
                        code="agent.materialize_failed")
        self._materialized[msg.identity] = time.time()
        self._prune(msg.keep)
        return Ok(Materialized(msg.identity, str(target), len(missing), len(msg.files) - len(missing)))

    def _prune(self, keep_also: list[Identity]) -> None:
        """Keep the snapshots processes started from, those the supervisor says they still
        load code from (after in-place changes), those being made, and the newest others."""
        in_use = {rec.spec.identity for rec in self.procs.values() if rec.exit_code is None}
        keep = in_use | set(keep_also) | self._materializing
        others = sorted((t, i) for i, t in self._materialized.items() if i not in keep)
        for _, identity in others[: max(0, len(others) - self.tun.snapshot_retention)]:
            _rmtree(self.snapshot_dir(identity))
            del self._materialized[identity]

    # ----------------------------------------------------------- processes

    async def spawn(self, spec: ProcSpec) -> Result[msgspec.Struct]:
        if spec.proc.host != self.host:
            return fail(COMPONENT, "spawn", f"{spec.proc} belongs to another host", code="agent.wrong_host")
        old = self.procs.get(spec.proc)
        if old is not None and old.spec.incarnation == spec.incarnation:  # a repeated Spawn
            if old.exit_code is not None:  # never started again under the same incarnation
                return fail(COMPONENT, "spawn", f"incarnation {spec.incarnation} already ran and exited with "
                            f"{old.exit_code}", subject=str(spec.proc), code="agent.already_exited")
            return Ok(Spawned(spec.proc, old.process.pid))
        if old is not None and old.spec.incarnation > spec.incarnation:
            return fail(COMPONENT, "spawn", f"incarnation {spec.incarnation} is older than the last one here "
                        f"({old.spec.incarnation})", subject=str(spec.proc), code="agent.stale_spawn")
        if old is not None and old.exit_code is None:
            return fail(COMPONENT, "spawn", f"{spec.proc} is already running", code="agent.running")
        snapshot = self.snapshot_dir(spec.identity)
        if not snapshot.is_dir():
            return fail(COMPONENT, "spawn", f"identity {spec.identity} is not materialized",
                        subject=str(spec.proc), code="agent.not_materialized")
        run_dir = self.run_root / f"{spec.proc.role}-{spec.proc.index}-{next(self._runs)}"
        run_dir.mkdir(parents=True)
        spec_path = run_dir / "spec.msgpack"
        spec_path.write_bytes(msgspec.msgpack.encode(spec))
        log = open(run_dir / "log", "wb")  # noqa: SIM115 - owned by the child
        process = await asyncio.create_subprocess_exec(
            self.python, "-P", "-s", "-m", "unit.runtime.host", str(spec_path),
            cwd=run_dir, env=child_env(os.environ, spec, snapshot / "engine"),
            stdin=asyncio.subprocess.DEVNULL, stdout=log, stderr=log,
            preexec_fn=_die_with_parent,
        )
        log.close()
        rec = self.procs[spec.proc] = _Proc(spec, process, run_dir)
        asyncio.create_task(self._watch(rec))
        return Ok(Spawned(spec.proc, process.pid))

    async def _watch(self, rec: _Proc) -> None:
        code = await rec.process.wait()
        rec.exit_code = code
        rec.exited.set()
        for name in rec.shm:
            _unlink_shm(name)
        await self.client.call(self.supervisor,
                               Exited(rec.spec.proc, rec.spec.incarnation, code, killed=code == -signal.SIGKILL),
                               timeout=self.tun.rpc_timeout_s)

    async def stop(self, proc: ProcId, timeout_s: float, incarnation: int) -> Result[msgspec.Struct]:
        rec = self.procs.get(proc)
        if rec is None:
            return fail(COMPONENT, "stop", f"{proc} is unknown", code="agent.unknown_proc")
        if rec.spec.incarnation != incarnation:  # a late Stop must not hit the successor
            return fail(COMPONENT, "stop", f"incarnation {incarnation} is not the one here ({rec.spec.incarnation})",
                        subject=str(proc), code="agent.stale_stop")
        killed = False
        if rec.exit_code is None:
            rec.stopping = True
            rec.process.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(rec.exited.wait(), timeout_s)
            except TimeoutError:
                rec.process.kill()
                killed = True
                await rec.exited.wait()
        assert rec.exit_code is not None
        return Ok(Stopped(proc, rec.exit_code, killed))

    def alloc_shm(self, msg: AllocShm) -> Result[msgspec.Struct]:
        rec = self.procs.get(msg.proc)
        name = f"rsinfer-{self.host}-{next(self._shm)}-{uuid.uuid4().hex[:8]}"
        if rec is not None:
            rec.shm.append(name)
        return Ok(ShmAllocated(name))


def _rmtree(path: Path) -> None:
    if not path.exists():
        return
    for d in [path, *(p for p in path.rglob("*") if p.is_dir())]:
        d.chmod(0o755)
    shutil.rmtree(path, ignore_errors=True)


def _unlink_shm(name: str) -> None:
    try:
        os.unlink(f"/dev/shm/{name}")
    except FileNotFoundError:
        pass
