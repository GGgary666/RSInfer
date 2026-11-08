"""The supervisor (doc/specs/supervisor.md): holds the topology and makes the
processes match it, only through node agents. It reports lost processes and
machines; deciding what to do about them is not its job (nor the framework's).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal

import msgspec
import zmq.asyncio

from core.change.blobs import BlobStore
from core.common.ids import Endpoint, HostId, Identity, ProcId
from core.common.messages import Envelope, SchemaRegistry
from core.common.result import Diagnostic, Failed, Ok, Result, diag, fail
from core.ipc.rpc import RpcClient, RpcServer
from core.ipc.stream import StreamAddress
from core.supervisor.protocol import (
    AllocShm,
    Blobs,
    CheckedIn,
    CheckIn,
    Exited,
    FetchBlobs,
    Found,
    Heartbeat,
    Lookup,
    Materialize,
    ProcSpec,
    PublishEndpoint,
    Register,
    Registered,
    Spawn,
    Spawned,
    Stop,
    Stopped,
    UnitBrief,
    control_registry,
)
from core.supervisor.topology import DesiredProc, Topology, TopologyPlan, desired_procs, diff_topology
from platforms.base import PlatformInfo
from unit.runtime.types import InstanceState, ProcessPlan

COMPONENT = "supervisor"

ProcState = Literal["DECLARED", "SPAWNING", "RUNNING", "STOPPING", "STOPPED", "LOST"]


class SupervisorTunables(msgspec.Struct, frozen=True):
    heartbeat_timeout_s: float = 5.0
    """A node agent silent for this long means the whole machine is lost."""
    checkin_timeout_s: float = 60.0
    rpc_timeout_s: float = 30.0
    stop_timeout_s: float = 10.0
    monitor_interval_s: float = 0.2


class ProcStatus(msgspec.Struct, frozen=True):
    proc: ProcId
    state: ProcState
    identity: Identity | None
    units: list[UnitBrief]
    exit_code: int | None = None


class Loss(msgspec.Struct, frozen=True):
    """Reported to ``core.change``: the unit of failure is a process or a machine."""

    kind: Literal["process_lost", "machine_lost"]
    host: HostId
    procs: list[ProcId]
    reason: str


@dataclass
class _Host:
    id: HostId
    agent: str
    platform: PlatformInfo
    last_seen: float
    lost: bool = False


@dataclass
class _Proc:
    desired: DesiredProc
    spec: ProcSpec
    state: ProcState = "DECLARED"
    checkin: CheckIn | None = None
    published: dict[str, StreamAddress] = field(default_factory=dict[str, StreamAddress])
    """Readers it opened after checking in (``PublishEndpoint``)."""
    units: list[UnitBrief] = field(default_factory=list[UnitBrief])
    """What it runs: from its check-in, then from each change it confirmed (``Resumed``)."""
    exit_code: int | None = None
    local_pid: int | None = None
    checked_in: asyncio.Event = field(default_factory=asyncio.Event)
    snapshots: set[Identity] = field(default_factory=set[Identity])
    """Snapshots the process loads code from: the one it started from, and those of in-place changes."""


type PlanFor = Callable[[DesiredProc], ProcessPlan]
type ControlHandler = Callable[[Envelope, msgspec.Struct], Awaitable[Result[msgspec.Struct]]]


class Supervisor:
    def __init__(
        self, zctx: zmq.asyncio.Context, *, host: HostId, bind_address: str, blobs: BlobStore,
        tun: SupervisorTunables | None = None, registry: SchemaRegistry | None = None,
        proc_tunables: dict[str, float | int] | None = None,
    ) -> None:
        self.tun = tun or SupervisorTunables()
        self.registry = registry or control_registry()
        self.me = Endpoint(ProcId(host, "control", 0), "supervisor")
        self.blobs = blobs
        self.bind_address = bind_address
        self.proc_tunables = dict(proc_tunables or {})
        self.server = RpcServer(zctx, self.registry, self.me, bind_address)
        self.address = self.server.address
        self.client = RpcClient(zctx, self.registry, self.me)
        self.hosts: dict[HostId, _Host] = {}
        self.procs: dict[ProcId, _Proc] = {}
        self.topology: Topology | None = None
        self.reference_platform: PlatformInfo | None = None
        self.losses: asyncio.Queue[Loss] = asyncio.Queue()
        self.loss_listeners: list[Callable[[Loss], None]] = []
        self._tasks: list[asyncio.Task[None]] = []
        self._host_changed = asyncio.Condition()
        self._previous: dict[ProcId, tuple[DesiredProc, ProcSpec]] = {}
        self._incarnation: dict[ProcId, int] = {}
        self.handlers: dict[type[msgspec.Struct], ControlHandler] = {}
        """Other control-plane messages arriving at this endpoint, e.g. the coordinator's ``QueryDecision``."""

    async def start(self) -> None:
        self._tasks.append(asyncio.create_task(self.server.serve(self._dispatch)))
        self._tasks.append(asyncio.create_task(self._monitor()))

    def close(self) -> None:
        for t in self._tasks:
            t.cancel()
        self.server.close()
        self.client.close()

    # ---------------------------------------------------------- inbound

    async def _dispatch(self, env: Envelope, msg: msgspec.Struct) -> Result[msgspec.Struct]:
        match msg:
            case Register():
                return await self._register(msg)
            case Heartbeat():
                return self._heartbeat(msg)
            case Exited():
                return self._exited(msg)
            case CheckIn():
                return self._check_in(msg)
            case FetchBlobs():
                return self._fetch(msg)
            case Lookup():
                return Ok(Found(self.endpoint(msg.proc, msg.channel)))
            case PublishEndpoint():
                return self._publish(msg)
            case AllocShm():
                # A process host asks for a ring name for a reader of its own: the node agent of
                # that host names it and removes the segment when the process is gone.
                host = self.hosts.get(msg.proc.host)
                if host is None or host.lost:
                    return fail(COMPONENT, "alloc_shm", "the process's host is not registered",
                                subject=str(msg.proc), code="supervisor.no_host")
                return await self.client.call(host.agent, msg, timeout=self.tun.rpc_timeout_s)
            case _:
                handler = self.handlers.get(type(msg))
                if handler is not None:
                    return await handler(env, msg)
                return fail(COMPONENT, "dispatch", f"unexpected {env.schema}", code="supervisor.unhandled")

    async def _register(self, msg: Register) -> Result[msgspec.Struct]:
        if self.reference_platform is None:
            self.reference_platform = msg.platform
        elif msg.platform != self.reference_platform:
            return Ok(Registered(False, [diag(
                COMPONENT, "register", f"platform {msg.platform} differs from {self.reference_platform}",
                subject=msg.host, code="supervisor.platform_mismatch",
                hint="every host needs the same driver, CUDA, torch and native libraries")]))
        self.hosts[msg.host] = _Host(msg.host, msg.agent, msg.platform, time.monotonic())
        async with self._host_changed:
            self._host_changed.notify_all()
        return Ok(Registered(True))

    def _heartbeat(self, msg: Heartbeat) -> Result[msgspec.Struct]:
        h = self.hosts.get(msg.host)
        if h is None or h.lost:
            return fail(COMPONENT, "heartbeat", "host is not registered", subject=msg.host,
                        code="supervisor.unregistered")
        h.last_seen = time.monotonic()
        for beat in msg.procs:
            rec = self.procs.get(beat.proc)
            if rec is None or rec.spec.incarnation != beat.incarnation:
                continue  # about an earlier process with this ProcId
            if beat.state == "exited" and rec.state in ("SPAWNING", "RUNNING"):
                self._lose_proc(rec, f"exited with {beat.exit_code}")
        return Ok(CheckedIn())

    def _exited(self, msg: Exited) -> Result[msgspec.Struct]:
        rec = self.procs.get(msg.proc)
        if rec is not None and rec.spec.incarnation == msg.incarnation:
            rec.exit_code = msg.exit_code
            if rec.state == "STOPPING":
                rec.state = "STOPPED"
            elif rec.state in ("SPAWNING", "RUNNING"):
                self._lose_proc(rec, f"exited with {msg.exit_code}")
            rec.checked_in.set()
        return Ok(CheckedIn())

    def _check_in(self, msg: CheckIn) -> Result[msgspec.Struct]:
        rec = self.procs.get(msg.proc)
        if rec is not None and rec.state == "RUNNING" and rec.checkin == msg:
            return Ok(CheckedIn())  # a repeated check-in
        if rec is None or rec.state != "SPAWNING" or rec.spec.incarnation != msg.incarnation:
            return fail(COMPONENT, "check_in", "process was not being spawned", subject=str(msg.proc),
                        code="supervisor.unexpected_check_in")
        if msg.identity != rec.spec.identity:
            return fail(COMPONENT, "check_in", f"runs {msg.identity}, expected {rec.spec.identity}",
                        subject=str(msg.proc), code="supervisor.wrong_identity")
        rec.checkin = msg
        rec.units = list(msg.units)
        rec.state = "RUNNING"
        rec.checked_in.set()
        return Ok(CheckedIn())

    def _fetch(self, msg: FetchBlobs) -> Result[msgspec.Struct]:
        found: dict[str, bytes] = {}
        missing: list[str] = []
        for h in msg.hashes:
            data = self.blobs.get(h)
            if data is None:
                missing.append(h)
            else:
                found[h] = data
        return Ok(Blobs(found, missing))

    # ------------------------------------------------------------ losses

    def _report(self, loss: Loss) -> None:
        self.losses.put_nowait(loss)
        for listener in self.loss_listeners:
            listener(loss)

    def _lose_proc(self, rec: _Proc, reason: str) -> None:
        rec.state = "LOST"
        rec.checked_in.set()
        self._report(Loss("process_lost", rec.desired.proc.host, [rec.desired.proc], reason))

    async def _monitor(self) -> None:
        while True:
            await asyncio.sleep(self.tun.monitor_interval_s)
            now = time.monotonic()
            for h in self.hosts.values():
                if h.lost or now - h.last_seen <= self.tun.heartbeat_timeout_s:
                    continue
                h.lost = True
                procs = sorted(p for p, rec in self.procs.items()
                               if p.host == h.id and rec.state in ("SPAWNING", "RUNNING", "STOPPING"))
                for p in procs:
                    self.procs[p].state = "LOST"
                    self.procs[p].checked_in.set()
                self._report(Loss("machine_lost", h.id, procs,
                                  f"no heartbeat for {now - h.last_seen:.1f}s"))

    # ------------------------------------------------------------ queries

    def endpoint(self, proc: ProcId, channel: str) -> StreamAddress | None:
        rec = self.procs.get(proc)
        if rec is None or rec.checkin is None or rec.state != "RUNNING":
            return None
        return rec.published.get(channel) or rec.checkin.endpoints.get(channel)

    def _publish(self, msg: PublishEndpoint) -> Result[msgspec.Struct]:
        rec = self.procs.get(msg.proc)
        if rec is None or rec.checkin is None or rec.spec.incarnation != msg.incarnation:
            return fail(COMPONENT, "publish", "process has not checked in as this incarnation",
                        subject=str(msg.proc), code="supervisor.unexpected_publish")
        rec.published[msg.channel] = msg.address
        return Ok(CheckedIn())

    def status(self) -> list[ProcStatus]:
        return [
            ProcStatus(p, rec.state, rec.spec.identity, list(rec.units), rec.exit_code)
            for p, rec in sorted(self.procs.items())
        ]

    def proc_state(self, proc: ProcId) -> ProcState | None:
        rec = self.procs.get(proc)
        return rec.state if rec else None

    def checkin_of(self, proc: ProcId) -> CheckIn | None:
        rec = self.procs.get(proc)
        return rec.checkin if rec else None

    def local_pid(self, proc: ProcId) -> int | None:
        rec = self.procs.get(proc)
        return rec.local_pid if rec else None

    async def call(self, proc: ProcId, msg: msgspec.Struct, *, timeout: float | None = None,
                   channel: str = "rpc") -> Result[msgspec.Struct]:
        address = self.endpoint(proc, channel)
        if address is None or address.tcp is None:
            return fail(COMPONENT, "call", f"no {channel} endpoint", subject=str(proc), code="supervisor.no_endpoint")
        return await self.client.call(address.tcp, msg, timeout=timeout or self.tun.rpc_timeout_s)

    async def wait_for_hosts(self, hosts: Iterable[HostId], timeout: float) -> Result[None]:
        wanted = set(hosts)

        def ready() -> bool:
            return all(h in self.hosts and not self.hosts[h].lost for h in wanted)

        try:
            async with self._host_changed:
                await asyncio.wait_for(self._host_changed.wait_for(ready), timeout)
        except TimeoutError:
            missing = sorted(h for h in wanted if h not in self.hosts or self.hosts[h].lost)
            return fail(COMPONENT, "hosts", f"hosts did not register: {missing}", code="supervisor.no_host")
        return Ok(None)

    # ------------------------------------------------------------ actions

    def _spec(self, topo: Topology, d: DesiredProc, identity: Identity, plan_for: PlanFor) -> ProcSpec:
        return ProcSpec(
            proc=d.proc, identity=identity, plan=plan_for(d), control=self.address,
            bind_address=topo.host(d.proc.host).address, env=dict(d.env), gpus=list(d.gpus),
            platform=d.platform, platform_options=dict(d.platform_options), tunables=dict(self.proc_tunables),
        )

    async def apply_topology(
        self, topo: Topology, identity: Identity, files: dict[str, str], plan_for: PlanFor,
    ) -> Result[TopologyPlan]:
        """Make the running processes match ``topo``; only affected processes are touched."""
        problems = topo.check()
        if problems:
            return Failed([diag(COMPONENT, "topology", p, code="supervisor.bad_topology") for p in problems])
        r = await self.wait_for_hosts([h.id for h in topo.hosts], self.tun.rpc_timeout_s)
        if isinstance(r, Failed):
            return r
        desired = desired_procs(topo)
        specs = {p: self._spec(topo, d, identity, plan_for) for p, d in desired.items()}
        current = {p: rec.desired for p, rec in self.procs.items() if rec.state == "RUNNING"}
        changed = {p for p in desired
                   if p in current and specs[p] != msgspec.structs.replace(self.procs[p].spec, incarnation=0)}
        plan = diff_topology(current, desired, changed)

        to_spawn = [*plan.restart, *plan.start]
        hosts = sorted({p.host for p in to_spawn})
        mats = await asyncio.gather(*(self.materialize(h, identity, files) for h in hosts))
        errors = [d for m in mats if isinstance(m, Failed) for d in m.diagnostics]
        if errors:
            return Failed(errors)
        stops = await asyncio.gather(*(self.stop(p) for p in [*plan.stop, *plan.restart]))
        errors = [d for s in stops if isinstance(s, Failed) for d in s.diagnostics]
        if errors:
            return Failed(errors)
        for p in plan.stop:
            del self.procs[p]
        spawns = await asyncio.gather(*(self.spawn(desired[p], specs[p]) for p in to_spawn))
        errors = [d for s in spawns if isinstance(s, Failed) for d in s.diagnostics]
        self.topology = topo
        return Failed(errors) if errors else Ok(plan)

    async def materialize(self, host: HostId, identity: Identity, files: dict[str, str]) -> Result[msgspec.Struct]:
        h = self.hosts[host]
        # every record: a stopped or lost process may still be started again from its spec
        keep = sorted({i for p, rec in self.procs.items() if p.host == host for i in rec.snapshots})
        return await self.client.call(h.agent, Materialize(identity, files, keep), timeout=self.tun.rpc_timeout_s)

    async def spawn(self, desired: DesiredProc, spec: ProcSpec) -> Result[None]:
        self._incarnation[desired.proc] = self._incarnation.get(desired.proc, 0) + 1
        spec = msgspec.structs.replace(spec, incarnation=self._incarnation[desired.proc])
        rec = self.procs[desired.proc] = _Proc(desired, spec, "SPAWNING", snapshots={spec.identity})
        r = await self.client.call(self.hosts[desired.proc.host].agent, Spawn(spec), timeout=self.tun.rpc_timeout_s)
        if isinstance(r, Failed):
            rec.state = "LOST"
            return r
        assert isinstance(r.value, Spawned)
        rec.local_pid = r.value.local_pid
        try:
            await asyncio.wait_for(rec.checked_in.wait(), self.tun.checkin_timeout_s)
        except TimeoutError:
            return fail(COMPONENT, "spawn", "did not check in", subject=str(desired.proc), code="supervisor.no_check_in")
        if rec.state != "RUNNING" or rec.checkin is None:
            return fail(COMPONENT, "spawn", f"is {rec.state} (exit code {rec.exit_code})", subject=str(desired.proc),
                        code="supervisor.spawn_failed")
        bad = [u for u in rec.checkin.units if u.state is InstanceState.FAILED]
        if bad:
            return Failed([diag(COMPONENT, "spawn", u.error or "failed", subject=f"{desired.proc}:{u.unit}",
                                code="supervisor.unit_failed") for u in bad])
        return Ok(None)

    async def stop(self, proc: ProcId, timeout_s: float | None = None) -> Result[None]:
        rec = self.procs.get(proc)
        if rec is None or rec.state == "STOPPED":
            return Ok(None)
        if self.hosts[proc.host].lost:
            return fail(COMPONENT, "stop", "host is lost; process stop is unconfirmed", subject=str(proc),
                        code="supervisor.stop_unconfirmed")
        rec.state = "STOPPING"
        r = await self.client.call(self.hosts[proc.host].agent,
                                   Stop(proc, timeout_s if timeout_s is not None else self.tun.stop_timeout_s,
                                        rec.spec.incarnation),
                                   timeout=self.tun.rpc_timeout_s + (timeout_s or self.tun.stop_timeout_s))
        if isinstance(r, Failed) and r.has_code("agent.stale_stop"):
            rec.state = "STOPPED"  # this incarnation is not running there any more
            return Ok(None)
        if isinstance(r, Failed):
            return r
        assert isinstance(r.value, Stopped)
        rec.state = "STOPPED"
        rec.exit_code = r.value.exit_code
        return Ok(None)

    async def launch(self, plan: ProcessPlan, identity: Identity, topo: Topology, *,
                     start_paused: bool = False) -> Result[None]:
        """Start one process of ``topo`` with ``plan``."""
        d = desired_procs(topo).get(plan.proc)
        if d is None:
            return fail(COMPONENT, "launch", "not part of the topology", subject=str(plan.proc),
                        code="supervisor.not_in_topology")
        plan = msgspec.structs.replace(plan, identity=identity)
        spec = msgspec.structs.replace(self._spec(topo, d, identity, lambda _: plan), start_paused=start_paused)
        return await self.spawn(d, spec)

    async def relaunch(self, plan: ProcessPlan, identity: Identity, topo: Topology, *,
                       start_paused: bool = False) -> Result[None]:
        """Replace a process; ``relaunch_previous`` undoes it."""
        old = self.procs.get(plan.proc)
        if old is not None:
            self._previous[plan.proc] = (old.desired, old.spec)
            r = await self.stop(plan.proc)
            if isinstance(r, Failed):
                return r
        return await self.launch(plan, identity, topo, start_paused=start_paused)

    async def relaunch_previous(self, proc: ProcId) -> Result[None]:
        prev = self._previous.get(proc)
        if prev is None:
            return Ok(None)
        r = await self.stop(proc)
        if isinstance(r, Failed):
            return r
        r = await self.spawn(*prev)
        if isinstance(r, Ok):
            self._previous.pop(proc, None)
        return r

    def committed(self, proc: ProcId, identity: Identity, plan_after: ProcessPlan | None) -> None:
        """A participant changed in place and now serves ``identity``."""
        rec = self.procs.get(proc)
        if rec is None:
            return
        plan = plan_after or rec.spec.plan
        rec.spec = msgspec.structs.replace(rec.spec, identity=identity, start_paused=False,
                                           plan=msgspec.structs.replace(plan, identity=identity))
        rec.snapshots.add(identity)
        self._previous.pop(proc, None)

    def runs_units(self, proc: ProcId, units: Iterable[UnitBrief] | None) -> None:
        """What a process reported it runs once a change committed or rolled back there. ``None``:
        it did not say, so the record stays; an empty list: it runs no unit."""
        rec = self.procs.get(proc)
        if rec is not None and units is not None:
            rec.units = list(units)

    def uses_snapshots(self, proc: ProcId, snapshots: Iterable[Identity]) -> None:
        """What a process reported it still loads code from (after a commit or an abort)."""
        rec, reported = self.procs.get(proc), set(snapshots)
        if rec is not None and reported:
            rec.snapshots = {rec.spec.identity, *reported}

    def forget(self, proc: ProcId) -> None:
        self.procs.pop(proc, None)

    async def restart(self, procs: list[ProcId]) -> Result[None]:
        """Stop + spawn with the same spec. State units in these processes are lost."""
        errors: list[Diagnostic] = []
        for p in procs:
            rec = self.procs[p]
            r = await self.stop(p)
            if isinstance(r, Failed):
                errors += r.diagnostics
                continue
            r = await self.spawn(rec.desired, rec.spec)
            if isinstance(r, Failed):
                errors += r.diagnostics
        return Failed(errors) if errors else Ok(None)

    async def stop_all(self) -> None:
        await asyncio.gather(*(self.stop(p) for p in list(self.procs)))
