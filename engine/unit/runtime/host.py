"""The process host: the entry point of every engine process.

``python -P -s -m unit.runtime.host <spec.msgpack>``, started by a node agent
with the snapshot's ``engine/`` directory as the only engine code on sys.path.

The host owns the outermost loop. Each turn is a safe point: it serves control
requests, runs safe-point hooks (the change participant), then calls ``step()``
of whichever unit provides ``proc.loop@1`` (the worker loop, the scheduler).
Because the loop unit is looked up again every turn, it can itself be reloaded:
reloads happen between two ``step()`` calls. ``step()`` must return promptly
when there is nothing to do, returning the epoch it finished or ``None``.
"""

from __future__ import annotations

import signal
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import Any, Protocol

import msgspec
import zmq

from core.change.config import ChangeTunables
from core.change.participant import Participant
from core.change.types import ChangeSet
from core.common.ids import Endpoint, Epoch, Identity, ProcId
from core.common.messages import Envelope
from core.common.result import Diagnostic, Failed, Ok, Result, diag, fail
from core.common.tunables import load_tunables
from core.control.protocol import ChangeQueued, SubmitChange
from core.ipc.config import IpcTunables
from core.ipc.rpc import Deferred, SyncRpcClient, SyncRpcServer
from core.ipc.stream import StreamAddress
from core.supervisor.protocol import (
    CheckIn,
    Introspect,
    ProcIntrospection,
    ProcSpec,
    ProcStatusReport,
    ReadRecord,
    Status,
    UnitBrief,
    control_registry,
)
from platforms.base import Platform
from unit.runtime.runtime import UnitRuntime

LOOP_SERVICE = "proc.loop@1"

type Handler = Callable[[Envelope, Any], Result[msgspec.Struct] | Deferred]


class Loop(Protocol):
    def step(self) -> Epoch | None: ...


class HostTunables(msgspec.Struct, frozen=True):
    idle_poll_s: float = 0.02
    """How long the host waits for control requests when there is no loop to run."""
    checkin_timeout_s: float = 10.0


def make_platform(spec: ProcSpec) -> Platform:
    if spec.platform == "fake":
        from platforms.fake import FakePlatform

        return FakePlatform(devices=spec.platform_options.get("devices", max(1, len(spec.gpus))))
    raise ValueError(f"unknown platform {spec.platform!r}")


CONTROL_LINK = "proc.control@1"
"""Host service: what a unit may ask of the control plane (submit an internal change)."""


class ControlLink:
    """``proc.control@1``. A unit submits internal change-sets (EPLB) through it; they are
    queued at the control plane, not waited for (doc/specs/change.md §6)."""

    def __init__(self, host: ProcessHost) -> None:
        self._host = host

    @property
    def identity(self) -> Identity:
        """The identity this process serves: what an internal change is ``computed_under``."""
        return self._host.identity

    def submit(self, cs: ChangeSet) -> Result[int]:
        h = self._host
        r = h.client.call(h.spec.control, SubmitChange(cs), timeout=h.ipc.rpc_timeout_s)
        if isinstance(r, Failed):
            return r
        assert isinstance(r.value, ChangeQueued)
        return Ok(r.value.queued)


def introspection(runtime: UnitRuntime, proc: ProcId, identity: Identity) -> ProcIntrospection:
    """What ``core.deps`` needs to know about one process: its units, who read which config key
    when, and the derived artifacts registered there (doc/specs/control-plane.md §4)."""
    snap = runtime.snapshot()
    live = {u.instance for u in snap.units if u.instance is not None}
    config = runtime.config
    reads: list[ReadRecord] = []
    recorded = config.reads()
    for phase, table in (("runtime", recorded.runtime), ("startup", recorded.startup)):
        for r in table:
            if r.reader in live:
                reads.append(ReadRecord(r.key, r.reader.unit, phase, r.reader in config.hooked(r.key)))
    for artifact, plan_reads in recorded.plan_build.items():
        for r in plan_reads:
            if r.reader in live:
                reads.append(ReadRecord(r.key, r.reader.unit, "plan_build", artifact=artifact))
    return ProcIntrospection(
        proc=proc, identity=identity,
        units=[UnitBrief(u.unit, u.state, u.error, u.generation) for u in snap.units],
        reads=sorted(set(reads), key=lambda r: (r.key, r.unit, r.phase, r.artifact)),
        artifacts=runtime.derived(),
        config={k: e.value for k, e in config.entries().items()},
    )


def loaded_engine_files() -> list[str]:
    """Files of loaded modules outside the interpreter's own prefixes."""
    prefixes = tuple({sys.prefix, sys.base_prefix, sys.exec_prefix})
    out: list[str] = []
    for mod in list(sys.modules.values()):
        file = getattr(mod, "__file__", None)
        if file and not file.startswith(prefixes):
            out.append(str(Path(file).resolve()))
    return sorted(set(out))


class ProcessHost:
    def __init__(self, spec: ProcSpec) -> None:
        self.spec = spec
        self.tun = load_tunables(HostTunables, "host", dict(spec.tunables))
        self.ipc = load_tunables(IpcTunables, "ipc", dict(spec.tunables))
        self.zctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
        self.registry = control_registry()
        self.me = Endpoint(spec.proc, "rpc")
        self.platform = make_platform(spec)
        self.runtime = UnitRuntime(spec.proc, platform=self.platform)
        self.rpc = SyncRpcServer(self.zctx, self.registry, self.me, spec.bind_address)
        self.client = SyncRpcClient(self.zctx, self.registry, self.me)
        self.handlers: dict[type[msgspec.Struct], Handler] = {Status: self._status, Introspect: self._introspect}
        self.safe_point_hooks: list[Callable[[], None]] = []
        self.endpoints: dict[str, StreamAddress] = {"rpc": StreamAddress(spec.proc.host, tcp=self.rpc.address)}
        self.identity = spec.identity
        """The identity this process serves; a committed change moves it."""
        self.epoch: Epoch | None = None
        self.paused = spec.start_paused
        """Set by the change participant: no ``step()`` while a change executes."""
        self.stopping = False
        self.diagnostics: list[Diagnostic] = []
        self.participant = Participant(self, load_tunables(ChangeTunables, "change", dict(spec.tunables)))
        self.runtime.provide_host(CONTROL_LINK, ControlLink(self))

    # ------------------------------------------------------------- startup

    def start(self) -> Result[msgspec.Struct]:
        """Load the units, then check in; unit failures are reported, not fatal."""
        plan = self.spec.plan
        if plan.code_root is None:
            # The code root is wherever this host's own code came from: the snapshot.
            plan = msgspec.structs.replace(plan, code_root=str(Path(__file__).resolve().parents[2]))
        loaded = self.runtime.load_process_plan(plan)
        self.diagnostics += loaded.diagnostics
        return self.check_in()

    def check_in(self) -> Result[msgspec.Struct]:
        msg = CheckIn(
            proc=self.spec.proc,
            incarnation=self.spec.incarnation,
            identity=self.spec.identity,
            endpoints=dict(self.endpoints),
            units=self.unit_briefs(),
            diagnostics=list(self.diagnostics),
            sys_path=list(sys.path),
            module_files=loaded_engine_files(),
        )
        return self.client.call(self.spec.control, msg, timeout=self.tun.checkin_timeout_s)

    def unit_briefs(self) -> list[UnitBrief]:
        return [UnitBrief(u.unit, u.state, u.error, u.generation) for u in self.runtime.snapshot().units]

    # ---------------------------------------------------------------- loop

    def _dispatch(self, env: Envelope, msg: msgspec.Struct) -> Result[msgspec.Struct] | Deferred:
        handler = self.handlers.get(type(msg))
        if handler is None:
            return fail("host", "dispatch", f"{env.schema} is not handled by {self.spec.proc.role}",
                        subject=str(self.spec.proc), code="host.unhandled")
        return handler(env, msg)

    def _status(self, env: Envelope, msg: Status) -> Result[msgspec.Struct]:
        return Ok(ProcStatusReport(self.spec.proc, self.identity, self.epoch, self.unit_briefs(),
                                   self.paused, list(self.participant.diagnostics)))

    def _introspect(self, env: Envelope, msg: Introspect) -> Result[msgspec.Struct]:
        return Ok(introspection(self.runtime, self.spec.proc, self.identity))

    def safe_point(self, wait_s: float = 0.0) -> None:
        self.rpc.poll(self._dispatch, wait_s)
        for hook in list(self.safe_point_hooks):
            hook()

    def run(self) -> int:
        while not self.stopping:
            self.safe_point()
            if self.stopping:
                break
            loop: Loop | None = None if self.paused else self.runtime.lookup(LOOP_SERVICE)
            if loop is None:
                self.safe_point(self.tun.idle_poll_s)
                continue
            try:
                epoch = loop.step()
            except Exception as e:  # noqa: BLE001 - a unit bug must not kill the host
                self.diagnostics.append(diag("host", "step", f"{type(e).__name__}: {e}",
                                             subject=str(self.spec.proc), code="host.step_failed"))
                traceback.print_exc()
                time.sleep(self.tun.idle_poll_s)
                continue
            if epoch is not None:
                self.epoch = epoch
        return self.shutdown()

    def shutdown(self) -> int:
        self.runtime.unload_all()
        self.rpc.close()
        self.client.close()
        self.zctx.term()
        return 0

    def request_stop(self, signum: int = 0, frame: FrameType | None = None) -> None:
        del signum, frame
        self.stopping = True


def main(argv: list[str]) -> int:
    spec = msgspec.msgpack.decode(Path(argv[1]).read_bytes(), type=ProcSpec)
    host = ProcessHost(spec)
    signal.signal(signal.SIGTERM, host.request_stop)
    r = host.start()
    if isinstance(r, Failed):
        print(f"check-in failed: {r}", file=sys.stderr)
        host.shutdown()
        return 3
    return host.run()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
