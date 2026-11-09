"""The process host: the entry point of every engine process.

``python -P -s -m unit.runtime.host <spec.msgpack>``, started by a node agent
with the snapshot's ``engine/`` directory as the only engine code on sys.path.

The host owns the outermost loop. Each turn is a safe point: it serves control
requests, runs safe-point hooks (the change participant), then calls ``step()``
of whichever unit provides ``proc.loop@1`` (the worker loop, the scheduler).
Because the loop unit is looked up again every turn, it can itself be reloaded:
reloads happen between two ``step()`` calls. ``step()`` must return promptly
when there is nothing to do, returning the epoch it finished or ``None``.

A loop may leave work running when ``step()`` returns: GPU work launched
asynchronously, or the next step started before the last one's results are
taken (overlap scheduling). Before the process stops stepping (a pause, a
shutdown), the host calls the loop's ``quiesce()`` so that nothing an Execute
unloads is still in use.
"""

from __future__ import annotations

import signal
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import Any, Protocol, cast

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
from core.ipc.channels import Channels
from core.ipc.config import IpcTunables
from core.ipc.rpc import Deferred, SyncRpcClient, SyncRpcServer
from core.ipc.stream import StreamAddress
from core.supervisor.protocol import (
    AllocShm,
    CheckIn,
    Found,
    Introspect,
    Lookup,
    ProcIntrospection,
    ProcSpec,
    ProcStatusReport,
    PublishEndpoint,
    ReadRecord,
    ShmAllocated,
    Status,
    UnitBrief,
    control_registry,
)
from platforms.base import Platform
from unit.runtime.runtime import UnitRuntime

LOOP_SERVICE = "proc.loop@1"
EXIT_QUIESCE_FAILED = 4
"""Exit code when the loop could not finish its steps: the units were left loaded."""

type Handler = Callable[[Envelope, Any], Result[msgspec.Struct] | Deferred]


class Loop(Protocol):
    def step(self) -> Epoch | None: ...


class Quiescing(Protocol):
    def quiesce(self) -> Epoch | None:
        """Finish every step already started: its device work and its results. Return the
        last epoch finished, or ``None`` if none was. Optional for a loop that leaves nothing
        running when ``step()`` returns. Called before the process stops stepping, possibly
        more than once: while a pause waits for its epoch, ``step()`` goes on in between."""
        ...


class HostTunables(msgspec.Struct, frozen=True):
    idle_poll_s: float = 0.02
    """How long the host waits for control requests when there is no loop to run."""
    checkin_timeout_s: float = 10.0


def make_platform(spec: ProcSpec) -> Platform:
    if spec.platform == "fake":
        from platforms.fake import FakePlatform

        return FakePlatform(devices=spec.platform_options.get("devices", max(1, len(spec.gpus))))
    if spec.platform == "torch_eager":
        from platforms.torch_eager import TorchEagerPlatform

        return TorchEagerPlatform(cuda=bool(spec.platform_options.get("cuda", 0)))
    raise ValueError(f"unknown platform {spec.platform!r}")


CONTROL_LINK = "proc.control@1"
"""Host service: what a unit may ask of the control plane (submit an internal change)."""
STREAMS = "proc.streams@1"
"""Host service: the data channels the host keeps for its units (``core.ipc.channels``)."""


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
        """Set by the change participant: no ``step()`` while a change executes. Normally set
        after a successful ``quiesce()``; an undo that fails sets it even when quiesce failed,
        so it says nothing about work in flight (``settled`` does)."""
        self.settled = True
        """No step has been started since the last successful ``quiesce()`` (or ever)."""
        self.stopping = False
        self.diagnostics: list[Diagnostic] = []
        self.participant = Participant(self, load_tunables(ChangeTunables, "change", dict(spec.tunables)))
        self.runtime.provide_host(CONTROL_LINK, ControlLink(self))
        self.checked_in = False
        self.channels = Channels(self.zctx, local=spec.proc.host, bind=spec.bind_address, tun=self.ipc,
                                 allocate=self._allocate_ring, lookup=self._lookup, publish=self._publish)
        self.runtime.provide_host(STREAMS, self.channels)

    # ------------------------------------------------------------- startup

    def start(self) -> Result[msgspec.Struct]:
        """Load the units, then check in; unit failures are reported, not fatal."""
        plan = self.spec.plan
        if plan.code_root is None:
            # The code root is wherever this host's own code came from: the snapshot.
            plan = msgspec.structs.replace(plan, code_root=str(Path(__file__).resolve().parents[2]))
        loaded = self.runtime.load_process_plan(plan)
        self.diagnostics += loaded.diagnostics
        checked = self.check_in()
        self.checked_in = isinstance(checked, Ok)
        return checked

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

    def _allocate_ring(self, name: str) -> str:
        """A ring name from this host's node agent, which removes the ring when the process is gone."""
        r = self.client.call(self.spec.control, AllocShm(self.spec.proc, name), timeout=self.ipc.rpc_timeout_s)
        allocated = r.unwrap()  # raises in the unit that asked: it cannot have the reader
        assert isinstance(allocated, ShmAllocated)
        return allocated.name

    def _lookup(self, proc: ProcId, channel: str) -> StreamAddress | None:
        r = self.client.call(self.spec.control, Lookup(proc, channel), timeout=self.ipc.rpc_timeout_s)
        if isinstance(r, Failed) or not isinstance(r.value, Found):
            return None  # the writer tries again later
        return r.value.address

    def _publish(self, channel: str, address: StreamAddress) -> None:
        """Readers opened before the check-in go with it; later ones are sent on their own."""
        self.endpoints[channel] = address
        if self.checked_in:
            msg = PublishEndpoint(self.spec.proc, self.spec.incarnation, channel, address)
            self.client.call(self.spec.control, msg, timeout=self.ipc.rpc_timeout_s).unwrap()

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

    def has_loop(self) -> bool:
        """Whether a loop unit steps in this process (it numbers epochs)."""
        return self.runtime.lookup(LOOP_SERVICE) is not None

    def quiesce(self) -> Result[None]:
        """Let the loop unit finish the steps it has started. ``epoch`` moves to the last one
        finished. Called before the process stops stepping; while a pause waits for its epoch it
        is called at every safe point, and stepping goes on in between. Nothing to do once
        ``settled``; a failure leaves it unsettled, so the next call tries again."""
        if self.settled:
            return Ok(None)
        loop = self.runtime.lookup(LOOP_SERVICE)
        quiesce = getattr(loop, "quiesce", None) if loop is not None else None
        if quiesce is not None:
            try:
                epoch = cast(Quiescing, loop).quiesce()
            except Exception as e:  # noqa: BLE001 - a unit bug must not kill the host
                traceback.print_exc()
                return fail("host", "quiesce", f"{type(e).__name__}: {e}", subject=str(self.spec.proc),
                            code="host.quiesce_failed")
            if epoch is not None:
                self.epoch = epoch
        self.settled = True
        return Ok(None)

    def run(self) -> int:
        while not self.stopping:
            self.turn()
        return self.shutdown()

    def turn(self) -> None:
        """One turn of the main loop: a safe point, then one step unless paused or stopping."""
        self.safe_point()
        if self.stopping:
            return
        loop: Loop | None = None if self.paused else self.runtime.lookup(LOOP_SERVICE)
        if loop is None:
            self.safe_point(self.tun.idle_poll_s)
            return
        self.settled = False
        try:
            epoch = loop.step()
        except Exception as e:  # noqa: BLE001 - a unit bug must not kill the host
            self.diagnostics.append(diag("host", "step", f"{type(e).__name__}: {e}",
                                         subject=str(self.spec.proc), code="host.step_failed"))
            traceback.print_exc()
            time.sleep(self.tun.idle_poll_s)
            return
        if epoch is not None:
            self.epoch = epoch

    def shutdown(self) -> int:
        quiesced = self.quiesce()
        if isinstance(quiesced, Failed):
            # Device work may still be running: unit teardown could free memory it uses. Leave
            # the units loaded; exiting the process releases everything once the device is done.
            self.diagnostics += quiesced.diagnostics
            print(f"quiesce before shutdown failed, units not unloaded: {quiesced}", file=sys.stderr)
        else:
            self.runtime.unload_all()
        self.channels.close()
        self.rpc.close()
        self.client.close()
        self.zctx.term()
        return EXIT_QUIESCE_FAILED if isinstance(quiesced, Failed) else 0

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
