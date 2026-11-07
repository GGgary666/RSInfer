"""The control-plane process (doc/specs/control-plane.md §1).

It holds the framework pieces that live once per engine (blob store, event log,
supervisor, change coordinator) and a unit runtime for the control-plane units
(``core.identity``, ``core.manifest``, ``core.deps``, ``core.api``). Those units
reach the framework pieces through host services; the coordinator reaches the
units through their services, looked up afresh for every change, so a reloaded
unit takes effect at the next change.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import msgspec
import zmq.asyncio

from core.change.blobs import BlobStore
from core.change.config import ChangeTunables
from core.change.coordinator import Coordinator, PlanInput
from core.change.eventlog import EventLog
from core.change.participant import with_fresh_modules
from core.change.state import IdentityManifest
from core.change.types import EMPTY, ActionPlan, ChangeResult, ProcActions
from core.common.ids import HostId, Identity, ProcId
from core.common.manifest import discover
from core.common.messages import Envelope
from core.common.result import Diagnostic, Failed, Ok, Result, diag, fail
from core.context.config import ConfigCommit, ConfigEntry
from core.control.protocol import API_REQUESTS
from core.supervisor.protocol import ProcIntrospection
from core.supervisor.supervisor import Supervisor, SupervisorTunables
from core.supervisor.topology import Topology
from unit.runtime.host import introspection
from unit.runtime.runtime import UnitRuntime
from unit.runtime.types import InstanceState, ProcessPlan, ReloadPlan, UnitSpec

COMPONENT = "control"
CONTROL_ROLE = "control"

# Services the host offers the control-plane units.
HOST_CHANGE = "ctrl.change@1"
HOST_SUPERVISOR = "ctrl.supervisor@1"
HOST_BLOBS = "ctrl.blobs@1"
HOST_EVENTS = "ctrl.events@1"
HOST_SELF = "ctrl.self@1"

# Services of the control-plane units the host uses.
IDENTITY = "ctrl.identity@1"
MANIFEST = "ctrl.manifest@1"
DEPS = "ctrl.deps@1"
API = "ctrl.api@1"
CONTROL_SERVICES = (IDENTITY, MANIFEST, DEPS, API)


def files_digest(files: dict[str, str]) -> str:
    """Names a file list (``path -> content hash``); the same list always gets the same name."""
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]


def ingest(blobs: BlobStore, source: Path) -> dict[str, str]:
    """A workspace's ``engine/`` tree as ``{"engine/...": hash}``, its contents in the blob store."""
    return blobs.ingest_tree(source / "engine", prefix="engine/")


def materialize(blobs: BlobStore, files: dict[str, str], root: Path) -> Path:
    """A local copy of ``files`` under ``root``, named by their digest; returns its ``engine/``."""
    target = root / files_digest(files)
    if not target.is_dir():
        root.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".tmp-", dir=root))
        try:
            for rel, h in sorted(files.items()):
                if rel.startswith("/") or ".." in Path(rel).parts:
                    raise ValueError(f"bad snapshot path {rel!r}")
                data = blobs.get(h)
                if data is None:
                    raise FileNotFoundError(f"blob {h} of {rel} is not in the blob store")
                dest = tmp / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            os.rename(tmp, target)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    return target / "engine"


class ControlPlane:
    def __init__(
        self, zctx: zmq.asyncio.Context, *, host: HostId, bind: str, state_dir: Path,
        sup_tun: SupervisorTunables | None = None, change_tun: ChangeTunables | None = None,
        proc_tunables: dict[str, float | int] | None = None,
    ) -> None:
        self.proc = ProcId(host, CONTROL_ROLE, 0)
        self.state_dir = state_dir
        self.blobs = BlobStore(state_dir / "blobs")
        self.log = EventLog(state_dir / "events.sqlite", self.blobs)
        self.sup = Supervisor(zctx, host=host, bind_address=bind, blobs=self.blobs, tun=sup_tun,
                              proc_tunables=proc_tunables)
        self.runtime = UnitRuntime(self.proc)
        self.local = ControlParticipant(self)
        self.coord = Coordinator(self.sup, self.log, planner=_Planner(self), checker=_Checker(self),
                                 identity_fn=self._identity, tun=change_tun, local=self.local)
        for name, obj in [(HOST_CHANGE, self.coord), (HOST_SUPERVISOR, self.sup), (HOST_BLOBS, self.blobs),
                          (HOST_EVENTS, self.log), (HOST_SELF, self)]:
            self.runtime.provide_host(name, obj)
        for type_ in API_REQUESTS:
            self.sup.handlers[type_] = self._api

    @property
    def address(self) -> str:
        """The control endpoint: clients, node agents and processes all use it."""
        return self.sup.address

    async def start(self) -> None:
        await self.sup.start()
        await self.coord.start()

    def close(self) -> None:
        self.coord.close()
        self.runtime.unload_all()
        self.sup.close()

    # -------------------------------------------------------- for the units

    def snapshot(self, files: dict[str, str]) -> Path:
        """A local copy of a file list to analyse (``core.manifest``, ``core.deps``)."""
        return materialize(self.blobs, files, self.state_dir / "snapshots")

    def introspect_self(self) -> ProcIntrospection:
        return introspection(self.runtime, self.proc, self.coord.identity)

    # ------------------------------------------------------------- startup

    async def boot(self, files: dict[str, str], topology: Topology, config: dict[str, Any] | None = None
                   ) -> Result[ChangeResult]:
        """Load the control-plane units from ``files``, then start the engine: the startup change."""
        config = dict(config or {})
        try:
            root = self.snapshot(files)
        except (OSError, ValueError) as e:
            return fail(COMPONENT, "boot", f"cannot materialize the source: {e}", code="control.bad_source")
        manifests, problems = discover(root)
        if problems:
            return Failed(problems)
        specs = [UnitSpec(m) for m in manifests if CONTROL_ROLE in m.unit.process]
        entries = {key: ConfigEntry(s.id, decl, config.get(key, decl.default))
                   for s in specs for key, decl in s.manifest.config.items()}
        loaded = self.runtime.load_process_plan(ProcessPlan(self.proc, EMPTY, specs, entries, code_root=str(root)))
        problems = [*(loaded.diagnostics if isinstance(loaded, Failed) else []), *self.control_problems("boot")]
        if problems:
            return Failed(problems)
        return Ok(await self.coord.startup(files, topology, config=config))

    def control_problems(self, phase: str) -> list[Diagnostic]:
        """Why the control-plane units cannot serve: one is not active, or a service the host
        relies on is missing. Checked at boot and after every change to them, before it commits."""
        out = [diag(COMPONENT, phase, f"{u.unit} is {u.state}: {u.error}", subject=u.unit, code="control.unit_failed")
               for u in self.runtime.snapshot().units if u.state is not InstanceState.ACTIVE]
        missing = [s for s in CONTROL_SERVICES if self.runtime.lookup(s) is None]
        if missing:
            out.append(diag(COMPONENT, phase, f"no unit provides {missing}", code="control.unit_missing",
                            hint="the source needs core.identity, core.manifest, core.deps and core.api"))
        return out

    # ------------------------------------------------------------ plumbing

    def service(self, name: str) -> Any:
        """A control-plane unit's service, looked up now (a reloaded unit is used from the next call)."""
        svc = self.runtime.lookup(name)
        if svc is None:
            raise _Missing(name)
        return svc

    def _identity(self, manifest: IdentityManifest) -> Identity:
        return self.service(IDENTITY).identity(manifest, self.sup.reference_platform)

    async def _api(self, env: Envelope, msg: msgspec.Struct) -> Result[msgspec.Struct]:
        try:
            api = self.service(API)
        except _Missing as e:
            return e.failed()
        return await api.handle(msg)


class _Missing(Exception):
    def __init__(self, service: str) -> None:
        super().__init__(service)
        self.service = service

    def failed(self) -> Failed:
        return fail(COMPONENT, "lookup", f"no control-plane unit provides {self.service}",
                    code="control.unit_missing", hint="a control-plane unit failed to load; see introspect")


class _Checker:
    def __init__(self, cp: ControlPlane) -> None:
        self.cp = cp

    async def check(self, inp: PlanInput) -> Result[None]:
        try:
            svc = self.cp.service(MANIFEST)
        except _Missing as e:
            return e.failed()
        return await svc.check(inp)


class _Planner:
    def __init__(self, cp: ControlPlane) -> None:
        self.cp = cp

    async def plan(self, inp: PlanInput) -> Result[ActionPlan]:
        try:
            svc = self.cp.service(DEPS)
        except _Missing as e:
            return e.failed()
        return await svc.plan(inp)


class _Local:
    def __init__(self, change_id: int, actions: ProcActions, root: str) -> None:
        self.change_id = change_id
        self.actions = actions
        self.root = root
        self.reverse: ReloadPlan | None = None
        self.old_config: dict[str, Any] = {}
        self.modules_before: set[str] = set()
        self.undo_errors: list[Diagnostic] = []
        """Errors of an undo that already ran; abort reports them, so the engine needs recovery."""


class ControlParticipant:
    """Runs ``ActionPlan.control`` on the control-plane units (doc/specs/control-plane.md §1).
    The coordinator calls it between its own steps, so the control runtime is at a safe point."""

    def __init__(self, cp: ControlPlane) -> None:
        self.cp = cp
        self.rt = cp.runtime
        self.change: _Local | None = None

    async def prepare(self, change_id: int, identity: Identity, actions: ProcActions,
                      files: dict[str, str]) -> Result[None]:
        try:
            root = self.cp.snapshot(files)
        except (OSError, ValueError) as e:
            return fail(COMPONENT, "prepare", f"cannot materialize the candidate: {e}", code="control.bad_source")
        if actions.config:
            r = self.rt.config.stage(actions.config)
            if isinstance(r, Failed):
                return r
            self.rt.config.discard_staged()  # validated; taken inside the reload at execute
        self.change = _Local(change_id, actions, str(root))
        return Ok(None)

    def execute(self, change_id: int) -> Result[None]:
        c = self.change
        if c is None or c.change_id != change_id:
            return fail(COMPONENT, "execute", f"change {change_id} was not prepared here", code="control.not_prepared")
        a = c.actions
        plan = a.reload or ReloadPlan()
        if plan.new_code_root is None and c.root != self.rt.code_root:
            plan = msgspec.structs.replace(plan, new_code_root=c.root)
        c.root = plan.new_code_root or c.root
        if a.stage:
            plan = msgspec.structs.replace(plan, units_to_add=[*plan.units_to_add, *a.stage])
        reverse = self.rt.reverse_plan(plan)
        c.old_config = {k: self.rt.config.entries()[k].value for k in a.config if k in self.rt.config.entries()}
        c.modules_before = set(sys.modules)
        r = self.rt.reload(plan, values=dict(a.config))
        if isinstance(r, Failed):
            # the runtime put the old units back with the old values, unless its rollback failed
            c.old_config = {}
            if any(d.phase == "rollback" and d.severity == "error" for d in r.diagnostics):
                c.undo_errors = [diag(COMPONENT, "abort", "the reload failed and the old control-plane units "
                                      "did not all come back", code="control.rollback_failed")]
            return r
        c.reverse = reverse
        changed = {k: v for k, v in a.config.items() if c.old_config.get(k) != v}
        hooks: ConfigCommit = self.rt.config.run_hooks(changed, skip_units=set(r.value.loaded))
        problems = [*hooks.hook_failures, *self.cp.control_problems("execute")]
        if problems:
            c.undo_errors = self._undo(c)  # abort reports them
            return Failed(problems)
        return Ok(None)

    def commit(self, change_id: int) -> None:
        if self.change is not None and self.change.change_id == change_id:
            self.change = None

    def abort(self, change_id: int) -> list[Diagnostic]:
        """Undo the change here. Errors mean the old control-plane units are not all back: the
        coordinator then requires recovery, as for a participant that did not confirm its rollback."""
        c = self.change
        self.change = None
        if c is None or c.change_id != change_id:
            return []
        return [*c.undo_errors, *self._undo(c)]

    def _undo(self, c: _Local) -> list[Diagnostic]:
        out: list[Diagnostic] = []
        if c.old_config:
            self.rt.config.stage(c.old_config)
            out += self.rt.config.commit().hook_failures
            c.old_config = {}
        if c.reverse is not None:
            r = self.rt.reload(with_fresh_modules(self.rt, c.reverse, c.root, c.modules_before))
            c.reverse = None
            if isinstance(r, Failed):
                out += [d for d in r.diagnostics if d.severity == "error"]
                out.append(diag(COMPONENT, "abort", "the old control-plane units could not be reloaded",
                                code="control.rollback_failed"))
            else:
                out += self.cp.control_problems("abort")
        return out
