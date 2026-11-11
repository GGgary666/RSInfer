"""A real control plane in the test's event loop, node agents as child processes."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import sys
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import zmq.asyncio

from core.change.blobs import BlobStore
from core.common.ids import HostId, Identity, ProcId, UnitId
from core.context.config import ConfigEntry
from core.supervisor import (
    DesiredProc,
    HostSpec,
    Loss,
    Mesh,
    RankPlacement,
    RoleSpec,
    Supervisor,
    SupervisorTunables,
    Topology,
)
from tests.framework.support import spec, write_tree
from unit.runtime import ProcessPlan, UnitSpec

REPO = Path(__file__).resolve().parents[3]
ENGINE = REPO / "engine"
ADDRESS = "127.0.0.1"  # test data: every fake host lives on this machine

TOY = {
    "toyc/builder.py": '''
"""A build step (``[unit] build``): after ``TOYC_BUILD_S`` seconds, appends a line to the file
``TOYC_BUILD_LOG`` names (the step's process does not know the toy log); exits 3 when
``TOYC_BUILD_FAIL`` is set."""
import os
import sys
import time


def make():
    time.sleep(float(os.environ.get("TOYC_BUILD_S", "0")))
    with open(os.environ["TOYC_BUILD_LOG"], "a") as f:
        f.write(f"built {sys.path[0] if sys.path else ''}\\n")
    if os.environ.get("TOYC_BUILD_FAIL"):
        sys.exit(3)
''',
    "toyc/__init__.py": "",
    "toyc/units.py": '''
import time


class Idle:
    def __init__(self, ctx, config):
        pass


class Loop:
    """Never blocks for long. Every process numbers its steps the same way, as workers follow
    the scheduler's step numbers (change.md 4.2): the fake hosts share this machine's clock,
    so a restarted worker joins the current number instead of counting from zero."""
    STEP_S = 0.005

    def __init__(self, ctx, config):
        ctx.provide("proc.loop@1", self)

    def step(self):
        time.sleep(self.STEP_S)
        return int(time.time() / self.STEP_S)


class Stubborn:
    """A loop stuck inside a step: it never reaches a safe point again. It logs once it is stuck,
    so a test can wait until the process no longer answers (before that, it still does)."""
    def __init__(self, ctx, config):
        self.ctx, self.config = ctx, config
        ctx.provide("proc.loop@1", self)

    def step(self):
        from toyc.logutil import log
        log(self.config, self.ctx, "stubborn: stuck in step")
        time.sleep(3600)
''',
    "toyc/logutil.py": '''
import os


def log(config, ctx, text):
    """One file per process, so tests can see what each participant did."""
    d = config["toyc.log_dir"]
    if not d:
        return
    p = ctx.instance.proc
    with open(os.path.join(d, f"{p.host}-{p.role}-{p.index}.log"), "a") as f:
        f.write(text + "\\n")
''',
    "toyc/channels.py": '''
import time
import uuid

from core.common.ids import HostId, ProcId
from toyc.logutil import log


class Sender:
    """Writes numbered records to the sink's reader "in" through the host's data channels; a
    record the channel could not take is sent again, so the numbers have no gaps on its side."""
    EVERY_S = 0.02

    def __init__(self, ctx, config):
        self.ctx, self.config = ctx, config
        sink = config["toyc.sink"]
        self.out = ctx.use("proc.streams@1").writer(ProcId(HostId(sink["host"]), sink["role"], sink["index"]), "in")
        self.token, self.n, self.gen, self.last = uuid.uuid4().hex[:6], 0, 0, 0.0
        log(config, ctx, f"sender up {self.token}")
        ctx.provide("proc.loop@1", self)

    def step(self):
        time.sleep(0.002)
        if time.monotonic() - self.last < self.EVERY_S:
            return None
        self.last = time.monotonic()
        if self.out.write(f"{self.token}:{self.n}".encode(), timeout=1.0):
            self.n += 1
            if self.out.generation != self.gen:
                self.gen = self.out.generation
                log(self.config, self.ctx, f"sender gen {self.gen}")
        return None


class Sink:
    """Logs what reaches its reader "in". ``toyc.late_reader``: open the reader at the first step,
    after the process checked in, so the host publishes it on its own."""

    def __init__(self, ctx, config):
        self.ctx, self.config = ctx, config
        self.inp = None if config["toyc.late_reader"] else self._open()
        log(config, ctx, f"sink up {uuid.uuid4().hex[:6]}")
        ctx.provide("proc.loop@1", self)

    def _open(self):
        return self.ctx.use("proc.streams@1").reader("in", HostId(self.config["toyc.source_host"]))

    def step(self):
        if self.inp is None:
            self.inp = self._open()
        data = self.inp.read(timeout=0.005)
        if data is not None:
            log(self.config, self.ctx, f"got {data.decode()} gen {self.inp.generation}")
        return None
''',
    "toyc/tunable.py": '''
import time

from toyc.logutil import log


class Op:
    """Takes part in changes as "toy.op"; args can make a phase fail.

    ``slices``: prepare in that many slices (``slice_s`` each, ``fail_slice`` raises);
    ``prepare_s``: a one-shot prepare that takes that long; ``needs_prepare``: execute
    fails unless this very instance was prepared.
    """
    def __init__(self, ctx, config):
        self.ctx, self.config = ctx, config
        self.left = 0
        self.never = False
        self.prepared = False

    def _phase(self, name, args=None):
        log(self.config, self.ctx, f"op {name}")
        if (args or {}).get("fail_at") == name:
            raise RuntimeError(f"injected {name} failure")

    def prepare(self, args):
        self.prepared = True
        self._phase("prepare", args)
        if "slices" in args:
            return self._slices(args)
        time.sleep(args.get("prepare_s", 0))

    def _slices(self, args):
        for i in range(args["slices"]):
            log(self.config, self.ctx, f"op slice {i}")
            if args.get("fail_slice") == i:
                raise RuntimeError(f"injected failure in slice {i}")
            time.sleep(args.get("slice_s", 0))
            yield

    def execute(self, args):
        self._phase("execute", args)
        if args.get("needs_prepare") and not self.prepared:
            raise RuntimeError("executed on an instance that was never prepared")

    def commit(self): self._phase("commit")
    def abort(self): self._phase("abort")

    def start_drain(self, args):
        self.left = args.get("ticks", 0)
        self.never = args.get("never", False)
        self.deadline_fails = args.get("deadline_fails", False)
        log(self.config, self.ctx, "drain start")
        if args.get("refuse"):
            raise RuntimeError("the toy holds requests")

    def drained(self):
        self.left -= 1
        return not self.never and self.left <= 0

    def drain_deadline(self):
        log(self.config, self.ctx, "drain timeout")
        if self.deadline_fails:
            raise RuntimeError("the toy did not empty in time")
        return {"aborted": 3}  # this toy's policy: the requests left end at the deadline


class Tunable:
    def __init__(self, ctx, config):
        self.config = config
        time.sleep(config["toyc.init_delay_s"])  # e.g. loading weights
        ctx.on_action("toy.op", Op(ctx, config))
        ctx.on_config("toyc.chunk", lambda v: log(config, ctx, f"chunk {v}"))
        log(config, ctx, f"tunable loaded gen{ctx.instance.generation} flavor {config['toyc.flavor']}")
''',
    "toyc/tensor.py": '''
from toyc.logutil import log


class Tensor:
    """Holds one buffer of its process's platform and logs the tensor behind it."""

    def __init__(self, ctx, config):
        platform = ctx.platform
        buf = platform.alloc(0, 0, shape=(2, 3), dtype="f32")
        ctx.effect(lambda: lambda: platform.free(buf))
        t = platform.tensor(buf)
        t.fill_(1.5)
        platform.synchronize(0)
        log(config, ctx, f"tensor {platform.info().name} {type(t).__module__}.{type(t).__name__} "
                         f"{tuple(t.shape)} {t.device.type} {float(t.sum())}")
''',
    "toyc/versioned.py": '''
from toyc.logutil import log

VERSION = 1


class Versioned:
    def __init__(self, ctx, config):
        log(config, ctx, f"versioned v{VERSION}")
''',
}

VERSIONED_V2 = "from toyc.logutil import log\n\nVERSION = 2\n\n\nclass Versioned:\n" \
    "    def __init__(self, ctx, config):\n        log(config, ctx, f\"versioned v{VERSION}\")\n"

ALL_ROLES = ["sched", "worker", "frontend"]

SPECS: dict[str, UnitSpec] = {
    "toyc.idle": spec("toyc.idle", "toyc.units:Idle", process=ALL_ROLES,
                      config={"toyc.mesh": {"type": "any", "default": {}, "binding": "startup"}}),
    "toyc.tunable": spec("toyc.tunable", "toyc.tunable:Tunable", process=ALL_ROLES,
                         capabilities="accept_config = true", config={
                             "toyc.log_dir": {"type": "str", "default": "", "binding": "startup"},
                             "toyc.init_delay_s": {"type": "float", "default": 0.0, "binding": "startup"},
                             "toyc.flavor": {"type": "str", "default": "plain", "binding": "startup"},
                             "toyc.chunk": {"type": "int", "default": 8192, "binding": "live"}}),
    "toyc.versioned": spec("toyc.versioned", "toyc.versioned:Versioned", process=ALL_ROLES,
                           reads=["toyc.log_dir"]),
    "toyc.loop": spec("toyc.loop", "toyc.units:Loop", provides=["proc.loop@1"], process=["sched", "worker"]),
    "toyc.sender": spec("toyc.sender", "toyc.channels:Sender", provides=["proc.loop@1"],
                        requires=["proc.streams@1"], process=["sched"], reads=["toyc.log_dir"],
                        config={"toyc.sink": {"type": "any", "default": {}, "binding": "startup"}}),
    "toyc.sink": spec("toyc.sink", "toyc.channels:Sink", provides=["proc.loop@1"], requires=["proc.streams@1"],
                      process=["worker"], reads=["toyc.log_dir"],
                      config={"toyc.source_host": {"type": "str", "default": "", "binding": "startup"},
                              "toyc.late_reader": {"type": "bool", "default": False, "binding": "startup"}}),
    "toyc.stubborn": spec("toyc.stubborn", "toyc.units:Stubborn", provides=["proc.loop@1"], process=["worker"],
                          reads=["toyc.log_dir"]),
    "toyc.tensor": spec("toyc.tensor", "toyc.tensor:Tensor", process=["worker"], reads=["toyc.log_dir"]),
}

FAST = {
    "nodeagent.heartbeat_interval_s": 0.1,
    "nodeagent.rpc_timeout_s": 5.0,
}


def build_tree(root: Path, extra: dict[str, str] | None = None) -> Path:
    """An engine source tree: the framework from this repo plus toy units."""
    src = root / "src"
    shutil.copytree(ENGINE, src / "engine", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    write_tree(src / "engine", {**TOY, **(extra or {})})
    return src


def topology(tmp: Path, hosts: list[str], ranks: list[str], roles: list[RoleSpec]) -> Topology:
    return Topology(
        hosts=[HostSpec(HostId(h), ADDRESS, str(tmp / h / "snapshots")) for h in hosts],
        mesh=Mesh({"tp": len(ranks)}),
        placement=[RankPlacement(i, HostId(h), i) for i, h in enumerate(ranks)],
        roles=roles,
    )


def role(name: str, units: list[str], hosts: list[str] | None = None) -> RoleSpec:
    return RoleSpec(name, [UnitId(u) for u in units], per_rank=hosts is None,
                    hosts=[HostId(h) for h in hosts or []])


@dataclass
class Agent:
    host: str
    process: asyncio.subprocess.Process
    root: Path

    def kill(self) -> None:
        """The machine goes silent: the agent dies first, so it cannot report its processes' exits
        (killing the group at once gives no order; on macOS a worker may die first and be reported),
        then whatever it started."""
        if self.process.returncode is None:
            self.process.kill()
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)


@dataclass
class Cluster:
    tmp: Path
    sup: Supervisor
    blobs: BlobStore
    agents: dict[str, Agent] = field(default_factory=dict[str, Agent])
    losses: list[Loss] = field(default_factory=list[Loss])
    agent_tunables: dict[str, float] = field(default_factory=lambda: dict(FAST))

    async def start_agent(self, host: str, platform: dict[str, object] | None = None) -> Agent:
        root = self.tmp / host / "snapshots"
        env = {**os.environ, "PYTHONPATH": str(ENGINE)}
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "core.nodeagent", "--host", host, "--address", ADDRESS,
            "--supervisor", self.sup.address, "--root", str(root),
            "--platform", json.dumps(platform or {"name": "fake", "versions": {"driver": "0"}}),
            "--tunables", json.dumps(self.agent_tunables),
            env=env, start_new_session=True,
            stdout=open(self.tmp / f"agent-{host}.log", "wb"), stderr=asyncio.subprocess.STDOUT,
        )
        agent = self.agents[host] = Agent(host, process, root)
        return agent

    def ingest(self, src: Path) -> tuple[Identity, dict[str, str]]:
        files = self.blobs.ingest_tree(src)
        identity = Identity(self.blobs.put(json.dumps(files, sort_keys=True).encode())[:16])
        return identity, files

    def plan_for(self, identity: Identity, extra: Callable[[DesiredProc], dict[str, object]] | None = None
                 ) -> Callable[[DesiredProc], ProcessPlan]:
        return lambda d: process_plan(d, identity, {"toyc.log_dir": str(self.logs), **(extra(d) if extra else {})})

    @property
    def logs(self) -> Path:
        d = self.tmp / "logs"
        d.mkdir(exist_ok=True)
        return d

    def log_of(self, p: ProcId) -> list[str]:
        f = self.logs / f"{p.host}-{p.role}-{p.index}.log"
        return f.read_text().splitlines() if f.exists() else []

    def output_of(self, p: ProcId) -> list[str]:
        """What the process (every incarnation of it) wrote to its stdout and stderr."""
        runs = sorted((self.tmp / str(p.host) / "snapshots" / ".run").glob(f"{p.role}-{p.index}-*/log"))
        return [line for f in runs for line in f.read_text(errors="replace").splitlines()]

    async def wait_loss(self, kind: str, timeout: float = 10.0) -> Loss:
        loss = await asyncio.wait_for(self.sup.losses.get(), timeout)
        while loss.kind != kind:
            loss = await asyncio.wait_for(self.sup.losses.get(), timeout)
        return loss


def process_plan(d: DesiredProc, identity: Identity, values: dict[str, object]) -> ProcessPlan:
    """The units of ``d`` with their owned keys set from ``values`` (defaults otherwise)."""
    units = [SPECS[u] for u in d.units]
    config = {
        key: ConfigEntry(s.id, decl, values.get(key, decl.default))
        for s in units for key, decl in s.manifest.config.items()
    }
    return ProcessPlan(d.proc, identity, units, config=config)


@asynccontextmanager
async def cluster(tmp: Path, hosts: list[str], proc_tunables: dict[str, float] | None = None,
                  agent_tunables: dict[str, float] | None = None, sup_tunables: dict[str, float] | None = None,
                  ) -> AsyncGenerator[Cluster]:
    zctx = zmq.asyncio.Context()
    # 10 s, as in tests/scenarios/harness.py (#29): under `pytest -n auto` a busy runner can stall a
    # node agent or this event loop for seconds, and 2 s then declared a live machine lost. The
    # machine-loss test, which measures detection, passes 2 s itself.
    tun = SupervisorTunables(**{"heartbeat_timeout_s": 10.0, "monitor_interval_s": 0.05,
                                "checkin_timeout_s": 30.0, "rpc_timeout_s": 10.0, **(sup_tunables or {})})
    sup = Supervisor(zctx, host=HostId("host-a"), bind_address=ADDRESS, blobs=BlobStore(tmp / "blobs"),
                     tun=tun, proc_tunables={"host.idle_poll_s": 0.01, **(proc_tunables or {})})
    await sup.start()
    c = Cluster(tmp, sup, sup.blobs, agent_tunables={**FAST, **(agent_tunables or {})})
    try:
        for h in hosts:
            await c.start_agent(h)
        (await sup.wait_for_hosts([HostId(h) for h in hosts], 20)).unwrap()
        yield c
    finally:
        for a in c.agents.values():
            a.kill()
        for a in c.agents.values():
            await a.process.wait()
        sup.close()
        zctx.destroy(linger=0)


def proc(host: str, role_: str, index: int) -> ProcId:
    return ProcId(HostId(host), role_, index)
