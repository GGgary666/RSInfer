"""Versioned messages between the supervisor, node agents and processes
(doc/specs/nodeagent.md §6, supervisor.md §4). Only these cross the
boundaries of the independent framework processes, so each of them can later
be rewritten in another language.
"""

from __future__ import annotations

from typing import Literal

import msgspec

from core.common.ids import HostId, Identity, ProcId
from core.common.messages import SchemaRegistry
from core.common.result import Diagnostic
from core.ipc.stream import StreamAddress
from platforms.base import PlatformInfo
from unit.runtime.types import InstanceState, ProcessPlan

# ---------------------------------------------------------------- spawning


class ProcSpec(msgspec.Struct, frozen=True):
    """Everything a node agent needs to start one process."""

    proc: ProcId
    identity: Identity
    """Selects the snapshot directory the process imports code from."""
    plan: ProcessPlan
    control: str
    """Supervisor RPC address the process checks in with."""
    bind_address: str
    env: dict[str, str] = {}
    gpus: list[int] = []
    platform: str = "fake"
    platform_options: dict[str, int] = {}
    tunables: dict[str, float | int] = {}
    """Framework tunables for the process (``host.*``, ``ipc.*``, ``change.*``)."""
    incarnation: int = 0
    """Grows every time the supervisor spawns this ProcId; reports about older ones are stale."""
    start_paused: bool = False
    """A transaction starts candidate processes fenced until all participants commit."""


# ------------------------------------------------------ agent -> supervisor


class Register(msgspec.Struct, frozen=True):
    host: HostId
    agent: str
    """The agent's RPC address."""
    platform: PlatformInfo


class Registered(msgspec.Struct, frozen=True):
    accepted: bool
    diagnostics: list[Diagnostic] = []


ProcRunState = Literal["running", "exited"]


class ProcBeat(msgspec.Struct, frozen=True):
    proc: ProcId
    incarnation: int
    state: ProcRunState
    exit_code: int | None = None


class Heartbeat(msgspec.Struct, frozen=True):
    host: HostId
    procs: list[ProcBeat]


class Exited(msgspec.Struct, frozen=True):
    proc: ProcId
    incarnation: int
    exit_code: int
    killed: bool = False
    """The agent had to kill it after the stop timeout."""


class FetchBlobs(msgspec.Struct, frozen=True):
    hashes: list[str]


class Blobs(msgspec.Struct, frozen=True):
    blobs: dict[str, bytes]
    missing: list[str] = []


# --------------------------------------------------- supervisor -> agent


class Materialize(msgspec.Struct, frozen=True):
    identity: Identity
    files: dict[str, str]
    """Snapshot-relative path -> content hash."""
    keep: list[Identity] = []
    """Snapshots the processes on this host still load code from: never pruned."""


class Materialized(msgspec.Struct, frozen=True):
    identity: Identity
    path: str
    fetched: int
    reused: int


class Spawn(msgspec.Struct, frozen=True):
    spec: ProcSpec


class Spawned(msgspec.Struct, frozen=True):
    proc: ProcId
    local_pid: int | None = None
    """For operators and tests on that host only; never used to address the process."""


class Stop(msgspec.Struct, frozen=True):
    proc: ProcId
    timeout_s: float
    incarnation: int
    """Which process of this ProcId: a late Stop must not hit its successor."""


class Stopped(msgspec.Struct, frozen=True):
    proc: ProcId
    exit_code: int
    killed: bool


class AllocShm(msgspec.Struct, frozen=True):
    proc: ProcId
    purpose: str


class ShmAllocated(msgspec.Struct, frozen=True):
    name: str


# ------------------------------------------------- process -> supervisor


class UnitBrief(msgspec.Struct, frozen=True):
    unit: str
    state: InstanceState
    error: str | None = None
    generation: int = 0


class CheckIn(msgspec.Struct, frozen=True):
    proc: ProcId
    incarnation: int
    identity: Identity
    endpoints: dict[str, StreamAddress]
    units: list[UnitBrief]
    diagnostics: list[Diagnostic]
    sys_path: list[str]
    module_files: list[str]
    """Files of every loaded module under a code root; for the snapshot-only check."""


class CheckedIn(msgspec.Struct, frozen=True):
    pass


class Lookup(msgspec.Struct, frozen=True):
    proc: ProcId
    channel: str


class Found(msgspec.Struct, frozen=True):
    address: StreamAddress | None


# ------------------------------------------------- anyone -> process host


class Status(msgspec.Struct, frozen=True):
    pass


class ProcStatusReport(msgspec.Struct, frozen=True):
    proc: ProcId
    identity: Identity
    epoch: int | None
    units: list[UnitBrief]
    paused: bool = False
    diagnostics: list[Diagnostic] = []


MESSAGES: list[tuple[str, type[msgspec.Struct]]] = [
    ("proc.spec@1", ProcSpec),
    ("agent.register@1", Register),
    ("agent.registered@1", Registered),
    ("agent.heartbeat@1", Heartbeat),
    ("agent.exited@1", Exited),
    ("blobs.fetch@1", FetchBlobs),
    ("blobs.blobs@1", Blobs),
    ("agent.materialize@1", Materialize),
    ("agent.materialized@1", Materialized),
    ("agent.spawn@1", Spawn),
    ("agent.spawned@1", Spawned),
    ("agent.stop@1", Stop),
    ("agent.stopped@1", Stopped),
    ("agent.alloc_shm@1", AllocShm),
    ("agent.shm_allocated@1", ShmAllocated),
    ("proc.check_in@1", CheckIn),
    ("proc.checked_in@1", CheckedIn),
    ("proc.lookup@1", Lookup),
    ("proc.found@1", Found),
    ("proc.status@1", Status),
    ("proc.status_report@1", ProcStatusReport),
]


def register_all(registry: SchemaRegistry) -> SchemaRegistry:
    from core.change.protocol import MESSAGES as CHANGE_MESSAGES

    for name, type_ in [*MESSAGES, *CHANGE_MESSAGES]:
        registry.register(name, type_)
    return registry


def control_registry() -> SchemaRegistry:
    return register_all(SchemaRegistry())
