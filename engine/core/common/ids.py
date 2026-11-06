"""Identifiers shared by every framework component (doc/specs/README.md).

No identifier contains a local PID, a local path, or a loopback address (P7).
"""

from __future__ import annotations

from typing import NewType

import msgspec

HostId = NewType("HostId", str)
"""A machine, as named in the topology data, e.g. ``"host-a"``."""

UnitId = NewType("UnitId", str)
"""A unit at the code level: the ``id`` in its ``unit.toml``."""

Identity = NewType("Identity", str)
"""A system identity: the hash of the identity manifest (see change.md)."""

Epoch = int
"""Step number. Monotonic; changes only take effect between two epochs."""


class ProcId(msgspec.Struct, frozen=True, order=True):
    """A process, addressed by host, role and index; never by PID."""

    host: HostId
    role: str
    index: int

    def __str__(self) -> str:
        return f"{self.host}/{self.role}/{self.index}"


class Endpoint(msgspec.Struct, frozen=True, order=True):
    """An addressable communication endpoint of a process."""

    proc: ProcId
    channel: str

    def __str__(self) -> str:
        return f"{self.proc}#{self.channel}"


class InstanceId(msgspec.Struct, frozen=True, order=True):
    """One load of a unit in one process; ``generation`` grows on every reload."""

    proc: ProcId
    unit: UnitId
    generation: int

    def __str__(self) -> str:
        return f"{self.proc}:{self.unit}#{self.generation}"
