"""Topology data (doc/specs/supervisor.md §3, P1, P7).

Which machines exist, the parallel mesh, where each rank lives, and which
process roles run where with which units. A single machine is the special case
of one host. Topology is data: it goes into the identity, and changing it is a
change-set like any other.
"""

from __future__ import annotations

import msgspec

from core.common.ids import HostId, ProcId, UnitId


class HostSpec(msgspec.Struct, frozen=True):
    id: HostId
    address: str
    """Where this host's processes bind and are reached; data, never a literal in code."""
    snapshot_root: str
    """Local directory for code snapshots (not on shared storage)."""
    gpus: list[int] = []
    run_root: str = ""
    """Local directory for per-process run directories; defaults under snapshot_root."""
    weights_root: str = ""


class Mesh(msgspec.Struct, frozen=True):
    dims: dict[str, int]
    """e.g. ``{"tp": 2}`` or ``{"dp": 8, "ep": 8}``."""

    @property
    def size(self) -> int:
        n = 1
        for d in self.dims.values():
            n *= d
        return n


class RankPlacement(msgspec.Struct, frozen=True):
    rank: int
    host: HostId
    device: int
    nvlink_domain: str = ""


class RoleSpec(msgspec.Struct, frozen=True):
    name: str
    units: list[UnitId]
    per_rank: bool = False
    """One process per rank, placed by ``Topology.placement`` (workers)."""
    hosts: list[HostId] = []
    """For roles that are not per rank: process ``i`` runs on ``hosts[i]``."""
    env: dict[str, str] = {}


class Topology(msgspec.Struct, frozen=True):
    hosts: list[HostSpec]
    mesh: Mesh
    placement: list[RankPlacement]
    roles: list[RoleSpec]
    rendezvous: str = ""

    def host(self, host: HostId) -> HostSpec:
        for h in self.hosts:
            if h.id == host:
                return h
        raise KeyError(host)

    def check(self) -> list[str]:
        problems: list[str] = []
        ids = [h.id for h in self.hosts]
        if len(set(ids)) != len(ids):
            problems.append("duplicate host ids")
        ranks = sorted(p.rank for p in self.placement)
        if ranks != list(range(self.mesh.size)):
            problems.append(f"placement covers ranks {ranks}, mesh has {self.mesh.size}")
        for p in self.placement:
            if p.host not in ids:
                problems.append(f"rank {p.rank} placed on unknown host {p.host}")
        for r in self.roles:
            for h in r.hosts:
                if h not in ids:
                    problems.append(f"role {r.name} placed on unknown host {h}")
            if r.per_rank and r.hosts:
                problems.append(f"role {r.name} is per rank; it is placed by the mesh, not hosts")
        names = [r.name for r in self.roles]
        if len(set(names)) != len(names):
            problems.append("duplicate role names")
        return problems


class DesiredProc(msgspec.Struct, frozen=True):
    """What one process should be: the unit of comparison when topology changes."""

    proc: ProcId
    units: list[UnitId]
    gpus: list[int]
    env: dict[str, str]
    rank: int | None = None


def desired_procs(topo: Topology) -> dict[ProcId, DesiredProc]:
    out: dict[ProcId, DesiredProc] = {}
    for role in topo.roles:
        if role.per_rank:
            for p in sorted(topo.placement, key=lambda p: p.rank):
                pid = ProcId(p.host, role.name, p.rank)
                out[pid] = DesiredProc(pid, list(role.units), [p.device], dict(role.env), p.rank)
        else:
            for i, host in enumerate(role.hosts):
                pid = ProcId(host, role.name, i)
                out[pid] = DesiredProc(pid, list(role.units), [], dict(role.env))
    return out


class TopologyPlan(msgspec.Struct, frozen=True):
    start: list[ProcId]
    stop: list[ProcId]
    restart: list[ProcId]
    keep: list[ProcId]


def diff_topology(
    current: dict[ProcId, DesiredProc], desired: dict[ProcId, DesiredProc],
    changed: set[ProcId] | None = None,
) -> TopologyPlan:
    """``changed``: processes whose plan differs for reasons outside the topology
    (e.g. the mesh they were built for)."""
    changed = changed or set()
    start = sorted(p for p in desired if p not in current)
    stop = sorted(p for p in current if p not in desired)
    both = sorted(p for p in desired if p in current)
    restart = [p for p in both if desired[p] != current[p] or p in changed]
    keep = [p for p in both if p not in restart]
    return TopologyPlan(start, stop, restart, keep)
