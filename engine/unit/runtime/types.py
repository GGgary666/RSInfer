"""Inputs and outputs of the unit runtime (doc/specs/unit-runtime.md §7)."""

from __future__ import annotations

from enum import StrEnum

import msgspec

from core.common.ids import Identity, InstanceId, ProcId, UnitId
from core.common.manifest import UnitManifest
from core.common.result import Diagnostic
from core.context.config import ConfigEntry


class InstanceState(StrEnum):
    PENDING = "PENDING"
    LOADING = "LOADING"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"
    UNLOADING = "UNLOADING"
    DISPOSED = "DISPOSED"


class UnitSpec(msgspec.Struct, frozen=True):
    manifest: UnitManifest

    @property
    def id(self) -> UnitId:
        return self.manifest.unit.id


class ProcessPlan(msgspec.Struct, frozen=True):
    """Which units this process installs, and the config keys they may read."""

    proc: ProcId
    identity: Identity
    units: list[UnitSpec]
    config: dict[str, ConfigEntry] = {}
    code_root: str | None = None
    """The snapshot's ``engine/`` directory to import code from; ``None`` keeps sys.path."""


class ReloadPlan(msgspec.Struct, frozen=True):
    """A reload instruction from ``core.change``; executed only at a safe point."""

    changed_modules: list[str] = []
    units_to_reload: list[UnitSpec] = []
    """New specs of units whose code (or manifest) changed."""
    units_to_add: list[UnitSpec] = []
    units_to_remove: list[UnitId] = []
    new_code_root: str | None = None
    config: dict[str, ConfigEntry] = {}
    """Declarations of keys introduced by added or reloaded units."""


class ReloadReport(msgspec.Struct, frozen=True):
    unloaded: list[UnitId]
    """In the order they were unloaded (users before providers)."""
    loaded: list[UnitId]
    """In the order they were loaded (providers before users)."""
    reimported: list[str]
    """Modules dropped from the module cache and imported again."""
    unplanned: list[UnitId] = []
    """Units the plan did not list but whose code depends on a changed module."""
    diagnostics: list[Diagnostic] = []


class InstanceInfo(msgspec.Struct, frozen=True):
    unit: UnitId
    kind: str
    state: InstanceState
    generation: int
    instance: InstanceId | None
    fingerprint: list[InstanceId]
    """Providers of the required services this instance was loaded against."""
    provides: list[str]
    uses: dict[str, InstanceId]
    registrations: list[str]
    staged: bool = False
    error: str | None = None


class RuntimeSnapshot(msgspec.Struct, frozen=True):
    proc: ProcId
    code_root: str | None
    units: list[InstanceInfo]
    services: dict[str, InstanceId]
    diagnostics: list[Diagnostic]
    """Findings since the process started (leaks, failed undo functions, ...)."""

    def unit(self, unit: str) -> InstanceInfo:
        for u in self.units:
            if u.unit == unit:
                return u
        raise KeyError(unit)

    def states(self) -> dict[str, InstanceState]:
        return {u.unit: u.state for u in self.units}
