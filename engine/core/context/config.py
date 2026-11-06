"""Config keys: the only way to read config (doc/specs/context.md).

Every key is declared in exactly one owner's manifest with a type, a default and
a binding. Reading an undeclared key, or a key not listed in the reader's own
manifest, raises. Every read is recorded with its phase, for ``core.deps``.
Changing a value never restarts the engine; values change only at an epoch
boundary through ``stage()`` + ``commit()``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Collection, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal

import msgspec

from core.common.ids import InstanceId, UnitId
from core.common.manifest import ConfigKeyDecl, UnitManifest, check_value_type
from core.common.result import Diagnostic, Failed, Ok, Result, diag
from core.context.capture import ValueDep, observing, report_value

ReadPhase = Literal["runtime", "plan_build", "startup"]

_phase: ContextVar[ReadPhase] = ContextVar("config_phase", default="runtime")
_plan: ContextVar[str | None] = ContextVar("config_plan", default=None)


class ConfigError(Exception):
    pass


class UndeclaredConfigKey(ConfigError, KeyError):
    pass


class ConfigTypeError(ConfigError, TypeError):
    pass


class ConfigRead(msgspec.Struct, frozen=True):
    key: str
    reader: InstanceId
    phase: ReadPhase
    value_hash: str


class ConfigEntry(msgspec.Struct, frozen=True):
    """A declared key as shipped to a process: owner, declaration, current value."""

    owner: UnitId
    decl: ConfigKeyDecl
    value: Any


class ConfigReads(msgspec.Struct, frozen=True):
    runtime: list[ConfigRead]
    """One record per (key, reader); ``value_hash`` is of the first read."""
    startup: list[ConfigRead]
    plan_build: dict[str, list[ConfigRead]]
    """Per plan id, every read in order."""


class ConfigCommit(msgspec.Struct, frozen=True):
    changed: dict[str, Any]
    hooks_called: list[tuple[str, InstanceId]]
    hook_failures: list[Diagnostic]


def value_hash(value: Any) -> str:
    return hashlib.sha256(msgspec.msgpack.encode(value)).hexdigest()[:16]


@contextmanager
def read_phase(phase: ReadPhase) -> Generator[None]:
    token = _phase.set(phase)
    try:
        yield
    finally:
        _phase.reset(token)


@contextmanager
def plan_scope(plan_id: str) -> Generator[None]:
    """Reads inside are recorded one by one against ``plan_id`` (opened by plan units)."""
    t1 = _phase.set("plan_build")
    t2 = _plan.set(plan_id)
    try:
        yield
    finally:
        _plan.reset(t2)
        _phase.reset(t1)


@dataclass
class _Key:
    owner: UnitId
    decl: ConfigKeyDecl
    value: Any


@dataclass
class _Reads:
    runtime: dict[tuple[str, InstanceId], ConfigRead] = field(default_factory=dict)
    startup: dict[tuple[str, InstanceId], ConfigRead] = field(default_factory=dict)
    plan_build: dict[str, list[ConfigRead]] = field(default_factory=dict)


class ConfigStore:
    """One per process."""

    def __init__(self) -> None:
        self._keys: dict[str, _Key] = {}
        self._staged: dict[str, Any] | None = None
        self._reads = _Reads()
        self._hooks: dict[str, dict[InstanceId, Callable[[Any], None]]] = {}

    # -- declarations -------------------------------------------------------

    def declare(self, key: str, owner: UnitId, decl: ConfigKeyDecl, value: Any = ...) -> Result[None]:
        if value is ...:
            value = decl.default
        known = self._keys.get(key)
        if known is not None and known.owner != owner:
            return Failed([diag(
                "context", "declare", f"key {key!r} is already owned by {known.owner}",
                subject=owner, code="config.owner_conflict",
                hint="exactly one unit declares a key's default; others list it under [reads]",
            )])
        if not check_value_type(decl.type, value):
            return Failed([diag(
                "context", "declare", f"value {value!r} of {key!r} is not of type {decl.type}",
                subject=owner, code="config.type",
            )])
        self._keys[key] = _Key(owner, decl, value)
        return Ok(None)

    def declare_manifest(self, manifest: UnitManifest, values: dict[str, Any] | None = None) -> Result[None]:
        diags: list[Diagnostic] = []
        for key, decl in manifest.config.items():
            r = self.declare(key, manifest.id, decl, (values or {}).get(key, ...))
            if isinstance(r, Failed):
                diags += r.diagnostics
        return Failed(diags) if diags else Ok(None)

    def declare_entries(self, entries: dict[str, ConfigEntry]) -> Result[None]:
        diags: list[Diagnostic] = []
        for key, e in entries.items():
            r = self.declare(key, e.owner, e.decl, e.value)
            if isinstance(r, Failed):
                diags += r.diagnostics
        return Failed(diags) if diags else Ok(None)

    def retract_owner(self, owner: UnitId) -> None:
        for key in [k for k, v in self._keys.items() if v.owner == owner]:
            del self._keys[key]

    def check_readable(self, manifest: UnitManifest) -> list[Diagnostic]:
        """Every key a unit may read must be declared in this process."""
        return [
            diag("context", "declare", f"reads undeclared key {key!r}", subject=manifest.id,
                 code="config.undeclared", hint="no unit in the system owns this key")
            for key in sorted(manifest.readable_keys())
            if key not in self._keys
        ]

    def entries(self) -> dict[str, ConfigEntry]:
        return {k: ConfigEntry(v.owner, v.decl, v.value) for k, v in self._keys.items()}

    def restore_entries(self, entries: dict[str, ConfigEntry]) -> None:
        """Put declarations back as ``entries()`` returned them (reload rollback)."""
        self._keys = {k: _Key(e.owner, e.decl, e.value) for k, e in entries.items()}

    def owner_of(self, key: str) -> UnitId | None:
        k = self._keys.get(key)
        return k.owner if k else None

    def binding_of(self, key: str) -> str:
        return self._key(key).decl.binding

    # -- values -------------------------------------------------------------

    def _key(self, key: str) -> _Key:
        try:
            return self._keys[key]
        except KeyError:
            raise UndeclaredConfigKey(f"config key {key!r} is not declared") from None

    def _validate(self, values: dict[str, Any]) -> list[Diagnostic]:
        diags: list[Diagnostic] = []
        for key, value in values.items():
            k = self._keys.get(key)
            if k is None:
                diags.append(diag("context", "validate", f"key {key!r} is not declared",
                                  subject=key, code="config.undeclared"))
            elif not check_value_type(k.decl.type, value):
                diags.append(diag("context", "validate",
                                  f"value {value!r} is not of type {k.decl.type}",
                                  subject=key, code="config.type"))
        return diags

    def stage(self, values: dict[str, Any]) -> Result[dict[str, Any]]:
        """Validate new values; they take effect at ``commit()`` (an epoch boundary)."""
        diags = self._validate(values)
        if diags:
            return Failed(diags)
        changed = {k: v for k, v in values.items() if self._keys[k].value != v}
        self._staged = changed
        return Ok(dict(changed))

    def discard_staged(self) -> None:
        self._staged = None

    def commit(self) -> ConfigCommit:
        """Apply staged values and run ``on_config`` hooks of their readers."""
        changed = self._staged or {}
        self._staged = None
        for key, value in changed.items():
            self._keys[key].value = value
        return self.run_hooks(changed)

    def apply(self, values: dict[str, Any]) -> Result[dict[str, Any]]:
        """Set values without running hooks: part of a larger operation (a reload) that
        runs the hooks once it succeeded, or puts the old values back if it fails."""
        diags = self._validate(values)
        if diags:
            return Failed(diags)
        changed = {k: v for k, v in values.items() if self._keys[k].value != v}
        for key, value in changed.items():
            self._keys[key].value = value
        return Ok(changed)

    def run_hooks(self, changed: dict[str, Any], *, skip_units: Collection[str] = ()) -> ConfigCommit:
        """``on_config`` hooks for ``changed`` keys; units in ``skip_units`` were built after the change."""
        called: list[tuple[str, InstanceId]] = []
        failures: list[Diagnostic] = []
        for key, value in changed.items():
            for inst, hook in list(self._hooks.get(key, {}).items()):
                if inst.unit in skip_units:
                    continue
                try:
                    hook(value)
                    called.append((key, inst))
                except Exception as e:  # noqa: BLE001 - reported, never raised across units
                    failures.append(diag("context", "commit", f"on_config({key!r}) raised {e!r}",
                                         subject=str(inst), code="config.hook_failed"))
        return ConfigCommit(dict(changed), called, failures)

    def set_now(self, values: dict[str, Any]) -> Result[None]:
        """Set values outside the epoch protocol (process start, tests)."""
        r = self.stage(values)
        if isinstance(r, Failed):
            return r
        self.commit()
        return Ok(None)

    # -- reads --------------------------------------------------------------

    def read(self, key: str, reader: InstanceId) -> Any:
        k = self._key(key)
        phase = _phase.get()
        if phase == "plan_build":
            plan = _plan.get() or ""
            self._reads.plan_build.setdefault(plan, []).append(
                ConfigRead(key, reader, phase, value_hash(k.value)))
        else:
            table = self._reads.runtime if phase == "runtime" else self._reads.startup
            if (key, reader) not in table:
                table[(key, reader)] = ConfigRead(key, reader, phase, value_hash(k.value))
        if observing():
            report_value(ValueDep("config", key, k.value, lambda: self._keys[key].value))
        return k.value

    def reads(self) -> ConfigReads:
        r = self._reads
        return ConfigReads(
            sorted(r.runtime.values(), key=_read_key),
            sorted(r.startup.values(), key=_read_key),
            {plan: list(v) for plan, v in r.plan_build.items()},
        )

    def readers_of(self, key: str) -> set[InstanceId]:
        r = self._reads
        out = {inst for (k, inst) in r.runtime if k == key}
        out |= {inst for (k, inst) in r.startup if k == key}
        for reads in r.plan_build.values():
            out |= {x.reader for x in reads if x.key == key}
        return out

    def forget_reader(self, reader: InstanceId) -> None:
        r = self._reads
        for table in (r.runtime, r.startup):
            for k in [k for k in table if k[1] == reader]:
                del table[k]

    def drop_plan(self, plan_id: str) -> None:
        self._reads.plan_build.pop(plan_id, None)

    # -- hooks --------------------------------------------------------------

    def add_hook(self, key: str, reader: InstanceId, handler: Callable[[Any], None]) -> Callable[[], None]:
        self._key(key)
        hooks = self._hooks.setdefault(key, {})
        if reader in hooks:
            raise ConfigError(f"{reader} already has an on_config hook for {key!r}")
        hooks[reader] = handler

        def remove() -> None:
            self._hooks.get(key, {}).pop(reader, None)

        return remove

    def hooked(self, key: str) -> set[InstanceId]:
        return set(self._hooks.get(key, {}))

    def hooks_of(self, reader: InstanceId) -> list[str]:
        return sorted(k for k, hooks in self._hooks.items() if reader in hooks)


def _read_key(r: ConfigRead) -> tuple[str, InstanceId]:
    return (r.key, r.reader)


class ConfigView:
    """What a unit sees as ``config``: only keys in its own manifest."""

    __slots__ = ("_store", "_reader", "_allowed")

    def __init__(self, store: ConfigStore, reader: InstanceId, allowed: frozenset[str]) -> None:
        self._store = store
        self._reader = reader
        self._allowed = allowed

    def __getitem__(self, key: str) -> Any:
        if key not in self._allowed:
            raise UndeclaredConfigKey(
                f"{self._reader.unit} reads {key!r}, which is not in its manifest "
                "([config] or [reads])"
            )
        return self._store.read(key, self._reader)

    def get(self, key: str) -> Any:
        return self[key]

    def keys(self) -> frozenset[str]:
        return self._allowed

    def __contains__(self, key: object) -> bool:
        return key in self._allowed
