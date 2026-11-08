"""The unit manifest, ``unit.toml`` (doc/specs/README.md, "单元清单").

The manifest is read without importing unit code, so static checks and identity
computation never execute anything.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any, Literal

import msgspec

from core.common.ids import UnitId
from core.common.result import Diagnostic, Failed, Ok, Result, diag, fail

UnitKind = Literal["logic", "state"]
Binding = Literal["live", "plan", "startup"]
ConfigType = Literal["int", "float", "str", "bool", "list", "dict", "any"]
StateAction = str
"""Named by the state unit that offers and runs it (``ctx.on_action``); the framework keeps no list
of them (modules.html P4). Common ones: ``refit`` (new contents, same layout), ``grow`` (more
capacity, same addresses), ``relocate`` (move contents within the same version: EPLB moves experts
between ranks), ``invalidate`` (drop the contents). Carrying contents to a new version of the
unit's code is ``[state] migrate`` (doc/specs/state-changes.md)."""

_SERVICE_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)*@[0-9]+$")
_UNIT_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)*$")
_ACTION_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ENTRY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$")


def is_service_name(name: str) -> bool:
    """Service names carry an interface version: ``"sched.chunk_policy@1"``."""
    return _SERVICE_RE.match(name) is not None


class UnitSection(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    id: UnitId
    kind: UnitKind
    process: list[str]
    """Process roles this unit is installed in."""
    entry: str
    """``"module:Class"``."""

    @property
    def entry_module(self) -> str:
        return self.entry.partition(":")[0]

    @property
    def entry_class(self) -> str:
        return self.entry.partition(":")[2]


class Provides(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    services: list[str] = []


class Requires(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    services: list[str] = []
    """Required: the unit stays PENDING until all of them are provided."""
    optional: list[str] = []
    """Optional: fetched with ``ctx.get()``, may be ``None``."""


class ConfigKeyDecl(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """A config key owned by this unit; only the owner declares the default."""

    type: ConfigType
    default: Any
    binding: Binding
    doc: str = ""


class Reads(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    config: list[str] = []
    """Config keys owned by other units that this unit reads."""


class Capabilities(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    accept_config: bool = False
    """Can take new config values in place (``ctx.on_config``)."""
    actions: list[StateAction] = []
    """State units only: the actions the unit offers, by its own names (``StateAction``)."""
    graph_capturable: bool = False
    """Kernel solutions only: may be recorded into a graph."""
    captures_graph: bool = False
    """Plan units only: may call ``platform.graph()``."""
    native_extensions: list[str] = []
    """Native modules (``.so``) this unit loads; changing them restarts the process."""
    changes_kv_numerics: bool = False


class StateSection(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    drain_on_reload: bool = True
    migrate: str = ""
    """``"module:function"`` migrating state from the previous version."""


class UnitManifest(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    unit: UnitSection
    provides: Provides = Provides()
    requires: Requires = Requires()
    config: dict[str, ConfigKeyDecl] = {}
    reads: Reads = Reads()
    capabilities: Capabilities = Capabilities()
    state: StateSection | None = None

    @property
    def id(self) -> UnitId:
        return self.unit.id

    def readable_keys(self) -> set[str]:
        return set(self.config) | set(self.reads.config)


def check_manifest(m: UnitManifest) -> list[str]:
    """Structural checks that do not need other manifests. Returns problems."""
    problems: list[str] = []
    if not _UNIT_ID_RE.match(m.unit.id):
        problems.append(f"bad unit id {m.unit.id!r}")
    if not _ENTRY_RE.match(m.unit.entry):
        problems.append(f"bad entry {m.unit.entry!r}; expected 'module:Class'")
    if not m.unit.process:
        problems.append("unit.process is empty")
    for name in [*m.provides.services, *m.requires.services, *m.requires.optional]:
        if not is_service_name(name):
            problems.append(f"bad service name {name!r}; expected 'name@version'")
    overlap = set(m.requires.services) & set(m.requires.optional)
    if overlap:
        problems.append(f"services both required and optional: {sorted(overlap)}")
    self_dep = set(m.provides.services) & (set(m.requires.services) | set(m.requires.optional))
    if self_dep:
        problems.append(f"unit requires its own services: {sorted(self_dep)}")
    for key in set(m.config) & set(m.reads.config):
        problems.append(f"config key {key!r} both owned and read-only")
    if m.unit.kind == "logic":
        if m.state is not None:
            problems.append("[state] is only allowed on state units")
        if m.capabilities.actions:
            problems.append("capabilities.actions is only allowed on state units")
    for name in m.capabilities.actions:
        if not _ACTION_RE.match(name):
            problems.append(f"bad action name {name!r}; expected lowercase letters, digits and '_'")
    for key, decl in m.config.items():
        if not check_value_type(decl.type, decl.default):
            problems.append(f"default of {key!r} is not of type {decl.type}")
    return problems


def check_value_type(type_: ConfigType, value: Any) -> bool:
    """Strict: ``True`` is not an int, ``1`` is not a float."""
    match type_:
        case "int":
            return type(value) is int
        case "float":
            return type(value) is float
        case "str":
            return type(value) is str
        case "bool":
            return type(value) is bool
        case "list":
            return type(value) is list
        case "dict":
            return type(value) is dict
        case "any":
            return True


def parse_manifest(text: str, *, source: str = "<string>") -> Result[UnitManifest]:
    try:
        raw = tomllib.loads(text)
        m = msgspec.convert(raw, UnitManifest)
    except (tomllib.TOMLDecodeError, msgspec.ValidationError) as e:
        return fail("manifest", "parse", str(e), subject=source, code="manifest.invalid")
    problems = check_manifest(m)
    if problems:
        return Failed(
            [diag("manifest", "parse", p, subject=source, code="manifest.invalid") for p in problems]
        )
    return Ok(m)


def load_manifest(path: Path) -> Result[UnitManifest]:
    return parse_manifest(path.read_text(), source=str(path))


MANIFEST_FILE = "unit.toml"


def discover(root: Path) -> tuple[list[UnitManifest], list[Diagnostic]]:
    """Every ``unit.toml`` under a code root, read without importing anything. The control
    plane uses it to load its own units; ``core.manifest`` does the full checks."""
    manifests: list[UnitManifest] = []
    problems: list[Diagnostic] = []
    for path in sorted(root.rglob(MANIFEST_FILE)):
        r = load_manifest(path)
        if isinstance(r, Failed):
            problems += r.diagnostics
        else:
            manifests.append(r.value)
    return manifests, problems
