"""Static checks of an engine source tree (doc/specs/control-plane.md §3).

Everything here reads files; nothing imports or runs unit code. The rules are
the ones that make "unload, then load again" reliable (modules.html ③ §6), plus
the cross-checks between what a unit declares and what its code does.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from core.common.framework import FrameworkModules
from core.common.ids import UnitId
from core.common.result import Diagnostic, diag
from core.sourcetree import Tree, read_tree
from core.supervisor.topology import Topology
from unit.runtime.imports import imports_of

COMPONENT = "core.manifest"

CONTROL_ROLE = "control"
HOST_SERVICES: Mapping[str, frozenset[str]] = MappingProxyType({
    CONTROL_ROLE: frozenset({"ctrl.change@1", "ctrl.supervisor@1", "ctrl.blobs@1", "ctrl.events@1", "ctrl.self@1"}),
})
"""Services the control host provides; every other process host provides ``proc.control@1`` and
``proc.streams@1``."""
PROCESS_HOST_SERVICES = frozenset({"proc.control@1", "proc.streams@1"})

MUTABLE_CALLS = frozenset({"list", "dict", "set", "bytearray", "defaultdict", "deque", "OrderedDict", "Counter"})
NATIVE_SUFFIXES = (".so", ".pyd")
READ_METHODS = frozenset({"get", "peek"})
"""``ConfigView`` methods that read a key: ``peek`` reads without recording the read, but the key
must still be declared."""


def check_tree(root: Path, topology: Topology | None, framework: FrameworkModules
               ) -> tuple[list[Diagnostic], list[Diagnostic]]:
    """Errors (nothing may run) and warnings."""
    tree = read_tree(root)
    errors = list(tree.errors)
    warnings: list[Diagnostic] = []
    parsed = _parse_all(tree, errors)
    _entries(tree, errors)
    _config(tree, parsed, errors, warnings)
    _roles(tree, topology, errors)
    _imports(tree, parsed, framework, errors)
    _modules(tree, parsed, framework, errors)
    _natives(tree, errors, warnings)
    return errors, warnings


def _parse_all(tree: Tree, errors: list[Diagnostic]) -> dict[str, ast.Module]:
    """Every module's syntax tree. A module that does not parse is an error, and the checks that
    read code skip it: they cannot tell what it imports or reads."""
    out: dict[str, ast.Module] = {}
    for module, path in sorted(tree.modules.items()):
        try:
            out[module] = ast.parse(path.read_bytes(), filename=str(path))
        except (SyntaxError, ValueError) as e:  # ValueError: undecodable source
            where = f"{path.relative_to(tree.root).as_posix()}"
            if isinstance(e, SyntaxError) and e.lineno is not None:
                where += f", line {e.lineno}"
            reason = e.msg if isinstance(e, SyntaxError) else str(e)
            errors.append(diag(COMPONENT, "syntax", f"{module} does not parse ({where}): {reason}",
                               subject=tree.owner.get(module) or module, code="manifest.syntax_error"))
    return out


def _entries(tree: Tree, errors: list[Diagnostic]) -> None:
    for uid, s in tree.units.items():
        if s.manifest.unit.entry_module not in tree.modules:
            errors.append(diag(COMPONENT, "manifest", f"entry module {s.manifest.unit.entry_module} does not exist",
                               subject=uid, code="manifest.missing_entry"))


def _config(tree: Tree, parsed: Mapping[str, ast.Module], errors: list[Diagnostic],
            warnings: list[Diagnostic]) -> None:
    owners: dict[str, list[UnitId]] = {}
    for uid, s in tree.units.items():
        for key in s.manifest.config:
            owners.setdefault(key, []).append(uid)
    for key, us in sorted(owners.items()):
        if len(us) > 1:
            errors.append(diag(COMPONENT, "config", f"key {key!r} is declared by {sorted(us)}", subject=key,
                               code="manifest.config_owner", hint="one unit owns a key; the others list it in [reads]"))
    for uid, s in sorted(tree.units.items()):
        for key in s.manifest.reads.config:
            if key not in owners:
                errors.append(diag(COMPONENT, "config", f"reads {key!r}, which no unit declares", subject=uid,
                                   code="manifest.undeclared_read"))
        readable = s.manifest.readable_keys()
        used: set[str] = set()
        for module in s.modules:
            if module in parsed:
                used |= _config_keys(parsed[module])
        for key in sorted(used - readable):
            errors.append(diag(COMPONENT, "config", f"code reads {key!r}, which its manifest does not declare",
                               subject=uid, code="manifest.read_not_declared",
                               hint="add it to [config] (if this unit owns it) or [reads]"))
        if any(module not in parsed for module in s.modules):
            continue  # its unread keys may be read by the code that does not parse
        for key in sorted(set(s.manifest.reads.config) - used):
            warnings.append(diag(COMPONENT, "config", f"declares it reads {key!r}, but its code never does",
                                 subject=uid, code="manifest.unused_read", severity="warning"))


def _config_keys(mod: ast.Module) -> set[str]:
    """String keys in ``config["k"]``, ``x.config["k"]``, ``config.get("k")``, ``config.peek("k")``."""
    keys: set[str] = set()
    for node in ast.walk(mod):
        if isinstance(node, ast.Subscript) and _is_config(node.value) and isinstance(node.slice, ast.Constant) \
                and isinstance(node.slice.value, str):
            keys.add(node.slice.value)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in READ_METHODS \
                and _is_config(node.func.value) and node.args and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str):
            keys.add(node.args[0].value)
    return keys


def _is_config(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "config") or \
        (isinstance(node, ast.Attribute) and node.attr == "config")


def _roles(tree: Tree, topology: Topology | None, errors: list[Diagnostic]) -> None:
    roles = [CONTROL_ROLE, *(r.name for r in topology.roles)] if topology is not None else [CONTROL_ROLE]
    for role in roles:
        units = tree.role_units(topology, role)
        provided = set(HOST_SERVICES.get(role, PROCESS_HOST_SERVICES))
        for uid in units:
            s = tree.units.get(uid)
            if s is None:
                errors.append(diag(COMPONENT, "topology", f"role {role!r} lists {uid}, which does not exist",
                                   subject=uid, code="manifest.unknown_unit"))
                continue
            if role not in s.manifest.unit.process:
                errors.append(diag(COMPONENT, "topology", f"is placed in role {role!r} but its process list is "
                                   f"{s.manifest.unit.process}", subject=uid, code="manifest.wrong_role"))
            provided |= set(s.manifest.provides.services)
        for uid in units:
            s = tree.units.get(uid)
            if s is None:
                continue
            for name in s.manifest.requires.services:
                if name not in provided:
                    errors.append(diag(COMPONENT, "services", f"requires {name}, which nothing in role {role!r} "
                                       "provides", subject=uid, code="manifest.unprovided_service",
                                       hint="it would wait in PENDING forever"))


def _imports(tree: Tree, parsed: Mapping[str, ast.Module], framework: FrameworkModules,
             errors: list[Diagnostic]) -> None:
    for module, uid in sorted(tree.owner.items()):
        if module not in parsed:
            continue
        path = tree.modules[module]
        for name in sorted(imports_of(path, module, path.name == "__init__.py")):
            target = _resolve(name, tree.modules)
            if target is None or framework.contains(target):
                continue
            other = tree.owner.get(target)
            if other is not None and other != uid:
                errors.append(diag(COMPONENT, "imports", f"{module} imports {target}, code of {other}", subject=uid,
                                   code="manifest.cross_unit_import",
                                   hint="reach other units through their services (ctx.use), never their objects"))


def _resolve(name: str, modules: dict[str, Path]) -> str | None:
    """The module an imported name refers to (``a.b.c`` may be attribute ``c`` of module ``a.b``)."""
    parts = name.split(".")
    while parts:
        candidate = ".".join(parts)
        if candidate in modules:
            return candidate
        parts.pop()
    return None


def _modules(tree: Tree, parsed: Mapping[str, ast.Module], framework: FrameworkModules,
             errors: list[Diagnostic]) -> None:
    for module, mod in sorted(parsed.items()):
        if framework.contains(module):
            continue
        uid = tree.owner.get(module)
        subject = uid or module
        for node in mod.body:
            target, value = _assignment(node)
            if target is not None and value is not None and not target.startswith("__") and _mutable(value):
                errors.append(diag(COMPONENT, "rules", f"{module}: module-level {target!r} is mutable", subject=subject,
                                   code="manifest.mutable_global",
                                   hint="keep it on the instance, or in a state unit if it must survive a reload; "
                                        "constants can be tuples, frozensets or MappingProxyType"))
        for node in ast.walk(mod):
            if isinstance(node, ast.Global):
                errors.append(diag(COMPONENT, "rules", f"{module}: 'global {', '.join(node.names)}'",
                                   subject=subject, code="manifest.mutable_global"))
        captures = uid is not None and tree.units[uid].manifest.capabilities.captures_graph
        if uid is not None and not captures and _calls_graph(mod):
            errors.append(diag(COMPONENT, "rules", f"{module} calls platform.graph() but {uid} is not a plan unit",
                               subject=uid, code="manifest.graph_outside_plan",
                               hint="only units with capabilities.captures_graph record graphs"))


def _assignment(node: ast.stmt) -> tuple[str | None, ast.expr | None]:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id, node.value
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id, node.value
    return None, None


def _mutable(value: ast.expr) -> bool:
    if isinstance(value, ast.List | ast.Dict | ast.Set | ast.ListComp | ast.DictComp | ast.SetComp):
        return True
    if isinstance(value, ast.Call):
        func = value.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        return name in MUTABLE_CALLS
    return False


def _calls_graph(mod: ast.Module) -> bool:
    for node in ast.walk(mod):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "graph":
            owner = node.func.value
            if (isinstance(owner, ast.Attribute) and owner.attr == "platform") or \
                    (isinstance(owner, ast.Name) and owner.id == "platform"):
                return True
    return False


def _natives(tree: Tree, errors: list[Diagnostic], warnings: list[Diagnostic]) -> None:
    for uid, s in sorted(tree.units.items()):
        declared = set(s.manifest.capabilities.native_extensions)
        present = {p.relative_to(tree.root).as_posix() for p in s.directory.rglob("*")
                   if p.is_file() and p.name.endswith(NATIVE_SUFFIXES)}
        for path in sorted(present - declared):
            errors.append(diag(COMPONENT, "native", f"{path} is not in capabilities.native_extensions", subject=uid,
                               code="manifest.undeclared_native",
                               hint="a native module cannot be unloaded; declared ones restart the process"))
        for path in sorted(declared - present):
            warnings.append(diag(COMPONENT, "native", f"declares {path}, which is not in the tree", subject=uid,
                                 code="manifest.missing_native", severity="warning"))

