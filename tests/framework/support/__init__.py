"""Helpers for building toy units and snapshot directories in tests."""

from __future__ import annotations

import json
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core.common.ids import HostId, ProcId
from core.common.manifest import parse_manifest
from unit.runtime import UnitSpec

events: list[tuple[str, str]] = []
"""Shared log that toy units append to; not an engine module, never purged."""

PROC = ProcId(HostId("host-a"), "worker", 0)


def _toml_list(items: list[str]) -> str:
    return "[" + ", ".join(f'"{i}"' for i in items) + "]"


def manifest_toml(
    unit_id: str,
    entry: str,
    *,
    provides: Sequence[str] = (),
    requires: Sequence[str] = (),
    optional: Sequence[str] = (),
    kind: str = "logic",
    process: Sequence[str] = ("worker",),
    config: dict[str, dict[str, Any]] | None = None,
    reads: Sequence[str] = (),
    capabilities: str = "",
    build: str = "",
) -> str:
    lines = [
        "[unit]",
        f'id = "{unit_id}"',
        f'kind = "{kind}"',
        f"process = {_toml_list(list(process))}",
        f'entry = "{entry}"',
        *([f'build = "{build}"'] if build else []),
        "[provides]",
        f"services = {_toml_list(list(provides))}",
        "[requires]",
        f"services = {_toml_list(list(requires))}",
        f"optional = {_toml_list(list(optional))}",
        "[reads]",
        f"config = {_toml_list(list(reads))}",
    ]
    if config:
        lines.append("[config]")
        for key, d in config.items():
            default = d["default"]
            lit = json.dumps(default)
            lines.append(f'"{key}" = {{ type = "{d["type"]}", default = {lit}, binding = "{d.get("binding", "live")}" }}')
    if capabilities:
        lines.append("[capabilities]")
        lines.append(capabilities)
    return "\n".join(lines) + "\n"


def spec(unit_id: str, entry: str, **kw: Any) -> UnitSpec:
    return UnitSpec(parse_manifest(manifest_toml(unit_id, entry, **kw)).unwrap())


def write_tree(root: Path, files: dict[str, str]) -> Path:
    """Write ``{"pkg/mod.py": code}`` under ``root``; package dirs get an ``__init__.py``."""
    for rel, code in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(code))
        d = path.parent
        while d != root:
            init = d / "__init__.py"
            if not init.exists():
                init.write_text("")
            d = d.parent
    return root


GENERIC_UNIT = '''
from tests.framework.support import events


class Service:
    def __init__(self, name, owner):
        self.name = name
        self.owner = owner

    def ping(self):
        return (self.name, str(self.owner))


class Unit:
    def __init__(self, ctx, config):
        self.ctx = ctx
        self.used = {name: ctx.use(name) for name in ctx.manifest.requires.services}
        for name in ctx.manifest.provides.services:
            ctx.provide(name, Service(name, ctx.instance))
        events.append(("load", ctx.id))
        ctx.effect(lambda: (lambda: events.append(("unload", ctx.id))), "log")
'''
"""A unit that uses all its required services and provides all declared ones."""
