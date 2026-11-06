"""Framework tunables are config items, never constants (doc/specs/README.md, "可调参数").

Each component declares a ``Tunables`` struct whose fields are its parameters
with defaults; keys are ``"<component>.<field>"``, e.g. ``"ipc.peer_timeout_s"``.
"""

from __future__ import annotations

from typing import Any

import msgspec


def load_tunables[T: msgspec.Struct](cls: type[T], component: str, values: dict[str, Any] | None = None) -> T:
    """Build ``cls`` from the ``component.*`` entries of ``values``; unknown keys are errors."""
    prefix = component + "."
    mine = {k[len(prefix):]: v for k, v in (values or {}).items() if k.startswith(prefix)}
    unknown = set(mine) - set(cls.__struct_fields__)
    if unknown:
        raise KeyError(f"unknown {component} tunables: {sorted(prefix + k for k in unknown)}")
    return msgspec.convert(mine, cls, strict=True)


def tunable_keys(cls: type[msgspec.Struct], component: str) -> list[str]:
    return [f"{component}.{f}" for f in cls.__struct_fields__]
