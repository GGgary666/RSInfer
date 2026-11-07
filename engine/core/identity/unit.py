"""``core.identity``: a name for "this system" (doc/specs/control-plane.md §2).

The identity is the hash of what the system is: file contents (the units'
``unit.toml`` among them), config, the dependency lock, artifacts (weights
manifests), the topology, and the platform every host runs. Candidate identity =
f(base manifest, change-set) is deterministic: the manifest is encoded with its
keys sorted, so the order in which a change-set lists things does not matter.
"""

from __future__ import annotations

import hashlib
from typing import Any

import msgspec

from core.change.state import IdentityManifest
from core.common.ids import Identity
from platforms.base import PlatformInfo


class _Named(msgspec.Struct, frozen=True):
    manifest: IdentityManifest
    platform: PlatformInfo | None


class IdentityFunction:
    """``ctrl.identity@1``."""

    def __init__(self) -> None:
        self._encode = msgspec.msgpack.Encoder(order="sorted").encode

    def identity(self, manifest: IdentityManifest, platform: PlatformInfo | None = None) -> Identity:
        return Identity(hashlib.sha256(self._encode(_Named(manifest, platform))).hexdigest()[:16])

    def diff(self, a: IdentityManifest, b: IdentityManifest) -> dict[str, Any]:
        """What differs from ``a`` to ``b``: the change-set between two identities."""
        paths = sorted(set(a.files) | set(b.files))
        keys = sorted(set(a.config) | set(b.config))
        return {
            "files": {p: (a.files.get(p), b.files.get(p)) for p in paths if a.files.get(p) != b.files.get(p)},
            "config": {k: (a.config.get(k), b.config.get(k)) for k in keys if a.config.get(k) != b.config.get(k)},
            "deps": {k: (a.deps.get(k), b.deps.get(k)) for k in sorted(set(a.deps) | set(b.deps))
                     if a.deps.get(k) != b.deps.get(k)},
            "artifacts": {k: (a.artifacts.get(k), b.artifacts.get(k))
                          for k in sorted(set(a.artifacts) | set(b.artifacts))
                          if a.artifacts.get(k) != b.artifacts.get(k)},
            "topology": a.topology != b.topology,
        }


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        ctx.provide("ctrl.identity@1", IdentityFunction())
