"""What an identity names, and how a change-set moves from one to the next.

``core.identity`` (a control-plane unit) owns the full identity definition:
platform info, weights, kernel bindings, ... Until it exists, the coordinator
uses this default: the hash of the files, config, dependency lock, topology and
artifacts. Either way, candidate identity = f(base identity, change-set).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import msgspec

from core.change.types import ChangeSet
from core.common.ids import Identity
from core.supervisor.topology import Topology

_canonical = msgspec.msgpack.Encoder(order="sorted")


class IdentityManifest(msgspec.Struct, frozen=True):
    files: dict[str, str] = {}
    config: dict[str, Any] = {}
    deps: dict[str, str] = {}
    topology: Topology | None = None
    artifacts: dict[str, str] = {}


def apply_changeset(base: IdentityManifest, cs: ChangeSet) -> IdentityManifest:
    files = dict(base.files)
    for path, h in cs.files.items():
        if h is None:
            files.pop(path, None)
        else:
            files[path] = h
    return IdentityManifest(
        files=files,
        config={**base.config, **cs.config},
        deps={**base.deps, **cs.deps},
        topology=cs.topology if cs.topology is not None else base.topology,
        artifacts={**base.artifacts, **cs.artifacts},
    )


def encode_manifest(m: IdentityManifest) -> bytes:
    return _canonical.encode(m)


def decode_manifest(data: bytes) -> IdentityManifest:
    return msgspec.msgpack.decode(data, type=IdentityManifest)


def identity_of(m: IdentityManifest) -> Identity:
    return Identity(hashlib.sha256(encode_manifest(m)).hexdigest()[:16])


type IdentityFn = Callable[[IdentityManifest], Identity]
