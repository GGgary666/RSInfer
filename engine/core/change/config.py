"""Tunables of change transactions (``change.*`` config items), shared by both sides."""

from __future__ import annotations

import msgspec


class ChangeTunables(msgspec.Struct, frozen=True):
    coordinator_timeout_s: float = 60.0
    """A participant that hears nothing from the coordinator for this long asks it (§4.7)."""
    keepalive_interval_s: float = 10.0
    """The coordinator renews every participant's lease this often while a change runs."""
    prepare_timeout_s: float = 600.0
    """How long the coordinator waits for ``Prepared`` (sliced prepares take a while)."""
    drain_timeout_s: float = 10.0
    commit_retry_interval_s: float = 1.0
    """Pause between rounds of ``Commit`` to participants that did not answer yet."""
    blob_retention: int = 20
    """Blobs of this many recent identities survive ``Coordinator.prune_blobs``."""
