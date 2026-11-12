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
    execute_timeout_s: float = 3600.0
    """How long the coordinator waits for ``Executed``, and for the ``Abort`` that follows a change
    that failed in or after execute. The reload at the safe point may read a checkpoint (waking
    from ``holder.sleep``, a change to the weight store) and record CUDA graphs: about 30 s for
    a 64-layer model on a SM100, more than ``supervisor.rpc_timeout_s``."""
    build_stop_grace_s: float = 5.0
    """An undo stops a running build step with SIGTERM, then SIGKILLs what is left after this."""
    drain_timeout_s: float = 10.0
    commit_retry_interval_s: float = 1.0
    """Pause between attempts of ``Commit``/``Resume`` to participants that did not confirm yet."""
    commit_attempts: int = 3
    """Maximum attempts per Commit/Resume confirmation phase."""
    commit_ack_timeout_s: float = 120.0
    """Total time allowed for Commit and Resume confirmation, including RPC waits."""
    blob_retention: int = 20
    """Blobs of this many recent identities survive ``Coordinator.prune_blobs``."""
    blob_grace_s: float = 3600.0
    """Blobs younger than this are never pruned: agents upload before they submit."""
    blob_prune_interval_s: float = 300.0
    """Between changes, blobs are pruned at most this often (§5.1)."""
    blob_prune_batch: int = 500
    """Blobs deleted per batch; the event loop runs between batches, and a queued change goes first."""
