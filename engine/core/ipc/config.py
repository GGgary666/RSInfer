"""Tunables of ``core.ipc`` (keys ``ipc.<field>``)."""

from __future__ import annotations

import msgspec


class IpcTunables(msgspec.Struct, frozen=True):
    rpc_timeout_s: float = 5.0
    lookup_retry_s: float = 0.1
    """How often a data-channel writer asks the supervisor again for a reader it has not found."""
    heartbeat_interval_s: float = 0.1
    """How often each end of a same-host step stream proves it is alive."""
    peer_timeout_s: float = 2.0
    """A step-stream peer silent for this long is reported lost."""
    stream_hwm: int = 64
    """Records in flight per TCP step stream before the producer blocks."""
    stream_socket_buffer: int = 1 << 16
    shm_capacity: int = 1 << 22
    """Bytes in a same-host step-stream ring."""
    stream_hold_bytes: int = 1 << 26
    """What a replaceable TCP reader holds of a later producer while the current one has not
    ended; more means two producers write at once."""
    close_timeout_s: float = 10.0
    """How long an orderly close waits for the consumer to take what is queued."""
