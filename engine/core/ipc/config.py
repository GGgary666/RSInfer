"""Tunables of ``core.ipc`` (keys ``ipc.<field>``)."""

from __future__ import annotations

import msgspec


class IpcTunables(msgspec.Struct, frozen=True):
    rpc_timeout_s: float = 5.0
    heartbeat_interval_s: float = 0.1
    """How often each end of a same-host step stream proves it is alive."""
    peer_timeout_s: float = 2.0
    """A step-stream peer silent for this long is reported lost."""
    stream_hwm: int = 64
    """Records in flight per TCP step stream before the producer blocks."""
    stream_socket_buffer: int = 1 << 16
    shm_capacity: int = 1 << 22
    """Bytes in a same-host step-stream ring."""
    close_timeout_s: float = 10.0
    """How long an orderly close waits for the consumer to take what is queued."""
