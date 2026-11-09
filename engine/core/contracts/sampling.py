"""Sampling parameters: the frontend gives them with a request (``core.contracts.requests``),
the scheduler passes them on to the worker's persistent batch (``core.contracts.records``).
"""

from __future__ import annotations

import msgspec


class SamplingParams(msgspec.Struct, frozen=True):
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int | None = None
    """Set by the scheduler at admission when the client gave none. The worker draws from
    (seed, position) and keeps no random state: the same tokens however requests are batched,
    and a rebuilt persistent batch samples what the old one would have (§4.3)."""
