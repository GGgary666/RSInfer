"""Tensors and the buffers a unit hands to others, in the worker (stage2-contract.md §5.2,
§9.3). Used by the storage units, the weight store, the backends, the persistent batch
and the plan.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

type Tensor = Any
"""A ``torch.Tensor`` on ``torch_cpu`` and ``cuda_sm100``; never constructed by the contracts."""


class Binding(Protocol):
    """Buffers a unit hands to others. ``generation`` moves whenever any of them changes
    address, shape or dtype, so a graph checks one integer per provider before replay
    instead of every tensor (GPU probes, proposal 1). Same address, new contents: no change."""

    @property
    def generation(self) -> int: ...
    def buffers(self) -> Mapping[str, Tensor]: ...
