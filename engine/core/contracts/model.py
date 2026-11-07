"""Weights and the model's assembly (stage2-contract.md §7, v0-plan 5.10).

A logical tensor is a format (a plugin: BF16, FP8 block-quantized plus scales, ...) plus a
placement (data: how the mesh splits it). Layers never own weights: they get views from the
weight store, so a refit changes contents in place and no graph needs recording again.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

import msgspec

from core.contracts.state import Binding, StateDecl


class Placement(msgspec.Struct, frozen=True):
    split: Literal["replicated", "rows", "columns"] = "replicated"
    """How TP splits the tensor across the ``tp`` mesh dimension."""
    axis_size: int = 0
    """For ``rows`` / ``columns``: the full size of the split axis, for the shard bounds."""


class WeightDecl(msgspec.Struct, frozen=True):
    name: str
    """Logical name, e.g. ``layers.3.attn.q_proj.weight``."""
    shape: tuple[int, ...]
    """Full (unsplit) logical shape."""
    format: str = "bf16"
    placement: Placement = Placement()
    source: str = ""
    """Checkpoint tensor name if it differs (fused or renamed); resolved by the model unit."""


class WeightStore(Protocol):
    """``state.weights@1`` in every worker: loads from safetensors (paths resolved per
    machine), addresses by logical name, carries a version; ``refit`` replaces contents in
    place, same layout."""

    @property
    def version(self) -> str: ...

    def declare(self, decls: Sequence[WeightDecl]) -> None: ...
    def bind(self, prefix: str) -> Binding:
        """Views of every declared tensor under ``prefix``, this rank's shard."""
        ...


class Assembly(msgspec.Struct, frozen=True):
    """What a model unit (``models.qwen38``) declares at start: the layer units it uses,
    what each layer stores, and every weight."""

    layers: list[str]
    """One entry per layer: the layer kind's service, e.g. ``layers.gdn@1``, ``layers.attn@1``."""
    states: list[StateDecl]
    weights: list[WeightDecl]
    config: Mapping[str, int | float | str] = {}
    """The checkpoint config values the layers read (hidden size, head counts, ...)."""
