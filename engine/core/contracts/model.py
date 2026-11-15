"""Weights and the model's assembly (stage2-contract.md §7, v0-plan 5.10).

A logical tensor is a format (a plugin: BF16, FP8 block-quantized plus scales, ...) plus a
placement (data: how the mesh splits it). Layers never own weights: they get views from the
weight store, so a refit changes contents in place and no graph needs recording again.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, Protocol

import msgspec

from core.contracts.state import StateDecl
from core.contracts.tensor import Binding, Tensor

type WeightLoader = Callable[[Iterable[tuple[str, Tensor]], Mapping[str, Tensor]], set[str]]
"""A model's ``load_weights`` (SGLang's): checkpoint ``(name, tensor)`` pairs in, written into the
declared tensors (name -> tensor), fused or renamed as the model's parameters are; returns the
names it loaded."""


class Placement(msgspec.Struct, frozen=True):
    split: Literal["replicated", "rows", "columns"] = "replicated"
    """How TP splits the tensor across the ``tp`` mesh dimension."""


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
    def register(self, loader: WeightLoader) -> Callable[[], None]:
        """The model's loader (``ctx.register``), used by ``load`` and ``refit``."""
        ...
    def load(self) -> None:
        """Fill every declared tensor not yet filled: from the checkpoint through the loader, or
        dummy weights without one. Contents survive a reload of the model."""
        ...
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


class Model(Protocol):
    """``model@1``: the forward of SGLang's ``*ForCausalLM`` with its ``LogitsProcessor``. The
    backends of ``assembly.layers`` have prepared the step (``Backend.prepare``) before it runs."""

    @property
    def assembly(self) -> Assembly: ...

    @property
    def backends(self) -> Mapping[str, Any]:
        """The backends of ``assembly.layers`` by their service names (``Backend``, plan.py): what the
        plan prepares before a step and records graphs over."""
        ...

    @property
    def stores(self) -> Mapping[str, Any]:
        """The stores of ``assembly.states`` by state kind (``StateStore``, storage.py): what the
        executor clears, copies and reports the capacity of."""
        ...

    def forward(self, input_ids: Tensor, positions: Tensor, last_index: Tensor) -> Tensor:
        """``input_ids``, ``positions`` ``[tokens]``; logits ``[rows, vocab]`` at ``last_index``,
        each row's last token."""
        ...


class DraftModel(Model, Protocol):
    """``model@1`` with speculation on (doc/specs/mtp.md): the target model, and its MTP layer as the
    draft model (SGLang's ``Qwen3_5ForCausalLMMTP``), which shares its embedding and LM head and
    keeps its KV in the same pages, as layer ``draft_layer``."""

    @property
    def draft_layer(self) -> int: ...

    @property
    def hidden_size(self) -> int: ...

    @property
    def dtype(self) -> str:
        """The activations' dtype (``model.dtype``): the draft model's hidden states are in it."""
        ...

    def forward_hidden(self, input_ids: Tensor, positions: Tensor, last_index: Tensor) -> tuple[Tensor, Tensor]:
        """``forward``, and every token's hidden state after the final norm ``[tokens, hidden]``
        (what SGLang's ``LogitsProcessor`` captures for the draft model)."""
        ...

    def draft_forward(self, input_ids: Tensor, positions: Tensor, hidden: Tensor, last_index: Tensor
                      ) -> tuple[Tensor, Tensor]:
        """The draft model on ``input_ids`` with ``hidden`` ``[tokens, hidden]``, the target's hidden
        states or its own last ones: logits ``[rows, vocab]`` and its hidden state after its final
        norm ``[rows, hidden]``, both at ``last_index``. Of the backends only ``layers.attn@1`` is
        prepared for it. ``hidden`` is given up: a norm may write over it."""
        ...
