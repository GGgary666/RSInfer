# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""``core.state.weights``: the weights of a worker (``state.weights@1``, stage2-contract.md §7).

Adapted from SGLang's model loading (``DefaultModelLoader``, ``DummyModelLoader``,
srt/model_loader/weight_utils.py, e838360; LICENSE.sglang). SGLang's parameters live in the
model; here they live in this state unit, so a reload of the model keeps them. The model declares
its parameters and registers its ``load_weights`` (the stacked-parameter mapping); ``load`` feeds
it ``safetensors_weights_iterator`` over the checkpoint, or fills dummy weights
(``initialize_dummy_weights``). ``refit`` loads a checkpoint again into the same tensors.

Asleep (``holder.sleep``), the declared tensors are on the meta device: they have shapes and no
memory, nothing is loaded, and nothing may run on them (the scheduler refuses requests, and
``core.plan`` records no graph). Waking reloads this unit, which loads the checkpoint again, as
vLLM's level-2 sleep does; contents a ``refit`` put there are gone then.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import safetensors
import torch

from core.contracts.model import WeightDecl, WeightLoader
from unit.runtime.registry import Registry


META_DTYPES = MappingProxyType({"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32, "f64": torch.float64,
                                "i8": torch.int8, "u8": torch.uint8, "i32": torch.int32, "i64": torch.int64})


def safetensors_files(path: str) -> list[str]:
    """``DefaultModelLoader._prepare_weights`` for a local path: the file itself, or every
    ``*.safetensors`` of the directory but ``consolidated.safetensors`` when others exist."""
    if os.path.isfile(path):
        return [path]
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    if not files:
        raise FileNotFoundError(f"no *.safetensors in {path}")
    return files


def safetensors_weights_iterator(hf_weights_files: Sequence[str]) -> Iterator[tuple[str, torch.Tensor]]:
    """Iterate over the weights in the model safetensor files."""
    for st_file in hf_weights_files:
        with safetensors.safe_open(st_file, framework="pt", device="cpu") as f:  # pyright: ignore[reportUnknownMemberType]
            for name in f.keys():  # noqa: SIM118
                yield name, f.get_tensor(name)


def initialize_dummy_weights(params: Mapping[str, torch.Tensor], low: float = -1e-3, high: float = 1e-3,
                             seed: int = 1234) -> None:
    """Values in ``[low, high)`` from one seed per parameter, so they depend only on its size and
    dtype, not on the order or the partitioning."""
    for param in params.values():
        if torch.is_floating_point(param):
            generator = torch.Generator(device=param.device)
            generator.manual_seed(seed)
            if torch.finfo(param.dtype).bits < 16:
                tmp_param = param.to(torch.float16).uniform_(low, high, generator=generator)
                param.copy_(tmp_param.to(param.dtype))
            else:
                param.uniform_(low, high, generator=generator)


@dataclass(frozen=True)
class Views:
    """``Binding``: the declared tensors under a prefix, with the prefix taken off."""

    store: WeightStore
    prefix: str

    @property
    def generation(self) -> int:
        return self.store.generation

    def buffers(self) -> Mapping[str, torch.Tensor]:
        p = self.prefix
        return {name[len(p):]: t for name, t in self.store.tensors.items() if name.startswith(p)}


@dataclass
class WeightStore:
    """``state.weights@1``."""

    platform: Any
    path: str
    seed: int
    decls: dict[str, WeightDecl] = field(default_factory=dict[str, WeightDecl])
    tensors: dict[str, torch.Tensor] = field(default_factory=dict[str, torch.Tensor])
    buffers_: dict[str, Any] = field(default_factory=dict[str, Any])
    filled: set[str] = field(default_factory=set[str])
    loaders: Registry[WeightLoader] = field(default_factory=lambda: Registry[WeightLoader]("weights.loaders"))
    generation: int = 0
    loads: int = 0
    loaded_from: str = ""
    source: str = ""
    """What ``fill`` last read: a checkpoint path, or ``""`` for the dummy weights."""
    asleep: bool = False
    """``holder.sleep``: declared tensors take no memory (the meta device), and nothing is loaded."""

    @property
    def version(self) -> str:
        """Which contents: ``"<n>:<path>"`` after the ``n``-th load, ``"<n>:dummy:<seed>"`` for dummy ones."""
        return f"{self.loads}:{self.loaded_from}" if self.loads else ""

    def declare(self, decls: Sequence[WeightDecl]) -> None:
        moved = False
        existing = bool(self.decls)
        decls = [d for d in decls if self.decls.get(d.name) != d]
        for d in decls:
            if d.name in self.buffers_:
                self.platform.free(self.buffers_.pop(d.name))
                moved = True
        if decls and not self.asleep:
            # As the state stores do: memory freed before (this store's, or that of the instances a
            # change unloaded) stays in PyTorch's cache, and these allocations would be cut out of
            # its blocks. A rollback after a failed reload then had the old weights pin the blocks
            # the failed attempt freed, and the old state pools no longer fit
            # (doc/framework-bugs/2026-11-13-a-weight-dtype-change-runs-out-of-memory-and-its-rollback-is-unconfirmed.md).
            torch.cuda.empty_cache()
        for d in decls:
            moved |= existing
            if self.asleep:
                self.tensors[d.name] = torch.empty(d.shape, dtype=META_DTYPES[d.format], device="meta")
            else:
                buf = self.platform.alloc(0, 0, shape=d.shape, dtype=d.format)
                self.buffers_[d.name], self.tensors[d.name] = buf, self.platform.tensor(buf)
            self.decls[d.name] = d
            self.filled.discard(d.name)
        self.generation += moved

    def register(self, loader: WeightLoader) -> Callable[[], None]:
        return self.loaders.register(loader)

    def load(self) -> None:
        if self.asleep or set(self.decls) <= self.filled:
            return
        self.fill(self.path)

    def fill(self, path: str) -> None:
        """Every declared tensor from the checkpoint at ``path``, or dummy weights for ``""``."""
        if self.asleep:
            raise RuntimeError("the weights are asleep (holder.sleep): wake the engine first")
        if path:
            loaders = self.loaders.items()
            if len(loaders) != 1:
                raise RuntimeError(f"weights need exactly one registered loader, got {len(loaders)}")
            loaded = loaders[0](safetensors_weights_iterator(safetensors_files(path)), self.tensors)
            missing = sorted(set(self.decls) - loaded)
            if missing:
                raise RuntimeError(f"the checkpoint at {path} has no weights for {missing[:8]}")
            self.loaded_from = path
        else:
            initialize_dummy_weights(self.tensors, seed=self.seed)
            self.loaded_from = f"dummy:{self.seed}"
        self.source = path
        self.filled = set(self.decls)
        self.loads += 1

    def bind(self, prefix: str) -> Views:
        return Views(self, prefix)

    def free(self) -> None:
        self.generation += 1
        for buf in self.buffers_.values():
            self.platform.free(buf)
        self.buffers_.clear()
        self.tensors.clear()


class Refit:
    """``refit``: new contents from a checkpoint, same tensors (so no graph records again).
    ``args``: ``{"path": ...}``, default the configured path. Prepare checks the path; execute
    loads it in place and keeps no copy; abort loads the previous contents again from where they
    came from (their checkpoint, or the dummy weights of the same seed). SGLang's
    ``update_weights_from_disk`` (model_runner_components/weight_updater.py) rolls back the same
    way, by loading again, so a refit needs no second copy of the weights in device memory."""

    def __init__(self, store: WeightStore) -> None:
        self.store = store
        self.path = ""
        self.previous: tuple[str, int, str] | None = None

    def prepare(self, args: dict[str, Any]) -> None:
        self.path = str(args.get("path", self.store.path))
        if self.store.asleep:
            raise RuntimeError("the weights are asleep (holder.sleep): wake the engine first")
        safetensors_files(self.path)

    def execute(self, args: dict[str, Any]) -> None:
        s = self.store
        self.previous = (s.source, s.loads, s.loaded_from)
        s.fill(self.path)

    def commit(self) -> None:
        self.previous = None

    def abort(self) -> None:
        if self.previous is not None:
            source, loads, loaded_from = self.previous
            self.store.fill(source)
            self.store.loads, self.store.loaded_from = loads, loaded_from
            self.previous = None


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        self.store = WeightStore(ctx.platform, config["weights.path"], config["weights.dummy_seed"],
                                 asleep=config["holder.sleep"])
        self.model = {"model.path": config["model.path"], "model.hf_config": config["model.hf_config"],
                      "model.dtype": config["model.dtype"]}
        """The model config the parameters are declared from (and the state dtypes' ``auto``). Read here
        so that changing it reloads this unit, which ``core.deps`` refuses in place (§5.5.5)."""
        ctx.effect(lambda: self.store.free, label="weights")
        ctx.provide("state.weights@1", self.store)
        ctx.on_action("refit", Refit(self.store))
