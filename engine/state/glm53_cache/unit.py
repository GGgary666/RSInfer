"""Transfer quiesced RSInfer pages and KDA slots to an independent MemCache pool."""

from __future__ import annotations

import hashlib
import os
import sys
import time
from collections.abc import Iterable
from contextlib import ExitStack
from importlib import import_module
from pathlib import Path
from typing import Any

import msgspec
import torch

from models.glm53_config import KDA, KV
from state.prefix_checkpoint import PrefixCheckpoint

COPY_L2G, COPY_G2L = 0, 1
TRANSFER_BATCH = 32


def signature(state: object, code: str, checkpoint: str) -> str:
    source = Path(checkpoint)
    shards = sorted(source.glob("*.safetensors")) if source.is_dir() else [source]
    files = []
    for f in shards:
        stat = f.stat()
        files.append((f.name, stat.st_size, stat.st_mtime_ns))
    return hashlib.sha256(msgspec.json.encode(("glm53-native-bf16-v1", state, code, files))).hexdigest()[:24]


class Cache:
    def __init__(self, ctx: Any, config: Any) -> None:
        self.kv = ctx.use("state.kv.paged@1")
        self.kda = ctx.use("state.gdn.slots@1")
        self.weights = ctx.use("state.weights@1")
        self.group = ctx.use("comm.group@1")
        self.platform = ctx.platform
        self.client_config = config["glm53.memcache_config"]
        # A manifest seeds a new worker. A hot change can have freed or reused its old page IDs.
        self.path = config["alloc.prefix_checkpoint"] if ctx.instance.generation == 1 else ""
        self.asleep = config["holder.sleep"]
        self.model_state: object = None
        self.layer_code = ""
        self.store: Any = None
        self.restored = False

    def bind(self, declarations: object, config: object, model_code: str, layer_code: str) -> None:
        self.model_state = (declarations, config, model_code)
        self.layer_code = layer_code
        if not self.asleep and self.path and not self.restored:
            self.transfer(PrefixCheckpoint.load(Path(self.path)), write=False)
            self.restored = True

    def connect(self) -> Any:
        if self.store is None:
            if not self.client_config or not Path(self.client_config).is_file():
                raise ValueError("GLM MemCache needs --memcache-client-config")
            os.environ["MMC_LOCAL_CONFIG_PATH"] = self.client_config
            store = import_module("memcache_hybrid").DistributedObjectStore()
            if (code := store.init(0)) != 0:
                raise RuntimeError(f"MemCache client initialization failed ({code})")
            self.store = store
        return self.store

    def layout(self, kind: str) -> tuple[list[torch.Tensor], list[tuple[int, int]]]:
        """Registration buffers and (base address, item stride bytes) in layer order."""
        if kind == KV:
            buffers: list[torch.Tensor] = []
            spans: list[tuple[int, int]] = []
            for layer, d in sorted(self.kv.decls.items()):
                for name, tensor in sorted(self.kv.bind(layer).buffers().items()):
                    stride = d.tensors[name][0][0] * tensor.stride(0) * tensor.element_size()
                    buffers.append(tensor)
                    spans.append((tensor.data_ptr(), stride))
            return buffers, spans
        conv, temporal, _ = self.kda.pools()
        spans = [(t[i].data_ptr(), t.stride(1) * t.element_size())
                 for i in range(conv.shape[0]) for t in (conv, temporal)]
        return [conv, temporal], spans

    def items(self, checkpoint: PrefixCheckpoint) -> Iterable[tuple[str, list[int]]]:
        if (checkpoint.world != self.group.world_size or checkpoint.page != self.kv.page_size
                or checkpoint.slot_kind != KDA or len(checkpoint.domains) != self.group.world_size // self.group.size):
            raise ValueError("MemCache checkpoint mesh, page size, or state layout changed")
        domain = self.group.world_rank // self.group.size
        nodes = checkpoint.domains[domain]
        yield KV, sorted({p for n in nodes for p in n.pages})
        yield KDA, sorted({n.slot for n in nodes if n.slot is not None})

    def transfer(self, checkpoint: PrefixCheckpoint, *, write: bool) -> None:
        started = time.monotonic()
        store = self.connect()
        if self.model_state is None or self.asleep:
            raise RuntimeError("MemCache requires the awake GLM worker's allocated state")
        fingerprint = signature(self.model_state, self.layer_code, self.weights.source)
        prefix = f"rsinfer.glm53.{checkpoint.nonce}.{fingerprint}.r{self.group.world_rank}"
        self.platform.synchronize(0)
        transferred = 0
        for kind, items in self.items(checkpoint):
            if not items:
                continue
            capacity = self.kv.capacity() if kind == KV else self.kda.capacity()
            if items[0] < 0 or items[-1] >= capacity:
                raise ValueError(f"MemCache {kind} checkpoint exceeds worker capacity")
            buffers, spans = self.layout(kind)
            sizes = [size for _, size in spans]
            with ExitStack() as registration:
                for t in buffers:
                    ptr, size = t.data_ptr(), t.numel() * t.element_size()
                    if (code := store.register_buffer(ptr, size)) != 0:
                        raise RuntimeError(f"MemCache buffer registration failed ({code})")
                    registration.callback(store.unregister_buffer, ptr, size)
                for start in range(0, len(items), TRANSFER_BATCH):
                    ids = items[start:start + TRANSFER_BATCH]
                    keys = [f"{prefix}.{kind}.{i}" for i in ids]
                    pointers = [[base + (i + 1) * size for base, size in spans] for i in ids]
                    lens = [sizes] * len(ids)
                    result = (store.batch_put_from_layers(keys, pointers, lens, COPY_L2G) if write
                              else store.batch_get_into_layers(keys, pointers, lens, COPY_G2L))
                    if len(result) != len(ids) or any(code != 0 for code in result):
                        raise RuntimeError(f"MemCache transfer failed for {kind}: {result}")
                    transferred += len(ids)
        mode = "put" if write else "get"
        print(f"state.glm53_cache: {mode} {transferred} objects, rank {self.group.world_rank}, "
              f"{time.monotonic() - started:.3f}s", file=sys.stderr, flush=True)

    def prepare(self, args: dict[str, Any]) -> None:
        if not self.client_config:
            raise ValueError("GLM MemCache needs --memcache-client-config")

    def execute(self, args: dict[str, Any]) -> None:
        path = Path(args["path"])
        until = time.monotonic() + float(args["timeout"])
        while not path.is_file():
            if time.monotonic() >= until:
                raise TimeoutError("scheduler prefix checkpoint was not published")
            time.sleep(0.02)
        self.transfer(PrefixCheckpoint.load(path), write=True)

    def commit(self) -> None:
        pass

    def abort(self) -> None:
        pass


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        cache = Cache(ctx, config)
        ctx.provide("state.glm53_cache@1", cache)
        ctx.on_action("save_prefix", cache)
