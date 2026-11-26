"""``worker.comm``: this worker's place in its communication group (``comm.group@1``;
doc/specs/v1-ds1017.md stage 3). The ranks of a group compute each step together: with expert
parallelism every rank holds part of each layer's experts, and a token's experts are run by the ranks
that hold them.

It is ``torch.distributed``, set up as DeepInfer does (``worker/base.py``,
``parallel_utils/parallel_state.py``, a679bb8): the default group over gloo, found through
``comm.rendezvous``, with a second group for NCCL or HCCL tensors on the workers' devices.
Device kernels can use that ``ProcessGroup`` (``Group.device_group``).

A process joins once. ``torch.distributed``'s default group is the process's, not this unit's: a
reload of this unit finds it joined and keeps it, and a change of the group's size or rendezvous
needs new processes. The group for the cards is this unit's: made when it loads, given up when it
unloads; a change reloads this unit on every rank at once, so they make the new one together. A rank
that dies takes the group with it: the others' next step fails, and the engine has to be started
again.

``Group.gather`` and ``Group.reduce`` are the two collectives the reference kernels are written
with (``kernels.ds1017_ep``). Ranks in a group call them in the same order. A model with experts
shared across scheduling domains also takes part on ranks without rows (``model.idle``).
Independent groups use local rank numbers; the scheduler and checkpoint use the world ranks.
"""

from __future__ import annotations

import datetime
import math
import os
from collections.abc import Sequence
from importlib import import_module
from types import MappingProxyType
from typing import Any

import torch
import torch.distributed as dist

DEVICE_TYPES = MappingProxyType({"nccl": "cuda", "hccl": "npu"})
BACKENDS = ("gloo", *DEVICE_TYPES)


class Group:
    """``comm.group@1``."""

    def __init__(self, size: int = 1, rank: int = 0, device_group: Any = None, device_type: str = "cuda",
                 *, host_group: Any = None, world_size: int | None = None, world_rank: int | None = None) -> None:
        self.size, self.rank = size, rank
        self.world_size = size if world_size is None else world_size
        self.world_rank = rank if world_rank is None else world_rank
        self.host_group = host_group
        self.device_group = device_group
        self.prefix_group: Any = None
        self.device_type = device_type
        """Tensors of this device type use the device group when one is configured."""
        self.collectives = 0
        """How many ``gather`` and ``reduce`` calls this rank made: equal within its group after a step."""

    def gather(self, tensors: Sequence[torch.Tensor]) -> tuple[list[torch.Tensor], list[int]]:
        """Each tensor ``[n, ...]`` of this rank with the other ranks' (each rank its own ``n``, 0
        too) as ``[sum of n, ...]``, in rank order; and every rank's ``n``."""
        n = tensors[0].shape[0]
        if self.size == 1:
            return list(tensors), [n]
        self.collectives += 1
        counts = self._all_gather(torch.tensor([n], dtype=torch.int64)).tolist()
        width = max(counts)
        out: list[torch.Tensor] = []
        for t in tensors:
            if t.shape[0] != n:
                raise ValueError("the tensors gathered together have one row count")
            # As bytes: the collectives of gloo take few dtypes (no bf16). A rank without rows knows
            # a row's size too.
            size = t.element_size() * math.prod(t.shape[1:])
            padded = torch.zeros(width, size, dtype=torch.uint8, device=t.device)
            padded[:n] = t.contiguous().view(torch.uint8).reshape(n, size)
            every = self._all_gather(padded)
            kept = torch.cat([every[r * width:r * width + c] for r, c in enumerate(counts)])
            out.append(kept.view(t.dtype).reshape(sum(counts), *t.shape[1:]))
        return out, counts

    def reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """The sum of the ranks' ``tensor`` (the same shape on every rank; fp32 or an integer type)."""
        if self.size == 1:
            return tensor
        self.collectives += 1
        if self.device_group is not None and tensor.device.type == self.device_type:
            out = tensor.clone(memory_format=torch.contiguous_format)
            dist.all_reduce(out, group=self.device_group)
            return out
        out = tensor.cpu().clone()
        dist.all_reduce(out, group=self.host_group)
        return out.to(tensor.device)

    def _all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """``[size * n, ...]``: every rank's ``tensor`` ``[n, ...]``, in rank order."""
        if self.device_group is not None and tensor.device.type == self.device_type:
            out = torch.empty(self.size * tensor.shape[0], *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
            dist.all_gather_into_tensor(out, tensor.contiguous(), group=self.device_group)
            return out
        host = tensor.cpu().contiguous()
        out = torch.empty(self.size * host.shape[0], *host.shape[1:], dtype=host.dtype)
        dist.all_gather_into_tensor(out, host, group=self.host_group)
        return out.to(tensor.device)

    def prefix_ready(self, ready: bool) -> bool:
        if self.prefix_group is None:
            raise RuntimeError("prefix copies require a peer communication group")
        value = torch.tensor([int(ready)], dtype=torch.int32)
        dist.all_reduce(value, op=dist.ReduceOp.MIN)
        return bool(value.item())

    def broadcast_prefix(self, data: torch.Tensor, domain: int) -> None:
        dist.broadcast(data, src=domain * self.size + self.rank, group=self.prefix_group)


def subgroup(size: int, rank: int, width: int, backend: str, timeout: datetime.timedelta,
             *, peers: bool = False) -> dist.ProcessGroup:
    own = None
    # Every rank creates groups in the same order. The resulting rendezvous keys are fresh on reload.
    groups = (range(i, size, width) for i in range(width)) if peers else (
        range(i, i + width) for i in range(0, size, width))
    for indices in groups:
        ranks = list(indices)
        member = dist.new_group(ranks, backend=backend, timeout=timeout)
        if rank in ranks:
            own = member
    if not isinstance(own, dist.ProcessGroup):
        raise RuntimeError(f"rank {rank} did not join its {backend} group")
    return own


def join(size: int, rank: int, rendezvous: str, device_backend: str, timeout_s: float, domains: int = 1,
         *, prefix_peer: bool = False) -> Group:
    """This process's place in a group of ``size`` ranks; it waits for the others to come."""
    if not 0 <= rank < size:
        raise ValueError(f"rank {rank} of a group of comm.size = {size}")
    if device_backend not in BACKENDS:
        raise ValueError(f"comm.device_backend = {device_backend!r}: one of {BACKENDS}")
    if domains < 1 or size % domains:
        raise ValueError(f"comm.domains = {domains} must divide the {size} ranks")
    if prefix_peer and (domains <= 1 or device_backend != "hccl"):
        raise ValueError("peer prefix transfer requires independent HCCL domains")
    timeout = datetime.timedelta(seconds=timeout_s)
    # Every rank is a process that sees its one card as device 0. torch's symmetric memory (the buffer
    # DeepGEMM's fused MoE kernel shares between the ranks) takes ranks with the same device index for
    # ranks on one card unless told otherwise; DeepInfer sets the same (deepinfer/__init__.py).
    os.environ.setdefault("TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES", "1")
    if dist.is_initialized():  # a reload of this unit: the process joined already
        if dist.get_world_size() != size or dist.get_rank() != rank:
            raise ValueError(f"this process is rank {dist.get_rank()} of {dist.get_world_size()}, and a group is "
                             f"joined once: rank {rank} of {size} needs a new process")
    else:
        if not rendezvous:
            raise ValueError(f"comm.size = {size} needs comm.rendezvous (tcp://host:port or file:///path)")
        dist.init_process_group(backend="gloo", init_method=rendezvous, rank=rank, world_size=size, timeout=timeout)
    width = size // domains
    device_type = DEVICE_TYPES.get(device_backend, "cuda")
    group = Group(width, rank % width, device_type=device_type, world_size=size, world_rank=rank)
    if width == 1 and not prefix_peer:
        return group
    if device_type == "npu":
        import_module("torch_npu").npu.set_device(0)
    try:
        if domains > 1 and width > 1:
            group.host_group = subgroup(size, rank, width, "gloo", timeout)
        if device_backend == "gloo":
            return group
        if width > 1:
            group.device_group = subgroup(size, rank, width, device_backend, timeout)
            # Materialize the communicator before the worker becomes ready.
            dist.all_reduce(torch.zeros(1, device=device_type), group=group.device_group)
        if prefix_peer:
            group.prefix_group = subgroup(size, rank, width, device_backend, timeout, peers=True)
            dist.all_reduce(torch.zeros(1, device=device_type), group=group.prefix_group)
    except Exception:
        leave(group)
        raise
    return group


def leave(group: Group) -> None:
    """Give up the group for the cards; the default group stays, as the process's."""
    if group.prefix_group is not None:
        dist.destroy_process_group(group.prefix_group)
        group.prefix_group = None
    if group.device_group is not None:
        dist.destroy_process_group(group.device_group)
        group.device_group = None
    if group.host_group is not None:
        dist.destroy_process_group(group.host_group)
        group.host_group = None


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        size, domains = config["comm.size"], config["comm.domains"]
        if size < 1:
            raise ValueError(f"comm.size = {size} must be at least 1")
        group = Group() if size == domains == 1 else join(size, ctx.instance.proc.index, config["comm.rendezvous"],
                                                          config["comm.device_backend"], config["comm.timeout_s"], domains,
                                                          prefix_peer=config["comm.prefix_peer"])
        ctx.effect(lambda: lambda: leave(group), label="device group")
        ctx.provide("comm.group@1", group)
