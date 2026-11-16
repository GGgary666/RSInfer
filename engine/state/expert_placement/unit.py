"""``state.expert_placement`` (``expert.placement@1``): which slot of which rank holds each expert of
each layer, among ranks that share a model's experts (``comm.group@1``), and placing them again from
the tokens they take: expert parallel load balancing (doc/specs/v1-ds1017.md stage 3), after
DeepInfer's ``model_executor/eplb`` (a679bb8).

- A layer has ``logical`` experts, the ones the gate chooses, in ``physical`` slots, ``local`` of them a
  rank: rank ``r`` holds the slots ``r * local`` onwards. ``eplb.extra_experts`` adds spare slots, which
  hold copies of the experts that take most tokens. At start the slots hold the experts in the
  checkpoint's order, then the first experts again (DeepInfer's initial layout).
- The kernels read the maps here: ``physical`` gives each chosen expert one of its slots, and the
  routed-experts kernel adds the tokens each of this rank's slots took to ``count``.
- ``step``, once per forward on every rank: the step's counts go into a window of the last
  ``eplb.window`` steps. Every ``eplb.interval`` steps (the first time after a quarter of it) the
  ranks sum their windows per expert, ``policy.rebalance@1`` decides the new placement, and each layer's
  weights are moved: within a rank by copies, between ranks through one ``gather`` of the rows that
  change rank. Every rank computes the same moves from the same maps.
- Where the experts are is kept with the weights (``record``, the model's ``experts.placed``), and
  written with every move: a reload of the model or of this unit finds the experts where the last
  moves put them, and only weights loaded again are back in the initial layout.

Moving an expert does not change what the model computes: its weights are the same in every slot,
and a token's term is the same on whichever rank computes it.

Without spare slots and with ``eplb.interval = 0``, or on one rank, nothing is placed again (``active``
is false); the kernels still read the maps while the weights are where earlier moves put them.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import torch


class Policy(Protocol):
    """``policy.rebalance@1``."""

    def place(self, tokens: torch.Tensor, physical: int, local: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


def initial(logical: int, physical: int) -> list[int]:
    """The expert each slot holds at start."""
    return [*range(logical), *(i % logical for i in range(physical - logical))]


def moves(old: Sequence[int], new: Sequence[int], local: int) -> list[tuple[int, int]]:
    """``(from slot, to slot)`` for every slot whose expert changes from ``old`` to ``new`` (the expert
    each slot holds): from a slot of the same rank that held the expert where there is one, else from
    the first slot that did. Sorted, so every rank lists the same moves in the same order."""
    where: dict[int, list[int]] = {}
    for slot, expert in enumerate(old):
        where.setdefault(expert, []).append(slot)
    out: list[tuple[int, int]] = []
    for slot, expert in enumerate(new):
        if old[slot] == expert:
            continue
        held = where[expert]
        near = [s for s in held if s // local == slot // local]
        out.append(((near or held)[0], slot))
    return sorted(out)


class Placement:
    """``expert.placement@1``."""

    def __init__(self, group: Any, policy: Policy, interval: int, window: int, extra: int) -> None:
        if interval < 0 or extra < 0 or (interval and not 1 <= window <= interval):
            raise ValueError(f"eplb.interval = {interval}, eplb.window = {window}, eplb.extra_experts = {extra}: the "
                             "window is 1 to the interval's steps, and none is negative")
        self.group, self.policy = group, policy
        self.interval, self.window_steps, self.extra = interval, window, extra
        self.layers = self.logical = self.slots_all = self.local = 0
        self.held: torch.Tensor | None = None
        """``[layers, physical]`` on the host: the expert each slot holds."""
        self.to_physical: torch.Tensor | None = None
        """``[layers, logical, spare + 1]``: each expert's slots, ``-1`` after them."""
        self.copies: torch.Tensor | None = None
        """``[layers, logical]``: how many slots each expert has."""
        self.counts: torch.Tensor | None = None
        """``[layers, local]``: tokens each of this rank's slots took in the step so far."""
        self.window: torch.Tensor | None = None
        self.stacks: Callable[[int], Sequence[torch.Tensor]] | None = None
        self.record: torch.Tensor | None = None
        """``[layers, physical]`` int32 kept with the weights (the model's ``experts.placed``): the expert
        each slot's rows hold. Moving the rows writes it too, so whatever reads the weights later (a
        reloaded model, a reloaded placement) finds the experts where they are."""
        self.mapped = False
        """The kernels read the maps: placing is on, or the weights are not in the initial layout."""
        self.steps = 0
        self.placed = 0
        """How many times the experts were placed again."""

    @property
    def active(self) -> bool:
        return self.group.size > 1 and (self.interval > 0 or self.extra > 0)

    def slots(self, logical: int) -> int:
        """Slots of a layer of ``logical`` experts, over all ranks."""
        if not self.active:
            return logical
        ranks = self.group.size
        return -(-(logical + self.extra) // ranks) * ranks

    def start(self, layers: int, logical: int, stacks: Callable[[int], Sequence[torch.Tensor]], device: torch.device,
              record: torch.Tensor | None = None) -> None:
        """Where the experts are: as ``record`` says (``[layers, physical]``, kept with the weights and
        written where they were loaded, ``initial`` then), else in the ``initial`` layout.
        ``stacks(layer)`` are the layer's tensors with one row per slot of this rank. Called again
        whenever the weights' tensors or contents change, and by every new instance of the model."""
        self.layers, self.logical, self.stacks = layers, logical, stacks
        self.slots_all = self.slots(logical)
        self.local = self.slots_all // self.group.size
        self.steps = 0
        # ``start`` may run inside CUDA graph warmup (``torch.inference_mode``), while ``step`` and
        # ``rebalance`` change these in place from ``model.stepped`` outside it. Buffers of the same
        # shape are written in place: a graph recorded over them keeps reading them.
        with torch.inference_mode(False):
            first = initial(logical, self.slots_all)
            self.record = record if record is not None and tuple(record.shape) == (layers, self.slots_all) else None
            held = (self.record.cpu().to(torch.int64) if self.record is not None
                    else torch.tensor(first, dtype=torch.int64).expand(layers, -1).contiguous())
            self.held = held
            self.mapped = self.group.size > 1 and (self.active or not bool((held == torch.tensor(first)).all()))
            if not self.mapped:
                return
            spare = self.slots_all - logical
            where: list[list[list[int]]] = []
            for row in held.tolist():
                each: list[list[int]] = [[] for _ in range(logical)]
                for slot, expert in enumerate(row):
                    each[expert].append(slot)
                where.append([slots + [-1] * (spare + 1 - len(slots)) for slots in each])
            to_physical = torch.tensor(where, dtype=torch.int32)
            copies = (to_physical >= 0).sum(-1, dtype=torch.int32)
            self.to_physical = self._put(self.to_physical, to_physical, device)
            self.copies = self._put(self.copies, copies, device)
            self.counts = self._put(self.counts, torch.zeros(layers, self.local, dtype=torch.int32), device)
            self.window = self._put(self.window, torch.zeros(max(1, self.window_steps), layers, self.local,
                                                             dtype=torch.int32), device)

    @staticmethod
    def _put(old: torch.Tensor | None, value: torch.Tensor, device: torch.device) -> torch.Tensor:
        if old is not None and old.shape == value.shape and old.device == torch.device(device):
            return old.copy_(value)
        return value.to(device)

    # ------------------------------------------------------------------ what the kernels read

    def physical(self, layer: int, indices: torch.Tensor) -> torch.Tensor:
        """The slot of each chosen expert ``[tokens, k]``: of an expert's copies, by the token's place
        in the step, so that they share its tokens."""
        assert self.to_physical is not None and self.copies is not None
        copies = self.copies[layer][indices]
        which = torch.arange(indices.shape[0], device=indices.device).unsqueeze(-1) % copies
        return self.to_physical[layer][indices, which].long()

    def maps(self, layer: int) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """What a layer's MoE kernel takes: each expert's slots ``[logical, spare + 1]`` (``-1`` after
        them), how many each has ``[logical]``, and where it adds the tokens each of this rank's slots
        took ``[local]`` (``None`` while not placing again: nothing would read the counts); three
        ``None`` where every slot holds its initial expert and nothing is placed again."""
        if not self.mapped:
            return None, None, None
        assert self.to_physical is not None and self.copies is not None and self.counts is not None
        return self.to_physical[layer], self.copies[layer], self.counts[layer] if self.active else None

    def count(self, layer: int) -> torch.Tensor:
        """``[local]`` int32: the routed-experts kernel adds the tokens each slot of this rank took."""
        assert self.counts is not None
        return self.counts[layer]

    # ------------------------------------------------------------------ placing again

    def step(self) -> None:
        """A forward ended (on every rank: one with no rows of its own took part too)."""
        if not self.active or self.interval <= 0:
            return
        assert self.counts is not None and self.window is not None
        self.window[self.steps % self.window.shape[0]].copy_(self.counts)
        self.counts.zero_()
        self.steps += 1
        first = max(1, self.interval // 4)  # early the first time: the start's order knows nothing
        if self.steps == first or (self.steps > first and (self.steps - first) % self.interval == 0):
            self.rebalance()

    def rebalance(self) -> None:
        assert self.held is not None and self.window is not None and self.to_physical is not None \
            and self.copies is not None and self.stacks is not None
        local, rank = self.local, self.group.rank
        device = self.window.device
        steps = min(self.steps, self.window.shape[0])
        mine = self.held[:, rank * local:(rank + 1) * local].to(device)
        tokens = torch.zeros(steps, self.layers, self.logical, dtype=torch.int32, device=device)
        tokens.scatter_add_(-1, mine.expand(steps, -1, -1), self.window[:steps])
        tokens = self.group.reduce(tokens).cpu()
        held, to_physical, copies = self.policy.place(tokens, self.slots_all, local)
        held = held.to(torch.int64)
        moved = 0
        for layer in range(self.layers):
            moved += self._move(self.stacks(layer), self.held[layer].tolist(), held[layer].tolist())
        if rank == 0:
            before, after = self._balance(tokens, self.held, self.copies.cpu()), self._balance(tokens, held, copies)
            print(f"state.expert_placement: experts placed again at step {self.steps}: {moved} slots changed, the "
                  f"busiest rank takes {before:.3f} of an even share's tokens before, {after:.3f} after "
                  f"({self.slots_all - self.logical} spare slots a layer)", file=sys.stderr, flush=True)
        self.held = held
        self.to_physical.copy_(to_physical.to(device))
        self.copies.copy_(copies.to(device))
        if self.record is not None:  # with the weights, which now hold the experts as ``held`` says
            self.record.copy_(held.to(self.record.device, self.record.dtype))
        self.placed += 1

    def _move(self, stacks: Sequence[torch.Tensor], old: Sequence[int], new: Sequence[int]) -> int:
        """One layer's weights from the slots of ``old`` to those of ``new``."""
        plan = moves(old, new, self.local)
        if not plan:
            return 0
        local, first = self.local, self.group.rank * self.local
        between = [(src, dst) for src, dst in plan if src // local != dst // local]
        within = [(src - first, dst - first) for src, dst in plan
                  if src // local == dst // local == self.group.rank]
        for stack in stacks:
            # Read everything before anything is written: a slot may be both a source and a target.
            # Gather cross-rank rows in bounded chunks so a 96-expert rebalance never spikes multi-GiB
            # temporary collective buffers on tight GPU memory budgets.
            kept = stack[[src for src, _ in within]].clone() if within else None
            chunks: list[tuple[list[int], torch.Tensor]] = []
            for off in range(0, len(between), 16):
                batch = between[off:off + 16]
                leaving = [src - first for src, _ in batch if src // local == self.group.rank]
                arriving = [(i, dst - first) for i, (_, dst) in enumerate(batch) if dst // local == self.group.rank]
                (got,), _ = self.group.gather([stack[leaving]])
                if arriving:
                    chunks.append(([dst for _, dst in arriving], got[[i for i, _ in arriving]]))
            if kept is not None and within:
                stack[[dst for _, dst in within]] = kept
            for dsts, rows in chunks:
                stack[dsts] = rows
        return len(plan)

    def _balance(self, tokens: torch.Tensor, held: torch.Tensor, copies: torch.Tensor) -> float:
        """The busiest rank's tokens over an even share, averaged over the window's steps and the layers."""
        each = tokens.float() / copies.float().clamp_min(1).unsqueeze(0)
        ranks = each.gather(-1, held.unsqueeze(0).expand(each.shape[0], -1, -1)).unflatten(-1, (-1, self.local)).sum(-1)
        return float((ranks.max(-1).values / ranks.mean(-1).clamp_min(1e-6)).mean())


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        ctx.provide("expert.placement@1", Placement(ctx.use("comm.group@1"), ctx.use("policy.rebalance@1"),
                                                    config["eplb.interval"], config["eplb.window"],
                                                    config["eplb.extra_experts"]))
