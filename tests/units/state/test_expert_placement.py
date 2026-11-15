"""``state.expert_placement`` and ``policies.eplb`` (expert parallel load balancing,
doc/specs/v1-ds1017.md stage 3): several ranks in one process, each on a thread, their collectives a
barrier (``Ranks``). A slot's rows hold the number of the expert whose weights they are, so a slot
that was given the wrong rows shows."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any

import pytest
import torch

from policies.eplb.unit import Greedy, place
from state.expert_placement.unit import Placement, initial, moves


class Ranks:
    """``comm.group@1`` for ``size`` ranks on threads of one process: a collective returns once every
    rank has called it."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.barrier = threading.Barrier(size)
        self.given: dict[int, Any] = {}

    def of(self, rank: int) -> Any:
        ranks = self

        class Group:
            size = ranks.size

            def __init__(self) -> None:
                self.rank = rank

            def _meet(self, value: Any) -> list[Any]:
                ranks.given[rank] = value
                ranks.barrier.wait(10)
                every = [ranks.given[r] for r in range(ranks.size)]
                ranks.barrier.wait(10)  # all have read before anyone gives again
                return every

            def gather(self, tensors: Sequence[torch.Tensor]) -> tuple[list[torch.Tensor], list[int]]:
                every = self._meet([t.clone() for t in tensors])
                return [torch.cat([e[i] for e in every]) for i in range(len(tensors))], [e[0].shape[0] for e in every]

            def reduce(self, tensor: torch.Tensor) -> torch.Tensor:
                return torch.stack(self._meet(tensor.clone())).sum(0).to(tensor.dtype)

        return Group()

    def run(self, fn: Callable[[int], Any]) -> list[Any]:
        out: dict[int, Any] = {}
        errors: list[BaseException] = []

        def body(rank: int) -> None:
            try:
                out[rank] = fn(rank)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
                self.barrier.abort()

        threads = [threading.Thread(target=body, args=(r,)) for r in range(self.size)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(20)
        assert not errors, errors
        return [out[r] for r in range(self.size)]


def stacks(placement: Placement, rank: int, layers: int, logical: int) -> list[list[torch.Tensor]]:
    """Per layer two tensors with one row a slot of ``rank``, each row filled with its expert's number
    (plus the layer's, so a row of another layer shows)."""
    physical = placement.slots(logical)
    local = physical // placement.group.size
    held = initial(logical, physical)[rank * local:(rank + 1) * local]
    return [[torch.tensor(held, dtype=torch.int32).view(-1, 1, 1).expand(-1, 2, 3).contiguous() + 1000 * layer,
             torch.tensor(held, dtype=torch.uint8).view(-1, 1).expand(-1, 5).contiguous()] for layer in range(layers)]


def test_the_moves_take_each_changed_slot_from_a_slot_that_held_its_expert_the_same_rank_first() -> None:
    old = [0, 1, 2, 3, 4, 5]  # rank 0 holds slots 0..2
    assert moves(old, old, 3) == []
    assert moves(old, [1, 0, 2, 3, 4, 5], 3) == [(0, 1), (1, 0)]  # a swap within a rank
    assert moves(old, [3, 1, 2, 0, 4, 5], 3) == [(0, 3), (3, 0)]  # and between ranks
    assert moves([0, 1, 0, 2, 3, 0], [0, 1, 2, 0, 3, 0], 3) == [(3, 2), (5, 3)]  # a copy on the same rank is preferred


@pytest.mark.parametrize(("ranks", "logical", "extra"), [(2, 4, 2), (4, 8, 4), (2, 6, 0), (4, 64, 8)])
def test_placing_again_moves_the_weights_to_where_the_maps_say(ranks: int, logical: int, extra: int) -> None:
    layers, interval = 3, 4
    bus = Ranks(ranks)
    placements = [Placement(bus.of(r), Greedy(), interval, interval, extra) for r in range(ranks)]
    weights = [stacks(p, r, layers, logical) for r, p in enumerate(placements)]
    for r, p in enumerate(placements):
        p.start(layers, logical, lambda layer, r=r: weights[r][layer], torch.device("cpu"))
    physical = placements[0].slots(logical)
    assert physical == -(-(logical + extra) // ranks) * ranks and all(p.active for p in placements)
    torch.manual_seed(0)
    load = torch.randint(1, 5, (layers, logical))
    load[:, 0] = 60  # one expert takes most of the tokens

    def steps(rank: int) -> int:
        p = placements[rank]
        for _ in range(interval + 1):
            for layer in range(layers):
                # What ds.ep_experts@1 does: this rank's slots take the tokens of the experts they hold.
                held = p.held[layer, rank * p.local:(rank + 1) * p.local]  # type: ignore[index]
                p.count(layer).add_((load[layer][held] // p.copies[layer][held]).to(torch.int32))  # type: ignore[index]
            p.step()
        return p.placed

    assert bus.run(steps) == [2] * ranks  # after a quarter of the interval, then an interval later
    held = placements[0].held
    assert held is not None
    for p in placements[1:]:  # every rank decided the same
        assert torch.equal(p.held, held) and torch.equal(p.to_physical, placements[0].to_physical)  # type: ignore[arg-type]
    for layer in range(layers):
        row = held[layer].tolist()
        assert sorted(set(row)) == list(range(logical))  # every expert has a slot
        if extra:
            assert row.count(0) > 1  # the busy one has copies
        for rank in range(ranks):
            local = placements[rank].local
            want = torch.tensor(row[rank * local:(rank + 1) * local])
            a, b = weights[rank][layer]
            assert torch.equal(a[:, 0, 0], (want + 1000 * layer).to(torch.int32)) and (a == a[:, :1, :1]).all()
            assert torch.equal(b[:, 0], want.to(torch.uint8))
        # The maps the kernels read agree with where the weights are.
        maps, copies = placements[0].to_physical[layer], placements[0].copies[layer]  # type: ignore[index]
        for expert in range(logical):
            slots = [s for s in maps[expert].tolist() if s >= 0]
            assert len(slots) == int(copies[expert]) and all(row[s] == expert for s in slots)


def test_a_chosen_expert_runs_in_one_of_its_slots_and_its_copies_share_the_tokens() -> None:
    bus = Ranks(2)
    p = Placement(bus.of(0), Greedy(), 0, 1, 2)  # spare slots, never placed again
    p.start(1, 4, lambda layer: [], torch.device("cpu"))
    assert p.slots(4) == 6 and initial(4, 6) == [0, 1, 2, 3, 0, 1]
    indices = torch.tensor([[0, 2], [0, 3], [1, 0], [0, 1]])
    slots = p.physical(0, indices)
    assert torch.equal(torch.tensor(initial(4, 6))[slots], indices)  # each slot holds the expert chosen
    assert set(slots[indices == 0].tolist()) == {0, 4}  # expert 0's two copies both take tokens


def test_alone_or_switched_off_there_is_nothing_to_place() -> None:
    alone = Placement(Ranks(1).of(0), Greedy(), 8, 4, 4)
    off = Placement(Ranks(2).of(0), Greedy(), 0, 100, 0)
    for p in (alone, off):
        assert not p.active and p.slots(384) == 384
        p.start(2, 384, lambda layer: [], torch.device("cpu"))
        p.step()
        assert p.placed == 0
    with pytest.raises(ValueError, match="window"):
        Placement(Ranks(2).of(0), Greedy(), 4, 8, 0)


def test_the_policy_gives_spare_slots_to_the_busiest_and_evens_out_the_ranks() -> None:
    tokens = torch.tensor([[[90, 10, 10, 10, 10, 10, 10, 10]], [[80, 20, 10, 10, 10, 10, 10, 10]]], dtype=torch.int32)
    held, to_physical, copies = place(tokens, physical=12, local=6)
    assert copies[0].tolist()[0] == 5 and sum(copies[0].tolist()) == 12  # the four spare slots go to expert 0
    each = tokens.sum(0)[0].float() / copies[0]
    loads = each[held[0].long()].view(2, 6).sum(-1)
    assert float(loads.max() / loads.mean()) < 1.1  # the two ranks carry about the same
    for expert in range(8):
        slots = sorted(s for s in to_physical[0, expert].tolist() if s >= 0)
        assert slots == [i for i, e in enumerate(held[0].tolist()) if e == expert]


def test_placement_started_inside_inference_mode_steps_and_rebalances_outside_it() -> None:
    """With ``cuda_graph``, ``Placement.start`` runs for the first time inside ``torch.inference_mode``
    while ``Placement.step`` runs from ``model.stepped`` outside it."""
    bus = Ranks(2)
    placements = [Placement(bus.of(r), Greedy(), 4, 4, 2) for r in range(2)]
    weights = [stacks(p, r, 2, 4) for r, p in enumerate(placements)]
    with torch.inference_mode():
        for r, p in enumerate(placements):
            p.start(2, 4, lambda layer, r=r: weights[r][layer], torch.device("cpu"))

    def run_steps(rank: int) -> int:
        p = placements[rank]
        with torch.inference_mode():
            p.count(0).add_(10 * (rank + 1))
        p.step()  # outside torch.inference_mode(), as PyTorchExecutor calls model.stepped()
        return p.placed

    assert bus.run(run_steps) == [1, 1]
