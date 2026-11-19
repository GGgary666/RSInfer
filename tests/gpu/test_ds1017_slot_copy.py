"""The real CUDA slot-copy operator, including unaligned fields and chunked launch parameters."""
from __future__ import annotations

import pytest
import torch

from state.swa_slots.slot_copy import operator


@pytest.mark.parametrize("ordered", [False, True])
def test_slot_batches_match_sequential_clones_on_the_calling_stream(ordered: bool) -> None:
    generator = torch.Generator().manual_seed(19)
    fields = [torch.randint(256, (150, width), dtype=torch.uint8, generator=generator)
              for width in [67584, 24, 17, 8193, *([7] * 63)]]
    moves = [(0, 1), (1, 2), (3, 2)] if ordered else [(i, i + 70) for i in range(69)]
    zeros = [3, 7, 7, 101]
    expected = [v.clone() for v in fields]
    for f in expected:
        for src, dst in moves:
            f[dst].copy_(f[src])
        f[zeros] = 0
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        device = [f.cuda(non_blocking=True) for f in fields]
        operator().clone_and_zero(device, moves, zeros)
        result = [f.cpu() for f in device]
    assert all(torch.equal(a, b) for a, b in zip(result, expected, strict=True))
