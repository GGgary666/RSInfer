"""kernels.torch_graph against the reference it stands in for in captured steps: the same decode
attention, whatever the width of the table, whatever sits past a row's length, GQA or not."""

from __future__ import annotations

import pytest
import torch

from kernels.reference import ops as reference
from kernels.torch_graph import ops


def paged(n: int, dtype: torch.dtype, gen: torch.Generator, *, heads: int, kv_heads: int, width: int = 24,
          stale: int = 0) -> dict[str, torch.Tensor]:
    """``n`` rows of 1..12 tokens at scattered slots (slot 0 is padding); past a row's length the
    table holds ``stale``."""
    dim = 32
    lens = torch.randint(1, 13, (n,), generator=gen)
    slots = torch.randperm(int(lens.sum()), generator=gen) + 1
    table = torch.full((n + 1, width), stale, dtype=torch.int32)
    table[0] = 0
    for r, row in enumerate(torch.split(slots, lens.tolist())):
        table[r + 1, :len(row)] = row
    return {"q": torch.randn(n, heads, dim, generator=gen).to(dtype),
            "k": torch.randn(int(lens.sum()) + 1, kv_heads, dim, generator=gen).to(dtype),
            "v": torch.randn(int(lens.sum()) + 1, kv_heads, dim, generator=gen).to(dtype),
            "table": table, "rows": torch.arange(1, n + 1), "lens": lens}


def both(p: dict[str, torch.Tensor], gqa: bool, table: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    args = (p["k"], p["v"], p["table"] if table is None else table, p["rows"], p["lens"], 32 ** -0.5, gqa)
    return (reference.attention_decode(p["q"], torch.empty_like(p["q"]), *args),
            ops.attention_decode(p["q"], torch.empty_like(p["q"]), *args))


@pytest.mark.parametrize(("dtype", "tolerance"), [(torch.float32, 1e-5), (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize(("heads", "kv_heads"), [(4, 2), (6, 1), (3, 3)])
def test_the_fixed_extent_decode_attention_equals_the_reference(dtype: torch.dtype, tolerance: float, heads: int,
                                                                kv_heads: int) -> None:
    gen = torch.Generator().manual_seed(5)
    for n in (1, 3, 8):
        p = paged(n, dtype, gen, heads=heads, kv_heads=kv_heads)
        want, got = both(p, heads != kv_heads)
        torch.testing.assert_close(got.float(), want.float(), rtol=tolerance, atol=tolerance)


def test_what_sits_past_a_rows_length_is_never_read() -> None:
    """A row's table keeps the slots of the request that held it before, beyond its own tokens; here
    they point outside the cache. A wider table (a larger ``batch.max_context_len``) changes
    nothing either, up to the order the sums are taken in."""
    gen = torch.Generator().manual_seed(6)
    p = paged(5, torch.float32, gen, heads=4, kv_heads=2, stale=10 ** 6)
    want, got = both(p, True)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    wide = torch.full((6, 512), 10 ** 6, dtype=torch.int32)
    wide[:, :24] = p["table"]
    torch.testing.assert_close(both(p, True, wide)[1], want, rtol=1e-5, atol=1e-5)


def test_a_padded_lane_reads_the_padding_slot_and_leaves_the_rows_alone() -> None:
    """A graph pads its rows with the padding row, index 0, one token long: its slot is slot 0."""
    gen = torch.Generator().manual_seed(7)
    p = paged(3, torch.float32, gen, heads=4, kv_heads=2)
    want, _ = both(p, True)
    q = torch.cat([p["q"], torch.zeros(5, 4, 32)])
    rows = torch.cat([p["rows"], torch.zeros(5, dtype=torch.int64)])
    lens = torch.cat([p["lens"], torch.ones(5, dtype=torch.int64)])
    got = ops.attention_decode(q, torch.empty_like(q), p["k"], p["v"], p["table"], rows, lens, 32 ** -0.5, True)
    torch.testing.assert_close(got[:3], want, rtol=1e-5, atol=1e-5)
    assert torch.isfinite(got).all()
    with pytest.raises(ValueError, match="length mask"):
        ops.attention_decode(q, torch.empty_like(q), p["k"], p["v"], p["table"], rows, lens, 32 ** -0.5, True, True)
