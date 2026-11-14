"""Arguments for the DS definitions over the states' caches (``kernels.ds1017_compose``), and how to
compare two solutions' results: shared by the GPU tests of the TileLang kernels (small shapes, the
4090 D) and of DeepInfer's (the released shapes, the SM100).

A case runs a solution on its own copy of the arguments; what it is compared on is what it returns
and what it writes (a cache's records, read back as values; a ring of the compressor), never bytes
alone: two quantizers that agree on every value but a tie write other bytes for that one."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from core.contracts.kernels import Tolerance
from kernels import ds1017_formats as formats

F32, BF16, U8 = torch.float32, torch.bfloat16, torch.uint8


@dataclass(frozen=True)
class Shapes:
    heads: int
    head_dim: int
    rotary: int
    window: int
    index_heads: int
    index_dim: int
    entries: int
    """Latents a page (a page of 128 tokens at a ratio of 2)."""
    rows: tuple[int, ...]
    """The latents of the indexer's rows."""
    topk: int
    topk_blocks: int
    block: int


SMALL = Shapes(heads=16, head_dim=128, rotary=32, window=128, index_heads=16, index_dim=64, entries=8,
               rows=(5, 70, 300), topk=64, topk_blocks=16, block=8)
"""``small_config``'s."""
REAL = Shapes(heads=64, head_dim=512, rotary=64, window=128, index_heads=32, index_dim=128, entries=64,
              rows=(37, 700, 3000), topk=512, topk_blocks=128, block=8)
"""The released model's, but for fewer candidate blocks than it keeps (2048), so that 3000 latents
have more blocks than candidates."""


def _rand(g: torch.Generator, *shape: int, scale: float = 1.0, dtype: torch.dtype = BF16) -> torch.Tensor:
    return (torch.randn(*shape, generator=g) * scale).to(dtype).cuda()


def _table(s: Shapes, positions: int = 4096) -> torch.Tensor:
    freqs = formats.rope_table(s.rotary, positions, 256, 160000.0, 16, 32, 1, torch.device("cuda"))
    return torch.cat([freqs.real, freqs.imag], dim=-1).contiguous()


def _keys(g: torch.Generator, count: int, dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random index keys as FP4 bytes and scales, and their values."""
    data, scales = formats.quantize_fp4(torch.randn(count, dim, generator=g) * 0.5)
    return data.cuda(), scales.cuda(), formats.dequant_fp4(data, scales).cuda()


def _segments(rows: list[tuple[int, list[int]]], entries: int) -> tuple[torch.Tensor, int]:
    """Load segments of rows ``(latents, pages)``, one after the other."""
    out: list[tuple[int, int, int, int]] = []
    base = 0
    for count, pages in rows:
        for j in range(-(-count // entries)):
            out.append((pages[j], 0, min(entries, count - j * entries), base + j * entries))
        base += count
    return torch.tensor(out, dtype=torch.int32).view(-1, 4).cuda(), base


def cases(s: Shapes, sources: Callable[[str], Callable[..., Any]] | None = None, seed: int = 17
          ) -> dict[str, tuple[Any, ...]]:
    """``"<definition> <case>"`` -> arguments, at shapes ``s``, on the GPU. ``sources`` gives the
    reference of a definition: a candidate user's blocks are the source's choice."""
    g = torch.Generator().manual_seed(seed)
    t, d, rd = 70, s.head_dim, s.rotary
    table = _table(s)
    positions = torch.randint(0, 4096, (t,), generator=g).cuda()
    norm = 1 + _rand(g, d, scale=0.1)
    out: dict[str, tuple[Any, ...]] = {}
    out["ds.rope@1"] = (_rand(g, t, s.heads, d, scale=0.5), table, positions, rd)
    out["ds.rope@1 inverse"] = (_rand(g, t, s.heads, d, scale=0.5), table, positions, rd, True)
    out["ds.transform@1"] = (_rand(g, t, d), norm, 1e-6, table, positions, rd, "fp8")

    pages = 24
    rows = torch.randperm(pages * s.entries - 1, generator=g)[:t].cuda() + 1  # row 0 is the scratch entry
    rows[::9] = -1
    ring = torch.zeros(6 * s.window, d + d // 32, dtype=U8, device="cuda")
    ring_rows = torch.randperm(6 * s.window - 1, generator=g)[:t].cuda() + 1
    out["ds.store_cache@1 fp8"] = (_rand(g, t, d), norm, 1e-6, table, positions, rd, ring, ring_rows, "fp8", s.window)
    values = formats.unpack_kv(formats.pack_kv(_rand(g, t, d))).to(BF16)
    out["ds.store_cache@1 fp8 values"] = (values, None, 1e-6, None, None, 0, ring.clone(), ring_rows, "fp8", s.window)
    latents = torch.zeros(pages * s.entries, d // 2 + d // 16, dtype=U8, device="cuda")
    out["ds.store_cache@1 latent"] = (_rand(g, t, d), norm, 1e-6, table, positions, rd, latents, rows, "latent",
                                      s.entries, True)
    keys = torch.zeros(pages * s.entries, s.index_dim // 2 + s.index_dim // 32, dtype=U8, device="cuda")
    out["ds.store_cache@1 key"] = (_rand(g, t, s.index_dim), 1 + _rand(g, s.index_dim, scale=0.1), 1e-6, table, positions,
                                   rd, keys, rows, "key", s.entries)

    # Loads: random records, rows of the page table in a random order.
    perm = torch.randperm(pages, generator=g).tolist()
    layout: list[tuple[int, list[int]]] = []
    for count in (13, s.entries * 2 + 5, s.entries):  # a part of a page, pages and a part, a whole page
        need = -(-count // s.entries)
        layout.append((count, perm[:need]))
        perm = perm[need:]
    segments, total = _segments(layout, s.entries)
    held = formats.pack_latent(formats.unpack_latent(formats.pack_latent(_rand(g, pages * s.entries, d))))
    out["ds.load_values@1 latent"] = (held, s.entries, segments, total, "latent")
    held8 = formats.pack_kv(_rand(g, pages * s.entries, d))
    out["ds.load_values@1 fp8"] = (held8, s.entries, segments, total, "fp8")
    data, scales, _ = _keys(g, pages * s.entries, s.index_dim)
    key_cache = torch.zeros(pages * s.entries, s.index_dim // 2 + s.index_dim // 32, dtype=U8, device="cuda")
    held_data, held_scales = formats.key_sections(key_cache, s.entries, s.index_dim)
    held_data.copy_(data.view(pages, s.entries, -1))
    held_scales.copy_(scales.view(pages, s.entries, -1))
    out["ds.load_keys@1"] = (key_cache, s.entries, segments, total, s.index_dim)

    q = _rand(g, t, s.index_heads, s.index_dim)
    out["ds.quant_index_q@1"] = (q, table, positions, rd)

    # The indexer over loaded keys: rows of different lengths, each with a few queries at its end.
    k_data, k_scales, _ = _keys(g, sum(s.rows), s.index_dim)
    ks: list[int] = []
    ke: list[int] = []
    base = 0
    for count in s.rows:
        for j in range(4):
            ks.append(base)
            ke.append(base + max(count - 3 + j, 0) if j < 3 else base + count)
        base += count
    n = len(ks)
    q_data, q_scales = formats.quantize_fp4(torch.randn(n, s.index_heads, s.index_dim, generator=g))
    weights = _rand(g, n, s.index_heads, scale=0.1)
    common = (q_data.cuda(), q_scales.cuda(), weights, k_data, k_scales, torch.tensor(ks).cuda(), torch.tensor(ke).cuda(),
              4096, min(s.topk, max(s.rows)))
    out["ds.index_topk@1"] = (*common, "dense", None, s.topk_blocks, s.block)
    out["ds.index_topk@1 source"] = (*common, "source", None, s.topk_blocks, s.block)
    if sources is not None:
        picked = sources("ds.index_topk@1")(*common, "source", None, s.topk_blocks, s.block)[1]
        out["ds.index_topk@1 user"] = (*common, "user", picked, s.topk_blocks, s.block)

    # The indexer through pages: each query its own row of pages, at the longest row's width.
    width = max(s.rows)
    count_pages = -(-width // s.entries)
    total_pages = count_pages * n + 1
    k_data, k_scales, _ = _keys(g, total_pages * s.entries, s.index_dim)
    paged_keys = torch.zeros(total_pages * s.entries, s.index_dim // 2 + s.index_dim // 32, dtype=U8, device="cuda")
    held_data, held_scales = formats.key_sections(paged_keys, s.entries, s.index_dim)
    held_data.copy_(k_data.view(total_pages, s.entries, -1))
    held_scales.copy_(k_scales.view(total_pages, s.entries, -1))
    block_table = (torch.randperm(total_pages - 1, generator=g)[:n * count_pages].view(n, count_pages) + 1).cuda()
    lens = torch.tensor([ke[i] - ks[i] for i in range(n)]).cuda()
    common = (q_data.cuda(), q_scales.cuda(), weights, paged_keys, s.entries, block_table, lens, width, min(s.topk, width))
    out["ds.index_topk_paged@1"] = (*common, "dense", None, s.topk_blocks, s.block)
    out["ds.index_topk_paged@1 source"] = (*common, "source", None, s.topk_blocks, s.block)
    if sources is not None:
        picked = sources("ds.index_topk_paged@1")(*common, "source", None, s.topk_blocks, s.block)[1]
        out["ds.index_topk_paged@1 user"] = (*common, "user", picked, s.topk_blocks, s.block)

    # The shared expert: FP8 w1 over w3 and w2, with their block scales and those packed.
    dim, inter = (1024, 1024) if s is SMALL else (5120, 2304)

    def fp8(n: int, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data, scale = formats.quantize_fp8(torch.randn(n, k, generator=g) * k ** -0.5)
        return data.cuda(), scale.cuda(), formats.pack_scales(formats.block_rows(scale, n)).cuda()

    out["ds.shared_expert@1"] = (_rand(g, t, dim), *fp8(2 * inter, dim), *fp8(dim, inter), 10.0)

    # Engram's hash: two layers, n-grams of 4 with 8 heads each, tables of prime sizes.
    tokens = torch.randint(0, 99092, (t, 4), generator=g).cuda()
    multipliers = (torch.randint(0, 1 << 20, (2, 4), generator=g) * 2 + 1).cuda()
    primes = torch.tensor([[[1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049]] * 3] * 2).cuda() * torch.tensor([1, 3]).cuda()[:, None, None] + 2
    offsets = torch.cumsum(primes.flatten(1), dim=1) - primes.flatten(1)
    out["ds.engram_hash@1"] = (tokens, multipliers, primes, offsets)

    # The compressor at a ratio of 2: three sequences, one from position 0, two going on (one of
    # them from the middle of a group), one completing no group.
    ratio = 2
    seqs = [(0, 9), (5, 6), (6, 1)]  # (earlier positions, tokens of this step)
    tokens = sum(c for _, c in seqs)
    slots = 8
    state = _rand(g, slots, ratio, 2, d, dtype=F32)
    seq_slot = torch.tensor([3, 1, 6]).cuda()
    cu_q, cu_k, cu_c, c_seq = [0], [0], [0], []
    for i, (p, c) in enumerate(seqs):
        cu_q.append(cu_q[-1] + c)
        cu_k.append(cu_k[-1] + p + c)
        reserved = max((p + c) // ratio - p // ratio, 1)
        cu_c.append(cu_c[-1] + reserved)
        c_seq += [i] * reserved
    out["ds.compress@1"] = (_rand(g, tokens, d, dtype=F32), _rand(g, tokens, d, dtype=F32), state, seq_slot,
                            torch.tensor(cu_q).cuda(), torch.tensor(cu_k).cuda(), torch.tensor(cu_c).cuda(),
                            torch.tensor(c_seq).cuda(), ratio)
    context = torch.tensor([1, 2, 7, 8, 100, 3001]).cuda()
    out["ds.compress_decode@1"] = (_rand(g, 6, d, dtype=F32), _rand(g, 6, d, dtype=F32), state.clone(),
                                   torch.tensor([0, 2, 3, 4, 5, 7]).cuda(), context, ratio)
    return out


# ---------------------------------------------------------------- running and comparing

def _clone(args: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in args)


def observe(case: str, fn: Callable[..., Any], args: tuple[Any, ...]) -> list[torch.Tensor]:
    """What ``fn`` gives for ``case`` on a copy of ``args``, as tensors to compare."""
    a = _clone(args)
    return view(case, fn(*a), a)


def view(case: str, result: Any, args: tuple[Any, ...]) -> list[torch.Tensor]:
    """A solution's ``result`` for ``case`` and what it wrote into ``args``, as tensors to compare."""
    name = case.split()[0]
    a = args
    got = [r for r in (result if isinstance(result, tuple) else (result,)) if isinstance(r, torch.Tensor)]
    if name == "ds.store_cache@1":  # the records written, as values (row 0 is the scratch entry)
        cache, fmt, entries, dim = a[6], a[8], a[9], a[0].shape[-1]
        if fmt == "key":
            data, scales = formats.key_sections(cache, entries, dim)
            values = formats.dequant_fp4(data.flatten(0, 1), scales.flatten(0, 1))
        else:
            values = (formats.unpack_kv if fmt == "fp8" else formats.unpack_latent)(cache)
        got.append(values[1:])
    if name in ("ds.quant_index_q@1", "ds.load_keys@1"):
        got = [formats.dequant_fp4(got[0], got[1])]
    if name in ("ds.compress@1", "ds.compress_decode@1"):
        got.append(a[2])  # the rings
        if name == "ds.compress_decode@1":  # a token that completes no group pools nothing defined
            complete = a[4] % a[5] == 0  # (score, kv, state, slot, context_lens, ratio)
            got[0] = got[0][complete]
    return got


DECODE = ("ds.rope@1", "ds.rope@1 inverse", "ds.store_cache@1 fp8", "ds.store_cache@1 latent", "ds.store_cache@1 key",
          "ds.quant_index_q@1", "ds.index_topk_paged@1", "ds.index_topk_paged@1 source", "ds.index_topk_paged@1 user",
          "ds.compress_decode@1", "ds.shared_expert@1", "ds.engram_hash@1")
"""The cases a decode step runs: what a graph records."""


@dataclass(frozen=True)
class Report:
    case: str
    differ: list[float]
    """Per compared tensor: the fraction of its elements that differ."""
    largest: list[float]
    overlap: float | None = None
    """The indexer: the fraction of chosen latents both choose."""


def _choices(a: torch.Tensor, b: torch.Tensor) -> float:
    """How many of the chosen (non -1) entries the two lists share, row by row."""
    shared = chosen = 0
    for x, y in zip(a.tolist(), b.tolist(), strict=True):
        sx, sy = {v for v in x if v >= 0}, {v for v in y if v >= 0}
        shared += len(sx & sy)
        chosen += max(len(sx), len(sy))
    return shared / max(chosen, 1)


def compare(case: str, got: list[torch.Tensor], want: list[torch.Tensor], tolerance: Tolerance,
            min_overlap: float = 0.97, ties: float = 0.0) -> Report:
    """``got`` against ``want`` within ``tolerance``: the indexer's choices must share ``min_overlap``
    of their latents; values may differ beyond the tolerance in at most a fraction ``ties`` of the
    elements (a value exactly between two of a quantized format, rounded the other way)."""
    name = case.split()[0]
    assert len(got) == len(want), (case, len(got), len(want))
    if name in ("ds.index_topk@1", "ds.index_topk_paged@1"):
        overlap = _choices(got[0], want[0])
        blocks = _choices(got[1], want[1]) if len(got) > 1 else None
        assert overlap >= min_overlap, f"{case}: the choices share {overlap:.3f} of their latents"
        assert blocks is None or blocks >= min_overlap, f"{case}: the candidate blocks share {blocks:.3f}"
        return Report(case, [], [], overlap)
    differ: list[float] = []
    largest: list[float] = []
    for x, y in zip(got, want, strict=True):
        assert x.shape == y.shape, (case, x.shape, y.shape)
        xf, yf = x.float(), y.float()
        differ.append(float((xf != yf).float().mean()) if xf.numel() else 0.0)
        largest.append(float((xf - yf).abs().max()) if xf.numel() else 0.0)
        if not (tolerance.atol or tolerance.rtol):
            assert torch.equal(xf, yf), f"{case}: {int((xf != yf).sum())} of {xf.numel()} differ"
            continue
        outside = ~torch.isclose(xf, yf, atol=tolerance.atol, rtol=tolerance.rtol)
        assert float(outside.float().mean()) <= ties, (
            f"{case}: {int(outside.sum())} of {xf.numel()} beyond {tolerance}, the largest {largest[-1]}")
    return Report(case, differ, largest)
