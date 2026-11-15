"""``kernels.deepinfer``: DeepInfer's kernel libraries as solutions of the DS definitions, each against
the definition's reference at the released model's shapes (64 heads of 512, 5120 wide, experts of
2304). Needs a GPU of SM 10.0 and DeepInfer's environment, so it runs on the SM100
(``tests/gpu/sm100_follow.sh``), not on the 4090 D."""
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from kernels import ds1017_formats as formats
from kernels.deepinfer import ops
from kernels.deepinfer.unit import HOST_DEPENDENT, SOLUTIONS
from kernels.ds1017 import ops as reference
from kernels.ds1017.unit import DEFINITIONS
from tests.gpu.support_ds1017_cases import DECODE, REAL, cases, compare, observe, view
from tests.units.support.ds1017 import expert_stacks

DIM, HEADS, HEAD_DIM, INTER, EXPERTS, TOPK, HC, TOKENS = 5120, 64, 512, 2304, 32, 6, 4, 70
F32, BF16 = torch.float32, torch.bfloat16
TOLERANCE = {name: tolerance for name, _, _, tolerance, _ in DEFINITIONS}


@pytest.fixture(scope="module", autouse=True)
def aligned() -> None:
    ops.align_experts()  # what the unit does when it loads


def rand(g: torch.Generator, *shape: int, scale: float = 1.0, dtype: torch.dtype = BF16) -> torch.Tensor:
    return (torch.randn(*shape, generator=g) * scale).to(dtype).cuda()


def experts(g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    stacks: list[torch.Tensor] = []
    for n, k in ((INTER, DIM), (DIM, INTER), (INTER, DIM)):
        pairs = [formats.quantize_fp4(torch.randn(n, k, generator=g).cuda() * k ** -0.5) for _ in range(EXPERTS)]
        stacks += [torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs])]
    w1, s1, w2, s2, w3, s3 = stacks
    return expert_stacks(w1, s1, w2, s2, w3, s3)


def inputs(case: str, tokens: int = TOKENS, seed: int = 0) -> tuple[Any, ...]:
    """Arguments for one case of a definition, at the released model's shapes, on the GPU."""
    g = torch.Generator().manual_seed(len(case) + 1000 * seed)
    t = tokens
    match case:
        case "ds.fake_quant_fp8@1":
            return (rand(g, t, HEAD_DIM) * torch.exp(rand(g, t, 1, dtype=F32) * 3).to(BF16), 32)
        case "ds.fake_quant_fp4@1":
            return (rand(g, t, 32, 128), 32, False)
        case "ds.fake_quant_fp4@1 e4m3":
            return (rand(g, t, HEAD_DIM), 16, True)
        case "ds.linear_fp8@1":
            data, scale = formats.quantize_fp8(torch.randn(1280, DIM, generator=g).cuda() * DIM ** -0.5)
            return (rand(g, t, DIM), data, scale, formats.pack_scales(formats.block_rows(scale, 1280)))
        case "ds.linear@1":
            return (rand(g, t, DIM), rand(g, 32, DIM, scale=1 / 64))
        case "ds.linear@1 fp32":
            return (rand(g, t, DIM), rand(g, 1024, DIM, scale=1 / 64, dtype=F32))
        case "ds.grouped_linear@1":
            return (rand(g, t, 8, 4096), rand(g, 8, 1024, 4096, scale=1 / 64))
        case "ds.index_score@1":
            return (rand(g, 1, t, 32, 128), rand(g, 1, 300, 128), rand(g, 1, t, 32, scale=0.1))
        case "ds.rms_norm@1":
            return (rand(g, t, DIM), 1 + rand(g, DIM, scale=0.1), 1e-6)
        case "ds.sparse_attn@1":
            idx = torch.stack([torch.randperm(900, generator=g)[:598] for _ in range(t)])  # not whole blocks of keys
            idx[torch.rand(t, 598, generator=g) < 0.2] = -1
            idx[0, 5:] = -1
            return (rand(g, t, HEADS, HEAD_DIM, scale=0.3), rand(g, 900, HEAD_DIM, scale=0.3), idx.cuda(),
                    rand(g, HEADS, dtype=F32), HEAD_DIM ** -0.5)
        case "ds.cache_attn@1" | "ds.cache_attn@1 window":
            # As a decode step lists them: a window's records oldest first, the missing ones before
            # them; of the latents the ones reached first.
            window = formats.pack_kv(rand(g, 6 * 128, HEAD_DIM, scale=0.3)).view(6, 128, -1)
            latents = formats.pack_latent(rand(g, 900, HEAD_DIM, scale=0.3)).view(60, 15, -1)  # pages of latents
            held = torch.randint(1, 129, (t,), generator=g)
            at = torch.arange(128) - 128 + held[:, None]
            window_idx = torch.where(at >= 0, (torch.arange(t) % 6)[:, None] * 128 + at, -1)
            latent_idx = torch.stack([torch.randperm(900, generator=g)[:470] for _ in range(t)])
            latent_idx[torch.arange(470) >= torch.randint(0, 471, (t, 1), generator=g)] = -1
            alone = case.endswith("window")
            return (rand(g, t, HEADS, HEAD_DIM, scale=0.3), window, window_idx.cuda(), None if alone else latents,
                    None if alone else latent_idx.cuda(), rand(g, HEADS, dtype=F32), HEAD_DIM ** -0.5)
        case "ds.pack_kv@1":
            return (rand(g, t, HEAD_DIM) * torch.exp(rand(g, t, 1, dtype=F32) * 3).to(BF16),)
        case "ds.hc_mixes@1":
            return (rand(g, t, HC, DIM), rand(g, 24, HC * DIM, scale=1 / 64, dtype=F32),
                    1 + rand(g, 3, scale=0.2, dtype=F32), rand(g, 24, scale=0.5, dtype=F32), 1e-6, 20, 1e-6)
        case "ds.hc_pre@1":
            return (rand(g, t, HC, DIM), torch.rand(t, HC, generator=g).cuda())
        case "ds.hc_post@1":
            return (rand(g, t, DIM), rand(g, t, HC, DIM), torch.rand(t, HC, generator=g).cuda(),
                    torch.rand(t, HC, HC, generator=g).cuda())
        case "ds.moe_gate@1":
            return (rand(g, t, DIM), rand(g, 384, DIM, scale=1 / 64).float(), rand(g, 384, scale=0.1, dtype=F32), TOPK, 2.5)
        case "ds.moe_experts@1":
            x, weights = rand(g, t, DIM), torch.rand(t, TOPK, generator=g).cuda()
            indices = torch.stack([torch.randperm(EXPERTS, generator=g)[:TOPK] for _ in range(t)]).cuda()
            return (x, weights, indices, *experts(g), 10.0)
        case "ds.top@1":
            return (rand(g, t, 5000, dtype=F32), 512)
        case "ds.top@1 bf16":
            return (rand(g, t, 1, 2500), 512)
        case "ds.silu@1":
            return (rand(g, t, INTER, dtype=F32),)
        case "ds.pool@1":
            return (rand(g, t, 2, HEAD_DIM, dtype=F32), rand(g, t, 2, HEAD_DIM, dtype=F32))
        case "ds.engram_gate@1":
            return (rand(g, t, HC, DIM), rand(g, t, (HC + 1) * DIM), 1 + rand(g, HC, DIM, scale=0.1),
                    1 + rand(g, HC, DIM, scale=0.1), 1e-6)
        case _:
            raise KeyError(case)


OVER_CACHES = ("ds.rope@1", "ds.transform@1", "ds.store_cache@1", "ds.load_values@1", "ds.load_keys@1",
               "ds.quant_index_q@1", "ds.index_topk@1", "ds.index_topk_paged@1", "ds.compress@1", "ds.compress_decode@1",
               "ds.shared_expert@1", "ds.engram_hash@1")
"""The definitions over the caches: their cases are ``support_ds1017_cases``'."""
CASES = [name for name, _, _ in SOLUTIONS if name not in OVER_CACHES] + [
    "ds.fake_quant_fp4@1 e4m3", "ds.linear@1 fp32", "ds.top@1 bf16", "ds.cache_attn@1 window"]
ROWS_ONLY = tuple(name for name, _, _ in SOLUTIONS if name not in OVER_CACHES)


def close(case: str, got: Any, want: Any, args: tuple[Any, ...]) -> None:
    name = case.split()[0]
    tolerance = TOLERANCE[name]
    pairs = list(zip(got if isinstance(got, tuple) else (got,), want if isinstance(want, tuple) else (want,), strict=True))
    if name == "ds.top@1":  # the k largest values, whichever of equal ones
        scores: torch.Tensor = args[0]
        assert all(a.shape == b.shape and a.dtype == torch.int64 for a, b in pairs)
        pairs = [(scores.gather(-1, a).sort(dim=-1).values, scores.gather(-1, b).sort(dim=-1).values) for a, b in pairs]
    if name == "ds.moe_gate@1":  # the same experts with the same weights, in whichever order
        (gw, gi), (ww, wi) = got, want
        go, wo = gi.argsort(-1), wi.argsort(-1)
        pairs = [(gi.gather(1, go), wi.gather(1, wo)), (gw.gather(1, go), ww.gather(1, wo))]
    for a, b in pairs:
        assert a.shape == b.shape and a.dtype == b.dtype, (case, a.shape, b.shape, a.dtype, b.dtype)
        if a.dtype == torch.int64 or not (tolerance.atol or tolerance.rtol):
            assert torch.equal(a, b), f"{case}: {int((a != b).sum())} of {a.numel()} differ"
        else:
            torch.testing.assert_close(a.float(), b.float(), atol=tolerance.atol, rtol=tolerance.rtol,
                                       msg=lambda m, case=case: f"{case}: {m}")


def expected(case: str) -> Callable[..., Any]:
    """The definition's reference; for the routed experts, the reference with DeepInfer's rounding."""
    name = case.split()[0]
    if name == "ds.moe_experts@1":
        return lambda *args: reference.moe_experts(*args, direct=True)
    return getattr(reference, name.split(".", 1)[1].split("@")[0])


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("case", CASES)
def test_a_library_kernel_computes_what_its_definitions_reference_computes(case: str) -> None:
    args = inputs(case)
    fn = getattr(ops, case.split()[0].split(".", 1)[1].split("@")[0])
    got, want = fn(*args), expected(case)(*args)
    pairs = [(a.float(), b.float()) for a, b in
             zip(got if isinstance(got, tuple) else (got,), want if isinstance(want, tuple) else (want,), strict=True)
             if a.shape == b.shape]
    print(f"{case}: fraction of elements that differ {[round(float((a != b).float().mean()), 5) for a, b in pairs]}, "
          f"largest difference {[round(float((a - b).abs().max()), 6) for a, b in pairs]} "
          f"of {[round(float(b.abs().max()), 3) for _, b in pairs]}")
    if case == "ds.fake_quant_fp4@1 e4m3":
        a, b = pairs[0]
        blocks = (a != b).unflatten(-1, (-1, 16))
        where = blocks.any(-1).nonzero()[:6].tolist()
        print(f"  differing blocks {int(blocks.any(-1).sum())}, elements in them {blocks.sum(-1)[blocks.any(-1)][:8].tolist()}")
        for r, c in where[:3]:
            print("  x  ", [round(v, 4) for v in args[0][r, c * 16:c * 16 + 16].float().tolist()])
            print("  got", a[r, c * 16:c * 16 + 16].tolist())
            print("  ref", b[r, c * 16:c * 16 + 16].tolist())
    if case == "ds.fake_quant_fp4@1 e4m3":
        # TileKernels multiplies by the scale's reciprocal where the reference divides by the scale:
        # a value exactly halfway between two FP4 values (x / scale of 2.5 or 5) lands on the other
        # one. About one element in 2000, one FP4 step apart; nothing else may differ.
        a, b = pairs[0]
        x: torch.Tensor = args[0].float()
        amax = x.unflatten(-1, (-1, 16)).abs().amax(-1, keepdim=True).expand(-1, -1, 16).flatten(-2)
        assert float((a != b).float().mean()) < 2e-3 and bool(((a - b).abs() <= 0.36 * amax).all())
        return
    close(case, got, want, args)


ROW_INVARIANT = tuple(name for name, _, _ in SOLUTIONS if name not in ("ds.moe_gate@1", "ds.index_score@1"))
"""Whose kernels give a token the same bits whatever else is in the step: all but the gate, whose
logits are torch's fp32 product."""


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("case", [name for name in ROWS_ONLY if name not in ("ds.index_score@1",)])
def test_a_tokens_result_does_not_depend_on_the_rest_of_the_step(case: str) -> None:
    """One token alone against the same token among 70: printed for every kernel, required of
    ``ROW_INVARIANT``."""
    args = inputs(case)
    fn = getattr(ops, case.split(".", 1)[1].split("@")[0])
    rows = TOKENS
    alone = tuple(a[:1].contiguous() if isinstance(a, torch.Tensor) and a.dim() and a.shape[0] == rows else a for a in args)
    full, one = fn(*args), fn(*alone)
    if case == "ds.top@1":
        scores = args[0]
        full, one = scores.gather(-1, full).sort(dim=-1).values, scores[:1].gather(-1, one).sort(dim=-1).values
    pairs = list(zip(full if isinstance(full, tuple) else (full,), one if isinstance(one, tuple) else (one,), strict=True))
    same = all(torch.equal(a[:1], b) for a, b in pairs)
    print(f"{case}: one token alone {'equals' if same else 'DIFFERS FROM'} its row of {rows}")
    assert same or case not in ROW_INVARIANT


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("case", ROWS_ONLY)
def test_a_library_kernel_replays_from_a_graph_on_new_inputs(case: str) -> None:
    """A kernel recorded in a CUDA graph on one set of tensors, then replayed after other values were
    written into them (weights too), against the kernel run on those values."""
    name = case.split(".", 1)[1].split("@")[0]
    eager = getattr(ops, name)
    recorded = ops.moe_experts_fixed if case == "ds.moe_experts@1" else eager
    static, fresh = inputs(case, 4), inputs(case, 4, seed=1)
    recorded(*static)  # compiled and warm before the capture
    torch.cuda.synchronize()
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    problem = ""
    try:
        with torch.cuda.graph(graph, stream=stream):
            out = recorded(*static)
        want = eager(*fresh)
        for held, new in zip(static, fresh, strict=True):
            if isinstance(held, torch.Tensor):
                held.copy_(new)
        graph.replay()
        torch.cuda.synchronize()
        pairs = list(zip(out if isinstance(out, tuple) else (out,), want if isinstance(want, tuple) else (want,), strict=True))
        if not all(torch.equal(a, b) for a, b in pairs):
            problem = f"replay differs: {[float((a.float() - b.float()).abs().max()) for a, b in pairs]}"
    except Exception as e:  # a kernel that reads the device during capture, or launches off the stream
        problem = f"{type(e).__name__}: {str(e)[:200]}"
    print(f"{case}: {'replays from a graph' if not problem else 'NOT CAPTURABLE: ' + problem}")
    assert not problem or case not in CAPTURABLE


CAPTURABLE = tuple(name for name, _, _ in SOLUTIONS)
"""Every one: the routed experts (``HOST_DEPENDENT``) as their fixed-row form."""
assert {"ds.moe_experts@1"} <= HOST_DEPENDENT


REFERENCE = {name: getattr(reference, name.split(".", 1)[1].split("@")[0]) for name in OVER_CACHES}
MINE = {name: getattr(ops, name.split(".", 1)[1].split("@")[0]) for name in OVER_CACHES}


def _real(seed: int = 17) -> dict[str, tuple[Any, ...]]:
    return cases(REAL, lambda name: REFERENCE[name], seed)


@pytest.fixture(scope="module")
def real() -> dict[str, tuple[Any, ...]]:
    return _real()


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("case", [  # the case names of support_ds1017_cases (known without a GPU)
    "ds.rope@1", "ds.rope@1 inverse", "ds.transform@1", "ds.store_cache@1 fp8", "ds.store_cache@1 fp8 values",
    "ds.store_cache@1 latent", "ds.store_cache@1 key", "ds.load_values@1 latent", "ds.load_values@1 fp8",
    "ds.load_keys@1", "ds.quant_index_q@1", "ds.index_topk@1", "ds.index_topk@1 source", "ds.index_topk@1 user",
    "ds.index_topk_paged@1", "ds.index_topk_paged@1 source", "ds.index_topk_paged@1 user", "ds.compress@1",
    "ds.compress_decode@1", "ds.shared_expert@1", "ds.engram_hash@1"])
def test_deepinfers_operators_compute_what_the_references_compute(case: str, real: dict[str, tuple[Any, ...]]) -> None:
    """DeepInfer's TileLang operators, DeepGEMM's indexer kernels and DeepSelect, against the references,
    at the released shapes: on what they return and what they write into the caches."""
    name = case.split()[0]
    # The shared expert's SwiGLU is quantized from fp32 here: against the reference's rounding to bf16
    # first, about one value in sixteen is the next FP8 one (as the routed experts', ops' docstring).
    ties = 5e-2 if name == "ds.shared_expert@1" else 5e-3
    report = compare(case, observe(case, MINE[name], real[case]), observe(case, REFERENCE[name], real[case]),
                     TOLERANCE[name], min_overlap=0.95, ties=ties)
    print(report)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("case", DECODE)
def test_deepinfers_decode_operators_replay_from_a_graph_on_new_inputs(case: str) -> None:
    """What a decode step runs of them, recorded on one set of tensors and replayed after other values
    were written into them, against the operators run on those values."""
    name = case.split()[0]
    fn = MINE[name]
    static, fresh = _real(17)[case], _real(18)[case]
    static = tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in static)
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    with torch.cuda.stream(stream):  # warm on the stream it is recorded on (DeepGEMM keeps a workspace a stream)
        fn(*tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in static))
    torch.cuda.synchronize()
    problem = ""
    try:
        with torch.cuda.graph(graph, stream=stream):
            out = fn(*static)
        for held, new in zip(static, fresh, strict=True):
            if isinstance(held, torch.Tensor):
                held.copy_(new)
        graph.replay()
        torch.cuda.synchronize()
        got, want = view(case, out, static), observe(case, fn, fresh)
        if not all(torch.equal(a, b) for a, b in zip(got, want, strict=True)):
            problem = "replay differs"
    except Exception as e:  # a kernel that reads the device during capture, or launches off the stream
        problem = f"{type(e).__name__}: {str(e)[:300]}"
    print(f"{case}: {'replays from a graph' if not problem else 'NOT CAPTURABLE: ' + problem}")
    assert not problem


def fused_case(tokens: int, seed: int = 0) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """A sublayer step before the MoE and that MoE, at the released model's shapes: the step's
    arguments (without ``stage``) and the MoE's after its input, the weights in the checkpoint's row
    order."""
    g = torch.Generator().manual_seed(77 + seed)
    f32 = torch.float32
    mix = (2 + HC) * HC
    step = (rand(g, tokens, DIM), rand(g, tokens, HC, DIM), torch.rand(tokens, HC, generator=g).cuda(),
            torch.rand(tokens, HC, generator=g).cuda(), torch.rand(tokens, HC, HC, generator=g).cuda(),
            rand(g, mix, HC * DIM, scale=1 / 64, dtype=f32), 1 + rand(g, 3, scale=0.2, dtype=f32),
            rand(g, mix, scale=0.5, dtype=f32), 1 + rand(g, DIM, scale=0.1), 1e-6, 20, 1e-6)
    gate = rand(g, EXPERTS, DIM, scale=1 / 16)
    shared: list[torch.Tensor] = []
    for n, k in ((2 * INTER, DIM), (DIM, INTER)):
        data, scale = formats.quantize_fp8(torch.randn(n, k, generator=g).cuda() * k ** -0.5)
        shared += [data, scale, formats.pack_scales(formats.block_rows(scale, n))]
    return step, (gate.float(), rand(g, EXPERTS, scale=0.1, dtype=f32), gate, *experts(g), *shared, TOPK, 2.5, 10.0)


def mega_weights(moe: tuple[Any, ...]) -> tuple[Any, ...]:
    """The MoE's arguments with the expert stacks and the shared expert's weights in the fused kernel's row order."""
    gate, bias, gate16, w13, p13, w2, p2, sw13, ss13, sp13, sw2, ss2, sp2, *rest = moe
    w13, p13, w2, p2, sw13, sp13, sp2 = (t.clone() for t in (w13, p13, w2, p2, sw13, sp13, sp2))
    formats.expert_layout(w13, p13, w2, p2, mega=True)
    formats.shared_layout(sw13, sp13, sp2, mega=True)
    return (gate, bias, gate16, w13, p13, w2, p2, sw13, ss13, sp13, sw2, ss2, sp2, *rest)


@pytest.mark.timeout(1800)
def test_the_fused_sublayer_step_and_moe_compute_what_the_unfused_kernels_compute() -> None:
    """``ds.hc_step@1`` and ``ds.moe@1`` of this backend (``mega``: ``deep_gemm.mega_mhc``,
    ``bf16_mega_gate``, ``fp8_fp4_mega_moe`` with the shared expert, for a group of one rank) against
    the same from DeepInfer's unfused kernels (``ops.hc_step``, ``ops.moe``): the step writes the MoE's
    FP8 input into the MoE buffer, the gate its choice, and the MoE kernel reads both. The fused gate's
    product is bf16, so a token whose experts are nearly tied may choose others: compared are the
    tokens that chose the same. Also: the tokens each slot took, a step without tokens, and the pair
    recorded in a graph and replayed on other values."""
    from kernels.deepinfer.mega import Mega
    from worker.comm.unit import Group

    group = Group()
    step, moe = fused_case(TOKENS)
    stage = (group, EXPERTS, TOPK, INTER)
    unfused_step: Any = ops.hc_step  # the arguments are built as tuples: called as they are
    unfused_moe: Any = ops.moe
    want_y, want_stream, want_pre, want_post, want_comb = unfused_step(*step)
    want = unfused_moe(group, want_y, *moe)
    chosen = ops.moe_gate(want_y, moe[0], moe[1], TOPK, 2.5)[1].sort(dim=-1).values
    fused = mega_weights(moe)
    mega = Mega(256)
    fused_step: Any = mega.hc_step
    fused_moe: Any = mega.moe
    try:
        taken = torch.zeros(EXPERTS, dtype=torch.int32, device="cuda")
        y, stream, pre, post, comb = fused_step(*step, stage)
        out = fused_moe(group, y, *fused, None, None, taken)
        torch.cuda.synchronize()
        symm = mega.symm(group, EXPERTS, TOPK, DIM, INTER)
        same = (symm.topk_idx[:TOKENS].sort(dim=-1).values == chosen).all(dim=-1)
        for name, a, b in (("input", y, want_y), ("stream", stream, want_stream), ("pre", pre, want_pre),
                           ("post", post, want_post), ("comb", comb, want_comb), ("moe", out[same], want[same])):
            apart = float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-12))
            print(f"fused {name}: farthest {apart:.5f} of the largest value")
            assert a.shape == b.shape and apart < (0.1 if name == "moe" else 0.03), (name, apart)
        print(f"fused gate: {int(same.sum())} of {TOKENS} tokens chose the same experts; counted {int(taken.sum())}")
        assert int(same.sum()) >= TOKENS * 0.9 and int(taken.sum()) == TOKENS * TOPK
        assert torch.equal(taken.long(), torch.bincount(symm.topk_idx[:TOKENS].flatten(), minlength=EXPERTS))
        assert fused_moe(group, y[:0], *fused).shape == (0, DIM)  # a rank without tokens launches the kernel too
        # Recorded and replayed on other inputs.
        held, _ = fused_case(4)
        fresh, _ = fused_case(4, seed=1)
        graph, stream_ = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        # The fused kernels keep a small buffer per stream, made at their first launch on it: not
        # inside a capture. The engine's recorder warms a step up on its capture stream too.
        with torch.cuda.stream(stream_):
            fused_moe(group, fused_step(*held, stage)[0], *fused)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=stream_):
            recorded = fused_moe(group, fused_step(*held, stage)[0], *fused)
        expect = fused_moe(group, fused_step(*fresh, stage)[0], *fused)
        for kept, new in zip(held, fresh, strict=True):
            if isinstance(kept, torch.Tensor):
                kept.copy_(new)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(recorded, expect), float((recorded.float() - expect.float()).abs().max())
        # Hot-applying eplb.extra_experts changes slot count on the same Mega instance: SymmBuffer rebuilds cleanly.
        rebuilt = mega.symm(group, EXPERTS + 8, TOPK, DIM, INTER)
        assert mega.made_for == (1, EXPERTS + 8, TOPK, DIM, INTER) and rebuilt is mega.buffer
    finally:
        mega.close()


def _fp8_weight(g: torch.Generator, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data, scale = formats.quantize_fp8(torch.randn(n, k, generator=g).cuda() * k ** -0.5)
    return data, scale, formats.pack_scales(formats.block_rows(scale, n))


def fused_attn_case(tokens: int, seed: int = 0) -> dict[str, Any]:
    """Arguments for ``hc_step(..., 'attn')`` + ``attn_qkv`` + ``{cache,sparse}_attn_out`` at released shapes."""
    g = torch.Generator().manual_seed(177 + seed)
    q_lora, o_groups, o_lora, rotary, index_heads, index_dim = 1536, 8, 1024, 64, 32, 128
    step, _ = fused_case(tokens, seed=10 + seed)
    freqs = formats.rope_table(rotary, 4096, 0, 10000.0, 1.0, 32, 1, torch.device("cuda"))
    cos_sin = torch.cat([freqs.real, freqs.imag], dim=-1).contiguous()
    pos = torch.randint(0, 2048, (tokens,), generator=g).cuda()
    window = formats.pack_kv(rand(g, 8 * 128, HEAD_DIM, scale=0.3)).view(8, 128, -1)
    latents = formats.pack_latent(rand(g, 900, HEAD_DIM, scale=0.3)).view(60, 15, -1)
    held = torch.randint(1, 129, (tokens,), generator=g)
    at = torch.arange(128) - 128 + held[:, None]
    window_idx = torch.where(at >= 0, (torch.arange(tokens) % 8)[:, None] * 128 + at, -1).cuda()
    latent_idx = torch.stack([torch.randperm(900, generator=g)[:470] for _ in range(tokens)])
    latent_idx[torch.arange(470) >= torch.randint(0, 471, (tokens, 1), generator=g)] = -1
    sparse_src = rand(g, 900, HEAD_DIM, scale=0.3)
    sparse_idx = torch.stack([torch.randperm(900, generator=g)[:598] for _ in range(tokens)])
    sparse_idx[torch.rand(tokens, 598, generator=g) < 0.2] = -1
    sparse_idx[0, 5:] = -1
    return {
        "step": step,
        "wq_a": _fp8_weight(g, q_lora, DIM),
        "wkv": _fp8_weight(g, HEAD_DIM, DIM),
        "q_norm": 1 + rand(g, q_lora, scale=0.1),
        "wq_b": _fp8_weight(g, HEADS * HEAD_DIM, q_lora),
        "iwq_b": _fp8_weight(g, index_heads * index_dim, q_lora),
        "wo_a": rand(g, o_groups * o_lora, HEADS * HEAD_DIM // o_groups, scale=(HEADS * HEAD_DIM // o_groups) ** -0.5),
        "wo_b": _fp8_weight(g, DIM, o_groups * o_lora),
        "sink": rand(g, HEADS, dtype=F32),
        "cos_sin": cos_sin,
        "pos": pos,
        "rotary": rotary,
        "o_groups": o_groups,
        "window": window,
        "window_idx": window_idx,
        "latents": latents,
        "latent_idx": latent_idx.cuda(),
        "sparse_src": sparse_src,
        "sparse_idx": sparse_idx.cuda(),
    }


@pytest.mark.timeout(1800)
def test_the_fused_attention_computes_what_the_unfused_kernels_compute() -> None:
    """``Mega.hc_step(..., 'attn')`` + ``Mega.attn_qkv`` + ``Mega.{cache,sparse}_attn_out`` against
    ``ops.*`` at the released model's shapes: decode (with and without latents), prefill sparse
    attention, 0-token idle step, in-place weight refit invalidating ``_cached``, and CUDA graph replay."""
    from kernels.deepinfer.mega import Mega

    c = fused_attn_case(TOKENS)
    mega = Mega(256)
    scale = HEAD_DIM ** -0.5
    unfused_step: Any = ops.hc_step
    unfused_qkv: Any = ops.attn_qkv
    fused_step: Any = mega.hc_step
    fused_qkv: Any = mega.attn_qkv
    try:
        uy = unfused_step(*c["step"])[0]
        uq, ukv, uiq = unfused_qkv(uy, *c["wq_a"], *c["wkv"], c["q_norm"], 1e-6, *c["wq_b"], *c["iwq_b"], HEADS, HEAD_DIM)
        want_decode = ops.cache_attn_out(uq, c["cos_sin"], c["pos"], c["rotary"], c["window"], c["window_idx"],
                                         c["latents"], c["latent_idx"], c["sink"], scale, c["o_groups"], c["wo_a"], *c["wo_b"])
        want_win = ops.cache_attn_out(uq, c["cos_sin"], c["pos"], c["rotary"], c["window"], c["window_idx"],
                                      None, None, c["sink"], scale, c["o_groups"], c["wo_a"], *c["wo_b"])
        want_prefill = ops.sparse_attn_out(uq, c["cos_sin"], c["pos"], c["rotary"], c["sparse_src"], c["sparse_idx"],
                                           c["sink"], scale, c["o_groups"], c["wo_a"], *c["wo_b"])

        fy = fused_step(*c["step"], "attn")[0]
        fq, fkv, fiq = fused_qkv(fy, *c["wq_a"], *c["wkv"], c["q_norm"], 1e-6, *c["wq_b"], *c["iwq_b"], HEADS, HEAD_DIM)
        got_decode = mega.cache_attn_out(fq, c["cos_sin"], c["pos"], c["rotary"], c["window"], c["window_idx"],
                                         c["latents"], c["latent_idx"], c["sink"], scale, c["o_groups"], c["wo_a"], *c["wo_b"])
        got_win = mega.cache_attn_out(fq, c["cos_sin"], c["pos"], c["rotary"], c["window"], c["window_idx"],
                                      None, None, c["sink"], scale, c["o_groups"], c["wo_a"], *c["wo_b"])
        got_prefill = mega.sparse_attn_out(fq, c["cos_sin"], c["pos"], c["rotary"], c["sparse_src"], c["sparse_idx"],
                                           c["sink"], scale, c["o_groups"], c["wo_a"], *c["wo_b"])
        assert fiq is not None and uiq is not None
        for name, a, b, lim in (("kv", fkv, ukv, 0.03), ("iq", fiq, uiq, 0.03), ("decode", got_decode, want_decode, 0.08),
                                ("decode_win", got_win, want_win, 0.08), ("prefill", got_prefill, want_prefill, 0.08)):
            apart = float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-12))
            print(f"fused attn {name}: farthest {apart:.5f} of the largest value")
            assert a.shape == b.shape and apart < lim, (name, apart)

        # Zero-token idle rank returns empty tensors cleanly.
        eq, ekv, eiq = fused_qkv(fy[:0], *c["wq_a"], *c["wkv"], c["q_norm"], 1e-6, *c["wq_b"], None, None, None, HEADS, HEAD_DIM)
        assert eq.shape == (0, HEADS, HEAD_DIM) and ekv.shape == (0, HEAD_DIM) and eiq is None
        assert mega.cache_attn_out(eq, c["cos_sin"], c["pos"][:0], c["rotary"], c["window"], c["window_idx"][:0],
                                   None, None, c["sink"], scale, c["o_groups"], c["wo_a"], *c["wo_b"]).shape == (0, DIM)

        # In-place weight update bumps _version and invalidates Mega._cached.
        wo_a_mut = c["wo_a"].clone()
        out1 = mega.cache_attn_out(fq, c["cos_sin"], c["pos"], c["rotary"], c["window"], c["window_idx"],
                                   None, None, c["sink"], scale, c["o_groups"], wo_a_mut, *c["wo_b"])
        wo_a_mut.zero_()
        out2 = mega.cache_attn_out(fq, c["cos_sin"], c["pos"], c["rotary"], c["window"], c["window_idx"],
                                   None, None, c["sink"], scale, c["o_groups"], wo_a_mut, *c["wo_b"])
        assert not torch.equal(out1, out2) and float(out2.float().abs().max()) == 0.0

        # CUDA graph capture + replay on fresh activation inputs (weights stay fixed across steps).
        d0, d1 = fused_attn_case(4, seed=2), fused_attn_case(4, seed=3)
        for wk in ("wq_a", "wkv", "q_norm", "wq_b", "iwq_b", "wo_a", "wo_b", "sink", "cos_sin"):
            d1[wk] = d0[wk]

        def run_decode(d: dict[str, Any]) -> torch.Tensor:
            y0 = fused_step(*d["step"], "attn")[0]
            q0, _, _ = fused_qkv(y0, *d["wq_a"], *d["wkv"], d["q_norm"], 1e-6, *d["wq_b"], *d["iwq_b"], HEADS, HEAD_DIM)
            return mega.cache_attn_out(q0, d["cos_sin"], d["pos"], d["rotary"], d["window"], d["window_idx"],
                                       d["latents"], d["latent_idx"], d["sink"], scale, d["o_groups"], d["wo_a"], *d["wo_b"])

        graph, stream_ = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        with torch.cuda.stream(stream_):
            run_decode(d0)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=stream_):
            recorded = run_decode(d0)
        expect = run_decode(d1)
        for k in ("step", "pos", "window", "window_idx", "latents", "latent_idx"):
            v = d0[k]
            if isinstance(v, torch.Tensor):
                v.copy_(d1[k])
            elif isinstance(v, tuple):
                for a, b in zip(v, d1[k], strict=True):
                    if isinstance(a, torch.Tensor):
                        a.copy_(b)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(recorded, expect), float((recorded.float() - expect.float()).abs().max())
    finally:
        mega.close()


def _engram_ep_worker(rank: int, port: int) -> None:
    import torch.distributed as dist
    from kernels.deepinfer.mega import Mega
    from worker.comm.unit import join, leave

    torch.cuda.set_device(rank)
    group = join(2, rank, f"tcp://127.0.0.1:{port}", "nccl", 60.0)
    mega = Mega(64)
    try:
        g = torch.Generator().manual_seed(500)
        rows, cols, eh = 256, 8, 256
        full_w, full_s = formats.quantize_rows_fp8(torch.randn(rows, eh, generator=g).cuda() * 0.2)
        part = rows // 2
        local_w = full_w[rank * part:(rank + 1) * part].contiguous()
        local_s = full_s[rank * part:(rank + 1) * part].contiguous()
        wkv = _fp8_weight(g, (HC + 1) * DIM, cols * eh)
        qw, kw = 1 + rand(g, HC, DIM, scale=0.1), 1 + rand(g, HC, DIM, scale=0.1)

        rg = torch.Generator().manual_seed(600 + rank)
        x = rand(rg, 8, HC, DIM)
        ids = torch.randint(0, rows, (8, 1, cols), generator=rg).cuda()
        want = ops.engram(0, x, full_w[ids[:, 0]], full_s[ids[:, 0]], None, local_w, local_s, *wkv, qw, kw, 1e-6)

        mega.engram_fetch(group, ids, [local_w], [local_s], stage=True)
        got = mega.engram(0, x, None, None, None, local_w, local_s, *wkv, qw, kw, 1e-6)
        apart = float((got.float() - want.float()).abs().max() / want.float().abs().max().clamp_min(1e-12))
        assert got.shape == want.shape and apart < 0.02, (rank, apart)

        # CUDA graph capture + replay with EngramBuffer.fetch inside the graph.
        ids2 = torch.randint(0, rows, (8, 1, cols), generator=rg).cuda()
        x2 = rand(rg, 8, HC, DIM)
        graph, stream_ = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        with torch.cuda.stream(stream_):
            mega.engram_fetch(group, ids, [local_w], [local_s], stage=True)
            mega.engram(0, x, None, None, None, local_w, local_s, *wkv, qw, kw, 1e-6)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=stream_):
            mega.engram_fetch(group, ids, [local_w], [local_s], stage=True)
            rec = mega.engram(0, x, None, None, None, local_w, local_s, *wkv, qw, kw, 1e-6)
        mega.engram_fetch(group, ids2, [local_w], [local_s], stage=True)
        exp = mega.engram(0, x2, None, None, None, local_w, local_s, *wkv, qw, kw, 1e-6)
        ids.copy_(ids2)
        x.copy_(x2)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(rec, exp)
    finally:
        mega.close()
        leave(group)
        dist.destroy_process_group()


@pytest.mark.timeout(1800)
def test_the_fused_engram_computes_what_the_unfused_kernels_compute() -> None:
    """``Mega.engram`` (and on >=2 GPUs with ``deep_ep``, ``Mega.engram_fetch`` over ``EngramBuffer``)
    against ``ops.engram``: direct ``fp8_fp4_gemm_nt`` without bf16 round-trip, 0-token step, weight
    refit invalidation, and CUDA graph replay."""
    import importlib.util
    import socket
    from torch.multiprocessing.spawn import spawn
    from kernels.deepinfer.mega import Mega

    g = torch.Generator().manual_seed(301)
    rows, cols, eh = 256, 8, 256
    embed_w, embed_s = formats.quantize_rows_fp8(torch.randn(rows, eh, generator=g).cuda() * 0.2)
    wkv = _fp8_weight(g, (HC + 1) * DIM, cols * eh)
    qw, kw = 1 + rand(g, HC, DIM, scale=0.1), 1 + rand(g, HC, DIM, scale=0.1)
    x = rand(g, TOKENS, HC, DIM)
    ids = torch.randint(0, rows, (TOKENS, cols), generator=g).cuda()
    mega = Mega(256)
    try:
        want = ops.engram(0, x, None, None, ids, embed_w, embed_s, *wkv, qw, kw, 1e-6)
        got = mega.engram(0, x, None, None, ids, embed_w, embed_s, *wkv, qw, kw, 1e-6)
        apart = float((got.float() - want.float()).abs().max() / want.float().abs().max().clamp_min(1e-12))
        print(f"fused engram: farthest {apart:.5f} of the largest value")
        assert got.shape == want.shape and apart < 0.02, apart
        assert mega.engram(0, x[:0], None, None, ids[:0], embed_w, embed_s, *wkv, qw, kw, 1e-6).shape == (0, HC, DIM)

        # In-place weight update bumps _version and invalidates cached fused_weight.
        qw_mut = qw.clone()
        out1 = mega.engram(0, x, None, None, ids, embed_w, embed_s, *wkv, qw_mut, kw, 1e-6)
        qw_mut.neg_()
        out2 = mega.engram(0, x, None, None, ids, embed_w, embed_s, *wkv, qw_mut, kw, 1e-6)
        assert not torch.equal(out1, out2)

        # CUDA graph capture + replay on fresh inputs.
        x0, ids0 = rand(g, 4, HC, DIM), torch.randint(0, rows, (4, cols), generator=g).cuda()
        x1, ids1 = rand(g, 4, HC, DIM), torch.randint(0, rows, (4, cols), generator=g).cuda()
        graph, stream_ = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        with torch.cuda.stream(stream_):
            mega.engram(0, x0, None, None, ids0, embed_w, embed_s, *wkv, qw, kw, 1e-6)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=stream_):
            rec = mega.engram(0, x0, None, None, ids0, embed_w, embed_s, *wkv, qw, kw, 1e-6)
        exp = mega.engram(0, x1, None, None, ids1, embed_w, embed_s, *wkv, qw, kw, 1e-6)
        x0.copy_(x1)
        ids0.copy_(ids1)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(rec, exp)
    finally:
        mega.close()

    if torch.cuda.device_count() >= 2 and importlib.util.find_spec("deep_ep") is not None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        spawn(_engram_ep_worker, args=(port,), nprocs=2, join=True)


