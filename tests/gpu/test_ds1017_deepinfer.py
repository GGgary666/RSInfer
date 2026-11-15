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


@pytest.mark.timeout(1800)
def test_the_fused_moe_kernel_on_one_rank_computes_the_routed_experts() -> None:
    """``ds.ep_experts@1`` of this backend (``mega``: ``deep_gemm.fp8_fp4_mega_moe``) for a group of
    one rank, against the routed experts' reference with DeepInfer's rounding: the expert stacks in the
    fused kernel's row order, the tokens each slot took counted, slots that are not the experts'
    numbers, a step without tokens, and the kernel recorded in a graph and replayed on other values.
    It adds a token's terms in bf16 with fast math, so it agrees to the routed experts' tolerance,
    not to the bit."""
    from types import SimpleNamespace

    from kernels.deepinfer.mega import Mega

    args = inputs("ds.moe_experts@1")
    x, weights, indices, w13, p13, w2, p2, limit = args
    want = reference.moe_experts(*args, direct=True)
    m13, q13, m2, q2 = (t.clone() for t in (w13, p13, w2, p2))
    formats.expert_layout(m13, q13, m2, q2, mega=True)
    group = SimpleNamespace(size=1, rank=0, device_group=None)
    mega = Mega(256)
    try:
        taken = torch.zeros(EXPERTS, dtype=torch.int32, device="cuda")
        got = mega.ep_experts(group, x, weights, indices, m13, q13, m2, q2, limit, None, taken)
        torch.cuda.synchronize()
        apart = float((got - want).abs().max() / want.abs().max())
        print(f"fused MoE: farthest {apart:.4f} of the largest value; counts {int(taken.sum())} of {TOKENS * TOPK}")
        close("ds.moe_experts@1", got, want, args)
        assert torch.equal(taken.long(), torch.bincount(indices.flatten(), minlength=EXPERTS))
        # A slot is where an expert's weights are, not its number: the stacks' experts in another order.
        order = torch.randperm(EXPERTS, generator=torch.Generator().manual_seed(5)).cuda()
        o13, r13, o2, r2 = (t[order].contiguous() for t in (m13, q13, m2, q2))
        slot_of = torch.empty_like(order).scatter_(0, order, torch.arange(EXPERTS, device="cuda"))
        elsewhere = mega.ep_experts(group, x, weights, indices, o13, r13, o2, r2, limit, slot_of[indices])
        assert torch.equal(elsewhere, got)
        assert mega.ep_experts(group, x[:0], weights[:0], indices[:0], m13, q13, m2, q2, limit).shape == (0, DIM)
        # One token alone against the same token among the others.
        alone = mega.ep_experts(group, x[:1], weights[:1], indices[:1], m13, q13, m2, q2, limit)
        print(f"fused MoE: one token alone {'equals' if torch.equal(alone, got[:1]) else 'DIFFERS FROM'} its row of {TOKENS}")
        # Recorded and replayed on other inputs.
        sx, sw, si = (t.clone() for t in (x[:4], weights[:4], indices[:4]))
        fx, fw, fi = inputs("ds.moe_experts@1", 4, seed=1)[:3]
        mega.ep_experts(group, sx, sw, si, m13, q13, m2, q2, limit)
        torch.cuda.synchronize()
        graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        with torch.cuda.graph(graph, stream=stream):
            out = mega.ep_experts(group, sx, sw, si, m13, q13, m2, q2, limit)
        expect = mega.ep_experts(group, fx, fw, fi, m13, q13, m2, q2, limit)
        for held, new in ((sx, fx), (sw, fw), (si, fi)):
            held.copy_(new)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expect), float((out - expect).abs().max())
    finally:
        mega.close()
