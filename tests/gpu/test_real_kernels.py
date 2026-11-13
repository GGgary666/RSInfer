"""The real kernels on one CUDA GPU of 24 GB or more: SGLang's Triton GDN kernels
(``kernels.triton``) and FlashInfer's paged attention (``kernels.flashinfer``), at Qwen3.8-27B's
shapes.

Manual (``tests/gpu`` is not collected by default; it needs the ``gpu`` extra, and ``ninja`` and a
CUDA compiler on ``PATH`` for FlashInfer's first plan of a shape, as under ``uv run``):

    uv sync --extra gpu --extra frontend
    uv run pytest tests/gpu/test_real_kernels.py -s

- Each solution against the reference of its definition, on the same inputs and state pools,
  within the definition's tolerance (stage2-contract.md §6): the Triton GDN kernels, FlashInfer's
  paged attention and its Gemma RMSNorm.
- The worker stack on the first 4 layers of the model in bf16 (``QWEN38_MODEL``, or random weights
  from ``make_checkpoint``): ``auto`` picks the real kernels for every phase; a step replayed from
  a graph, with padded lanes and changing buckets, gives the logits and leaves the states an
  eager step does; the kernel units reload under running requests; a library that does not take
  fp32 is refused at start when it is named.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from core.contracts.kernels import Tolerance
from kernels.reference import ops as ref
from kernels.reference import paged as ref_paged
from kernels.reference.unit import DEFINITIONS
from layers.attn.paged import indptr_of
from platforms.cuda_graph import CudaGraphPlatform
from platforms.torch_eager import TorchEagerPlatform
from tests.gpu.support import free_gpu, make_checkpoint
from tests.units.support.worker import Driver, Worker, spec
from unit.runtime import ReloadPlan

DEVICE = torch.device("cuda")
HK, HV, K, V, WIDTH = 16, 48, 128, 128, 4  # Qwen3.8-27B's GDN heads, at TP1
CONV_DIM = 2 * HK * K + HV * V
HEADS, KV_HEADS, HEAD_DIM = 24, 4, 256
TOLERANCE = {name: tolerance for name, _, _, tolerance, _ in DEFINITIONS}


def close(definition: str, got: torch.Tensor, want: torch.Tensor) -> None:
    t: Tolerance = TOLERANCE[definition]
    torch.testing.assert_close(got.float(), want.float(), rtol=t.rtol, atol=t.atol, msg=lambda m: f"{definition}: {m}")


@pytest.fixture(scope="module")
def triton() -> Any:
    from kernels.triton import ops
    return ops


@pytest.mark.parametrize(("dtype", "ssm_dtype"), [(torch.bfloat16, torch.float32), (torch.bfloat16, torch.bfloat16),
                                                  (torch.float32, torch.float32)])
def test_triton_gdn_decode_kernels_stay_within_their_definitions(triton: Any, dtype: torch.dtype,
                                                                 ssm_dtype: torch.dtype) -> None:
    """One decode step of five rows at scattered slots, and a padded lane (slot -1), which the
    kernels leave alone: its output is undefined, the pools are untouched. The two input
    projections go in as the GEMMs leave them, and the packed result of the conv into the
    delta rule."""
    gen = torch.Generator(device="cuda").manual_seed(0)
    rand: Callable[..., torch.Tensor] = lambda *shape: torch.randn(*shape, device=DEVICE, generator=gen)
    weight, A_log, dt_bias = (0.3 * rand(CONV_DIM, WIDTH)).to(dtype), (0.5 * rand(HV)).to(dtype), (0.5 * rand(HV)).to(dtype)
    conv0, ssm0 = (0.5 * rand(9, CONV_DIM, WIDTH - 1)).to(dtype), (0.1 * rand(9, HV, V, K)).to(ssm_dtype)
    slots = torch.tensor([3, 1, 7, -1, 5, 2], device=DEVICE, dtype=torch.int32)
    real = slots >= 0
    qkvz, ba = rand(6, CONV_DIM + HV * V).to(dtype), rand(6, 2 * HV).to(dtype)
    unpack = (CONV_DIM, HV * V, HV, V, "silu")

    conv_ref, conv_tri = conv0.clone(), conv0.clone()
    want = ref.fused_qkvzba_causal_conv1d_update(qkvz[real], ba[real], conv_ref, weight, None, slots[real], *unpack)
    got = triton.fused_qkvzba_causal_conv1d_update(qkvz, ba, conv_tri, weight, None, slots, *unpack)
    close("gdn.qkvzba_conv1d_update@1", got[0][real], want[0])
    for mine, theirs in zip(got[1:], want[1:], strict=True):  # z, b, a: moved, not computed
        assert mine.is_contiguous() and torch.equal(mine[real], theirs)
    assert got[1].shape == (6, HV, V) and got[2].shape == got[3].shape == (6, HV)
    assert torch.equal(conv_tri[1:], conv_ref[1:]) and torch.equal(conv_tri[0], conv0[0])

    mixed_qkv, _, b, a = got
    ssm_ref, ssm_tri = ssm0.clone(), ssm0.clone()
    want_out = ref.fused_recurrent_gated_delta_rule_packed_decode(mixed_qkv[real], a[real], b[real], A_log, dt_bias,
                                                                  K ** -0.5, ssm_ref, slots[real], HV, V)
    got_out = triton.fused_recurrent_gated_delta_rule_packed_decode(mixed_qkv, a, b, A_log, dt_bias, K ** -0.5,
                                                                    ssm_tri, slots, HV, V)
    assert got_out.shape == (1, 6, HV, V)
    close("gdn.packed_decode@1", got_out[:, real], want_out)
    close("gdn.packed_decode@1", ssm_tri, ssm_ref)
    assert torch.equal(ssm_tri[0], ssm0[0])

    lanes = int(real.sum())
    xn, z, wn = rand(lanes * HV, V).to(dtype), rand(lanes * HV, V).to(dtype), rand(V).to(dtype)
    close("norm.rms_gated@1", triton.rms_norm_gated(xn, wn, None, z, 1e-6, None, True),
          ref.rms_norm_gated(xn, wn, None, z, 1e-6, None, True))
    # a prefill step's z: a view of the projection, by token and head
    strided = rand(lanes, CONV_DIM + HV * V).to(dtype)[:, CONV_DIM:].view(lanes, HV, V)
    assert not strided.is_contiguous()
    close("norm.rms_gated@1", triton.rms_norm_gated(xn, wn, None, strided, 1e-6, None, True),
          ref.rms_norm_gated(xn, wn, None, strided, 1e-6, None, True))


@pytest.mark.parametrize(("dtype", "has_gate"), [(torch.bfloat16, True), (torch.float16, True), (torch.bfloat16, False)])
def test_triton_attention_glue_stays_within_its_definitions(triton: Any, dtype: torch.dtype, has_gate: bool) -> None:
    """What an attention layer runs around its attention: Q/K norm, RoPE on the first quarter of
    each head and the split of the output gate in one kernel, at a decode step's positions and a
    prefill step's; and the output gate, written over the attention's output."""
    from layers.rope.unit import rope_cos_sin_cache

    gen = torch.Generator(device="cuda").manual_seed(5)
    rand: Callable[..., torch.Tensor] = lambda *shape: torch.randn(*shape, device=DEVICE, generator=gen)
    rotary_dim = HEAD_DIM // 4
    cache = rope_cos_sin_cache(rotary_dim, 4096, 10_000_000.0).to(DEVICE)
    q_weight, k_weight = (0.1 * rand(HEAD_DIM)).to(dtype), (0.1 * rand(HEAD_DIM)).to(dtype)
    for positions in (torch.tensor([0, 4095, 17, 300], device=DEVICE), torch.arange(700, 1900, device=DEVICE)):
        tokens = positions.shape[0]
        q_gate = rand(tokens, HEADS * (1 + has_gate) * HEAD_DIM).to(dtype)
        k = rand(tokens, KV_HEADS * HEAD_DIM).to(dtype)
        args = (q_gate, k, q_weight, k_weight, cache, positions, 1e-6, HEADS, KV_HEADS, HEAD_DIM, rotary_dim, has_gate)
        got, want = triton.fused_qk_gemma_rmsnorm_rope_gate(*args), ref.fused_qk_gemma_rmsnorm_rope_gate(*args)
        close("attn.qk_norm_rope_gate@1", got[0], want[0])
        close("attn.qk_norm_rope_gate@1", got[1], want[1])
        assert got[0].shape == (tokens, HEADS * HEAD_DIM) and got[1].shape == (tokens, KV_HEADS * HEAD_DIM)
        if not has_gate:
            assert got[2] is None and want[2] is None
            continue
        assert got[2] is not None and want[2] is not None and torch.equal(got[2], want[2])
        attn_output = rand(tokens, HEADS * HEAD_DIM).to(dtype)
        want_gated = ref.fused_sigmoid_mul(attn_output, want[2])
        gated = triton.fused_sigmoid_mul(mine := attn_output.clone(), got[2])
        assert gated is mine
        close("act.sigmoid_mul@1", gated, want_gated)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_triton_gdn_prefill_kernels_stay_within_their_definitions(triton: Any, dtype: torch.dtype) -> None:
    """Four sequences in one extend, two of them continuing from their states, one of a single
    token and one longer than two chunks of the delta rule. The chunked rule takes no fp32."""
    gen = torch.Generator(device="cuda").manual_seed(1)
    rand: Callable[..., torch.Tensor] = lambda *shape: torch.randn(*shape, device=DEVICE, generator=gen)
    lens = [70, 1, 133, 64]
    total = sum(lens)
    weight, A_log, dt_bias = (0.3 * rand(CONV_DIM, WIDTH)).to(dtype), (0.5 * rand(HV)).to(dtype), (0.5 * rand(HV)).to(dtype)
    conv0, ssm0 = (0.5 * rand(9, CONV_DIM, WIDTH - 1)).to(torch.bfloat16), 0.1 * rand(9, HV, V, K)
    slots = torch.tensor([4, 2, 8, 6], device=DEVICE, dtype=torch.int32)
    cu = torch.tensor([0, 70, 71, 204, 268], device=DEVICE, dtype=torch.int32)
    has_initial = torch.tensor([False, True, True, False], device=DEVICE)
    x = rand(total, CONV_DIM).to(dtype)

    conv_ref, conv_tri = conv0.clone(), conv0.clone()
    want = ref.causal_conv1d_fn(x.transpose(0, 1), weight, None, "silu", conv_ref, has_initial, slots, cu)
    got = triton.causal_conv1d_fn(x.transpose(0, 1), weight, None, "silu", conv_tri, has_initial, slots, cu, lens)
    close("gdn.conv1d_fn@1", got, want)
    assert torch.equal(conv_tri, conv_ref)
    # Without the host lengths the solution reads them from the device: the same result.
    again = triton.causal_conv1d_fn(x.transpose(0, 1), weight, None, "silu", conv0.clone(), has_initial, slots, cu)
    assert torch.equal(again, got)

    a, b = rand(total, HV).to(dtype), rand(total, HV).to(dtype)
    g, beta = ref.fused_gdn_gating(A_log, a, b, dt_bias)
    for got_gate, want_gate in zip(triton.fused_gdn_gating(A_log, a, b, dt_bias), (g, beta), strict=True):
        close("gdn.gating@1", got_gate, want_gate)

    mixed = want.transpose(0, 1).contiguous()
    q, k, v = (t.contiguous() for t in torch.split(mixed, [HK * K, HK * K, HV * V], dim=-1))
    q, k, v = q.view(1, total, HK, K), k.view(1, total, HK, K), v.view(1, total, HV, V)
    ssm_ref, ssm_tri = ssm0.clone(), ssm0.clone()
    if dtype == torch.float32:
        with pytest.raises(AssertionError, match="does not support float32"):
            triton.chunk_gated_delta_rule(q, k, v, g, beta, ssm_tri, slots, cu)
        return
    want_rule = ref.chunk_gated_delta_rule(q, k, v, g, beta, ssm_ref, slots, cu)[0]
    close("gdn.chunk_delta_rule@1", triton.chunk_gated_delta_rule(q, k, v, g, beta, ssm_tri, slots, cu)[0], want_rule)
    close("gdn.chunk_delta_rule@1", ssm_tri, ssm_ref)


def kv_pool(gen: torch.Generator, rows: int, tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """K and V of ``rows`` sequences of ``tokens`` at scattered token slots, and their table
    (row 0 and slot 0 are padding)."""
    slots = rows * tokens + 1
    k_cache = torch.randn(slots, KV_HEADS, HEAD_DIM, device=DEVICE, generator=gen).to(torch.bfloat16)
    v_cache = torch.randn(slots, KV_HEADS, HEAD_DIM, device=DEVICE, generator=gen).to(torch.bfloat16)
    req_to_token = torch.zeros(rows + 1, tokens + 8, dtype=torch.int32, device=DEVICE)
    req_to_token[1:, :tokens] = (torch.randperm(slots - 1, device=DEVICE, generator=gen) + 1).view(rows, tokens)
    return k_cache, v_cache, req_to_token


def test_flashinfer_paged_attention_stays_within_its_definitions() -> None:
    """Decode and extend against the reference operators over the same token lists: eagerly, and
    over a graph bucket's fixed buffers, planned first by FlashInfer's own plan and then, with
    other lengths, by the fast one."""
    from kernels.flashinfer import ops

    gen = torch.Generator(device="cuda").manual_seed(2)
    k_cache, v_cache, req_to_token = kv_pool(gen, 8, 300)
    workspace = ops.Workspace()
    shape = (HEADS, KV_HEADS, HEAD_DIM, torch.bfloat16, torch.bfloat16, DEVICE)
    scaling = HEAD_DIM ** -0.5

    def layout(rows: list[int], lengths: list[int], out: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
        kv_indptr_cpu = indptr_of(lengths)
        kv_indptr = kv_indptr_cpu.to(DEVICE)
        if out is None:
            out = torch.empty(int(kv_indptr_cpu[-1]), dtype=torch.int32, device=DEVICE)
        ref.create_flashinfer_kv_indices(req_to_token, torch.tensor(rows, device=DEVICE),
                                         torch.tensor(lengths, device=DEVICE), kv_indptr, out)
        return kv_indptr_cpu, kv_indptr, out

    eager, want = ops.PagedDecode(workspace, *shape), ref_paged.PagedDecode(*shape)
    for rows, lengths in (([3, 1, 7, 2, 5], [200, 1, 77, 300, 33]), ([4, 8], [300, 150]), ([6], [9])):
        q = torch.randn(len(rows), HEADS, HEAD_DIM, device=DEVICE, generator=gen).to(torch.bfloat16)
        qo = torch.arange(len(rows) + 1, dtype=torch.int32)
        for operator in (eager, want):
            operator.plan(qo, *layout(rows, lengths))
        close("attn.paged_decode@1", eager.run(q, k_cache, v_cache, scaling), want.run(q, k_cache, v_cache, scaling))

    size = 4
    buffers = {"kv_indptr": torch.zeros(size + 1, dtype=torch.int32, device=DEVICE),
               "kv_indices": torch.zeros(size * req_to_token.shape[1], dtype=torch.int32, device=DEVICE),
               "kv_last_page_len": torch.ones(size, dtype=torch.int32, device=DEVICE)}
    bucket = ops.PagedDecode(workspace, *shape, buffers)
    for rows, lengths in (([2, 5, 0, 0], [10, 20, 1, 1]), ([1, 2, 3, 4], [300, 1, 250, 7]), ([8, 0, 0, 0], [123, 1, 1, 1])):
        q = torch.randn(size, HEADS, HEAD_DIM, device=DEVICE, generator=gen).to(torch.bfloat16)
        kv_indptr_cpu, kv_indptr, kv_indices = layout(rows, lengths, buffers["kv_indices"])
        buffers["kv_indptr"].copy_(kv_indptr)
        qo = torch.arange(size + 1, dtype=torch.int32)
        bucket.plan(qo, kv_indptr_cpu, buffers["kv_indptr"], kv_indices)
        want.plan(qo, kv_indptr_cpu, kv_indptr, kv_indices)
        close("attn.paged_decode@1", bucket.run(q, k_cache, v_cache, scaling), want.run(q, k_cache, v_cache, scaling))

    extend, want_extend = ops.PagedExtend(workspace, *shape), ref_paged.PagedExtend(*shape)
    for rows, prefix, new in (([4, 6, 8], [0, 120, 40], [64, 50, 1]), ([1], [0], [300])):
        q = torch.randn(sum(new), HEADS, HEAD_DIM, device=DEVICE, generator=gen).to(torch.bfloat16)
        for operator in (extend, want_extend):
            operator.plan(indptr_of(new), *layout(rows, [p + n for p, n in zip(prefix, new, strict=True)]))
        close("attn.paged_extend@1", extend.run(q, k_cache, v_cache, scaling),
              want_extend.run(q, k_cache, v_cache, scaling))
    with pytest.raises(ValueError, match="not replayed from a graph"):
        ops.PagedExtend(workspace, *shape, buffers)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_flashinfer_rmsnorm_stays_within_its_definition(dtype: torch.dtype) -> None:
    """The hidden states of a decode step and of a prefill step, and the per-head Q/K norm; with
    a residual the solution writes the sum over ``residual`` and the norm over ``x``, and returns
    those two tensors."""
    from kernels.flashinfer import ops

    gen = torch.Generator(device="cuda").manual_seed(3)
    for rows, width in ((8, 5120), (2000, 5120), (8 * HEADS, HEAD_DIM)):
        x = torch.randn(rows, width, device=DEVICE, generator=gen).to(dtype)
        residual = torch.randn(rows, width, device=DEVICE, generator=gen).to(dtype)
        weight = (0.1 * torch.randn(width, device=DEVICE, generator=gen)).to(dtype)
        plain, want_plain = ops.gemma_rms_norm(x, weight, 1e-6), ref.gemma_rms_norm(x, weight, 1e-6)
        assert isinstance(plain, torch.Tensor) and isinstance(want_plain, torch.Tensor)
        close("norm.gemma_rms@1", plain, want_plain)
        fused, want_fused = ops.gemma_rms_norm(mine := x.clone(), weight, 1e-6, mine_sum := residual.clone()), \
            ref.gemma_rms_norm(x, weight, 1e-6, residual)
        assert isinstance(fused, tuple) and isinstance(want_fused, tuple)
        (got, got_sum), (want, want_sum) = fused, want_fused
        assert got is mine and got_sum is mine_sum
        close("norm.gemma_rms@1", got, want)
        assert torch.equal(got_sum, want_sum)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_flashinfer_silu_and_mul_stays_within_its_definition(dtype: torch.dtype) -> None:
    """The MLP's activation at Qwen3.8-27B's intermediate size, for a decode step and a prefill
    step (FlashInfer's takes no fp32)."""
    from kernels.flashinfer import ops

    gen = torch.Generator(device="cuda").manual_seed(4)
    for rows in (8, 2000):
        x = torch.randn(rows, 2 * 17408, device=DEVICE, generator=gen).to(dtype)
        got = ops.silu_and_mul(x)
        assert got.shape == (rows, 17408) and got.dtype == dtype
        close("act.silu_and_mul@1", got, ref.silu_and_mul(x))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_cuda_kv_store_stays_within_its_definition(dtype: torch.dtype) -> None:
    """``cuda.kv.store@1`` against the reference, bit for bit: K and V as the attention layer hands
    them over (K its own tensor, V a strided slice of the QKV projection), one token, a decode
    batch and a prefill step, int64 and int32 slots; the slots it was not given stay as they were.
    Where the kernel does not take the tensors (another dtype than the pool's, a row that is not
    whole 16-byte vectors) the solution still stores what the reference stores."""
    from kernels.cuda import ops

    module = ops.build()
    gen = torch.Generator(device="cuda").manual_seed(7)
    slots = 4096
    q_width = 2 * HEADS * HEAD_DIM

    def pools(kv_heads: int = KV_HEADS, head_dim: int = HEAD_DIM) -> tuple[torch.Tensor, ...]:
        k = torch.randn(slots, kv_heads, head_dim, device=DEVICE, generator=gen).to(dtype)
        v = torch.randn(slots, kv_heads, head_dim, device=DEVICE, generator=gen).to(dtype)
        return k, v, k.clone(), v.clone()

    for tokens in (1, 13, 2000):
        qkv = torch.randn(tokens, q_width + 2 * KV_HEADS * HEAD_DIM, device=DEVICE, generator=gen).to(dtype)
        k = torch.randn(tokens, KV_HEADS, HEAD_DIM, device=DEVICE, generator=gen).to(dtype)
        v = qkv[:, q_width + KV_HEADS * HEAD_DIM:].view(tokens, KV_HEADS, HEAD_DIM)
        assert tokens == 1 or not v.is_contiguous()
        for loc_dtype in (torch.int64, torch.int32):
            loc = (torch.randperm(slots - 1, device=DEVICE, generator=gen)[:tokens] + 1).to(loc_dtype)
            got_k, got_v, want_k, want_v = pools()
            assert module.store(got_k.clone(), got_v.clone(), loc, k, v)  # the kernel takes these: no fallback below
            ops.kv_store(module, got_k, got_v, loc, k, v)
            ref.kv_store(want_k, want_v, loc.long(), k, v)
            assert torch.equal(got_k, want_k) and torch.equal(got_v, want_v)

    # The fallbacks: the model's K/V in another dtype than the pool, and rows of 12 bytes.
    loc = torch.tensor([5, 9, 2], device=DEVICE)
    other = torch.float32 if dtype != torch.float32 else torch.bfloat16
    k, v = (torch.randn(3, KV_HEADS, HEAD_DIM, device=DEVICE, generator=gen).to(other) for _ in range(2))
    got_k, got_v, want_k, want_v = pools()
    ops.kv_store(module, got_k, got_v, loc, k, v)
    ref.kv_store(want_k, want_v, loc, k, v)
    assert torch.equal(got_k, want_k) and torch.equal(got_v, want_v)
    width = 12 // torch.empty(0, dtype=dtype).element_size()
    k, v = (torch.randn(3, 1, width, device=DEVICE, generator=gen).to(dtype) for _ in range(2))
    got_k, got_v, want_k, want_v = pools(1, width)
    assert not module.store(got_k.clone(), got_v.clone(), loc, k, v)
    ops.kv_store(module, got_k, got_v, loc, k, v)
    ref.kv_store(want_k, want_v, loc, k, v)
    assert torch.equal(got_k, want_k) and torch.equal(got_v, want_v)


def test_triton_kv_indices_stays_within_its_definition(triton: Any) -> None:
    """``triton.attn.kv_indices@1``: slot lists and optional per-row last slot, keeping the tail of
    a padded graph bucket buffer untouched."""
    gen = torch.Generator(device="cuda").manual_seed(9)
    req_to_token = torch.randint(1, 50000, (16, 1024), device=DEVICE, generator=gen, dtype=torch.int32)
    rows = torch.tensor([5, 1, 9, 0, 12], device=DEVICE, dtype=torch.int64)
    lengths = [1, 513, 64, 1, 700]  # spans multiple 512-token Triton blocks and 1-token padded rows
    lens = torch.tensor(lengths, device=DEVICE, dtype=torch.int64)
    kv_indptr = indptr_of(lengths).to(DEVICE)
    buf_len = sum(lengths) + 37
    got_idx = torch.full((buf_len,), -7, device=DEVICE, dtype=torch.int32)
    want_idx = got_idx.clone()
    got_last = torch.full((len(lengths),), -1, device=DEVICE, dtype=torch.int64)
    want_last = got_last.clone()
    triton.create_flashinfer_kv_indices(req_to_token, rows, lens, kv_indptr, got_idx, got_last)
    ref.create_flashinfer_kv_indices(req_to_token, rows, lens, kv_indptr, want_idx, want_last)
    assert torch.equal(got_idx, want_idx) and torch.equal(got_last, want_last)
    got_idx2 = torch.full((buf_len,), -3, device=DEVICE, dtype=torch.int32)
    want_idx2 = got_idx2.clone()
    triton.create_flashinfer_kv_indices(req_to_token, rows, lens, kv_indptr, got_idx2)
    ref.create_flashinfer_kv_indices(req_to_token, rows, lens, kv_indptr, want_idx2)
    assert torch.equal(got_idx2, want_idx2)


# ------------------------------------------------------------------ the worker stack, 4 layers in bf16


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return Path(os.environ["QWEN38_MODEL"]) if os.environ.get("QWEN38_MODEL") else make_checkpoint(tmp_path_factory, 4)


def worker(checkpoint: Path, graphs: bool, **config: Any) -> Worker:
    cfg = json.loads((checkpoint / "config.json").read_text())
    return Worker(cfg, platform=CudaGraphPlatform() if graphs else TorchEagerPlatform(cuda=True), **{
        "model.dtype": "bf16", "state.gdn.conv_dtype": "bf16", "weights.path": str(checkpoint),
        "state.kv.page_size": 16, "state.kv.pages": 256, "state.gdn.slots": 8, "batch.max_rows": 4,
        "batch.max_context_len": 512, "plan.cuda_graph": "required" if graphs else "off", **config})


def state(w: Worker) -> list[torch.Tensor]:
    kv, gdn = w.service("state.kv.paged@1"), w.service("state.gdn.slots@1")
    return [*(kv.bind(layer).buffers()[name][kv.page_size:(kv.pages + 1) * kv.page_size].float()
              for layer in kv.k_buffer for name in ("k", "v")),
            gdn.conv[:, 1:gdn.size + 1].float(), gdn.temporal[:, 1:gdn.size + 1].float()]


PROMPTS = {0: [9707, 11, 1879, 0, 3555, 374, 279, 1888], 1: list(range(1000, 1040)), 2: [151, 2]}


def real_kernels(w: Worker, graphs: bool) -> None:
    """``auto`` chose the real kernels for every phase of ``w``."""
    resolved = w.service("plan@1").resolved
    tables = ("prefill", "decode", "graph") if graphs else ("prefill", "decode")
    assert {resolved[t]["layers.attn@1"][d] for t, d in zip(tables, (
        "attn.paged_extend@1", "attn.paged_decode@1", "attn.paged_decode@1"), strict=False)} == {
        "flashinfer.attn.paged_extend@1", "flashinfer.attn.paged_decode@1"}
    assert {resolved[t]["layers.attn@1"]["kv.store@1"] for t in tables} == {"cuda.kv.store@1"}
    assert {resolved[t]["layers.attn@1"]["attn.kv_indices@1"] for t in tables} == {"triton.attn.kv_indices@1"}
    gdn = {s for t in tables for d, s in resolved[t]["layers.gdn@1"].items() if d != "backend"}
    assert gdn and all(s.startswith("triton.gdn.") for s in gdn), gdn
    assert resolved["decode"]["layers.gdn@1"] == {
        "backend": "auto", "gdn.qkvzba_conv1d_update@1": "triton.gdn.qkvzba_conv1d_update@1",
        "gdn.packed_decode@1": "triton.gdn.packed_decode@1", "gdn.track_states@1": "triton.gdn.track_states@1"}
    catalog = w.service("kernels.catalog@1")  # the layers' kernels are chosen at their first call
    for definition, shape, solution in (
            ("norm.gemma_rms@1", {"hidden": 5120}, "flashinfer.norm.gemma_rms@1"),
            ("norm.rms_gated@1", {"hidden": 128}, "triton.norm.rms_gated@1"),
            ("act.silu_and_mul@1", {"hidden": 17408, "fp32": 0}, "flashinfer.act.silu_and_mul@1"),
            ("act.sigmoid_mul@1", {"hidden": 6144}, "triton.act.sigmoid_mul@1"),
            ("attn.qk_norm_rope_gate@1", {"head_dim": 256, "rotary_dim": 64, "fp32": 0},
             "triton.attn.qk_norm_rope_gate@1")):
        assert catalog.select(definition, shape).solution == solution


ROUNDS: tuple[dict[int, int], ...] = ({0: 8, 1: 40, 2: 2}, {2: 1, 0: 1, 1: 1}, {1: 1}, {0: 1, 1: 1}, {1: 1, 2: 1, 0: 1},
                                      {2: 1}, {0: 1, 2: 1}, {0: 1, 1: 1, 2: 1})
"""Three rows prefill, then decode three, one, two, three, one, two and three at a time."""


@pytest.mark.timeout(1800)
def test_auto_runs_the_real_kernels_and_a_replayed_step_is_the_eager_step(checkpoint: Path) -> None:
    """With a bucket for every number of rows a replayed step runs what the eager step runs, at
    the same batch size: the logits and the states agree to bf16 rounding and the greedy tokens
    are the same, also after a row leaves and another takes its place."""
    eager, graph = worker(checkpoint, False), worker(checkpoint, True, **{"plan.graph_buckets": [1, 2, 3, 4]})
    try:
        real_kernels(eager, False)
        real_kernels(graph, True)
        assert graph.service("plan@1").buckets("decode") == [1, 2, 3, 4]
        a, b = Driver(eager), Driver(graph)
        for d in (a, b):
            for row, prompt in PROMPTS.items():
                d.add(row, prompt)
        for chunks in ROUNDS:
            x, y = a.step(chunks), b.step(chunks)
            torch.testing.assert_close(y.logits.float(), x.logits.float(), rtol=2e-2, atol=2e-2)
            assert y.finish() == x.finish()
        for d in (a, b):
            d.remove(1)
            d.add(1, [8, 3, 7, 6])
        for chunks in ({1: 4, 0: 1}, {1: 1, 0: 1, 2: 1}, {1: 1, 0: 1, 2: 1}):
            x, y = a.step(chunks), b.step(chunks)
            torch.testing.assert_close(y.logits.float(), x.logits.float(), rtol=2e-2, atol=2e-2)
            assert y.finish() == x.finish()
        for x, y in zip(state(eager), state(graph), strict=True):
            torch.testing.assert_close(y, x, rtol=2e-2, atol=2e-2)
    finally:
        eager.close()
        graph.close()
        free_gpu()


@pytest.mark.timeout(1800)
def test_a_padded_bucket_gives_the_real_rows_what_they_get_alone(checkpoint: Path) -> None:
    """Three rows replayed from the bucket of four: one lane is padding, with no GDN slot and
    the padding row's KV. The GEMMs then run four rows where the eager step runs three, and a
    BLAS library may take another kernel for another number of rows (a SM100 does), so the logits
    are held to their direction, and the greedy tokens to being the same wherever the eager
    step's two best candidates are not nearly tied (stage2-contract.md §1.3)."""
    eager, graph = worker(checkpoint, False), worker(checkpoint, True)
    try:
        assert graph.service("plan@1").buckets("decode") == [1, 2, 4]
        a, b = Driver(eager), Driver(graph)
        for d in (a, b):
            for row, prompt in PROMPTS.items():
                d.add(row, prompt)
        for chunks in ROUNDS:
            x, y = a.step(chunks), b.step(chunks)
            lx, ly = x.logits.float(), y.logits.float()
            assert bool((torch.nn.functional.cosine_similarity(lx, ly, dim=-1) > 0.999).all())
            best = lx.topk(2, dim=-1).values
            tied = (best[:, 0] - best[:, 1]) < 0.5
            tx, ty = list(x.finish().tokens.values()), list(y.finish().tokens.values())
            parted = [i for i, (p, q) in enumerate(zip(tx, ty, strict=True)) if p != q]
            assert all(bool(tied[i]) for i in parted), (chunks, parted)
            if parted:  # from a near tie on, the two runs decode different texts
                break
    finally:
        eager.close()
        graph.close()
        free_gpu()


def test_what_the_graphs_keep_is_counted_alike_cold_and_after_a_reload(checkpoint: Path) -> None:
    """FlashInfer's workspace (384 MiB, for every step, eager ones too) is allocated before the
    graphs' memory is counted: a budget below it still records the graphs on a cold start, and a
    reload of the plan alone counts what the cold start counted."""
    budget = 200 << 20
    w = worker(checkpoint, True, **{"plan.graph_memory_bytes": budget})
    try:
        cold = w.service("plan@1").resolved["cuda_graph"]
        assert cold["buckets"] == [1, 2, 4] and 0 < cold["kept_bytes"] < budget, cold
        w.runtime.reload(ReloadPlan(units_to_reload=[spec("core.plan")])).unwrap()
        again = w.service("plan@1").resolved["cuda_graph"]
        assert again["buckets"] == [1, 2, 4], again
        assert abs(again["kept_bytes"] - cold["kept_bytes"]) <= 16 << 20, (cold, again)
    finally:
        w.close()
        free_gpu()


def test_graphs_that_are_refused_or_only_checked_leave_nothing_in_the_backends(checkpoint: Path) -> None:
    """Without a bucket of one row, the graph recorded for the check alone is closed and the
    backends keep nothing for it. Graphs over their budget are closed, and the backends keep no
    bucket; eager steps go on with the same kernels."""
    w = worker(checkpoint, True, **{"plan.graph_buckets": [2, 4]})
    try:
        attn, gdn = w.service("layers.attn@1"), w.service("layers.gdn@1")
        assert w.service("plan@1").buckets("decode") == [2, 4]
        assert sorted(attn.paged.graph_metadata) == [2, 4] and sorted(gdn.graph_metadata) == [2, 4]
        assert sorted(key for key in attn.paged.operators if key[0] == "graph") == [("graph", 2), ("graph", 4)]
        assert not [name for name in (*attn.binding.buffers(), *gdn.binding.buffers()) if name.split(".")[-2] == "1"]
    finally:
        w.close()
        free_gpu()
    w = worker(checkpoint, True, **{"plan.cuda_graph": "auto", "plan.graph_memory_bytes": 4096})
    try:
        attn, gdn = w.service("layers.attn@1"), w.service("layers.gdn@1")
        refused = w.service("plan@1").resolved["cuda_graph"]
        assert refused["buckets"] == [] and "plan.graph_memory_bytes" in refused["reason"], refused
        assert not attn.paged.graph_metadata and not gdn.graph_metadata and not attn.binding.buffers()
        assert not [key for key in attn.paged.operators if key[0] == "graph"]
        d = Driver(w)
        d.add(0, PROMPTS[0])
        d.generate([0], 4)
        assert len(d.requests[0].outputs) == 4
    finally:
        w.close()
        free_gpu()


def test_a_cache_of_another_16_bit_dtype_than_the_model_is_not_flashinfers(checkpoint: Path) -> None:
    """FlashInfer loads a 16-bit cache as the queries' type: a bf16 model over an fp16 cache would
    read wrong numbers. ``auto`` leaves that combination to the native kernels, which convert;
    named, FlashInfer fails the start."""
    mixed = {"state.kv.dtype": "f16"}
    w = worker(checkpoint, True, **{"plan.cuda_graph": "auto", **mixed})
    try:
        resolved = w.service("plan@1").resolved
        assert resolved["decode"]["layers.attn@1"]["attn.decode@1"] == "reference.attn.decode@1 (fallback)"
        assert resolved["prefill"]["layers.attn@1"]["attn.extend@1"] == "reference.attn.extend@1 (fallback)"
        assert resolved["graph"]["layers.attn@1"]["attn.decode@1"] == "torch_graph.attn.decode@1"
        d = Driver(w)
        d.add(0, PROMPTS[0])
        d.generate([0], 4)
    finally:
        w.close()
        free_gpu()
    with pytest.raises(Exception, match=r"attn.backend = 'flashinfer': backend 'flashinfer' has no solution of "
                                        r"attn\.paged_\w+@1 for \{'head_dim': 256, 'fp32': 0, 'kv_as_q': 0\}"):
        worker(checkpoint, False, **{"attn.backend": "flashinfer", **mixed})
    free_gpu()


@pytest.mark.timeout(1800)
def test_the_kernel_units_reload_under_running_requests(checkpoint: Path) -> None:
    """The kernel units are reloaded, with the plan that recorded graphs over them, between two
    decode steps of live requests: the catalog hands out the new instances' solutions, the
    attention operators are made again, the graphs are recorded again, and generation goes on to
    the tokens of a worker that never reloaded."""
    steady, reloaded = worker(checkpoint, True), worker(checkpoint, True)
    try:
        a, b = Driver(steady), Driver(reloaded)
        for d in (a, b):
            for row in (0, 2):
                d.add(row, PROMPTS[row])
            d.step({0: 8, 2: 2})
            for _ in range(3):
                d.step({0: 1, 2: 1})
        attn = reloaded.service("layers.attn@1")
        before = attn.active.operators["graph", 2][0]
        version = reloaded.service("kernels.catalog@1").version
        for _ in range(2):
            reloaded.runtime.reload(ReloadPlan(units_to_reload=[
                spec("kernels.triton"), spec("kernels.flashinfer"), spec("kernels.cuda"),
                spec("core.plan")])).unwrap()
        assert reloaded.service("kernels.catalog@1").version > version
        assert reloaded.service("plan@1").buckets("decode") == [1, 2, 4]
        for _ in range(4):
            x, y = a.step({0: 1, 2: 1}), b.step({0: 1, 2: 1})
            torch.testing.assert_close(y.logits.float(), x.logits.float(), rtol=2e-2, atol=2e-2)
            assert y.finish() == x.finish()
        assert attn.active.operators["graph", 2][0] is not before
    finally:
        steady.close()
        reloaded.close()
        free_gpu()


def test_the_backends_start_when_named_as_sglang_names_them(checkpoint: Path) -> None:
    """``--attention-backend flashinfer --linear-attn-backend triton``. FlashInfer brings no store
    into the KV pool (in SGLang that is the pool's, under every backend): it runs as under
    ``auto``."""
    w = worker(checkpoint, True, **{"attn.backend": "flashinfer", "gdn.backend": "triton"})
    try:
        resolved = w.service("plan@1").resolved
        for table, definition in (("prefill", "attn.paged_extend@1"), ("decode", "attn.paged_decode@1"),
                                  ("graph", "attn.paged_decode@1")):
            assert resolved[table]["layers.attn@1"] == {
                "backend": "flashinfer", "kv.store@1": "cuda.kv.store@1",
                "attn.kv_indices@1": "triton.attn.kv_indices@1",
                definition: f"flashinfer.{definition}"}
            assert resolved[table]["layers.gdn@1"]["backend"] == "triton"
        d = Driver(w)
        d.add(0, PROMPTS[0])
        d.generate([0], 4)
        assert len(d.requests[0].outputs) == 4
    finally:
        w.close()
        free_gpu()


@pytest.mark.parametrize("key", ["attn.backend", "attn.decode_backend"])
def test_flashinfer_named_for_an_fp32_model_fails_the_start(checkpoint: Path, key: str) -> None:
    """FlashInfer takes no fp32: named explicitly it is a hard constraint (§5.5.3), and the start
    fails saying so instead of running the reference in its place. ``auto`` runs fp32 on the
    native kernels (tests/gpu/test_qwen38_partial.py)."""
    with pytest.raises(Exception, match=rf"{key} = 'flashinfer': backend 'flashinfer' has no solution of "
                                        r"attn\.paged_\w+@1 for \{'head_dim': 256, 'fp32': 1, 'kv_as_q': 1\}"):
        worker(checkpoint, False, **{"model.dtype": "f32", "state.gdn.conv_dtype": "f32", key: "flashinfer"})
    free_gpu()
