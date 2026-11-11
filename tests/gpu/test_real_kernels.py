"""The real kernels on one CUDA GPU of 24 GB or more: SGLang's Triton GDN kernels
(``kernels.triton``) and FlashInfer's paged attention (``kernels.flashinfer``), at Qwen3.8-27B's
shapes.

Manual (``tests/gpu`` is not collected by default; it needs the ``gpu`` extra, and ``ninja`` and a
CUDA compiler on ``PATH`` for FlashInfer's first plan of a shape, as under ``uv run``):

    uv sync --extra gpu --extra frontend
    uv run pytest tests/gpu/test_real_kernels.py -s

- Each solution against the reference of its definition, on the same inputs and state pools,
  within the definition's tolerance (stage2-contract.md §6).
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
from layers.attn.paged import gather_kv_indices, indptr_of
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
    kernels leave alone: its output is undefined, the pools are untouched."""
    gen = torch.Generator(device="cuda").manual_seed(0)
    rand: Callable[..., torch.Tensor] = lambda *shape: torch.randn(*shape, device=DEVICE, generator=gen)
    weight, A_log, dt_bias = (0.3 * rand(CONV_DIM, WIDTH)).to(dtype), (0.5 * rand(HV)).to(dtype), (0.5 * rand(HV)).to(dtype)
    conv0, ssm0 = (0.5 * rand(9, CONV_DIM, WIDTH - 1)).to(torch.bfloat16), (0.1 * rand(9, HV, V, K)).to(ssm_dtype)
    slots = torch.tensor([3, 1, 7, -1, 5, 2], device=DEVICE, dtype=torch.int32)
    real = slots >= 0
    x = rand(6, CONV_DIM).to(dtype)

    conv_ref, conv_tri = conv0.clone(), conv0.clone()
    want = ref.causal_conv1d_update(x[real].clone(), conv_ref, weight, None, "silu", slots[real])
    got = triton.causal_conv1d_update(x.clone(), conv_tri, weight, None, "silu", slots)
    close("gdn.conv1d_update@1", got[real], want)
    assert torch.equal(conv_tri[1:], conv_ref[1:]) and torch.equal(conv_tri[0], conv0[0])

    q, k, v = (t.contiguous() for t in torch.split(want, [HK * K, HK * K, HV * V], dim=-1))
    lanes = int(real.sum())
    q, k, v = q.view(1, lanes, HK, K), k.view(1, lanes, HK, K), v.view(1, lanes, HV, V)
    a, b = rand(lanes, HV).to(dtype), rand(lanes, HV).to(dtype)
    cu = torch.arange(lanes + 1, device=DEVICE, dtype=torch.int32)
    ssm_ref, ssm_tri = ssm0.clone(), ssm0.clone()
    want = ref.fused_sigmoid_gating_delta_rule_update(A_log, dt_bias, q, k, v, a, b, ssm_ref, slots[real], cu)
    got = triton.fused_sigmoid_gating_delta_rule_update(A_log, dt_bias, q, k, v, a, b, ssm_tri, slots[real], cu)
    close("gdn.recurrent_update@1", got, want)
    close("gdn.recurrent_update@1", ssm_tri, ssm_ref)

    xn, z, wn = rand(lanes * HV, V).to(dtype), rand(lanes * HV, V).to(dtype), rand(V).to(dtype)
    close("norm.rms_gated@1", triton.rms_norm_gated(xn, wn, None, z, 1e-6, None, True),
          ref.rms_norm_gated(xn, wn, None, z, 1e-6, None, True))


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
        return kv_indptr_cpu, kv_indptr, gather_kv_indices(req_to_token, torch.tensor(rows, device=DEVICE), kv_indptr,
                                                           int(kv_indptr_cpu[-1]), out)

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


@pytest.mark.timeout(1800)
def test_auto_runs_the_real_kernels_and_a_replayed_step_is_the_eager_step(checkpoint: Path) -> None:
    """Three rows in a bucket of four (one padded lane), then two, one, and three again; a row
    leaves and another takes its place. The same kernels run eagerly and in the graph, so the
    logits and the states agree to bf16 rounding and the greedy tokens are the same."""
    eager, graph = worker(checkpoint, False), worker(checkpoint, True)
    try:
        for w in (eager, graph):
            resolved = w.service("plan@1").resolved
            tables = ("prefill", "decode", "graph") if w is graph else ("prefill", "decode")
            assert {resolved[t]["layers.attn@1"][d] for t, d in zip(tables, (
                "attn.paged_extend@1", "attn.paged_decode@1", "attn.paged_decode@1"), strict=False)} == {
                "flashinfer.attn.paged_extend@1", "flashinfer.attn.paged_decode@1"}
            gdn = {s for t in tables for d, s in resolved[t]["layers.gdn@1"].items() if d != "backend"}
            assert gdn and all(s.startswith("triton.gdn.") for s in gdn), gdn
        assert graph.service("plan@1").buckets("decode") == [1, 2, 4]
        a, b = Driver(eager), Driver(graph)
        for d in (a, b):
            for row, prompt in PROMPTS.items():
                d.add(row, prompt)
        rounds: list[dict[int, int]] = [{0: 8, 1: 40, 2: 2}, {2: 1, 0: 1, 1: 1}, {1: 1}, {0: 1, 1: 1},
                                       {1: 1, 2: 1, 0: 1}, {2: 1}, {0: 1, 2: 1}, {0: 1, 1: 1, 2: 1}]
        for chunks in rounds:
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
                spec("kernels.triton"), spec("kernels.flashinfer"), spec("core.plan")])).unwrap()
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
                "backend": "flashinfer", "kv.store@1": "reference.kv.store@1 (fallback)",
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
                                        r"attn\.paged_\w+@1 for \{'head_dim': 256, 'fp32': 1\}"):
        worker(checkpoint, False, **{"model.dtype": "f32", "state.gdn.conv_dtype": "f32", key: "flashinfer"})
    free_gpu()
