"""DS 1017 (DeepSeek-V4.1) on the GPU against the reference implementation shipped with its checkpoint
(``ds1017_reference``; doc/specs/v1-ds1017.md), on a random checkpoint in the real checkpoint's
formats and a small config with every kind of layer (``support_ds1017.small_config``).

- With that implementation's own kernels and torch calls as the layers' kernels (``dsv4.backend =
  tilelang``), the engine's logits are its logits, bit for bit: the prompt prefilled in one piece and
  every decode step after it. That holds the model, the layers and the stores to it. The prompts are
  longer than the window: for a shorter one that implementation lists a token's keys in other
  columns (``layers.ds1017``), which is another last bit and, in a model with random weights, soon
  another token.
- The PyTorch references (``kernels.ds1017``) against those kernels, definition by definition,
  within each definition's tolerance. End to end they are not comparable to that implementation on
  random weights: a GEMM's last bit moves a routing decision a few layers on.
- With the references, a token's result does not depend on how its step is laid out: chunks of a
  prompt, other rows in the batch, a decode step replayed from a CUDA graph. Bit for bit.

Needs ``tilelang`` (0.1.12) in the GPU environment."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

from kernels import ds1017_formats as formats
from platforms.cuda_graph import CudaGraphPlatform
from tests.gpu.support import free_gpu
from tests.gpu.support_ds1017 import reference_greedy, reference_model, small_config, worker
from tests.gpu.support_ds1017_cases import SMALL, cases, compare, observe
from tests.units.support.ds1017 import checkpoint as make_checkpoint
from tests.units.support.ds1017 import driver, expert_stacks

CONTEXT = 1024
ROWS: dict[str, Any] = {"batch.max_context_len": CONTEXT, "state.kv.pages": 1024, "state.swa.slots": 6, "batch.max_rows": 4}


def prompt(tokens: int, seed: int) -> list[int]:
    return torch.randint(3, 4096, (tokens,), generator=torch.Generator().manual_seed(seed)).tolist()


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_checkpoint(tmp_path_factory.mktemp("ds1017"), small_config(), seed=1)


def generate(checkpoint: Path, prompts: dict[int, list[int]], steps: int, config: dict[str, Any], chunk: int | None = None,
             platform: Any = None) -> tuple[dict[int, list[int]], dict[int, list[torch.Tensor]]]:
    w = worker(small_config(), checkpoint, platform, **{**ROWS, **config})
    try:
        d = driver(w)
        for row, p in prompts.items():
            d.add(row, p)
        logits = d.generate(list(prompts), steps, chunk)
        return {row: list(d.requests[row].outputs) for row in prompts}, {r: [x.clone() for x in v] for r, v in logits.items()}
    finally:
        w.close()
        free_gpu()


def same(a: dict[int, list[torch.Tensor]], b: dict[int, list[torch.Tensor]]) -> None:
    for row in a:
        for step, (x, y) in enumerate(zip(a[row], b[row], strict=True)):
            assert torch.equal(x, y), f"row {row}, step {step}: differs by {float((x - y).abs().max())}"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(("tokens", "steps"), [(300, 40), (517, 24)])
def test_the_engine_computes_what_the_reference_implementation_computes(checkpoint: Path, tokens: int, steps: int) -> None:
    """300 tokens: past the window, more latents than the indexers keep, more blocks than the
    candidate layer keeps. 517: an odd length, so the two-to-one compression leaves a group open."""
    given = prompt(tokens, 7)
    reference = reference_model(checkpoint, CONTEXT)
    want_tokens, want_logits = reference_greedy(reference, given, steps)
    del reference
    free_gpu()
    tokens_out, logits = generate(checkpoint, {0: given}, steps, {"dsv4.backend": "tilelang"})
    assert tokens_out[0] == want_tokens
    same({0: logits[0]}, {0: want_logits})


def inputs(name: str) -> tuple[Any, ...]:
    """Arguments for one definition, in this model's small shapes, on the GPU."""
    g = torch.Generator().manual_seed(len(name))

    def rand(*shape: int, scale: float = 1.0, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return (torch.randn(*shape, generator=g) * scale).to(dtype).cuda()

    def fp8(n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        data, scale = formats.quantize_fp8(torch.randn(n, k, generator=g) * k ** -0.5)
        return data.cuda(), scale.cuda()

    f32 = torch.float32
    match name:
        case "ds.fake_quant_fp8@1":
            return (rand(70, 1024) * torch.exp(rand(70, 1, dtype=f32) * 3).to(torch.bfloat16), 32)
        case "ds.fake_quant_fp4@1":
            return (rand(70, 128), 16, True)
        case "ds.linear_fp8@1":
            data, scale = fp8(2048, 1024)
            return (rand(70, 1024), data, scale, formats.pack_scales(formats.block_rows(scale, 2048)))
        case "ds.linear@1":
            return (rand(70, 1024), rand(128, 1024, scale=1 / 32, dtype=f32))
        case "ds.grouped_linear@1":
            return (rand(70, 8, 256), rand(8, 256, 256, scale=1 / 16))
        case "ds.index_score@1":
            return (rand(1, 70, 16, 64), rand(1, 150, 64), rand(1, 70, 16, scale=0.1))
        case "ds.rms_norm@1":
            return (rand(70, 1024), 1 + rand(1024, scale=0.1), 1e-20)
        case "ds.sparse_attn@1":
            idx = torch.randint(-1, 400, (70, 190), generator=g).cuda()
            return (rand(70, 16, 128, scale=0.3), rand(400, 128, scale=0.3), idx, rand(16, dtype=f32), 128 ** -0.5)
        case "ds.cache_attn@1":
            window = formats.pack_kv(rand(6 * 32, 128, scale=0.3)).view(6, 32, -1)
            latents = formats.pack_latent(rand(400, 128, scale=0.3)).view(1, 400, -1)
            return (rand(70, 16, 128, scale=0.3), window, torch.randint(-1, 6 * 32, (70, 32), generator=g).cuda(), latents,
                    torch.randint(-1, 400, (70, 158), generator=g).cuda(), rand(16, dtype=f32), 128 ** -0.5)
        case "ds.hc_mixes@1":
            return (rand(70, 4, 1024), rand(24, 4096, scale=1 / 64, dtype=f32), 1 + rand(3, scale=0.2, dtype=f32),
                    rand(24, scale=0.5, dtype=f32), 1e-20, 20, 1e-6)
        case "ds.hc_pre@1":
            return (rand(70, 4, 1024), torch.rand(70, 4, generator=g).cuda())
        case "ds.hc_post@1":
            return (rand(70, 1024), rand(70, 4, 1024), torch.rand(70, 4, generator=g).cuda(), torch.rand(70, 4, 4, generator=g).cuda())
        case "ds.moe_gate@1":
            return (rand(70, 1024), rand(8, 1024, scale=1 / 32, dtype=f32), rand(8, scale=0.1, dtype=f32), 2, 1.5)
        case "ds.moe_experts@1":
            x = rand(70, 1024)
            weights = torch.rand(70, 2, generator=g).cuda()
            indices = torch.stack([torch.randperm(8, generator=g)[:2] for _ in range(70)]).cuda()
            stacks: list[torch.Tensor] = []
            for n, k in ((1024, 1024), (1024, 1024), (1024, 1024)):
                pairs = [formats.quantize_fp4(torch.randn(n, k, generator=g) * k ** -0.5) for _ in range(8)]
                stacks += [torch.stack([p[0] for p in pairs]).cuda(), torch.stack([p[1] for p in pairs]).cuda()]
            w1, s1, w2, s2, w3, s3 = stacks
            return (x, weights, indices, *expert_stacks(w1, s1, w2, s2, w3, s3), 10.0)
        case "ds.silu@1":
            return (rand(70, 1024, dtype=f32),)
        case "ds.pool@1":
            return (rand(70, 2, 128, dtype=f32), rand(70, 2, 128, dtype=f32))
        case "ds.engram_gate@1":
            return (rand(70, 4, 1024), rand(70, 5 * 1024), 1 + rand(4, 1024, scale=0.1), 1 + rand(4, 1024, scale=0.1), 1e-20)
        case "ds.top@1":
            return (rand(70, 300, dtype=f32), 64)
        case _:
            raise KeyError(name)


def test_the_references_are_within_tolerance_of_the_reference_implementations_kernels(checkpoint: Path) -> None:
    w = worker(small_config(), checkpoint, **ROWS)
    try:
        catalog = w.service("kernels.catalog@1")
        compared: list[str] = []
        over_caches = cases(SMALL, lambda name: catalog.resolve(f"tilelang.{name}"))
        for definition in (d for d in catalog.definitions() if d.name.startswith("ds.")):
            solution = f"tilelang.{definition.name}"
            if solution not in catalog.solutions(definition.name):
                continue  # elementwise, or the same torch call: one solution for both
            mine = [c for c in over_caches if c.split()[0] == definition.name]
            if mine:  # a composition over the caches: on what it returns and what it writes
                for case in mine:
                    report = compare(case, observe(case, catalog.resolve(definition.reference), over_caches[case]),
                                     observe(case, catalog.resolve(solution), over_caches[case]), definition.tolerance,
                                     ties=2e-3)
                    print(report)
                compared.append(definition.name)
                continue
            args = inputs(definition.name)
            want: Any = catalog.resolve(solution)(*args)
            got: Any = catalog.resolve(definition.reference)(*args)
            tolerance = definition.tolerance
            for a, b in zip(got if isinstance(got, tuple) else (got,), want if isinstance(want, tuple) else (want,), strict=True):
                if definition.name == "ds.top@1":  # the same set: neither promises an order
                    a, b = a.sort(dim=-1).values, b.sort(dim=-1).values
                if a.dtype == torch.int64 or not (tolerance.atol or tolerance.rtol):
                    assert torch.equal(a, b), definition.name
                else:
                    torch.testing.assert_close(a.float(), b.float(), atol=tolerance.atol, rtol=tolerance.rtol,
                                               msg=lambda m, name=definition.name: f"{name}: {m}")
            compared.append(definition.name)
        assert len(compared) >= 25, compared
    finally:
        w.close()
        free_gpu()


@pytest.mark.timeout(1800)
def test_a_tokens_result_does_not_depend_on_how_its_step_is_laid_out(checkpoint: Path) -> None:
    """The references on the GPU: a prompt in one piece, in chunks (33 cuts a compression group in two),
    one token a step, and beside rows of other lengths."""
    prompts = {0: prompt(300, 7), 1: prompt(141, 8), 2: prompt(17, 9)}
    backend: dict[str, Any] = {"dsv4.backend": "reference"}
    whole = generate(checkpoint, {0: prompts[0]}, 8, backend)
    for chunk in (33, 100):
        pieces = generate(checkpoint, {0: prompts[0]}, 8, backend, chunk)
        assert pieces[0] == whole[0]
        same(pieces[1], whole[1])
    short = generate(checkpoint, {1: prompts[1]}, 8, backend)
    one_by_one = generate(checkpoint, {1: prompts[1]}, 8, backend, 1)
    assert one_by_one[0] == short[0]
    same(one_by_one[1], short[1])
    together = generate(checkpoint, prompts, 8, backend)
    assert together[0][0] == whole[0][0] and together[0][1] == short[0][1]
    same({0: together[1][0], 1: together[1][1]}, {0: whole[1][0], 1: short[1][1]})


@pytest.mark.timeout(1800)
def test_decode_steps_replayed_from_graphs_give_what_eager_steps_give(checkpoint: Path) -> None:
    """Graphs required: a step no graph holds would fail. Buckets of 1, 2 and 4 for three rows."""
    prompts = {0: prompt(300, 7), 1: prompt(141, 8), 2: prompt(17, 9)}
    eager = generate(checkpoint, prompts, 12, {"dsv4.backend": "reference", "plan.cuda_graph": "off"})
    replayed = generate(checkpoint, prompts, 12, {"dsv4.backend": "reference", "plan.cuda_graph": "required"},
                        platform=CudaGraphPlatform())
    assert replayed[0] == eager[0]
    same(replayed[1], eager[1])


def per_row(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> Iterator[int]:
    whole = fn(x)
    for size in (1, 7, 16):
        if not torch.equal(torch.cat([fn(x[a:a + size]) for a in range(0, x.shape[0], size)]), whole):
            yield size


def test_library_products_and_reductions_follow_the_batch_and_the_references_do_not() -> None:
    """Why the references sum and multiply themselves: on this GPU a library product of fewer rows,
    and a mean of 16 rows or fewer, round the last bit another way."""
    from kernels.ds1017 import ops

    x, w = torch.randn(300, 1024, device="cuda"), torch.randn(128, 1024, device="cuda")
    assert list(per_row(lambda t: torch.nn.functional.linear(t, w), x))
    assert list(per_row(lambda t: t.square().mean(-1), x))
    assert not list(per_row(lambda t: ops.matmul(t, w), x))
    assert not list(per_row(lambda t: ops.total(t.square()), x))
    assert not list(per_row(lambda t: ops.rms_norm(t.bfloat16(), w[0].bfloat16(), 1e-20), x))
