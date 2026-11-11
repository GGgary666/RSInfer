"""The paged attention backend of ``layers.attn`` (``PagedAttnBackend``, SGLang's
``FlashInferAttnBackend``) on the CPU. No library provides paged kernels there, so the backend
never chooses that family by itself: the test puts it on the reference operators, and the worker
stack must generate what it generates with the native attention kernels, however the rows are
batched and chunked. With FlashInfer the same backend is tested in ``tests/gpu``."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

from kernels.reference.paged import PagedDecode
from layers.attn.paged import gather_kv_indices, indptr_of
from tests.units.support.worker import Driver, Worker, hf_model, save_checkpoint, tiny_config

PROMPTS = {0: [5, 17, 3, 44, 9, 61, 2], 1: [8, 30, 30, 1, 77, 12, 40, 3, 19, 6, 55], 2: [4]}
STEPS = 6


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], Path]:
    cfg = tiny_config()
    return cfg.to_dict(), save_checkpoint(hf_model(cfg), tmp_path_factory.mktemp("ckpt"))


def paged(w: Worker) -> dict[str, Any]:
    """Put both phases of ``w``'s attention on the paged family; what it then resolved."""
    attn = w.service("layers.attn@1")
    attn._paged = lambda phase, shape: True  # pyright: ignore[reportUnknownLambdaType]
    return dict(attn.resolve(w.service("model@1").assembly.states))


@pytest.fixture
def workers(checkpoint: tuple[dict[str, Any], Path]) -> Iterator[tuple[Worker, Worker]]:
    native, other = (Worker(checkpoint[0], **{"weights.path": str(checkpoint[1])}) for _ in range(2))
    try:
        resolved = paged(other)
        assert resolved["decode"]["attn.paged_decode@1"] == "reference.attn.paged_decode@1 (fallback)"
        assert resolved["prefill"]["attn.paged_extend@1"] == "reference.attn.paged_extend@1 (fallback)"
        yield native, other
    finally:
        native.close()
        other.close()


def run(w: Worker, rows: list[int], chunk: int | None) -> tuple[dict[int, list[int]], dict[int, list[torch.Tensor]]]:
    d = Driver(w)
    for row in rows:
        d.add(row, PROMPTS[row])
    logits = d.generate(rows, STEPS, chunk)
    return {row: d.requests[row].outputs for row in rows}, logits


@pytest.mark.parametrize(("rows", "chunk"), [([0], None), ([1], 3), ([0, 1, 2], None), ([0, 1, 2], 4)])
def test_the_paged_backend_generates_what_the_native_one_generates(workers: tuple[Worker, Worker], rows: list[int],
                                                                  chunk: int | None) -> None:
    """Whole and chunked prefills, mixed steps (a short row decodes while a long one prefills),
    and decode steps: the same tokens, and the same logits to the last bits of fp32."""
    native, other = workers
    want_tokens, want_logits = run(native, rows, chunk)
    tokens, logits = run(other, rows, chunk)
    assert tokens == want_tokens
    for row in rows:
        torch.testing.assert_close(torch.stack(logits[row]), torch.stack(want_logits[row]), rtol=1e-5, atol=1e-5)
    attn = other.service("layers.attn@1")
    assert isinstance(attn.active.operators["decode", 0][0], PagedDecode)
    assert set(attn.active.operators) == {("prefill", 0), ("decode", 0)}


def test_a_planned_operator_is_made_again_after_the_kernel_catalog_changed(workers: tuple[Worker, Worker]) -> None:
    """An operator is made by a solution: once the catalog's registrations moved (a kernel unit
    reloaded), the next step makes its operators anew instead of running the old code's."""
    _, other = workers
    run(other, [0], None)
    attn, catalog = other.service("layers.attn@1"), other.service("kernels.catalog@1")
    before = attn.active.operators["decode", 0][0]
    catalog._changed()
    run(other, [0], None)
    assert attn.active.operators["decode", 0][0] is not before


def test_the_token_lists_are_each_rows_first_slots_in_row_order() -> None:
    req_to_token = torch.arange(40, dtype=torch.int32).reshape(4, 10) + 100
    rows = torch.tensor([3, 1, 2])
    kv_indptr = indptr_of([2, 5, 1])
    assert kv_indptr.tolist() == [0, 2, 7, 8] and kv_indptr.dtype == torch.int32
    got = gather_kv_indices(req_to_token, rows, kv_indptr, 8)
    assert got.tolist() == [130, 131, 110, 111, 112, 113, 114, 120] and got.dtype == torch.int32
    buffer = torch.full((12,), -1, dtype=torch.int32)  # a graph bucket's: filled in place, the rest untouched
    assert gather_kv_indices(req_to_token, rows, kv_indptr, 8, buffer) is buffer
    assert buffer.tolist() == [130, 131, 110, 111, 112, 113, 114, 120, -1, -1, -1, -1]
