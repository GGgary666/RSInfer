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

from kernels.reference import ops as reference
from kernels.reference.paged import PagedDecode
from layers.attn.paged import indptr_of
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
    assert isinstance(attn.active.operators["decode", 0, False][0], PagedDecode)
    assert set(attn.active.operators) == {("prefill", 0, False), ("decode", 0, False)}


def test_a_planned_operator_is_made_again_after_the_kernel_catalog_changed(workers: tuple[Worker, Worker]) -> None:
    """An operator is made by a solution: once the catalog's registrations moved (a kernel unit
    reloaded), the next step makes its operators anew instead of running the old code's."""
    _, other = workers
    run(other, [0], None)
    attn, catalog = other.service("layers.attn@1"), other.service("kernels.catalog@1")
    before = attn.active.operators["decode", 0, False][0]
    catalog._changed()
    run(other, [0], None)
    assert attn.active.operators["decode", 0, False][0] is not before


def test_the_kv_indices_reference_lays_out_the_same_lists_and_each_rows_last_slot() -> None:
    """``attn.kv_indices@1`` (what ``prepare`` lays the lists out with): each row's first slots, row
    after row, and each row's last slot. It reads nothing on the host, so a graph bucket's buffer
    past the lists keeps what it held, and a row alone gets what it gets among others."""
    req_to_token = torch.arange(40, dtype=torch.int32).reshape(4, 10) + 100
    rows, lens = torch.tensor([3, 1, 2]), torch.tensor([2, 5, 1])
    kv_indptr = indptr_of([2, 5, 1])
    assert kv_indptr.tolist() == [0, 2, 7, 8] and kv_indptr.dtype == torch.int32
    buffer = torch.full((12,), -1, dtype=torch.int32)
    last = torch.zeros(3, dtype=torch.int64)
    reference.create_flashinfer_kv_indices(req_to_token, rows, lens, kv_indptr, buffer, last)
    assert buffer.tolist() == [130, 131, 110, 111, 112, 113, 114, 120, -1, -1, -1, -1]
    assert last.tolist() == [131, 114, 120]
    for r in range(3):
        alone, one = torch.full((5,), -1, dtype=torch.int32), torch.zeros(1, dtype=torch.int64)
        reference.create_flashinfer_kv_indices(req_to_token, rows[r:r + 1], lens[r:r + 1], indptr_of([int(lens[r])]),
                                               alone, one)
        s, e = int(kv_indptr[r]), int(kv_indptr[r + 1])
        assert alone[:e - s].tolist() == buffer[s:e].tolist() and int(one) == int(last[r])
