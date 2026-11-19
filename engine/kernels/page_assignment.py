"""Expand a batch of physical page IDs directly into the persistent request-to-token table."""
# pyright: reportMissingImports=false, reportUntypedFunctionDecorator=false
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit  # pyright: ignore[reportAttributeAccessIssue]
def _assign(Table, Packet, row_stride, nrows, PAGE: tl.constexpr, BLOCK: tl.constexpr):
    seq = tl.program_id(1)
    row = tl.load(Packet + seq * 4)
    dest_page = tl.load(Packet + seq * 4 + 1)
    begin = tl.load(Packet + seq * 4 + 2)
    count = tl.load(Packet + seq * 4 + 3)
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < count * PAGE
    page = tl.load(Packet + 4 * nrows + begin + i // PAGE, mask=valid, other=0) + 1
    tl.store(Table + row.to(tl.int64) * row_stride + dest_page * PAGE + i, page * PAGE + i % PAGE, valid)


def assign_pages(table: torch.Tensor, packet: torch.Tensor, nrows: int, max_pages: int, page_size: int) -> None:
    _assign[(triton.cdiv(max_pages * page_size, 512), nrows)](  # pyright: ignore[reportIndexIssue, reportAttributeAccessIssue]
        table, packet, table.stride(0), nrows, PAGE=page_size, BLOCK=512)
