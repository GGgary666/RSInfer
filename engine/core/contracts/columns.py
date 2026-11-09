"""Columnar arrays for per-step records (stage2-contract.md §4.5).

A step record carries one value per request row. Building one Python object per request
per step is the CPU cost vLLM V0 paid (stage2-contract.md §4); a column is one typed byte
string instead, built from ``array.array`` and read back without copying, e.g. with
``torch.frombuffer(memoryview(col.data), dtype=...)``. Standard library only.
"""

from __future__ import annotations

import sys
from array import array
from collections.abc import Iterable
from types import MappingProxyType
from typing import Literal

import msgspec

type DType = Literal["u8", "i32", "i64", "f32"]

_TYPECODE = MappingProxyType({"u8": "B", "i32": "i", "i64": "q", "f32": "f"})
_ITEMSIZE = MappingProxyType({"u8": 1, "i32": 4, "i64": 8, "f32": 4})
assert all(array(c).itemsize == _ITEMSIZE[d] for d, c in _TYPECODE.items())
assert sys.byteorder == "little", "columns are little-endian on the wire"


class Column(msgspec.Struct, frozen=True, array_like=True):
    dtype: DType
    data: bytes = b""

    @staticmethod
    def of(dtype: DType, values: Iterable[int] | Iterable[float] = ()) -> Column:
        return Column(dtype, array(_TYPECODE[dtype], values).tobytes())

    def values(self) -> array[int] | array[float]:
        a: array[int] | array[float] = array(_TYPECODE[self.dtype])
        a.frombytes(self.data)
        return a

    def __len__(self) -> int:
        return len(self.data) // _ITEMSIZE[self.dtype]


def ragged(dtype: DType, lists: Iterable[Iterable[int]]) -> tuple[Column, Column]:
    """``(offsets, flat)``: row ``i`` owns ``flat[offsets[i]:offsets[i + 1]]``."""
    offsets, flat = array("i", [0]), array(_TYPECODE[dtype])
    for xs in lists:
        flat.extend(xs)
        offsets.append(len(flat))
    return Column("i32", offsets.tobytes()), Column(dtype, flat.tobytes())
