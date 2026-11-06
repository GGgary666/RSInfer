"""ipc.md §5 test 3: open records."""

from __future__ import annotations

import msgspec

from core.common.result import Failed, Ok
from core.ipc import RecordBuilder, decode_record, encode_record, read_sections


class Batch(msgspec.Struct, frozen=True):
    reqs: list[int]


class AttnMeta(msgspec.Struct, frozen=True):
    seq_lens: list[int]


class Spec(msgspec.Struct, frozen=True):
    draft_len: int


def test_new_sections_do_not_disturb_old_readers() -> None:
    old = RecordBuilder("sched.step_plan@1", epoch=7).add("sched.batch@1", Batch([1, 2])).build()
    new = (RecordBuilder("sched.step_plan@1", epoch=8)
           .add("sched.batch@1", Batch([3]))
           .add("attn.metadata@1", AttnMeta([5]))
           .add("spec.verify@1", Spec(4))
           .build())
    for rec in (old, new):
        got = read_sections(decode_record(encode_record(rec)), {"sched.batch@1": Batch})
        assert isinstance(got, Ok) and list(got.value) == ["sched.batch@1"]
    got = read_sections(new, {"sched.batch@1": Batch}, {"attn.metadata@1": AttnMeta})
    assert got == Ok({"sched.batch@1": Batch([3]), "attn.metadata@1": AttnMeta([5])})


def test_missing_required_section_is_an_error() -> None:
    rec = RecordBuilder("sched.step_plan@1", epoch=1).add("sched.batch@1", Batch([])).build()
    got = read_sections(rec, {"sched.batch@1": Batch, "attn.metadata@1": AttnMeta})
    assert isinstance(got, Failed) and got.has_code("ipc.missing_section")
    assert read_sections(rec, {"sched.batch@1": Batch}, {"attn.metadata@1": AttnMeta}) == Ok(
        {"sched.batch@1": Batch([])})


def test_rank_indexed_records() -> None:
    per_rank = [RecordBuilder("sched.step_plan@1", 3, rank=r).add("sched.batch@1", Batch([r])).build()
                for r in range(4)]
    assert [decode_record(encode_record(r)).rank for r in per_rank] == [0, 1, 2, 3]
