"""Open records (doc/specs/ipc.md §3, P2).

A record such as StepPlan or StepResult is not one fixed struct: each unit
contributes its own versioned section. Readers decode only the sections they
know; a missing required section is an error, extra sections are ignored.
"""

from __future__ import annotations

from typing import Any

import msgspec

from core.common.ids import Epoch
from core.common.result import Diagnostic, Failed, Ok, Result, diag

_encoder = msgspec.msgpack.Encoder()


class OpenRecord(msgspec.Struct, frozen=True):
    schema: str
    """e.g. ``"sched.step_plan@1"``"""
    epoch: Epoch
    sections: dict[str, bytes]
    """Section name (``"attn.metadata@1"``) -> msgpack-encoded section."""
    rank: int | None = None
    """Set for rank-indexed records (P3): one record per rank."""


class RecordBuilder:
    def __init__(self, schema: str, epoch: Epoch, rank: int | None = None) -> None:
        self._schema = schema
        self._epoch = epoch
        self._rank = rank
        self._sections: dict[str, bytes] = {}

    def add(self, name: str, section: Any) -> RecordBuilder:
        if name in self._sections:
            raise ValueError(f"section {name!r} added twice")
        self._sections[name] = _encoder.encode(section)
        return self

    def build(self) -> OpenRecord:
        return OpenRecord(self._schema, self._epoch, dict(self._sections), self._rank)


def read_sections(
    record: OpenRecord,
    required: dict[str, type[Any]],
    optional: dict[str, type[Any]] | None = None,
) -> Result[dict[str, Any]]:
    """Decode the sections this reader knows; ignore the rest."""
    out: dict[str, Any] = {}
    errors: list[Diagnostic] = []
    for name, type_ in [*required.items(), *(optional or {}).items()]:
        raw = record.sections.get(name)
        if raw is None:
            if name in required:
                errors.append(diag("ipc", "decode", f"{record.schema} lacks required section {name!r}",
                                   subject=f"epoch {record.epoch}", code="ipc.missing_section"))
            continue
        try:
            out[name] = msgspec.msgpack.decode(raw, type=type_)
        except msgspec.DecodeError as e:
            errors.append(diag("ipc", "decode", f"section {name!r}: {e}", code="ipc.malformed"))
    return Failed(errors) if errors else Ok(out)


def encode_record(record: OpenRecord) -> bytes:
    return _encoder.encode(record)


def decode_record(data: bytes) -> OpenRecord:
    return msgspec.msgpack.decode(data, type=OpenRecord)
