"""The message envelope and schema registry (doc/specs/README.md, "消息信封").

Every cross-process message is a versioned ``msgspec.Struct`` wrapped in an
``Envelope``. An unknown schema or a version mismatch is rejected, never guessed.
"""

from __future__ import annotations

import msgspec

from core.common.ids import Endpoint, Epoch
from core.common.result import Ok, Result, fail


class Envelope(msgspec.Struct, frozen=True):
    schema: str
    """Message type name and version, e.g. ``"sched.step_plan@1"``."""
    sender: Endpoint
    epoch: Epoch | None
    body: bytes
    """msgpack-encoded message body."""


class SchemaRegistry:
    """Maps ``"name@version"`` to a message type. One registry per process."""

    def __init__(self) -> None:
        self._types: dict[str, type[msgspec.Struct]] = {}
        self._names: dict[type[msgspec.Struct], str] = {}
        self._decoders: dict[str, msgspec.msgpack.Decoder[msgspec.Struct]] = {}
        self._encoder = msgspec.msgpack.Encoder()

    def register(self, schema: str, type_: type[msgspec.Struct]) -> None:
        known = self._types.get(schema)
        if known is type_:
            return
        if known is not None:
            raise ValueError(f"schema {schema!r} already registered for {known.__qualname__}")
        if type_ in self._names:
            raise ValueError(f"{type_.__qualname__} already registered as {self._names[type_]!r}")
        self._types[schema] = type_
        self._names[type_] = schema
        self._decoders[schema] = msgspec.msgpack.Decoder(type_)

    def schema_of(self, msg: msgspec.Struct) -> str:
        try:
            return self._names[type(msg)]
        except KeyError:
            raise KeyError(f"message type {type(msg).__qualname__} is not registered") from None

    def wrap(self, msg: msgspec.Struct, sender: Endpoint, epoch: Epoch | None = None) -> Envelope:
        return Envelope(self.schema_of(msg), sender, epoch, self._encoder.encode(msg))

    def unwrap(self, env: Envelope) -> Result[msgspec.Struct]:
        decoder = self._decoders.get(env.schema)
        if decoder is None:
            name = env.schema.partition("@")[0]
            known = sorted(s for s in self._types if s.partition("@")[0] == name)
            hint = f"this process knows {known}" if known else "no version of this schema is known"
            return fail(
                "ipc", "decode", f"unknown schema {env.schema!r}", subject=str(env.sender),
                hint=hint, code="ipc.unknown_schema",
            )
        try:
            return Ok(decoder.decode(env.body))
        except msgspec.DecodeError as e:
            return fail(
                "ipc", "decode", f"malformed {env.schema!r}: {e}", subject=str(env.sender),
                code="ipc.malformed",
            )

    def known(self) -> list[str]:
        return sorted(self._types)


_ENVELOPE_ENCODER = msgspec.msgpack.Encoder()
_ENVELOPE_DECODER = msgspec.msgpack.Decoder(Envelope)


def encode_envelope(env: Envelope) -> bytes:
    return _ENVELOPE_ENCODER.encode(env)


def decode_envelope(data: bytes) -> Result[Envelope]:
    try:
        return Ok(_ENVELOPE_DECODER.decode(data))
    except msgspec.DecodeError as e:
        return fail("ipc", "decode", f"not an envelope: {e}", code="ipc.malformed")

