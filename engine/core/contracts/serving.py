"""Routing section supplied by the frontend alongside Requests.

The scheduler opens a token writer to this address when it first receives a request.
Addresses are supplied by deployment data / the existing step-stream factory. A
frontend incarnation's request ids are distinct, so late replies cannot reach a new session.
"""

from typing import ClassVar

import msgspec

from core.ipc.stream import StreamAddress


class ReturnRoute(msgspec.Struct, frozen=True):
    SECTION: ClassVar[str] = "frontend.return_route@1"
    tokens: StreamAddress
