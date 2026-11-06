"""Service provisions and the proxies handed to users.

A user never holds the provider object itself: ``ctx.use()`` returns a proxy,
and once the provider is unloaded every access through an old proxy raises
``StaleServiceError`` on the spot (unit-runtime.md §4).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from core.common.ids import InstanceId
from unit.runtime.errors import StaleServiceError


class Provision:
    __slots__ = ("name", "owner", "obj", "revoked")

    def __init__(self, name: str, owner: InstanceId, obj: object) -> None:
        self.name = name
        self.owner = owner
        self.obj = obj
        self.revoked = False

    def target(self) -> Any:
        if self.revoked:
            raise StaleServiceError(
                f"service {self.name!r} from {self.owner} was unloaded; "
                "a unit kept a reference across a reload"
            )
        return self.obj

    def __repr__(self) -> str:
        return f"Provision({self.name!r}, {self.owner}{', revoked' if self.revoked else ''})"


class ServiceProxy:
    """Forwards attribute access and the common protocols to the live provider."""

    __slots__ = ("_provision",)

    def __init__(self, provision: Provision) -> None:
        object.__setattr__(self, "_provision", provision)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provision.target(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._provision.target(), name, value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._provision.target()(*args, **kwargs)

    def __getitem__(self, key: Any) -> Any:
        return self._provision.target()[key]

    def __len__(self) -> int:
        return len(self._provision.target())

    def __iter__(self) -> Iterator[Any]:
        return iter(self._provision.target())

    def __contains__(self, item: Any) -> bool:
        return item in self._provision.target()

    def __repr__(self) -> str:
        return f"<proxy {self._provision!r}>"


def provision_of(proxy: ServiceProxy) -> Provision:
    return object.__getattribute__(proxy, "_provision")
