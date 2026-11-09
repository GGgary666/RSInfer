"""``UnitContext``: everything a unit may touch goes through ``ctx``.

Every registration made through ``ctx`` records its undo function on the
instance; unloading runs them in reverse order (unit-runtime.md §5).
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from core.common.ids import InstanceId, UnitId
from core.common.manifest import UnitManifest
from core.context.config import ConfigView, UndeclaredConfigKey
from core.context.config import plan_scope as _plan_scope
from core.context.flags import RuntimeFlags
from platforms.base import GuardedPlatform, Stream
from unit.runtime.errors import (
    InactiveContextError,
    MissingServiceError,
    RevocationError,
    UndeclaredServiceError,
    UnitRuleError,
)
from unit.runtime.proxy import ServiceProxy
from unit.runtime.registry import running_as
from unit.runtime.types import DerivedArtifact, InstanceState

if TYPE_CHECKING:
    from unit.runtime.runtime import UnitRuntime, _Slot


class UnitContext:
    def __init__(self, runtime: UnitRuntime, slot: _Slot, instance: InstanceId, config: ConfigView) -> None:
        self._rt = runtime
        self._slot = slot
        self.instance = instance
        self.config = config

    @property
    def id(self) -> UnitId:
        return self.instance.unit

    @property
    def manifest(self) -> UnitManifest:
        return self._slot.spec.manifest

    @property
    def flags(self) -> RuntimeFlags:
        return self._rt.flags

    @property
    def platform(self) -> GuardedPlatform:
        if self._rt.platform is None:
            raise UnitRuleError("this process has no platform")
        return GuardedPlatform(
            self._rt.platform,
            may_capture=self.manifest.capabilities.captures_graph,
            who=self.id,
        )

    def _check_live(self) -> None:
        slot = self._slot
        if slot.instance != self.instance or slot.state not in (InstanceState.LOADING, InstanceState.ACTIVE):
            raise InactiveContextError(f"context of {self.instance} is no longer active")

    def _record(self, label: str, undo: Callable[[], object]) -> Callable[[], None]:
        if not callable(undo):
            raise RevocationError(f"{self.id}: {label} did not return an undo function")
        return self._rt._add_registration(self._slot, label, undo)

    # -- services -----------------------------------------------------------

    def provide(self, name: str, obj: object) -> None:
        self._check_live()
        if name not in self.manifest.provides.services:
            raise UndeclaredServiceError(f"{self.id} provides {name!r}, which is not in [provides]")
        prov = self._rt._provide(self._slot, name, obj)
        self._record(f"provide({name})", lambda: self._rt._revoke(prov))

    def use(self, name: str) -> Any:
        """A required service. Its provider is fixed when the instance loads."""
        self._check_live()
        if name not in self.manifest.requires.services:
            raise UndeclaredServiceError(f"{self.id} uses {name!r}, which is not in requires.services")
        prov = self._slot.uses.get(name)
        if prov is None:
            raise MissingServiceError(f"{self.id}: required service {name!r} has no provider")
        return ServiceProxy(prov)

    def get(self, name: str) -> Any | None:
        """An optional service, or ``None``. Looked up at call time."""
        self._check_live()
        if name in self.manifest.requires.services:
            return self.use(name)
        if name not in self.manifest.requires.optional:
            raise UndeclaredServiceError(f"{self.id} gets {name!r}, which is not in requires.optional")
        prov = self._rt._services.get(name)
        if prov is None or prov.revoked:
            return None
        self._slot.uses[name] = prov
        return ServiceProxy(prov)

    # -- revocable registrations ----------------------------------------------

    def effect(self, setup: Callable[[], Callable[[], object]], label: str = "effect") -> Callable[[], None]:
        """Run ``setup()`` now; the undo function it returns runs on unload.

        Returns a function that undoes the effect early (at most once).
        """
        self._check_live()
        with running_as(self.instance):
            undo = setup()
        return self._record(label, undo)

    def register(self, registry: str, item: object, label: str | None = None) -> Callable[[], None]:
        """Register ``item`` with another unit's registry service; it must return an undo."""
        self._check_live()
        svc = self.use(registry) if registry in self.manifest.requires.services else self.get(registry)
        if svc is None:
            raise MissingServiceError(f"{self.id}: registry {registry!r} is not available")
        with running_as(self.instance):
            undo = svc.register(item)
        return self._record(label or f"register({registry})", undo)

    def on_config(self, key: str, handler: Callable[[Any], None]) -> None:
        """Accept new values of ``key`` in place, at the epoch boundary."""
        self._check_live()
        if key not in self.manifest.readable_keys():
            raise UndeclaredConfigKey(f"{self.id} hooks {key!r}, which is not in its manifest")
        if not self.manifest.capabilities.accept_config:
            raise UnitRuleError(f"{self.id} registers on_config but does not declare capabilities.accept_config")
        instance = self.instance

        def run(value: Any) -> None:
            with running_as(instance):
                handler(value)

        self._record(f"on_config({key})", self._rt.config.add_hook(key, instance, run))

    def on_action(self, name: str, handler: object) -> None:
        """Take part in changes as ``name`` (e.g. ``"rebuild_graphs"``, ``"refit"``).

        ``handler`` has ``prepare(args)``, ``execute(args)``, ``commit()`` and
        ``abort()``; draining units also ``start_drain(args)``, ``drained()`` and
        ``drain_deadline() -> dict[str, int]`` (at the deadline, settle what is left by the unit's
        own policy and say how many requests went which way, or raise to fail the change). A drain
        hears ``commit()`` or ``abort()`` too, on the instance serving then, even if it raised. ``prepare`` runs before the pause while the old
        version keeps serving, the rest at the epoch boundary. ``prepare`` either does
        everything at once (returns ``None``; serving stops meanwhile) or returns an
        iterator whose every ``next()`` does one small slice between two steps
        (change.md §4.6); it must not change what is serving. If the same change
        reloads this unit, ``prepare`` runs at the safe point instead, on the new
        instance, so one instance sees every phase. Raising fails the change, and
        every participant rolls back; ``abort()`` is called once and must also undo
        a ``prepare`` or ``execute`` that raised halfway.
        """
        self._check_live()
        self._record(f"on_action({name})", self._rt._add_action(self._slot, name, handler))

    # -- resources ----------------------------------------------------------

    def lease_stream(self, name: str, device: int = 0) -> Stream:
        self._check_live()
        stream, release = self._rt.resources.lease_stream(self.instance, name, device)
        self._record(f"lease_stream({name}@{device})", release)
        return stream

    def lease_buffer[T](self, name: str, factory: Callable[[], T]) -> T:
        self._check_live()
        buf, release = self._rt.resources.lease_buffer(self.instance, name, factory)
        self._record(f"lease_buffer({name})", release)
        return buf

    def derived(self, name: str, *, rebuild: str, config: Iterable[str] = (), units: Iterable[str] = ()) -> None:
        """Register a derived artifact this unit built (a plan, a graph, a cached table).

        ``config``: keys recorded into it by value (for a graph, the ``config`` entries of
        ``graph.values``); ``units``: units whose code it was built with; ``rebuild``: the
        ``ctx.on_action`` that rebuilds it next to the old one. ``core.deps`` reads these
        to decide what an upstream change rebuilds (modules.html, "派生"). Registering the
        same name again (after a rebuild) replaces the record.
        """
        self._check_live()
        artifact = DerivedArtifact(self.id, name, rebuild, sorted(set(config)),
                                   sorted(UnitId(u) for u in set(units)))
        undo = self._rt._add_derived(self._slot, artifact)
        if undo is not None:
            self._record(f"derived({name})", undo)

    @contextmanager
    def plan_scope(self, plan_id: str) -> Generator[None]:
        """Config reads inside are recorded against ``plan_id`` (plan units only)."""
        self._check_live()
        if not self.manifest.capabilities.captures_graph:
            raise UnitRuleError(f"{self.id} opens a plan scope but is not a plan unit")
        with _plan_scope(plan_id):
            yield
