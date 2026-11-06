"""Errors raised inside a process when a unit breaks the unit rules."""


class UnitRuleError(RuntimeError):
    """A unit used the runtime in a way its manifest does not allow."""


class StaleServiceError(UnitRuleError):
    """A proxy to a provider that has since been unloaded was used."""


class UndeclaredServiceError(UnitRuleError):
    """``use``/``get``/``provide`` of a service not in the manifest."""


class ServiceConflictError(UnitRuleError):
    """Two providers for one service name in one process."""


class MissingServiceError(UnitRuleError):
    """A required service has no provider (should not happen once ACTIVE)."""


class InactiveContextError(UnitRuleError):
    """A unit context was used after its instance stopped being LOADING/ACTIVE."""


class RevocationError(UnitRuleError):
    """A registration did not produce a callable undo function."""


class InjectedFault(RuntimeError):
    """Raised at a fault point configured by a test."""
