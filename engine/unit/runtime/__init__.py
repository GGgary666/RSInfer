"""The unit runtime ``unit.runtime`` (doc/specs/unit-runtime.md)."""

from unit.runtime.context import UnitContext
from unit.runtime.errors import (
    InactiveContextError,
    InjectedFault,
    MissingServiceError,
    RevocationError,
    ServiceConflictError,
    StaleServiceError,
    UndeclaredServiceError,
    UnitRuleError,
)
from unit.runtime.registry import Registry, current_instance
from unit.runtime.runtime import ESCALATE_ENGINE, ESCALATE_PROCESS, UnitRuntime
from unit.runtime.types import (
    InstanceInfo,
    InstanceState,
    ProcessPlan,
    ReloadPlan,
    ReloadReport,
    RuntimeSnapshot,
    UnitSpec,
)

__all__ = [
    "ESCALATE_ENGINE",
    "ESCALATE_PROCESS",
    "InactiveContextError",
    "InjectedFault",
    "InstanceInfo",
    "InstanceState",
    "MissingServiceError",
    "ProcessPlan",
    "Registry",
    "ReloadPlan",
    "ReloadReport",
    "RevocationError",
    "RuntimeSnapshot",
    "ServiceConflictError",
    "StaleServiceError",
    "UndeclaredServiceError",
    "UnitContext",
    "UnitRuleError",
    "UnitRuntime",
    "UnitSpec",
    "current_instance",
]
