"""Types shared by all framework components."""

from core.common.ids import Endpoint, Epoch, HostId, Identity, InstanceId, ProcId, UnitId
from core.common.result import Diagnostic, Failed, FailedError, Ok, Result, diag, fail

__all__ = [
    "Diagnostic",
    "Endpoint",
    "Epoch",
    "Failed",
    "FailedError",
    "HostId",
    "Identity",
    "InstanceId",
    "Ok",
    "ProcId",
    "Result",
    "UnitId",
    "diag",
    "fail",
]
