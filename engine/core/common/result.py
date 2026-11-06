"""Results and diagnostics (doc/specs/README.md, "结果和错误").

Framework operations return ``Ok(value)`` or ``Failed(diagnostics)``; exceptions
never cross a process boundary.
"""

from __future__ import annotations

from typing import Literal, NoReturn

import msgspec

Severity = Literal["error", "warning", "info"]


class Diagnostic(msgspec.Struct, frozen=True):
    """One finding: which component, which phase, about what, why, and what to do."""

    component: str
    phase: str
    reason: str
    subject: str = ""
    """The unit, process or key the finding is about."""
    hint: str = ""
    """An actionable suggestion."""
    code: str = ""
    """Stable machine-readable code, e.g. ``"unit.leak"``."""
    severity: Severity = "error"

    def __str__(self) -> str:
        head = f"[{self.component}/{self.phase}]"
        if self.code:
            head += f" {self.code}"
        text = f"{head} {self.subject + ': ' if self.subject else ''}{self.reason}"
        if self.hint:
            text += f" (hint: {self.hint})"
        return text


class Ok[T](msgspec.Struct, frozen=True, tag=True):
    value: T
    diagnostics: list[Diagnostic] = []
    """Non-fatal findings (warnings) gathered on the way."""

    @property
    def ok(self) -> Literal[True]:
        return True

    def unwrap(self) -> T:
        return self.value


class Failed(msgspec.Struct, frozen=True, tag=True):
    diagnostics: list[Diagnostic]

    @property
    def ok(self) -> Literal[False]:
        return False

    def unwrap(self) -> NoReturn:
        raise FailedError(self)

    def has_code(self, code: str) -> bool:
        return any(d.code == code for d in self.diagnostics)

    def __str__(self) -> str:
        return "\n".join(str(d) for d in self.diagnostics) or "failed"


type Result[T] = Ok[T] | Failed


class FailedError(Exception):
    """Raised by ``Failed.unwrap()``; for tests and in-process callers only."""

    def __init__(self, failed: Failed) -> None:
        super().__init__(str(failed))
        self.failed = failed


def diag(
    component: str,
    phase: str,
    reason: str,
    *,
    subject: str = "",
    hint: str = "",
    code: str = "",
    severity: Severity = "error",
) -> Diagnostic:
    return Diagnostic(component, phase, reason, subject, hint, code, severity)


def fail(
    component: str,
    phase: str,
    reason: str,
    *,
    subject: str = "",
    hint: str = "",
    code: str = "",
) -> Failed:
    return Failed([diag(component, phase, reason, subject=subject, hint=hint, code=code)])
