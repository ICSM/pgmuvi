"""Canonical status and failure records for wavelength-advisory attempts.

The maintained wavelength-advisory workflow historically exposed a mixture of
``passed``/``failed`` strings and Boolean aliases.  This module adds orthogonal,
JSON-safe status dimensions without removing those compatibility fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from collections.abc import Mapping
from typing import Any

__all__ = [
    "AttemptDisposition",
    "ComparisonEligibility",
    "DiagnosticValidity",
    "ExecutionStage",
    "ScientificUsability",
    "TechnicalOutcome",
    "WarningSeverity",
    "WavelengthAttemptStatus",
    "WavelengthFailureRecord",
    "coerce_execution_stage",
    "derive_wavelength_attempt_status",
]


class _StringEnum(str, Enum):
    """Enum whose members serialize and display as their string values."""

    def __str__(self) -> str:
        return self.value


class AttemptDisposition(_StringEnum):
    """Whether a configured attempt was actually executed."""

    NOT_ATTEMPTED = "not_attempted"
    ATTEMPTED = "attempted"
    SKIPPED = "skipped"


class ExecutionStage(_StringEnum):
    """Furthest execution stage reached by an attempt."""

    NOT_STARTED = "not_started"
    PRECONDITION = "precondition"
    INGESTION = "ingestion"
    PREPROCESSING = "preprocessing"
    SETUP = "setup"
    CONSENSUS = "consensus"
    OPTIMIZATION = "optimization"
    DIAGNOSTICS = "diagnostics"
    EXPORT = "export"
    COMPLETED = "completed"
    UNKNOWN = "unknown"


class TechnicalOutcome(_StringEnum):
    """Technical completion state of an attempt."""

    NOT_ATTEMPTED = "not_attempted"
    SKIPPED = "skipped"
    FAILED = "failed"
    INITIALIZED_ONLY = "initialized_only"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    COMPLETED_WITH_RECOVERY = "completed_with_recovery"


class DiagnosticValidity(_StringEnum):
    """Validity of diagnostics produced by the attempt."""

    NOT_EVALUATED = "not_evaluated"
    VALID = "valid"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


class ScientificUsability(_StringEnum):
    """Whether the attempt can support scientific interpretation."""

    NOT_EVALUATED = "not_evaluated"
    USABLE = "usable"
    LIMITED = "limited"
    UNUSABLE = "unusable"


class ComparisonEligibility(_StringEnum):
    """Whether the attempt is eligible for like-for-like comparison."""

    NOT_EVALUATED = "not_evaluated"
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"


class WarningSeverity(_StringEnum):
    """Severity used by structured warning records."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class WavelengthAttemptStatus:
    """Orthogonal status dimensions for one wavelength-model attempt."""

    disposition: AttemptDisposition
    execution_stage: ExecutionStage
    technical_outcome: TechnicalOutcome
    diagnostic_validity: DiagnosticValidity
    scientific_usability: ScientificUsability
    comparison_eligibility: ComparisonEligibility
    warning_severity: WarningSeverity | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Return a JSON-safe dictionary using stable public field names."""
        return {
            "attempt_disposition": self.disposition.value,
            "execution_stage": self.execution_stage.value,
            "technical_outcome": self.technical_outcome.value,
            "diagnostic_validity": self.diagnostic_validity.value,
            "scientific_usability": self.scientific_usability.value,
            "comparison_eligibility": self.comparison_eligibility.value,
            "warning_severity": (
                self.warning_severity.value
                if self.warning_severity is not None
                else None
            ),
        }


@dataclass(frozen=True)
class WavelengthFailureRecord:
    """Structured, JSON-safe record for a failed wavelength-model attempt."""

    failure_code: str
    stage: ExecutionStage
    substage: str | None
    exception_type: str | None
    message: str
    diagnostics: Mapping[str, Any]
    traceback_reference: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow JSON-safe representation.

        Nested diagnostic values are expected to have been sanitized by the
        caller because the advisory workflow already owns the package-wide
        scalar-cleaning rules.
        """
        return {
            "failure_code": self.failure_code,
            "failure_stage": self.stage.value,
            "failure_substage": self.substage,
            "exception_type": self.exception_type,
            "exception_message": self.message,
            "failure_diagnostics": dict(self.diagnostics),
            "traceback_reference": self.traceback_reference,
        }


def coerce_execution_stage(value: str | ExecutionStage | None) -> ExecutionStage:
    """Map legacy stage labels onto the canonical execution-stage enum."""
    if isinstance(value, ExecutionStage):
        return value
    normalized = str(value or "").strip().lower()
    mapping = {
        "": ExecutionStage.UNKNOWN,
        "not_started": ExecutionStage.NOT_STARTED,
        "input_validation": ExecutionStage.PRECONDITION,
        "data_quality": ExecutionStage.PRECONDITION,
        "precondition": ExecutionStage.PRECONDITION,
        "ingestion": ExecutionStage.INGESTION,
        "preprocessing": ExecutionStage.PREPROCESSING,
        "parameter_constraint": ExecutionStage.SETUP,
        "fit_execution": ExecutionStage.OPTIMIZATION,
        "numerical_stability": ExecutionStage.OPTIMIZATION,
        "optimization": ExecutionStage.OPTIMIZATION,
        "consensus": ExecutionStage.CONSENSUS,
        "diagnostics": ExecutionStage.DIAGNOSTICS,
        "export": ExecutionStage.EXPORT,
        "completed": ExecutionStage.COMPLETED,
    }
    return mapping.get(normalized, ExecutionStage.UNKNOWN)


def derive_wavelength_attempt_status(
    *,
    legacy_status: str | None,
    training_iter: int | None = None,
    recovered_from_failure: bool = False,
    diagnostics_available: bool | None = None,
    diagnostics_partial: bool = False,
    warning_count: int = 0,
    failure_stage: str | ExecutionStage | None = None,
) -> WavelengthAttemptStatus:
    """Derive canonical dimensions from the maintained legacy attempt fields.

    This helper intentionally leaves the legacy ``status``, ``fit_success`` and
    ``fit_failed`` fields untouched.  It is therefore safe to introduce before
    the later typed-result migration.
    """
    status = str(legacy_status or "").strip().lower()

    if status in {"skipped", "not_attempted"}:
        disposition = (
            AttemptDisposition.SKIPPED
            if status == "skipped"
            else AttemptDisposition.NOT_ATTEMPTED
        )
        technical = (
            TechnicalOutcome.SKIPPED
            if status == "skipped"
            else TechnicalOutcome.NOT_ATTEMPTED
        )
        stage = (
            ExecutionStage.PRECONDITION
            if status == "skipped"
            else ExecutionStage.NOT_STARTED
        )
        return WavelengthAttemptStatus(
            disposition=disposition,
            execution_stage=stage,
            technical_outcome=technical,
            diagnostic_validity=DiagnosticValidity.NOT_EVALUATED,
            scientific_usability=ScientificUsability.NOT_EVALUATED,
            comparison_eligibility=(
                ComparisonEligibility.INELIGIBLE
                if status == "skipped"
                else ComparisonEligibility.NOT_EVALUATED
            ),
            warning_severity=None,
        )

    disposition = AttemptDisposition.ATTEMPTED
    if status in {"failed", "failure"}:
        return WavelengthAttemptStatus(
            disposition=disposition,
            execution_stage=coerce_execution_stage(failure_stage),
            technical_outcome=TechnicalOutcome.FAILED,
            diagnostic_validity=(
                DiagnosticValidity.PARTIAL
                if diagnostics_partial
                else DiagnosticValidity.UNAVAILABLE
            ),
            scientific_usability=ScientificUsability.UNUSABLE,
            comparison_eligibility=ComparisonEligibility.INELIGIBLE,
            warning_severity=WarningSeverity.ERROR,
        )

    try:
        initialized_only = training_iter is not None and int(training_iter) == 0
    except (TypeError, ValueError):
        initialized_only = False
    if initialized_only:
        technical = TechnicalOutcome.INITIALIZED_ONLY
    elif recovered_from_failure:
        technical = TechnicalOutcome.COMPLETED_WITH_RECOVERY
    elif warning_count > 0:
        technical = TechnicalOutcome.COMPLETED_WITH_WARNINGS
    else:
        technical = TechnicalOutcome.COMPLETED

    if diagnostics_available is True:
        diagnostic_validity = DiagnosticValidity.VALID
    elif diagnostics_partial:
        diagnostic_validity = DiagnosticValidity.PARTIAL
    elif diagnostics_available is False:
        diagnostic_validity = DiagnosticValidity.UNAVAILABLE
    else:
        diagnostic_validity = DiagnosticValidity.NOT_EVALUATED

    if initialized_only or recovered_from_failure:
        usability = ScientificUsability.LIMITED
    elif diagnostic_validity is DiagnosticValidity.VALID:
        usability = ScientificUsability.USABLE
    elif diagnostic_validity in {
        DiagnosticValidity.PARTIAL,
        DiagnosticValidity.UNAVAILABLE,
        DiagnosticValidity.NOT_EVALUATED,
    }:
        usability = ScientificUsability.LIMITED
    else:
        usability = ScientificUsability.UNUSABLE

    comparison = (
        ComparisonEligibility.ELIGIBLE
        if (
            diagnostic_validity is DiagnosticValidity.VALID
            and not initialized_only
        )
        else ComparisonEligibility.INELIGIBLE
    )

    severity = WarningSeverity.WARNING if warning_count > 0 else None
    return WavelengthAttemptStatus(
        disposition=disposition,
        execution_stage=ExecutionStage.COMPLETED,
        technical_outcome=technical,
        diagnostic_validity=diagnostic_validity,
        scientific_usability=usability,
        comparison_eligibility=comparison,
        warning_severity=severity,
    )
