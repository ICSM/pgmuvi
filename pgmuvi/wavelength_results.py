"""Typed, JSON-safe primitives for wavelength-advisory results.

The maintained wavelength-advisory APIs currently return dictionaries.  Those
payloads are intentionally preserved for backward compatibility.  This module
provides immutable typed adapters that can be constructed from the existing
mappings without mutating them or discarding unknown legacy fields.

The adapters are a foundation for later result-schema work; they do not change
model ranking, fitting, or scientific interpretation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

from .wavelength_hypotheses import (
    WavelengthModelHypothesis,
    describe_wavelength_model_hypothesis,
)
from .wavelength_status import (
    AttemptDisposition,
    ComparisonEligibility,
    DiagnosticValidity,
    ExecutionStage,
    ScientificUsability,
    TechnicalOutcome,
    WarningSeverity,
    WavelengthAttemptStatus,
    WavelengthFailureRecord,
    coerce_execution_stage,
    derive_wavelength_attempt_status,
)

WAVELENGTH_RESULT_SCHEMA_VERSION = "pgmuvi-wavelength-results-v1"

__all__ = [
    "WAVELENGTH_RESULT_SCHEMA_VERSION",
    "WavelengthAdvisoryResult",
    "WavelengthEvidenceKind",
    "WavelengthEvidenceRecord",
    "WavelengthModelAttemptResult",
    "WavelengthProvenanceRecord",
    "WavelengthWarningRecord",
    "as_wavelength_advisory_result",
    "as_wavelength_model_attempt_result",
]


class _StringEnum(str, Enum):
    """Enum whose members serialize and display as stable string values."""

    def __str__(self) -> str:
        return self.value


class WavelengthEvidenceKind(_StringEnum):
    """Epistemic role of one result item.

    These categories keep direct observations and derived statistics separate
    from heuristic interpretation, formal comparison claims, workflow warnings,
    and explicitly documented limitations.
    """

    OBSERVED_FACT = "observed_fact"
    DERIVED_STATISTIC = "derived_statistic"
    HEURISTIC_INTERPRETATION = "heuristic_interpretation"
    FORMAL_COMPARISON_RESULT = "formal_comparison_result"
    WORKFLOW_WARNING = "workflow_warning"
    FUTURE_WORK_LIMITATION = "future_work_limitation"


def _json_safe(value: Any) -> Any:
    """Return a recursively JSON-safe copy without importing heavy packages."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (Sequence, Set)) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_json_safe(item) for item in value]

    # NumPy scalar/array compatibility without a module-level dependency.
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe(item_method())
        except (TypeError, ValueError, RuntimeError):
            pass
    tolist_method = getattr(value, "tolist", None)
    if callable(tolist_method):
        try:
            return _json_safe(tolist_method())
        except (TypeError, ValueError, RuntimeError):
            pass

    # Torch tensor compatibility without importing torch at module import time.
    detach_method = getattr(value, "detach", None)
    if callable(detach_method):
        try:
            detached = detach_method()
            cpu_method = getattr(detached, "cpu", None)
            if callable(cpu_method):
                detached = cpu_method()
            tolist_method = getattr(detached, "tolist", None)
            if callable(tolist_method):
                return _json_safe(tolist_method())
        except (TypeError, ValueError, RuntimeError):
            pass

    return str(value)


def _mapping_copy(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _json_safe(item) for key, item in value.items()}


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return (str(value),)


def _coerce_string_enum(enum_type, value: Any, default):
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class WavelengthEvidenceRecord:
    """One typed item of wavelength-advisory evidence or interpretation."""

    name: str
    kind: WavelengthEvidenceKind
    value: Any = None
    available: bool | None = None
    units: str | None = None
    summary: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("WavelengthEvidenceRecord.name must be non-empty.")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "value", _json_safe(self.value))
        object.__setattr__(self, "provenance", _mapping_copy(self.provenance))
        object.__setattr__(self, "limitations", _string_tuple(self.limitations))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> WavelengthEvidenceRecord:
        """Construct a record from its stable public mapping representation."""
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        kind = _coerce_string_enum(
            WavelengthEvidenceKind,
            payload.get("kind"),
            WavelengthEvidenceKind.DERIVED_STATISTIC,
        )
        return cls(
            name=str(payload.get("name") or payload.get("key") or ""),
            kind=kind,
            value=payload.get("value"),
            available=payload.get("available"),
            units=payload.get("units"),
            summary=payload.get("summary"),
            provenance=payload.get("provenance") or {},
            limitations=_string_tuple(payload.get("limitations")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-safe representation."""
        return {
            "name": self.name,
            "kind": self.kind.value,
            "value": _json_safe(self.value),
            "available": self.available,
            "units": self.units,
            "summary": self.summary,
            "provenance": _mapping_copy(self.provenance),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class WavelengthWarningRecord:
    """Structured warning emitted while producing a wavelength result."""

    message: str
    severity: WarningSeverity = WarningSeverity.WARNING
    code: str | None = None
    stage: ExecutionStage = ExecutionStage.UNKNOWN
    category: str | None = None
    source: str | None = None
    filename: str | None = None
    lineno: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.message).strip():
            raise ValueError("WavelengthWarningRecord.message must be non-empty.")
        object.__setattr__(self, "message", str(self.message))
        object.__setattr__(self, "details", _mapping_copy(self.details))

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        default_severity: str | WarningSeverity | None = None,
        default_stage: str | ExecutionStage | None = None,
    ) -> WavelengthWarningRecord:
        """Construct from current warning dictionaries or the canonical schema."""
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        severity = _coerce_string_enum(
            WarningSeverity,
            payload.get("severity") or default_severity,
            WarningSeverity.WARNING,
        )
        stage = coerce_execution_stage(payload.get("stage") or default_stage)
        lineno = payload.get("lineno")
        try:
            lineno = int(lineno) if lineno is not None else None
        except (TypeError, ValueError):
            lineno = None
        return cls(
            message=str(payload.get("message") or ""),
            severity=severity,
            code=payload.get("code"),
            stage=stage,
            category=payload.get("category"),
            source=payload.get("source"),
            filename=payload.get("filename"),
            lineno=lineno,
            details=payload.get("details") or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-safe representation."""
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "stage": self.stage.value,
            "category": self.category,
            "source": self.source,
            "filename": self.filename,
            "lineno": self.lineno,
            "details": _mapping_copy(self.details),
        }


@dataclass(frozen=True)
class WavelengthProvenanceRecord:
    """Configuration and origin metadata for a typed wavelength result."""

    workflow_kind: str
    schema_version: str = WAVELENGTH_RESULT_SCHEMA_VERSION
    package_version: str | None = None
    source_identifier: str | None = None
    configuration: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.workflow_kind).strip():
            raise ValueError("workflow_kind must be non-empty.")
        object.__setattr__(self, "workflow_kind", str(self.workflow_kind))
        object.__setattr__(self, "configuration", _mapping_copy(self.configuration))
        object.__setattr__(self, "notes", _string_tuple(self.notes))

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        workflow_kind: str | None = None,
    ) -> WavelengthProvenanceRecord:
        """Construct provenance from an existing workflow result mapping."""
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        explicit = payload.get("provenance")
        explicit = explicit if isinstance(explicit, Mapping) else {}
        kind = (
            workflow_kind
            or explicit.get("workflow_kind")
            or payload.get("kind")
            or "wavelength_advisory_result"
        )
        configuration = explicit.get("configuration")
        if not isinstance(configuration, Mapping):
            configuration = {
                key: payload.get(key)
                for key in (
                    "advisory_only",
                    "runs_fits",
                    "scores_fit_quality",
                    "formats_report",
                    "makes_plots",
                    "model_kernel_config_state_isolated",
                    "mutates_input_lightcurve",
                    "automatic_model_selection_applied",
                    "automatic_constraints_applied",
                    "automatic_initialization_applied",
                )
                if key in payload
            }
        return cls(
            workflow_kind=str(kind),
            schema_version=str(
                explicit.get("schema_version")
                or payload.get("result_schema_version")
                or WAVELENGTH_RESULT_SCHEMA_VERSION
            ),
            package_version=explicit.get("package_version"),
            source_identifier=(
                explicit.get("source_identifier")
                or payload.get("source_identifier")
                or payload.get("source_id")
            ),
            configuration=configuration,
            notes=_string_tuple(explicit.get("notes")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-safe representation."""
        return {
            "workflow_kind": self.workflow_kind,
            "schema_version": self.schema_version,
            "package_version": self.package_version,
            "source_identifier": self.source_identifier,
            "configuration": _mapping_copy(self.configuration),
            "notes": list(self.notes),
        }


def _attempt_status_from_mapping(payload: Mapping[str, Any]) -> WavelengthAttemptStatus:
    canonical_keys = {
        "attempt_disposition",
        "execution_stage",
        "technical_outcome",
        "diagnostic_validity",
        "scientific_usability",
        "comparison_eligibility",
    }
    if canonical_keys.issubset(payload):
        return WavelengthAttemptStatus(
            disposition=_coerce_string_enum(
                AttemptDisposition,
                payload.get("attempt_disposition"),
                AttemptDisposition.NOT_ATTEMPTED,
            ),
            execution_stage=coerce_execution_stage(payload.get("execution_stage")),
            technical_outcome=_coerce_string_enum(
                TechnicalOutcome,
                payload.get("technical_outcome"),
                TechnicalOutcome.NOT_ATTEMPTED,
            ),
            diagnostic_validity=_coerce_string_enum(
                DiagnosticValidity,
                payload.get("diagnostic_validity"),
                DiagnosticValidity.NOT_EVALUATED,
            ),
            scientific_usability=_coerce_string_enum(
                ScientificUsability,
                payload.get("scientific_usability"),
                ScientificUsability.NOT_EVALUATED,
            ),
            comparison_eligibility=_coerce_string_enum(
                ComparisonEligibility,
                payload.get("comparison_eligibility"),
                ComparisonEligibility.NOT_EVALUATED,
            ),
            warning_severity=(
                _coerce_string_enum(
                    WarningSeverity,
                    payload.get("warning_severity"),
                    WarningSeverity.WARNING,
                )
                if payload.get("warning_severity") is not None
                else None
            ),
        )

    fit_kwargs = payload.get("fit_kwargs")
    fit_kwargs = fit_kwargs if isinstance(fit_kwargs, Mapping) else {}
    recovery = bool(
        payload.get("training_recovered_from_failure")
        or payload.get("recovered_from_failure")
    )
    fit_quality = payload.get("fit_quality")
    fit_quality = fit_quality if isinstance(fit_quality, Mapping) else {}
    diagnostics_available = payload.get("fit_quality_available")
    if diagnostics_available is None and "available" in fit_quality:
        diagnostics_available = fit_quality.get("available")
    warning_records = payload.get("warning_records")
    warning_count = payload.get("warning_count")
    if warning_count is None and isinstance(warning_records, Sequence):
        warning_count = len(warning_records)
    try:
        warning_count = int(warning_count or 0)
    except (TypeError, ValueError):
        warning_count = 0
    structured = payload.get("structured_failure_diagnostics")
    diagnostics_partial = bool(
        isinstance(structured, Mapping)
        and any(
            key in structured
            for key in (
                "accepted_bands",
                "rejected_bands",
                "per_band_diagnostics",
                "period_summaries",
            )
        )
    )
    return derive_wavelength_attempt_status(
        legacy_status=payload.get("status"),
        training_iter=fit_kwargs.get("training_iter"),
        recovered_from_failure=recovery,
        diagnostics_available=diagnostics_available,
        diagnostics_partial=diagnostics_partial,
        warning_count=warning_count,
        failure_stage=payload.get("failure_stage"),
    )


def _failure_from_mapping(
    payload: Mapping[str, Any],
) -> WavelengthFailureRecord | None:
    structured = payload.get("structured_failure_record")
    structured = structured if isinstance(structured, Mapping) else {}
    failure_code = structured.get("failure_code") or payload.get("failure_code")
    if not failure_code:
        return None
    diagnostics = (
        structured.get("failure_diagnostics")
        or payload.get("structured_failure_diagnostics")
        or payload.get("fit_failure_diagnostics")
        or {}
    )
    return WavelengthFailureRecord(
        failure_code=str(failure_code),
        stage=coerce_execution_stage(
            structured.get("failure_stage") or payload.get("failure_stage")
        ),
        substage=(
            structured.get("failure_substage")
            or payload.get("failure_substage")
        ),
        exception_type=(
            structured.get("exception_type") or payload.get("exception_type")
        ),
        message=str(
            structured.get("exception_message")
            or payload.get("exception_message")
            or payload.get("failure_stage_reason")
            or failure_code
        ),
        diagnostics=_mapping_copy(diagnostics),
        traceback_reference=(
            structured.get("traceback_reference")
            or payload.get("traceback_reference")
        ),
    )


def _warning_records_from_attempt(
    payload: Mapping[str, Any],
) -> tuple[WavelengthWarningRecord, ...]:
    records = payload.get("warning_records")
    if not isinstance(records, Sequence) or isinstance(records, str):
        return ()
    output = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        message = record.get("message")
        if not str(message or "").strip():
            continue
        output.append(
            WavelengthWarningRecord.from_mapping(
                record,
                default_severity=payload.get("warning_severity"),
                default_stage=payload.get("execution_stage"),
            )
        )
    return tuple(output)


@dataclass(frozen=True)
class WavelengthModelAttemptResult:
    """Typed adapter for one existing model/kernel-config outcome mapping."""

    model_kernel_config_id: str | None
    rank: int | None
    model: str | None
    status: WavelengthAttemptStatus
    failure: WavelengthFailureRecord | None = None
    warnings: tuple[WavelengthWarningRecord, ...] = ()
    fit_kwargs: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    scores: Mapping[str, Any] = field(default_factory=dict)
    evidence: tuple[WavelengthEvidenceRecord, ...] = ()
    legacy_payload: Mapping[str, Any] = field(default_factory=dict, repr=False)
    hypothesis: WavelengthModelHypothesis | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fit_kwargs", _mapping_copy(self.fit_kwargs))
        if self.hypothesis is None:
            object.__setattr__(
                self,
                "hypothesis",
                describe_wavelength_model_hypothesis(
                    self.model or "", fit_kwargs=self.fit_kwargs
                ),
            )
        object.__setattr__(self, "diagnostics", _mapping_copy(self.diagnostics))
        object.__setattr__(self, "scores", _mapping_copy(self.scores))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(
            self, "legacy_payload", _mapping_copy(self.legacy_payload)
        )

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> WavelengthModelAttemptResult:
        """Adapt a maintained candidate-outcome dictionary without mutation."""
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        rank = payload.get("rank")
        try:
            rank = int(rank) if rank is not None else None
        except (TypeError, ValueError):
            rank = None

        diagnostics = {
            key: payload.get(key)
            for key in (
                "fit_quality",
                "fit_failure_diagnostics",
                "structured_failure_diagnostics",
                "consensus_diagnostics",
                "sm_ard_diagnostics",
                "sm_ard_scale_diagnostics",
            )
            if key in payload
        }
        consensus_fields = {
            key: payload.get(key)
            for key in (
                "consensus_success",
                "consensus_frequency",
                "consensus_period",
                "consensus_time_kernel_constraint_mode",
                "n_accepted_bands",
                "n_rejected_bands",
                "accepted_bands",
                "rejected_bands",
            )
            if key in payload
        }
        if consensus_fields:
            diagnostics["consensus"] = consensus_fields

        scores = {
            key: payload.get(key)
            for key in (
                "fit_quality_score",
                "score_kind",
                "score_components",
                "training_log_marginal_likelihood",
                "training_log_marginal_likelihood_total",
                "training_map_objective",
                "training_map_objective_total",
                "training_normalized_rmse",
                "training_reduced_chi2",
            )
            if key in payload
        }

        evidence_payload = payload.get("evidence")
        evidence = []
        if isinstance(evidence_payload, Sequence) and not isinstance(
            evidence_payload, str
        ):
            for record in evidence_payload:
                if isinstance(record, Mapping):
                    evidence.append(WavelengthEvidenceRecord.from_mapping(record))

        return cls(
            model_kernel_config_id=(
                str(payload.get("model_kernel_config_id"))
                if payload.get("model_kernel_config_id") is not None
                else None
            ),
            rank=rank,
            model=(
                str(payload.get("model"))
                if payload.get("model") is not None
                else None
            ),
            status=_attempt_status_from_mapping(payload),
            hypothesis=(
                WavelengthModelHypothesis.from_mapping(payload["model_hypothesis"])
                if isinstance(payload.get("model_hypothesis"), Mapping)
                else describe_wavelength_model_hypothesis(
                    str(payload.get("model") or ""),
                    fit_kwargs=payload.get("fit_kwargs") or {},
                )
            ),
            failure=_failure_from_mapping(payload),
            warnings=_warning_records_from_attempt(payload),
            fit_kwargs=payload.get("fit_kwargs") or {},
            diagnostics=diagnostics,
            scores=scores,
            evidence=tuple(evidence),
            legacy_payload=payload,
        )

    def to_dict(self, *, include_legacy_payload: bool = False) -> dict[str, Any]:
        """Return the canonical typed schema as a JSON-safe dictionary."""
        output = {
            "schema_version": WAVELENGTH_RESULT_SCHEMA_VERSION,
            "kind": "wavelength_model_attempt_result",
            "model_kernel_config_id": self.model_kernel_config_id,
            "rank": self.rank,
            "model": self.model,
            "status": self.status.to_dict(),
            "hypothesis": (
                self.hypothesis.to_dict() if self.hypothesis is not None else None
            ),
            "failure": self.failure.to_dict() if self.failure is not None else None,
            "warnings": [record.to_dict() for record in self.warnings],
            "fit_kwargs": _mapping_copy(self.fit_kwargs),
            "diagnostics": _mapping_copy(self.diagnostics),
            "scores": _mapping_copy(self.scores),
            "evidence": [record.to_dict() for record in self.evidence],
        }
        if include_legacy_payload:
            output["legacy_payload"] = _mapping_copy(self.legacy_payload)
        return output

    def to_legacy_dict(self) -> dict[str, Any]:
        """Return a defensive JSON-safe copy of the original flat payload."""
        return _mapping_copy(self.legacy_payload)


@dataclass(frozen=True)
class WavelengthAdvisoryResult:
    """Typed envelope around a maintained advisory-workflow dictionary.

    The envelope intentionally uses flexible named sections.  Later packages can
    promote those sections into more specific dataclasses without changing the
    current dictionary-returning public workflow.
    """

    kind: str
    advisory_only: bool
    attempts: tuple[WavelengthModelAttemptResult, ...]
    evidence: tuple[WavelengthEvidenceRecord, ...]
    warnings: tuple[WavelengthWarningRecord, ...]
    sections: Mapping[str, Any]
    provenance: WavelengthProvenanceRecord
    legacy_payload: Mapping[str, Any] = field(default_factory=dict, repr=False)
    schema_version: str = WAVELENGTH_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not str(self.kind).strip():
            raise ValueError("kind must be non-empty.")
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "attempts", tuple(self.attempts))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "sections", _mapping_copy(self.sections))
        object.__setattr__(
            self, "legacy_payload", _mapping_copy(self.legacy_payload)
        )

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> WavelengthAdvisoryResult:
        """Adapt a current single-source advisory workflow result mapping."""
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")

        run_report = payload.get("run_report")
        run_report = run_report if isinstance(run_report, Mapping) else {}
        raw_attempts = run_report.get("outcomes")
        if not isinstance(raw_attempts, Sequence) or isinstance(raw_attempts, str):
            raw_attempts = payload.get("outcomes")
        if not isinstance(raw_attempts, Sequence) or isinstance(raw_attempts, str):
            raw_attempts = []
        attempts = tuple(
            WavelengthModelAttemptResult.from_mapping(item)
            for item in raw_attempts
            if isinstance(item, Mapping)
        )

        raw_evidence = payload.get("evidence")
        evidence = []
        if isinstance(raw_evidence, Sequence) and not isinstance(raw_evidence, str):
            for item in raw_evidence:
                if isinstance(item, Mapping):
                    evidence.append(WavelengthEvidenceRecord.from_mapping(item))

        raw_warnings = payload.get("warnings")
        warnings = []
        if isinstance(raw_warnings, Sequence) and not isinstance(raw_warnings, str):
            for item in raw_warnings:
                if isinstance(item, Mapping) and str(item.get("message") or "").strip():
                    warnings.append(WavelengthWarningRecord.from_mapping(item))

        sections = {
            name: payload.get(name)
            for name in (
                "input_data_summary",
                "sampling_diagnostics",
                "period_diagnostics",
                "period_independent_wavelength_diagnostics",
                "fixed_period_wavelength_diagnostics",
                "model_kernel_config_report",
                "run_report",
                "quality_report",
                "fallback_report",
                "interpretation",
                "advisory_conclusions",
                "unresolved_ambiguities",
                "advisory_conclusion_summary",
            )
            if name in payload
        }

        return cls(
            kind=str(payload.get("kind") or "wavelength_advisory_result"),
            advisory_only=bool(payload.get("advisory_only", True)),
            attempts=attempts,
            evidence=tuple(evidence),
            warnings=tuple(warnings),
            sections=sections,
            provenance=WavelengthProvenanceRecord.from_mapping(payload),
            legacy_payload=payload,
            schema_version=str(
                payload.get("result_schema_version")
                or WAVELENGTH_RESULT_SCHEMA_VERSION
            ),
        )

    def to_dict(self, *, include_legacy_payload: bool = False) -> dict[str, Any]:
        """Return the canonical typed envelope as a JSON-safe dictionary."""
        output = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "advisory_only": self.advisory_only,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "evidence": [record.to_dict() for record in self.evidence],
            "warnings": [record.to_dict() for record in self.warnings],
            "sections": _mapping_copy(self.sections),
            "provenance": self.provenance.to_dict(),
        }
        if include_legacy_payload:
            output["legacy_payload"] = _mapping_copy(self.legacy_payload)
        return output

    def to_legacy_dict(self) -> dict[str, Any]:
        """Return a defensive JSON-safe copy of the original workflow payload."""
        return _mapping_copy(self.legacy_payload)


def as_wavelength_model_attempt_result(
    payload: Mapping[str, Any] | WavelengthModelAttemptResult,
) -> WavelengthModelAttemptResult:
    """Return ``payload`` as a typed model-attempt result."""
    if isinstance(payload, WavelengthModelAttemptResult):
        return payload
    return WavelengthModelAttemptResult.from_mapping(payload)


def as_wavelength_advisory_result(
    payload: Mapping[str, Any] | WavelengthAdvisoryResult,
) -> WavelengthAdvisoryResult:
    """Return ``payload`` as a typed advisory result envelope."""
    if isinstance(payload, WavelengthAdvisoryResult):
        return payload
    return WavelengthAdvisoryResult.from_mapping(payload)
