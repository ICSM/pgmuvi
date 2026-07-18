"""Typed, JSON-safe records for wavelength-model validation.

This module defines the data contracts used by the D1--D3 wavelength-model
validation programme: synthetic recovery, robustness and failure-boundary
experiments, and representative LPV validation.  The records are deliberately
independent of fitting code.  They preserve truth, configuration, provenance,
status, metrics, failures, and aggregate summaries without performing model
ranking or automatic model selection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

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
)

WAVELENGTH_VALIDATION_SCHEMA_VERSION = "pgmuvi-wavelength-validation-v1"

__all__ = [
    "WAVELENGTH_VALIDATION_SCHEMA_VERSION",
    "RecoveryMetricDirection",
    "WavelengthRecoveryMetric",
    "WavelengthValidationAggregate",
    "WavelengthValidationFailureExpectation",
    "WavelengthValidationPhase",
    "WavelengthValidationProvenance",
    "WavelengthValidationRun",
    "WavelengthValidationScenario",
    "WavelengthValidationSourceKind",
    "WavelengthValidationTruth",
    "as_wavelength_validation_aggregate",
    "as_wavelength_validation_run",
    "as_wavelength_validation_scenario",
]


class _StringEnum(str, Enum):
    """Enum whose members serialize and display as stable strings."""

    def __str__(self) -> str:
        return self.value


class WavelengthValidationPhase(_StringEnum):
    """Validation tranche owning a scenario, run, or aggregate."""

    D1_SYNTHETIC_RECOVERY = "d1_synthetic_recovery"
    D2_ROBUSTNESS_FAILURE_BOUNDARY = "d2_robustness_failure_boundary"
    D3_REPRESENTATIVE_LPV = "d3_representative_lpv"


class WavelengthValidationSourceKind(_StringEnum):
    """Origin of the light curve used by a validation scenario."""

    SYNTHETIC = "synthetic"
    OBSERVED = "observed"


class RecoveryMetricDirection(_StringEnum):
    """How a recovery metric should be interpreted against its threshold."""

    INFORMATIONAL = "informational"
    LOWER_IS_BETTER = "lower_is_better"
    HIGHER_IS_BETTER = "higher_is_better"
    INSIDE_INTERVAL = "inside_interval"
    EXACT_MATCH = "exact_match"


_PROHIBITED_SELECTION_KEYS = {
    "automatic_model_selection",
    "automatic_model_selection_applied",
    "best_model",
    "model_selection",
    "selected_model",
    "winner",
    "winning_model",
}


def _json_safe(value: Any) -> Any:
    """Return a recursively JSON-safe copy without heavy imports."""
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

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe(item_method())
        except (TypeError, ValueError, RuntimeError):
            pass

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

    tolist_method = getattr(value, "tolist", None)
    if callable(tolist_method):
        try:
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


def _float_tuple(value: Any) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("numeric tuple fields cannot be strings")
    try:
        return tuple(float(item) for item in value)
    except TypeError as exc:
        raise TypeError("numeric tuple fields must be iterable") from exc


def _coerce_string_enum(enum_type, value: Any, default):
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError):
        return default


def _require_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty.")
    return text


def _extra_fields(
    payload: Mapping[str, Any],
    known_fields: set[str],
) -> dict[str, Any]:
    explicit = payload.get("extra_fields")
    output = _mapping_copy(explicit if isinstance(explicit, Mapping) else {})
    for key, value in payload.items():
        key = str(key)
        if key not in known_fields and key != "extra_fields":
            output[key] = _json_safe(value)
    return output


def _validate_no_selection_claims(value: Mapping[str, Any]) -> None:
    prohibited = sorted(_PROHIBITED_SELECTION_KEYS.intersection(value))
    if prohibited:
        joined = ", ".join(prohibited)
        raise ValueError(
            "Validation records cannot contain model-selection claims: "
            f"{joined}."
        )


def _status_from_mapping(value: Any) -> WavelengthAttemptStatus | None:
    if value is None:
        return None
    if isinstance(value, WavelengthAttemptStatus):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("status must be a WavelengthAttemptStatus or mapping")
    return WavelengthAttemptStatus(
        disposition=_coerce_string_enum(
            AttemptDisposition,
            value.get("attempt_disposition") or value.get("disposition"),
            AttemptDisposition.NOT_ATTEMPTED,
        ),
        execution_stage=coerce_execution_stage(
            value.get("execution_stage") or value.get("stage")
        ),
        technical_outcome=_coerce_string_enum(
            TechnicalOutcome,
            value.get("technical_outcome"),
            TechnicalOutcome.NOT_ATTEMPTED,
        ),
        diagnostic_validity=_coerce_string_enum(
            DiagnosticValidity,
            value.get("diagnostic_validity"),
            DiagnosticValidity.NOT_EVALUATED,
        ),
        scientific_usability=_coerce_string_enum(
            ScientificUsability,
            value.get("scientific_usability"),
            ScientificUsability.NOT_EVALUATED,
        ),
        comparison_eligibility=_coerce_string_enum(
            ComparisonEligibility,
            value.get("comparison_eligibility"),
            ComparisonEligibility.NOT_EVALUATED,
        ),
        warning_severity=(
            _coerce_string_enum(
                WarningSeverity,
                value.get("warning_severity"),
                WarningSeverity.WARNING,
            )
            if value.get("warning_severity") is not None
            else None
        ),
    )


def _failure_from_mapping(value: Any) -> WavelengthFailureRecord | None:
    if value is None:
        return None
    if isinstance(value, WavelengthFailureRecord):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("failure must be a WavelengthFailureRecord or mapping")
    code = value.get("failure_code")
    if not str(code or "").strip():
        raise ValueError("failure.failure_code must be non-empty.")
    return WavelengthFailureRecord(
        failure_code=str(code),
        stage=coerce_execution_stage(
            value.get("failure_stage") or value.get("stage")
        ),
        substage=value.get("failure_substage") or value.get("substage"),
        exception_type=value.get("exception_type"),
        message=str(
            value.get("exception_message")
            or value.get("message")
            or code
        ),
        diagnostics=_mapping_copy(
            value.get("failure_diagnostics") or value.get("diagnostics") or {}
        ),
        traceback_reference=value.get("traceback_reference"),
    )


@dataclass(frozen=True)
class WavelengthValidationProvenance:
    """Environment and input provenance for a validation record."""

    schema_version: str = WAVELENGTH_VALIDATION_SCHEMA_VERSION
    package_version: str | None = None
    package_commit: str | None = None
    created_at_utc: str | None = None
    python_version: str | None = None
    platform: str | None = None
    dependencies: Mapping[str, Any] = field(default_factory=dict)
    torch_default_dtype: str | None = None
    device: str | None = None
    source_identifier: str | None = None
    source_checksum: str | None = None
    configuration: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    extra_fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", str(self.schema_version))
        object.__setattr__(self, "dependencies", _mapping_copy(self.dependencies))
        object.__setattr__(self, "configuration", _mapping_copy(self.configuration))
        object.__setattr__(self, "notes", _string_tuple(self.notes))
        extras = _mapping_copy(self.extra_fields)
        _validate_no_selection_claims(extras)
        object.__setattr__(self, "extra_fields", extras)

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> WavelengthValidationProvenance:
        """Construct provenance while retaining unknown fields."""
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        known = {
            "schema_version",
            "package_version",
            "package_commit",
            "created_at_utc",
            "python_version",
            "platform",
            "dependencies",
            "torch_default_dtype",
            "device",
            "source_identifier",
            "source_checksum",
            "configuration",
            "notes",
            "extra_fields",
        }
        return cls(
            schema_version=str(
                payload.get("schema_version")
                or WAVELENGTH_VALIDATION_SCHEMA_VERSION
            ),
            package_version=payload.get("package_version"),
            package_commit=payload.get("package_commit"),
            created_at_utc=payload.get("created_at_utc"),
            python_version=payload.get("python_version"),
            platform=payload.get("platform"),
            dependencies=payload.get("dependencies") or {},
            torch_default_dtype=payload.get("torch_default_dtype"),
            device=payload.get("device"),
            source_identifier=payload.get("source_identifier"),
            source_checksum=payload.get("source_checksum"),
            configuration=payload.get("configuration") or {},
            notes=_string_tuple(payload.get("notes")),
            extra_fields=_extra_fields(payload, known),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-safe representation."""
        return {
            "schema_version": self.schema_version,
            "package_version": self.package_version,
            "package_commit": self.package_commit,
            "created_at_utc": self.created_at_utc,
            "python_version": self.python_version,
            "platform": self.platform,
            "dependencies": _mapping_copy(self.dependencies),
            "torch_default_dtype": self.torch_default_dtype,
            "device": self.device,
            "source_identifier": self.source_identifier,
            "source_checksum": self.source_checksum,
            "configuration": _mapping_copy(self.configuration),
            "notes": list(self.notes),
            "extra_fields": _mapping_copy(self.extra_fields),
        }


@dataclass(frozen=True)
class WavelengthValidationTruth:
    """Known generating truth for a synthetic validation scenario."""

    generating_model: str
    truth_kind: str
    schema_version: str = WAVELENGTH_VALIDATION_SCHEMA_VERSION
    physical_wavelengths: tuple[float, ...] = ()
    band_labels: tuple[str, ...] = ()
    temporal_parameters: Mapping[str, Any] = field(default_factory=dict)
    wavelength_mean_parameters: Mapping[str, Any] = field(default_factory=dict)
    wavelength_covariance_parameters: Mapping[str, Any] = field(default_factory=dict)
    latent_parameters: Mapping[str, Any] = field(default_factory=dict)
    noiseless_summary: Mapping[str, Any] = field(default_factory=dict)
    coordinate_transforms: Mapping[str, Any] = field(default_factory=dict)
    parameter_ownership: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    extra_fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "generating_model",
            _require_text(self.generating_model, "generating_model"),
        )
        object.__setattr__(
            self, "truth_kind", _require_text(self.truth_kind, "truth_kind")
        )
        object.__setattr__(self, "schema_version", str(self.schema_version))
        wavelengths = _float_tuple(self.physical_wavelengths)
        if any(not math.isfinite(value) or value <= 0.0 for value in wavelengths):
            raise ValueError(
                "physical_wavelengths must contain finite positive values."
            )
        bands = _string_tuple(self.band_labels)
        if bands and len(bands) != len(wavelengths):
            raise ValueError(
                "band_labels and physical_wavelengths must have equal lengths."
            )
        object.__setattr__(self, "physical_wavelengths", wavelengths)
        object.__setattr__(self, "band_labels", bands)
        for name in (
            "temporal_parameters",
            "wavelength_mean_parameters",
            "wavelength_covariance_parameters",
            "latent_parameters",
            "noiseless_summary",
            "coordinate_transforms",
            "parameter_ownership",
            "metadata",
        ):
            object.__setattr__(self, name, _mapping_copy(getattr(self, name)))
        extras = _mapping_copy(self.extra_fields)
        _validate_no_selection_claims(extras)
        object.__setattr__(self, "extra_fields", extras)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> WavelengthValidationTruth:
        """Construct a truth record while retaining unknown fields."""
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        known = {
            "schema_version",
            "generating_model",
            "truth_kind",
            "physical_wavelengths",
            "band_labels",
            "temporal_parameters",
            "wavelength_mean_parameters",
            "wavelength_covariance_parameters",
            "latent_parameters",
            "noiseless_summary",
            "coordinate_transforms",
            "parameter_ownership",
            "metadata",
            "extra_fields",
        }
        return cls(
            generating_model=payload.get("generating_model") or "",
            truth_kind=payload.get("truth_kind") or "",
            schema_version=str(
                payload.get("schema_version")
                or WAVELENGTH_VALIDATION_SCHEMA_VERSION
            ),
            physical_wavelengths=_float_tuple(payload.get("physical_wavelengths")),
            band_labels=_string_tuple(payload.get("band_labels")),
            temporal_parameters=payload.get("temporal_parameters") or {},
            wavelength_mean_parameters=(
                payload.get("wavelength_mean_parameters") or {}
            ),
            wavelength_covariance_parameters=(
                payload.get("wavelength_covariance_parameters") or {}
            ),
            latent_parameters=payload.get("latent_parameters") or {},
            noiseless_summary=payload.get("noiseless_summary") or {},
            coordinate_transforms=payload.get("coordinate_transforms") or {},
            parameter_ownership=payload.get("parameter_ownership") or {},
            metadata=payload.get("metadata") or {},
            extra_fields=_extra_fields(payload, known),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-safe representation."""
        return {
            "schema_version": self.schema_version,
            "generating_model": self.generating_model,
            "truth_kind": self.truth_kind,
            "physical_wavelengths": list(self.physical_wavelengths),
            "band_labels": list(self.band_labels),
            "temporal_parameters": _mapping_copy(self.temporal_parameters),
            "wavelength_mean_parameters": _mapping_copy(
                self.wavelength_mean_parameters
            ),
            "wavelength_covariance_parameters": _mapping_copy(
                self.wavelength_covariance_parameters
            ),
            "latent_parameters": _mapping_copy(self.latent_parameters),
            "noiseless_summary": _mapping_copy(self.noiseless_summary),
            "coordinate_transforms": _mapping_copy(self.coordinate_transforms),
            "parameter_ownership": _mapping_copy(self.parameter_ownership),
            "metadata": _mapping_copy(self.metadata),
            "extra_fields": _mapping_copy(self.extra_fields),
        }


@dataclass(frozen=True)
class WavelengthValidationFailureExpectation:
    """Expected classified failure for one failure-boundary scenario."""

    failure_code: str
    stage: ExecutionStage
    technical_outcome: TechnicalOutcome = TechnicalOutcome.FAILED
    diagnostic_validity: DiagnosticValidity = DiagnosticValidity.UNAVAILABLE
    scientific_usability: ScientificUsability = ScientificUsability.UNUSABLE
    comparison_eligibility: ComparisonEligibility = ComparisonEligibility.INELIGIBLE
    schema_version: str = WAVELENGTH_VALIDATION_SCHEMA_VERSION
    acceptable_exception_types: tuple[str, ...] = ()
    rationale: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    extra_fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "failure_code", _require_text(self.failure_code, "failure_code")
        )
        object.__setattr__(self, "stage", coerce_execution_stage(self.stage))
        object.__setattr__(
            self,
            "technical_outcome",
            _coerce_string_enum(
                TechnicalOutcome,
                self.technical_outcome,
                TechnicalOutcome.FAILED,
            ),
        )
        object.__setattr__(
            self,
            "diagnostic_validity",
            _coerce_string_enum(
                DiagnosticValidity,
                self.diagnostic_validity,
                DiagnosticValidity.UNAVAILABLE,
            ),
        )
        object.__setattr__(
            self,
            "scientific_usability",
            _coerce_string_enum(
                ScientificUsability,
                self.scientific_usability,
                ScientificUsability.UNUSABLE,
            ),
        )
        object.__setattr__(
            self,
            "comparison_eligibility",
            _coerce_string_enum(
                ComparisonEligibility,
                self.comparison_eligibility,
                ComparisonEligibility.INELIGIBLE,
            ),
        )
        if self.comparison_eligibility is ComparisonEligibility.ELIGIBLE:
            raise ValueError("Expected failures cannot be comparison-eligible.")
        if self.technical_outcome not in {
            TechnicalOutcome.FAILED,
            TechnicalOutcome.SKIPPED,
        }:
            raise ValueError(
                "Expected failures must have a failed or skipped technical outcome."
            )
        object.__setattr__(self, "schema_version", str(self.schema_version))
        object.__setattr__(
            self,
            "acceptable_exception_types",
            _string_tuple(self.acceptable_exception_types),
        )
        object.__setattr__(self, "diagnostics", _mapping_copy(self.diagnostics))
        extras = _mapping_copy(self.extra_fields)
        _validate_no_selection_claims(extras)
        object.__setattr__(self, "extra_fields", extras)

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> WavelengthValidationFailureExpectation:
        """Construct an expected-failure record from a mapping."""
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        known = {
            "schema_version",
            "failure_code",
            "stage",
            "failure_stage",
            "technical_outcome",
            "diagnostic_validity",
            "scientific_usability",
            "comparison_eligibility",
            "acceptable_exception_types",
            "rationale",
            "diagnostics",
            "extra_fields",
        }
        return cls(
            failure_code=payload.get("failure_code") or "",
            stage=coerce_execution_stage(
                payload.get("stage") or payload.get("failure_stage")
            ),
            technical_outcome=_coerce_string_enum(
                TechnicalOutcome,
                payload.get("technical_outcome"),
                TechnicalOutcome.FAILED,
            ),
            diagnostic_validity=_coerce_string_enum(
                DiagnosticValidity,
                payload.get("diagnostic_validity"),
                DiagnosticValidity.UNAVAILABLE,
            ),
            scientific_usability=_coerce_string_enum(
                ScientificUsability,
                payload.get("scientific_usability"),
                ScientificUsability.UNUSABLE,
            ),
            comparison_eligibility=_coerce_string_enum(
                ComparisonEligibility,
                payload.get("comparison_eligibility"),
                ComparisonEligibility.INELIGIBLE,
            ),
            schema_version=str(
                payload.get("schema_version")
                or WAVELENGTH_VALIDATION_SCHEMA_VERSION
            ),
            acceptable_exception_types=_string_tuple(
                payload.get("acceptable_exception_types")
            ),
            rationale=payload.get("rationale"),
            diagnostics=payload.get("diagnostics") or {},
            extra_fields=_extra_fields(payload, known),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-safe representation."""
        return {
            "schema_version": self.schema_version,
            "failure_code": self.failure_code,
            "failure_stage": self.stage.value,
            "technical_outcome": self.technical_outcome.value,
            "diagnostic_validity": self.diagnostic_validity.value,
            "scientific_usability": self.scientific_usability.value,
            "comparison_eligibility": self.comparison_eligibility.value,
            "acceptable_exception_types": list(self.acceptable_exception_types),
            "rationale": self.rationale,
            "diagnostics": _mapping_copy(self.diagnostics),
            "extra_fields": _mapping_copy(self.extra_fields),
        }


@dataclass(frozen=True)
class WavelengthRecoveryMetric:
    """One recovery, stability, or workflow-contract metric."""

    name: str
    value: Any = None
    truth_value: Any = None
    available: bool = True
    passed: bool | None = None
    direction: RecoveryMetricDirection = RecoveryMetricDirection.INFORMATIONAL
    schema_version: str = WAVELENGTH_VALIDATION_SCHEMA_VERSION
    units: str | None = None
    threshold: Mapping[str, Any] = field(default_factory=dict)
    scope: str | None = None
    model: str | None = None
    parameter: str | None = None
    ard_dimension: str | None = None
    component_index: int | None = None
    summary: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    extra_fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_text(self.name, "name"))
        object.__setattr__(self, "value", _json_safe(self.value))
        object.__setattr__(self, "truth_value", _json_safe(self.truth_value))
        object.__setattr__(
            self,
            "direction",
            _coerce_string_enum(
                RecoveryMetricDirection,
                self.direction,
                RecoveryMetricDirection.INFORMATIONAL,
            ),
        )
        object.__setattr__(self, "schema_version", str(self.schema_version))
        object.__setattr__(self, "threshold", _mapping_copy(self.threshold))
        object.__setattr__(self, "provenance", _mapping_copy(self.provenance))
        object.__setattr__(self, "limitations", _string_tuple(self.limitations))
        object.__setattr__(self, "metadata", _mapping_copy(self.metadata))
        if self.component_index is not None:
            try:
                component_index = int(self.component_index)
            except (TypeError, ValueError) as exc:
                raise TypeError("component_index must be an integer or None") from exc
            if component_index < 0:
                raise ValueError("component_index cannot be negative.")
            object.__setattr__(self, "component_index", component_index)
        extras = _mapping_copy(self.extra_fields)
        _validate_no_selection_claims(extras)
        object.__setattr__(self, "extra_fields", extras)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> WavelengthRecoveryMetric:
        """Construct a metric while retaining unknown fields."""
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        known = {
            "schema_version",
            "name",
            "value",
            "truth_value",
            "available",
            "passed",
            "direction",
            "units",
            "threshold",
            "scope",
            "model",
            "parameter",
            "ard_dimension",
            "component_index",
            "summary",
            "provenance",
            "limitations",
            "metadata",
            "extra_fields",
        }
        return cls(
            name=payload.get("name") or "",
            value=payload.get("value"),
            truth_value=payload.get("truth_value"),
            available=bool(payload.get("available", True)),
            passed=payload.get("passed"),
            direction=_coerce_string_enum(
                RecoveryMetricDirection,
                payload.get("direction"),
                RecoveryMetricDirection.INFORMATIONAL,
            ),
            schema_version=str(
                payload.get("schema_version")
                or WAVELENGTH_VALIDATION_SCHEMA_VERSION
            ),
            units=payload.get("units"),
            threshold=payload.get("threshold") or {},
            scope=payload.get("scope"),
            model=payload.get("model"),
            parameter=payload.get("parameter"),
            ard_dimension=payload.get("ard_dimension"),
            component_index=payload.get("component_index"),
            summary=payload.get("summary"),
            provenance=payload.get("provenance") or {},
            limitations=_string_tuple(payload.get("limitations")),
            metadata=payload.get("metadata") or {},
            extra_fields=_extra_fields(payload, known),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-safe representation."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "value": _json_safe(self.value),
            "truth_value": _json_safe(self.truth_value),
            "available": self.available,
            "passed": self.passed,
            "direction": self.direction.value,
            "units": self.units,
            "threshold": _mapping_copy(self.threshold),
            "scope": self.scope,
            "model": self.model,
            "parameter": self.parameter,
            "ard_dimension": self.ard_dimension,
            "component_index": self.component_index,
            "summary": self.summary,
            "provenance": _mapping_copy(self.provenance),
            "limitations": list(self.limitations),
            "metadata": _mapping_copy(self.metadata),
            "extra_fields": _mapping_copy(self.extra_fields),
        }


@dataclass(frozen=True)
class WavelengthValidationScenario:
    """Configuration and expected behavior for one validation scenario."""

    scenario_id: str
    phase: WavelengthValidationPhase
    source_kind: WavelengthValidationSourceKind
    description: str
    schema_version: str = WAVELENGTH_VALIDATION_SCHEMA_VERSION
    truth: WavelengthValidationTruth | None = None
    sampling_configuration: Mapping[str, Any] = field(default_factory=dict)
    noise_configuration: Mapping[str, Any] = field(default_factory=dict)
    preprocessing_configuration: Mapping[str, Any] = field(default_factory=dict)
    fit_configuration: Mapping[str, Any] = field(default_factory=dict)
    expected_failure: WavelengthValidationFailureExpectation | None = None
    tags: tuple[str, ...] = ()
    provenance: WavelengthValidationProvenance | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    advisory_only: bool = True
    extra_fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "scenario_id", _require_text(self.scenario_id, "scenario_id")
        )
        object.__setattr__(
            self,
            "phase",
            _coerce_string_enum(
                WavelengthValidationPhase,
                self.phase,
                WavelengthValidationPhase.D1_SYNTHETIC_RECOVERY,
            ),
        )
        object.__setattr__(
            self,
            "source_kind",
            _coerce_string_enum(
                WavelengthValidationSourceKind,
                self.source_kind,
                WavelengthValidationSourceKind.SYNTHETIC,
            ),
        )
        object.__setattr__(
            self, "description", _require_text(self.description, "description")
        )
        object.__setattr__(self, "schema_version", str(self.schema_version))
        if self.truth is not None and not isinstance(
            self.truth, WavelengthValidationTruth
        ):
            object.__setattr__(
                self, "truth", WavelengthValidationTruth.from_mapping(self.truth)
            )
        if (
            self.source_kind is WavelengthValidationSourceKind.SYNTHETIC
            and self.truth is None
        ):
            raise ValueError("Synthetic validation scenarios require a truth record.")
        if self.expected_failure is not None and not isinstance(
            self.expected_failure, WavelengthValidationFailureExpectation
        ):
            object.__setattr__(
                self,
                "expected_failure",
                WavelengthValidationFailureExpectation.from_mapping(
                    self.expected_failure
                ),
            )
        if self.provenance is not None and not isinstance(
            self.provenance, WavelengthValidationProvenance
        ):
            object.__setattr__(
                self,
                "provenance",
                WavelengthValidationProvenance.from_mapping(self.provenance),
            )
        for name in (
            "sampling_configuration",
            "noise_configuration",
            "preprocessing_configuration",
            "fit_configuration",
            "metadata",
        ):
            object.__setattr__(self, name, _mapping_copy(getattr(self, name)))
        object.__setattr__(self, "tags", _string_tuple(self.tags))
        if self.advisory_only is not True:
            raise ValueError("Wavelength validation scenarios must remain advisory-only.")
        extras = _mapping_copy(self.extra_fields)
        _validate_no_selection_claims(extras)
        object.__setattr__(self, "extra_fields", extras)

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> WavelengthValidationScenario:
        """Construct a scenario while retaining unknown fields."""
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        known = {
            "schema_version",
            "scenario_id",
            "phase",
            "source_kind",
            "description",
            "truth",
            "sampling_configuration",
            "noise_configuration",
            "preprocessing_configuration",
            "fit_configuration",
            "expected_failure",
            "tags",
            "provenance",
            "metadata",
            "advisory_only",
            "extra_fields",
        }
        truth = payload.get("truth")
        expected_failure = payload.get("expected_failure")
        provenance = payload.get("provenance")
        return cls(
            scenario_id=payload.get("scenario_id") or "",
            phase=_coerce_string_enum(
                WavelengthValidationPhase,
                payload.get("phase"),
                WavelengthValidationPhase.D1_SYNTHETIC_RECOVERY,
            ),
            source_kind=_coerce_string_enum(
                WavelengthValidationSourceKind,
                payload.get("source_kind"),
                WavelengthValidationSourceKind.SYNTHETIC,
            ),
            description=payload.get("description") or "",
            schema_version=str(
                payload.get("schema_version")
                or WAVELENGTH_VALIDATION_SCHEMA_VERSION
            ),
            truth=(
                WavelengthValidationTruth.from_mapping(truth)
                if isinstance(truth, Mapping)
                else truth
            ),
            sampling_configuration=payload.get("sampling_configuration") or {},
            noise_configuration=payload.get("noise_configuration") or {},
            preprocessing_configuration=(
                payload.get("preprocessing_configuration") or {}
            ),
            fit_configuration=payload.get("fit_configuration") or {},
            expected_failure=(
                WavelengthValidationFailureExpectation.from_mapping(
                    expected_failure
                )
                if isinstance(expected_failure, Mapping)
                else expected_failure
            ),
            tags=_string_tuple(payload.get("tags")),
            provenance=(
                WavelengthValidationProvenance.from_mapping(provenance)
                if isinstance(provenance, Mapping)
                else provenance
            ),
            metadata=payload.get("metadata") or {},
            advisory_only=payload.get("advisory_only", True),
            extra_fields=_extra_fields(payload, known),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-safe representation."""
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "phase": self.phase.value,
            "source_kind": self.source_kind.value,
            "description": self.description,
            "truth": self.truth.to_dict() if self.truth is not None else None,
            "sampling_configuration": _mapping_copy(
                self.sampling_configuration
            ),
            "noise_configuration": _mapping_copy(self.noise_configuration),
            "preprocessing_configuration": _mapping_copy(
                self.preprocessing_configuration
            ),
            "fit_configuration": _mapping_copy(self.fit_configuration),
            "expected_failure": (
                self.expected_failure.to_dict()
                if self.expected_failure is not None
                else None
            ),
            "tags": list(self.tags),
            "provenance": (
                self.provenance.to_dict() if self.provenance is not None else None
            ),
            "metadata": _mapping_copy(self.metadata),
            "advisory_only": True,
            "extra_fields": _mapping_copy(self.extra_fields),
        }


@dataclass(frozen=True)
class WavelengthValidationRun:
    """Outcome of one model fit or classified validation attempt."""

    run_id: str
    scenario_id: str
    model: str
    schema_version: str = WAVELENGTH_VALIDATION_SCHEMA_VERSION
    seed: int | None = None
    status: WavelengthAttemptStatus | None = None
    failure: WavelengthFailureRecord | None = None
    fit_configuration: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    parameter_workflow: Mapping[str, Any] = field(default_factory=dict)
    metrics: tuple[WavelengthRecoveryMetric, ...] = ()
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    provenance: WavelengthValidationProvenance | None = None
    notes: tuple[str, ...] = ()
    advisory_only: bool = True
    extra_fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_text(self.run_id, "run_id"))
        object.__setattr__(
            self, "scenario_id", _require_text(self.scenario_id, "scenario_id")
        )
        object.__setattr__(self, "model", _require_text(self.model, "model"))
        object.__setattr__(self, "schema_version", str(self.schema_version))
        if self.seed is not None:
            try:
                object.__setattr__(self, "seed", int(self.seed))
            except (TypeError, ValueError) as exc:
                raise TypeError("seed must be an integer or None") from exc
        object.__setattr__(self, "status", _status_from_mapping(self.status))
        object.__setattr__(self, "failure", _failure_from_mapping(self.failure))
        if (
            self.failure is not None
            and self.status is not None
            and self.status.technical_outcome is not TechnicalOutcome.FAILED
        ):
            raise ValueError("A failure record requires a failed technical outcome.")
        for name in (
            "fit_configuration",
            "diagnostics",
            "parameter_workflow",
            "artifacts",
        ):
            object.__setattr__(self, name, _mapping_copy(getattr(self, name)))
        metrics = []
        for metric in self.metrics:
            if isinstance(metric, WavelengthRecoveryMetric):
                metrics.append(metric)
            elif isinstance(metric, Mapping):
                metrics.append(WavelengthRecoveryMetric.from_mapping(metric))
            else:
                raise TypeError("metrics must contain recovery metric records")
        object.__setattr__(self, "metrics", tuple(metrics))
        if self.provenance is not None and not isinstance(
            self.provenance, WavelengthValidationProvenance
        ):
            object.__setattr__(
                self,
                "provenance",
                WavelengthValidationProvenance.from_mapping(self.provenance),
            )
        object.__setattr__(self, "notes", _string_tuple(self.notes))
        if self.advisory_only is not True:
            raise ValueError("Wavelength validation runs must remain advisory-only.")
        extras = _mapping_copy(self.extra_fields)
        _validate_no_selection_claims(extras)
        object.__setattr__(self, "extra_fields", extras)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> WavelengthValidationRun:
        """Construct a run while retaining unknown fields."""
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        known = {
            "schema_version",
            "run_id",
            "scenario_id",
            "model",
            "seed",
            "status",
            "failure",
            "fit_configuration",
            "diagnostics",
            "parameter_workflow",
            "metrics",
            "artifacts",
            "provenance",
            "notes",
            "advisory_only",
            "extra_fields",
        }
        metrics = payload.get("metrics") or ()
        provenance = payload.get("provenance")
        return cls(
            run_id=payload.get("run_id") or "",
            scenario_id=payload.get("scenario_id") or "",
            model=payload.get("model") or "",
            schema_version=str(
                payload.get("schema_version")
                or WAVELENGTH_VALIDATION_SCHEMA_VERSION
            ),
            seed=payload.get("seed"),
            status=_status_from_mapping(payload.get("status")),
            failure=_failure_from_mapping(payload.get("failure")),
            fit_configuration=payload.get("fit_configuration") or {},
            diagnostics=payload.get("diagnostics") or {},
            parameter_workflow=payload.get("parameter_workflow") or {},
            metrics=tuple(
                WavelengthRecoveryMetric.from_mapping(item)
                if isinstance(item, Mapping)
                else item
                for item in metrics
            ),
            artifacts=payload.get("artifacts") or {},
            provenance=(
                WavelengthValidationProvenance.from_mapping(provenance)
                if isinstance(provenance, Mapping)
                else provenance
            ),
            notes=_string_tuple(payload.get("notes")),
            advisory_only=payload.get("advisory_only", True),
            extra_fields=_extra_fields(payload, known),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-safe representation."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "model": self.model,
            "seed": self.seed,
            "status": self.status.to_dict() if self.status is not None else None,
            "failure": self.failure.to_dict() if self.failure is not None else None,
            "fit_configuration": _mapping_copy(self.fit_configuration),
            "diagnostics": _mapping_copy(self.diagnostics),
            "parameter_workflow": _mapping_copy(self.parameter_workflow),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "artifacts": _mapping_copy(self.artifacts),
            "provenance": (
                self.provenance.to_dict() if self.provenance is not None else None
            ),
            "notes": list(self.notes),
            "advisory_only": True,
            "extra_fields": _mapping_copy(self.extra_fields),
        }


@dataclass(frozen=True)
class WavelengthValidationAggregate:
    """Failure-aware aggregate of validation runs without model selection."""

    aggregate_id: str
    phase: WavelengthValidationPhase
    schema_version: str = WAVELENGTH_VALIDATION_SCHEMA_VERSION
    scenario_ids: tuple[str, ...] = ()
    run_ids: tuple[str, ...] = ()
    n_runs: int = 0
    status_counts: Mapping[str, Any] = field(default_factory=dict)
    metric_summaries: Mapping[str, Any] = field(default_factory=dict)
    model_summaries: Mapping[str, Any] = field(default_factory=dict)
    failure_summaries: Mapping[str, Any] = field(default_factory=dict)
    provenance: WavelengthValidationProvenance | None = None
    notes: tuple[str, ...] = ()
    advisory_only: bool = True
    automatic_model_selection_applied: bool = False
    extra_fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "aggregate_id", _require_text(self.aggregate_id, "aggregate_id")
        )
        object.__setattr__(
            self,
            "phase",
            _coerce_string_enum(
                WavelengthValidationPhase,
                self.phase,
                WavelengthValidationPhase.D1_SYNTHETIC_RECOVERY,
            ),
        )
        object.__setattr__(self, "schema_version", str(self.schema_version))
        object.__setattr__(self, "scenario_ids", _string_tuple(self.scenario_ids))
        object.__setattr__(self, "run_ids", _string_tuple(self.run_ids))
        try:
            n_runs = int(self.n_runs)
        except (TypeError, ValueError) as exc:
            raise TypeError("n_runs must be an integer") from exc
        if n_runs < 0:
            raise ValueError("n_runs cannot be negative.")
        if self.run_ids and n_runs not in {0, len(self.run_ids)}:
            raise ValueError("n_runs must match the number of run_ids when provided.")
        if n_runs == 0 and self.run_ids:
            n_runs = len(self.run_ids)
        object.__setattr__(self, "n_runs", n_runs)
        for name in (
            "status_counts",
            "metric_summaries",
            "model_summaries",
            "failure_summaries",
        ):
            value = _mapping_copy(getattr(self, name))
            _validate_no_selection_claims(value)
            object.__setattr__(self, name, value)
        if self.provenance is not None and not isinstance(
            self.provenance, WavelengthValidationProvenance
        ):
            object.__setattr__(
                self,
                "provenance",
                WavelengthValidationProvenance.from_mapping(self.provenance),
            )
        object.__setattr__(self, "notes", _string_tuple(self.notes))
        if self.advisory_only is not True:
            raise ValueError("Wavelength validation aggregates must remain advisory-only.")
        if self.automatic_model_selection_applied is not False:
            raise ValueError(
                "Wavelength validation aggregates cannot apply automatic model selection."
            )
        extras = _mapping_copy(self.extra_fields)
        _validate_no_selection_claims(extras)
        object.__setattr__(self, "extra_fields", extras)

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> WavelengthValidationAggregate:
        """Construct an aggregate while retaining unknown fields."""
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        known = {
            "schema_version",
            "aggregate_id",
            "phase",
            "scenario_ids",
            "run_ids",
            "n_runs",
            "status_counts",
            "metric_summaries",
            "model_summaries",
            "failure_summaries",
            "provenance",
            "notes",
            "advisory_only",
            "automatic_model_selection_applied",
            "extra_fields",
        }
        provenance = payload.get("provenance")
        return cls(
            aggregate_id=payload.get("aggregate_id") or "",
            phase=_coerce_string_enum(
                WavelengthValidationPhase,
                payload.get("phase"),
                WavelengthValidationPhase.D1_SYNTHETIC_RECOVERY,
            ),
            schema_version=str(
                payload.get("schema_version")
                or WAVELENGTH_VALIDATION_SCHEMA_VERSION
            ),
            scenario_ids=_string_tuple(payload.get("scenario_ids")),
            run_ids=_string_tuple(payload.get("run_ids")),
            n_runs=payload.get("n_runs", 0),
            status_counts=payload.get("status_counts") or {},
            metric_summaries=payload.get("metric_summaries") or {},
            model_summaries=payload.get("model_summaries") or {},
            failure_summaries=payload.get("failure_summaries") or {},
            provenance=(
                WavelengthValidationProvenance.from_mapping(provenance)
                if isinstance(provenance, Mapping)
                else provenance
            ),
            notes=_string_tuple(payload.get("notes")),
            advisory_only=payload.get("advisory_only", True),
            automatic_model_selection_applied=payload.get(
                "automatic_model_selection_applied", False
            ),
            extra_fields=_extra_fields(payload, known),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-safe representation."""
        return {
            "schema_version": self.schema_version,
            "aggregate_id": self.aggregate_id,
            "phase": self.phase.value,
            "scenario_ids": list(self.scenario_ids),
            "run_ids": list(self.run_ids),
            "n_runs": self.n_runs,
            "status_counts": _mapping_copy(self.status_counts),
            "metric_summaries": _mapping_copy(self.metric_summaries),
            "model_summaries": _mapping_copy(self.model_summaries),
            "failure_summaries": _mapping_copy(self.failure_summaries),
            "provenance": (
                self.provenance.to_dict() if self.provenance is not None else None
            ),
            "notes": list(self.notes),
            "advisory_only": True,
            "automatic_model_selection_applied": False,
            "extra_fields": _mapping_copy(self.extra_fields),
        }


def as_wavelength_validation_scenario(
    value: WavelengthValidationScenario | Mapping[str, Any],
) -> WavelengthValidationScenario:
    """Return ``value`` as a typed validation scenario."""
    if isinstance(value, WavelengthValidationScenario):
        return value
    return WavelengthValidationScenario.from_mapping(value)


def as_wavelength_validation_run(
    value: WavelengthValidationRun | Mapping[str, Any],
) -> WavelengthValidationRun:
    """Return ``value`` as a typed validation run."""
    if isinstance(value, WavelengthValidationRun):
        return value
    return WavelengthValidationRun.from_mapping(value)


def as_wavelength_validation_aggregate(
    value: WavelengthValidationAggregate | Mapping[str, Any],
) -> WavelengthValidationAggregate:
    """Return ``value`` as a typed validation aggregate."""
    if isinstance(value, WavelengthValidationAggregate):
        return value
    return WavelengthValidationAggregate.from_mapping(value)
