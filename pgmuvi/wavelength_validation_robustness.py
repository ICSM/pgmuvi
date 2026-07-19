"""Synthetic D2 robustness and failure-boundary validation.

This module promotes truth-preserving synthetic cases into the D2 validation
phase, executes matched-truth or explicit case-by-model robustness matrices,
and records expected failures separately from unexpected technical failures.
It deliberately reuses the maintained D1 fitting and recovery extraction path
without reusing the D1 aggregate gates as D2 acceptance criteria.

The D2 outputs remain advisory.  They describe where wavelength-parameter
recovery, constraint diagnostics, or fitting cease to be reliable; they do not
rank models, install a winner, or perform automatic model selection.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
import math
from typing import Any

import numpy as np

from .wavelength_status import (
    ComparisonEligibility,
    DiagnosticValidity,
    ExecutionStage,
    ScientificUsability,
    TechnicalOutcome,
)
from .wavelength_validation import (
    RecoveryMetricDirection,
    WavelengthRecoveryMetric,
    WavelengthValidationAggregate,
    WavelengthValidationFailureExpectation,
    WavelengthValidationPhase,
    WavelengthValidationRun,
)
from .wavelength_validation_recovery import (
    SUPPORTED_SYNTHETIC_RECOVERY_MODELS,
    SyntheticWavelengthRecoveryThresholds,
    run_synthetic_wavelength_recovery,
)
from .wavelength_validation_synthetic import (
    DEFAULT_VALIDATION_BANDS,
    DEFAULT_VALIDATION_WAVELENGTHS,
    SyntheticWavelengthValidationCase,
    make_synthetic_wavelength_validation_case,
)

SYNTHETIC_WAVELENGTH_ROBUSTNESS_SCHEMA_VERSION = (
    "pgmuvi-synthetic-wavelength-robustness-v1"
)

_LightcurveFactory = Callable[[SyntheticWavelengthValidationCase], Any]
_SyntheticFitRunner = Callable[
    [Any, Mapping[str, Any], SyntheticWavelengthValidationCase], Any
]

__all__ = [
    "SYNTHETIC_WAVELENGTH_ROBUSTNESS_SCHEMA_VERSION",
    "SyntheticWavelengthRobustnessAxis",
    "SyntheticWavelengthRobustnessReport",
    "SyntheticWavelengthRobustnessSeverity",
    "SyntheticWavelengthRobustnessSpecification",
    "aggregate_synthetic_wavelength_robustness_runs",
    "canonical_synthetic_wavelength_robustness_cases",
    "canonical_synthetic_wavelength_robustness_specifications",
    "make_synthetic_wavelength_robustness_case",
    "run_synthetic_wavelength_robustness",
    "run_synthetic_wavelength_robustness_matrix",
]


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class SyntheticWavelengthRobustnessAxis(_StringEnum):
    """Single controlled perturbation represented by a D2 scenario."""

    REFERENCE = "reference"
    INDEPENDENT_TIME_GRIDS = "independent_time_grids"
    SPARSE_SAMPLING = "sparse_sampling"
    UNEVEN_BAND_COUNTS = "uneven_band_counts"
    LONGER_SPARSE_BASELINE = "longer_sparse_baseline"
    MISSING_BLUE_EDGE_BAND = "missing_blue_edge_band"
    MISSING_RED_EDGE_BAND = "missing_red_edge_band"
    MISSING_INTERIOR_BAND = "missing_interior_band"
    LARGE_WAVELENGTH_GAP = "large_wavelength_gap"
    HETEROSCEDASTIC_NOISE = "heteroscedastic_noise"
    HIGH_NOISE = "high_noise"
    WEAK_WAVELENGTH_DEPENDENCE = "weak_wavelength_dependence"
    STRONG_WAVELENGTH_DEPENDENCE = "strong_wavelength_dependence"
    JOINT_SM_SPARSE_ARD = "joint_sm_sparse_ard"
    INSUFFICIENT_PER_BAND_SAMPLING = "insufficient_per_band_sampling"


class SyntheticWavelengthRobustnessSeverity(_StringEnum):
    """Expected qualitative difficulty of a D2 perturbation."""

    REFERENCE = "reference"
    MILD = "mild"
    CHALLENGING = "challenging"
    BOUNDARY = "boundary"
    INVALID = "invalid"


def _coerce_enum(enum_type, value: Any, *, field_name: str):
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{field_name} must be one of: {allowed}.") from exc


def _json_safe(value: Any) -> Any:
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
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError, RuntimeError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except (TypeError, ValueError, RuntimeError):
            pass
    return str(value)


def _mapping_copy(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _json_safe(item) for key, item in value.items()}


def _deduplicated_strings(values: Sequence[Any]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item) for item in values))


def _as_expected_failure(
    value: WavelengthValidationFailureExpectation | Mapping[str, Any] | None,
) -> WavelengthValidationFailureExpectation | None:
    if value is None or isinstance(value, WavelengthValidationFailureExpectation):
        return value
    if isinstance(value, Mapping):
        return WavelengthValidationFailureExpectation.from_mapping(value)
    raise TypeError("expected_failure must be a failure expectation, mapping, or None")


@dataclass(frozen=True)
class SyntheticWavelengthRobustnessSpecification:
    """Declarative generator specification for one D2 scenario."""

    scenario_id: str
    axis: SyntheticWavelengthRobustnessAxis
    severity: SyntheticWavelengthRobustnessSeverity
    description: str
    generator_configuration: Mapping[str, Any]
    reference_scenario_id: str | None = None
    expected_failure: WavelengthValidationFailureExpectation | None = None
    tags: tuple[str, ...] = ()
    schema_version: str = SYNTHETIC_WAVELENGTH_ROBUSTNESS_SCHEMA_VERSION
    extra_fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        scenario_id = str(self.scenario_id or "").strip()
        description = str(self.description or "").strip()
        if not scenario_id or not description:
            raise ValueError("scenario_id and description must be non-empty.")
        object.__setattr__(self, "scenario_id", scenario_id)
        object.__setattr__(self, "description", description)
        object.__setattr__(
            self,
            "axis",
            _coerce_enum(
                SyntheticWavelengthRobustnessAxis,
                self.axis,
                field_name="axis",
            ),
        )
        object.__setattr__(
            self,
            "severity",
            _coerce_enum(
                SyntheticWavelengthRobustnessSeverity,
                self.severity,
                field_name="severity",
            ),
        )
        configuration = _mapping_copy(self.generator_configuration)
        for required in ("generating_model", "mean_kind", "covariance_kind"):
            if not str(configuration.get(required) or "").strip():
                raise ValueError(
                    f"generator_configuration requires {required!r}."
                )
        object.__setattr__(self, "generator_configuration", configuration)
        reference = (
            str(self.reference_scenario_id).strip()
            if self.reference_scenario_id is not None
            else None
        )
        object.__setattr__(self, "reference_scenario_id", reference or None)
        object.__setattr__(
            self,
            "expected_failure",
            _as_expected_failure(self.expected_failure),
        )
        object.__setattr__(
            self,
            "tags",
            _deduplicated_strings(self.tags),
        )
        object.__setattr__(self, "schema_version", str(self.schema_version))
        object.__setattr__(self, "extra_fields", _mapping_copy(self.extra_fields))

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> SyntheticWavelengthRobustnessSpecification:
        """Reconstruct a specification while retaining unknown fields."""
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        known = {
            "schema_version",
            "scenario_id",
            "axis",
            "severity",
            "description",
            "generator_configuration",
            "reference_scenario_id",
            "expected_failure",
            "tags",
            "extra_fields",
        }
        extras = _mapping_copy(payload.get("extra_fields"))
        for key, value in payload.items():
            if key not in known:
                extras[str(key)] = _json_safe(value)
        return cls(
            scenario_id=str(payload.get("scenario_id") or ""),
            axis=_coerce_enum(
                SyntheticWavelengthRobustnessAxis,
                payload.get("axis"),
                field_name="axis",
            ),
            severity=_coerce_enum(
                SyntheticWavelengthRobustnessSeverity,
                payload.get("severity"),
                field_name="severity",
            ),
            description=str(payload.get("description") or ""),
            generator_configuration=payload.get("generator_configuration") or {},
            reference_scenario_id=payload.get("reference_scenario_id"),
            expected_failure=_as_expected_failure(payload.get("expected_failure")),
            tags=tuple(payload.get("tags") or ()),
            schema_version=str(
                payload.get("schema_version")
                or SYNTHETIC_WAVELENGTH_ROBUSTNESS_SCHEMA_VERSION
            ),
            extra_fields=extras,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-safe specification."""
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "axis": self.axis.value,
            "severity": self.severity.value,
            "description": self.description,
            "generator_configuration": _mapping_copy(
                self.generator_configuration
            ),
            "reference_scenario_id": self.reference_scenario_id,
            "expected_failure": (
                self.expected_failure.to_dict()
                if self.expected_failure is not None
                else None
            ),
            "tags": list(self.tags),
            "extra_fields": _mapping_copy(self.extra_fields),
        }


def make_synthetic_wavelength_robustness_case(
    specification: SyntheticWavelengthRobustnessSpecification | Mapping[str, Any],
    *,
    seed: int = 0,
) -> SyntheticWavelengthValidationCase:
    """Build one truth-preserving D2 case from a declarative specification."""
    if not isinstance(specification, SyntheticWavelengthRobustnessSpecification):
        specification = SyntheticWavelengthRobustnessSpecification.from_mapping(
            specification
        )
    configuration = dict(specification.generator_configuration)
    configuration.update(
        {
            "scenario_id": specification.scenario_id,
            "description": specification.description,
            "seed": int(seed),
        }
    )
    case = make_synthetic_wavelength_validation_case(**configuration)
    scenario = case.scenario
    tags = _deduplicated_strings(
        (
            "d2",
            "synthetic",
            specification.axis.value,
            specification.severity.value,
            *(item for item in scenario.tags if item != "d1"),
            *specification.tags,
        )
    )
    metadata = dict(scenario.metadata)
    metadata.update(
        {
            "robustness_axis": specification.axis.value,
            "robustness_severity": specification.severity.value,
            "reference_scenario_id": specification.reference_scenario_id,
            "robustness_schema_version": (
                SYNTHETIC_WAVELENGTH_ROBUSTNESS_SCHEMA_VERSION
            ),
        }
    )
    sampling = dict(scenario.sampling_configuration)
    sampling.update(
        {
            "robustness_axis": specification.axis.value,
            "robustness_severity": specification.severity.value,
            "reference_scenario_id": specification.reference_scenario_id,
        }
    )
    fit_configuration = dict(scenario.fit_configuration)
    fit_configuration.update(
        {
            "validation_phase": (
                WavelengthValidationPhase.D2_ROBUSTNESS_FAILURE_BOUNDARY.value
            ),
            "selection_performed": False,
        }
    )
    provenance = scenario.provenance
    if provenance is not None:
        provenance_configuration = dict(provenance.configuration)
        provenance_configuration.update(
            {
                "robustness_schema_version": (
                    SYNTHETIC_WAVELENGTH_ROBUSTNESS_SCHEMA_VERSION
                ),
                "robustness_axis": specification.axis.value,
                "robustness_severity": specification.severity.value,
            }
        )
        provenance = replace(
            provenance,
            source_identifier=specification.scenario_id,
            configuration=provenance_configuration,
        )
    promoted = replace(
        scenario,
        phase=WavelengthValidationPhase.D2_ROBUSTNESS_FAILURE_BOUNDARY,
        description=specification.description,
        sampling_configuration=sampling,
        fit_configuration=fit_configuration,
        expected_failure=specification.expected_failure,
        tags=tags,
        provenance=provenance,
        metadata=metadata,
        extra_fields={
            **dict(scenario.extra_fields),
            **dict(specification.extra_fields),
        },
    )
    return replace(
        case,
        scenario=promoted,
        lightcurve_configuration={
            **dict(case.lightcurve_configuration),
            "name": specification.scenario_id,
        },
    )


def _base_configuration(**updates: Any) -> dict[str, Any]:
    configuration = {
        "generating_model": "2DSeparable",
        "mean_kind": "constant",
        "covariance_kind": "separable_quasi_periodic_rbf",
        "strength": "moderate",
        "wavelengths": DEFAULT_VALIDATION_WAVELENGTHS,
        "band_labels": DEFAULT_VALIDATION_BANDS,
        "n_per_band": 72,
        "period": 500.0,
        "n_cycles": 3.2,
        "irregular": True,
        "shared_time_grid": True,
        "noise_sigma": 0.15,
        "heteroscedastic_fraction": 0.0,
        "xtransform": "minmax",
    }
    configuration.update(updates)
    return configuration


def canonical_synthetic_wavelength_robustness_specifications(
) -> tuple[SyntheticWavelengthRobustnessSpecification, ...]:
    """Return the canonical controlled D2 perturbation specifications."""
    reference_id = "d2-reference-separable-moderate"
    insufficient_sampling = WavelengthValidationFailureExpectation(
        failure_code="synthetic_recovery_fit_failed",
        stage=ExecutionStage.OPTIMIZATION,
        acceptable_exception_types=(
            "FitError",
            "PSDError",
            "RuntimeError",
            "ValueError",
        ),
        rationale=(
            "Two rows per band are below the maintained wavelength-estimation "
            "and consensus-quality requirements; the matched separable fit is "
            "expected to fail cleanly rather than produce recovery evidence."
        ),
        diagnostics={
            "applicable_models": ["2DSeparable"],
            "boundary_kind": "insufficient_per_band_sampling",
        },
    )
    definitions = (
        (
            reference_id,
            SyntheticWavelengthRobustnessAxis.REFERENCE,
            SyntheticWavelengthRobustnessSeverity.REFERENCE,
            "Dense shared-grid separable reference for paired D2 comparisons.",
            _base_configuration(),
            None,
        ),
        (
            "d2-independent-time-grids",
            SyntheticWavelengthRobustnessAxis.INDEPENDENT_TIME_GRIDS,
            SyntheticWavelengthRobustnessSeverity.MILD,
            "Dense independent irregular time grids across wavelength bands.",
            _base_configuration(shared_time_grid=False),
            None,
        ),
        (
            "d2-sparse-shared-grid",
            SyntheticWavelengthRobustnessAxis.SPARSE_SAMPLING,
            SyntheticWavelengthRobustnessSeverity.CHALLENGING,
            "Thirty-six shared-grid observations per band over 3.2 periods.",
            _base_configuration(
                generating_model="2DWavelengthDependent",
                mean_kind="quadratic",
                n_per_band=36,
            ),
            None,
        ),
        (
            "d2-uneven-band-counts",
            SyntheticWavelengthRobustnessAxis.UNEVEN_BAND_COUNTS,
            SyntheticWavelengthRobustnessSeverity.CHALLENGING,
            "Strongly uneven observation counts on independent band grids.",
            _base_configuration(
                generating_model="2DDustMean",
                mean_kind="dust",
                strength="strong",
                n_per_band=(72, 54, 42, 30, 24, 18),
                shared_time_grid=False,
            ),
            None,
        ),
        (
            "d2-longer-sparse-baseline",
            SyntheticWavelengthRobustnessAxis.LONGER_SPARSE_BASELINE,
            SyntheticWavelengthRobustnessSeverity.CHALLENGING,
            "Thirty-six independent observations per band over 6.4 periods.",
            _base_configuration(
                generating_model="2DPowerLawMean",
                mean_kind="power_law",
                n_per_band=36,
                n_cycles=6.4,
                shared_time_grid=False,
            ),
            None,
        ),
        (
            "d2-missing-blue-edge-band",
            SyntheticWavelengthRobustnessAxis.MISSING_BLUE_EDGE_BAND,
            SyntheticWavelengthRobustnessSeverity.MILD,
            "Dense sampling after removal of the shortest-wavelength band.",
            _base_configuration(
                wavelengths=DEFAULT_VALIDATION_WAVELENGTHS[1:],
                band_labels=DEFAULT_VALIDATION_BANDS[1:],
            ),
            None,
        ),
        (
            "d2-missing-red-edge-band",
            SyntheticWavelengthRobustnessAxis.MISSING_RED_EDGE_BAND,
            SyntheticWavelengthRobustnessSeverity.MILD,
            "Dense sampling after removal of the longest-wavelength band.",
            _base_configuration(
                wavelengths=DEFAULT_VALIDATION_WAVELENGTHS[:-1],
                band_labels=DEFAULT_VALIDATION_BANDS[:-1],
            ),
            None,
        ),
        (
            "d2-missing-interior-band",
            SyntheticWavelengthRobustnessAxis.MISSING_INTERIOR_BAND,
            SyntheticWavelengthRobustnessSeverity.CHALLENGING,
            "Turning-point mean recovery after removal of an interior band.",
            _base_configuration(
                generating_model="2DWavelengthDependent",
                mean_kind="quadratic",
                strength="strong",
                turning_point=True,
                wavelengths=(0.55, 0.80, 1.25, 3.40, 4.60),
                band_labels=("V", "I", "J", "W1", "W2"),
            ),
            None,
        ),
        (
            "d2-large-wavelength-gap",
            SyntheticWavelengthRobustnessAxis.LARGE_WAVELENGTH_GAP,
            SyntheticWavelengthRobustnessSeverity.BOUNDARY,
            "Four-band dust-mean case with a large J-to-W2 wavelength gap.",
            _base_configuration(
                generating_model="2DDustMean",
                mean_kind="dust",
                strength="strong",
                wavelengths=(0.55, 0.80, 1.25, 4.60),
                band_labels=("V", "I", "J", "W2"),
            ),
            None,
        ),
        (
            "d2-heteroscedastic-noise",
            SyntheticWavelengthRobustnessAxis.HETEROSCEDASTIC_NOISE,
            SyntheticWavelengthRobustnessSeverity.CHALLENGING,
            "Power-law mean with strongly heteroscedastic reported errors.",
            _base_configuration(
                generating_model="2DPowerLawMean",
                mean_kind="power_law",
                heteroscedastic_fraction=1.0,
            ),
            None,
        ),
        (
            "d2-high-noise",
            SyntheticWavelengthRobustnessAxis.HIGH_NOISE,
            SyntheticWavelengthRobustnessSeverity.BOUNDARY,
            "Quadratic wavelength mean with three-times nominal noise.",
            _base_configuration(
                generating_model="2DWavelengthDependent",
                mean_kind="quadratic",
                noise_sigma=0.45,
            ),
            None,
        ),
        (
            "d2-weak-wavelength-dependence",
            SyntheticWavelengthRobustnessAxis.WEAK_WAVELENGTH_DEPENDENCE,
            SyntheticWavelengthRobustnessSeverity.BOUNDARY,
            "Weak separable wavelength covariance and nearly constant mean.",
            _base_configuration(strength="weak"),
            None,
        ),
        (
            "d2-strong-wavelength-dependence",
            SyntheticWavelengthRobustnessAxis.STRONG_WAVELENGTH_DEPENDENCE,
            SyntheticWavelengthRobustnessSeverity.CHALLENGING,
            "Strong dust mean and short wavelength-correlation scale.",
            _base_configuration(
                generating_model="2DDustMean",
                mean_kind="dust",
                strength="strong",
            ),
            None,
        ),
        (
            "d2-joint-sm-sparse-independent",
            SyntheticWavelengthRobustnessAxis.JOINT_SM_SPARSE_ARD,
            SyntheticWavelengthRobustnessSeverity.CHALLENGING,
            "Joint spectral-mixture ARD case with sparse independent grids.",
            _base_configuration(
                generating_model="2D",
                covariance_kind="joint_spectral_mixture_ard",
                n_per_band=36,
                shared_time_grid=False,
            ),
            None,
        ),
        (
            "d2-insufficient-per-band-sampling",
            SyntheticWavelengthRobustnessAxis.INSUFFICIENT_PER_BAND_SAMPLING,
            SyntheticWavelengthRobustnessSeverity.INVALID,
            "Two-band, two-row-per-band consensus failure boundary.",
            _base_configuration(
                wavelengths=(0.80, 3.40),
                band_labels=("I", "W1"),
                n_per_band=2,
                shared_time_grid=False,
            ),
            insufficient_sampling,
        ),
    )
    return tuple(
        SyntheticWavelengthRobustnessSpecification(
            scenario_id=scenario_id,
            axis=axis,
            severity=severity,
            description=description,
            generator_configuration=configuration,
            reference_scenario_id=(
                None
                if axis is SyntheticWavelengthRobustnessAxis.REFERENCE
                else reference_id
            ),
            expected_failure=expected_failure,
        )
        for (
            scenario_id,
            axis,
            severity,
            description,
            configuration,
            expected_failure,
        ) in definitions
    )


def canonical_synthetic_wavelength_robustness_cases(
    *, seed: int = 0
) -> tuple[SyntheticWavelengthValidationCase, ...]:
    """Build one deterministic case for every canonical D2 perturbation."""
    return tuple(
        make_synthetic_wavelength_robustness_case(
            specification,
            seed=int(seed) + index,
        )
        for index, specification in enumerate(
            canonical_synthetic_wavelength_robustness_specifications()
        )
    )


def _as_case(value: Any) -> SyntheticWavelengthValidationCase:
    if isinstance(value, SyntheticWavelengthValidationCase):
        case = value
    elif isinstance(value, Mapping):
        case = SyntheticWavelengthValidationCase.from_mapping(value)
    else:
        raise TypeError("case must be a synthetic validation case or mapping")
    if (
        case.scenario.phase
        is not WavelengthValidationPhase.D2_ROBUSTNESS_FAILURE_BOUNDARY
    ):
        raise ValueError("D2 robustness runners require a D2 scenario.")
    return case


def _is_failed(run: WavelengthValidationRun) -> bool:
    return bool(
        run.status is not None
        and run.status.technical_outcome is TechnicalOutcome.FAILED
    )


def _expectation_applies(
    expectation: WavelengthValidationFailureExpectation,
    model: str,
) -> bool:
    models = expectation.diagnostics.get("applicable_models")
    if not models:
        return True
    return str(model) in {str(item) for item in models}


def _evaluate_failure_expectation(
    case: SyntheticWavelengthValidationCase,
    run: WavelengthValidationRun,
) -> dict[str, Any]:
    expectation = case.scenario.expected_failure
    actual_stage = (
        run.failure.stage
        if run.failure is not None
        else (
            run.status.execution_stage
            if run.status is not None
            else ExecutionStage.UNKNOWN
        )
    )
    actual_technical = (
        run.status.technical_outcome
        if run.status is not None
        else TechnicalOutcome.NOT_ATTEMPTED
    )
    actual_diagnostic = (
        run.status.diagnostic_validity
        if run.status is not None
        else DiagnosticValidity.NOT_EVALUATED
    )
    actual_usability = (
        run.status.scientific_usability
        if run.status is not None
        else ScientificUsability.NOT_EVALUATED
    )
    actual_eligibility = (
        run.status.comparison_eligibility
        if run.status is not None
        else ComparisonEligibility.NOT_EVALUATED
    )
    if expectation is None:
        return {
            "expected": False,
            "applicable": False,
            "matched": None,
            "unexpected_failure": _is_failed(run),
            "actual_failure_code": (
                run.failure.failure_code if run.failure is not None else None
            ),
            "actual_failure_stage": actual_stage.value,
            "actual_exception_type": (
                run.failure.exception_type if run.failure is not None else None
            ),
        }

    applicable = _expectation_applies(expectation, run.model)
    if not applicable:
        return {
            "expected": True,
            "applicable": False,
            "matched": None,
            "unexpected_failure": _is_failed(run),
            "expected_failure": expectation.to_dict(),
            "actual_failure_code": (
                run.failure.failure_code if run.failure is not None else None
            ),
            "actual_failure_stage": actual_stage.value,
            "actual_exception_type": (
                run.failure.exception_type if run.failure is not None else None
            ),
        }

    actual_code = run.failure.failure_code if run.failure is not None else None
    actual_exception = (
        run.failure.exception_type if run.failure is not None else None
    )
    checks = {
        "failure_code": actual_code == expectation.failure_code,
        "stage": actual_stage is expectation.stage,
        "technical_outcome": actual_technical is expectation.technical_outcome,
        "diagnostic_validity": (
            actual_diagnostic is expectation.diagnostic_validity
        ),
        "scientific_usability": (
            actual_usability is expectation.scientific_usability
        ),
        "comparison_eligibility": (
            actual_eligibility is expectation.comparison_eligibility
        ),
        "exception_type": (
            not expectation.acceptable_exception_types
            or actual_exception in expectation.acceptable_exception_types
        ),
    }
    matched = all(checks.values())
    return {
        "expected": True,
        "applicable": True,
        "matched": matched,
        "unexpected_failure": False,
        "checks": checks,
        "expected_failure": expectation.to_dict(),
        "actual_failure_code": actual_code,
        "actual_failure_stage": actual_stage.value,
        "actual_exception_type": actual_exception,
    }


def _annotate_robustness_run(
    case: SyntheticWavelengthValidationCase,
    run: WavelengthValidationRun,
) -> WavelengthValidationRun:
    evaluation = _evaluate_failure_expectation(case, run)
    if evaluation["expected"] and evaluation["applicable"]:
        metric = WavelengthRecoveryMetric(
            name="expected_failure_contract_matched",
            value=evaluation["matched"],
            truth_value=True,
            available=True,
            passed=bool(evaluation["matched"]),
            direction=RecoveryMetricDirection.EXACT_MATCH,
            scope="d2_failure_boundary",
            model=run.model,
            summary="Whether the classified failure matched the D2 contract.",
            provenance={
                "robustness_schema_version": (
                    SYNTHETIC_WAVELENGTH_ROBUSTNESS_SCHEMA_VERSION
                )
            },
        )
    else:
        absent = not bool(evaluation["unexpected_failure"])
        metric = WavelengthRecoveryMetric(
            name="unexpected_technical_failure_absent",
            value=absent,
            truth_value=True,
            available=True,
            passed=absent,
            direction=RecoveryMetricDirection.EXACT_MATCH,
            scope="d2_robustness",
            model=run.model,
            summary="Whether a non-expected D2 attempt avoided technical failure.",
            provenance={
                "robustness_schema_version": (
                    SYNTHETIC_WAVELENGTH_ROBUSTNESS_SCHEMA_VERSION
                )
            },
        )
    scenario = case.scenario
    return replace(
        run,
        metrics=(*run.metrics, metric),
        notes=(*run.notes, "D2 robustness annotation applied."),
        extra_fields={
            **dict(run.extra_fields),
            "validation_phase": (
                WavelengthValidationPhase.D2_ROBUSTNESS_FAILURE_BOUNDARY.value
            ),
            "robustness_schema_version": (
                SYNTHETIC_WAVELENGTH_ROBUSTNESS_SCHEMA_VERSION
            ),
            "robustness_axis": scenario.metadata.get("robustness_axis"),
            "robustness_severity": scenario.metadata.get("robustness_severity"),
            "reference_scenario_id": scenario.metadata.get(
                "reference_scenario_id"
            ),
            "expected_failure_evaluation": evaluation,
            "generating_model": scenario.truth.generating_model,
        },
    )


def run_synthetic_wavelength_robustness(
    case: SyntheticWavelengthValidationCase | Mapping[str, Any],
    model: str | None = None,
    *,
    fit_kwargs: Mapping[str, Any] | None = None,
    thresholds: SyntheticWavelengthRecoveryThresholds | Mapping[str, Any] | None = None,
    lightcurve_factory: _LightcurveFactory | None = None,
    fit_runner: _SyntheticFitRunner | None = None,
    stop_on_error: bool = False,
) -> WavelengthValidationRun:
    """Execute and annotate one D2 synthetic robustness attempt."""
    case = _as_case(case)
    resolved_model = str(
        model or case.scenario.truth.generating_model or ""
    ).strip()
    run = run_synthetic_wavelength_recovery(
        case,
        resolved_model,
        fit_kwargs=fit_kwargs,
        thresholds=thresholds,
        lightcurve_factory=lightcurve_factory,
        fit_runner=fit_runner,
        stop_on_error=stop_on_error,
    )
    return _annotate_robustness_run(case, run)


def _finite_float(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _metric_summaries(
    runs: Sequence[WavelengthValidationRun],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[WavelengthRecoveryMetric]] = {}
    for run in runs:
        for metric in run.metrics:
            grouped.setdefault(metric.name, []).append(metric)
    output: dict[str, dict[str, Any]] = {}
    for name, metrics in grouped.items():
        available = [metric for metric in metrics if metric.available]
        passed = [metric for metric in available if metric.passed is True]
        failed = [metric for metric in available if metric.passed is False]
        numeric = []
        for metric in available:
            if isinstance(metric.value, bool):
                continue
            value = _finite_float(metric.value)
            if value is not None:
                numeric.append(value)
        summary: dict[str, Any] = {
            "n_reported": len(metrics),
            "n_available": len(available),
            "n_unavailable": len(metrics) - len(available),
            "n_passed": len(passed),
            "n_failed": len(failed),
            "pass_fraction": (
                len(passed) / (len(passed) + len(failed))
                if passed or failed
                else None
            ),
        }
        if numeric:
            array = np.asarray(numeric, dtype=float)
            summary.update(
                {
                    "minimum": float(np.min(array)),
                    "median": float(np.median(array)),
                    "maximum": float(np.max(array)),
                    "p90": float(np.percentile(array, 90.0)),
                }
            )
        output[name] = summary
    return output


def _group_summary(
    runs: Sequence[WavelengthValidationRun],
) -> dict[str, Any]:
    completed_outcomes = {
        TechnicalOutcome.COMPLETED,
        TechnicalOutcome.COMPLETED_WITH_WARNINGS,
        TechnicalOutcome.COMPLETED_WITH_RECOVERY,
    }
    n_completed = sum(
        run.status is not None
        and run.status.technical_outcome in completed_outcomes
        for run in runs
    )
    evaluations = [
        run.extra_fields.get("expected_failure_evaluation") or {}
        for run in runs
    ]
    n_expected = sum(
        bool(item.get("expected")) and bool(item.get("applicable"))
        for item in evaluations
    )
    n_expected_matched = sum(item.get("matched") is True for item in evaluations)
    n_expected_mismatched = sum(
        item.get("matched") is False for item in evaluations
    )
    n_unexpected_failures = sum(
        bool(item.get("unexpected_failure")) for item in evaluations
    )
    return {
        "n_runs": len(runs),
        "n_completed": n_completed,
        "n_failed": len(runs) - n_completed,
        "completion_fraction": n_completed / len(runs) if runs else None,
        "n_expected_failure_contracts": n_expected,
        "n_expected_failures_matched": n_expected_matched,
        "n_expected_failures_mismatched": n_expected_mismatched,
        "n_unexpected_failures": n_unexpected_failures,
        "metric_summaries": _metric_summaries(runs),
    }


def aggregate_synthetic_wavelength_robustness_runs(
    runs: Sequence[WavelengthValidationRun | Mapping[str, Any]],
    *,
    aggregate_id: str = "d2-synthetic-wavelength-robustness",
) -> WavelengthValidationAggregate:
    """Aggregate D2 runs without importing or weakening D1 gates."""
    normalized = []
    for run in runs:
        if isinstance(run, WavelengthValidationRun):
            normalized_run = run
        elif isinstance(run, Mapping):
            normalized_run = WavelengthValidationRun.from_mapping(run)
        else:
            raise TypeError("runs must contain validation runs or mappings")
        phase = str(normalized_run.extra_fields.get("validation_phase") or "")
        if phase != WavelengthValidationPhase.D2_ROBUSTNESS_FAILURE_BOUNDARY.value:
            raise ValueError("D2 aggregates require D2-annotated robustness runs.")
        normalized.append(normalized_run)

    status_counts: dict[str, int] = {}
    failures_by_stage: dict[str, int] = {}
    failures_by_code: dict[str, int] = {}
    by_model: dict[str, list[WavelengthValidationRun]] = {}
    by_axis: dict[str, list[WavelengthValidationRun]] = {}
    by_severity: dict[str, list[WavelengthValidationRun]] = {}
    for run in normalized:
        technical = (
            run.status.technical_outcome.value
            if run.status is not None
            else "not_evaluated"
        )
        status_counts[technical] = status_counts.get(technical, 0) + 1
        if run.failure is not None:
            stage = run.failure.stage.value
            code = run.failure.failure_code
            failures_by_stage[stage] = failures_by_stage.get(stage, 0) + 1
            failures_by_code[code] = failures_by_code.get(code, 0) + 1
        by_model.setdefault(run.model, []).append(run)
        axis = str(run.extra_fields.get("robustness_axis") or "unclassified")
        severity = str(
            run.extra_fields.get("robustness_severity") or "unclassified"
        )
        by_axis.setdefault(axis, []).append(run)
        by_severity.setdefault(severity, []).append(run)

    axis_summaries = {
        name: _group_summary(group) for name, group in by_axis.items()
    }
    severity_summaries = {
        name: _group_summary(group) for name, group in by_severity.items()
    }
    model_summaries = {
        name: _group_summary(group) for name, group in by_model.items()
    }
    overall = _group_summary(normalized)
    return WavelengthValidationAggregate(
        aggregate_id=aggregate_id,
        phase=WavelengthValidationPhase.D2_ROBUSTNESS_FAILURE_BOUNDARY,
        scenario_ids=tuple(
            dict.fromkeys(run.scenario_id for run in normalized)
        ),
        run_ids=tuple(run.run_id for run in normalized),
        n_runs=len(normalized),
        status_counts=status_counts,
        metric_summaries=_metric_summaries(normalized),
        model_summaries=model_summaries,
        failure_summaries={
            "by_stage": failures_by_stage,
            "by_code": failures_by_code,
            "n_failures": sum(failures_by_stage.values()),
            "n_expected_failure_contracts": overall[
                "n_expected_failure_contracts"
            ],
            "n_expected_failures_matched": overall[
                "n_expected_failures_matched"
            ],
            "n_expected_failures_mismatched": overall[
                "n_expected_failures_mismatched"
            ],
            "n_unexpected_failures": overall["n_unexpected_failures"],
        },
        notes=(
            "D2 summaries describe robustness and failure boundaries; they are "
            "not D1 recovery gates or automatic model-selection evidence.",
        ),
        automatic_model_selection_applied=False,
        extra_fields={
            "robustness_schema_version": (
                SYNTHETIC_WAVELENGTH_ROBUSTNESS_SCHEMA_VERSION
            ),
            "axis_summaries": axis_summaries,
            "severity_summaries": severity_summaries,
            "overall_summary": overall,
            "d2_boundary_summary": {
                "empirical_boundaries_calibrated": False,
                "d1_gates_reused_as_d2_gates": False,
                "d1_thresholds_available_as_descriptive_metrics": True,
                "automatic_model_selection_applied": False,
            },
        },
    )


@dataclass(frozen=True)
class SyntheticWavelengthRobustnessReport:
    """Typed collection of D2 runs and their failure-aware aggregate."""

    report_id: str
    cases: tuple[SyntheticWavelengthValidationCase, ...]
    runs: tuple[WavelengthValidationRun, ...]
    aggregate: WavelengthValidationAggregate
    schema_version: str = SYNTHETIC_WAVELENGTH_ROBUSTNESS_SCHEMA_VERSION
    notes: tuple[str, ...] = ()
    advisory_only: bool = True
    automatic_model_selection_applied: bool = False
    extra_fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        report_id = str(self.report_id or "").strip()
        if not report_id:
            raise ValueError("report_id must be non-empty.")
        object.__setattr__(self, "report_id", report_id)
        normalized_cases = tuple(_as_case(case) for case in self.cases)
        object.__setattr__(self, "cases", normalized_cases)
        normalized_runs = []
        for run in self.runs:
            if isinstance(run, WavelengthValidationRun):
                normalized_runs.append(run)
            elif isinstance(run, Mapping):
                normalized_runs.append(WavelengthValidationRun.from_mapping(run))
            else:
                raise TypeError("runs must contain validation run records")
        object.__setattr__(self, "runs", tuple(normalized_runs))
        if not isinstance(self.aggregate, WavelengthValidationAggregate):
            object.__setattr__(
                self,
                "aggregate",
                WavelengthValidationAggregate.from_mapping(self.aggregate),
            )
        if (
            self.aggregate.phase
            is not WavelengthValidationPhase.D2_ROBUSTNESS_FAILURE_BOUNDARY
        ):
            raise ValueError("Robustness reports require a D2 aggregate.")
        if self.aggregate.n_runs != len(self.runs):
            raise ValueError("aggregate.n_runs must match the number of runs.")
        object.__setattr__(self, "schema_version", str(self.schema_version))
        object.__setattr__(
            self,
            "notes",
            tuple(str(item) for item in self.notes),
        )
        if self.advisory_only is not True:
            raise ValueError("Robustness reports must remain advisory-only.")
        if self.automatic_model_selection_applied is not False:
            raise ValueError("Robustness reports cannot select a model.")
        object.__setattr__(self, "extra_fields", _mapping_copy(self.extra_fields))

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> SyntheticWavelengthRobustnessReport:
        """Reconstruct a D2 report while retaining unknown fields."""
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        known = {
            "schema_version",
            "report_id",
            "cases",
            "runs",
            "aggregate",
            "notes",
            "advisory_only",
            "automatic_model_selection_applied",
            "extra_fields",
        }
        extras = _mapping_copy(payload.get("extra_fields"))
        for key, value in payload.items():
            if key not in known:
                extras[str(key)] = _json_safe(value)
        return cls(
            report_id=str(payload.get("report_id") or ""),
            cases=tuple(
                SyntheticWavelengthValidationCase.from_mapping(item)
                if isinstance(item, Mapping)
                else item
                for item in payload.get("cases") or ()
            ),
            runs=tuple(
                WavelengthValidationRun.from_mapping(item)
                if isinstance(item, Mapping)
                else item
                for item in payload.get("runs") or ()
            ),
            aggregate=WavelengthValidationAggregate.from_mapping(
                payload.get("aggregate") or {}
            ),
            schema_version=str(
                payload.get("schema_version")
                or SYNTHETIC_WAVELENGTH_ROBUSTNESS_SCHEMA_VERSION
            ),
            notes=tuple(payload.get("notes") or ()),
            advisory_only=payload.get("advisory_only", True),
            automatic_model_selection_applied=payload.get(
                "automatic_model_selection_applied", False
            ),
            extra_fields=extras,
        )

    def to_dict(self, *, include_observations: bool = True) -> dict[str, Any]:
        """Return a stable JSON-safe D2 report envelope."""
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "cases": [
                case.to_dict(include_observations=include_observations)
                for case in self.cases
            ],
            "runs": [run.to_dict() for run in self.runs],
            "aggregate": self.aggregate.to_dict(),
            "notes": list(self.notes),
            "advisory_only": True,
            "automatic_model_selection_applied": False,
            "extra_fields": _mapping_copy(self.extra_fields),
        }


def run_synthetic_wavelength_robustness_matrix(
    cases: Sequence[SyntheticWavelengthValidationCase | Mapping[str, Any]],
    *,
    models: Sequence[str] | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
    per_model_fit_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    thresholds: SyntheticWavelengthRecoveryThresholds | Mapping[str, Any] | None = None,
    lightcurve_factory: _LightcurveFactory | None = None,
    fit_runner: _SyntheticFitRunner | None = None,
    stop_on_error: bool = False,
    report_id: str = "d2-synthetic-wavelength-robustness",
) -> SyntheticWavelengthRobustnessReport:
    """Execute a failure-aware D2 matrix without model selection.

    By default each case is fitted only with its generating model.  Passing
    ``models`` requests an explicit case-by-model cross product.
    """
    normalized_cases = tuple(_as_case(case) for case in cases)
    if not normalized_cases:
        raise ValueError("cases must contain at least one D2 synthetic case.")
    resolved_models = None
    if models is not None:
        resolved_models = tuple(str(model) for model in models)
        if not resolved_models:
            raise ValueError("models must contain at least one model.")
        unsupported = [
            model
            for model in resolved_models
            if model not in SUPPORTED_SYNTHETIC_RECOVERY_MODELS
        ]
        if unsupported:
            raise ValueError(
                "Unsupported synthetic robustness model(s): "
                + ", ".join(unsupported)
            )
    per_model_fit_kwargs = dict(per_model_fit_kwargs or {})
    runs = []
    for case in normalized_cases:
        case_models = (
            resolved_models
            if resolved_models is not None
            else (case.scenario.truth.generating_model,)
        )
        for model in case_models:
            model_configuration = dict(fit_kwargs or {})
            model_configuration.update(
                dict(per_model_fit_kwargs.get(model) or {})
            )
            runs.append(
                run_synthetic_wavelength_robustness(
                    case,
                    model,
                    fit_kwargs=model_configuration,
                    thresholds=thresholds,
                    lightcurve_factory=lightcurve_factory,
                    fit_runner=fit_runner,
                    stop_on_error=stop_on_error,
                )
            )
    aggregate = aggregate_synthetic_wavelength_robustness_runs(
        runs,
        aggregate_id=report_id,
    )
    return SyntheticWavelengthRobustnessReport(
        report_id=report_id,
        cases=normalized_cases,
        runs=tuple(runs),
        aggregate=aggregate,
        notes=(
            "D2 completion, recovery, boundary, and expected-failure summaries "
            "are descriptive validation outputs; no winner is installed.",
        ),
    )
