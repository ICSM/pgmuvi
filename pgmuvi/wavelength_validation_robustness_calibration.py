"""Multi-seed D2 robustness calibration and empirical boundary summaries.

This module executes truth-matched reference/perturbation populations, retains
classified technical failures, summarizes recovery degradation and parameter
boundary pressure, and assigns advisory empirical boundary classes.  The D2
policy is deliberately distinct from the frozen D1 aggregate gates.  Per-run D1
recovery metrics remain descriptive inputs, but D2 classes are based on paired
changes relative to a truth-matched reference population.

No function in this module ranks candidate models, installs a winner, or
performs automatic model selection.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np

from .wavelength_status import TechnicalOutcome
from .wavelength_validation import (
    WavelengthRecoveryMetric,
    WavelengthValidationAggregate,
    WavelengthValidationPhase,
    WavelengthValidationRun,
)
from .wavelength_validation_recovery import (
    SyntheticWavelengthRecoveryThresholds,
)
from .wavelength_validation_robustness import (
    SYNTHETIC_WAVELENGTH_ROBUSTNESS_SCHEMA_VERSION,
    SyntheticWavelengthRobustnessAxis,
    SyntheticWavelengthRobustnessSpecification,
    aggregate_synthetic_wavelength_robustness_runs,
    canonical_synthetic_wavelength_robustness_specifications,
    make_synthetic_wavelength_robustness_case,
    run_synthetic_wavelength_robustness,
)
from .wavelength_validation_synthetic import SyntheticWavelengthValidationCase

SYNTHETIC_WAVELENGTH_ROBUSTNESS_CALIBRATION_SCHEMA_VERSION = (
    "pgmuvi-synthetic-wavelength-robustness-calibration-v1"
)

_CORE_RECOVERY_METRICS = (
    "period_relative_error",
    "wavelength_lengthscale_factor_error",
    "mean_law_normalized_rmse",
    "sm_ard_factor_error",
)

_LightcurveFactory = Callable[[SyntheticWavelengthValidationCase], Any]
_SyntheticFitRunner = Callable[
    [Any, Mapping[str, Any], SyntheticWavelengthValidationCase], Any
]

__all__ = [
    "DEFAULT_SYNTHETIC_WAVELENGTH_ROBUSTNESS_CALIBRATION_THRESHOLDS",
    "SYNTHETIC_WAVELENGTH_ROBUSTNESS_CALIBRATION_SCHEMA_VERSION",
    "SyntheticWavelengthRobustnessBoundaryClass",
    "SyntheticWavelengthRobustnessBoundaryRecord",
    "SyntheticWavelengthRobustnessCalibrationReport",
    "SyntheticWavelengthRobustnessCalibrationThresholds",
    "calibrate_synthetic_wavelength_robustness_runs",
    "run_synthetic_wavelength_robustness_population",
    "summarize_synthetic_wavelength_robustness_population",
    "validate_synthetic_wavelength_robustness_reference_pairs",
]


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class SyntheticWavelengthRobustnessBoundaryClass(_StringEnum):
    """Advisory empirical class assigned to one D2 scenario population."""

    REFERENCE = "reference"
    ROBUST = "robust"
    DEGRADED = "degraded"
    FAILURE_BOUNDARY = "failure_boundary"
    EXPECTED_FAILURE_BOUNDARY = "expected_failure_boundary"
    INCONCLUSIVE = "inconclusive"


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


def _finite_float(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _coerce_boundary_class(value: Any) -> SyntheticWavelengthRobustnessBoundaryClass:
    if isinstance(value, SyntheticWavelengthRobustnessBoundaryClass):
        return value
    try:
        return SyntheticWavelengthRobustnessBoundaryClass(str(value))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in SyntheticWavelengthRobustnessBoundaryClass)
        raise ValueError(f"classification must be one of: {allowed}.") from exc


@dataclass(frozen=True)
class SyntheticWavelengthRobustnessCalibrationThresholds:
    """D2 population-comparison policy, distinct from D1 aggregate gates."""

    minimum_seed_count: int = 10
    robust_completion_drop_maximum: float = 0.10
    degraded_completion_drop_maximum: float = 0.25
    robust_metric_pass_fraction_drop_maximum: float = 0.15
    degraded_metric_pass_fraction_drop_maximum: float = 0.35
    failure_boundary_completion_fraction_maximum: float = 0.50
    failure_boundary_unexpected_failure_fraction_minimum: float = 0.50
    expected_failure_match_fraction_minimum: float = 0.90
    boundary_pressure_warning_fraction: float = 0.50
    near_bound_tolerance_fraction: float = 0.05
    at_bound_tolerance_fraction: float = 1.0e-6

    def __post_init__(self) -> None:
        minimum = int(self.minimum_seed_count)
        if minimum <= 0:
            raise ValueError("minimum_seed_count must be positive.")
        object.__setattr__(self, "minimum_seed_count", minimum)
        fraction_fields = (
            "robust_completion_drop_maximum",
            "degraded_completion_drop_maximum",
            "robust_metric_pass_fraction_drop_maximum",
            "degraded_metric_pass_fraction_drop_maximum",
            "failure_boundary_completion_fraction_maximum",
            "failure_boundary_unexpected_failure_fraction_minimum",
            "expected_failure_match_fraction_minimum",
            "boundary_pressure_warning_fraction",
            "near_bound_tolerance_fraction",
            "at_bound_tolerance_fraction",
        )
        for name in fraction_fields:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1].")
            object.__setattr__(self, name, value)
        if (
            self.robust_completion_drop_maximum
            > self.degraded_completion_drop_maximum
        ):
            raise ValueError(
                "robust_completion_drop_maximum cannot exceed the degraded limit."
            )
        if (
            self.robust_metric_pass_fraction_drop_maximum
            > self.degraded_metric_pass_fraction_drop_maximum
        ):
            raise ValueError(
                "robust metric degradation cannot exceed the degraded limit."
            )
        if self.at_bound_tolerance_fraction > self.near_bound_tolerance_fraction:
            raise ValueError(
                "at_bound_tolerance_fraction cannot exceed the near-bound tolerance."
            )

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> SyntheticWavelengthRobustnessCalibrationThresholds:
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        return cls(
            minimum_seed_count=payload.get("minimum_seed_count", 10),
            robust_completion_drop_maximum=payload.get(
                "robust_completion_drop_maximum", 0.10
            ),
            degraded_completion_drop_maximum=payload.get(
                "degraded_completion_drop_maximum", 0.25
            ),
            robust_metric_pass_fraction_drop_maximum=payload.get(
                "robust_metric_pass_fraction_drop_maximum", 0.15
            ),
            degraded_metric_pass_fraction_drop_maximum=payload.get(
                "degraded_metric_pass_fraction_drop_maximum", 0.35
            ),
            failure_boundary_completion_fraction_maximum=payload.get(
                "failure_boundary_completion_fraction_maximum", 0.50
            ),
            failure_boundary_unexpected_failure_fraction_minimum=payload.get(
                "failure_boundary_unexpected_failure_fraction_minimum", 0.50
            ),
            expected_failure_match_fraction_minimum=payload.get(
                "expected_failure_match_fraction_minimum", 0.90
            ),
            boundary_pressure_warning_fraction=payload.get(
                "boundary_pressure_warning_fraction", 0.50
            ),
            near_bound_tolerance_fraction=payload.get(
                "near_bound_tolerance_fraction", 0.05
            ),
            at_bound_tolerance_fraction=payload.get(
                "at_bound_tolerance_fraction", 1.0e-6
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_seed_count": self.minimum_seed_count,
            "robust_completion_drop_maximum": (
                self.robust_completion_drop_maximum
            ),
            "degraded_completion_drop_maximum": (
                self.degraded_completion_drop_maximum
            ),
            "robust_metric_pass_fraction_drop_maximum": (
                self.robust_metric_pass_fraction_drop_maximum
            ),
            "degraded_metric_pass_fraction_drop_maximum": (
                self.degraded_metric_pass_fraction_drop_maximum
            ),
            "failure_boundary_completion_fraction_maximum": (
                self.failure_boundary_completion_fraction_maximum
            ),
            "failure_boundary_unexpected_failure_fraction_minimum": (
                self.failure_boundary_unexpected_failure_fraction_minimum
            ),
            "expected_failure_match_fraction_minimum": (
                self.expected_failure_match_fraction_minimum
            ),
            "boundary_pressure_warning_fraction": (
                self.boundary_pressure_warning_fraction
            ),
            "near_bound_tolerance_fraction": self.near_bound_tolerance_fraction,
            "at_bound_tolerance_fraction": self.at_bound_tolerance_fraction,
        }


DEFAULT_SYNTHETIC_WAVELENGTH_ROBUSTNESS_CALIBRATION_THRESHOLDS = (
    SyntheticWavelengthRobustnessCalibrationThresholds()
)


@dataclass(frozen=True)
class SyntheticWavelengthRobustnessBoundaryRecord:
    """One paired, advisory empirical D2 boundary classification."""

    scenario_id: str
    classification: SyntheticWavelengthRobustnessBoundaryClass
    axis: str
    severity: str
    reference_scenario_id: str | None
    n_runs: int
    summary: Mapping[str, Any]
    comparison: Mapping[str, Any] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    schema_version: str = (
        SYNTHETIC_WAVELENGTH_ROBUSTNESS_CALIBRATION_SCHEMA_VERSION
    )
    advisory_only: bool = True
    automatic_model_selection_applied: bool = False
    extra_fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        scenario_id = str(self.scenario_id or "").strip()
        if not scenario_id:
            raise ValueError("scenario_id must be non-empty.")
        object.__setattr__(self, "scenario_id", scenario_id)
        object.__setattr__(
            self,
            "classification",
            _coerce_boundary_class(self.classification),
        )
        object.__setattr__(self, "axis", str(self.axis or "unclassified"))
        object.__setattr__(self, "severity", str(self.severity or "unclassified"))
        reference = (
            str(self.reference_scenario_id).strip()
            if self.reference_scenario_id is not None
            else None
        )
        object.__setattr__(self, "reference_scenario_id", reference or None)
        n_runs = int(self.n_runs)
        if n_runs < 0:
            raise ValueError("n_runs cannot be negative.")
        object.__setattr__(self, "n_runs", n_runs)
        object.__setattr__(self, "summary", _mapping_copy(self.summary))
        object.__setattr__(self, "comparison", _mapping_copy(self.comparison))
        object.__setattr__(
            self, "reasons", tuple(str(item) for item in self.reasons)
        )
        object.__setattr__(
            self, "warnings", tuple(str(item) for item in self.warnings)
        )
        object.__setattr__(self, "schema_version", str(self.schema_version))
        if self.advisory_only is not True:
            raise ValueError("D2 boundary records must remain advisory-only.")
        if self.automatic_model_selection_applied is not False:
            raise ValueError("D2 boundary records cannot select a model.")
        object.__setattr__(self, "extra_fields", _mapping_copy(self.extra_fields))

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> SyntheticWavelengthRobustnessBoundaryRecord:
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        known = {
            "schema_version",
            "scenario_id",
            "classification",
            "axis",
            "severity",
            "reference_scenario_id",
            "n_runs",
            "summary",
            "comparison",
            "reasons",
            "warnings",
            "advisory_only",
            "automatic_model_selection_applied",
            "extra_fields",
        }
        extras = _mapping_copy(payload.get("extra_fields"))
        for key, value in payload.items():
            if key not in known:
                extras[str(key)] = _json_safe(value)
        return cls(
            scenario_id=str(payload.get("scenario_id") or ""),
            classification=_coerce_boundary_class(payload.get("classification")),
            axis=str(payload.get("axis") or "unclassified"),
            severity=str(payload.get("severity") or "unclassified"),
            reference_scenario_id=payload.get("reference_scenario_id"),
            n_runs=int(payload.get("n_runs") or 0),
            summary=payload.get("summary") or {},
            comparison=payload.get("comparison") or {},
            reasons=tuple(payload.get("reasons") or ()),
            warnings=tuple(payload.get("warnings") or ()),
            schema_version=str(
                payload.get("schema_version")
                or SYNTHETIC_WAVELENGTH_ROBUSTNESS_CALIBRATION_SCHEMA_VERSION
            ),
            advisory_only=payload.get("advisory_only", True),
            automatic_model_selection_applied=payload.get(
                "automatic_model_selection_applied", False
            ),
            extra_fields=extras,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "classification": self.classification.value,
            "axis": self.axis,
            "severity": self.severity,
            "reference_scenario_id": self.reference_scenario_id,
            "n_runs": self.n_runs,
            "summary": _mapping_copy(self.summary),
            "comparison": _mapping_copy(self.comparison),
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "advisory_only": True,
            "automatic_model_selection_applied": False,
            "extra_fields": _mapping_copy(self.extra_fields),
        }


def _as_specification(
    value: SyntheticWavelengthRobustnessSpecification | Mapping[str, Any],
) -> SyntheticWavelengthRobustnessSpecification:
    if isinstance(value, SyntheticWavelengthRobustnessSpecification):
        return value
    if isinstance(value, Mapping):
        return SyntheticWavelengthRobustnessSpecification.from_mapping(value)
    raise TypeError("specifications must contain robustness specifications")


def _truth_pair_signature(
    specification: SyntheticWavelengthRobustnessSpecification,
) -> dict[str, Any]:
    configuration = specification.generator_configuration
    return {
        "generating_model": configuration.get("generating_model"),
        "mean_kind": configuration.get("mean_kind"),
        "covariance_kind": configuration.get("covariance_kind"),
        "period": _finite_float(configuration.get("period", 500.0)),
        "turning_point": bool(configuration.get("turning_point", False)),
        "mean_parameters": _mapping_copy(configuration.get("mean_parameters")),
        "covariance_parameters": _mapping_copy(
            configuration.get("covariance_parameters")
        ),
        "temporal_components": _json_safe(
            configuration.get("temporal_components")
        ),
        "strength": configuration.get("strength", "moderate"),
    }


def validate_synthetic_wavelength_robustness_reference_pairs(
    specifications: Sequence[
        SyntheticWavelengthRobustnessSpecification | Mapping[str, Any]
    ],
) -> dict[str, Any]:
    """Validate that every perturbation points to a compatible reference."""
    normalized = tuple(_as_specification(item) for item in specifications)
    by_id = {item.scenario_id: item for item in normalized}
    duplicate_count = len(normalized) - len(by_id)
    issues: list[dict[str, Any]] = []
    if duplicate_count:
        issues.append(
            {
                "code": "duplicate_scenario_ids",
                "count": duplicate_count,
            }
        )
    for specification in normalized:
        if specification.axis is SyntheticWavelengthRobustnessAxis.REFERENCE:
            if specification.reference_scenario_id is not None:
                issues.append(
                    {
                        "scenario_id": specification.scenario_id,
                        "code": "reference_points_to_reference",
                    }
                )
            continue
        reference_id = specification.reference_scenario_id
        reference = by_id.get(str(reference_id)) if reference_id else None
        if reference is None:
            issues.append(
                {
                    "scenario_id": specification.scenario_id,
                    "code": "missing_reference",
                    "reference_scenario_id": reference_id,
                }
            )
            continue
        if reference.axis is not SyntheticWavelengthRobustnessAxis.REFERENCE:
            issues.append(
                {
                    "scenario_id": specification.scenario_id,
                    "code": "reference_target_is_not_reference_axis",
                    "reference_scenario_id": reference_id,
                }
            )
        candidate_signature = _truth_pair_signature(specification)
        reference_signature = _truth_pair_signature(reference)
        allowed_differences = set()
        if specification.axis in {
            SyntheticWavelengthRobustnessAxis.WEAK_WAVELENGTH_DEPENDENCE,
            SyntheticWavelengthRobustnessAxis.STRONG_WAVELENGTH_DEPENDENCE,
        }:
            allowed_differences.add("strength")
        differences = {
            key: {
                "scenario": candidate_signature[key],
                "reference": reference_signature[key],
            }
            for key in candidate_signature
            if key not in allowed_differences
            and candidate_signature[key] != reference_signature[key]
        }
        if differences:
            issues.append(
                {
                    "scenario_id": specification.scenario_id,
                    "code": "truth_family_mismatch",
                    "reference_scenario_id": reference_id,
                    "differences": differences,
                }
            )
    return {
        "valid": not issues,
        "n_specifications": len(normalized),
        "n_references": sum(
            item.axis is SyntheticWavelengthRobustnessAxis.REFERENCE
            for item in normalized
        ),
        "issues": issues,
    }


def _numeric_summary(values: Sequence[Any]) -> dict[str, Any]:
    numeric = [
        item for item in (_finite_float(value) for value in values) if item is not None
    ]
    if not numeric:
        return {}
    array = np.asarray(numeric, dtype=float)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
        "p90": float(np.percentile(array, 90.0)),
    }


def _strict_metric(metric: WavelengthRecoveryMetric) -> bool:
    return metric.metadata.get("strict_recovery_eligible") is not False


def _metric_summaries(
    runs: Sequence[WavelengthValidationRun],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[WavelengthRecoveryMetric]] = {}
    for run in runs:
        for metric in run.metrics:
            if not _strict_metric(metric):
                continue
            grouped.setdefault(metric.name, []).append(metric)
    output: dict[str, dict[str, Any]] = {}
    for name, metrics in grouped.items():
        available = [metric for metric in metrics if metric.available]
        passed = [metric for metric in available if metric.passed is True]
        failed = [metric for metric in available if metric.passed is False]
        summary = {
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
        summary.update(_numeric_summary([metric.value for metric in available]))
        output[name] = summary
    return output


def _interval_boundary_record(
    *,
    value: float,
    lower: float,
    upper: float,
    parameter: str,
    dimension_name: str,
    source: str,
    thresholds: SyntheticWavelengthRobustnessCalibrationThresholds,
) -> dict[str, Any] | None:
    if not all(math.isfinite(item) for item in (value, lower, upper)):
        return None
    if upper <= lower:
        return None
    width = upper - lower
    distance_lower = value - lower
    distance_upper = upper - value
    normalized_lower = distance_lower / width
    normalized_upper = distance_upper / width
    nearest = min(normalized_lower, normalized_upper)
    side = "lower" if normalized_lower <= normalized_upper else "upper"
    outside = value < lower or value > upper
    at_bound = outside or nearest <= thresholds.at_bound_tolerance_fraction
    near_bound = outside or nearest <= thresholds.near_bound_tolerance_fraction
    return {
        "parameter": parameter,
        "dimension_name": dimension_name,
        "source": source,
        "value": value,
        "lower_bound": lower,
        "upper_bound": upper,
        "distance_to_lower": distance_lower,
        "distance_to_upper": distance_upper,
        "normalized_distance_to_lower": normalized_lower,
        "normalized_distance_to_upper": normalized_upper,
        "normalized_distance_to_nearest_bound": nearest,
        "bound_side": side,
        "outside_bounds": outside,
        "near_bound": near_bound,
        "at_bound": at_bound,
    }


def _separable_boundary_records(
    run: WavelengthValidationRun,
    thresholds: SyntheticWavelengthRobustnessCalibrationThresholds,
) -> list[dict[str, Any]]:
    fitted = _finite_float(run.diagnostics.get("fitted_wavelength_lengthscale"))
    if fitted is None:
        return []
    report = (run.parameter_workflow.get("report") or {})
    applied = report.get("applied") if isinstance(report, Mapping) else None
    if not isinstance(applied, Sequence):
        return []
    for item in applied:
        if not isinstance(item, Mapping):
            continue
        provenance = item.get("wavelength_estimate_provenance")
        if not isinstance(provenance, Mapping):
            continue
        effective = provenance.get("effective_constraint")
        if not isinstance(effective, Sequence) or len(effective) != 2:
            continue
        model_lower = _finite_float(effective[0])
        model_upper = _finite_float(effective[1])
        diagnostics = provenance.get("diagnostics") or {}
        raw_initial = _finite_float(
            diagnostics.get("raw_recommended_lengthscale_initial")
        )
        model_initial = _finite_float(
            diagnostics.get("model_recommended_lengthscale_initial")
        )
        factor = (
            raw_initial / model_initial
            if raw_initial is not None
            and model_initial is not None
            and raw_initial > 0.0
            and model_initial > 0.0
            else None
        )
        if (
            model_lower is not None
            and model_upper is not None
            and factor is not None
        ):
            lower = model_lower * factor
            upper = model_upper * factor
        else:
            raw_bounds = diagnostics.get("raw_recommended_lengthscale_bounds")
            if not isinstance(raw_bounds, Sequence) or len(raw_bounds) != 2:
                continue
            lower = _finite_float(raw_bounds[0])
            upper = _finite_float(raw_bounds[1])
        if lower is None or upper is None:
            continue
        record = _interval_boundary_record(
            value=fitted,
            lower=lower,
            upper=upper,
            parameter=str(item.get("parameter") or "wavelength_lengthscale"),
            dimension_name="wavelength_lengthscale",
            source="parameter_workflow_wavelength_estimate",
            thresholds=thresholds,
        )
        return [record] if record is not None else []
    return []


def _sm_boundary_records(
    run: WavelengthValidationRun,
    thresholds: SyntheticWavelengthRobustnessCalibrationThresholds,
) -> list[dict[str, Any]]:
    diagnostics = run.diagnostics.get("sm_ard_diagnostics")
    if not isinstance(diagnostics, Mapping) or not diagnostics.get("available"):
        return []
    output = []
    parameters = diagnostics.get("parameters") or {}
    if not isinstance(parameters, Mapping):
        return []
    for parameter_name, parameter_record in parameters.items():
        if not isinstance(parameter_record, Mapping):
            continue
        components = parameter_record.get("component_diagnostics") or ()
        for component in components:
            if not isinstance(component, Mapping):
                continue
            value = _finite_float(component.get("raw_input_coordinate_value"))
            lower = _finite_float(
                component.get("raw_input_coordinate_lower_bound")
            )
            upper = _finite_float(
                component.get("raw_input_coordinate_upper_bound")
            )
            if value is None or lower is None or upper is None:
                value = _finite_float(component.get("model_coordinate_value"))
                lower = _finite_float(
                    component.get("model_coordinate_lower_bound")
                )
                upper = _finite_float(
                    component.get("model_coordinate_upper_bound")
                )
            if value is None or lower is None or upper is None:
                continue
            record = _interval_boundary_record(
                value=value,
                lower=lower,
                upper=upper,
                parameter=str(parameter_name),
                dimension_name=str(
                    component.get("dimension_name") or "unclassified"
                ),
                source="spectral_mixture_ard_diagnostics",
                thresholds=thresholds,
            )
            if record is not None:
                record["component_index"] = int(
                    component.get("component_index") or 0
                )
                output.append(record)
    return output


def _run_boundary_summary(
    run: WavelengthValidationRun,
    thresholds: SyntheticWavelengthRobustnessCalibrationThresholds,
) -> dict[str, Any]:
    records = _sm_boundary_records(run, thresholds)
    if not records:
        records = _separable_boundary_records(run, thresholds)
    wavelength = [
        item
        for item in records
        if item.get("dimension_name")
        in {"wavelength_frequency", "wavelength_lengthscale"}
    ]
    temporal = [
        item
        for item in records
        if item.get("dimension_name") == "temporal_frequency"
    ]
    wavelength_distances = [
        item["normalized_distance_to_nearest_bound"] for item in wavelength
    ]
    temporal_distances = [
        item["normalized_distance_to_nearest_bound"] for item in temporal
    ]
    return {
        "available": bool(records),
        "records": records,
        "n_records": len(records),
        "wavelength_near_bound": any(item["near_bound"] for item in wavelength),
        "wavelength_at_bound": any(item["at_bound"] for item in wavelength),
        "temporal_near_bound": any(item["near_bound"] for item in temporal),
        "temporal_at_bound": any(item["at_bound"] for item in temporal),
        "minimum_wavelength_normalized_distance": (
            min(wavelength_distances) if wavelength_distances else None
        ),
        "minimum_temporal_normalized_distance": (
            min(temporal_distances) if temporal_distances else None
        ),
    }


def _scenario_summary(
    runs: Sequence[WavelengthValidationRun],
    thresholds: SyntheticWavelengthRobustnessCalibrationThresholds,
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
    boundary_summaries = [
        _run_boundary_summary(run, thresholds) for run in runs
    ]
    available_boundaries = [
        item for item in boundary_summaries if item["available"]
    ]
    wavelength_distances = [
        item["minimum_wavelength_normalized_distance"]
        for item in available_boundaries
        if item["minimum_wavelength_normalized_distance"] is not None
    ]
    temporal_distances = [
        item["minimum_temporal_normalized_distance"]
        for item in available_boundaries
        if item["minimum_temporal_normalized_distance"] is not None
    ]
    first = runs[0]
    return {
        "scenario_id": first.scenario_id,
        "model": first.model,
        "axis": str(first.extra_fields.get("robustness_axis") or "unclassified"),
        "severity": str(
            first.extra_fields.get("robustness_severity") or "unclassified"
        ),
        "reference_scenario_id": first.extra_fields.get(
            "reference_scenario_id"
        ),
        "n_runs": len(runs),
        "n_completed": n_completed,
        "n_failed": len(runs) - n_completed,
        "completion_fraction": n_completed / len(runs) if runs else None,
        "n_expected_failure_contracts": n_expected,
        "n_expected_failures_matched": n_expected_matched,
        "n_expected_failures_mismatched": n_expected_mismatched,
        "expected_failure_match_fraction": (
            n_expected_matched / n_expected if n_expected else None
        ),
        "n_unexpected_failures": n_unexpected_failures,
        "unexpected_failure_fraction": (
            n_unexpected_failures / len(runs) if runs else None
        ),
        "metric_summaries": _metric_summaries(runs),
        "boundary_summary": {
            "n_available": len(available_boundaries),
            "n_wavelength_near_bound": sum(
                bool(item["wavelength_near_bound"])
                for item in available_boundaries
            ),
            "wavelength_near_bound_fraction": (
                sum(
                    bool(item["wavelength_near_bound"])
                    for item in available_boundaries
                )
                / len(available_boundaries)
                if available_boundaries
                else None
            ),
            "n_wavelength_at_bound": sum(
                bool(item["wavelength_at_bound"])
                for item in available_boundaries
            ),
            "wavelength_at_bound_fraction": (
                sum(
                    bool(item["wavelength_at_bound"])
                    for item in available_boundaries
                )
                / len(available_boundaries)
                if available_boundaries
                else None
            ),
            "n_temporal_near_bound": sum(
                bool(item["temporal_near_bound"])
                for item in available_boundaries
            ),
            "temporal_near_bound_fraction": (
                sum(
                    bool(item["temporal_near_bound"])
                    for item in available_boundaries
                )
                / len(available_boundaries)
                if available_boundaries
                else None
            ),
            "minimum_wavelength_normalized_distance": (
                min(wavelength_distances) if wavelength_distances else None
            ),
            "median_wavelength_normalized_distance": (
                float(np.median(wavelength_distances))
                if wavelength_distances
                else None
            ),
            "minimum_temporal_normalized_distance": (
                min(temporal_distances) if temporal_distances else None
            ),
        },
        "run_ids": [run.run_id for run in runs],
    }


def summarize_synthetic_wavelength_robustness_population(
    runs: Sequence[WavelengthValidationRun | Mapping[str, Any]],
    *,
    thresholds: (
        SyntheticWavelengthRobustnessCalibrationThresholds
        | Mapping[str, Any]
        | None
    ) = None,
) -> dict[str, dict[str, Any]]:
    """Summarize repeated D2 runs by scenario without assigning a winner."""
    resolved_thresholds = (
        thresholds
        if isinstance(thresholds, SyntheticWavelengthRobustnessCalibrationThresholds)
        else SyntheticWavelengthRobustnessCalibrationThresholds.from_mapping(
            thresholds or {}
        )
    )
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
            raise ValueError("D2 population summaries require D2 runs.")
        normalized.append(normalized_run)
    grouped: dict[str, list[WavelengthValidationRun]] = {}
    for run in normalized:
        grouped.setdefault(run.scenario_id, []).append(run)
    return {
        scenario_id: _scenario_summary(group, resolved_thresholds)
        for scenario_id, group in grouped.items()
    }


def _scenario_comparison(
    summary: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    completion = _finite_float(summary.get("completion_fraction"))
    reference_completion = _finite_float(reference.get("completion_fraction"))
    metric_drops = {}
    for name in _CORE_RECOVERY_METRICS:
        candidate_metric = (summary.get("metric_summaries") or {}).get(name) or {}
        reference_metric = (reference.get("metric_summaries") or {}).get(name) or {}
        candidate_pass = _finite_float(candidate_metric.get("pass_fraction"))
        reference_pass = _finite_float(reference_metric.get("pass_fraction"))
        if candidate_pass is None or reference_pass is None:
            continue
        metric_drops[name] = reference_pass - candidate_pass
    boundary = summary.get("boundary_summary") or {}
    reference_boundary = reference.get("boundary_summary") or {}
    candidate_pressure = _finite_float(
        boundary.get("wavelength_near_bound_fraction")
    )
    reference_pressure = _finite_float(
        reference_boundary.get("wavelength_near_bound_fraction")
    )
    return {
        "reference_scenario_id": reference.get("scenario_id"),
        "completion_fraction_drop": (
            reference_completion - completion
            if completion is not None and reference_completion is not None
            else None
        ),
        "metric_pass_fraction_drops": metric_drops,
        "worst_metric_pass_fraction_drop": (
            max(metric_drops.values()) if metric_drops else None
        ),
        "wavelength_boundary_pressure_fraction_increase": (
            candidate_pressure - reference_pressure
            if candidate_pressure is not None and reference_pressure is not None
            else None
        ),
    }


def _classify_summary(
    summary: Mapping[str, Any],
    reference: Mapping[str, Any] | None,
    thresholds: SyntheticWavelengthRobustnessCalibrationThresholds,
) -> tuple[
    SyntheticWavelengthRobustnessBoundaryClass,
    dict[str, Any],
    tuple[str, ...],
    tuple[str, ...],
]:
    axis = str(summary.get("axis") or "unclassified")
    reasons = []
    warnings = []
    if axis == SyntheticWavelengthRobustnessAxis.REFERENCE.value:
        return (
            SyntheticWavelengthRobustnessBoundaryClass.REFERENCE,
            {},
            ("Truth-matched nominal reference population.",),
            (),
        )
    n_runs = int(summary.get("n_runs") or 0)
    if n_runs < thresholds.minimum_seed_count:
        return (
            SyntheticWavelengthRobustnessBoundaryClass.INCONCLUSIVE,
            {},
            (
                "Population does not meet the configured minimum seed count.",
            ),
            (),
        )
    n_expected = int(summary.get("n_expected_failure_contracts") or 0)
    if n_expected:
        match_fraction = _finite_float(
            summary.get("expected_failure_match_fraction")
        )
        mismatched = int(summary.get("n_expected_failures_mismatched") or 0)
        unexpected = int(summary.get("n_unexpected_failures") or 0)
        if (
            match_fraction is not None
            and match_fraction
            >= thresholds.expected_failure_match_fraction_minimum
            and mismatched == 0
            and unexpected == 0
        ):
            return (
                SyntheticWavelengthRobustnessBoundaryClass.EXPECTED_FAILURE_BOUNDARY,
                {},
                ("The classified expected-failure contract is reproducible.",),
                (),
            )
        return (
            SyntheticWavelengthRobustnessBoundaryClass.INCONCLUSIVE,
            {},
            ("Expected-failure outcomes are not sufficiently reproducible.",),
            (),
        )
    if reference is None:
        return (
            SyntheticWavelengthRobustnessBoundaryClass.INCONCLUSIVE,
            {},
            ("No truth-matched reference population is available.",),
            (),
        )
    reference_runs = int(reference.get("n_runs") or 0)
    reference_completion = _finite_float(reference.get("completion_fraction"))
    reference_unexpected = _finite_float(
        reference.get("unexpected_failure_fraction")
    )
    if (
        reference_runs < thresholds.minimum_seed_count
        or reference_completion is None
        or reference_completion
        <= thresholds.failure_boundary_completion_fraction_maximum
        or reference_unexpected is None
        or reference_unexpected
        >= thresholds.failure_boundary_unexpected_failure_fraction_minimum
    ):
        return (
            SyntheticWavelengthRobustnessBoundaryClass.INCONCLUSIVE,
            {},
            ("The truth-matched reference population is not sufficiently stable.",),
            (),
        )
    comparison = _scenario_comparison(summary, reference)
    completion = _finite_float(summary.get("completion_fraction"))
    unexpected_fraction = _finite_float(
        summary.get("unexpected_failure_fraction")
    )
    if (
        completion is not None
        and completion
        <= thresholds.failure_boundary_completion_fraction_maximum
    ) or (
        unexpected_fraction is not None
        and unexpected_fraction
        >= thresholds.failure_boundary_unexpected_failure_fraction_minimum
    ):
        reasons.append(
            "Technical completion or unexpected-failure frequency reaches the "
            "configured empirical failure boundary."
        )
        classification = SyntheticWavelengthRobustnessBoundaryClass.FAILURE_BOUNDARY
    else:
        completion_drop = _finite_float(
            comparison.get("completion_fraction_drop")
        )
        metric_drop = _finite_float(
            comparison.get("worst_metric_pass_fraction_drop")
        )
        if completion_drop is None or metric_drop is None:
            classification = SyntheticWavelengthRobustnessBoundaryClass.INCONCLUSIVE
            reasons.append(
                "Paired completion and recovery-retention metrics are incomplete."
            )
        elif (
            completion_drop <= thresholds.robust_completion_drop_maximum
            and metric_drop
            <= thresholds.robust_metric_pass_fraction_drop_maximum
        ):
            classification = SyntheticWavelengthRobustnessBoundaryClass.ROBUST
            reasons.append(
                "Completion and per-run recovery retention remain within the "
                "configured robust-degradation envelope."
            )
        elif (
            completion_drop <= thresholds.degraded_completion_drop_maximum
            and metric_drop
            <= thresholds.degraded_metric_pass_fraction_drop_maximum
        ):
            classification = SyntheticWavelengthRobustnessBoundaryClass.DEGRADED
            reasons.append(
                "The perturbation is measurably degraded but remains short of "
                "the configured failure boundary."
            )
        else:
            classification = SyntheticWavelengthRobustnessBoundaryClass.FAILURE_BOUNDARY
            reasons.append(
                "Paired completion or recovery retention exceeds the configured "
                "degraded envelope."
            )
    boundary = summary.get("boundary_summary") or {}
    pressure = _finite_float(boundary.get("wavelength_near_bound_fraction"))
    at_bound = _finite_float(boundary.get("wavelength_at_bound_fraction"))
    if (
        pressure is not None
        and pressure >= thresholds.boundary_pressure_warning_fraction
    ):
        warnings.append(
            "Wavelength-parameter boundary pressure is frequent; this is "
            "reported separately from recovery classification."
        )
    if at_bound is not None and at_bound > 0.0:
        warnings.append(
            "At least one fitted wavelength parameter reached its effective bound."
        )
    return classification, comparison, tuple(reasons), tuple(warnings)


def calibrate_synthetic_wavelength_robustness_runs(
    runs: Sequence[WavelengthValidationRun | Mapping[str, Any]],
    *,
    thresholds: (
        SyntheticWavelengthRobustnessCalibrationThresholds
        | Mapping[str, Any]
        | None
    ) = None,
) -> tuple[
    dict[str, dict[str, Any]],
    tuple[SyntheticWavelengthRobustnessBoundaryRecord, ...],
]:
    """Assign advisory D2 boundary classes to repeated scenario runs."""
    resolved_thresholds = (
        thresholds
        if isinstance(thresholds, SyntheticWavelengthRobustnessCalibrationThresholds)
        else SyntheticWavelengthRobustnessCalibrationThresholds.from_mapping(
            thresholds or {}
        )
    )
    summaries = summarize_synthetic_wavelength_robustness_population(
        runs, thresholds=resolved_thresholds
    )
    records = []
    for scenario_id, summary in summaries.items():
        reference_id = summary.get("reference_scenario_id")
        reference = summaries.get(str(reference_id)) if reference_id else None
        classification, comparison, reasons, warnings = _classify_summary(
            summary, reference, resolved_thresholds
        )
        records.append(
            SyntheticWavelengthRobustnessBoundaryRecord(
                scenario_id=scenario_id,
                classification=classification,
                axis=str(summary.get("axis") or "unclassified"),
                severity=str(summary.get("severity") or "unclassified"),
                reference_scenario_id=(
                    str(reference_id) if reference_id is not None else None
                ),
                n_runs=int(summary.get("n_runs") or 0),
                summary=summary,
                comparison=comparison,
                reasons=reasons,
                warnings=warnings,
            )
        )
    return summaries, tuple(records)


@dataclass(frozen=True)
class SyntheticWavelengthRobustnessCalibrationReport:
    """Typed multi-seed D2 population report and boundary classifications."""

    report_id: str
    base_seeds: tuple[int, ...]
    specifications: tuple[SyntheticWavelengthRobustnessSpecification, ...]
    runs: tuple[WavelengthValidationRun, ...]
    aggregate: WavelengthValidationAggregate
    scenario_summaries: Mapping[str, Any]
    boundary_records: tuple[SyntheticWavelengthRobustnessBoundaryRecord, ...]
    thresholds: SyntheticWavelengthRobustnessCalibrationThresholds = field(
        default_factory=SyntheticWavelengthRobustnessCalibrationThresholds
    )
    schema_version: str = (
        SYNTHETIC_WAVELENGTH_ROBUSTNESS_CALIBRATION_SCHEMA_VERSION
    )
    notes: tuple[str, ...] = ()
    advisory_only: bool = True
    automatic_model_selection_applied: bool = False
    extra_fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        report_id = str(self.report_id or "").strip()
        if not report_id:
            raise ValueError("report_id must be non-empty.")
        object.__setattr__(self, "report_id", report_id)
        seeds = tuple(int(seed) for seed in self.base_seeds)
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError("base_seeds must be non-empty and unique.")
        object.__setattr__(self, "base_seeds", seeds)
        object.__setattr__(
            self,
            "specifications",
            tuple(_as_specification(item) for item in self.specifications),
        )
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
            raise ValueError("Calibration reports require a D2 aggregate.")
        if self.aggregate.n_runs != len(self.runs):
            raise ValueError("aggregate.n_runs must match the number of runs.")
        object.__setattr__(
            self, "scenario_summaries", _mapping_copy(self.scenario_summaries)
        )
        object.__setattr__(
            self,
            "boundary_records",
            tuple(
                item
                if isinstance(item, SyntheticWavelengthRobustnessBoundaryRecord)
                else SyntheticWavelengthRobustnessBoundaryRecord.from_mapping(item)
                for item in self.boundary_records
            ),
        )
        if not isinstance(
            self.thresholds, SyntheticWavelengthRobustnessCalibrationThresholds
        ):
            object.__setattr__(
                self,
                "thresholds",
                SyntheticWavelengthRobustnessCalibrationThresholds.from_mapping(
                    self.thresholds
                ),
            )
        object.__setattr__(self, "schema_version", str(self.schema_version))
        object.__setattr__(
            self, "notes", tuple(str(item) for item in self.notes)
        )
        if self.advisory_only is not True:
            raise ValueError("Calibration reports must remain advisory-only.")
        if self.automatic_model_selection_applied is not False:
            raise ValueError("Calibration reports cannot select a model.")
        object.__setattr__(self, "extra_fields", _mapping_copy(self.extra_fields))

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> SyntheticWavelengthRobustnessCalibrationReport:
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        known = {
            "schema_version",
            "report_id",
            "base_seeds",
            "specifications",
            "runs",
            "aggregate",
            "scenario_summaries",
            "boundary_records",
            "thresholds",
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
            base_seeds=tuple(payload.get("base_seeds") or ()),
            specifications=tuple(
                SyntheticWavelengthRobustnessSpecification.from_mapping(item)
                if isinstance(item, Mapping)
                else item
                for item in payload.get("specifications") or ()
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
            scenario_summaries=payload.get("scenario_summaries") or {},
            boundary_records=tuple(
                SyntheticWavelengthRobustnessBoundaryRecord.from_mapping(item)
                if isinstance(item, Mapping)
                else item
                for item in payload.get("boundary_records") or ()
            ),
            thresholds=SyntheticWavelengthRobustnessCalibrationThresholds.from_mapping(
                payload.get("thresholds") or {}
            ),
            schema_version=str(
                payload.get("schema_version")
                or SYNTHETIC_WAVELENGTH_ROBUSTNESS_CALIBRATION_SCHEMA_VERSION
            ),
            notes=tuple(payload.get("notes") or ()),
            advisory_only=payload.get("advisory_only", True),
            automatic_model_selection_applied=payload.get(
                "automatic_model_selection_applied", False
            ),
            extra_fields=extras,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "base_seeds": list(self.base_seeds),
            "specifications": [item.to_dict() for item in self.specifications],
            "runs": [run.to_dict() for run in self.runs],
            "aggregate": self.aggregate.to_dict(),
            "scenario_summaries": _mapping_copy(self.scenario_summaries),
            "boundary_records": [
                item.to_dict() for item in self.boundary_records
            ],
            "thresholds": self.thresholds.to_dict(),
            "notes": list(self.notes),
            "advisory_only": True,
            "automatic_model_selection_applied": False,
            "extra_fields": _mapping_copy(self.extra_fields),
        }


def _safe_filename_token(value: Any) -> str:
    text = str(value or "").strip()
    return "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "-"
        for character in text
    )


def _run_output_path(
    output_dir: Path,
    *,
    base_seed: int,
    scenario_id: str,
    model: str,
) -> Path:
    filename = (
        f"base-seed-{base_seed:04d}__"
        f"{_safe_filename_token(scenario_id)}__"
        f"{_safe_filename_token(model)}.json"
    )
    return output_dir / "runs" / filename


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_persisted_run(
    path: Path,
    *,
    base_seed: int,
    scenario_id: str,
    model: str,
) -> WavelengthValidationRun:
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = payload.get("record") or {}
    if int(record.get("base_seed")) != int(base_seed):
        raise ValueError(f"Persisted run has the wrong base seed: {path}")
    if str(record.get("scenario_id")) != str(scenario_id):
        raise ValueError(f"Persisted run has the wrong scenario id: {path}")
    if str(record.get("model")) != str(model):
        raise ValueError(f"Persisted run has the wrong model: {path}")
    return WavelengthValidationRun.from_mapping(payload.get("run") or {})


@contextmanager
def _temporary_optimizer_seed(seed: int | None):
    if seed is None:
        yield
        return
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_module = None
    torch_state = None
    cuda_states = None
    try:
        random.seed(int(seed))
        np.random.seed(int(seed) % (2**32 - 1))
        try:
            import torch

            torch_module = torch
            torch_state = torch.random.get_rng_state()
            if torch.cuda.is_available():
                cuda_states = torch.cuda.get_rng_state_all()
            torch.manual_seed(int(seed))
        except ImportError:
            torch_module = None
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch_module is not None and torch_state is not None:
            torch_module.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch_module.cuda.set_rng_state_all(cuda_states)


def run_synthetic_wavelength_robustness_population(
    *,
    base_seeds: Sequence[int],
    specifications: Sequence[
        SyntheticWavelengthRobustnessSpecification | Mapping[str, Any]
    ]
    | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
    per_model_fit_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    recovery_thresholds: (
        SyntheticWavelengthRecoveryThresholds | Mapping[str, Any] | None
    ) = None,
    calibration_thresholds: (
        SyntheticWavelengthRobustnessCalibrationThresholds
        | Mapping[str, Any]
        | None
    ) = None,
    lightcurve_factory: _LightcurveFactory | None = None,
    fit_runner: _SyntheticFitRunner | None = None,
    stop_on_error: bool = False,
    optimizer_seed_base: int | None = 61000,
    output_dir: str | Path | None = None,
    resume: bool = True,
    write_report: bool = True,
    report_id: str = "d2-synthetic-wavelength-robustness-calibration",
) -> SyntheticWavelengthRobustnessCalibrationReport:
    """Execute a matched-model, multi-seed D2 calibration population."""
    seeds = tuple(int(seed) for seed in base_seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("base_seeds must be non-empty and unique.")
    if any(seed < 0 for seed in seeds):
        raise ValueError("base_seeds cannot contain negative values.")
    resolved_output_dir = Path(output_dir) if output_dir is not None else None
    if write_report and resolved_output_dir is None:
        write_report = False
    normalized_specifications = tuple(
        _as_specification(item)
        for item in (
            specifications
            if specifications is not None
            else canonical_synthetic_wavelength_robustness_specifications()
        )
    )
    reference_validation = validate_synthetic_wavelength_robustness_reference_pairs(
        normalized_specifications
    )
    if not reference_validation["valid"]:
        raise ValueError(
            "D2 robustness reference pairs are invalid: "
            + str(reference_validation["issues"])
        )
    resolved_calibration_thresholds = (
        calibration_thresholds
        if isinstance(
            calibration_thresholds,
            SyntheticWavelengthRobustnessCalibrationThresholds,
        )
        else SyntheticWavelengthRobustnessCalibrationThresholds.from_mapping(
            calibration_thresholds or {}
        )
    )
    per_model_fit_kwargs = dict(per_model_fit_kwargs or {})
    runs = []
    n_resumed_runs = 0
    n_executed_runs = 0
    for base_seed in seeds:
        optimizer_seed = (
            int(optimizer_seed_base) + int(base_seed)
            if optimizer_seed_base is not None
            else None
        )
        for specification in normalized_specifications:
            case = make_synthetic_wavelength_robustness_case(
                specification,
                seed=base_seed,
            )
            model = case.scenario.truth.generating_model
            run_path = (
                _run_output_path(
                    resolved_output_dir,
                    base_seed=base_seed,
                    scenario_id=specification.scenario_id,
                    model=model,
                )
                if resolved_output_dir is not None
                else None
            )
            if resume and run_path is not None and run_path.is_file():
                run = _load_persisted_run(
                    run_path,
                    base_seed=base_seed,
                    scenario_id=specification.scenario_id,
                    model=model,
                )
                n_resumed_runs += 1
            else:
                configuration = dict(fit_kwargs or {})
                configuration.update(dict(per_model_fit_kwargs.get(model) or {}))
                with _temporary_optimizer_seed(optimizer_seed):
                    run = run_synthetic_wavelength_robustness(
                        case,
                        model,
                        fit_kwargs=configuration,
                        thresholds=recovery_thresholds,
                        lightcurve_factory=lightcurve_factory,
                        fit_runner=fit_runner,
                        stop_on_error=stop_on_error,
                    )
                run = replace(
                    run,
                    extra_fields={
                        **dict(run.extra_fields),
                        "base_seed": base_seed,
                        "optimizer_seed": optimizer_seed,
                        "calibration_schema_version": (
                            SYNTHETIC_WAVELENGTH_ROBUSTNESS_CALIBRATION_SCHEMA_VERSION
                        ),
                    },
                )
                n_executed_runs += 1
                if run_path is not None:
                    _atomic_json_write(
                        run_path,
                        {
                            "schema_version": (
                                SYNTHETIC_WAVELENGTH_ROBUSTNESS_CALIBRATION_SCHEMA_VERSION
                            ),
                            "record": {
                                "base_seed": base_seed,
                                "optimizer_seed": optimizer_seed,
                                "scenario_id": specification.scenario_id,
                                "model": model,
                                "reference_scenario_id": (
                                    specification.reference_scenario_id
                                ),
                            },
                            "run": run.to_dict(),
                        },
                    )
            runs.append(run)
    aggregate = aggregate_synthetic_wavelength_robustness_runs(
        runs, aggregate_id=report_id
    )
    summaries, boundary_records = calibrate_synthetic_wavelength_robustness_runs(
        runs,
        thresholds=resolved_calibration_thresholds,
    )
    nonreference_records = [
        item
        for item in boundary_records
        if item.classification
        is not SyntheticWavelengthRobustnessBoundaryClass.REFERENCE
    ]
    calibration_complete = bool(nonreference_records) and all(
        item.classification
        is not SyntheticWavelengthRobustnessBoundaryClass.INCONCLUSIVE
        for item in nonreference_records
    )
    classification_counts: dict[str, int] = {}
    for item in boundary_records:
        name = item.classification.value
        classification_counts[name] = classification_counts.get(name, 0) + 1
    aggregate = replace(
        aggregate,
        extra_fields={
            **dict(aggregate.extra_fields),
            "calibration_schema_version": (
                SYNTHETIC_WAVELENGTH_ROBUSTNESS_CALIBRATION_SCHEMA_VERSION
            ),
            "reference_pair_validation": reference_validation,
            "scenario_population_summaries": summaries,
            "boundary_records": [item.to_dict() for item in boundary_records],
            "d2_boundary_summary": {
                "empirical_boundaries_calibrated": calibration_complete,
                "n_base_seeds": len(seeds),
                "classification_counts": classification_counts,
                "minimum_seed_count": (
                    resolved_calibration_thresholds.minimum_seed_count
                ),
                "d1_gates_reused_as_d2_gates": False,
                "d1_per_run_metrics_used_descriptively": True,
                "automatic_model_selection_applied": False,
            },
        },
    )
    report = SyntheticWavelengthRobustnessCalibrationReport(
        report_id=report_id,
        base_seeds=seeds,
        specifications=normalized_specifications,
        runs=tuple(runs),
        aggregate=aggregate,
        scenario_summaries=summaries,
        boundary_records=boundary_records,
        thresholds=resolved_calibration_thresholds,
        notes=(
            "D2 classes compare each perturbation with a truth-matched reference "
            "population and remain advisory.",
            "D1 aggregate gates are not reused as D2 boundary gates.",
            "Boundary pressure is reported separately from recovery degradation.",
        ),
        extra_fields={
            "robustness_schema_version": (
                SYNTHETIC_WAVELENGTH_ROBUSTNESS_SCHEMA_VERSION
            ),
            "output_dir": (
                str(resolved_output_dir)
                if resolved_output_dir is not None
                else None
            ),
            "n_executed_runs": n_executed_runs,
            "n_resumed_runs": n_resumed_runs,
            "resume_enabled": bool(resume),
        },
    )
    if write_report and resolved_output_dir is not None:
        _atomic_json_write(resolved_output_dir / "report.json", report.to_dict())
    return report
