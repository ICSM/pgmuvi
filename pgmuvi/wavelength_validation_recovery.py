"""Synthetic recovery metrics and runners for wavelength-model validation.

This module implements the D1 execution layer for the truth-preserving cases in
:mod:`pgmuvi.wavelength_validation_synthetic`.  It measures recovery against
known generating truth, classifies fit and diagnostic failures, and aggregates
runs without selecting a model.

The default runner can execute real :class:`~pgmuvi.lightcurve.Lightcurve`
fits.  ``lightcurve_factory`` and ``fit_runner`` hooks keep the workflow
unit-testable and allow external validation drivers to control resources.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

import numpy as np

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
)
from .wavelength_validation import (
    RecoveryMetricDirection,
    WavelengthRecoveryMetric,
    WavelengthValidationAggregate,
    WavelengthValidationPhase,
    WavelengthValidationProvenance,
    WavelengthValidationRun,
)
from .wavelength_validation_synthetic import (
    DEFAULT_VALIDATION_CANDIDATES,
    SyntheticWavelengthValidationCase,
)

SYNTHETIC_WAVELENGTH_RECOVERY_SCHEMA_VERSION = (
    "pgmuvi-synthetic-wavelength-recovery-v1"
)

SUPPORTED_SYNTHETIC_RECOVERY_MODELS = tuple(DEFAULT_VALIDATION_CANDIDATES)

__all__ = [
    "DEFAULT_SYNTHETIC_RECOVERY_THRESHOLDS",
    "SUPPORTED_SYNTHETIC_RECOVERY_MODELS",
    "SYNTHETIC_WAVELENGTH_RECOVERY_SCHEMA_VERSION",
    "SyntheticWavelengthRecoveryReport",
    "SyntheticWavelengthRecoveryThresholds",
    "aggregate_synthetic_wavelength_recovery_runs",
    "build_synthetic_wavelength_recovery_metrics",
    "evaluate_synthetic_wavelength_recovery",
    "run_synthetic_wavelength_recovery",
    "run_synthetic_wavelength_recovery_matrix",
]


@dataclass(frozen=True)
class SyntheticWavelengthRecoveryThresholds:
    """Frozen D1 per-run recovery thresholds.

    Aggregate percentile gates are intentionally evaluated outside this record.
    These thresholds only decide whether an individual available metric passes.
    """

    coordinate_round_trip_max_abs_error: float = 1.0e-10
    period_relative_error: float = 0.15
    wavelength_lengthscale_factor_error: float = 4.0
    strong_mean_normalized_rmse: float = 0.15
    weak_mean_normalized_rmse: float = 0.25

    def __post_init__(self) -> None:
        for name in (
            "coordinate_round_trip_max_abs_error",
            "period_relative_error",
            "wavelength_lengthscale_factor_error",
            "strong_mean_normalized_rmse",
            "weak_mean_normalized_rmse",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        if self.wavelength_lengthscale_factor_error < 1.0:
            raise ValueError(
                "wavelength_lengthscale_factor_error must be at least one."
            )

    def mean_normalized_rmse(self, strength: str | None) -> float:
        """Return the mean-law threshold for a scenario strength label."""
        if str(strength or "").lower() in {"moderate", "strong"}:
            return self.strong_mean_normalized_rmse
        return self.weak_mean_normalized_rmse

    def to_dict(self) -> dict[str, float]:
        """Return a stable JSON-safe mapping."""
        return {
            "coordinate_round_trip_max_abs_error": (
                self.coordinate_round_trip_max_abs_error
            ),
            "period_relative_error": self.period_relative_error,
            "wavelength_lengthscale_factor_error": (
                self.wavelength_lengthscale_factor_error
            ),
            "strong_mean_normalized_rmse": self.strong_mean_normalized_rmse,
            "weak_mean_normalized_rmse": self.weak_mean_normalized_rmse,
        }


DEFAULT_SYNTHETIC_RECOVERY_THRESHOLDS = (
    SyntheticWavelengthRecoveryThresholds()
)


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
    detach = getattr(value, "detach", None)
    if callable(detach):
        try:
            value = detach()
            cpu = getattr(value, "cpu", None)
            if callable(cpu):
                value = cpu()
        except (TypeError, ValueError, RuntimeError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except (TypeError, ValueError, RuntimeError):
            pass
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError, RuntimeError):
            pass
    return str(value)


def _finite_float(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _finite_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size == 0 or not np.all(np.isfinite(array)):
        return None
    return array


def _as_case(value: Any) -> SyntheticWavelengthValidationCase:
    if isinstance(value, SyntheticWavelengthValidationCase):
        return value
    if isinstance(value, Mapping):
        return SyntheticWavelengthValidationCase.from_mapping(value)
    raise TypeError("case must be a SyntheticWavelengthValidationCase or mapping")


def _as_thresholds(value: Any) -> SyntheticWavelengthRecoveryThresholds:
    if value is None:
        return DEFAULT_SYNTHETIC_RECOVERY_THRESHOLDS
    if isinstance(value, SyntheticWavelengthRecoveryThresholds):
        return value
    if isinstance(value, Mapping):
        return SyntheticWavelengthRecoveryThresholds(**dict(value))
    raise TypeError(
        "thresholds must be SyntheticWavelengthRecoveryThresholds, mapping, or None"
    )


def _metric(
    *,
    name: str,
    value: Any,
    truth_value: Any,
    available: bool,
    passed: bool | None,
    direction: RecoveryMetricDirection,
    threshold: Mapping[str, Any] | None = None,
    units: str | None = None,
    scope: str | None = None,
    model: str | None = None,
    parameter: str | None = None,
    ard_dimension: str | None = None,
    component_index: int | None = None,
    summary: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    limitations: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> WavelengthRecoveryMetric:
    return WavelengthRecoveryMetric(
        name=name,
        value=value,
        truth_value=truth_value,
        available=available,
        passed=passed,
        direction=direction,
        threshold=dict(threshold or {}),
        units=units,
        scope=scope,
        model=model,
        parameter=parameter,
        ard_dimension=ard_dimension,
        component_index=component_index,
        summary=summary,
        provenance=dict(provenance or {}),
        limitations=tuple(str(item) for item in limitations),
        metadata=dict(metadata or {}),
    )


def _coordinate_round_trip_error(
    case: SyntheticWavelengthValidationCase,
) -> float | None:
    truth = case.scenario.truth
    if truth is None:
        return None
    physical = _finite_array(truth.physical_wavelengths)
    model = _finite_array(
        truth.noiseless_summary.get("model_wavelength_by_band")
    )
    if physical is None or model is None or physical.shape != model.shape:
        return None
    transform = truth.coordinate_transforms
    kind = str(transform.get("kind") or "").lower()
    wavelength_record = transform.get("wavelength") or {}
    if kind == "identity":
        reconstructed = model
    else:
        origin = _finite_float(wavelength_record.get("origin"))
        scale = _finite_float(wavelength_record.get("scale"))
        if origin is None or scale is None or scale <= 0.0:
            return None
        reconstructed = origin + scale * model
    return float(np.max(np.abs(reconstructed - physical)))


def _relative_error(value: float | None, truth: float | None) -> float | None:
    if value is None or truth is None or truth == 0.0:
        return None
    return abs(value - truth) / abs(truth)


def _factor_error(value: float | None, truth: float | None) -> float | None:
    if value is None or truth is None or value <= 0.0 or truth <= 0.0:
        return None
    return max(value / truth, truth / value)


def _normalized_rmse(value: Any, truth: Any) -> float | None:
    fitted = _finite_array(value)
    expected = _finite_array(truth)
    if fitted is None or expected is None or fitted.shape != expected.shape:
        return None
    rmse = float(np.sqrt(np.mean((fitted - expected) ** 2)))
    target_scale = max(
        float(np.ptp(expected)),
        float(np.median(np.abs(expected))),
        1.0e-12,
    )
    return rmse / target_scale


def _centered_normalized_rmse(value: Any, truth: Any) -> float | None:
    """Return prediction-shape error after removing a constant offset."""
    fitted = _finite_array(value)
    expected = _finite_array(truth)
    if fitted is None or expected is None or fitted.shape != expected.shape:
        return None
    fitted_centered = fitted - float(np.median(fitted))
    expected_centered = expected - float(np.median(expected))
    rmse = float(
        np.sqrt(np.mean((fitted_centered - expected_centered) ** 2))
    )
    target_scale = max(float(np.ptp(expected_centered)), 1.0e-12)
    return rmse / target_scale


def _rbf_correlation_matrix(
    wavelengths: Any,
    lengthscale: float | None,
) -> np.ndarray | None:
    values = _finite_array(wavelengths)
    lengthscale = _finite_float(lengthscale)
    if values is None or lengthscale is None or lengthscale <= 0.0:
        return None
    delta = values[:, None] - values[None, :]
    return np.exp(-0.5 * (delta / lengthscale) ** 2)


def _wavelength_correlation_rmse(
    wavelengths: Any,
    fitted_lengthscale: float | None,
    truth_lengthscale: float | None,
) -> tuple[float | None, float | None, float | None]:
    fitted = _rbf_correlation_matrix(wavelengths, fitted_lengthscale)
    truth = _rbf_correlation_matrix(wavelengths, truth_lengthscale)
    if fitted is None or truth is None or fitted.shape != truth.shape:
        return None, None, None
    rmse = float(np.sqrt(np.mean((fitted - truth) ** 2)))
    endpoint_truth = float(truth[0, -1]) if truth.size else None
    endpoint_fitted = float(fitted[0, -1]) if fitted.size else None
    return rmse, endpoint_fitted, endpoint_truth


def _realized_wavelength_correlation_rmse(
    case: SyntheticWavelengthValidationCase,
    truth_lengthscale: float | None,
) -> tuple[float | None, float | None, float | None, int | None]:
    """Compare one shared-grid latent realization with declared RBF truth."""
    configuration = case.scenario.sampling_configuration
    if not bool(configuration.get("shared_time_grid")):
        return None, None, None, None

    covariance_truth = (
        case.scenario.truth.wavelength_covariance_parameters
    )
    if (
        str(covariance_truth.get("covariance_kind"))
        == "joint_spectral_mixture_ard"
        and abs(float(covariance_truth.get("wavelength_frequency", 0.0)))
        > 1.0e-12
    ):
        return None, None, None, None

    labels = np.asarray(case.band_labels, dtype=str)
    times = np.asarray(case.time_values, dtype=float)
    latent = np.asarray(case.latent_process, dtype=float)
    rows = []
    reference_times = None
    for band in case.scenario.truth.band_labels:
        mask = labels == str(band)
        order = np.argsort(times[mask])
        band_times = times[mask][order]
        band_latent = latent[mask][order]
        if band_times.size < 2:
            return None, None, None, None
        if reference_times is None:
            reference_times = band_times
        elif (
            band_times.shape != reference_times.shape
            or not np.allclose(band_times, reference_times)
        ):
            return None, None, None, None
        rows.append(band_latent)

    matrix = np.vstack(rows)
    if np.any(np.std(matrix, axis=1) <= 0.0):
        return None, None, None, None
    realized = np.corrcoef(matrix)
    truth = _rbf_correlation_matrix(
        case.scenario.truth.physical_wavelengths,
        truth_lengthscale,
    )
    if truth is None or realized.shape != truth.shape:
        return None, None, None, None
    rmse = float(np.sqrt(np.mean((realized - truth) ** 2)))
    endpoint_realized = float(realized[0, -1]) if realized.size else None
    endpoint_truth = float(truth[0, -1]) if truth.size else None
    return rmse, endpoint_realized, endpoint_truth, int(matrix.shape[1])


def _wavelength_covariance_recovery_limitations(
    case: SyntheticWavelengthValidationCase,
    model: str,
) -> tuple[str, ...]:
    """Return limitations that make strict lengthscale recovery ineligible."""
    temporal = case.scenario.truth.temporal_parameters
    if (
        bool(temporal.get("fundamental_plus_harmonic"))
        and model in {
            "2DWavelengthDependent",
            "2DDustMean",
            "2DPowerLawMean",
            "2DSeparable",
        }
    ):
        return (
            "multiple_temporal_components_fitted_with_single_quasi_periodic_kernel",
        )
    return ()


def _boundary_label_metric(
    boundary_hits: Sequence[Mapping[str, Any]] | None,
) -> tuple[bool, int, list[dict[str, Any]]]:
    normalized: list[dict[str, Any]] = []
    valid = True
    for raw_hit in boundary_hits or ():
        if not isinstance(raw_hit, Mapping):
            valid = False
            normalized.append({"unparsed": _json_safe(raw_hit)})
            continue
        hit = {str(key): _json_safe(item) for key, item in raw_hit.items()}
        dimension = str(hit.get("dimension_name") or "")
        parameter = str(
            hit.get("parameter_name") or hit.get("parameter") or ""
        )
        if dimension not in {"temporal_frequency", "wavelength_frequency"}:
            valid = False
        if parameter and parameter not in {"mixture_means", "mixture_scales"}:
            valid = False
        normalized.append(hit)
    return valid, len(normalized), normalized


def _sm_raw_values(
    diagnostics: Mapping[str, Any] | None,
    parameter_name: str,
) -> np.ndarray | None:
    if not isinstance(diagnostics, Mapping):
        return None
    parameter = (diagnostics.get("parameters") or {}).get(parameter_name) or {}
    values = parameter.get("raw_input_coordinate_values")
    if values is None:
        values = parameter.get("model_coordinate_values")
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.size == 0 or not np.all(np.isfinite(array)):
        return None
    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    return array


def _sm_truth_values(
    case: SyntheticWavelengthValidationCase,
    parameter_name: str,
) -> np.ndarray | None:
    values = case.scenario.truth.wavelength_covariance_parameters.get(
        parameter_name
    )
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.size == 0 or not np.all(np.isfinite(array)):
        return None
    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    return array


def _sorted_sm_components(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if array.shape[1] == 0:
        return array, np.arange(array.shape[0])
    order = np.argsort(array[:, 0], kind="stable")
    return array[order], order


def _sm_recovery_metrics(
    case: SyntheticWavelengthValidationCase,
    model: str,
    diagnostics: Mapping[str, Any] | None,
    *,
    factor_threshold: float,
) -> list[WavelengthRecoveryMetric]:
    output: list[WavelengthRecoveryMetric] = []
    dimension_names = ("temporal_frequency", "wavelength_frequency")
    for parameter_name in ("mixture_means", "mixture_scales"):
        truth = _sm_truth_values(case, parameter_name)
        fitted = _sm_raw_values(diagnostics, parameter_name)
        if truth is None:
            continue
        if fitted is None or fitted.shape != truth.shape:
            for component_index in range(truth.shape[0]):
                for dimension_index in range(truth.shape[1]):
                    dimension = (
                        dimension_names[dimension_index]
                        if dimension_index < len(dimension_names)
                        else f"ard_dimension_{dimension_index}"
                    )
                    output.append(
                        _metric(
                            name="sm_ard_factor_error",
                            value=None,
                            truth_value=float(
                                truth[component_index, dimension_index]
                            ),
                            available=False,
                            passed=None,
                            direction=RecoveryMetricDirection.LOWER_IS_BETTER,
                            threshold={"maximum": factor_threshold},
                            scope="spectral_mixture_ard_recovery",
                            model=model,
                            parameter=parameter_name,
                            ard_dimension=dimension,
                            component_index=component_index,
                            summary=(
                                "Multiplicative recovery error in raw input "
                                "coordinates for one spectral-mixture ARD value."
                            ),
                        )
                    )
            continue
        truth_sorted, truth_order = _sorted_sm_components(truth)
        fitted_sorted, fitted_order = _sorted_sm_components(fitted)
        for component_index in range(truth_sorted.shape[0]):
            for dimension_index in range(truth_sorted.shape[1]):
                truth_value = float(truth_sorted[component_index, dimension_index])
                fitted_value = float(
                    fitted_sorted[component_index, dimension_index]
                )
                factor_error = _factor_error(fitted_value, truth_value)
                dimension = (
                    dimension_names[dimension_index]
                    if dimension_index < len(dimension_names)
                    else f"ard_dimension_{dimension_index}"
                )
                output.append(
                    _metric(
                        name="sm_ard_factor_error",
                        value=factor_error,
                        truth_value=truth_value,
                        available=factor_error is not None,
                        passed=(
                            factor_error <= factor_threshold
                            if factor_error is not None
                            else None
                        ),
                        direction=RecoveryMetricDirection.LOWER_IS_BETTER,
                        threshold={"maximum": factor_threshold},
                        scope="spectral_mixture_ard_recovery",
                        model=model,
                        parameter=parameter_name,
                        ard_dimension=dimension,
                        component_index=component_index,
                        summary=(
                            "Multiplicative recovery error in raw input "
                            "coordinates for one spectral-mixture ARD value."
                        ),
                        metadata={
                            "fitted_value": fitted_value,
                            "truth_component_index": int(
                                truth_order[component_index]
                            ),
                            "fitted_component_index": int(
                                fitted_order[component_index]
                            ),
                            "component_matching": (
                                "sorted_by_temporal_frequency"
                            ),
                        },
                    )
                )
                output.append(
                    _metric(
                        name="sm_ard_absolute_error",
                        value=abs(fitted_value - truth_value),
                        truth_value=truth_value,
                        available=True,
                        passed=None,
                        direction=RecoveryMetricDirection.INFORMATIONAL,
                        scope="spectral_mixture_ard_recovery",
                        model=model,
                        parameter=parameter_name,
                        ard_dimension=dimension,
                        component_index=component_index,
                        summary=(
                            "Absolute recovery error in raw input coordinates "
                            "for one spectral-mixture ARD value."
                        ),
                        metadata={"fitted_value": fitted_value},
                    )
                )
    return output


def build_synthetic_wavelength_recovery_metrics(
    case: SyntheticWavelengthValidationCase | Mapping[str, Any],
    model: str,
    *,
    fitted_period: float | None = None,
    fitted_wavelength_lengthscale: float | None = None,
    fitted_mean_by_band: Sequence[float] | None = None,
    boundary_hits: Sequence[Mapping[str, Any]] | None = None,
    sm_ard_diagnostics: Mapping[str, Any] | None = None,
    thresholds: SyntheticWavelengthRecoveryThresholds | Mapping[str, Any] | None = None,
) -> tuple[WavelengthRecoveryMetric, ...]:
    """Build D1 recovery metrics for one fitted synthetic case.

    Missing fitted quantities produce explicit unavailable metrics rather than
    being omitted or converted into zeros.
    """
    case = _as_case(case)
    model = str(model or "").strip()
    if not model:
        raise ValueError("model must be non-empty.")
    thresholds = _as_thresholds(thresholds)
    truth = case.scenario.truth
    if truth is None:
        raise ValueError("Synthetic recovery requires a truth record.")

    coordinate_error = _coordinate_round_trip_error(case)
    period_truth = _finite_float(
        truth.temporal_parameters.get("fundamental_period")
    )
    period_value = _finite_float(fitted_period)
    period_error = _relative_error(period_value, period_truth)
    lengthscale_truth = _finite_float(
        truth.wavelength_covariance_parameters.get("wavelength_lengthscale")
    )
    lengthscale_value = _finite_float(fitted_wavelength_lengthscale)
    lengthscale_error = _factor_error(lengthscale_value, lengthscale_truth)
    mean_truth = truth.noiseless_summary.get("mean_by_band")
    mean_error = _normalized_rmse(fitted_mean_by_band, mean_truth)
    mean_shape_error = _centered_normalized_rmse(
        fitted_mean_by_band, mean_truth
    )
    (
        correlation_rmse,
        fitted_endpoint_correlation,
        truth_endpoint_correlation,
    ) = _wavelength_correlation_rmse(
        truth.physical_wavelengths,
        lengthscale_value,
        lengthscale_truth,
    )
    (
        realized_correlation_rmse,
        realized_endpoint_correlation,
        realized_truth_endpoint_correlation,
        realized_time_points,
    ) = _realized_wavelength_correlation_rmse(case, lengthscale_truth)
    covariance_limitations = _wavelength_covariance_recovery_limitations(
        case, model
    )
    strict_covariance_recovery = not covariance_limitations
    strength = str(truth.metadata.get("dependence_strength") or "")
    mean_threshold = thresholds.mean_normalized_rmse(strength)
    labels_valid, boundary_count, normalized_hits = _boundary_label_metric(
        boundary_hits
    )

    metrics = [
        _metric(
            name="coordinate_round_trip_max_abs_error",
            value=coordinate_error,
            truth_value=0.0,
            available=coordinate_error is not None,
            passed=(
                coordinate_error <= thresholds.coordinate_round_trip_max_abs_error
                if coordinate_error is not None
                else None
            ),
            direction=RecoveryMetricDirection.LOWER_IS_BETTER,
            threshold={
                "maximum": thresholds.coordinate_round_trip_max_abs_error
            },
            units="physical_wavelength",
            scope="coordinate_transform",
            model=model,
            summary="Maximum wavelength-coordinate round-trip error.",
        ),
        _metric(
            name="period_relative_error",
            value=period_error,
            truth_value=period_truth,
            available=period_error is not None,
            passed=(
                period_error <= thresholds.period_relative_error
                if period_error is not None
                else None
            ),
            direction=RecoveryMetricDirection.LOWER_IS_BETTER,
            threshold={"maximum": thresholds.period_relative_error},
            units="fraction",
            scope="temporal_recovery",
            model=model,
            parameter="period",
            summary="Absolute period error divided by the generating period.",
            metadata={"fitted_period": period_value},
        ),
        _metric(
            name="wavelength_lengthscale_factor_error",
            value=lengthscale_error,
            truth_value=lengthscale_truth,
            available=lengthscale_error is not None,
            passed=(
                lengthscale_error
                <= thresholds.wavelength_lengthscale_factor_error
                if lengthscale_error is not None
                and strict_covariance_recovery
                else None
            ),
            direction=(
                RecoveryMetricDirection.LOWER_IS_BETTER
                if strict_covariance_recovery
                else RecoveryMetricDirection.INFORMATIONAL
            ),
            threshold=(
                {"maximum": thresholds.wavelength_lengthscale_factor_error}
                if strict_covariance_recovery
                else {}
            ),
            units="multiplicative_factor",
            scope="wavelength_covariance_recovery",
            model=model,
            parameter="wavelength_lengthscale",
            ard_dimension="wavelength_frequency",
            summary=(
                "Symmetric multiplicative error between fitted and generating "
                "physical wavelength lengthscales."
            ),
            limitations=covariance_limitations,
            metadata={
                "fitted_wavelength_lengthscale": lengthscale_value,
                "strict_recovery_eligible": strict_covariance_recovery,
            },
        ),
        _metric(
            name="wavelength_correlation_matrix_rmse",
            value=correlation_rmse,
            truth_value=0.0,
            available=correlation_rmse is not None,
            passed=None,
            direction=RecoveryMetricDirection.INFORMATIONAL,
            units="correlation_coefficient",
            scope="wavelength_covariance_recovery",
            model=model,
            parameter="wavelength_lengthscale",
            ard_dimension="wavelength_frequency",
            summary=(
                "RMSE between fitted and generating RBF wavelength "
                "correlation matrices at the observed bands."
            ),
            limitations=covariance_limitations,
            metadata={
                "fitted_endpoint_correlation": fitted_endpoint_correlation,
                "truth_endpoint_correlation": truth_endpoint_correlation,
                "strict_recovery_eligible": strict_covariance_recovery,
            },
        ),
        _metric(
            name="realized_wavelength_correlation_matrix_rmse",
            value=realized_correlation_rmse,
            truth_value=0.0,
            available=realized_correlation_rmse is not None,
            passed=None,
            direction=RecoveryMetricDirection.INFORMATIONAL,
            units="correlation_coefficient",
            scope="synthetic_realization_diagnostics",
            model=model,
            parameter="latent_process",
            ard_dimension="wavelength_frequency",
            summary=(
                "RMSE between the empirical shared-grid latent-process "
                "correlation matrix and the declared generating RBF matrix."
            ),
            metadata={
                "realized_endpoint_correlation": realized_endpoint_correlation,
                "truth_endpoint_correlation": (
                    realized_truth_endpoint_correlation
                ),
                "n_shared_time_points": realized_time_points,
            },
        ),
        _metric(
            name="mean_law_normalized_rmse",
            value=mean_error,
            truth_value=_json_safe(mean_truth),
            available=mean_error is not None,
            passed=(mean_error <= mean_threshold if mean_error is not None else None),
            direction=RecoveryMetricDirection.LOWER_IS_BETTER,
            threshold={"maximum": mean_threshold},
            units="fraction_of_target_scale",
            scope="wavelength_mean_recovery",
            model=model,
            parameter="mean_module",
            summary=(
                "RMSE of fitted versus generating mean at observed bands, "
                "normalized by the larger of target span and median magnitude."
            ),
            metadata={
                "fitted_mean_by_band": _json_safe(fitted_mean_by_band),
                "dependence_strength": strength,
            },
        ),
        _metric(
            name="mean_law_centered_normalized_rmse",
            value=mean_shape_error,
            truth_value=_json_safe(mean_truth),
            available=mean_shape_error is not None,
            passed=None,
            direction=RecoveryMetricDirection.INFORMATIONAL,
            units="fraction_of_target_span",
            scope="wavelength_mean_recovery",
            model=model,
            parameter="mean_module",
            summary=(
                "Prediction-shape RMSE after removing the median offset from "
                "both fitted and generating band means."
            ),
            metadata={
                "fitted_mean_by_band": _json_safe(fitted_mean_by_band),
                "dependence_strength": strength,
            },
        ),
        _metric(
            name="ard_boundary_hit_count",
            value=boundary_count,
            truth_value=None,
            available=True,
            passed=None,
            direction=RecoveryMetricDirection.INFORMATIONAL,
            scope="ard_boundary_diagnostics",
            model=model,
            summary="Number of reported spectral-mixture ARD boundary hits.",
            metadata={"boundary_hits": normalized_hits},
        ),
        _metric(
            name="ard_boundary_labels_valid",
            value=labels_valid,
            truth_value=True,
            available=True,
            passed=labels_valid,
            direction=RecoveryMetricDirection.EXACT_MATCH,
            scope="ard_boundary_diagnostics",
            model=model,
            summary=(
                "Whether every ARD hit uses the maintained parameter and "
                "temporal/wavelength dimension labels."
            ),
        ),
    ]
    metrics.extend(
        _sm_recovery_metrics(
            case,
            model,
            sm_ard_diagnostics,
            factor_threshold=thresholds.wavelength_lengthscale_factor_error,
        )
    )
    return tuple(metrics)


def _success_status(*, warning_count: int = 0) -> WavelengthAttemptStatus:
    return WavelengthAttemptStatus(
        disposition=AttemptDisposition.ATTEMPTED,
        execution_stage=ExecutionStage.COMPLETED,
        technical_outcome=(
            TechnicalOutcome.COMPLETED_WITH_WARNINGS
            if warning_count
            else TechnicalOutcome.COMPLETED
        ),
        diagnostic_validity=DiagnosticValidity.VALID,
        scientific_usability=ScientificUsability.USABLE,
        comparison_eligibility=ComparisonEligibility.ELIGIBLE,
        warning_severity=(WarningSeverity.WARNING if warning_count else None),
    )


def _failed_status(stage: ExecutionStage) -> WavelengthAttemptStatus:
    return WavelengthAttemptStatus(
        disposition=AttemptDisposition.ATTEMPTED,
        execution_stage=stage,
        technical_outcome=TechnicalOutcome.FAILED,
        diagnostic_validity=DiagnosticValidity.UNAVAILABLE,
        scientific_usability=ScientificUsability.UNUSABLE,
        comparison_eligibility=ComparisonEligibility.INELIGIBLE,
        warning_severity=WarningSeverity.ERROR,
    )


def _run_seed(case: SyntheticWavelengthValidationCase) -> int | None:
    seeds = case.scenario.sampling_configuration.get("purpose_specific_seeds")
    if not isinstance(seeds, Mapping):
        return None
    master = seeds.get("master")
    try:
        master = int(master)
    except (TypeError, ValueError):
        return None
    return master if master >= 0 else None


def evaluate_synthetic_wavelength_recovery(
    case: SyntheticWavelengthValidationCase | Mapping[str, Any],
    model: str,
    *,
    fitted_period: float | None = None,
    fitted_wavelength_lengthscale: float | None = None,
    fitted_mean_by_band: Sequence[float] | None = None,
    boundary_hits: Sequence[Mapping[str, Any]] | None = None,
    sm_ard_diagnostics: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    parameter_workflow: Mapping[str, Any] | None = None,
    fit_configuration: Mapping[str, Any] | None = None,
    status: WavelengthAttemptStatus | None = None,
    failure: WavelengthFailureRecord | None = None,
    thresholds: SyntheticWavelengthRecoveryThresholds | Mapping[str, Any] | None = None,
    run_id: str | None = None,
    provenance: WavelengthValidationProvenance | Mapping[str, Any] | None = None,
    notes: Sequence[str] = (),
) -> WavelengthValidationRun:
    """Evaluate one synthetic fit and return a typed validation run."""
    case = _as_case(case)
    model = str(model or "").strip()
    if model not in SUPPORTED_SYNTHETIC_RECOVERY_MODELS:
        allowed = ", ".join(SUPPORTED_SYNTHETIC_RECOVERY_MODELS)
        raise ValueError(f"model must be one of: {allowed}.")
    if failure is not None and status is None:
        status = _failed_status(failure.stage)
    if status is None:
        status = _success_status()
    metrics = build_synthetic_wavelength_recovery_metrics(
        case,
        model,
        fitted_period=fitted_period,
        fitted_wavelength_lengthscale=fitted_wavelength_lengthscale,
        fitted_mean_by_band=fitted_mean_by_band,
        boundary_hits=boundary_hits,
        sm_ard_diagnostics=sm_ard_diagnostics,
        thresholds=thresholds,
    )
    seed = _run_seed(case)
    run_id = run_id or f"{case.scenario.scenario_id}:{model}:seed-{seed}"
    if provenance is None:
        provenance = case.scenario.provenance
    diagnostic_payload = dict(diagnostics or {})
    diagnostic_payload.update(
        {
            "fitted_period": _finite_float(fitted_period),
            "fitted_wavelength_lengthscale": _finite_float(
                fitted_wavelength_lengthscale
            ),
            "fitted_mean_by_band": _json_safe(fitted_mean_by_band),
            "ard_boundary_hits": _json_safe(boundary_hits or ()),
            "sm_ard_diagnostics": _json_safe(sm_ard_diagnostics),
            "synthetic_recovery_schema_version": (
                SYNTHETIC_WAVELENGTH_RECOVERY_SCHEMA_VERSION
            ),
        }
    )
    return WavelengthValidationRun(
        run_id=run_id,
        scenario_id=case.scenario.scenario_id,
        model=model,
        seed=seed,
        status=status,
        failure=failure,
        fit_configuration=dict(fit_configuration or {}),
        diagnostics=diagnostic_payload,
        parameter_workflow=dict(parameter_workflow or {}),
        metrics=metrics,
        provenance=provenance,
        notes=tuple(str(item) for item in notes),
        advisory_only=True,
        extra_fields={
            "synthetic_recovery_schema_version": (
                SYNTHETIC_WAVELENGTH_RECOVERY_SCHEMA_VERSION
            ),
            "generating_model": case.scenario.truth.generating_model,
        },
    )


def _default_fit_configuration(
    case: SyntheticWavelengthValidationCase,
    model: str,
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    configuration: dict[str, Any] = {
        "model": model,
        "training_iter": 300,
        "miniter": 100,
        "learn_additional_noise": True,
        "verbose": False,
    }
    scenario_configuration = case.scenario.fit_configuration
    for name in (
        "training_iter",
        "miniter",
        "learn_additional_noise",
        "verbose",
        "lr",
        "stop",
        "stopavg",
    ):
        if name in scenario_configuration:
            configuration[name] = scenario_configuration[name]
    if model == "2D":
        configuration.update(
            {
                "num_mixtures": 1,
                "use_best_band_init": True,
            }
        )
    else:
        configuration.update(
            {
                "fit_strategy": "consensus",
                "time_kernel_type": "quasi_periodic",
                "use_acf": True,
            }
        )
    configuration.update(dict(overrides or {}))
    configuration["model"] = model
    return configuration


def _tensor_scalar(value: Any) -> float | None:
    if value is None:
        return None
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size != 1 or not np.isfinite(array[0]):
        return None
    return float(array[0])


def _iter_kernel_objects(lightcurve: Any):
    model = getattr(lightcurve, "model", None)
    stack = [getattr(model, "covar_module", None)] if model is not None else []
    seen: set[int] = set()
    while stack:
        item = stack.pop()
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        yield item
        for name in ("base_kernel", "data_covar_module", "covar_module"):
            child = getattr(item, name, None)
            if child is not None:
                stack.append(child)
        kernels = getattr(item, "kernels", None)
        if kernels is not None:
            try:
                stack.extend(list(kernels))
            except TypeError:
                pass


def _dominant_sm_component_index(kernel: Any) -> int:
    weights = getattr(kernel, "mixture_weights", None)
    try:
        values = np.asarray(_json_safe(weights), dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return 0
    if values.size == 0 or not np.any(np.isfinite(values)):
        return 0
    return int(np.nanargmax(values))


def _extract_fitted_period(
    lightcurve: Any,
    case: SyntheticWavelengthValidationCase,
) -> float | None:
    consensus = getattr(lightcurve, "consensus_diagnostics", None)
    if isinstance(consensus, Mapping):
        value = _finite_float(consensus.get("consensus_period"))
        if value is not None:
            return value
    time_scale = _finite_float(
        (case.scenario.truth.coordinate_transforms.get("time") or {}).get(
            "scale"
        )
    ) or 1.0
    for kernel in _iter_kernel_objects(lightcurve):
        value = _tensor_scalar(getattr(kernel, "period_length", None))
        if value is not None:
            return value * time_scale
        means = getattr(kernel, "mixture_means", None)
        try:
            array = np.asarray(_json_safe(means), dtype=float)
            array = array.reshape(array.shape[0], -1)
            component = _dominant_sm_component_index(kernel)
            temporal_frequency = float(array[component, 0])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(temporal_frequency) and temporal_frequency > 0.0:
            return time_scale / temporal_frequency
    return None


def _wavelength_model_scale(case: SyntheticWavelengthValidationCase) -> float:
    transform = case.scenario.truth.coordinate_transforms.get("wavelength") or {}
    return _finite_float(transform.get("scale")) or 1.0


def _extract_wavelength_lengthscale(
    lightcurve: Any,
    case: SyntheticWavelengthValidationCase,
) -> float | None:
    model = getattr(lightcurve, "model", None)
    covariance = getattr(model, "covar_module", None)
    kernels = getattr(covariance, "kernels", None)
    if kernels is not None:
        try:
            wavelength_kernel = list(kernels)[1]
        except (IndexError, TypeError):
            wavelength_kernel = None
        stack = [wavelength_kernel]
        seen: set[int] = set()
        while stack:
            item = stack.pop()
            if item is None or id(item) in seen:
                continue
            seen.add(id(item))
            value = _tensor_scalar(getattr(item, "lengthscale", None))
            if value is not None:
                return value * _wavelength_model_scale(case)
            stack.append(getattr(item, "base_kernel", None))
    truth = case.scenario.truth.wavelength_covariance_parameters
    if str(truth.get("covariance_kind")) != "joint_spectral_mixture_ard":
        return None
    try:
        scales = None
        for kernel in _iter_kernel_objects(lightcurve):
            candidate = getattr(kernel, "mixture_scales", None)
            if candidate is not None:
                scales = candidate
                break
        array = np.asarray(_json_safe(scales), dtype=float)
        array = array.reshape(array.shape[0], -1)
        kernel = next(
            item
            for item in _iter_kernel_objects(lightcurve)
            if getattr(item, "mixture_scales", None) is not None
        )
        component = _dominant_sm_component_index(kernel)
        wavelength_scale = float(array[component, 1])
    except (TypeError, ValueError, IndexError):
        return None
    if not math.isfinite(wavelength_scale) or wavelength_scale <= 0.0:
        return None
    model_lengthscale = 1.0 / (2.0 * math.pi * wavelength_scale)
    return model_lengthscale * _wavelength_model_scale(case)


def _extract_fitted_mean_by_band(
    lightcurve: Any,
    case: SyntheticWavelengthValidationCase,
) -> list[float] | None:
    model = getattr(lightcurve, "model", None)
    mean_module = getattr(model, "mean_module", None)
    if mean_module is None:
        return None
    model_wavelengths = _finite_array(
        case.scenario.truth.noiseless_summary.get("model_wavelength_by_band")
    )
    if model_wavelengths is None:
        return None
    try:
        import torch

        parameter = next(mean_module.parameters(), None)
        dtype = parameter.dtype if parameter is not None else torch.get_default_dtype()
        device = parameter.device if parameter is not None else None
        inputs = torch.tensor(
            np.column_stack([np.zeros_like(model_wavelengths), model_wavelengths]),
            dtype=dtype,
            device=device,
        )
        with torch.no_grad():
            values = mean_module(inputs)
        output = np.asarray(_json_safe(values), dtype=float).reshape(-1)
    except Exception:
        return None
    if output.shape != model_wavelengths.shape or not np.all(np.isfinite(output)):
        return None
    return [float(item) for item in output]


def _extract_parameter_workflow(lightcurve: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    summary_getter = getattr(
        lightcurve, "get_parameter_workflow_summary", None
    )
    if callable(summary_getter):
        try:
            value = summary_getter()
            if isinstance(value, Mapping):
                summary = dict(_json_safe(value))
        except Exception:
            summary = {
                "available": False,
                "reason": "summary_extraction_failed",
            }

    report_getter = getattr(lightcurve, "get_parameter_workflow_report", None)
    if callable(report_getter):
        try:
            report = report_getter()
            if isinstance(report, Mapping):
                summary["report"] = dict(_json_safe(report))
        except Exception:
            summary["report_error"] = "report_extraction_failed"

    if summary:
        return summary

    value = getattr(lightcurve, "parameter_workflow_result", None)
    if hasattr(value, "to_dict"):
        try:
            value = value.to_dict()
        except Exception:
            value = None
    return dict(_json_safe(value)) if isinstance(value, Mapping) else {}


def _extract_boundary_hits(lightcurve: Any, fit_kwargs: Mapping[str, Any]):
    try:
        from .spectral_mixture_ard_diagnostics import (
            diagnose_spectral_mixture_ard,
        )

        diagnostics = diagnose_spectral_mixture_ard(
            lightcurve,
            requested_num_mixtures=fit_kwargs.get("num_mixtures"),
        )
    except Exception:
        return [], {"available": False, "reason": "ard_diagnostics_failed"}
    return list(diagnostics.get("boundary_hits") or []), diagnostics


def _extract_default_fit_outputs(
    lightcurve: Any,
    case: SyntheticWavelengthValidationCase,
    fit_kwargs: Mapping[str, Any],
    fit_result: Any,
) -> dict[str, Any]:
    boundary_hits, ard_diagnostics = _extract_boundary_hits(
        lightcurve, fit_kwargs
    )
    consensus = getattr(lightcurve, "consensus_diagnostics", None)
    fit_history = getattr(lightcurve, "fit_history", None)
    return {
        "fit_result": fit_result,
        "fitted_period": _extract_fitted_period(lightcurve, case),
        "fitted_wavelength_lengthscale": _extract_wavelength_lengthscale(
            lightcurve, case
        ),
        "fitted_mean_by_band": _extract_fitted_mean_by_band(lightcurve, case),
        "boundary_hits": boundary_hits,
        "sm_ard_diagnostics": ard_diagnostics,
        "parameter_workflow": _extract_parameter_workflow(lightcurve),
        "diagnostics": {
            "consensus_diagnostics": _json_safe(consensus),
            "fit_history": _json_safe(fit_history),
            "sm_ard_diagnostics": _json_safe(ard_diagnostics),
            "fit_result_type": type(fit_result).__name__,
        },
    }


def _failure_record(
    exception: BaseException,
    *,
    stage: ExecutionStage,
    substage: str,
) -> WavelengthFailureRecord:
    exception_type = exception.__class__
    exception_mro = tuple(
        item.__name__ for item in exception_type.__mro__
    )
    diagnostics = {
        "exception_type": exception_type.__name__,
        "exception_module": exception_type.__module__,
        "exception_qualname": exception_type.__qualname__,
        "exception_mro": list(exception_mro),
    }
    structured = getattr(exception, "failure_diagnostics", None)
    if isinstance(structured, Mapping):
        diagnostics["structured_failure_diagnostics"] = _json_safe(structured)
    return WavelengthFailureRecord(
        failure_code=(
            "synthetic_recovery_fit_failed"
            if stage is ExecutionStage.OPTIMIZATION
            else "synthetic_recovery_diagnostics_failed"
        ),
        stage=stage,
        substage=substage,
        exception_type=exception_type.__name__,
        message=str(exception),
        diagnostics=diagnostics,
        traceback_reference=None,
    )


def run_synthetic_wavelength_recovery(
    case: SyntheticWavelengthValidationCase | Mapping[str, Any],
    model: str,
    *,
    fit_kwargs: Mapping[str, Any] | None = None,
    thresholds: SyntheticWavelengthRecoveryThresholds | Mapping[str, Any] | None = None,
    lightcurve_factory: Callable[[SyntheticWavelengthValidationCase], Any] | None = None,
    fit_runner: Callable[[Any, Mapping[str, Any], SyntheticWavelengthValidationCase], Any]
    | None = None,
    stop_on_error: bool = False,
) -> WavelengthValidationRun:
    """Fit and evaluate one synthetic case without selecting a model.

    A custom ``fit_runner`` may return a mapping containing fitted quantities,
    diagnostics, and parameter-workflow provenance.  When it returns ``None``,
    the maintained fit object is inspected after execution.
    """
    case = _as_case(case)
    model = str(model or "").strip()
    if model not in SUPPORTED_SYNTHETIC_RECOVERY_MODELS:
        allowed = ", ".join(SUPPORTED_SYNTHETIC_RECOVERY_MODELS)
        raise ValueError(f"model must be one of: {allowed}.")
    configuration = _default_fit_configuration(case, model, fit_kwargs)
    factory = lightcurve_factory or (lambda item: item.to_lightcurve())
    try:
        lightcurve = factory(case)
    except Exception as exc:
        if stop_on_error:
            raise
        failure = _failure_record(
            exc, stage=ExecutionStage.SETUP, substage="lightcurve_construction"
        )
        return evaluate_synthetic_wavelength_recovery(
            case,
            model,
            fit_configuration=configuration,
            failure=failure,
            status=_failed_status(ExecutionStage.SETUP),
            thresholds=thresholds,
        )

    try:
        if fit_runner is None:
            fit_result = lightcurve.fit(**configuration)
            output: Any = None
        else:
            fit_result = None
            output = fit_runner(lightcurve, dict(configuration), case)
    except Exception as exc:
        if stop_on_error:
            raise
        failure = _failure_record(
            exc, stage=ExecutionStage.OPTIMIZATION, substage="fit_execution"
        )
        return evaluate_synthetic_wavelength_recovery(
            case,
            model,
            fit_configuration=configuration,
            failure=failure,
            status=_failed_status(ExecutionStage.OPTIMIZATION),
            thresholds=thresholds,
        )

    try:
        if isinstance(output, WavelengthValidationRun):
            return output
        if output is None:
            output = _extract_default_fit_outputs(
                lightcurve, case, configuration, fit_result
            )
        if not isinstance(output, Mapping):
            raise TypeError("fit_runner must return a mapping, run record, or None")
        failure = output.get("failure")
        if isinstance(failure, Mapping):
            failure = WavelengthFailureRecord(
                failure_code=str(failure.get("failure_code") or "fit_failed"),
                stage=ExecutionStage(
                    str(failure.get("failure_stage") or "optimization")
                ),
                substage=failure.get("failure_substage"),
                exception_type=failure.get("exception_type"),
                message=str(failure.get("exception_message") or "fit failed"),
                diagnostics=dict(failure.get("failure_diagnostics") or {}),
                traceback_reference=failure.get("traceback_reference"),
            )
        status = output.get("status")
        if status is not None and not isinstance(status, WavelengthAttemptStatus):
            raise TypeError("fit_runner status must be a WavelengthAttemptStatus")
        return evaluate_synthetic_wavelength_recovery(
            case,
            model,
            fitted_period=output.get("fitted_period"),
            fitted_wavelength_lengthscale=output.get(
                "fitted_wavelength_lengthscale"
            ),
            fitted_mean_by_band=output.get("fitted_mean_by_band"),
            boundary_hits=output.get("boundary_hits"),
            sm_ard_diagnostics=output.get("sm_ard_diagnostics"),
            diagnostics=output.get("diagnostics") or {},
            parameter_workflow=output.get("parameter_workflow") or {},
            fit_configuration=configuration,
            status=status,
            failure=failure,
            thresholds=thresholds,
            notes=tuple(output.get("notes") or ()),
        )
    except Exception as exc:
        if stop_on_error:
            raise
        failure = _failure_record(
            exc,
            stage=ExecutionStage.DIAGNOSTICS,
            substage="recovery_extraction",
        )
        return evaluate_synthetic_wavelength_recovery(
            case,
            model,
            fit_configuration=configuration,
            failure=failure,
            status=_failed_status(ExecutionStage.DIAGNOSTICS),
            thresholds=thresholds,
        )


def _numeric_metric_summary(values: Sequence[Any]) -> dict[str, Any]:
    numeric = []
    for value in values:
        if isinstance(value, bool):
            continue
        finite = _finite_float(value)
        if finite is not None:
            numeric.append(finite)
    if not numeric:
        return {}
    array = np.asarray(numeric, dtype=float)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
        "p90": float(np.percentile(array, 90.0)),
    }


def _metric_summaries_from_runs(
    runs: Sequence[WavelengthValidationRun],
    *,
    strict_recovery_only: bool = False,
) -> dict[str, dict[str, Any]]:
    metrics_by_name: dict[str, list[WavelengthRecoveryMetric]] = {}
    for run in runs:
        for metric in run.metrics:
            if (
                strict_recovery_only
                and metric.metadata.get("strict_recovery_eligible") is False
            ):
                continue
            metrics_by_name.setdefault(metric.name, []).append(metric)
    output: dict[str, dict[str, Any]] = {}
    for name, metrics in metrics_by_name.items():
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
        summary.update(
            _numeric_metric_summary([item.value for item in available])
        )
        output[name] = summary
    return output


def _scenario_population_summaries(
    runs: Sequence[WavelengthValidationRun],
) -> dict[str, dict[str, Any]]:
    """Summarize repeated matched-truth runs by scenario identifier."""
    grouped: dict[str, list[WavelengthValidationRun]] = {}
    for run in runs:
        grouped.setdefault(run.scenario_id, []).append(run)

    completed_outcomes = {
        TechnicalOutcome.COMPLETED,
        TechnicalOutcome.COMPLETED_WITH_WARNINGS,
        TechnicalOutcome.COMPLETED_WITH_RECOVERY,
    }
    output: dict[str, dict[str, Any]] = {}
    for scenario_id, scenario_runs in grouped.items():
        n_completed = sum(
            run.status is not None
            and run.status.technical_outcome in completed_outcomes
            for run in scenario_runs
        )
        strict_metrics = _metric_summaries_from_runs(
            scenario_runs, strict_recovery_only=True
        )
        output[scenario_id] = {
            "n_runs": len(scenario_runs),
            "n_completed": n_completed,
            "n_failed": len(scenario_runs) - n_completed,
            "completion_fraction": (
                n_completed / len(scenario_runs) if scenario_runs else None
            ),
            "strict_metric_summaries": strict_metrics,
        }
    return output


def _aggregate_gate_summary(
    metric_summaries: Mapping[str, Mapping[str, Any]],
    *,
    n_failed_runs: int,
    n_matched_runs: int,
    scenario_population_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    gates: dict[str, dict[str, Any]] = {}
    population_mode = any(
        int(summary.get("n_runs") or 0) > 1
        for summary in scenario_population_summaries.values()
    )

    def add_gate(name: str, evaluated: bool, passed: bool | None, **details):
        gates[name] = {
            "evaluated": bool(evaluated),
            "passed": bool(passed) if passed is not None else None,
            **details,
        }

    coordinate = metric_summaries.get(
        "coordinate_round_trip_max_abs_error", {}
    )
    coordinate_available = int(coordinate.get("n_available") or 0) > 0
    coordinate_maximum = _finite_float(coordinate.get("maximum"))
    add_gate(
        "coordinate_round_trip",
        coordinate_available and coordinate_maximum is not None,
        (
            coordinate_maximum <= 1.0e-10
            if coordinate_available and coordinate_maximum is not None
            else None
        ),
        maximum=coordinate_maximum,
        required_maximum=1.0e-10,
    )

    period = metric_summaries.get("period_relative_error", {})
    period_median = _finite_float(period.get("median"))
    period_p90 = _finite_float(period.get("p90"))
    period_evaluated = (
        int(period.get("n_available") or 0) > 0
        and period_median is not None
        and period_p90 is not None
    )
    add_gate(
        "period_recovery",
        period_evaluated,
        (
            period_median <= 0.05 and period_p90 <= 0.15
            if period_evaluated
            else None
        ),
        median=period_median,
        p90=period_p90,
        required_median_maximum=0.05,
        required_p90_maximum=0.15,
    )

    lengthscale = metric_summaries.get(
        "wavelength_lengthscale_factor_error", {}
    )
    lengthscale_median = _finite_float(lengthscale.get("median"))
    lengthscale_p90 = _finite_float(lengthscale.get("p90"))
    lengthscale_pass_fraction = _finite_float(
        lengthscale.get("pass_fraction")
    )
    lengthscale_evaluated = (
        int(lengthscale.get("n_available") or 0) > 0
        and lengthscale_median is not None
        and lengthscale_p90 is not None
    )

    scenario_lengthscale_summaries = []
    for scenario_id, scenario_summary in scenario_population_summaries.items():
        strict_metrics = scenario_summary.get("strict_metric_summaries") or {}
        item = strict_metrics.get(
            "wavelength_lengthscale_factor_error", {}
        )
        if int(item.get("n_available") or 0) <= 0:
            continue
        scenario_lengthscale_summaries.append(
            {
                "scenario_id": scenario_id,
                "n_available": int(item.get("n_available") or 0),
                "median": _finite_float(item.get("median")),
                "p90": _finite_float(item.get("p90")),
                "pass_fraction": _finite_float(item.get("pass_fraction")),
            }
        )
    scenario_medians = [
        item["median"]
        for item in scenario_lengthscale_summaries
        if item["median"] is not None
    ]
    scenario_pass_fractions = [
        item["pass_fraction"]
        for item in scenario_lengthscale_summaries
        if item["pass_fraction"] is not None
    ]
    maximum_scenario_median = (
        max(scenario_medians) if scenario_medians else None
    )
    minimum_scenario_pass_fraction = (
        min(scenario_pass_fractions)
        if scenario_pass_fractions
        else None
    )

    if population_mode:
        population_lengthscale_evaluated = (
            lengthscale_evaluated
            and lengthscale_pass_fraction is not None
            and maximum_scenario_median is not None
            and minimum_scenario_pass_fraction is not None
        )
        lengthscale_passed = (
            lengthscale_median <= 2.0
            and lengthscale_pass_fraction >= 0.80
            and maximum_scenario_median <= 4.0
            and minimum_scenario_pass_fraction >= 0.60
            if population_lengthscale_evaluated
            else None
        )
        add_gate(
            "wavelength_lengthscale_recovery",
            population_lengthscale_evaluated,
            lengthscale_passed,
            evaluation_mode="population",
            median=lengthscale_median,
            p90=lengthscale_p90,
            pass_fraction=lengthscale_pass_fraction,
            maximum_scenario_median=maximum_scenario_median,
            minimum_scenario_pass_fraction=(
                minimum_scenario_pass_fraction
            ),
            required_median_maximum=2.0,
            required_pass_fraction_minimum=0.80,
            required_scenario_median_maximum=4.0,
            required_scenario_pass_fraction_minimum=0.60,
            tail_instability_warning=(
                lengthscale_p90 is not None and lengthscale_p90 > 4.0
            ),
            scenario_summaries=scenario_lengthscale_summaries,
        )
    else:
        add_gate(
            "wavelength_lengthscale_recovery",
            lengthscale_evaluated,
            (
                lengthscale_median <= 2.0 and lengthscale_p90 <= 4.0
                if lengthscale_evaluated
                else None
            ),
            evaluation_mode="single_realization_matrix",
            median=lengthscale_median,
            p90=lengthscale_p90,
            required_median_maximum=2.0,
            required_p90_maximum=4.0,
        )

    for gate_name, metric_name in (
        ("mean_law_recovery", "mean_law_normalized_rmse"),
        ("ard_dimension_labels", "ard_boundary_labels_valid"),
    ):
        summary = metric_summaries.get(metric_name, {})
        evaluated = int(summary.get("n_available") or 0) > 0
        add_gate(
            gate_name,
            evaluated,
            int(summary.get("n_failed") or 0) == 0 if evaluated else None,
            n_available=int(summary.get("n_available") or 0),
            n_failed=int(summary.get("n_failed") or 0),
        )

    sm = metric_summaries.get("sm_ard_factor_error", {})
    sm_p90 = _finite_float(sm.get("p90"))
    sm_evaluated = int(sm.get("n_available") or 0) > 0 and sm_p90 is not None
    add_gate(
        "spectral_mixture_ard_recovery",
        sm_evaluated,
        sm_p90 <= 4.0 if sm_evaluated else None,
        p90=sm_p90,
        required_p90_maximum=4.0,
    )

    completion_fraction = (
        (n_matched_runs - n_failed_runs) / n_matched_runs
        if n_matched_runs > 0
        else None
    )
    scenario_completion_fractions = [
        _finite_float(summary.get("completion_fraction"))
        for summary in scenario_population_summaries.values()
    ]
    scenario_completion_fractions = [
        value for value in scenario_completion_fractions if value is not None
    ]
    minimum_scenario_completion_fraction = (
        min(scenario_completion_fractions)
        if scenario_completion_fractions
        else None
    )
    if population_mode:
        completion_evaluated = (
            completion_fraction is not None
            and minimum_scenario_completion_fraction is not None
        )
        completion_passed = (
            completion_fraction >= 0.95
            and minimum_scenario_completion_fraction >= 0.95
            if completion_evaluated
            else None
        )
        add_gate(
            "matched_run_completion",
            completion_evaluated,
            completion_passed,
            evaluation_mode="population",
            n_matched_runs=n_matched_runs,
            n_failed_runs=int(n_failed_runs),
            completion_fraction=completion_fraction,
            minimum_scenario_completion_fraction=(
                minimum_scenario_completion_fraction
            ),
            required_completion_fraction_minimum=0.95,
            required_scenario_completion_fraction_minimum=0.95,
        )
    else:
        add_gate(
            "matched_run_completion",
            True,
            n_failed_runs == 0,
            evaluation_mode="single_realization_matrix",
            n_matched_runs=n_matched_runs,
            n_failed_runs=int(n_failed_runs),
        )

    evaluated_results = [
        gate["passed"] for gate in gates.values() if gate["evaluated"]
    ]
    return {
        "gates": gates,
        "n_evaluated": len(evaluated_results),
        "n_passed": sum(result is True for result in evaluated_results),
        "n_failed": sum(result is False for result in evaluated_results),
        "all_evaluated_gates_passed": bool(
            evaluated_results and all(evaluated_results)
        ),
        "evaluation_mode": (
            "population" if population_mode else "single_realization_matrix"
        ),
        "scope": "runs_whose_fitted_model_matches_generating_model",
        "automatic_model_selection_applied": False,
    }


def aggregate_synthetic_wavelength_recovery_runs(
    runs: Sequence[WavelengthValidationRun | Mapping[str, Any]],
    *,
    aggregate_id: str = "d1-synthetic-wavelength-recovery",
) -> WavelengthValidationAggregate:
    """Aggregate D1 runs while preserving failures and unavailable metrics."""
    normalized: list[WavelengthValidationRun] = []
    for run in runs:
        if isinstance(run, WavelengthValidationRun):
            normalized.append(run)
        elif isinstance(run, Mapping):
            normalized.append(WavelengthValidationRun.from_mapping(run))
        else:
            raise TypeError("runs must contain validation runs or mappings")

    status_counts: dict[str, int] = {}
    model_summaries: dict[str, dict[str, Any]] = {}
    failure_by_stage: dict[str, int] = {}
    failure_by_code: dict[str, int] = {}
    for run in normalized:
        technical = (
            run.status.technical_outcome.value
            if run.status is not None
            else "not_evaluated"
        )
        status_counts[technical] = status_counts.get(technical, 0) + 1
        model_summary = model_summaries.setdefault(
            run.model,
            {
                "n_runs": 0,
                "n_completed": 0,
                "n_failed": 0,
                "n_eligible": 0,
            },
        )
        model_summary["n_runs"] += 1
        if run.status is not None:
            if run.status.technical_outcome is TechnicalOutcome.FAILED:
                model_summary["n_failed"] += 1
            elif run.status.technical_outcome in {
                TechnicalOutcome.COMPLETED,
                TechnicalOutcome.COMPLETED_WITH_WARNINGS,
                TechnicalOutcome.COMPLETED_WITH_RECOVERY,
            }:
                model_summary["n_completed"] += 1
            if (
                run.status.comparison_eligibility
                is ComparisonEligibility.ELIGIBLE
            ):
                model_summary["n_eligible"] += 1
        if run.failure is not None:
            stage = run.failure.stage.value
            code = run.failure.failure_code
            failure_by_stage[stage] = failure_by_stage.get(stage, 0) + 1
            failure_by_code[code] = failure_by_code.get(code, 0) + 1

    metric_summaries = _metric_summaries_from_runs(normalized)
    matched_runs = [
        run
        for run in normalized
        if run.model == str(run.extra_fields.get("generating_model") or "")
    ]
    matched_metric_summaries = _metric_summaries_from_runs(matched_runs)
    matched_gate_metric_summaries = _metric_summaries_from_runs(
        matched_runs, strict_recovery_only=True
    )
    matched_failed_runs = sum(
        run.status is not None
        and run.status.technical_outcome is TechnicalOutcome.FAILED
        for run in matched_runs
    )
    scenario_population_summaries = _scenario_population_summaries(
        matched_runs
    )
    gate_summary = _aggregate_gate_summary(
        matched_gate_metric_summaries,
        n_failed_runs=matched_failed_runs,
        n_matched_runs=len(matched_runs),
        scenario_population_summaries=scenario_population_summaries,
    )

    return WavelengthValidationAggregate(
        aggregate_id=aggregate_id,
        phase=WavelengthValidationPhase.D1_SYNTHETIC_RECOVERY,
        scenario_ids=tuple(
            dict.fromkeys(run.scenario_id for run in normalized)
        ),
        run_ids=tuple(run.run_id for run in normalized),
        n_runs=len(normalized),
        status_counts=status_counts,
        metric_summaries=metric_summaries,
        model_summaries=model_summaries,
        failure_summaries={
            "by_stage": failure_by_stage,
            "by_code": failure_by_code,
            "n_failures": sum(failure_by_stage.values()),
        },
        notes=(
            "Aggregate recovery statistics are validation diagnostics, not "
            "automatic model-selection evidence.",
        ),
        automatic_model_selection_applied=False,
        extra_fields={
            "synthetic_recovery_schema_version": (
                SYNTHETIC_WAVELENGTH_RECOVERY_SCHEMA_VERSION
            ),
            "matched_truth_run_ids": [run.run_id for run in matched_runs],
            "matched_truth_metric_summaries": matched_metric_summaries,
            "matched_truth_gate_metric_summaries": (
                matched_gate_metric_summaries
            ),
            "scenario_population_summaries": (
                scenario_population_summaries
            ),
            "d1_gate_summary": gate_summary,
        },
    )


@dataclass(frozen=True)
class SyntheticWavelengthRecoveryReport:
    """Typed collection of D1 runs and their failure-aware aggregate."""

    report_id: str
    runs: tuple[WavelengthValidationRun, ...]
    aggregate: WavelengthValidationAggregate
    schema_version: str = SYNTHETIC_WAVELENGTH_RECOVERY_SCHEMA_VERSION
    notes: tuple[str, ...] = ()
    advisory_only: bool = True
    automatic_model_selection_applied: bool = False
    extra_fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        report_id = str(self.report_id or "").strip()
        if not report_id:
            raise ValueError("report_id must be non-empty.")
        object.__setattr__(self, "report_id", report_id)
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
        if self.aggregate.phase is not WavelengthValidationPhase.D1_SYNTHETIC_RECOVERY:
            raise ValueError("Synthetic recovery reports require a D1 aggregate.")
        if self.aggregate.n_runs != len(self.runs):
            raise ValueError("aggregate.n_runs must match the number of runs.")
        object.__setattr__(self, "schema_version", str(self.schema_version))
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))
        if self.advisory_only is not True:
            raise ValueError("Synthetic recovery reports must remain advisory-only.")
        if self.automatic_model_selection_applied is not False:
            raise ValueError("Synthetic recovery reports cannot select a model.")
        object.__setattr__(
            self,
            "extra_fields",
            {str(key): _json_safe(item) for key, item in self.extra_fields.items()},
        )

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> SyntheticWavelengthRecoveryReport:
        """Reconstruct a report from its JSON-safe mapping."""
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        known = {
            "schema_version",
            "report_id",
            "runs",
            "aggregate",
            "notes",
            "advisory_only",
            "automatic_model_selection_applied",
            "extra_fields",
        }
        extras = dict(payload.get("extra_fields") or {})
        for key, value in payload.items():
            if key not in known:
                extras[str(key)] = _json_safe(value)
        return cls(
            report_id=str(payload.get("report_id") or ""),
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
                or SYNTHETIC_WAVELENGTH_RECOVERY_SCHEMA_VERSION
            ),
            notes=tuple(payload.get("notes") or ()),
            advisory_only=payload.get("advisory_only", True),
            automatic_model_selection_applied=payload.get(
                "automatic_model_selection_applied", False
            ),
            extra_fields=extras,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-safe report envelope."""
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "runs": [run.to_dict() for run in self.runs],
            "aggregate": self.aggregate.to_dict(),
            "notes": list(self.notes),
            "advisory_only": True,
            "automatic_model_selection_applied": False,
            "extra_fields": _json_safe(self.extra_fields),
        }


def run_synthetic_wavelength_recovery_matrix(
    cases: Sequence[SyntheticWavelengthValidationCase | Mapping[str, Any]],
    *,
    models: Sequence[str] | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
    per_model_fit_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    thresholds: SyntheticWavelengthRecoveryThresholds | Mapping[str, Any] | None = None,
    lightcurve_factory: Callable[[SyntheticWavelengthValidationCase], Any] | None = None,
    fit_runner: Callable[[Any, Mapping[str, Any], SyntheticWavelengthValidationCase], Any]
    | None = None,
    stop_on_error: bool = False,
    report_id: str = "d1-synthetic-wavelength-recovery",
) -> SyntheticWavelengthRecoveryReport:
    """Execute a failure-aware case-by-model D1 recovery matrix."""
    normalized_cases = tuple(_as_case(case) for case in cases)
    if not normalized_cases:
        raise ValueError("cases must contain at least one synthetic case.")
    resolved_models = tuple(models or SUPPORTED_SYNTHETIC_RECOVERY_MODELS)
    if not resolved_models:
        raise ValueError("models must contain at least one model.")
    unsupported = [
        model for model in resolved_models
        if model not in SUPPORTED_SYNTHETIC_RECOVERY_MODELS
    ]
    if unsupported:
        raise ValueError(
            "Unsupported synthetic recovery model(s): " + ", ".join(unsupported)
        )
    per_model_fit_kwargs = dict(per_model_fit_kwargs or {})
    runs = []
    for case in normalized_cases:
        for model in resolved_models:
            model_kwargs = dict(fit_kwargs or {})
            model_kwargs.update(dict(per_model_fit_kwargs.get(model) or {}))
            runs.append(
                run_synthetic_wavelength_recovery(
                    case,
                    model,
                    fit_kwargs=model_kwargs,
                    thresholds=thresholds,
                    lightcurve_factory=lightcurve_factory,
                    fit_runner=fit_runner,
                    stop_on_error=stop_on_error,
                )
            )
    aggregate = aggregate_synthetic_wavelength_recovery_runs(
        runs, aggregate_id=report_id
    )
    return SyntheticWavelengthRecoveryReport(
        report_id=report_id,
        runs=tuple(runs),
        aggregate=aggregate,
        notes=(
            "Candidate completion and metric summaries are descriptive D1 "
            "validation outputs; no winner is installed.",
        ),
    )
