"""Dimension-aware estimates for two-dimensional spectral-mixture kernels.

The full non-separable ``2D`` model stores temporal-frequency and
wavelength-frequency parameters in the last axis of the same GPyTorch tensor.
This module derives separate values and bounds for those coordinates while
retaining the native :class:`gpytorch.kernels.SpectralMixtureKernel`
parameterization.  Bounds have shape ``(1, 1, 2)`` and therefore broadcast over
mixture components without collapsing the two ARD dimensions to one scalar
interval.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


SPECTRAL_MIXTURE_ARD_SCHEMA_VERSION = "pgmuvi-spectral-mixture-ard-v1"
ARD_COORDINATE_ORDER = (
    "temporal_frequency",
    "wavelength_frequency",
)


def _as_two_dimensional_array(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f"{name} must have shape (n_observations, >=2).")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one observation.")
    return array[:, :2]


def _sampling_summary(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    unique = np.unique(finite)
    unique.sort()

    summary: dict[str, Any] = {
        "n_finite": int(finite.size),
        "n_unique": int(unique.size),
        "minimum": None,
        "maximum": None,
        "span": None,
        "minimum_positive_spacing": None,
        "median_positive_spacing": None,
    }
    if unique.size == 0:
        return summary

    summary["minimum"] = float(unique[0])
    summary["maximum"] = float(unique[-1])
    if unique.size == 1:
        summary["span"] = 0.0
        return summary

    span = float(unique[-1] - unique[0])
    positive = np.diff(unique)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    summary["span"] = span
    if positive.size:
        summary["minimum_positive_spacing"] = float(np.min(positive))
        summary["median_positive_spacing"] = float(np.median(positive))
    return summary


def _safe_frequency_bounds(summary: dict[str, Any]) -> tuple[float, float]:
    span = summary.get("span")
    minimum_spacing = summary.get("minimum_positive_spacing")
    if span is None or not math.isfinite(span) or span <= 0:
        raise ValueError("A positive coordinate span is required.")

    lower = 1.0 / span
    upper = None
    if (
        minimum_spacing is not None
        and math.isfinite(minimum_spacing)
        and minimum_spacing > 0
    ):
        upper = 1.0 / (2.0 * minimum_spacing)

    # Two unique samples have a nominal Nyquist frequency below 1/span.  The
    # interval still needs to be valid, so retain an explicit conservative cap.
    if upper is None or not math.isfinite(upper) or upper <= lower:
        upper = 2.0 * lower
    return float(lower), float(upper)


def _trend_requires_nonzero_wavelength_frequency(
    wavelength_diagnostics: Any | None,
) -> bool:
    if wavelength_diagnostics is None:
        return False
    for name in (
        "median_flux_monotonicity_class",
        "amplitude_monotonicity_class",
        "scatter_monotonicity_class",
    ):
        if getattr(wavelength_diagnostics, name, None) == "non_monotonic":
            return True
    return False


def _lengthscale_recommendation(
    wavelength_diagnostics: Any | None,
    *,
    coordinate: str,
    sampling: dict[str, Any],
) -> tuple[float, tuple[float, float], str]:
    if coordinate not in {"raw", "model"}:
        raise ValueError("coordinate must be 'raw' or 'model'.")

    if wavelength_diagnostics is not None:
        if coordinate == "raw":
            initial = getattr(
                wavelength_diagnostics,
                "recommended_lengthscale_initial",
                None,
            )
            bounds = getattr(
                wavelength_diagnostics,
                "recommended_lengthscale_bounds",
                None,
            )
        else:
            initial = getattr(
                wavelength_diagnostics,
                "model_recommended_lengthscale_initial",
                None,
            )
            bounds = getattr(
                wavelength_diagnostics,
                "model_recommended_lengthscale_bounds",
                None,
            )
        if initial is not None and bounds is not None:
            lower, upper = (float(bounds[0]), float(bounds[1]))
            initial = float(initial)
            if (
                math.isfinite(initial)
                and math.isfinite(lower)
                and math.isfinite(upper)
                and 0 < lower < upper
            ):
                return (
                    min(max(initial, lower), upper),
                    (lower, upper),
                    "wavelength_estimation_context",
                )

    span = sampling.get("span")
    spacing = sampling.get("minimum_positive_spacing")
    if span is None or not math.isfinite(span) or span <= 0:
        # A single wavelength carries no resolved wavelength-covariance scale.
        # Keep a broad, explicit fallback for backward compatibility, and mark
        # it as degenerate in provenance rather than calling it data resolved.
        return 1.0, (1.0e-3, 1.0e3), "single_wavelength_degenerate_fallback"

    if spacing is None or not math.isfinite(spacing) or spacing <= 0:
        spacing = span
    lower = max(0.5 * spacing, 0.01 * span, 1.0e-12)
    upper = max(5.0 * span, 2.0 * lower)
    initial = math.sqrt(lower * upper)
    return initial, (lower, upper), "sampling_only_fallback"


def _spectral_scale_from_lengthscale(lengthscale: float) -> float:
    return 1.0 / (2.0 * math.pi * lengthscale)


def _coordinate_estimates(
    inputs: np.ndarray,
    *,
    num_mixtures: int,
    wavelength_diagnostics: Any | None,
    coordinate: str,
) -> dict[str, Any]:
    time_sampling = _sampling_summary(inputs[:, 0])
    wavelength_sampling = _sampling_summary(inputs[:, 1])

    temporal_lower, temporal_upper = _safe_frequency_bounds(time_sampling)
    temporal_values = np.asarray(
        [
            min(
                max(temporal_lower * (index + 1), temporal_lower),
                0.95 * temporal_upper,
            )
            for index in range(num_mixtures)
        ],
        dtype=float,
    )

    wavelength_span = wavelength_sampling.get("span")
    if (
        wavelength_span is None
        or not math.isfinite(wavelength_span)
        or wavelength_span <= 0
    ):
        wavelength_mean_lower = 1.0e-6
        wavelength_mean_upper = 1.0e3
        wavelength_mean_initial = 1.0e-3
        wavelength_mean_method = "single_wavelength_degenerate_fallback"
    else:
        wavelength_mean_lower = max(0.05 / wavelength_span, 1.0e-12)
        minimum_spacing = wavelength_sampling.get("minimum_positive_spacing")
        if (
            minimum_spacing is None
            or not math.isfinite(minimum_spacing)
            or minimum_spacing <= 0
        ):
            minimum_spacing = wavelength_span
        wavelength_mean_upper = 1.0 / (2.0 * minimum_spacing)
        if wavelength_mean_upper <= wavelength_mean_lower:
            wavelength_mean_upper = 2.0 * wavelength_mean_lower
        wavelength_mean_initial = (
            1.0 / wavelength_span
            if _trend_requires_nonzero_wavelength_frequency(
                wavelength_diagnostics
            )
            else 0.1 / wavelength_span
        )
        wavelength_mean_initial = min(
            max(wavelength_mean_initial, wavelength_mean_lower),
            0.95 * wavelength_mean_upper,
        )
        wavelength_mean_method = (
            "non_monotonic_wavelength_structure"
            if _trend_requires_nonzero_wavelength_frequency(
                wavelength_diagnostics
            )
            else "smooth_or_unresolved_wavelength_structure"
        )

    lengthscale_initial, lengthscale_bounds, lengthscale_source = (
        _lengthscale_recommendation(
            wavelength_diagnostics,
            coordinate=coordinate,
            sampling=wavelength_sampling,
        )
    )
    lengthscale_lower, lengthscale_upper = lengthscale_bounds
    wavelength_scale_initial = _spectral_scale_from_lengthscale(
        lengthscale_initial
    )
    wavelength_scale_lower = _spectral_scale_from_lengthscale(
        lengthscale_upper
    )
    wavelength_scale_upper = _spectral_scale_from_lengthscale(
        lengthscale_lower
    )

    temporal_scale_initial = _spectral_scale_from_lengthscale(
        float(time_sampling["span"])
    )
    temporal_scale_lower = max(0.01 * temporal_scale_initial, 1.0e-12)
    temporal_scale_upper = max(
        temporal_upper,
        100.0 * temporal_scale_initial,
        2.0 * temporal_scale_lower,
    )

    means = np.zeros((num_mixtures, 1, 2), dtype=float)
    means[:, 0, 0] = temporal_values
    means[:, 0, 1] = wavelength_mean_initial

    scales = np.zeros((num_mixtures, 1, 2), dtype=float)
    scales[:, 0, 0] = temporal_scale_initial
    scales[:, 0, 1] = wavelength_scale_initial

    return {
        "sampling": {
            "time": time_sampling,
            "wavelength": wavelength_sampling,
        },
        "mixture_means": {
            "initial_value": means.tolist(),
            "constraint_lower": [
                [[temporal_lower, wavelength_mean_lower]]
            ],
            "constraint_upper": [
                [[temporal_upper, wavelength_mean_upper]]
            ],
            "temporal_initialization": "baseline_frequency_sequence",
            "wavelength_initialization": wavelength_mean_method,
        },
        "mixture_scales": {
            "initial_value": scales.tolist(),
            "constraint_lower": [
                [[temporal_scale_lower, wavelength_scale_lower]]
            ],
            "constraint_upper": [
                [[temporal_scale_upper, wavelength_scale_upper]]
            ],
            "temporal_initialization": "inverse_time_span_spectral_scale",
            "wavelength_initialization": (
                "inverse_recommended_wavelength_lengthscale"
            ),
            "wavelength_lengthscale_initial": lengthscale_initial,
            "wavelength_lengthscale_bounds": list(lengthscale_bounds),
            "wavelength_lengthscale_source": lengthscale_source,
            "lengthscale_to_spectral_scale": "1/(2*pi*lengthscale)",
        },
    }


def build_dimension_aware_sm_ard_estimates(
    *,
    raw_inputs: Any,
    model_inputs: Any,
    num_mixtures: int,
    wavelength_diagnostics: Any | None = None,
) -> dict[str, Any]:
    """Build separate temporal and wavelength SM values and bounds.

    Parameters are returned in both raw-input and fitted-model coordinates.
    Model-coordinate arrays are the values consumed by the parameter workflow;
    raw arrays are retained solely as provenance and for raw-space
    initialization helpers.
    """
    if not isinstance(num_mixtures, int) or num_mixtures < 1:
        raise ValueError("num_mixtures must be a positive integer.")

    raw = _as_two_dimensional_array(raw_inputs, name="raw_inputs")
    model = _as_two_dimensional_array(model_inputs, name="model_inputs")
    if raw.shape[0] != model.shape[0]:
        raise ValueError("raw_inputs and model_inputs must have equal row counts.")

    raw_records = _coordinate_estimates(
        raw,
        num_mixtures=num_mixtures,
        wavelength_diagnostics=wavelength_diagnostics,
        coordinate="raw",
    )
    model_records = _coordinate_estimates(
        model,
        num_mixtures=num_mixtures,
        wavelength_diagnostics=wavelength_diagnostics,
        coordinate="model",
    )

    return {
        "schema_version": SPECTRAL_MIXTURE_ARD_SCHEMA_VERSION,
        "available": True,
        "parameterization": "broadcast_tensor_intervals",
        "coordinate_order": list(ARD_COORDINATE_ORDER),
        "ard_index": {
            "temporal_frequency": 0,
            "wavelength_frequency": 1,
        },
        "num_mixtures": num_mixtures,
        "constraint_shape": [1, 1, 2],
        "value_shape": [num_mixtures, 1, 2],
        "raw_coordinate": raw_records,
        "model_coordinate": model_records,
        "wavelength_diagnostics_schema_version": getattr(
            wavelength_diagnostics,
            "schema_version",
            None,
        ),
    }
