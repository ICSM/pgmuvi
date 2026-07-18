"""Wavelength-domain summaries for parameter estimation.

The routines in this module summarize the wavelength sampling and robust
per-band flux distributions of a multiband light curve.  They do not mutate a
GP model and they do not claim that a particular wavelength model is selected.
The resulting diagnostics are intended to provide a shared, auditable input to
later wavelength-kernel and wavelength-mean initialization work.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pgmuvi.parameter_context import (
    BandDiagnostics,
    WAVELENGTH_ESTIMATION_SCHEMA_VERSION,
    WavelengthEstimationDiagnostics,
    WavelengthMeanEstimationDiagnostics,
)
from pgmuvi.preprocess.quality import robust_scale


def _finite_array(values: Any) -> np.ndarray:
    """Return a flattened float array without changing its length."""
    return np.asarray(values, dtype=float).reshape(-1)


def _band_label_for_wavelength(wavelength: float) -> str:
    """Return a stable fallback label for one numeric wavelength."""
    return f"wavelength={wavelength:.17g}"


def _safe_ratio(values: list[float], *, absolute: bool = False) -> float | None:
    """Return max/min for finite positive values."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if absolute:
        arr = np.abs(arr)

    arr = arr[arr > 0.0]
    if arr.size < 2:
        return None

    return float(np.max(arr) / np.min(arr))


def _monotonicity_class(values: list[float]) -> str:
    """Classify a finite sequence without fitting a wavelength model."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size < 2:
        return "unavailable"

    scale = max(float(np.max(np.abs(arr))), 1.0)
    tolerance = 1.0e-10 * scale
    differences = np.diff(arr)
    positive = differences > tolerance
    negative = differences < -tolerance

    if not np.any(positive) and not np.any(negative):
        return "approximately_constant"
    if np.all(~negative) and np.any(positive):
        return "non_decreasing"
    if np.all(~positive) and np.any(negative):
        return "non_increasing"
    return "non_monotonic"


def _noise_corrected_scale(scale: float, median_uncertainty: float | None) -> float:
    """Subtract a representative measurement-noise variance in quadrature."""
    if median_uncertainty is None or not math.isfinite(median_uncertainty):
        return scale

    return float(math.sqrt(max(scale * scale - median_uncertainty**2, 0.0)))


def _build_band_diagnostics(
    *,
    band: str,
    wavelengths: np.ndarray,
    fluxes: np.ndarray,
    uncertainties: np.ndarray | None,
    min_points_per_band: int,
) -> BandDiagnostics:
    """Build robust physical-space summaries for one band."""
    finite = np.isfinite(wavelengths) & np.isfinite(fluxes)
    wavelength_values = wavelengths[finite]
    flux_values = fluxes[finite]

    n_points = int(flux_values.size)
    representative_wavelength = (
        float(np.median(wavelength_values)) if wavelength_values.size else None
    )
    wavelength_spread = (
        float(np.max(wavelength_values) - np.min(wavelength_values))
        if wavelength_values.size
        else None
    )

    if uncertainties is not None:
        uncertainty_values = uncertainties[finite]
        uncertainty_values = uncertainty_values[
            np.isfinite(uncertainty_values) & (uncertainty_values > 0.0)
        ]
    else:
        uncertainty_values = np.asarray([], dtype=float)

    median_uncertainty = (
        float(np.median(uncertainty_values)) if uncertainty_values.size else None
    )
    uncertainty_percentiles: dict[float, float] = {}
    if uncertainty_values.size:
        uq = np.percentile(uncertainty_values, [5.0, 50.0, 95.0])
        uncertainty_percentiles = {
            5.0: float(uq[0]),
            50.0: float(uq[1]),
            95.0: float(uq[2]),
        }

    if flux_values.size:
        quantile_levels = [2.5, 5.0, 10.0, 50.0, 90.0, 95.0, 97.5]
        q = np.percentile(flux_values, quantile_levels)
        flux_percentiles = {
            level: float(value)
            for level, value in zip(quantile_levels, q, strict=True)
        }
        median_flux = flux_percentiles[50.0]
        mad_flux = float(np.median(np.abs(flux_values - median_flux)))
        robust_scatter = float(robust_scale(flux_values))
        amplitude_02_97 = 0.5 * (
            flux_percentiles[97.5] - flux_percentiles[2.5]
        )
        amplitude_05_95 = 0.5 * (
            flux_percentiles[95.0] - flux_percentiles[5.0]
        )
        amplitude_10_90 = 0.5 * (
            flux_percentiles[90.0] - flux_percentiles[10.0]
        )
    else:
        flux_percentiles = {}
        median_flux = None
        mad_flux = None
        robust_scatter = None
        amplitude_02_97 = None
        amplitude_05_95 = None
        amplitude_10_90 = None

    fractional_denominator = (
        abs(median_flux)
        if median_flux is not None and math.isfinite(median_flux) and median_flux != 0.0
        else None
    )

    def fractional(value: float | None) -> float | None:
        if value is None or fractional_denominator is None:
            return None
        return float(value / fractional_denominator)

    def corrected(value: float | None) -> float | None:
        if value is None:
            return None
        return _noise_corrected_scale(value, median_uncertainty)

    wavelength_consistent = False
    if representative_wavelength is not None and wavelength_spread is not None:
        wavelength_tolerance = 1.0e-10 * max(
            abs(representative_wavelength),
            1.0,
        )
        wavelength_consistent = wavelength_spread <= wavelength_tolerance

    usable = (
        n_points >= min_points_per_band
        and representative_wavelength is not None
        and math.isfinite(representative_wavelength)
        and wavelength_consistent
    )

    exclusion_reason = None
    if not usable:
        if representative_wavelength is None:
            exclusion_reason = "finite_wavelength_unavailable"
        elif not wavelength_consistent:
            exclusion_reason = "inconsistent_wavelength_within_band"
        elif n_points < min_points_per_band:
            exclusion_reason = "insufficient_finite_flux_points"

    return BandDiagnostics(
        band=band,
        wavelength=representative_wavelength,
        median_flux=median_flux,
        mad_flux=mad_flux,
        flux_percentiles=flux_percentiles,
        n_points=n_points,
        median_uncertainty=median_uncertainty,
        uncertainty_percentiles=uncertainty_percentiles,
        robust_scatter=robust_scatter,
        raw_half_amplitude_q02_5_q97_5=amplitude_02_97,
        raw_half_amplitude_q05_q95=amplitude_05_95,
        raw_half_amplitude_q10_q90=amplitude_10_90,
        fractional_half_amplitude_q02_5_q97_5=fractional(amplitude_02_97),
        fractional_half_amplitude_q05_q95=fractional(amplitude_05_95),
        fractional_half_amplitude_q10_q90=fractional(amplitude_10_90),
        noise_corrected_robust_scatter=corrected(robust_scatter),
        noise_corrected_half_amplitude_q02_5_q97_5=corrected(amplitude_02_97),
        noise_corrected_half_amplitude_q05_q95=corrected(amplitude_05_95),
        noise_corrected_half_amplitude_q10_q90=corrected(amplitude_10_90),
        metadata={
            "usable_for_wavelength_estimation": usable,
            "exclusion_reason": exclusion_reason,
            "wavelength_spread_within_band": wavelength_spread,
            "wavelength_consistent_within_band": wavelength_consistent,
            "n_positive_finite_uncertainties": int(uncertainty_values.size),
        },
    )


def _coverage_class(n_bands: int) -> str:
    """Return a descriptive sampling class based only on usable band count."""
    if n_bands < 2:
        return "unresolved"
    if n_bands == 2:
        return "two_band"
    if n_bands <= 4:
        return "sparse"
    if n_bands <= 8:
        return "moderate"
    return "dense"


def _lengthscale_recommendation(
    wavelengths: np.ndarray,
) -> tuple[float | None, tuple[float, float] | None, str | None]:
    """Derive an auditable raw-coordinate length-scale recommendation."""
    if wavelengths.size < 2:
        return None, None, None

    spacings = np.diff(wavelengths)
    spacings = spacings[np.isfinite(spacings) & (spacings > 0.0)]
    if spacings.size == 0:
        return None, None, None

    span = float(wavelengths[-1] - wavelengths[0])
    if not math.isfinite(span) or span <= 0.0:
        return None, None, None

    minimum_spacing = float(np.min(spacings))
    median_spacing = float(np.median(spacings))
    largest_gap = float(np.max(spacings))

    lower = max(
        0.25 * minimum_spacing,
        np.finfo(float).eps * max(float(np.max(np.abs(wavelengths))), 1.0),
    )
    initial = float(math.sqrt(median_spacing * span))
    upper = max(5.0 * span, 4.0 * largest_gap, 2.0 * initial)
    initial = min(max(initial, lower), upper)

    return (
        initial,
        (float(lower), float(upper)),
        "geometric_mean_of_median_spacing_and_total_span",
    )


def build_wavelength_estimation_context(
    wavelengths: Any,
    fluxes: Any,
    uncertainties: Any | None = None,
    band_labels: Any | None = None,
    *,
    min_points_per_band: int = 3,
) -> tuple[WavelengthEstimationDiagnostics, dict[str, BandDiagnostics]]:
    """Build wavelength sampling and robust per-band diagnostics.

    Parameters
    ----------
    wavelengths
        Numeric wavelength coordinate for each observation.
    fluxes
        Flux or magnitude value for each observation.  The values are summarized
        in their supplied linear coordinate; no log-flux transformation is used.
    uncertainties
        Optional measurement uncertainties.  Only finite positive values enter
        the uncertainty and noise-corrected summaries.
    band_labels
        Optional row-wise labels.  When absent, exact numeric wavelengths define
        the groups.
    min_points_per_band
        Minimum number of finite wavelength/flux pairs required for a band to
        enter cross-wavelength trend and length-scale summaries.

    Returns
    -------
    tuple
        Two-element tuple containing an immutable wavelength-level summary and
        per-band diagnostics keyed by label.
    """
    if min_points_per_band < 1:
        raise ValueError("min_points_per_band must be at least 1.")

    wavelength_values = _finite_array(wavelengths)
    flux_values = _finite_array(fluxes)
    if wavelength_values.size != flux_values.size:
        raise ValueError("wavelengths and fluxes must have the same length.")

    if uncertainties is not None:
        uncertainty_values = _finite_array(uncertainties)
        if uncertainty_values.size != flux_values.size:
            raise ValueError(
                "uncertainties must have the same length as wavelengths and fluxes."
            )
    else:
        uncertainty_values = None

    if band_labels is None:
        labels = np.asarray(
            [
                _band_label_for_wavelength(value)
                if math.isfinite(value)
                else "wavelength=nonfinite"
                for value in wavelength_values
            ],
            dtype=str,
        )
    else:
        labels = np.asarray(band_labels, dtype=str).reshape(-1)
        if labels.size != flux_values.size:
            raise ValueError(
                "band_labels must have the same length as wavelengths and fluxes."
            )

    band_diagnostics: dict[str, BandDiagnostics] = {}
    for label in dict.fromkeys(labels.tolist()):
        mask = labels == label
        band_diagnostics[label] = _build_band_diagnostics(
            band=label,
            wavelengths=wavelength_values[mask],
            fluxes=flux_values[mask],
            uncertainties=(
                uncertainty_values[mask]
                if uncertainty_values is not None
                else None
            ),
            min_points_per_band=min_points_per_band,
        )

    usable = [
        item
        for item in band_diagnostics.values()
        if item.metadata.get("usable_for_wavelength_estimation")
    ]
    usable.sort(key=lambda item: float(item.wavelength))

    usable_wavelengths = np.asarray(
        [float(item.wavelength) for item in usable],
        dtype=float,
    )
    distinct_wavelengths = np.unique(usable_wavelengths)
    spacings = np.diff(distinct_wavelengths)
    positive_spacings = spacings[
        np.isfinite(spacings) & (spacings > 0.0)
    ]

    span = (
        float(distinct_wavelengths[-1] - distinct_wavelengths[0])
        if distinct_wavelengths.size >= 2
        else None
    )
    minimum_spacing = (
        float(np.min(positive_spacings)) if positive_spacings.size else None
    )
    median_spacing = (
        float(np.median(positive_spacings)) if positive_spacings.size else None
    )
    maximum_spacing = (
        float(np.max(positive_spacings)) if positive_spacings.size else None
    )
    gap_ratio = (
        float(maximum_spacing / median_spacing)
        if maximum_spacing is not None
        and median_spacing is not None
        and median_spacing > 0.0
        else None
    )
    spacing_ratio = (
        float(maximum_spacing / minimum_spacing)
        if maximum_spacing is not None
        and minimum_spacing is not None
        and minimum_spacing > 0.0
        else None
    )

    median_fluxes = [
        float(item.median_flux)
        for item in usable
        if item.median_flux is not None and math.isfinite(item.median_flux)
    ]
    amplitudes = [
        float(item.raw_half_amplitude_q05_q95)
        for item in usable
        if item.raw_half_amplitude_q05_q95 is not None
        and math.isfinite(item.raw_half_amplitude_q05_q95)
    ]
    scatters = [
        float(item.robust_scatter)
        for item in usable
        if item.robust_scatter is not None and math.isfinite(item.robust_scatter)
    ]

    initial, bounds, method = _lengthscale_recommendation(distinct_wavelengths)
    excluded = tuple(
        label
        for label, item in band_diagnostics.items()
        if not item.metadata.get("usable_for_wavelength_estimation")
    )

    diagnostics = WavelengthEstimationDiagnostics(
        available=bool(distinct_wavelengths.size >= 2 and span is not None),
        n_observations=int(
            np.sum(np.isfinite(wavelength_values) & np.isfinite(flux_values))
        ),
        n_distinct_wavelengths=int(distinct_wavelengths.size),
        n_usable_bands=len(usable),
        min_points_per_band=min_points_per_band,
        wavelengths=tuple(float(value) for value in distinct_wavelengths),
        wavelength_min=(
            float(distinct_wavelengths[0]) if distinct_wavelengths.size else None
        ),
        wavelength_max=(
            float(distinct_wavelengths[-1]) if distinct_wavelengths.size else None
        ),
        wavelength_span=span,
        adjacent_spacings=tuple(float(value) for value in positive_spacings),
        minimum_adjacent_spacing=minimum_spacing,
        median_adjacent_spacing=median_spacing,
        maximum_adjacent_spacing=maximum_spacing,
        largest_gap=maximum_spacing,
        largest_gap_ratio_to_median_spacing=gap_ratio,
        spacing_ratio_max_to_min=spacing_ratio,
        coverage_class=_coverage_class(int(distinct_wavelengths.size)),
        median_flux_monotonicity_class=_monotonicity_class(median_fluxes),
        amplitude_monotonicity_class=_monotonicity_class(amplitudes),
        scatter_monotonicity_class=_monotonicity_class(scatters),
        median_flux_ratio_max_to_min_abs=_safe_ratio(
            median_fluxes,
            absolute=True,
        ),
        amplitude_ratio_max_to_min=_safe_ratio(amplitudes),
        scatter_ratio_max_to_min=_safe_ratio(scatters),
        recommended_lengthscale_initial=initial,
        recommended_lengthscale_bounds=bounds,
        recommendation_method=method,
        model_coordinate_space="raw_input",
        model_recommended_lengthscale_initial=initial,
        model_recommended_lengthscale_bounds=bounds,
        lengthscale_transform_status="identity",
        lengthscale_transform_name=None,
        excluded_bands=excluded,
        metadata={
            "uses_log_flux": False,
            "lengthscale_units": "raw_wavelength_coordinate",
            "model_lengthscale_units": "raw_wavelength_coordinate",
            "lengthscale_applied_to_models": False,
            "trend_order": "ascending_wavelength",
        },
    )

    return diagnostics, band_diagnostics


def _wavelength_mean_band_points(
    raw_wavelengths: Any,
    model_wavelengths: Any,
    model_fluxes: Any,
    band_labels: Any | None,
    *,
    min_points_per_band: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Return robust per-band points for wavelength-mean estimation."""
    raw = _finite_array(raw_wavelengths)
    model = _finite_array(model_wavelengths)
    flux = _finite_array(model_fluxes)
    if raw.size != model.size or raw.size != flux.size:
        raise ValueError(
            "raw_wavelengths, model_wavelengths, and model_fluxes must "
            "have the same length."
        )

    if band_labels is None:
        labels = np.asarray(
            [
                _band_label_for_wavelength(value)
                if math.isfinite(value)
                else "wavelength=nonfinite"
                for value in raw
            ],
            dtype=str,
        )
    else:
        labels = np.asarray(band_labels, dtype=str).reshape(-1)
        if labels.size != raw.size:
            raise ValueError(
                "band_labels must have the same length as wavelength inputs."
            )

    rows: list[tuple[float, float, float, str]] = []
    excluded: list[str] = []
    for label in dict.fromkeys(labels.tolist()):
        mask = labels == label
        finite = mask & np.isfinite(raw) & np.isfinite(model) & np.isfinite(flux)
        raw_values = raw[finite]
        model_values = model[finite]
        flux_values = flux[finite]
        if flux_values.size < min_points_per_band:
            excluded.append(label)
            continue
        raw_median = float(np.median(raw_values))
        model_median = float(np.median(model_values))
        raw_spread = float(np.max(raw_values) - np.min(raw_values))
        model_spread = float(np.max(model_values) - np.min(model_values))
        raw_tol = 1.0e-10 * max(abs(raw_median), 1.0)
        model_tol = 1.0e-10 * max(abs(model_median), 1.0)
        if (
            raw_median <= 0.0
            or raw_spread > raw_tol
            or model_spread > model_tol
        ):
            excluded.append(label)
            continue
        rows.append(
            (
                raw_median,
                model_median,
                float(np.median(flux_values)),
                label,
            )
        )

    rows.sort(key=lambda item: item[0])
    return (
        np.asarray([item[0] for item in rows], dtype=float),
        np.asarray([item[1] for item in rows], dtype=float),
        np.asarray([item[2] for item in rows], dtype=float),
        tuple(excluded),
    )


def _wavelength_mean_flux_interval(values: np.ndarray) -> tuple[float, float, float]:
    """Return a padded finite target interval and a robust positive scale."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (-1.0, 1.0, 1.0)
    low = float(np.min(finite))
    high = float(np.max(finite))
    span = high - low
    floor = max(float(np.max(np.abs(finite))) * 1.0e-6, 1.0e-8)
    span = max(span, floor)
    return (low - 2.0 * span, high + 2.0 * span, span)


def _quadratic_mean_recommendation(
    model_wavelengths: np.ndarray,
    fluxes: np.ndarray,
) -> dict[str, Any]:
    """Fit a stable quadratic recommendation in the GP wavelength coordinate."""
    if model_wavelengths.size < 2:
        return {"available": False, "reason": "fewer_than_two_usable_bands"}
    if float(np.ptp(model_wavelengths)) <= 0.0:
        return {"available": False, "reason": "zero_model_wavelength_span"}

    degree = 2 if model_wavelengths.size >= 3 else 1
    coeff = np.polyfit(model_wavelengths, fluxes, deg=degree)
    if degree == 1:
        slope, bias = coeff
        weights = np.asarray([slope, 0.0], dtype=float)
    else:
        quad, slope, bias = coeff
        weights = np.asarray([slope, quad], dtype=float)
    predicted = bias + weights[0] * model_wavelengths
    predicted = predicted + weights[1] * model_wavelengths**2
    rmse = float(np.sqrt(np.mean((fluxes - predicted) ** 2)))

    offset_low, offset_high, flux_span = _wavelength_mean_flux_interval(fluxes)
    x_span = max(float(np.ptp(model_wavelengths)), 1.0e-8)
    slope_scale = max(abs(float(weights[0])), flux_span / x_span, 1.0e-8)
    quad_scale = max(abs(float(weights[1])), flux_span / (x_span**2), 1.0e-8)
    lower = [-5.0 * slope_scale, -5.0 * quad_scale]
    upper = [5.0 * slope_scale, 5.0 * quad_scale]

    return {
        "available": True,
        "coordinate_basis": "model_wavelength_and_model_flux",
        "initial_values": {
            "mean_module.bias": float(bias),
            "mean_module.weights": [float(weights[0]), float(weights[1])],
        },
        "constraints": {
            "mean_module.bias": [offset_low, offset_high],
            "mean_module.weights": [lower, upper],
        },
        "fit_rmse": rmse,
        "fit_degree": degree,
        "reason": None,
    }


def _power_law_mean_recommendation(
    raw_wavelengths: np.ndarray,
    fluxes: np.ndarray,
) -> dict[str, Any]:
    """Fit offset + weight * wavelength**exponent on robust band medians."""
    if raw_wavelengths.size < 2:
        return {"available": False, "reason": "fewer_than_two_usable_bands"}
    if np.any(raw_wavelengths <= 0.0):
        return {"available": False, "reason": "nonpositive_physical_wavelength"}

    flux_scale = max(float(np.max(np.abs(fluxes))), 1.0)
    approximately_constant = float(np.ptp(fluxes)) <= 1.0e-10 * flux_scale
    if raw_wavelengths.size == 2 or approximately_constant:
        candidates = np.asarray([-2.0])
        exponent_estimation = (
            "fixed_default_flat_trend"
            if approximately_constant
            else "fixed_default_two_band"
        )
    else:
        candidates = np.concatenate(
            [
                np.linspace(-10.0, -0.05, 240),
                np.linspace(0.05, 10.0, 240),
            ]
        )
        exponent_estimation = "grid_profile_fit"
    best: tuple[float, float, float, float] | None = None
    for exponent in candidates:
        basis = np.power(raw_wavelengths, exponent)
        if not np.all(np.isfinite(basis)) or float(np.ptp(basis)) <= 0.0:
            continue
        design = np.column_stack([np.ones_like(basis), basis])
        offset, weight = np.linalg.lstsq(design, fluxes, rcond=None)[0]
        predicted = offset + weight * basis
        mse = float(np.mean((fluxes - predicted) ** 2))
        if best is None or mse < best[0]:
            best = (mse, float(offset), float(weight), float(exponent))

    if best is None:
        return {"available": False, "reason": "power_law_fit_failed"}

    mse, offset, weight, exponent = best
    offset_low, offset_high, flux_span = _wavelength_mean_flux_interval(fluxes)
    weight_scale = max(
        abs(weight),
        flux_span,
        0.1 * float(np.max(np.abs(fluxes))),
        1.0e-3,
    )
    return {
        "available": True,
        "coordinate_basis": "physical_wavelength_and_model_flux",
        "initial_values": {
            "mean_module.offset": offset,
            "mean_module.weight": weight,
            "mean_module.exponent": exponent,
        },
        "constraints": {
            "mean_module.offset": [offset_low, offset_high],
            "mean_module.weight": [-5.0 * weight_scale, 5.0 * weight_scale],
            "mean_module.exponent": [-10.0, 10.0],
        },
        "fit_rmse": float(math.sqrt(mse)),
        "exponent_estimation": exponent_estimation,
        "reason": None,
    }


def _dust_mean_recommendation(
    raw_wavelengths: np.ndarray,
    fluxes: np.ndarray,
) -> dict[str, Any]:
    """Fit a coarse physical dust-attenuation recommendation."""
    if raw_wavelengths.size < 3:
        return {"available": False, "reason": "fewer_than_three_usable_bands"}
    if np.any(raw_wavelengths <= 0.0):
        return {"available": False, "reason": "nonpositive_physical_wavelength"}
    flux_scale = max(float(np.max(np.abs(fluxes))), 1.0)
    if float(np.ptp(fluxes)) <= 1.0e-10 * flux_scale:
        return {"available": False, "reason": "wavelength_mean_not_resolved"}

    alphas = np.geomspace(0.1, 10.0, 48)
    taus = np.geomspace(1.0e-3, 1.0e3, 72)
    best: tuple[float, float, float, float, float] | None = None
    for alpha in alphas:
        wavelength_term = np.power(raw_wavelengths, -alpha)
        for tau in taus:
            attenuation = np.exp(-tau * wavelength_term)
            if float(np.ptp(attenuation)) <= 1.0e-12:
                continue
            design = np.column_stack([np.ones_like(attenuation), attenuation])
            offset, amplitude = np.linalg.lstsq(design, fluxes, rcond=None)[0]
            if not math.isfinite(float(amplitude)) or amplitude <= 0.0:
                continue
            predicted = offset + amplitude * attenuation
            mse = float(np.mean((fluxes - predicted) ** 2))
            if best is None or mse < best[0]:
                best = (
                    mse,
                    float(offset),
                    float(amplitude),
                    float(tau),
                    float(alpha),
                )

    if best is None:
        return {"available": False, "reason": "dust_fit_failed"}

    mse, offset, amplitude, tau, alpha = best
    offset_low, offset_high, flux_span = _wavelength_mean_flux_interval(fluxes)
    amplitude_floor = max(1.0e-8, 1.0e-6 * flux_span)
    amplitude_high = max(5.0 * amplitude, 5.0 * flux_span, 10.0 * amplitude_floor)
    return {
        "available": True,
        "coordinate_basis": "physical_wavelength_and_model_flux",
        "initial_values": {
            "mean_module.offset": offset,
            "mean_module.log_amplitude": amplitude,
            "mean_module.log_tau": tau,
            "mean_module.log_alpha": alpha,
        },
        "constraints": {
            "mean_module.offset": [offset_low, offset_high],
            "mean_module.log_amplitude": [amplitude_floor, amplitude_high],
            "mean_module.log_tau": [1.0e-3, 1.0e3],
            "mean_module.log_alpha": [0.1, 10.0],
        },
        "fit_rmse": float(math.sqrt(mse)),
        "reason": None,
    }


def build_wavelength_mean_estimation_context(
    raw_wavelengths: Any,
    model_wavelengths: Any,
    model_fluxes: Any,
    band_labels: Any | None = None,
    *,
    min_points_per_band: int = 3,
) -> WavelengthMeanEstimationDiagnostics:
    """Build model-ready wavelength-mean estimates from robust band medians.

    Dust and power-law recommendations use raw positive wavelength values but
    fluxes in the transformed training-target coordinate.  The quadratic model
    uses the transformed wavelength coordinate because its coefficients are not
    assigned a physical wavelength interpretation.
    """
    raw, model, flux, excluded = _wavelength_mean_band_points(
        raw_wavelengths,
        model_wavelengths,
        model_fluxes,
        band_labels,
        min_points_per_band=min_points_per_band,
    )
    recommendations = {
        "2DWavelengthDependent": _quadratic_mean_recommendation(model, flux),
        "2DPowerLawMean": _power_law_mean_recommendation(raw, flux),
        "2DDustMean": _dust_mean_recommendation(raw, flux),
    }
    warnings = tuple(
        f"{model_name}: {record.get('reason')}"
        for model_name, record in recommendations.items()
        if not record.get("available")
    )
    return WavelengthMeanEstimationDiagnostics(
        available=any(record.get("available") for record in recommendations.values()),
        n_usable_bands=int(raw.size),
        raw_wavelengths=tuple(float(value) for value in raw),
        model_wavelengths=tuple(float(value) for value in model),
        model_median_fluxes=tuple(float(value) for value in flux),
        recommendations=recommendations,
        warnings=warnings,
        metadata={
            "uses_log_flux": False,
            "physical_mean_wavelength_coordinate": "raw_positive_wavelength",
            "quadratic_mean_wavelength_coordinate": "model_input_wavelength",
            "flux_coordinate": "model_training_target",
            "excluded_bands": list(excluded),
            "recommendations_applied_to_models": False,
        },
    )


__all__ = [
    "WAVELENGTH_ESTIMATION_SCHEMA_VERSION",
    "WavelengthEstimationDiagnostics",
    "WavelengthMeanEstimationDiagnostics",
    "build_wavelength_estimation_context",
    "build_wavelength_mean_estimation_context",
]
