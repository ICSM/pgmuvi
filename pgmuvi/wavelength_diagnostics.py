"""Wavelength-dependence diagnostics for multiband light curves.

This module contains cheap, pre-fit diagnostics used to decide what classes
of wavelength-dependent GP models are worth testing.  The initial API is
intentionally conservative: it does not run GP fitting and it does not try to
choose a final model.  It builds a band-by-band table that later model-
selection stages can reuse.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - pgmuvi normally depends on torch
    torch = None

from pgmuvi.preprocess.quality import assess_sampling_quality, robust_scale
from pgmuvi.preprocess.variability import is_variable


_TWO_PI = 2.0 * np.pi


def _finite_or_none(value: Any) -> float | int | bool | str | None:
    """Return JSON-friendly scalar values, replacing NaN/Inf with None."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, str) or value is None:
        return value
    return value


def _clean_scalar_dict(values: dict[str, Any]) -> dict[str, Any]:
    """Recursively clean scalar values in a dictionary for reports/tests."""
    cleaned = {}
    for key, value in values.items():
        if isinstance(value, dict):
            cleaned[key] = _clean_scalar_dict(value)
        elif isinstance(value, list):
            cleaned[key] = [_finite_or_none(v) for v in value]
        else:
            cleaned[key] = _finite_or_none(value)
    return cleaned


def _flux_summary(y: np.ndarray, yerr: np.ndarray | None) -> dict[str, Any]:
    """Compute robust, model-free flux/amplitude diagnostics for one band."""
    finite = np.isfinite(y)
    yf = np.asarray(y[finite], dtype=float)
    if yf.size == 0:
        return {
            "median_flux": None,
            "mean_flux": None,
            "robust_scatter": None,
            "robust_amplitude_5_95": None,
            "p05_flux": None,
            "p16_flux": None,
            "p50_flux": None,
            "p84_flux": None,
            "p95_flux": None,
            "median_yerr": None,
            "mean_yerr": None,
        }

    p05, p16, p50, p84, p95 = np.percentile(yf, [5.0, 16.0, 50.0, 84.0, 95.0])
    summary = {
        "median_flux": float(p50),
        "mean_flux": float(np.mean(yf)),
        "robust_scatter": float(robust_scale(yf)),
        "robust_amplitude_5_95": float(0.5 * (p95 - p05)),
        "p05_flux": float(p05),
        "p16_flux": float(p16),
        "p50_flux": float(p50),
        "p84_flux": float(p84),
        "p95_flux": float(p95),
        "median_yerr": None,
        "mean_yerr": None,
    }

    if yerr is not None:
        yef = np.asarray(yerr[np.isfinite(yerr) & (yerr > 0)], dtype=float)
        if yef.size > 0:
            summary["median_yerr"] = float(np.median(yef))
            summary["mean_yerr"] = float(np.mean(yef))

    return _clean_scalar_dict(summary)


def _band_labels_for_mask(band: np.ndarray | None, mask: np.ndarray) -> list[str]:
    """Return sorted unique string labels attached to one wavelength value."""
    if band is None:
        return []
    labels = np.asarray(band, dtype=np.str_)[mask]
    labels = labels[np.char.str_len(labels.astype(str)) > 0]
    return sorted({str(label) for label in labels})


def _resolve_frequency(
    *,
    frequency: float | None = None,
    period: float | None = None,
) -> float | None:
    """Resolve a supplied frequency or period into a positive frequency."""
    if frequency is not None and period is not None:
        raise ValueError("Specify only one of frequency or period, not both.")
    if period is not None:
        period = float(period)
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("period must be a positive finite value.")
        return 1.0 / period
    if frequency is not None:
        frequency = float(frequency)
        if not np.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("frequency must be a positive finite value.")
        return frequency
    return None


def _phase_to_lag(phase_radians: float, frequency: float) -> float:
    """Convert cosine-model phase to the nearest signed time lag."""
    lag = phase_radians / (_TWO_PI * frequency)
    period = 1.0 / frequency
    return float(((lag + 0.5 * period) % period) - 0.5 * period)


def _fit_fixed_frequency_sinusoid(
    t: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray | None,
    *,
    frequency: float,
    reference_time: float | None = None,
    min_points: int = 6,
) -> dict[str, Any]:
    """Fit offset + cos + sin terms at a fixed frequency for one band.

    The fitted model is

    ``y(t) = offset + c cos(theta) + s sin(theta)``

    with ``theta = 2 pi frequency (t - reference_time)``.  The returned phase
    uses the equivalent cosine convention

    ``y(t) = offset + amplitude cos(theta - phase)``.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(t) & np.isfinite(y)

    if yerr is not None:
        yerr = np.asarray(yerr, dtype=float)
        finite &= np.isfinite(yerr) & (yerr > 0.0)

    t = t[finite]
    y = y[finite]
    dy = yerr[finite] if yerr is not None else None

    result: dict[str, Any] = {
        "available": False,
        "status": "unavailable",
        "frequency": float(frequency),
        "period": float(1.0 / frequency),
        "reference_time": None,
        "n_points_used": int(t.size),
        "weighted": bool(dy is not None),
        "offset": None,
        "cos_coefficient": None,
        "sin_coefficient": None,
        "amplitude": None,
        "amplitude_uncertainty": None,
        "amplitude_snr": None,
        "peak_to_peak_amplitude": None,
        "fractional_amplitude": None,
        "phase_radians": None,
        "phase_cycles": None,
        "phase_uncertainty_radians": None,
        "lag": None,
        "lag_fraction_of_period": None,
        "lag_uncertainty": None,
        "residual_rms": None,
        "chi2": None,
        "reduced_chi2": None,
        "condition_number": None,
    }

    if t.size < min_points:
        result["status"] = "insufficient_points"
        return result

    if reference_time is None:
        reference_time = float(0.5 * (np.nanmin(t) + np.nanmax(t)))
    else:
        reference_time = float(reference_time)
        if not np.isfinite(reference_time):
            raise ValueError("reference_time must be finite when provided.")
    result["reference_time"] = reference_time

    theta = _TWO_PI * frequency * (t - reference_time)
    design = np.column_stack([np.ones_like(t), np.cos(theta), np.sin(theta)])

    if dy is not None:
        sqrt_weight = 1.0 / dy
        design_w = design * sqrt_weight[:, None]
        y_w = y * sqrt_weight
    else:
        design_w = design
        y_w = y

    try:
        coeffs, _, rank, singular_values = np.linalg.lstsq(design_w, y_w, rcond=None)
    except np.linalg.LinAlgError:
        result["status"] = "linear_solve_failed"
        return result

    if rank < design.shape[1]:
        result["status"] = "rank_deficient"
        return result

    if singular_values.size > 0 and singular_values[-1] > 0.0:
        condition_number = float(singular_values[0] / singular_values[-1])
    else:
        condition_number = None

    offset, cos_coeff, sin_coeff = [float(v) for v in coeffs]
    model = design @ coeffs
    residual = y - model
    dof = int(max(0, t.size - design.shape[1]))
    residual_rms = float(np.sqrt(np.mean(residual**2)))

    if dy is not None:
        chi2 = float(np.sum((residual / dy) ** 2))
        reduced_chi2 = float(chi2 / dof) if dof > 0 else None
        covariance_scale = reduced_chi2 if reduced_chi2 is not None else 1.0
    else:
        chi2 = float(np.sum(residual**2))
        reduced_chi2 = None
        covariance_scale = float(np.sum(residual**2) / dof) if dof > 0 else 1.0

    normal_matrix = design_w.T @ design_w
    covariance = np.linalg.pinv(normal_matrix) * covariance_scale

    amplitude = float(np.hypot(cos_coeff, sin_coeff))
    peak_to_peak = float(2.0 * amplitude)
    fractional_amplitude = None
    if np.isfinite(offset) and abs(offset) > 0.0:
        fractional_amplitude = float(amplitude / abs(offset))

    phase = float(np.arctan2(sin_coeff, cos_coeff))
    phase_cycles = float(phase / _TWO_PI)
    lag = _phase_to_lag(phase, frequency)

    amplitude_uncertainty = None
    phase_uncertainty = None
    lag_uncertainty = None
    amplitude_snr = None
    if amplitude > 0.0 and np.all(np.isfinite(covariance[1:3, 1:3])):
        grad_amp = np.array([cos_coeff / amplitude, sin_coeff / amplitude])
        cov_cs = covariance[1:3, 1:3]
        var_amp = float(grad_amp @ cov_cs @ grad_amp)
        if np.isfinite(var_amp) and var_amp >= 0.0:
            amplitude_uncertainty = float(np.sqrt(var_amp))
            if amplitude_uncertainty > 0.0:
                amplitude_snr = float(amplitude / amplitude_uncertainty)

        grad_phase = np.array([-sin_coeff / amplitude**2, cos_coeff / amplitude**2])
        var_phase = float(grad_phase @ cov_cs @ grad_phase)
        if np.isfinite(var_phase) and var_phase >= 0.0:
            phase_uncertainty = float(np.sqrt(var_phase))
            lag_uncertainty = float(phase_uncertainty / (_TWO_PI * frequency))

    result.update(
        {
            "available": True,
            "status": "ok",
            "offset": offset,
            "cos_coefficient": cos_coeff,
            "sin_coefficient": sin_coeff,
            "amplitude": amplitude,
            "amplitude_uncertainty": amplitude_uncertainty,
            "amplitude_snr": amplitude_snr,
            "peak_to_peak_amplitude": peak_to_peak,
            "fractional_amplitude": fractional_amplitude,
            "phase_radians": phase,
            "phase_cycles": phase_cycles,
            "phase_uncertainty_radians": phase_uncertainty,
            "lag": lag,
            "lag_fraction_of_period": float(lag * frequency),
            "lag_uncertainty": lag_uncertainty,
            "residual_rms": residual_rms,
            "chi2": chi2,
            "reduced_chi2": reduced_chi2,
            "condition_number": condition_number,
        }
    )
    return _clean_scalar_dict(result)


def _amplitude_phase_summary(band_table: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise fixed-frequency amplitude/phase diagnostics across bands."""
    wavelength_values: list[float] = []
    amplitude_values: list[float] = []
    fractional_amplitude_values: list[float] = []
    lag_values: list[float] = []

    for row in band_table:
        periodic = row.get("fixed_frequency_diagnostics", {})
        if periodic.get("status") != "ok":
            continue
        wl = periodic.get("wavelength", row.get("wavelength"))
        amp = periodic.get("amplitude")
        frac_amp = periodic.get("fractional_amplitude")
        lag = periodic.get("lag")
        if wl is not None and amp is not None and amp > 0.0:
            wavelength_values.append(float(wl))
            amplitude_values.append(float(amp))
        if frac_amp is not None and np.isfinite(frac_amp):
            fractional_amplitude_values.append(float(frac_amp))
        if lag is not None and np.isfinite(lag):
            lag_values.append(float(lag))

    summary: dict[str, Any] = {
        "available": bool(len(amplitude_values) > 0),
        "n_bands_with_fixed_frequency_fit": len(amplitude_values),
        "amplitude_min": None,
        "amplitude_max": None,
        "amplitude_ratio_max_to_min": None,
        "amplitude_loglog_slope": None,
        "fractional_amplitude_median": None,
        "fractional_amplitude_scatter": None,
        "lag_min": None,
        "lag_max": None,
        "lag_span": None,
    }

    if amplitude_values:
        amps = np.asarray(amplitude_values, dtype=float)
        summary["amplitude_min"] = float(np.min(amps))
        summary["amplitude_max"] = float(np.max(amps))
        if np.min(amps) > 0.0:
            summary["amplitude_ratio_max_to_min"] = float(np.max(amps) / np.min(amps))

    if len(amplitude_values) >= 2:
        wls = np.asarray(wavelength_values, dtype=float)
        amps = np.asarray(amplitude_values, dtype=float)
        positive = (wls > 0.0) & (amps > 0.0)
        if np.count_nonzero(positive) >= 2:
            slope, _ = np.polyfit(np.log(wls[positive]), np.log(amps[positive]), 1)
            summary["amplitude_loglog_slope"] = float(slope)

    if fractional_amplitude_values:
        frac = np.asarray(fractional_amplitude_values, dtype=float)
        summary["fractional_amplitude_median"] = float(np.median(frac))
        summary["fractional_amplitude_scatter"] = float(robust_scale(frac))

    if lag_values:
        lags = np.asarray(lag_values, dtype=float)
        summary["lag_min"] = float(np.min(lags))
        summary["lag_max"] = float(np.max(lags))
        summary["lag_span"] = float(np.max(lags) - np.min(lags))

    return _clean_scalar_dict(summary)


def _finite_float(value: Any) -> float | None:
    """Return finite float values from report fields, otherwise None."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _fixed_frequency_rows(
    band_table: list[dict[str, Any]],
) -> list[tuple[float, dict[str, Any]]]:
    """Return rows with successful fixed-frequency diagnostics."""
    rows: list[tuple[float, dict[str, Any]]] = []
    for row in band_table:
        wavelength = _finite_float(row.get("wavelength"))
        periodic = row.get("fixed_frequency_diagnostics")
        if wavelength is None or not isinstance(periodic, dict):
            continue
        if periodic.get("status") != "ok":
            continue
        rows.append((wavelength, periodic))
    rows.sort(key=lambda item: item[0])
    return rows


def _values_are_monotonic(
    values: np.ndarray, *, tolerance_fraction: float = 0.05
) -> bool:
    """Return True when a finite sequence is approximately monotonic."""
    if values.size < 3:
        return True
    span = float(np.nanmax(values) - np.nanmin(values))
    tolerance = tolerance_fraction * span if span > 0.0 else 0.0
    diffs = np.diff(values)
    return bool(
        np.all(diffs >= -tolerance)
        or np.all(diffs <= tolerance)
    )


def _add_candidate(
    candidates: list[dict[str, Any]],
    *,
    name: str,
    model: str | None,
    priority: str,
    reason: str,
    fit_strategy: str | None = None,
    options: dict[str, Any] | None = None,
) -> None:
    """Append a JSON-safe candidate-model recommendation."""
    entry: dict[str, Any] = {
        "name": name,
        "model": model,
        "priority": priority,
        "reason": reason,
    }
    if fit_strategy is not None:
        entry["fit_strategy"] = fit_strategy
    if options:
        entry["options"] = _clean_scalar_dict(dict(options))
    candidates.append(entry)


def classify_wavelength_diagnostics(
    report: dict[str, Any],
    *,
    achromatic_amplitude_ratio_tol: float = 1.25,
    wavelength_dependent_amplitude_ratio_min: float = 1.5,
    powerlaw_slope_min: float = 0.5,
    negligible_lag_fraction: float = 0.05,
    significant_lag_fraction: float = 0.10,
) -> dict[str, Any]:
    """Classify a pre-fit wavelength diagnostic report.

    The classifier is intentionally heuristic and conservative.  It does not
    choose a final GP model and it does not run any fitting.  It translates the
    pre-fit diagnostic table into evidence labels and a short candidate-model
    list that can be used by later model-comparison code.

    Parameters
    ----------
    report : dict
        Report returned by :func:`diagnose_wavelength_dependence_prefit`.
    achromatic_amplitude_ratio_tol : float, optional
        Maximum max/min fixed-frequency amplitude ratio still treated as
        consistent with constant amplitude.
    wavelength_dependent_amplitude_ratio_min : float, optional
        Minimum max/min amplitude ratio treated as evidence for wavelength-
        dependent amplitude.
    powerlaw_slope_min : float, optional
        Minimum absolute log-amplitude/log-wavelength slope used to label a
        smooth monotonic trend as power-law-like.
    negligible_lag_fraction : float, optional
        Maximum lag span as a fraction of the fixed period treated as
        consistent with no wavelength-dependent lag.
    significant_lag_fraction : float, optional
        Minimum lag span as a fraction of the fixed period used to flag a
        possible wavelength-dependent lag.

    Returns
    -------
    dict
        JSON-safe classification with evidence labels, candidate models, and
        warnings.  Model recommendations are candidate families, not a final
        selection.
    """
    if report.get("kind") != "wavelength_dependence_prefit_diagnostics":
        raise ValueError(
            "classify_wavelength_diagnostics() expects a report produced by "
            "diagnose_wavelength_dependence_prefit()."
        )

    for name, value in [
        ("achromatic_amplitude_ratio_tol", achromatic_amplitude_ratio_tol),
        (
            "wavelength_dependent_amplitude_ratio_min",
            wavelength_dependent_amplitude_ratio_min,
        ),
        ("powerlaw_slope_min", powerlaw_slope_min),
        ("negligible_lag_fraction", negligible_lag_fraction),
        ("significant_lag_fraction", significant_lag_fraction),
    ]:
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite value.")

    summary = report.get("summary", {})
    band_table = report.get("band_table", [])
    n_bands = int(summary.get("n_bands", 0) or 0)
    n_usable = int(summary.get("n_usable_for_wavelength_diagnostics", 0) or 0)
    fixed_frequency = _finite_float(report.get("fixed_frequency"))
    fixed_period = _finite_float(report.get("fixed_period"))
    fixed_rows = _fixed_frequency_rows(band_table)

    warnings = list(report.get("warnings", []))
    evidence: list[str] = []
    candidates: list[dict[str, Any]] = []

    classification: dict[str, Any] = {
        "available": False,
        "primary_class": "insufficient_data",
        "amplitude_class": "unavailable",
        "phase_lag_class": "unavailable",
        "n_bands": n_bands,
        "n_usable_bands": n_usable,
        "n_fixed_frequency_bands": len(fixed_rows),
        "evidence": evidence,
        "recommended_candidate_models": candidates,
        "warnings": warnings,
    }

    if n_bands < 2 or n_usable < 2:
        warnings.append(
            "Candidate wavelength-model recommendations are suppressed because "
            "fewer than two usable bands are available."
        )
        return _clean_scalar_dict(classification)

    classification["available"] = True
    _add_candidate(
        candidates,
        name="robust_2d_consensus_baseline",
        model="2D",
        fit_strategy="consensus",
        priority="baseline",
        reason=(
            "Use the stabilized 2D consensus path as the baseline period/PSD "
            "fit before interpreting more specific wavelength models."
        ),
        options={"learn_additional_noise": True},
    )

    if fixed_frequency is None or fixed_period is None:
        classification["primary_class"] = "prefit_table_only"
        warnings.append(
            "No fixed period/frequency was supplied, so amplitude and phase-lag "
            "candidate recommendations are not available yet."
        )
        _add_candidate(
            candidates,
            name="next_step_consensus_period_diagnostics",
            model=None,
            fit_strategy="consensus",
            priority="next_step",
            reason=(
                "Run LS/ACF or consensus period diagnostics, then rerun "
                "diagnose_wavelength_dependence(period=...) to classify "
                "period-locked amplitude and phase behavior."
            ),
        )
        return _clean_scalar_dict(classification)

    if len(fixed_rows) < 2:
        classification["primary_class"] = "fixed_frequency_inconclusive"
        warnings.append(
            "Fixed-frequency diagnostics succeeded in fewer than two bands; "
            "candidate recommendations are limited to the baseline model."
        )
        return _clean_scalar_dict(classification)

    wavelengths = np.asarray([item[0] for item in fixed_rows], dtype=float)
    amplitudes = np.asarray(
        [item[1].get("amplitude", np.nan) for item in fixed_rows],
        dtype=float,
    )
    lags = np.asarray(
        [item[1].get("lag", np.nan) for item in fixed_rows],
        dtype=float,
    )

    finite_amp = np.isfinite(wavelengths) & np.isfinite(amplitudes) & (amplitudes > 0.0)
    if np.count_nonzero(finite_amp) >= 2:
        amp_wls = wavelengths[finite_amp]
        amp_values = amplitudes[finite_amp]
        amp_ratio = float(np.nanmax(amp_values) / np.nanmin(amp_values))
        amp_monotonic = _values_are_monotonic(amp_values)
        amp_slope = None
        positive = (amp_wls > 0.0) & (amp_values > 0.0)
        if np.count_nonzero(positive) >= 2:
            amp_slope = float(
                np.polyfit(
                    np.log(amp_wls[positive]),
                    np.log(amp_values[positive]),
                    1,
                )[0]
            )

        classification["amplitude_ratio_max_to_min"] = amp_ratio
        classification["amplitude_loglog_slope"] = amp_slope
        classification["amplitude_monotonic"] = bool(amp_monotonic)

        if amp_ratio <= achromatic_amplitude_ratio_tol:
            classification["amplitude_class"] = "consistent_with_constant_amplitude"
            evidence.append(
                "Fixed-frequency amplitudes are consistent with "
                "wavelength-independent variability."
            )
            _add_candidate(
                candidates,
                name="achromatic_separable_candidate",
                model="2DAchromatic",
                priority="candidate",
                reason=(
                    "Fixed-frequency amplitudes vary weakly across usable bands; "
                    "an achromatic separable covariance is worth testing."
                ),
            )
        elif (
            amp_ratio >= wavelength_dependent_amplitude_ratio_min
            and amp_monotonic
            and amp_slope is not None
            and abs(amp_slope) >= powerlaw_slope_min
        ):
            classification["amplitude_class"] = "power_law_like_amplitude_trend"
            evidence.append(
                "Fixed-frequency amplitudes vary smoothly and monotonically "
                "with wavelength."
            )
            _add_candidate(
                candidates,
                name="smooth_wavelength_dependent_candidate",
                model="2DWavelengthDependent",
                priority="candidate",
                reason=(
                    "The accepted bands share a fixed frequency but show a smooth "
                    "wavelength-dependent amplitude trend."
                ),
                options={"wavelength_kernel_type": "rbf"},
            )
            _add_candidate(
                candidates,
                name="power_law_mean_candidate",
                model="2DPowerLawMean",
                priority="candidate",
                reason=(
                    "The amplitude trend is monotonic and approximately linear in "
                    "log amplitude versus log wavelength, so a power-law wavelength "
                    "mean family should be compared."
                ),
            )
        elif amp_ratio >= wavelength_dependent_amplitude_ratio_min:
            classification["amplitude_class"] = "smooth_or_band_dependent_amplitude"
            evidence.append(
                "Fixed-frequency amplitudes differ substantially across wavelength."
            )
            _add_candidate(
                candidates,
                name="smooth_wavelength_dependent_candidate",
                model="2DWavelengthDependent",
                priority="candidate",
                reason=(
                    "The accepted bands share a fixed frequency but have different "
                    "period-locked amplitudes."
                ),
                options={"wavelength_kernel_type": "rbf"},
            )
        else:
            classification["amplitude_class"] = "weak_or_ambiguous_amplitude_trend"
            evidence.append(
                "Fixed-frequency amplitudes are not constant enough for a clean "
                "achromatic label and not different enough for a strong "
                "wavelength-dependent label."
            )
    else:
        warnings.append(
            "Amplitude classification is unavailable because fewer than two bands "
            "have positive finite fixed-frequency amplitudes."
        )

    finite_lag = np.isfinite(lags)
    if np.count_nonzero(finite_lag) >= 2:
        lag_values = lags[finite_lag]
        lag_span = float(np.nanmax(lag_values) - np.nanmin(lag_values))
        lag_span_fraction = (
            float(lag_span / fixed_period) if fixed_period > 0.0 else None
        )
        classification["lag_span"] = lag_span
        classification["lag_span_fraction_of_period"] = lag_span_fraction
        if (
            lag_span_fraction is not None
            and lag_span_fraction <= negligible_lag_fraction
        ):
            classification["phase_lag_class"] = "consistent_with_zero_lag"
            evidence.append(
                "Fixed-frequency phases are consistent with no meaningful "
                "wavelength-dependent lag."
            )
        elif (
            lag_span_fraction is not None
            and lag_span_fraction >= significant_lag_fraction
        ):
            lag_monotonic = _values_are_monotonic(lag_values)
            classification["phase_lag_class"] = (
                "possible_monotonic_wavelength_lag"
                if lag_monotonic
                else "possible_wavelength_lag"
            )
            classification["lag_monotonic"] = bool(lag_monotonic)
            evidence.append(
                "Fixed-frequency phases show a potentially significant "
                "wavelength-dependent lag."
            )
            warnings.append(
                "A possible wavelength-dependent phase/lag was detected. Current "
                "PGMUVI separable wavelength models do not explicitly parameterize "
                "deterministic wavelength-dependent time delays."
            )
        else:
            classification["phase_lag_class"] = "weak_or_ambiguous_lag"
            evidence.append(
                "Fixed-frequency phases show only weak or ambiguous "
                "wavelength-dependent lag evidence."
            )
    else:
        warnings.append(
            "Phase-lag classification is unavailable because fewer than two bands "
            "have finite fixed-frequency lag estimates."
        )

    amplitude_class = classification.get("amplitude_class")
    phase_class = classification.get("phase_lag_class")
    if isinstance(phase_class, str) and phase_class.startswith("possible"):
        classification["primary_class"] = "possible_wavelength_dependent_lag"
    elif amplitude_class == "consistent_with_constant_amplitude":
        classification["primary_class"] = "achromatic_shared_variability_candidate"
    elif amplitude_class in {
        "power_law_like_amplitude_trend",
        "smooth_or_band_dependent_amplitude",
    }:
        classification["primary_class"] = (
            "wavelength_modulated_shared_variability_candidate"
        )
    elif amplitude_class == "weak_or_ambiguous_amplitude_trend":
        classification["primary_class"] = "ambiguous_wavelength_dependence"
    else:
        classification["primary_class"] = "fixed_frequency_inconclusive"

    return _clean_scalar_dict(classification)


def _normalise_candidate_models(
    candidates: list[Any] | tuple[Any, ...] | None,
) -> list[dict[str, Any]]:
    """Normalise model-comparison candidate specifications.

    Candidate dictionaries are intentionally aligned with the entries returned
    by :func:`classify_wavelength_diagnostics`, but strings are accepted as a
    compact user-facing form.
    """
    if candidates is None:
        return []

    normalised: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        if isinstance(candidate, str):
            entry = {
                "name": candidate,
                "model": candidate,
                "priority": "candidate",
                "reason": "User-supplied candidate model string.",
            }
        elif isinstance(candidate, dict):
            entry = dict(candidate)
            if "name" not in entry or entry.get("name") is None:
                model_name = entry.get("model")
                entry["name"] = str(model_name) if model_name is not None else f"candidate_{idx}"
        else:
            raise TypeError(
                "Candidate model specifications must be strings or dictionaries."
            )

        options = entry.get("options")
        if options is not None and not isinstance(options, dict):
            raise TypeError("Candidate 'options' entries must be dictionaries.")
        fit_kwargs = entry.get("fit_kwargs")
        if fit_kwargs is not None and not isinstance(fit_kwargs, dict):
            raise TypeError("Candidate 'fit_kwargs' entries must be dictionaries.")
        normalised.append(entry)

    return normalised


def _candidate_extra_kwargs(
    candidate: dict[str, Any], per_candidate_fit_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Return extra fit kwargs matching a candidate by name or model."""
    extras: dict[str, Any] = {}
    for key in (candidate.get("name"), candidate.get("model")):
        if key is None:
            continue
        value = per_candidate_fit_kwargs.get(str(key))
        if value is None:
            continue
        if not isinstance(value, dict):
            raise TypeError(
                "per_candidate_fit_kwargs values must be dictionaries keyed by "
                "candidate name or model string."
            )
        extras.update(value)
    return extras


def _build_candidate_fit_kwargs(
    candidate: dict[str, Any],
    *,
    base_fit_kwargs: dict[str, Any],
    per_candidate_fit_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Build the exact kwargs passed to ``Lightcurve.fit`` for a candidate."""
    model = candidate.get("model")
    if model is None:
        raise ValueError("Cannot build fit kwargs for a non-fit recommendation.")

    fit_kwargs = dict(base_fit_kwargs)
    fit_kwargs.update(dict(candidate.get("options") or {}))
    fit_kwargs.update(dict(candidate.get("fit_kwargs") or {}))
    fit_kwargs.update(_candidate_extra_kwargs(candidate, per_candidate_fit_kwargs))
    fit_kwargs["model"] = model
    if candidate.get("fit_strategy") is not None:
        fit_kwargs["fit_strategy"] = candidate.get("fit_strategy")
    return fit_kwargs


def _clone_for_model_comparison(lightcurve):
    """Return an isolated light-curve object for one candidate fit."""
    cloned = copy.deepcopy(lightcurve)
    if hasattr(cloned, "clear_fit_history"):
        try:
            cloned.clear_fit_history()
        except Exception:
            pass
    return cloned


def _latest_fit_history_entry(lightcurve) -> dict[str, Any] | None:
    """Return the most recent fit-history entry from a lightcurve-like object."""
    getter = getattr(lightcurve, "get_fit_history", None)
    if getter is None:
        return None
    try:
        history = getter()
    except Exception:
        return None
    if not history:
        return None
    last = history[-1]
    return dict(last) if isinstance(last, dict) else None


def _module_class_name(obj: Any) -> str | None:
    """Return a best-effort class name for a fitted model/likelihood object."""
    if obj is None:
        return None
    try:
        return obj.__class__.__name__
    except Exception:
        return None


def _tensor_scalar_or_list(value: Any) -> Any:
    """Convert small tensor/array-like values to JSON-safe scalars/lists."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            value = value.numpy()
        except Exception:
            pass
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _finite_or_none(float(value.reshape(-1)[0]))
        if value.size <= 8:
            return [_finite_or_none(float(v)) for v in value.reshape(-1)]
        finite = value[np.isfinite(value)]
        return {
            "shape": list(value.shape),
            "min": _finite_or_none(float(np.min(finite))) if finite.size else None,
            "max": _finite_or_none(float(np.max(finite))) if finite.size else None,
        }
    return _finite_or_none(value)


def _likelihood_noise_summary(likelihood: Any) -> dict[str, Any]:
    """Extract lightweight noise diagnostics from a fitted likelihood."""
    if likelihood is None:
        return {"available": False}

    summary: dict[str, Any] = {
        "available": True,
        "likelihood_class": _module_class_name(likelihood),
        "noise_parameters": {},
    }

    named_parameters = getattr(likelihood, "named_parameters", None)
    if named_parameters is not None:
        try:
            for name, value in named_parameters():
                if "noise" in str(name).lower():
                    summary["noise_parameters"][str(name)] = _tensor_scalar_or_list(value)
        except Exception:
            pass

    for attr_name in ("noise", "raw_noise"):
        if hasattr(likelihood, attr_name):
            try:
                summary[attr_name] = _tensor_scalar_or_list(getattr(likelihood, attr_name))
            except Exception:
                pass

    return _clean_scalar_dict(summary)


def _failure_category(exc: BaseException) -> str:
    """Classify a candidate-fit exception for model-comparison reports."""
    exc_name = exc.__class__.__name__
    if exc_name == "ConsensusFitError":
        return "consensus_data_rejection"
    if isinstance(exc, FloatingPointError):
        return "numerical_failure"
    if isinstance(exc, (ValueError, TypeError)):
        return "api_or_configuration_error"
    if isinstance(exc, RuntimeError):
        return "runtime_fit_failure"
    return "fit_exception"


def _comparison_result_entry(
    *,
    candidate: dict[str, Any],
    status: str,
    fit_kwargs: dict[str, Any] | None = None,
    elapsed_seconds: float | None = None,
    target_lightcurve: Any | None = None,
    exception: BaseException | None = None,
    fit_result: Any | None = None,
    residual_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one JSON-safe candidate-comparison row."""
    entry: dict[str, Any] = {
        "name": candidate.get("name"),
        "model": candidate.get("model"),
        "fit_strategy": candidate.get("fit_strategy"),
        "priority": candidate.get("priority"),
        "reason": candidate.get("reason"),
        "status": status,
        "fit_success": bool(status == "success"),
        "fit_failed": bool(status == "failed"),
        "success": bool(status == "success"),
        "failed": bool(status == "failed"),
        "skipped": bool(status == "skipped"),
        "elapsed_seconds": elapsed_seconds,
        "fit_kwargs": fit_kwargs,
    }

    if status == "skipped":
        entry["skip_reason"] = "candidate has no model string to fit"
    elif status == "success":
        entry["result_type"] = _module_class_name(fit_result)
        entry["resolved_model_class"] = _module_class_name(
            getattr(target_lightcurve, "model", None)
        )
        entry["likelihood_noise_summary"] = _likelihood_noise_summary(
            getattr(target_lightcurve, "likelihood", None)
        )
        entry["fit_history_entry"] = _latest_fit_history_entry(target_lightcurve)
        if residual_diagnostics is not None:
            entry["residual_diagnostics"] = residual_diagnostics
            entry["predictive_score"] = residual_diagnostics.get(
                "predictive_score", {"available": False}
            )
    elif status == "failed" and exception is not None:
        entry["exception_type"] = exception.__class__.__name__
        entry["exception_message"] = str(exception)
        entry["failure_category"] = _failure_category(exception)
        if hasattr(exception, "failure_diagnostics"):
            try:
                entry["failure_diagnostics"] = exception.failure_diagnostics
            except Exception:
                pass
        entry["fit_history_entry"] = _latest_fit_history_entry(target_lightcurve)

    return _clean_scalar_dict(entry)



def _array_from_tensor_like(value: Any) -> np.ndarray | None:
    """Convert tensor/array/scalar-like values to a NumPy array."""
    if value is None:
        return None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    try:
        return np.asarray(value)
    except Exception:
        return None


def _training_prediction_arrays(lightcurve) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Return training-point predictive mean/variance arrays when available."""
    custom = getattr(lightcurve, "training_predictions_for_wavelength_diagnostics", None)
    if callable(custom):
        prediction = custom()
        if isinstance(prediction, dict):
            mean = prediction.get("mean")
            variance = prediction.get("variance")
        else:
            mean, variance = prediction
        return (
            _array_from_tensor_like(mean),
            _array_from_tensor_like(variance),
            "custom_training_predictions",
        )

    mean_attr = getattr(lightcurve, "_wavelength_diagnostic_prediction_mean", None)
    if mean_attr is not None:
        return (
            _array_from_tensor_like(mean_attr),
            _array_from_tensor_like(
                getattr(lightcurve, "_wavelength_diagnostic_prediction_variance", None)
            ),
            "stored_training_predictions",
        )

    model = getattr(lightcurve, "model", None)
    likelihood = getattr(lightcurve, "likelihood", None)
    x_train = getattr(lightcurve, "_xdata_transformed", None)
    if model is None or x_train is None or torch is None:
        return None, None, "unavailable"

    try:
        if hasattr(model, "eval"):
            model.eval()
        if likelihood is not None and hasattr(likelihood, "eval"):
            likelihood.eval()
        with torch.no_grad():
            output = model(x_train)
            predictive = likelihood(output) if likelihood is not None else output
            mean = getattr(predictive, "mean", None)
            variance = getattr(predictive, "variance", None)
    except Exception:
        return None, None, "prediction_failed"

    return _array_from_tensor_like(mean), _array_from_tensor_like(variance), "gpytorch"


def _safe_standardised_residuals(
    residual: np.ndarray,
    variance: np.ndarray | None,
) -> np.ndarray | None:
    """Return residuals divided by predictive standard deviation."""
    if variance is None:
        return None
    variance = np.asarray(variance, dtype=float)
    if variance.shape != residual.shape:
        return None
    valid = np.isfinite(variance) & (variance > 0.0)
    if not np.any(valid):
        return None
    out = np.full_like(residual, np.nan, dtype=float)
    out[valid] = residual[valid] / np.sqrt(variance[valid])
    return out


def _gaussian_negative_log_predictive_density(
    residual: np.ndarray,
    variance: np.ndarray | None,
) -> np.ndarray | None:
    """Compute per-point Gaussian negative log predictive density."""
    if variance is None:
        return None
    variance = np.asarray(variance, dtype=float)
    if variance.shape != residual.shape:
        return None
    valid = np.isfinite(residual) & np.isfinite(variance) & (variance > 0.0)
    if not np.any(valid):
        return None
    out = np.full_like(residual, np.nan, dtype=float)
    out[valid] = 0.5 * (
        np.log(2.0 * np.pi * variance[valid])
        + (residual[valid] ** 2) / variance[valid]
    )
    return out


def _residual_scalar_summary(
    residual: np.ndarray,
    standardised: np.ndarray | None,
    nlpd: np.ndarray | None,
    variance: np.ndarray | None,
) -> dict[str, Any]:
    """Summarise residual and predictive-score arrays."""
    finite = np.isfinite(residual)
    summary: dict[str, Any] = {
        "n_points": int(np.count_nonzero(finite)),
        "residual_mean": None,
        "residual_median": None,
        "residual_rms": None,
        "residual_robust_scatter": None,
        "standardized_residual_mean": None,
        "standardized_residual_rms": None,
        "coverage_1sigma": None,
        "coverage_2sigma": None,
        "mean_negative_log_predictive_density": None,
        "median_negative_log_predictive_density": None,
        "median_predictive_std": None,
    }
    if np.any(finite):
        r = residual[finite]
        summary["residual_mean"] = float(np.mean(r))
        summary["residual_median"] = float(np.median(r))
        summary["residual_rms"] = float(np.sqrt(np.mean(r**2)))
        summary["residual_robust_scatter"] = float(robust_scale(r))

    if standardised is not None:
        zfinite = np.isfinite(standardised)
        if np.any(zfinite):
            z = standardised[zfinite]
            summary["standardized_residual_mean"] = float(np.mean(z))
            summary["standardized_residual_rms"] = float(np.sqrt(np.mean(z**2)))
            summary["coverage_1sigma"] = float(np.mean(np.abs(z) <= 1.0))
            summary["coverage_2sigma"] = float(np.mean(np.abs(z) <= 2.0))

    if nlpd is not None:
        nfinite = np.isfinite(nlpd)
        if np.any(nfinite):
            n = nlpd[nfinite]
            summary["mean_negative_log_predictive_density"] = float(np.mean(n))
            summary["median_negative_log_predictive_density"] = float(np.median(n))

    if variance is not None:
        vfinite = np.isfinite(variance) & (variance > 0.0)
        if np.any(vfinite):
            summary["median_predictive_std"] = float(np.median(np.sqrt(variance[vfinite])))

    return _clean_scalar_dict(summary)


def compute_wavelength_residual_diagnostics(
    lightcurve,
    *,
    frequency: float | None = None,
    period: float | None = None,
    min_points_per_band: int = 3,
) -> dict[str, Any]:
    """Compute residual and predictive diagnostics for a fitted light curve.

    The diagnostics are evaluated at the training coordinates.  They are meant
    for model-comparison reports: they score whether a fitted candidate leaves
    structured residuals by wavelength and, when predictive variances are
    available, compute Gaussian predictive-score and coverage summaries.
    """
    if isinstance(min_points_per_band, bool) or int(min_points_per_band) < 1:
        raise ValueError("min_points_per_band must be a positive integer.")
    min_points_per_band = int(min_points_per_band)
    fixed_frequency = _resolve_frequency(frequency=frequency, period=period)

    x_raw = _array_from_tensor_like(getattr(lightcurve, "_xdata_raw", None))
    y_train = _array_from_tensor_like(getattr(lightcurve, "_ydata_transformed", None))
    if y_train is None:
        y_train = _array_from_tensor_like(getattr(lightcurve, "_ydata_raw", None))
    if x_raw is None or y_train is None:
        return {
            "available": False,
            "status": "missing_training_data",
            "by_band": [],
            "overall": {},
            "predictive_score": {"available": False},
        }

    x_raw = np.asarray(x_raw, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    if x_raw.ndim != 2 or x_raw.shape[1] < 2 or x_raw.shape[0] != y_train.size:
        return {
            "available": False,
            "status": "requires_2d_multiband_training_data",
            "by_band": [],
            "overall": {},
            "predictive_score": {"available": False},
        }

    mean, variance, prediction_source = _training_prediction_arrays(lightcurve)
    if mean is None:
        return {
            "available": False,
            "status": prediction_source,
            "by_band": [],
            "overall": {},
            "predictive_score": {"available": False},
        }
    mean = np.asarray(mean, dtype=float).reshape(-1)
    if mean.size != y_train.size:
        return {
            "available": False,
            "status": "prediction_shape_mismatch",
            "prediction_source": prediction_source,
            "by_band": [],
            "overall": {},
            "predictive_score": {"available": False},
        }

    if variance is not None:
        variance = np.asarray(variance, dtype=float).reshape(-1)
        if variance.size != y_train.size:
            variance = None

    residual = y_train - mean
    standardised = _safe_standardised_residuals(residual, variance)
    nlpd = _gaussian_negative_log_predictive_density(residual, variance)
    wavelengths = np.unique(x_raw[:, 1])
    band_rows: list[dict[str, Any]] = []

    for wl in wavelengths:
        mask = x_raw[:, 1] == wl
        row = {
            "wavelength": float(wl),
            "n_points": int(np.count_nonzero(mask)),
            "status": "ok" if np.count_nonzero(mask) >= min_points_per_band else "insufficient_points",
            "band_labels": _band_labels_for_mask(getattr(lightcurve, "band", None), mask),
        }
        if np.count_nonzero(mask) >= min_points_per_band:
            row.update(
                _residual_scalar_summary(
                    residual[mask],
                    standardised[mask] if standardised is not None else None,
                    nlpd[mask] if nlpd is not None else None,
                    variance[mask] if variance is not None else None,
                )
            )
            if fixed_frequency is not None:
                periodic = _fit_fixed_frequency_sinusoid(
                    x_raw[mask, 0],
                    residual[mask],
                    None,
                    frequency=fixed_frequency,
                    min_points=max(3, min_points_per_band),
                )
                row["fixed_frequency_residual"] = periodic
        band_rows.append(_clean_scalar_dict(row))

    overall = _residual_scalar_summary(residual, standardised, nlpd, variance)
    predictive_score = {
        "available": bool(nlpd is not None and np.any(np.isfinite(nlpd))),
        "mean_negative_log_predictive_density": overall.get(
            "mean_negative_log_predictive_density"
        ),
        "median_negative_log_predictive_density": overall.get(
            "median_negative_log_predictive_density"
        ),
        "standardized_residual_rms": overall.get("standardized_residual_rms"),
        "coverage_1sigma": overall.get("coverage_1sigma"),
        "coverage_2sigma": overall.get("coverage_2sigma"),
    }

    report = {
        "available": True,
        "status": "ok",
        "prediction_source": prediction_source,
        "target_space": "transformed_y_training_space",
        "fixed_frequency": fixed_frequency,
        "fixed_period": (None if fixed_frequency is None else float(1.0 / fixed_frequency)),
        "overall": overall,
        "by_band": band_rows,
        "predictive_score": _clean_scalar_dict(predictive_score),
    }
    return _clean_scalar_dict(report)


def _score_comparison_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise scored successful model/kernel config fits."""
    scored: list[tuple[float, dict[str, Any]]] = []
    for result in results:
        if result.get("status") != "success":
            continue
        score = result.get("predictive_score", {})
        value = _finite_float(score.get("mean_negative_log_predictive_density"))
        if value is not None:
            scored.append((value, result))

    if not scored:
        return {
            "n_scored_successful": 0,
            "best_model_kernel_config": None,
            "selection_status": "not_scored",
            "selection_basis": None,
        }

    scored.sort(key=lambda item: item[0])
    best_value, best = scored[0]
    return {
        "n_scored_successful": len(scored),
        "best_model_kernel_config": {
            "name": best.get("name"),
            "model": best.get("model"),
            "fit_strategy": best.get("fit_strategy"),
            "mean_negative_log_predictive_density": float(best_value),
        },
        "selection_status": "scored_predictive",
        "selection_basis": "lowest training-point mean negative log predictive density",
    }

def _scored_candidate_rows(results: list[dict[str, Any]]) -> list[tuple[float, dict[str, Any]]]:
    """Return scored successful candidates sorted by predictive score."""
    scored: list[tuple[float, dict[str, Any]]] = []
    for result in results:
        if result.get("status") != "success":
            continue
        score = result.get("predictive_score", {})
        value = _finite_float(score.get("mean_negative_log_predictive_density"))
        if value is not None:
            scored.append((value, result))
    scored.sort(key=lambda item: item[0])
    return scored


def _candidate_identifier(result: dict[str, Any]) -> dict[str, Any]:
    """Return compact candidate-identification fields for reports."""
    return {
        "name": result.get("name"),
        "model": result.get("model"),
        "fit_strategy": result.get("fit_strategy"),
    }


def _candidate_residual_flags(
    result: dict[str, Any],
    *,
    standardized_residual_rms_warning: float,
    poor_coverage_2sigma_min: float,
    band_standardized_residual_rms_warning: float,
) -> list[dict[str, Any]]:
    """Build residual-quality flags for one successful comparison result."""
    diagnostics = result.get("residual_diagnostics")
    if not isinstance(diagnostics, dict) or not diagnostics.get("available"):
        return []

    flags: list[dict[str, Any]] = []
    overall = diagnostics.get("overall", {})
    z_rms = _finite_float(overall.get("standardized_residual_rms"))
    if z_rms is not None and z_rms >= standardized_residual_rms_warning:
        flags.append(
            {
                "flag": "large_standardized_residual_rms",
                "severity": "warning",
                "value": z_rms,
                "threshold": standardized_residual_rms_warning,
                "message": (
                    "Training-point standardized residual RMS is large; this "
                    "candidate may be underfitting, overconfident, or using "
                    "inadequate noise assumptions."
                ),
            }
        )

    coverage = _finite_float(overall.get("coverage_2sigma"))
    if coverage is not None and coverage < poor_coverage_2sigma_min:
        flags.append(
            {
                "flag": "poor_two_sigma_coverage",
                "severity": "warning",
                "value": coverage,
                "threshold": poor_coverage_2sigma_min,
                "message": (
                    "Training points have poor two-sigma predictive coverage; "
                    "the candidate may be overconfident or missing structure."
                ),
            }
        )

    bad_band_rows: list[dict[str, Any]] = []
    for row in diagnostics.get("by_band", []):
        if not isinstance(row, dict) or row.get("status") != "ok":
            continue
        band_z_rms = _finite_float(row.get("standardized_residual_rms"))
        if (
            band_z_rms is not None
            and band_z_rms >= band_standardized_residual_rms_warning
        ):
            bad_band_rows.append(
                {
                    "wavelength": row.get("wavelength"),
                    "band_labels": row.get("band_labels", []),
                    "standardized_residual_rms": band_z_rms,
                }
            )
    if bad_band_rows:
        flags.append(
            {
                "flag": "band_specific_residual_mismatch",
                "severity": "warning",
                "threshold": band_standardized_residual_rms_warning,
                "bands": bad_band_rows,
                "message": (
                    "One or more wavelength bands have unusually large "
                    "standardized residual scatter; inspect rejected bands, "
                    "band uncertainties, and wavelength-model adequacy."
                ),
            }
        )

    return _clean_scalar_dict({"flags": flags})["flags"]


def interpret_wavelength_model_comparison(
    comparison_report: dict[str, Any],
    *,
    score_tie_tolerance: float = 0.05,
    standardized_residual_rms_warning: float = 2.0,
    poor_coverage_2sigma_min: float = 0.80,
    band_standardized_residual_rms_warning: float = 2.5,
) -> dict[str, Any]:
    """Interpret a wavelength model-comparison report conservatively.

    The interpretation layer does not declare a final science model.  It turns
    PR45 predictive/residual scores into a compact ranking and explicit quality
    warnings.  The output is intended to help users decide whether the current
    model/kernel config set is informative, whether scores are effectively tied, and
    whether residual structure argues against trusting the provisional best
    model/kernel config.
    """
    if comparison_report.get("kind") != "wavelength_model_comparison":
        raise ValueError(
            "interpret_wavelength_model_comparison() expects a report produced "
            "by compare_wavelength_candidate_models()."
        )

    for name, value in [
        ("score_tie_tolerance", score_tie_tolerance),
        ("standardized_residual_rms_warning", standardized_residual_rms_warning),
        ("poor_coverage_2sigma_min", poor_coverage_2sigma_min),
        (
            "band_standardized_residual_rms_warning",
            band_standardized_residual_rms_warning,
        ),
    ]:
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a non-negative finite value.")

    summary = comparison_report.get("summary", {})
    results = comparison_report.get("results", [])
    warnings = list(comparison_report.get("warnings", []))
    scored = _scored_candidate_rows(results)

    interpretation: dict[str, Any] = {
        "available": False,
        "status": "not_interpretable",
        "decision": "no_successful_fit",
        "provisional_best_model_kernel_config": None,
        "model_kernel_config_rankings": [],
        "quality_flags": [],
        "warnings": warnings,
        "notes": [
            "This interpretation is diagnostic. It is not a substitute for "
            "science review of light curves, residuals, sampling, and candidate "
            "model assumptions."
        ],
    }

    if int(summary.get("n_successful", 0) or 0) == 0:
        warnings.append(
            "No model/kernel config fit succeeded, so wavelength-model comparison "
            "cannot be interpreted. Inspect failure categories before changing "
            "the model/kernel config set."
        )
        return _clean_scalar_dict(interpretation)

    if not scored:
        interpretation.update(
            {
                "available": True,
                "status": "unscored_successful_fits",
                "decision": "scores_unavailable",
            }
        )
        warnings.append(
            "At least one model/kernel config fit succeeded, but predictive scores were "
            "not available. Use fit-status and residual diagnostics only."
        )
        return _clean_scalar_dict(interpretation)

    best_score, _ = scored[0]
    denominator = max(1.0, abs(best_score))
    rankings: list[dict[str, Any]] = []
    for rank, (score, result) in enumerate(scored, start=1):
        delta = float(score - best_score)
        rankings.append(
            {
                "rank": int(rank),
                **_candidate_identifier(result),
                "mean_negative_log_predictive_density": float(score),
                "delta_from_best": delta,
                "relative_delta_from_best": float(delta / denominator),
            }
        )

    interpretation.update(
        {
            "available": True,
            "status": "ok",
            "decision": "provisional_predictive_preference",
            "provisional_best_model_kernel_config": rankings[0],
            "model_kernel_config_rankings": rankings,
            "selection_basis": "lowest training-point mean negative log predictive density",
        }
    )

    if len(scored) == 1:
        interpretation["decision"] = "single_scored_candidate"
        warnings.append(
            "Only one successful model/kernel config was scored. Treat it as a fitted "
            "baseline, not as evidence that it is preferred over alternatives."
        )
    else:
        second_score = float(scored[1][0])
        second_delta = second_score - best_score
        relative_delta = second_delta / denominator
        interpretation["second_best_delta"] = float(second_delta)
        interpretation["second_best_relative_delta"] = float(relative_delta)
        if relative_delta <= score_tie_tolerance:
            interpretation["decision"] = "scores_indistinguishable"
            warnings.append(
                "The two best predictive scores are within the configured tie "
                "tolerance. Prefer the simpler or more interpretable model unless "
                "residual plots show a meaningful difference."
            )

    quality_flags: list[dict[str, Any]] = []
    for _, result in scored:
        candidate_flags = _candidate_residual_flags(
            result,
            standardized_residual_rms_warning=standardized_residual_rms_warning,
            poor_coverage_2sigma_min=poor_coverage_2sigma_min,
            band_standardized_residual_rms_warning=band_standardized_residual_rms_warning,
        )
        for flag in candidate_flags:
            flag = dict(flag)
            flag["candidate"] = _candidate_identifier(result)
            quality_flags.append(flag)

    if quality_flags:
        interpretation["quality_flags"] = quality_flags
        warnings.append(
            "One or more successful candidates have residual or predictive-"
            "coverage warnings. Do not select a wavelength model from scalar "
            "scores alone."
        )

    return _clean_scalar_dict(interpretation)


def _format_value(value: Any, *, precision: int = 4) -> str:
    """Format scalar report values for compact human-readable summaries."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    number = _finite_float(value)
    if number is not None:
        return f"{number:.{precision}g}"
    return str(value)


def format_wavelength_diagnostics_report(
    diagnostic_report: dict[str, Any],
    *,
    comparison_report: dict[str, Any] | None = None,
    max_band_rows: int | None = None,
) -> str:
    """Return a compact Markdown summary of wavelength diagnostics.

    The formatter is intentionally report-only: it does not recompute any
    diagnostics, run GP fits, or mutate a light curve.  It is meant for users
    who need a science-facing text summary that can be pasted into notebooks,
    logs, or issue reports.
    """
    if diagnostic_report.get("kind") != "wavelength_dependence_prefit_diagnostics":
        raise ValueError(
            "format_wavelength_diagnostics_report() expects a pre-fit "
            "wavelength diagnostic report."
        )
    if max_band_rows is not None:
        if isinstance(max_band_rows, bool) or int(max_band_rows) < 1:
            raise ValueError("max_band_rows must be a positive integer or None.")
        max_band_rows = int(max_band_rows)

    summary = diagnostic_report.get("summary", {})
    classification = diagnostic_report.get("classification", {})
    amp_summary = diagnostic_report.get("amplitude_phase_summary", {})
    candidates = diagnostic_report.get("recommended_candidate_models", [])
    warnings = list(diagnostic_report.get("warnings", []))
    warnings.extend(classification.get("warnings", []) or [])

    lines: list[str] = []
    lines.append("# Wavelength-dependence diagnostics")
    lines.append("")
    lines.append("## Pre-fit summary")
    lines.append(
        "- Bands: "
        f"{_format_value(summary.get('n_bands'), precision=0)} total, "
        f"{_format_value(summary.get('n_sampling_pass'), precision=0)} passed sampling, "
        f"{_format_value(summary.get('n_variable'), precision=0)} variable, "
        f"{_format_value(summary.get('n_usable_for_wavelength_diagnostics'), precision=0)} usable."
    )
    lines.append(
        "- Fixed period/frequency: "
        f"period={_format_value(diagnostic_report.get('fixed_period'))}, "
        f"frequency={_format_value(diagnostic_report.get('fixed_frequency'))}."
    )
    lines.append(
        "- Primary class: "
        f"{classification.get('primary_class', 'unavailable')}."
    )
    lines.append(
        "- Amplitude class: "
        f"{classification.get('amplitude_class', 'unavailable')}; "
        "phase/lag class: "
        f"{classification.get('phase_lag_class', 'unavailable')}."
    )

    if amp_summary.get("available"):
        lines.append(
            "- Period-locked amplitude ratio max/min: "
            f"{_format_value(amp_summary.get('amplitude_ratio_max_to_min'))}; "
            "log-log slope: "
            f"{_format_value(amp_summary.get('amplitude_loglog_slope'))}; "
            "lag span: "
            f"{_format_value(amp_summary.get('lag_span'))}."
        )

    evidence = classification.get("evidence", []) or []
    if evidence:
        lines.append("")
        lines.append("## Evidence")
        for item in evidence:
            lines.append(f"- {item}")

    band_rows = list(diagnostic_report.get("band_table", []))
    if max_band_rows is not None:
        shown_rows = band_rows[:max_band_rows]
    else:
        shown_rows = band_rows
    if shown_rows:
        lines.append("")
        lines.append("## Band table")
        lines.append(
            "| wavelength | labels | n | sampling | variable | usable | "
            "median flux | robust amp | fixed amp | lag |"
        )
        lines.append(
            "|---:|---|---:|---|---|---|---:|---:|---:|---:|"
        )
        for row in shown_rows:
            flux = row.get("flux_summary", {})
            periodic = row.get("fixed_frequency_diagnostics", {})
            labels = ",".join(row.get("band_labels", []) or []) or "—"
            variable = row.get("variable")
            variable_text = "—" if variable is None else _format_value(variable)
            lines.append(
                "| "
                f"{_format_value(row.get('wavelength'))} | "
                f"{labels} | "
                f"{_format_value(row.get('n_points'), precision=0)} | "
                f"{_format_value(row.get('sampling_pass'))} | "
                f"{variable_text} | "
                f"{_format_value(row.get('usable_for_wavelength_diagnostics'))} | "
                f"{_format_value(flux.get('median_flux'))} | "
                f"{_format_value(flux.get('robust_amplitude_5_95'))} | "
                f"{_format_value(periodic.get('amplitude'))} | "
                f"{_format_value(periodic.get('lag'))} |"
            )
        if max_band_rows is not None and len(band_rows) > max_band_rows:
            lines.append(
                f"\n{len(band_rows) - max_band_rows} additional band row(s) omitted."
            )

    if candidates:
        lines.append("")
        lines.append("## Candidate model recommendations")
        for candidate in candidates:
            name = candidate.get("name", "unnamed_candidate")
            model = candidate.get("model") or "non-fit diagnostic step"
            fit_strategy = candidate.get("fit_strategy")
            priority = candidate.get("priority", "candidate")
            reason = candidate.get("reason", "")
            suffix = f", fit_strategy={fit_strategy}" if fit_strategy else ""
            lines.append(f"- **{name}** ({priority}): model={model}{suffix}. {reason}")

    if comparison_report is not None:
        if comparison_report.get("kind") != "wavelength_model_comparison":
            raise ValueError(
                "comparison_report must be a wavelength model-comparison report."
            )
        comp_summary = comparison_report.get("summary", {})
        interpretation = comparison_report.get("interpretation", {})
        lines.append("")
        lines.append("## Model-comparison summary")
        lines.append(
            "- Model/kernel configs: "
            f"{_format_value(comp_summary.get('n_model_kernel_configs'), precision=0)}; "
            f"successful={_format_value(comp_summary.get('n_successful'), precision=0)}, "
            f"failed={_format_value(comp_summary.get('n_failed'), precision=0)}, "
            f"skipped={_format_value(comp_summary.get('n_skipped'), precision=0)}."
        )
        best = comp_summary.get("best_model_kernel_config")
        if isinstance(best, dict):
            lines.append(
                "- Best scored model/kernel config: "
                f"{best.get('name')} / {best.get('model')} "
                "by mean NLPD="
                f"{_format_value(best.get('mean_negative_log_predictive_density'))}."
            )
        if interpretation:
            lines.append(
                "- Interpretation decision: "
                f"{interpretation.get('decision', interpretation.get('status', 'unavailable'))}."
            )
            quality_flags = interpretation.get("quality_flags", []) or []
            if quality_flags:
                lines.append(f"- Quality flags: {len(quality_flags)} warning(s).")

    unique_warnings = []
    for warning in warnings:
        if warning and warning not in unique_warnings:
            unique_warnings.append(warning)
    if unique_warnings:
        lines.append("")
        lines.append("## Warnings")
        for warning in unique_warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines).rstrip() + "\n"


def _import_pyplot():
    """Import matplotlib.pyplot lazily for optional plotting helpers."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - matplotlib is a dependency
        raise ImportError(
            "Matplotlib is required for wavelength diagnostic plotting."
        ) from exc
    return plt


def _plot_xy(
    x: list[float],
    y: list[float],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    show: bool,
):
    """Create one simple x-y figure for wavelength diagnostics."""
    plt = _import_pyplot()
    fig, ax = plt.subplots()
    ax.plot(x, y, marker="o", linestyle="none")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show:
        plt.show()
    return fig


def plot_wavelength_diagnostics(
    diagnostic_report: dict[str, Any],
    *,
    show: bool = False,
) -> dict[str, Any]:
    """Create science-facing pre-fit wavelength diagnostic figures.

    Returns a dictionary of Matplotlib figures.  Missing diagnostic quantities
    are skipped rather than synthesized.
    """
    if diagnostic_report.get("kind") != "wavelength_dependence_prefit_diagnostics":
        raise ValueError(
            "plot_wavelength_diagnostics() expects a pre-fit wavelength "
            "diagnostic report."
        )
    if not isinstance(show, bool):
        raise ValueError("show must be a bool.")

    rows = list(diagnostic_report.get("band_table", []))
    figures: dict[str, Any] = {}

    wavelengths: list[float] = []
    n_points: list[float] = []
    median_flux: list[float] = []
    robust_amp: list[float] = []
    fixed_amp: list[float] = []
    fixed_amp_wavelengths: list[float] = []
    lag_values: list[float] = []
    lag_wavelengths: list[float] = []

    for row in rows:
        wl = _finite_float(row.get("wavelength"))
        if wl is None:
            continue
        n = _finite_float(row.get("n_points"))
        if n is not None:
            wavelengths.append(wl)
            n_points.append(n)
        flux = row.get("flux_summary", {})
        med = _finite_float(flux.get("median_flux"))
        amp = _finite_float(flux.get("robust_amplitude_5_95"))
        if med is not None:
            median_flux.append(med)
        else:
            median_flux.append(np.nan)
        if amp is not None:
            robust_amp.append(amp)
        else:
            robust_amp.append(np.nan)
        periodic = row.get("fixed_frequency_diagnostics", {})
        famp = _finite_float(periodic.get("amplitude"))
        lag = _finite_float(periodic.get("lag"))
        if famp is not None:
            fixed_amp_wavelengths.append(wl)
            fixed_amp.append(famp)
        if lag is not None:
            lag_wavelengths.append(wl)
            lag_values.append(lag)

    if wavelengths and n_points:
        figures["points_per_wavelength"] = _plot_xy(
            wavelengths,
            n_points,
            xlabel="Wavelength",
            ylabel="Number of points",
            title="Sampling by wavelength",
            show=show,
        )

    if wavelengths and any(np.isfinite(median_flux)):
        figures["median_flux_by_wavelength"] = _plot_xy(
            wavelengths,
            median_flux,
            xlabel="Wavelength",
            ylabel="Median flux",
            title="Median flux by wavelength",
            show=show,
        )

    if wavelengths and any(np.isfinite(robust_amp)):
        figures["robust_amplitude_by_wavelength"] = _plot_xy(
            wavelengths,
            robust_amp,
            xlabel="Wavelength",
            ylabel="Robust amplitude (5–95 half-range)",
            title="Robust amplitude by wavelength",
            show=show,
        )

    if fixed_amp_wavelengths:
        figures["fixed_frequency_amplitude_by_wavelength"] = _plot_xy(
            fixed_amp_wavelengths,
            fixed_amp,
            xlabel="Wavelength",
            ylabel="Fixed-frequency amplitude",
            title="Period-locked amplitude by wavelength",
            show=show,
        )

    if lag_wavelengths:
        figures["fixed_frequency_lag_by_wavelength"] = _plot_xy(
            lag_wavelengths,
            lag_values,
            xlabel="Wavelength",
            ylabel="Lag",
            title="Fixed-frequency lag by wavelength",
            show=show,
        )

    return figures


def plot_wavelength_model_comparison(
    comparison_report: dict[str, Any],
    *,
    show: bool = False,
) -> dict[str, Any]:
    """Create model-comparison diagnostic figures from a comparison report."""
    if comparison_report.get("kind") != "wavelength_model_comparison":
        raise ValueError(
            "plot_wavelength_model_comparison() expects a wavelength "
            "model-comparison report."
        )
    if not isinstance(show, bool):
        raise ValueError("show must be a bool.")

    scored: list[tuple[str, float]] = []
    for result in comparison_report.get("results", []) or []:
        if result.get("status") != "success":
            continue
        score = result.get("predictive_score", {})
        value = _finite_float(score.get("mean_negative_log_predictive_density"))
        if value is None:
            continue
        label = str(result.get("name") or result.get("model") or "candidate")
        scored.append((label, value))

    figures: dict[str, Any] = {}
    if scored:
        plt = _import_pyplot()
        labels = [item[0] for item in scored]
        values = [item[1] for item in scored]
        fig, ax = plt.subplots()
        ax.plot(range(len(values)), values, marker="o", linestyle="none")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Mean negative log predictive density")
        ax.set_title("Wavelength model-comparison scores")
        fig.tight_layout()
        if show:
            plt.show()
        figures["predictive_score_by_candidate"] = fig

    return figures

def compare_wavelength_candidate_models(
    lightcurve,
    *,
    diagnostic_report: dict[str, Any] | None = None,
    candidates: list[Any] | tuple[Any, ...] | None = None,
    base_fit_kwargs: dict[str, Any] | None = None,
    per_candidate_fit_kwargs: dict[str, Any] | None = None,
    residual_diagnostic_kwargs: dict[str, Any] | None = None,
    score_successful_fits: bool = True,
    interpretation_kwargs: dict[str, Any] | None = None,
    interpret_results: bool = True,
    copy_lightcurve: bool = True,
    stop_on_error: bool = False,
) -> dict[str, Any]:
    """Run a controlled wavelength-candidate model comparison.

    This is an additive diagnostic wrapper around ``Lightcurve.fit``.  It does
    not change the behaviour of any model, trainer, constraint, or consensus
    pathway.  The function records which model/kernel config fits succeeded or failed and
    returns lightweight fit-history/noise diagnostics for later residual and
    predictive-scoring PRs.

    Parameters
    ----------
    lightcurve : pgmuvi.lightcurve.Lightcurve
        Light curve to fit.
    diagnostic_report : dict or None, optional
        Report returned by :func:`diagnose_wavelength_dependence_prefit`.  When
        ``candidates`` is omitted, the report's ``recommended_candidate_models``
        are used.
    candidates : list or tuple or None, optional
        Candidate model specifications.  Each item may be a model string or a
        dictionary with keys compatible with ``recommended_candidate_models``.
    base_fit_kwargs : dict or None, optional
        Fit keyword arguments applied to every model/kernel config, e.g.
        ``training_iter``, ``miniter``, ``lr``, or ``learn_additional_noise``.
    per_candidate_fit_kwargs : dict or None, optional
        Additional kwargs keyed by candidate ``name`` or ``model``.
    residual_diagnostic_kwargs : dict or None, optional
        Keyword arguments passed to :func:`compute_wavelength_residual_diagnostics`
        after each successful fit.
    score_successful_fits : bool, optional
        If True, compute residual and predictive diagnostics immediately after
        each successful model/kernel config fit.
    interpretation_kwargs : dict or None, optional
        Keyword arguments passed to :func:`interpret_wavelength_model_comparison`
        when ``interpret_results`` is True.
    interpret_results : bool, optional
        If True, attach a conservative interpretation block with candidate
        rankings and residual-quality warnings.
    copy_lightcurve : bool, optional
        If True, each candidate is fit on a deep copy of the input light curve.
        This is the safe default because it avoids reusing fitted model state.
    stop_on_error : bool, optional
        If True, re-raise the first fit exception after recording it.

    Returns
    -------
    dict
        JSON-safe comparison report with fit outcomes and, by default,
        training-point residual/predictive diagnostics for successful fits.
    """
    base_fit_kwargs = dict(base_fit_kwargs or {})
    per_candidate_fit_kwargs = dict(per_candidate_fit_kwargs or {})
    residual_diagnostic_kwargs = dict(residual_diagnostic_kwargs or {})
    interpretation_kwargs = dict(interpretation_kwargs or {})
    if not isinstance(score_successful_fits, bool):
        raise ValueError("score_successful_fits must be a bool.")
    if not isinstance(interpret_results, bool):
        raise ValueError("interpret_results must be a bool.")

    if candidates is None:
        if diagnostic_report is None:
            diagnostic_report = diagnose_wavelength_dependence_prefit(lightcurve)
        candidates = diagnostic_report.get("recommended_candidate_models", [])

    normalised = _normalise_candidate_models(candidates)
    results: list[dict[str, Any]] = []
    n_model_kernel_configs = 0
    n_successful = 0
    n_failed = 0
    n_skipped = 0

    for candidate in normalised:
        if candidate.get("model") is None:
            n_skipped += 1
            results.append(
                _comparison_result_entry(candidate=candidate, status="skipped")
            )
            continue

        n_model_kernel_configs += 1
        fit_kwargs = _build_candidate_fit_kwargs(
            candidate,
            base_fit_kwargs=base_fit_kwargs,
            per_candidate_fit_kwargs=per_candidate_fit_kwargs,
        )
        target = _clone_for_model_comparison(lightcurve) if copy_lightcurve else lightcurve

        start = time.perf_counter()
        try:
            fit_result = target.fit(**fit_kwargs)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            n_failed += 1
            results.append(
                _comparison_result_entry(
                    candidate=candidate,
                    status="failed",
                    fit_kwargs=fit_kwargs,
                    elapsed_seconds=elapsed,
                    target_lightcurve=target,
                    exception=exc,
                )
            )
            if stop_on_error:
                raise
        else:
            elapsed = time.perf_counter() - start
            n_successful += 1
            residual_diagnostics = None
            if score_successful_fits:
                residual_kwargs = dict(residual_diagnostic_kwargs)
                if diagnostic_report is not None:
                    report_frequency = diagnostic_report.get("fixed_frequency")
                    if (
                        "frequency" not in residual_kwargs
                        and "period" not in residual_kwargs
                        and report_frequency is not None
                    ):
                        residual_kwargs["frequency"] = report_frequency
                try:
                    residual_diagnostics = compute_wavelength_residual_diagnostics(
                        target,
                        **residual_kwargs,
                    )
                except Exception as exc:  # diagnostics must not turn fit success into failure
                    residual_diagnostics = {
                        "available": False,
                        "status": "residual_diagnostics_failed",
                        "exception_type": exc.__class__.__name__,
                        "exception_message": str(exc),
                        "predictive_score": {"available": False},
                    }
            results.append(
                _comparison_result_entry(
                    candidate=candidate,
                    status="success",
                    fit_kwargs=fit_kwargs,
                    elapsed_seconds=elapsed,
                    target_lightcurve=target,
                    fit_result=fit_result,
                    residual_diagnostics=residual_diagnostics,
                )
            )

    scoring_summary = _score_comparison_results(results)
    report = {
        "kind": "wavelength_model_comparison",
        "stage": "model_comparison",
        "summary": {
            "n_model_kernel_config_entries": len(normalised),
            "n_model_kernel_configs": int(n_model_kernel_configs),
            "n_successful": int(n_successful),
            "n_failed": int(n_failed),
            "n_skipped": int(n_skipped),
            "all_model_kernel_configs_succeeded": bool(
                n_model_kernel_configs > 0 and n_successful == n_model_kernel_configs
            ),
            "any_model_kernel_config_succeeded": bool(n_successful > 0),
            "n_scored_successful": scoring_summary["n_scored_successful"],
            "best_model_kernel_config": scoring_summary["best_model_kernel_config"],
            "selection_status": scoring_summary["selection_status"],
            "selection_basis": scoring_summary["selection_basis"],
        },
        "results": results,
        "warnings": [
            "Training-point residual and predictive scores are diagnostic, not "
            "a final scientific model-selection decision. Prefer simpler, more "
            "interpretable candidates when scores are indistinguishable."
        ],
    }
    if diagnostic_report is not None:
        report["diagnostic_classification"] = diagnostic_report.get("classification")
    if interpret_results:
        report["interpretation"] = interpret_wavelength_model_comparison(
            report,
            **interpretation_kwargs,
        )
    return _clean_scalar_dict(report)




# -----------------------------------------------------------------------------
# Period-independent wavelength diagnostics (Level 0)
# -----------------------------------------------------------------------------

def _piwd_float(value: Any) -> float | None:
    """Return a finite float, otherwise None."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _piwd_quantile(y: np.ndarray, q: float) -> float | None:
    """Finite-sample quantile with JSON-safe missing output."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return None
    return float(np.percentile(y, q))


def _piwd_ratio(values: list[float | None]) -> float | None:
    """Return max/min for positive finite values, otherwise None."""
    finite = np.asarray(
        [float(v) for v in values if v is not None and np.isfinite(float(v))],
        dtype=float,
    )
    finite = finite[finite > 0.0]
    if finite.size < 2:
        return None
    lower = float(np.nanmin(finite))
    if lower <= 0.0:
        return None
    return float(np.nanmax(finite) / lower)


def _piwd_loglog_slope(
    wavelengths: list[float | None], values: list[float | None]
) -> float | None:
    """Fit a diagnostic log(value)-log(wavelength) slope."""
    pairs: list[tuple[float, float]] = []
    for wl, value in zip(wavelengths, values, strict=False):
        wl_f = _piwd_float(wl)
        value_f = _piwd_float(value)
        if wl_f is None or value_f is None:
            continue
        if wl_f <= 0.0 or value_f <= 0.0:
            continue
        pairs.append((wl_f, value_f))
    if len(pairs) < 2:
        return None
    x = np.log(np.asarray([p[0] for p in pairs], dtype=float))
    y = np.log(np.asarray([p[1] for p in pairs], dtype=float))
    if np.nanmax(x) == np.nanmin(x):
        return None
    coeff = np.polyfit(x, y, deg=1)
    return float(coeff[0])


def _piwd_monotonic_class(
    wavelengths: list[float | None],
    values: list[float | None],
    *,
    flat_tolerance_fraction: float = 0.10,
) -> str:
    """Descriptively classify a wavelength trend without assuming a period.

    This is a tolerance-based exploratory label, not a formal statistical test
    of monotonicity/non-monotonicity.  A ``non_monotonic`` result should be
    treated as a soft warning that flexible wavelength structure may be worth
    testing, not as a hard rejection of monotonic dust/power-law models and not
    as a constraint on the fitted wavelength solution.

    TODO(feature): add an uncertainty-aware monotonicity diagnostic, e.g. a
    bootstrap/isotonic-regression comparison against monotonic increasing and
    monotonic decreasing null trends, plus rank-correlation trend diagnostics.
    """
    pairs: list[tuple[float, float]] = []
    for wl, value in zip(wavelengths, values, strict=False):
        wl_f = _piwd_float(wl)
        value_f = _piwd_float(value)
        if wl_f is None or value_f is None:
            continue
        pairs.append((wl_f, value_f))
    if len(pairs) < 2:
        return "unavailable"

    pairs.sort(key=lambda item: item[0])
    y = np.asarray([p[1] for p in pairs], dtype=float)
    span = float(np.nanmax(y) - np.nanmin(y))
    scale = max(float(np.nanmax(np.abs(y))), 1.0)
    if span <= flat_tolerance_fraction * scale:
        return "approximately_flat"

    # Descriptive tolerance rule only.  This deliberately avoids claiming
    # statistical evidence for non-monotonicity from noisy/sparsely sampled
    # per-band summaries.  Downstream model-planning code must treat
    # ``non_monotonic`` as advisory unless/until the TODO(feature) statistical
    # monotonicity test is implemented.
    tolerance = flat_tolerance_fraction * span
    diffs = np.diff(y)
    if np.all(diffs >= -tolerance):
        return "increasing"
    if np.all(diffs <= tolerance):
        return "decreasing"
    return "non_monotonic"


def _piwd_noise_corrected(value: float | None, noise: float | None) -> float | None:
    """Quadrature noise correction for positive scale-like quantities."""
    value_f = _piwd_float(value)
    noise_f = _piwd_float(noise)
    if value_f is None or noise_f is None:
        return None
    if value_f < 0.0 or noise_f < 0.0:
        return None
    return float(np.sqrt(max(0.0, value_f * value_f - noise_f * noise_f)))


def _period_independent_band_summary(
    y: np.ndarray,
    yerr: np.ndarray | None,
    *,
    min_points: int,
) -> dict[str, Any]:
    """Robust per-band flux statistics that do not use periods or phases."""
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    yf = y[finite]

    result: dict[str, Any] = {
        "available": False,
        "status": "insufficient_finite_flux",
        "n_finite_flux": int(yf.size),
        "median_flux": None,
        "trimmed_mean_flux_10_90": None,
        "mean_flux": None,
        "q02_5_flux": None,
        "q05_flux": None,
        "q10_flux": None,
        "q16_flux": None,
        "q25_flux": None,
        "q50_flux": None,
        "q75_flux": None,
        "q84_flux": None,
        "q90_flux": None,
        "q95_flux": None,
        "q97_5_flux": None,
        "mad_scatter": None,
        "iqr_scatter": None,
        "robust_scatter": None,
        "raw_peak_to_peak_q02_5_q97_5": None,
        "raw_half_amplitude_q02_5_q97_5": None,
        "raw_peak_to_peak_q05_q95": None,
        "raw_half_amplitude_q05_q95": None,
        "raw_peak_to_peak_q10_q90": None,
        "raw_half_amplitude_q10_q90": None,
        "fractional_half_amplitude_q02_5_q97_5": None,
        "fractional_half_amplitude_q05_q95": None,
        "fractional_half_amplitude_q10_q90": None,
        "median_yerr": None,
        "mean_yerr": None,
        "noise_corrected_robust_scatter": None,
        "noise_corrected_half_amplitude_q02_5_q97_5": None,
        "noise_corrected_half_amplitude_q05_q95": None,
        "noise_corrected_half_amplitude_q10_q90": None,
    }

    if yf.size < min_points:
        return result

    q02_5, q05, q10, q16, q25, q50, q75, q84, q90, q95, q97_5 = np.percentile(
        yf, [2.5, 5.0, 10.0, 16.0, 25.0, 50.0, 75.0, 84.0, 90.0, 95.0, 97.5]
    )
    trimmed = yf[(yf >= q10) & (yf <= q90)]
    if trimmed.size == 0:
        trimmed = yf

    mad = float(1.4826 * np.median(np.abs(yf - q50)))
    iqr_scatter = float(0.7413 * (q75 - q25))
    try:
        robust = float(robust_scale(yf))
    except Exception:
        robust = mad

    amp_02_97 = float(0.5 * (q97_5 - q02_5))
    amp_05_95 = float(0.5 * (q95 - q05))
    amp_10_90 = float(0.5 * (q90 - q10))
    median_abs = abs(float(q50))

    result.update(
        {
            "available": True,
            "status": "ok",
            "median_flux": float(q50),
            "trimmed_mean_flux_10_90": float(np.mean(trimmed)),
            "mean_flux": float(np.mean(yf)),
            "q02_5_flux": float(q02_5),
            "q05_flux": float(q05),
            "q10_flux": float(q10),
            "q16_flux": float(q16),
            "q25_flux": float(q25),
            "q50_flux": float(q50),
            "q75_flux": float(q75),
            "q84_flux": float(q84),
            "q90_flux": float(q90),
            "q95_flux": float(q95),
            "q97_5_flux": float(q97_5),
            "mad_scatter": mad,
            "iqr_scatter": iqr_scatter,
            "robust_scatter": robust,
            "raw_peak_to_peak_q02_5_q97_5": float(q97_5 - q02_5),
            "raw_half_amplitude_q02_5_q97_5": amp_02_97,
            "raw_peak_to_peak_q05_q95": float(q95 - q05),
            "raw_half_amplitude_q05_q95": amp_05_95,
            "raw_peak_to_peak_q10_q90": float(q90 - q10),
            "raw_half_amplitude_q10_q90": amp_10_90,
            "fractional_half_amplitude_q02_5_q97_5": (
                float(amp_02_97 / median_abs) if median_abs > 0.0 else None
            ),
            "fractional_half_amplitude_q05_q95": (
                float(amp_05_95 / median_abs) if median_abs > 0.0 else None
            ),
            "fractional_half_amplitude_q10_q90": (
                float(amp_10_90 / median_abs) if median_abs > 0.0 else None
            ),
        }
    )

    if yerr is not None:
        yerr = np.asarray(yerr, dtype=float)
        yef = yerr[np.isfinite(yerr) & (yerr > 0.0)]
        if yef.size > 0:
            median_yerr = float(np.median(yef))
            result["median_yerr"] = median_yerr
            result["mean_yerr"] = float(np.mean(yef))
            result["noise_corrected_robust_scatter"] = _piwd_noise_corrected(
                robust, median_yerr
            )
            # Approximate Gaussian quantile factors for half central intervals.
            result["noise_corrected_half_amplitude_q02_5_q97_5"] = _piwd_noise_corrected(
                amp_02_97, 1.95996 * median_yerr
            )
            result["noise_corrected_half_amplitude_q05_q95"] = _piwd_noise_corrected(
                amp_05_95, 1.64485 * median_yerr
            )
            result["noise_corrected_half_amplitude_q10_q90"] = _piwd_noise_corrected(
                amp_10_90, 1.28155 * median_yerr
            )

    return _clean_scalar_dict(result)


def diagnose_period_independent_wavelength_structure(
    lightcurve,
    *,
    min_points_per_band: int = 5,
) -> dict[str, Any]:
    """Diagnose wavelength structure using only per-band flux distributions.

    This is a Level-0, period-independent diagnostic.  It deliberately does not
    use Lomb-Scargle peaks, ACF peaks, consensus frequencies, phase folding, GP
    fits, or fitted model residuals.  It is therefore safe to use before and
    independently of temporal-consensus model selection.

    Parameters
    ----------
    lightcurve : pgmuvi.lightcurve.Lightcurve
        A 2-D/multiband light curve whose raw xdata has columns
        ``(time, wavelength)``.
    min_points_per_band : int, optional
        Minimum number of finite flux values required for a band's robust
        distribution summary.

    Returns
    -------
    dict
        JSON-safe diagnostics with a per-band table and cross-wavelength robust
        trend summaries.
    """
    min_points_per_band = int(min_points_per_band)
    if min_points_per_band < 2:
        raise ValueError("min_points_per_band must be at least 2.")

    x_raw = lightcurve._xdata_raw
    if x_raw.dim() != 2 or x_raw.shape[1] < 2:
        raise ValueError(
            "diagnose_period_independent_wavelength_structure() requires 2-D "
            "multiband data with raw xdata of shape (N, 2), where column 0 is "
            "time and column 1 is wavelength/band coordinate."
        )

    x_np = x_raw.detach().cpu().numpy()
    y_np = lightcurve._ydata_raw.detach().cpu().numpy()
    yerr_np = None
    if hasattr(lightcurve, "_yerr_raw") and lightcurve._yerr_raw is not None:
        yerr_np = lightcurve._yerr_raw.detach().cpu().numpy()

    wavelengths = np.unique(x_np[:, 1])
    band_table: list[dict[str, Any]] = []
    warnings: list[str] = []

    for wl in wavelengths:
        mask = x_np[:, 1] == wl
        y = np.asarray(y_np[mask], dtype=float)
        yerr = np.asarray(yerr_np[mask], dtype=float) if yerr_np is not None else None
        summary = _period_independent_band_summary(
            y,
            yerr,
            min_points=min_points_per_band,
        )
        row = {
            "wavelength": float(wl),
            "band_labels": _band_labels_for_mask(lightcurve.band, mask),
            "n_points": int(np.sum(mask)),
            "period_independent_flux_summary": summary,
        }
        band_table.append(_clean_scalar_dict(row))

    usable_rows = [
        row
        for row in band_table
        if row["period_independent_flux_summary"].get("available") is True
    ]
    usable_wavelengths = [row["wavelength"] for row in usable_rows]
    medians = [
        row["period_independent_flux_summary"].get("median_flux")
        for row in usable_rows
    ]
    raw_amp_02_97 = [
        row["period_independent_flux_summary"].get("raw_half_amplitude_q02_5_q97_5")
        for row in usable_rows
    ]
    raw_amp_05_95 = [
        row["period_independent_flux_summary"].get("raw_half_amplitude_q05_q95")
        for row in usable_rows
    ]
    raw_amp_10_90 = [
        row["period_independent_flux_summary"].get("raw_half_amplitude_q10_q90")
        for row in usable_rows
    ]
    scatters = [
        row["period_independent_flux_summary"].get("robust_scatter")
        for row in usable_rows
    ]
    nc_scatters = [
        row["period_independent_flux_summary"].get("noise_corrected_robust_scatter")
        for row in usable_rows
    ]

    if len(wavelengths) < 2:
        warnings.append(
            "Only one wavelength/band is present; wavelength structure cannot "
            "be diagnosed from this light curve."
        )
    if len(usable_rows) < 2:
        warnings.append(
            "Fewer than two bands have enough finite flux points for period-"
            "independent wavelength diagnostics."
        )
    if yerr_np is None:
        warnings.append(
            "No yerr values were provided; noise-corrected distribution summaries "
            "are unavailable."
        )

    summary = {
        "n_bands": len(wavelengths),
        "n_usable_bands": len(usable_rows),
        "usable_wavelengths": usable_wavelengths,
        "has_yerr": bool(yerr_np is not None),
        "uses_temporal_consensus": False,
        "uses_period_or_frequency": False,
        "median_flux_ratio_max_to_min_abs": _piwd_ratio([abs(v) if v is not None else None for v in medians]),
        "raw_half_amplitude_q02_5_q97_5_ratio_max_to_min": _piwd_ratio(raw_amp_02_97),
        "raw_half_amplitude_q05_q95_ratio_max_to_min": _piwd_ratio(raw_amp_05_95),
        "raw_half_amplitude_q10_q90_ratio_max_to_min": _piwd_ratio(raw_amp_10_90),
        "robust_scatter_ratio_max_to_min": _piwd_ratio(scatters),
        "noise_corrected_robust_scatter_ratio_max_to_min": _piwd_ratio(nc_scatters),
        "median_flux_monotonicity_class": _piwd_monotonic_class(
            usable_wavelengths, medians
        ),
        "raw_half_amplitude_q02_5_q97_5_monotonicity_class": _piwd_monotonic_class(
            usable_wavelengths, raw_amp_02_97
        ),
        "raw_half_amplitude_q05_q95_monotonicity_class": _piwd_monotonic_class(
            usable_wavelengths, raw_amp_05_95
        ),
        # These monotonicity classes are descriptive tolerance-rule summaries.
        # They are intentionally not p-value based and should not be used as
        # hard model-selection vetoes or hard constraints.  See the TODO in
        # _piwd_monotonic_class for the deferred uncertainty-aware feature.
        "robust_scatter_monotonicity_class": _piwd_monotonic_class(
            usable_wavelengths, scatters
        ),
        "median_flux_loglog_slope": _piwd_loglog_slope(
            usable_wavelengths, [abs(v) if v is not None else None for v in medians]
        ),
        "raw_half_amplitude_q02_5_q97_5_loglog_slope": _piwd_loglog_slope(
            usable_wavelengths, raw_amp_02_97
        ),
        "raw_half_amplitude_q05_q95_loglog_slope": _piwd_loglog_slope(
            usable_wavelengths, raw_amp_05_95
        ),
        "robust_scatter_loglog_slope": _piwd_loglog_slope(
            usable_wavelengths, scatters
        ),
    }

    return _clean_scalar_dict(
        {
            "kind": "period_independent_wavelength_diagnostics",
            "stage": "prefit_period_independent",
            "method": "robust_per_band_flux_distribution",
            "is_period_independent": True,
            "band_table": band_table,
            "summary": summary,
            "warnings": warnings,
        }
    )

def diagnose_wavelength_dependence_prefit(
    lightcurve,
    *,
    sampling_kwargs: dict[str, Any] | None = None,
    variability_kwargs: dict[str, Any] | None = None,
    frequency: float | None = None,
    period: float | None = None,
    amplitude_phase_kwargs: dict[str, Any] | None = None,
    classification_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a cheap pre-fit wavelength diagnostic report.

    Parameters
    ----------
    lightcurve : pgmuvi.lightcurve.Lightcurve
        A 2-D/multiband light curve with raw coordinates of shape ``(N, 2)``.
        Column 0 is time and column 1 is wavelength/band coordinate.
    sampling_kwargs : dict or None, optional
        Keyword arguments passed to
        :func:`pgmuvi.preprocess.quality.assess_sampling_quality`.
    variability_kwargs : dict or None, optional
        Keyword arguments passed to
        :func:`pgmuvi.preprocess.variability.is_variable`.
    frequency : float or None, optional
        Fixed temporal frequency at which to measure per-band sinusoidal
        amplitude, phase, and lag.  This should usually be a consensus
        frequency obtained from LS/ACF/consensus diagnostics.  No frequency is
        inferred automatically in this diagnostics-only function.
    period : float or None, optional
        Fixed temporal period.  Mutually exclusive with ``frequency``.
    amplitude_phase_kwargs : dict or None, optional
        Keyword arguments for the fixed-frequency sinusoid fit.  Currently
        supports ``reference_time`` and ``min_points``.
    classification_kwargs : dict or None, optional
        Keyword arguments for :func:`classify_wavelength_diagnostics`.

    Returns
    -------
    dict
        JSON-safe report containing ``band_table``, ``summary``, and
        ``warnings``.  The report is diagnostics-only: no GP model is fit.

    Raises
    ------
    ValueError
        If the light curve is not standard 2-D multiband data, or if invalid
        fixed-frequency arguments are supplied.
    """
    x_raw = lightcurve._xdata_raw
    if x_raw.dim() != 2 or x_raw.shape[1] < 2:
        raise ValueError(
            "diagnose_wavelength_dependence() requires 2-D multiband data "
            "with raw xdata of shape (N, 2), where column 0 is time and "
            "column 1 is wavelength/band coordinate."
        )

    sampling_kwargs = dict(sampling_kwargs or {})
    variability_kwargs = dict(variability_kwargs or {})
    amplitude_phase_kwargs = dict(amplitude_phase_kwargs or {})
    classification_kwargs = dict(classification_kwargs or {})
    fixed_frequency = _resolve_frequency(frequency=frequency, period=period)

    x_np = x_raw.detach().cpu().numpy()
    y_np = lightcurve._ydata_raw.detach().cpu().numpy()
    yerr_np = None
    if hasattr(lightcurve, "_yerr_raw") and lightcurve._yerr_raw is not None:
        yerr_np = lightcurve._yerr_raw.detach().cpu().numpy()

    wavelengths = np.unique(x_np[:, 1])
    band_table: list[dict[str, Any]] = []
    warnings: list[str] = []

    n_sampling_pass = 0
    n_variable = 0
    usable_wavelengths: list[float] = []
    sampling_pass_wavelengths: list[float] = []
    variable_wavelengths: list[float] = []

    for wl in wavelengths:
        mask = x_np[:, 1] == wl
        t = np.asarray(x_np[mask, 0], dtype=float)
        y = np.asarray(y_np[mask], dtype=float)
        yerr = np.asarray(yerr_np[mask], dtype=float) if yerr_np is not None else None

        sampling_pass, sampling_diag = assess_sampling_quality(
            t,
            y,
            yerr,
            verbose=False,
            **sampling_kwargs,
        )
        if sampling_pass:
            n_sampling_pass += 1
            sampling_pass_wavelengths.append(float(wl))

        variability_available = yerr is not None
        variability_diag: dict[str, Any]
        variable = None
        if variability_available:
            try:
                variable, variability_diag = is_variable(
                    y,
                    yerr,
                    **variability_kwargs,
                )
            except ValueError as exc:
                variability_available = False
                variable = None
                variability_diag = {
                    "decision": f"UNAVAILABLE: {exc}",
                    "tests_passed": {},
                }
        else:
            variability_diag = {
                "decision": "UNAVAILABLE: yerr was not provided",
                "tests_passed": {},
            }

        if variable is True:
            n_variable += 1
            variable_wavelengths.append(float(wl))

        usable = bool(sampling_pass and (variable is not False))
        if usable:
            usable_wavelengths.append(float(wl))

        table_row = {
            "wavelength": float(wl),
            "band_labels": _band_labels_for_mask(lightcurve.band, mask),
            "n_points": int(np.sum(mask)),
            "sampling_pass": bool(sampling_pass),
            "sampling_recommendation": sampling_diag.get("recommendation"),
            "sampling_warnings": list(sampling_diag.get("warnings", [])),
            "sampling_gates": sampling_diag.get("gates", {}),
            "sampling_metrics": sampling_diag.get("metrics", {}),
            "variability_available": bool(variability_available),
            "variable": variable if variable is None else bool(variable),
            "variability_decision": variability_diag.get("decision"),
            "variability_tests_passed": variability_diag.get("tests_passed", {}),
            "variability_metrics": {
                "chi2": variability_diag.get("chi2"),
                "dof": variability_diag.get("dof"),
                "p_value": variability_diag.get("p_value"),
                "fvar": variability_diag.get("fvar"),
                "stetson_k": variability_diag.get("stetson_k"),
            },
            "flux_summary": _flux_summary(y, yerr),
            "usable_for_wavelength_diagnostics": usable,
        }

        if fixed_frequency is not None:
            periodic = _fit_fixed_frequency_sinusoid(
                t,
                y,
                yerr,
                frequency=fixed_frequency,
                **amplitude_phase_kwargs,
            )
            periodic["wavelength"] = float(wl)
            table_row["fixed_frequency_diagnostics"] = periodic

        band_table.append(_clean_scalar_dict(table_row))

    if len(wavelengths) < 2:
        warnings.append(
            "Only one wavelength/band is present; wavelength dependence cannot "
            "be diagnosed from this light curve."
        )
    if len(usable_wavelengths) < 2:
        warnings.append(
            "Fewer than two bands are usable after sampling/variability checks; "
            "do not claim wavelength-dependent variability from this report alone."
        )
    if yerr_np is None:
        warnings.append(
            "No yerr values were provided; statistical variability tests were not "
            "run and amplitude diagnostics are not noise-corrected."
        )

    summary = {
        "n_bands": len(wavelengths),
        "n_sampling_pass": int(n_sampling_pass),
        "n_variable": int(n_variable),
        "n_usable_for_wavelength_diagnostics": len(usable_wavelengths),
        "sampling_pass_wavelengths": sampling_pass_wavelengths,
        "variable_wavelengths": variable_wavelengths,
        "usable_wavelengths": usable_wavelengths,
        "has_band_labels": bool(lightcurve.band is not None),
        "has_yerr": bool(yerr_np is not None),
    }

    report = {
        "kind": "wavelength_dependence_prefit_diagnostics",
        "stage": "prefit",
        "band_table": band_table,
        "summary": _clean_scalar_dict(summary),
        "warnings": warnings,
    }

    if fixed_frequency is not None:
        report["fixed_frequency"] = float(fixed_frequency)
        report["fixed_period"] = float(1.0 / fixed_frequency)
        report["amplitude_phase_summary"] = _amplitude_phase_summary(band_table)

    classification = classify_wavelength_diagnostics(report, **classification_kwargs)
    report["classification"] = classification
    report["recommended_candidate_models"] = classification[
        "recommended_candidate_models"
    ]

    return report


# -----------------------------------------------------------------------------
# Period-independent wavelength initialization / constraint planning (Level 0b)
# -----------------------------------------------------------------------------

def _piwd_plan_positive_floor(values: list[float | None], floor: float) -> float:
    finite = []
    for value in values:
        value_f = _piwd_float(value)
        if value_f is not None and value_f > 0.0:
            finite.append(value_f)
    if not finite:
        return float(floor)
    return float(max(min(finite) * 1.0e-6, floor))


def _piwd_plan_extract_arrays(report: dict[str, Any]) -> dict[str, Any]:
    """Extract finite per-band arrays from a PR56 diagnostics report."""
    rows = report.get("band_table", [])
    extracted: list[dict[str, float]] = []
    for row in rows:
        summary = row.get("period_independent_flux_summary", {})
        if summary.get("available") is not True:
            continue
        wl = _piwd_float(row.get("wavelength"))
        median = _piwd_float(summary.get("median_flux"))
        amp = _piwd_float(summary.get("raw_half_amplitude_q05_q95"))
        scatter = _piwd_float(summary.get("robust_scatter"))
        q05 = _piwd_float(summary.get("q05_flux"))
        q95 = _piwd_float(summary.get("q95_flux"))
        if wl is None or median is None or wl <= 0.0:
            continue
        extracted.append(
            {
                "wavelength": wl,
                "median_flux": median,
                "raw_half_amplitude_q05_q95": amp if amp is not None else 0.0,
                "robust_scatter": scatter if scatter is not None else 0.0,
                "q05_flux": q05 if q05 is not None else median,
                "q95_flux": q95 if q95 is not None else median,
            }
        )

    extracted.sort(key=lambda item: item["wavelength"])
    return {
        "rows": extracted,
        "wavelengths": [row["wavelength"] for row in extracted],
        "median_fluxes": [row["median_flux"] for row in extracted],
        "amplitudes": [row["raw_half_amplitude_q05_q95"] for row in extracted],
        "scatters": [row["robust_scatter"] for row in extracted],
        "q05_fluxes": [row["q05_flux"] for row in extracted],
        "q95_fluxes": [row["q95_flux"] for row in extracted],
    }


def _piwd_plan_loglog_fit(
    wavelengths: list[float],
    values: list[float],
    *,
    positive_floor: float,
) -> dict[str, Any]:
    pairs: list[tuple[float, float]] = []
    for wl, value in zip(wavelengths, values, strict=False):
        wl_f = _piwd_float(wl)
        value_f = _piwd_float(value)
        if wl_f is None or value_f is None:
            continue
        if wl_f <= 0.0:
            continue
        value_abs = max(abs(value_f), positive_floor)
        pairs.append((wl_f, value_abs))

    if len(pairs) < 2:
        return {
            "available": False,
            "slope": None,
            "intercept": None,
            "normalization": None,
            "reason": "fewer_than_two_positive_wavelength_points",
        }

    x = np.log(np.asarray([p[0] for p in pairs], dtype=float))
    y = np.log(np.asarray([p[1] for p in pairs], dtype=float))
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return {
            "available": False,
            "slope": None,
            "intercept": None,
            "normalization": None,
            "reason": "nonfinite_log_values",
        }
    if float(np.nanmax(x) - np.nanmin(x)) <= 0.0:
        return {
            "available": False,
            "slope": None,
            "intercept": None,
            "normalization": None,
            "reason": "zero_wavelength_span",
        }

    slope, intercept = np.polyfit(x, y, deg=1)
    return {
        "available": True,
        "slope": float(slope),
        "intercept": float(intercept),
        "normalization": float(np.exp(intercept)),
        "reason": None,
    }


def _piwd_plan_quadratic_fit(
    wavelengths: list[float], values: list[float]) -> dict[str, Any]:
    if len(wavelengths) < 2:
        return {
            "available": False,
            "bias": None,
            "weights": None,
            "coordinate_basis": "raw_wavelength",
            "reason": "fewer_than_two_wavelength_points",
        }
    x = np.asarray(wavelengths, dtype=float)
    y = np.asarray(values, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        return {
            "available": False,
            "bias": None,
            "weights": None,
            "coordinate_basis": "raw_wavelength",
            "reason": "fewer_than_two_finite_points",
        }
    degree = 2 if x.size >= 3 else 1
    coeff = np.polyfit(x, y, deg=degree)
    if degree == 1:
        slope, intercept = coeff
        weights = [float(slope), 0.0]
        bias = float(intercept)
    else:
        quad, linear, intercept = coeff
        weights = [float(linear), float(quad)]
        bias = float(intercept)
    return {
        "available": True,
        "bias": bias,
        "weights": weights,
        "coordinate_basis": "raw_wavelength",
        "reason": None,
        "warning": (
            "Advisory only: these coefficients are in the raw wavelength basis. "
            "Do not apply directly when the fit uses a transformed wavelength coordinate."
        ),
    }


def _piwd_plan_recommended_models(
    *,
    median_trend: str,
    median_slope: float | None,
    amplitude_trend: str,
) -> list[dict[str, Any]]:
    """Rank LPV-relevant wavelength model candidates from Level-0 summaries."""
    candidates: list[dict[str, Any]] = []
    slope = _piwd_float(median_slope)

    if median_trend == "non_monotonic" or amplitude_trend == "non_monotonic":
        candidates.append(
            {
                "model": "2DWavelengthDependent",
                "rank": 1,
                "reason": (
                    "The period-independent median-flux or amplitude trend is "
                    "non-monotonic, so a flexible wavelength-dependent mean/covariance "
                    "is safer than forcing a monotonic parametric mean."
                ),
            }
        )
        candidates.append(
            {
                "model": "2DDustMean",
                "rank": 2,
                "reason": "Retain as an LPV physically motivated monotonic mean comparison.",
            }
        )
        candidates.append(
            {
                "model": "2DPowerLawMean",
                "rank": 3,
                "reason": "Retain as a simple parametric wavelength-mean comparison.",
            }
        )
    elif slope is not None and slope > 0.0:
        candidates.append(
            {
                "model": "2DDustMean",
                "rank": 1,
                "reason": (
                    "The median flux rises with wavelength, consistent with an "
                    "attenuated short-wavelength mean for dusty LPVs."
                ),
            }
        )
        candidates.append(
            {
                "model": "2DPowerLawMean",
                "rank": 2,
                "reason": "The positive log-log slope can also be represented by a power-law mean.",
            }
        )
        candidates.append(
            {
                "model": "2DWavelengthDependent",
                "rank": 3,
                "reason": "Flexible fallback if the parametric monotonic means underfit.",
            }
        )
    elif slope is not None and slope < 0.0:
        candidates.append(
            {
                "model": "2DPowerLawMean",
                "rank": 1,
                "reason": "The median flux decreases with wavelength; a signed power-law mean is the least restrictive parametric option.",
            }
        )
        candidates.append(
            {
                "model": "2DWavelengthDependent",
                "rank": 2,
                "reason": "Flexible fallback for non-power-law wavelength structure.",
            }
        )
        candidates.append(
            {
                "model": "2DDustMean",
                "rank": 3,
                "reason": "Lower priority because the simple dust mean is naturally increasing with wavelength for positive amplitude/tau/alpha.",
            }
        )
    else:
        candidates.append(
            {
                "model": "2DWavelengthDependent",
                "rank": 1,
                "reason": "The period-independent wavelength trend is flat or unavailable; use the flexible separable baseline first.",
            }
        )
        candidates.append(
            {
                "model": "2DDustMean",
                "rank": 2,
                "reason": "Retain as an LPV motivated comparison.",
            }
        )
        candidates.append(
            {
                "model": "2DPowerLawMean",
                "rank": 3,
                "reason": "Retain as a simple parametric comparison.",
            }
        )

    return candidates


def build_period_independent_wavelength_parameter_plan(
    lightcurve_or_report,
    *,
    diagnostics_report: dict[str, Any] | None = None,
    min_points_per_band: int = 5,
    padding_factor: float = 2.0,
    positive_floor: float = 1.0e-12,
) -> dict[str, Any]:
    """Build advisory initialization/constraint suggestions from PR56 diagnostics.

    The returned plan is intentionally advisory.  It does not apply parameter
    values, register constraints, run consensus, or mutate a Lightcurve.  It is
    the bridge between Level-0 wavelength diagnostics and a later explicit
    parameter-workflow integration step.
    """
    padding_factor = float(padding_factor)
    if padding_factor < 1.0:
        raise ValueError("padding_factor must be >= 1.0")
    positive_floor = float(positive_floor)
    if positive_floor <= 0.0:
        raise ValueError("positive_floor must be > 0")

    if diagnostics_report is not None:
        report = diagnostics_report
    elif isinstance(lightcurve_or_report, dict):
        report = lightcurve_or_report
    else:
        report = diagnose_period_independent_wavelength_structure(
            lightcurve_or_report,
            min_points_per_band=min_points_per_band,
        )

    if report.get("kind") != "period_independent_wavelength_diagnostics":
        raise ValueError(
            "build_period_independent_wavelength_parameter_plan() requires a "
            "period-independent wavelength diagnostics report."
        )

    arrays = _piwd_plan_extract_arrays(report)
    rows = arrays["rows"]
    wavelengths = arrays["wavelengths"]
    medians = arrays["median_fluxes"]
    amps = arrays["amplitudes"]
    scatters = arrays["scatters"]
    q05 = arrays["q05_fluxes"]
    q95 = arrays["q95_fluxes"]

    warnings = list(report.get("warnings", []))
    if len(rows) < 2:
        warnings.append(
            "Fewer than two usable bands are available; wavelength-parameter "
            "initialization suggestions are limited."
        )

    finite_flux_values = [v for v in q05 + q95 + medians if _piwd_float(v) is not None]
    if finite_flux_values:
        flux_min = float(np.nanmin(np.asarray(finite_flux_values, dtype=float)))
        flux_max = float(np.nanmax(np.asarray(finite_flux_values, dtype=float)))
    else:
        flux_min = 0.0
        flux_max = positive_floor
    flux_span = max(float(flux_max - flux_min), positive_floor)
    median_flux = float(np.nanmedian(np.asarray(medians, dtype=float))) if medians else 0.0
    amplitude_scale = max(
        [positive_floor]
        + [float(v) for v in amps if _piwd_float(v) is not None and float(v) > 0.0]
    )
    scatter_scale = max(
        [positive_floor]
        + [float(v) for v in scatters if _piwd_float(v) is not None and float(v) > 0.0]
    )
    floor = _piwd_plan_positive_floor(medians + amps + scatters, positive_floor)

    loglog = _piwd_plan_loglog_fit(
        wavelengths,
        medians,
        positive_floor=floor,
    )
    median_slope = loglog.get("slope")
    alpha_guess = abs(float(median_slope)) if median_slope is not None else 1.7
    alpha_guess = float(np.clip(alpha_guess, 0.1, 10.0))

    offset_low = float(flux_min - padding_factor * flux_span)
    offset_high = float(flux_max + padding_factor * flux_span)
    positive_amp_high = float(max(padding_factor * flux_span, amplitude_scale, floor))

    powerlaw_weight = loglog.get("normalization")
    if powerlaw_weight is None:
        powerlaw_weight = float(np.sign(median_flux) * max(abs(median_flux), floor))
    powerlaw_exponent = float(median_slope) if median_slope is not None else 0.0

    dust_offset = max(0.0, float(flux_min - 0.25 * flux_span))
    dust_amplitude = max(float(flux_max - dust_offset), amplitude_scale, floor)

    quadratic = _piwd_plan_quadratic_fit(wavelengths, medians)

    median_trend = report.get("summary", {}).get("median_flux_monotonicity_class")
    amplitude_trend = report.get("summary", {}).get(
        "raw_half_amplitude_q05_q95_monotonicity_class"
    )
    candidates = _piwd_plan_recommended_models(
        median_trend=str(median_trend),
        median_slope=median_slope,
        amplitude_trend=str(amplitude_trend),
    )

    suggestions = {
        "2DPowerLawMean": {
            "parameter_basis": "physical_flux_and_raw_wavelength_advisory",
            "initial_values": {
                "mean_module.offset": 0.0,
                "mean_module.weight": float(powerlaw_weight),
                "mean_module.exponent": powerlaw_exponent,
            },
            "constraints": {
                "mean_module.offset": [offset_low, offset_high],
                "mean_module.weight": [-positive_amp_high, positive_amp_high],
                "mean_module.exponent": [-10.0, 10.0],
            },
        },
        "2DDustMean": {
            "parameter_basis": "physical_flux_advisory",
            "initial_values": {
                "mean_module.offset": float(dust_offset),
                "mean_module.log_amplitude": float(np.log(dust_amplitude)),
                "mean_module.log_tau": 0.0,
                "mean_module.log_alpha": float(np.log(alpha_guess)),
            },
            "physical_initial_values": {
                "amplitude": float(dust_amplitude),
                "tau": 1.0,
                "alpha": float(alpha_guess),
            },
            "constraints": {
                "mean_module.offset": [max(0.0, offset_low), max(offset_high, floor)],
                "mean_module.log_amplitude": [float(np.log(floor)), float(np.log(positive_amp_high))],
                "mean_module.log_tau": [float(np.log(1.0e-3)), float(np.log(1.0e3))],
                "mean_module.log_alpha": [float(np.log(0.1)), float(np.log(10.0))],
            },
        },
        "2DWavelengthDependent": {
            "parameter_basis": quadratic.get("coordinate_basis"),
            "initial_values": {
                "mean_module.bias": quadratic.get("bias"),
                "mean_module.weights": quadratic.get("weights"),
            },
            "constraints": {
                "mean_module.bias": [offset_low, offset_high],
                "mean_module.weights": None,
            },
            "warning": quadratic.get("warning"),
        },
        "shared_wavelength_kernel": {
            "parameter_basis": "raw_wavelength_advisory",
            "initial_values": {
                "wavelength_lengthscale": (
                    float(0.5 * (max(wavelengths) - min(wavelengths)))
                    if len(wavelengths) >= 2
                    else None
                )
            },
            "constraints": {
                "wavelength_lengthscale": (
                    [positive_floor, float(max(wavelengths) - min(wavelengths)) * padding_factor]
                    if len(wavelengths) >= 2
                    else None
                )
            },
        },
        "shared_flux_scales": {
            "median_flux": median_flux,
            "flux_min": flux_min,
            "flux_max": flux_max,
            "flux_span": flux_span,
            "amplitude_scale_q05_q95": amplitude_scale,
            "robust_scatter_scale": scatter_scale,
        },
    }

    return _clean_scalar_dict(
        {
            "kind": "period_independent_wavelength_parameter_plan",
            "stage": "prefit_period_independent_advisory",
            "source_report_kind": report.get("kind"),
            "is_period_independent": True,
            "uses_temporal_consensus": False,
            "uses_period_or_frequency": False,
            "applies_to_fit": False,
            "advisory_only": True,
            "hard_model_exclusions": False,
            "automatic_constraints_applied": False,
            "automatic_initialization_applied": False,
            "n_usable_bands": len(rows),
            "recommended_models": candidates,
            "ranked_candidates": [
                {
                    "rank": candidate.get("rank"),
                    "model": candidate.get("model"),
                    "recommendation_strength": "advisory",
                    "hard_exclusion": False,
                    "primary_reason": candidate.get("reason"),
                    "reason": candidate.get("reason"),
                }
                for candidate in candidates
            ],
            "primary_recommended_model": candidates[0]["model"] if candidates else None,
            "summary": {
                "median_flux_monotonicity_class": median_trend,
                "raw_half_amplitude_q05_q95_monotonicity_class": amplitude_trend,
                "median_flux_loglog_slope": median_slope,
                "median_flux_loglog_intercept": loglog.get("intercept"),
                "median_flux_loglog_normalization": loglog.get("normalization"),
                "flux_min": flux_min,
                "flux_max": flux_max,
                "flux_span": flux_span,
                "amplitude_scale_q05_q95": amplitude_scale,
                "robust_scatter_scale": scatter_scale,
            },
            "model_parameter_suggestions": suggestions,
            "warnings": warnings,
            "notes": [
                "This is an advisory Level-0 plan. It does not mutate the Lightcurve, register constraints, set hypers, or run a GP fit.",
                "Parameter suggestions are derived from raw per-band flux distributions, not from temporal consensus or phase-folded amplitudes.",
                "Raw-wavelength polynomial coefficients must not be applied blindly if the fit uses a transformed wavelength coordinate.",
            ],
        }
    )


# -----------------------------------------------------------------------------
# Period-independent wavelength model/kernel-config config generation (Level 0c)
# -----------------------------------------------------------------------------

_PIWD_LPV_SEPARABLE_MODELS = {
    "2DDustMean",
    "2DPowerLawMean",
    "2DWavelengthDependent",
    "2DSeparable",
}


def _piwd_candidate_model_entries(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Return ranked model recommendation entries from a PR57 plan."""
    entries = plan.get("ranked_candidates")
    if not entries:
        entries = plan.get("recommended_models", [])
    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            continue
        model = entry.get("model")
        if not model:
            continue
        normalized.append(
            {
                "rank": int(entry.get("rank") or index),
                "model": str(model),
                "recommendation_strength": str(
                    entry.get("recommendation_strength") or "advisory"
                ),
                "hard_exclusion": bool(entry.get("hard_exclusion", False)),
                "primary_reason": entry.get("primary_reason") or entry.get("reason"),
                "reason": entry.get("reason") or entry.get("primary_reason"),
            }
        )
    normalized.sort(key=lambda item: item["rank"])
    return normalized


def _piwd_candidate_fit_kwargs(
    model: str,
    *,
    base_fit_kwargs: dict[str, Any] | None,
    fit_strategy: str,
    lpv_time_kernel_type: str,
    learn_additional_noise: bool | None,
) -> dict[str, Any]:
    """Build explicit but non-executed fit kwargs for one candidate model."""
    kwargs: dict[str, Any] = {}
    if base_fit_kwargs:
        kwargs.update(dict(base_fit_kwargs))

    kwargs["model"] = model
    kwargs.setdefault("fit_strategy", fit_strategy)

    if learn_additional_noise is not None:
        kwargs.setdefault("learn_additional_noise", bool(learn_additional_noise))

    # PR55 made the LPV separable model family compatible with consensus via a
    # period_length handoff.  Use that path by default for these candidate
    # configs.  The full 2D baseline keeps its existing spectral-mixture default
    # and therefore should not receive a time_kernel_type kwarg unless the user
    # explicitly supplied one in base_fit_kwargs.
    if model in _PIWD_LPV_SEPARABLE_MODELS:
        kwargs.setdefault("time_kernel_type", lpv_time_kernel_type)

    return kwargs


def build_period_independent_wavelength_model_kernel_configs(
    lightcurve_or_plan,
    *,
    parameter_plan: dict[str, Any] | None = None,
    diagnostics_report: dict[str, Any] | None = None,
    include_models: list[str] | tuple[str, ...] | None = None,
    include_2d_baseline: bool = True,
    base_fit_kwargs: dict[str, Any] | None = None,
    fit_strategy: str = "consensus",
    lpv_time_kernel_type: str = "quasi_periodic",
    learn_additional_noise: bool | None = True,
    model_kernel_config_limit: int | None = None,
    include_parameter_suggestions: bool = True,
) -> dict[str, Any]:
    """Build advisory model/kernel-config configurations from a PR57 plan.

    The returned object is a planning artifact only.  It does not call ``fit``,
    mutate the Lightcurve, set hyperparameters, register constraints, or apply
    any PR57 parameter suggestions.  The suggestions are copied into each
    candidate only as metadata for user inspection or for a later explicit
    parameter-workflow PR.
    """
    if parameter_plan is not None:
        plan = parameter_plan
    elif isinstance(lightcurve_or_plan, dict):
        if lightcurve_or_plan.get("kind") == "period_independent_wavelength_parameter_plan":
            plan = lightcurve_or_plan
        else:
            raise ValueError(
                "build_period_independent_wavelength_model_kernel_configs() requires a "
                "period-independent wavelength parameter plan when a dict is passed."
            )
    else:
        plan = build_period_independent_wavelength_parameter_plan(
            lightcurve_or_plan,
            diagnostics_report=diagnostics_report,
        )

    if plan.get("kind") != "period_independent_wavelength_parameter_plan":
        raise ValueError(
            "build_period_independent_wavelength_model_kernel_configs() requires a "
            "period-independent wavelength parameter plan."
        )

    include_set = {str(model) for model in include_models} if include_models else None
    suggestions = plan.get("model_parameter_suggestions", {})

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in _piwd_candidate_model_entries(plan):
        model = entry["model"]
        if include_set is not None and model not in include_set:
            continue
        if model in seen:
            continue
        seen.add(model)

        fit_kwargs = _piwd_candidate_fit_kwargs(
            model,
            base_fit_kwargs=base_fit_kwargs,
            fit_strategy=fit_strategy,
            lpv_time_kernel_type=lpv_time_kernel_type,
            learn_additional_noise=learn_additional_noise,
        )
        candidate = {
            "model_kernel_config_id": f"rank{len(candidates) + 1}_{model}",
            "rank": len(candidates) + 1,
            "source": "parameter_plan",
            "source_plan_rank": entry.get("rank"),
            "model": model,
            "fit_kwargs": fit_kwargs,
            "recommendation_strength": entry.get("recommendation_strength", "advisory"),
            "hard_exclusion": False,
            "primary_reason": entry.get("primary_reason"),
            "reason": entry.get("reason"),
            "applies_parameter_suggestions": False,
            "parameter_suggestions_applied": False,
            "applies_constraints": False,
            "parameter_suggestions": (
                suggestions.get(model) if include_parameter_suggestions else None
            ),
        }
        candidates.append(candidate)

    if include_2d_baseline and (include_set is None or "2D" in include_set) and "2D" not in seen:
        baseline_kwargs = _piwd_candidate_fit_kwargs(
            "2D",
            base_fit_kwargs=base_fit_kwargs,
            fit_strategy=fit_strategy,
            lpv_time_kernel_type=lpv_time_kernel_type,
            learn_additional_noise=learn_additional_noise,
        )
        candidates.append(
            {
                "model_kernel_config_id": f"rank{len(candidates) + 1}_2D_baseline",
                "rank": len(candidates) + 1,
                "source": "baseline_comparison",
                "source_plan_rank": None,
                "model": "2D",
                "fit_kwargs": baseline_kwargs,
                "recommendation_strength": "baseline_comparison",
                "hard_exclusion": False,
                "primary_reason": (
                    "Full 2D spectral-mixture baseline retained as a comparison "
                    "against the LPV separable wavelength models."
                ),
                "reason": (
                    "Full 2D spectral-mixture baseline retained as a comparison "
                    "against the LPV separable wavelength models."
                ),
                "applies_parameter_suggestions": False,
                "parameter_suggestions_applied": False,
                "applies_constraints": False,
                "parameter_suggestions": None,
            }
        )

    if model_kernel_config_limit is not None:
        model_kernel_config_limit = int(model_kernel_config_limit)
        if model_kernel_config_limit < 1:
            raise ValueError("model_kernel_config_limit must be >= 1 when provided")
        candidates = candidates[:model_kernel_config_limit]
        for index, candidate in enumerate(candidates, start=1):
            candidate["rank"] = index
            candidate["model_kernel_config_id"] = f"rank{index}_{candidate['model']}"

    return _clean_scalar_dict(
        {
            "kind": "period_independent_wavelength_model_kernel_configs",
            "stage": "prefit_period_independent_model_kernel_config",
            "source_plan_kind": plan.get("kind"),
            "is_period_independent": True,
            "uses_temporal_consensus": False,
            "uses_period_or_frequency": False,
            "applies_to_fit": False,
            "advisory_only": True,
            "runs_fits": False,
            "hard_model_exclusions": False,
            "automatic_constraints_applied": False,
            "automatic_initialization_applied": False,
            "n_model_kernel_configs": len(candidates),
            "primary_model_kernel_config_model": candidates[0]["model"] if candidates else None,
            "model_kernel_configs": candidates,
            "notes": [
                "This object only contains model/kernel config fit kwargs; no GP fit has been run.",
                "Parameter suggestions are metadata only and are not inserted into fit_kwargs.",
                "LPV separable candidates default to time_kernel_type='quasi_periodic' to use the PR55 period_length handoff.",
                "The 2D baseline keeps its existing spectral-mixture time-kernel default unless base_fit_kwargs overrides it.",
            ],
        }
    )


# -----------------------------------------------------------------------------
# Period-independent wavelength model/kernel-config execution (Level 0d)
# -----------------------------------------------------------------------------

def _piwd_validate_model_kernel_config_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return model/kernel configs from a PR58 report, or raise a schema error."""
    if not isinstance(report, dict) or report.get("kind") != "period_independent_wavelength_model_kernel_configs":
        raise ValueError(
            "run_period_independent_wavelength_model_kernel_configs() requires a "
            "period-independent wavelength model/kernel-config report."
        )
    candidates = report.get("model_kernel_configs")
    if not isinstance(candidates, list):
        raise ValueError("model/kernel-config report must contain a 'model_kernel_configs' list")
    normalized: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            raise ValueError("every model/kernel config must be a dict")
        fit_kwargs = candidate.get("fit_kwargs")
        if not isinstance(fit_kwargs, dict):
            raise ValueError("every model/kernel config must contain a fit_kwargs dict")
        if not candidate.get("model") and not fit_kwargs.get("model"):
            raise ValueError("every model/kernel config must specify a model")
        copied = dict(candidate)
        copied["rank"] = int(copied.get("rank") or index)
        copied["model"] = str(copied.get("model") or fit_kwargs.get("model"))
        copied["fit_kwargs"] = dict(fit_kwargs)
        normalized.append(copied)
    return normalized




def _piwd_numpy_1d_or_none(value: Any) -> np.ndarray | None:
    """Return a finite 1-D numpy array when possible."""
    if value is None:
        return None
    try:
        import torch as _torch

        if isinstance(value, _torch.Tensor):
            value = value.detach().cpu().numpy()
    except Exception:
        pass
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return arr


def _piwd_safe_float(value: Any) -> float | None:
    """Return a finite float or None."""
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _piwd_training_fit_quality_unavailable(reason: str) -> dict[str, Any]:
    """Return the standard unavailable training-fit-quality payload."""
    return _clean_scalar_dict(
        {
            "available": False,
            "reason": reason,
            "space": "transformed_training_space",
            "metrics_used_for_quality_scoring": [],
            "n_points": 0,
        }
    )


def _piwd_compute_training_fit_quality(fitted_lightcurve: Any) -> dict[str, Any]:
    """Compute best-effort training-space residual diagnostics for a fitted GP.

    These diagnostics are intentionally descriptive.  They use predictions at
    the training coordinates in the transformed space used by the GP.  They are
    not cross-validation, AIC, BIC, or posterior predictive checks, but they are
    real fit-quality quantities and can distinguish model/kernel config fits that all pass
    the consensus/viability checks. Standardized residuals use the observed
    likelihood-wrapped predictive variance once; stored measurement errors are
    only a fallback when that variance is unavailable.
    """
    if fitted_lightcurve is None:
        return _piwd_training_fit_quality_unavailable("no fitted Lightcurve was provided")
    if not hasattr(fitted_lightcurve, "model") or not hasattr(fitted_lightcurve, "likelihood"):
        return _piwd_training_fit_quality_unavailable("fitted Lightcurve has no model/likelihood")
    if not hasattr(fitted_lightcurve, "_xdata_transformed") or not hasattr(fitted_lightcurve, "_ydata_transformed"):
        return _piwd_training_fit_quality_unavailable("fitted Lightcurve has no transformed training data")

    try:
        import torch as _torch
        import gpytorch as _gpytorch

        with _torch.no_grad(), _gpytorch.settings.fast_pred_var():
            if hasattr(fitted_lightcurve, "_eval"):
                fitted_lightcurve._eval()
            prediction = fitted_lightcurve.likelihood(
                fitted_lightcurve.model(fitted_lightcurve._xdata_transformed)
            )
            mean = prediction.mean.detach().cpu().numpy().reshape(-1)
            variance = getattr(prediction, "variance", None)
            if variance is not None:
                pred_var = variance.detach().cpu().numpy().reshape(-1)
            else:
                pred_var = None
            y = fitted_lightcurve._ydata_transformed.detach().cpu().numpy().reshape(-1)
    except Exception as exc:
        return _piwd_training_fit_quality_unavailable(
            f"could not evaluate training predictions: {type(exc).__name__}: {exc}"
        )

    if mean.shape != y.shape or y.size == 0:
        return _piwd_training_fit_quality_unavailable("prediction and target shapes are incompatible")

    finite = np.isfinite(mean) & np.isfinite(y)
    if not np.any(finite):
        return _piwd_training_fit_quality_unavailable("no finite prediction/target pairs")

    mean = mean[finite]
    y = y[finite]
    residual = y - mean
    n = int(residual.size)
    y_std = float(np.std(y)) if n > 1 else 0.0
    y_iqr = float(np.subtract(*np.percentile(y, [75, 25]))) if n > 1 else 0.0
    robust_scale = y_iqr / 1.349 if y_iqr > 0 else y_std
    if not np.isfinite(robust_scale) or robust_scale <= 0:
        robust_scale = float(np.mean(np.abs(y))) if np.mean(np.abs(y)) > 0 else 1.0

    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    med_abs = float(np.median(np.abs(residual)))
    bias = float(np.mean(residual))
    resid_std = float(np.std(residual)) if n > 1 else 0.0
    nrmse = float(rmse / robust_scale) if robust_scale > 0 else None

    yerr = _piwd_numpy_1d_or_none(
        getattr(fitted_lightcurve, "_yerr_transformed", None)
    )
    if yerr is not None and yerr.size == finite.size:
        yerr = yerr[finite]
    else:
        yerr = None

    pred_std = None
    if pred_var is not None and pred_var.shape == finite.shape:
        pred_var = pred_var[finite]
        pred_var = np.where(np.isfinite(pred_var) & (pred_var > 0), pred_var, np.nan)
        pred_std = np.sqrt(pred_var)

    sigma = None
    sigma_source = None
    predictive_variance_kind = None
    if yerr is not None:
        yerr = np.where(np.isfinite(yerr) & (yerr > 0), yerr, np.nan)
    if pred_std is not None and np.any(np.isfinite(pred_std) & (pred_std > 0)):
        sigma = pred_std.copy()
        sigma_source = "observed_predictive_standard_deviation"
        predictive_variance_kind = "observed"
        if yerr is not None:
            fill = (
                (~np.isfinite(sigma) | (sigma <= 0))
                & np.isfinite(yerr)
                & (yerr > 0)
            )
            if np.any(fill):
                sigma[fill] = yerr[fill]
                sigma_source = (
                    "observed_predictive_standard_deviation_with_"
                    "transformed_yerr_fallback"
                )
    elif yerr is not None and np.any(np.isfinite(yerr) & (yerr > 0)):
        sigma = yerr.copy()
        sigma_source = "transformed_measurement_uncertainty_fallback"

    normalized_rmse = None
    median_abs_standardized_residual = None
    outlier_fraction_3sigma = None
    reduced_chi2 = None
    if sigma is not None:
        ok = np.isfinite(sigma) & (sigma > 0)
        if np.any(ok):
            std_resid = residual[ok] / sigma[ok]
            normalized_rmse = float(np.sqrt(np.mean(std_resid**2)))
            median_abs_standardized_residual = float(np.median(np.abs(std_resid)))
            outlier_fraction_3sigma = float(np.mean(np.abs(std_resid) > 3.0))
            dof = max(int(std_resid.size) - 1, 1)
            reduced_chi2 = float(np.sum(std_resid**2) / dof)

    log_mll = None
    try:
        import torch as _torch
        import gpytorch as _gpytorch

        with _torch.no_grad():
            output = fitted_lightcurve.model(fitted_lightcurve._xdata_transformed)
            mll = _gpytorch.mlls.ExactMarginalLogLikelihood(
                fitted_lightcurve.likelihood, fitted_lightcurve.model
            )
            value = mll(output, fitted_lightcurve._ydata_transformed)
            log_mll = _piwd_safe_float(value.detach().cpu().item())
    except Exception:
        log_mll = None

    by_band = []
    try:
        x_raw = fitted_lightcurve.xdata
        import torch as _torch

        if isinstance(x_raw, _torch.Tensor) and x_raw.ndim == 2 and x_raw.shape[0] == finite.size:
            wavelengths = x_raw.detach().cpu().numpy().reshape((x_raw.shape[0], x_raw.shape[1]))[:, 1]
            wavelengths = wavelengths[finite]
            for wl in sorted(set(float(v) for v in wavelengths if np.isfinite(v))):
                mask = np.isclose(wavelengths, wl)
                if not np.any(mask):
                    continue
                r = residual[mask]
                by_band.append(
                    {
                        "wavelength": wl,
                        "n_points": int(r.size),
                        "rmse": float(np.sqrt(np.mean(r**2))),
                        "mae": float(np.mean(np.abs(r))),
                        "median_abs_residual": float(np.median(np.abs(r))),
                        "bias": float(np.mean(r)),
                    }
                )
    except Exception:
        by_band = []

    return _clean_scalar_dict(
        {
            "available": True,
            "reason": None,
            "space": "transformed_training_space",
            "metrics_used_for_quality_scoring": [
                "training_normalized_rmse",
                "training_median_abs_standardized_residual",
                "training_outlier_fraction_3sigma",
            ],
            "n_points": n,
            "rmse": rmse,
            "mae": mae,
            "median_abs_residual": med_abs,
            "bias": bias,
            "residual_std": resid_std,
            "target_robust_scale": robust_scale,
            "normalized_rmse_by_target_scale": nrmse,
            "predictive_variance_kind": predictive_variance_kind,
            "standardization_sigma_source": sigma_source,
            "measurement_uncertainty_added_separately": False,
            "normalized_rmse": normalized_rmse,
            "median_abs_standardized_residual": median_abs_standardized_residual,
            "outlier_fraction_3sigma": outlier_fraction_3sigma,
            "reduced_chi2": reduced_chi2,
            "log_marginal_likelihood": log_mll,
            "by_band": by_band,
        }
    )


def _piwd_to_numpy_array(value: Any) -> np.ndarray | None:
    """Best-effort conversion of tensors/arrays to a finite numpy array."""
    if value is None:
        return None
    try:
        if torch is not None and isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        elif hasattr(value, "detach") and hasattr(value.detach(), "cpu"):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value, dtype=float)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return arr


def _piwd_iter_kernel_like_objects(obj: Any):
    """Yield kernel-like objects reachable from a fitted lightcurve/model."""
    seen: set[int] = set()
    stack = []
    model = getattr(obj, "model", None)
    if model is not None:
        stack.extend(
            candidate
            for candidate in (
                getattr(model, "sci_kernel", None),
                getattr(model, "covar_module", None),
            )
            if candidate is not None
        )
    if obj is not None:
        stack.append(obj)

    while stack:
        item = stack.pop()
        ident = id(item)
        if ident in seen:
            continue
        seen.add(ident)
        yield item

        for attr in (
            "base_kernel",
            "data_covar_module",
            "covar_module",
            "module",
        ):
            child = getattr(item, attr, None)
            if child is not None:
                stack.append(child)

        kernels = getattr(item, "kernels", None)
        if kernels is not None:
            try:
                stack.extend(k for k in kernels if k is not None)
            except TypeError:
                pass


def _piwd_find_spectral_mixture_scale_array(fitted_lightcurve: Any) -> np.ndarray | None:
    """Return a 2-D component-by-ARD-dimension SM scale array if available."""
    for kernel in _piwd_iter_kernel_like_objects(fitted_lightcurve):
        if not hasattr(kernel, "mixture_scales"):
            continue
        arr = _piwd_to_numpy_array(kernel.mixture_scales)
        if arr is None:
            continue
        if arr.ndim == 0:
            continue
        if arr.ndim == 1:
            arr2 = arr.reshape((arr.shape[0], 1))
        elif arr.ndim == 2:
            arr2 = arr
        else:
            arr2 = arr.reshape((arr.shape[0], int(np.prod(arr.shape[1:]))))
        if arr2.size and np.all(np.isfinite(arr2)):
            return arr2.astype(float, copy=False)
    return None


def _piwd_sm_ard_dimension_names(n_dimensions: int) -> list[str]:
    """Return readable names for spectral-mixture ARD dimensions."""
    names = []
    for idx in range(int(n_dimensions)):
        if idx == 0:
            names.append("time_frequency")
        elif idx == 1:
            names.append("wavelength_frequency")
        else:
            names.append(f"ard_dimension_{idx}")
    return names


def _piwd_consensus_scale_upper_from_diagnostics(
    diagnostics: dict[str, Any],
) -> tuple[float | None, float | None]:
    """Extract lower/upper consensus SM-scale bounds from diagnostics."""
    bounds = diagnostics.get("consensus_scale_constraint_bounds")
    if not isinstance(bounds, (list, tuple)) or len(bounds) < 2:
        return None, None
    try:
        lower = float(bounds[0])
        upper = float(bounds[1])
    except (TypeError, ValueError):
        return None, None
    if not (np.isfinite(upper) and upper > 0):
        return None, None
    if not np.isfinite(lower):
        lower = None
    return lower, upper


def _piwd_extract_sm_ard_scale_diagnostics(
    fitted_lightcurve: Any,
    diagnostics: dict[str, Any] | None = None,
    *,
    ceiling_tolerance_fraction: float = 0.05,
) -> dict[str, Any]:
    """Summarize fitted SM ARD scales near the consensus scale ceiling.

    The primary use case is the full 2D spectral-mixture baseline in the
    wavelength advisory workflow.  When a consensus fit applies a single SM
    scale interval to an ARD kernel, this helper records which component and
    which ARD dimension are near the upper bound.  Dimension 0 is interpreted
    as the time-frequency axis and dimension 1 as the wavelength-frequency
    axis for 2D kernels.
    """
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    scales = _piwd_find_spectral_mixture_scale_array(fitted_lightcurve)
    if scales is None:
        return {
            "available": False,
            "reason": "fitted model does not expose spectral-mixture ARD scales",
            "fitted_sm_ard_scales": None,
            "sm_ard_dimension_names": [],
            "sm_scale_constraint_lower": None,
            "sm_scale_constraint_upper": None,
            "sm_scale_ceiling_tolerance_fraction": float(ceiling_tolerance_fraction),
            "constrained_sm_ard_components": [],
            "n_constrained_sm_ard_components": 0,
            "constrained_sm_ard_dimension_counts": {},
        }

    lower, upper = _piwd_consensus_scale_upper_from_diagnostics(diagnostics)
    dim_names = _piwd_sm_ard_dimension_names(scales.shape[1])
    tolerance = float(ceiling_tolerance_fraction)
    if not (np.isfinite(tolerance) and 0.0 <= tolerance < 1.0):
        tolerance = 0.05

    constrained = []
    counts = {name: 0 for name in dim_names}
    if upper is not None:
        threshold = (1.0 - tolerance) * upper
        for component_index in range(scales.shape[0]):
            for dimension_index in range(scales.shape[1]):
                scale = float(scales[component_index, dimension_index])
                if not np.isfinite(scale):
                    continue
                if scale >= threshold:
                    dim_name = dim_names[dimension_index]
                    counts[dim_name] += 1
                    constrained.append(
                        {
                            "component_index": int(component_index),
                            "dimension_index": int(dimension_index),
                            "dimension_name": dim_name,
                            "scale": scale,
                            "constraint_upper": float(upper),
                            "fraction_of_upper": float(scale / upper),
                            "ceiling_tolerance_fraction": tolerance,
                        }
                    )

    counts = {key: value for key, value in counts.items() if value}
    return _clean_scalar_dict(
        {
            "available": True,
            "reason": None if upper is not None else "no consensus SM scale upper bound was recorded",
            "fitted_sm_ard_scales": scales.tolist(),
            "sm_ard_dimension_names": dim_names,
            "sm_scale_constraint_lower": lower,
            "sm_scale_constraint_upper": upper,
            "sm_scale_ceiling_tolerance_fraction": tolerance,
            "constrained_sm_ard_components": constrained,
            "n_constrained_sm_ard_components": len(constrained),
            "constrained_sm_ard_dimension_counts": counts,
        }
    )


def _piwd_classify_model_kernel_config_failure(
    exception: BaseException | None,
) -> dict[str, Any]:
    """Classify a failed advisory model/kernel-config fit for reporting.

    The classification is intentionally descriptive.  It is used to make batch
    reports and fallback summaries easier to triage; it does not affect scoring,
    model ranking, fitting behavior, or automatic model selection.
    """
    if exception is None:
        return {
            "failure_stage": None,
            "failure_stage_reason": None,
            "is_consensus_failure": False,
            "is_numerical_failure": False,
            "is_input_validation_failure": False,
        }

    exception_type = type(exception).__name__
    message = str(exception)
    text = f"{exception_type} {message}".lower()

    stage = "fit_execution"
    reason = "model/kernel config fit raised an exception"
    is_consensus = False
    is_numerical = False
    is_input_validation = False

    if "consensus" in text or "period consensus" in text:
        stage = "consensus"
        reason = "fit failed while deriving or applying temporal consensus information"
        is_consensus = True
    elif (
        "notpsd" in text
        or "not positive definite" in text
        or "cholesky" in text
        or "psd" in text
        or "singular" in text
        or "nan" in text
    ):
        stage = "numerical_stability"
        reason = "fit failed due to a numerical stability or covariance-matrix issue"
        is_numerical = True
    elif (
        "constraint" in text
        or "out of bounds" in text
        or "interval" in text
        or "invalid value" in text
    ):
        stage = "parameter_constraint"
        reason = "fit failed while satisfying parameter values or constraints"
    elif (
        "sampling" in text
        or "variability" in text
        or "not enough" in text
        or "insufficient" in text
        or "no rows remain" in text
    ):
        stage = "data_quality"
        reason = "fit failed because the input data did not satisfy a quality or availability requirement"
        is_input_validation = True
    elif isinstance(exception, (ValueError, TypeError, AttributeError, KeyError)):
        stage = "input_validation"
        reason = "fit failed due to invalid or unsupported input for this model/kernel config"
        is_input_validation = True

    return {
        "failure_stage": stage,
        "failure_stage_reason": reason,
        "is_consensus_failure": bool(is_consensus),
        "is_numerical_failure": bool(is_numerical),
        "is_input_validation_failure": bool(is_input_validation),
    }

def _piwd_extract_fit_outcome(
    candidate: dict[str, Any],
    *,
    status: str,
    fitted_lightcurve=None,
    fit_result: Any = None,
    exception: BaseException | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe outcome record for one candidate execution."""
    diagnostics = None
    if fitted_lightcurve is not None:
        diagnostics = getattr(fitted_lightcurve, "consensus_diagnostics", None)
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    fit_quality = (
        _piwd_compute_training_fit_quality(fitted_lightcurve)
        if status == "passed"
        else _piwd_training_fit_quality_unavailable("model/kernel config fit did not complete")
    )
    sm_ard_diagnostics = _piwd_extract_sm_ard_scale_diagnostics(
        fitted_lightcurve,
        diagnostics,
    )
    sm_ard_counts = sm_ard_diagnostics.get("constrained_sm_ard_dimension_counts") or {}
    failure_info = _piwd_classify_model_kernel_config_failure(exception)

    outcome: dict[str, Any] = {
        "model_kernel_config_id": candidate.get("model_kernel_config_id"),
        "rank": candidate.get("rank"),
        "model": candidate.get("model"),
        "status": status,

        "fit_success": bool(status == "passed"),

        "fit_failed": bool(status == "failed"),
        "failure_stage": failure_info.get("failure_stage"),
        "failure_stage_reason": failure_info.get("failure_stage_reason"),
        "is_consensus_failure": failure_info.get("is_consensus_failure"),
        "is_numerical_failure": failure_info.get("is_numerical_failure"),
        "is_input_validation_failure": failure_info.get("is_input_validation_failure"),
        "fit_failure_diagnostics": failure_info,
        "fit_kwargs": dict(candidate.get("fit_kwargs") or {}),
        "recommendation_strength": candidate.get("recommendation_strength"),
        "hard_exclusion": bool(candidate.get("hard_exclusion", False)),
        "source": candidate.get("source"),
        "parameter_suggestions_applied": bool(
            candidate.get(
                "parameter_suggestions_applied",
                candidate.get("applies_parameter_suggestions", False),
            )
        ),
        "constraints_applied_from_plan": bool(candidate.get("applies_constraints", False)),
        "consensus_success": diagnostics.get("consensus_success"),
        "consensus_frequency": diagnostics.get("consensus_frequency"),
        "consensus_period": diagnostics.get("consensus_period"),
        "consensus_time_kernel_constraint_mode": diagnostics.get(
            "consensus_time_kernel_constraint_mode"
        ),
        "n_accepted_bands": diagnostics.get("n_accepted_bands"),
        "n_rejected_bands": diagnostics.get("n_rejected_bands"),
        "accepted_bands": diagnostics.get("accepted_bands"),
        "rejected_bands": diagnostics.get("rejected_bands"),
        "fit_result_type": type(fit_result).__name__ if fit_result is not None else None,
        "fit_quality": fit_quality,
        "fit_quality_available": fit_quality.get("available"),
        "training_rmse": fit_quality.get("rmse"),
        "training_mae": fit_quality.get("mae"),
        "training_median_abs_residual": fit_quality.get("median_abs_residual"),
        "training_normalized_rmse": fit_quality.get("normalized_rmse"),
        "training_nrmse_by_target_scale": fit_quality.get("normalized_rmse_by_target_scale"),
        "training_reduced_chi2": fit_quality.get("reduced_chi2"),
        "training_median_abs_standardized_residual": fit_quality.get("median_abs_standardized_residual"),
        "training_outlier_fraction_3sigma": fit_quality.get("outlier_fraction_3sigma"),
        "training_log_marginal_likelihood": fit_quality.get("log_marginal_likelihood"),
        "training_predictive_variance_kind": fit_quality.get(
            "predictive_variance_kind"
        ),
        "training_standardization_sigma_source": fit_quality.get(
            "standardization_sigma_source"
        ),
        "training_measurement_uncertainty_added_separately": fit_quality.get(
            "measurement_uncertainty_added_separately"
        ),
        "sm_ard_scale_diagnostics": sm_ard_diagnostics,
        "constrained_sm_ard_components": sm_ard_diagnostics.get(
            "constrained_sm_ard_components"
        ),
        "n_constrained_sm_ard_components": sm_ard_diagnostics.get(
            "n_constrained_sm_ard_components"
        ),
        "constrained_sm_ard_dimension_counts": sm_ard_counts,
        "n_constrained_sm_time_components": sm_ard_counts.get("time_frequency", 0),
        "n_constrained_sm_wavelength_components": sm_ard_counts.get(
            "wavelength_frequency", 0
        ),
    }

    if exception is not None:
        import traceback as _traceback

        outcome.update(
            {
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "traceback": "".join(
                    _traceback.format_exception(
                        type(exception), exception, exception.__traceback__
                    )
                ),
            }
        )

    return _clean_scalar_dict(outcome)


def run_period_independent_wavelength_model_kernel_configs(
    lightcurve,
    *,
    model_kernel_config_report: dict[str, Any] | None = None,
    model_kernel_config_limit: int | None = None,
    stop_on_error: bool = False,
    fit_runner=None,
    copy_lightcurve: bool = True,
    **model_kernel_config_builder_kwargs,
) -> dict[str, Any]:
    """Run PR58 wavelength model/kernel configs and return an outcome summary.

    This helper executes candidate ``fit_kwargs`` but deliberately does not
    install a winner, mutate the caller's main Lightcurve state, or apply PR57
    parameter suggestions as hyperparameters/constraints.  By default each
    candidate is run on a deep copy of the input Lightcurve.  ``fit_runner`` is
    an optional test/integration hook with signature
    ``fit_runner(lightcurve_copy, fit_kwargs, candidate)``.
    """
    if model_kernel_config_report is None:
        model_kernel_config_report = build_period_independent_wavelength_model_kernel_configs(
            lightcurve, **model_kernel_config_builder_kwargs
        )
    elif model_kernel_config_builder_kwargs:
        raise ValueError(
            "model_kernel_config_builder_kwargs may only be supplied when model_kernel_config_report is None"
        )

    candidates = _piwd_validate_model_kernel_config_report(model_kernel_config_report)

    if model_kernel_config_limit is not None:
        model_kernel_config_limit = int(model_kernel_config_limit)
        if model_kernel_config_limit < 1:
            raise ValueError("model_kernel_config_limit must be >= 1 when provided")
        candidates = candidates[:model_kernel_config_limit]

    outcomes: list[dict[str, Any]] = []
    for candidate in candidates:
        import copy as _copy

        if copy_lightcurve:
            try:
                lc_to_fit = _copy.deepcopy(lightcurve)
            except Exception as exc:  # pragma: no cover - defensive path
                raise RuntimeError(
                    "Could not deep-copy the Lightcurve for isolated model/kernel config fitting."
                ) from exc
        else:
            lc_to_fit = lightcurve

        fit_kwargs = dict(candidate.get("fit_kwargs") or {})
        try:
            if fit_runner is None:
                fit_result = lc_to_fit.fit(**fit_kwargs)
            else:
                fit_result = fit_runner(lc_to_fit, fit_kwargs, candidate)
            outcomes.append(
                _piwd_extract_fit_outcome(
                    candidate,
                    status="passed",
                    fitted_lightcurve=lc_to_fit,
                    fit_result=fit_result,
                )
            )
        except Exception as exc:
            outcomes.append(
                _piwd_extract_fit_outcome(
                    candidate,
                    status="failed",
                    fitted_lightcurve=lc_to_fit,
                    exception=exc,
                )
            )
            if stop_on_error:
                raise

    passed = [outcome for outcome in outcomes if outcome.get("status") == "passed"]
    failed = [outcome for outcome in outcomes if outcome.get("status") == "failed"]

    return _clean_scalar_dict(
        {
            "kind": "period_independent_wavelength_model_kernel_config_results",
            "stage": "model_kernel_config_fit_execution_summary",
            "source_model_kernel_config_report_kind": model_kernel_config_report.get("kind"),
            "is_period_independent": True,
            "uses_temporal_consensus": True,
            "uses_period_or_frequency": True,
            "runs_fits": True,
            "applies_to_fit": True,
            "mutates_input_lightcurve": bool(not copy_lightcurve),
            "model_kernel_config_state_isolated": bool(copy_lightcurve),
            "advisory_only": True,
            "automatic_model_selection_applied": False,
            "selected_model": None,
            "hard_model_exclusions": False,
            "automatic_constraints_applied": False,
            "automatic_initialization_applied": False,
            "parameter_suggestions_applied": False,
            "n_model_kernel_configs": len(candidates),
            "n_attempted": len(outcomes),
            "n_passed": len(passed),
            "n_failed": len(failed),
            "passed_models": [outcome.get("model") for outcome in passed],
            "failed_models": [outcome.get("model") for outcome in failed],
            "outcomes": outcomes,
            "model_kernel_config_results": outcomes,
            "notes": [
                "Candidate fits were executed for comparison/reporting only.",
                "No winning model is selected automatically.",
                "The input Lightcurve is deep-copied for each candidate by default.",
                "PR57 parameter suggestions remain metadata and are not applied as fit kwargs.",
            ],
        }
    )


# -----------------------------------------------------------------------------
# Period-independent wavelength model/kernel-config scoring (Level 0e)
# -----------------------------------------------------------------------------

def _piwd_bool_from_status(value: Any, *, status: str | None = None) -> bool:
    """Return a boolean success flag from a possibly-missing report field."""
    if isinstance(value, bool):
        return value
    if value is None and status is not None:
        return str(status) == "passed"
    return bool(value)


def _piwd_finite_float_or_none(value: Any) -> float | None:
    """Return a finite float or None."""
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _piwd_int_or_zero(value: Any) -> int:
    """Return an int count or zero when unavailable."""
    try:
        return int(value)
    except Exception:
        return 0


def _piwd_score_one_wavelength_fit_outcome(
    outcome: dict[str, Any],
    *,
    weights: dict[str, float],
) -> dict[str, Any]:
    """Score one PR59 candidate outcome without making a model-selection decision."""
    status = outcome.get("status")
    fit_success = _piwd_bool_from_status(outcome.get("fit_success"), status=status)
    consensus_success = outcome.get("consensus_success") is True
    n_accepted = _piwd_int_or_zero(outcome.get("n_accepted_bands"))
    n_rejected = _piwd_int_or_zero(outcome.get("n_rejected_bands"))
    has_exception = outcome.get("exception_type") is not None
    rank = _piwd_int_or_zero(outcome.get("rank")) or 999999

    score = 0.0
    reasons: list[str] = []

    if fit_success:
        score += weights["fit_success"]
        reasons.append("model/kernel config fit completed")
    else:
        score -= weights["fit_failure"]
        reasons.append("model/kernel config fit failed")

    if consensus_success:
        score += weights["consensus_success"]
        reasons.append("consensus diagnostics report success")
    elif fit_success:
        score -= weights["consensus_failure"]
        reasons.append("fit completed but consensus success is not explicitly true")

    if n_accepted:
        score += weights["accepted_band"] * float(n_accepted)
        reasons.append(f"accepted bands: {n_accepted}")
    if n_rejected:
        score -= weights["rejected_band"] * float(n_rejected)
        reasons.append(f"rejected bands: {n_rejected}")

    if has_exception:
        score -= weights["exception"]
        reasons.append(str(outcome.get("exception_type")))

    # Do not let the upstream advisory order dominate the outcome score, but use
    # a tiny deterministic penalty so equal-scored candidates keep the PR58 order.
    score -= weights["rank_tiebreak"] * float(rank)

    period = _piwd_finite_float_or_none(outcome.get("consensus_period"))
    mode = outcome.get("consensus_time_kernel_constraint_mode")

    return _clean_scalar_dict(
        {
            "rank": outcome.get("rank"),
            "model": outcome.get("model"),
            "status": status,
            "fit_success": fit_success,
            "consensus_success": consensus_success,
            "consensus_period": period,
            "consensus_time_kernel_constraint_mode": mode,
            "n_accepted_bands": n_accepted,
            "n_rejected_bands": n_rejected,
            "exception_type": outcome.get("exception_type"),
            "exception_message": outcome.get("exception_message"),
            "score": float(score),
            "viability_score": float(score),
            "score_kind": "completion_viability",
            "score_is_fit_quality_metric": False,
            "fit_quality_metrics_used": [],
            "score_components": reasons,
            "source_outcome": outcome,
        }
    )


def score_period_independent_wavelength_model_kernel_config_runs(
    run_report: dict[str, Any],
    *,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score and rank a PR59 wavelength model/kernel-config run report.

    This is a comparison report only.  It intentionally does not apply model
    selection, does not install a winning fit on any Lightcurve, and leaves
    ``selected_model`` as ``None``.  The highest-scoring candidate is exposed as
    ``top_ranked_model`` for user inspection, not as an automatic decision.
    """
    if not isinstance(run_report, dict) or run_report.get("kind") != "period_independent_wavelength_model_kernel_config_results":
        raise ValueError(
            "score_period_independent_wavelength_model_kernel_config_runs() requires a "
            "period-independent wavelength model/kernel-config results report."
        )

    default_weights = {
        "fit_success": 100.0,
        "fit_failure": 100.0,
        "consensus_success": 25.0,
        "consensus_failure": 25.0,
        "accepted_band": 2.0,
        "rejected_band": 1.0,
        "exception": 10.0,
        "rank_tiebreak": 0.001,
    }
    if weights:
        for key, value in weights.items():
            if key not in default_weights:
                raise ValueError(f"Unknown scoring weight: {key!r}")
            default_weights[key] = float(value)

    outcomes = run_report.get("model_kernel_config_results")
    if outcomes is None:
        outcomes = run_report.get("outcomes")
    if not isinstance(outcomes, list):
        raise ValueError("run report must contain an 'outcomes' or 'model_kernel_config_results' list")

    scored = [
        _piwd_score_one_wavelength_fit_outcome(outcome, weights=default_weights)
        for outcome in outcomes
        if isinstance(outcome, dict)
    ]
    scored.sort(key=lambda item: (-float(item.get("score", float("-inf"))), int(item.get("rank") or 999999)))

    ranked_results: list[dict[str, Any]] = []
    for index, item in enumerate(scored, start=1):
        copied = dict(item)
        copied["score_rank"] = index
        copied["is_top_ranked"] = index == 1
        ranked_results.append(copied)

    top = ranked_results[0] if ranked_results else None

    return _clean_scalar_dict(
        {
            "kind": "period_independent_wavelength_model_kernel_config_scores",
            "stage": "model_kernel_config_scoring_summary",
            "source_run_report_kind": run_report.get("kind"),
            "is_period_independent": True,
            "uses_temporal_consensus": True,
            "uses_period_or_frequency": True,
            "runs_fits": False,
            "scores_completed_fits": True,
            "score_kind": "completion_viability",
            "scores_fit_quality": False,
            "fit_quality_metrics_used": [],
            "score_interpretation": (
                "Completion/diagnostic viability score only; this is not a "
                "likelihood, residual, predictive, cross-validation, AIC, or BIC "
                "fit-quality metric."
            ),
            "applies_to_fit": False,
            "advisory_only": True,
            "automatic_model_selection_applied": False,
            "selected_model": None,
            "top_ranked_model": top.get("model") if top else None,
            "top_ranked_score": top.get("score") if top else None,
            "top_ranked_viability_score": top.get("viability_score") if top else None,
            "hard_model_exclusions": False,
            "automatic_constraints_applied": False,
            "automatic_initialization_applied": False,
            "parameter_suggestions_applied": False,
            "n_model_kernel_configs": len(outcomes),
            "n_scored": len(ranked_results),
            "n_passed": sum(1 for item in ranked_results if item.get("fit_success")),
            "n_failed": sum(1 for item in ranked_results if not item.get("fit_success")),
            "scoring_weights": dict(default_weights),
            "ranked_results": ranked_results,
            "scored_model_kernel_configs": ranked_results,
            "notes": [
                "Scores are completion/diagnostic viability summaries only, not scientific fit-quality scores.",
                "No likelihood, residual, predictive, cross-validation, AIC, or BIC metric is used by this scorer.",
                "No model is selected or installed automatically.",
                "Ties are broken by the upstream advisory candidate order with a tiny rank penalty.",
            ],
        }
    )


# -----------------------------------------------------------------------------
# Period-independent wavelength fit-quality scoring (Level 0f)
# -----------------------------------------------------------------------------

def _piwd_quality_metric(value: Any, *, fallback: float = 1.0e6) -> float:
    """Return a finite non-negative quality metric, lower is better."""
    try:
        out = float(value)
    except Exception:
        return float(fallback)
    if not np.isfinite(out):
        return float(fallback)
    return max(out, 0.0)


def _piwd_score_one_wavelength_fit_quality(scored_or_outcome: dict[str, Any]) -> dict[str, Any]:
    """Score one completed candidate using actual training-residual metrics."""
    outcome = scored_or_outcome.get("source_outcome")
    if not isinstance(outcome, dict):
        outcome = scored_or_outcome

    fit_quality = outcome.get("fit_quality")
    if not isinstance(fit_quality, dict):
        fit_quality = {}

    fit_success = _piwd_bool_from_status(
        outcome.get("fit_success"), status=outcome.get("status")
    )
    available = bool(fit_quality.get("available")) and fit_success

    nrmse = _piwd_quality_metric(
        fit_quality.get("normalized_rmse_by_target_scale"), fallback=1.0e6
    )
    standardized = _piwd_quality_metric(
        fit_quality.get("median_abs_standardized_residual"), fallback=nrmse
    )
    outlier_frac = _piwd_quality_metric(
        fit_quality.get("outlier_fraction_3sigma"), fallback=0.0 if available else 1.0
    )
    red_chi2 = _piwd_quality_metric(
        fit_quality.get("reduced_chi2"), fallback=standardized**2 if available else 1.0e6
    )

    if available:
        # Higher is better.  The penalties are deliberately simple and
        # transparent; this is a training-residual quality score, not a formal
        # evidence, AIC, BIC, or cross-validation metric.
        fit_quality_score = 100.0 - 25.0 * nrmse - 5.0 * standardized - 25.0 * outlier_frac
        fit_quality_score -= 0.25 * np.log1p(red_chi2)
        reason = "training residual diagnostics available"
    else:
        # Unavailable diagnostics are not a very poor scientific score.  Keep
        # them unscored so they cannot become a top-ranked candidate when every
        # fit failed or every diagnostic was unavailable.
        fit_quality_score = None
        reason = fit_quality.get("reason") or "training residual diagnostics unavailable"

    return _clean_scalar_dict(
        {
            "rank": outcome.get("rank"),
            "model": outcome.get("model"),
            "status": outcome.get("status"),
            "fit_success": fit_success,
            "consensus_success": outcome.get("consensus_success"),
            "consensus_period": outcome.get("consensus_period"),
            "consensus_time_kernel_constraint_mode": outcome.get("consensus_time_kernel_constraint_mode"),
            "fit_quality_available": available,
            "fit_quality_score": (
                float(fit_quality_score) if fit_quality_score is not None else None
            ),
            "score": float(fit_quality_score) if fit_quality_score is not None else None,
            "score_kind": "training_residual_fit_quality",
            "scores_fit_quality": True,
            "fit_quality_metrics_used": [
                "training_nrmse_by_target_scale",
                "training_median_abs_standardized_residual",
                "training_outlier_fraction_3sigma",
                "training_reduced_chi2",
            ],
            "training_rmse": fit_quality.get("rmse"),
            "training_mae": fit_quality.get("mae"),
            "training_nrmse_by_target_scale": fit_quality.get("normalized_rmse_by_target_scale"),
            "training_normalized_rmse": fit_quality.get("normalized_rmse"),
            "training_median_abs_standardized_residual": fit_quality.get("median_abs_standardized_residual"),
            "training_outlier_fraction_3sigma": fit_quality.get("outlier_fraction_3sigma"),
            "training_reduced_chi2": fit_quality.get("reduced_chi2"),
            "training_log_marginal_likelihood": fit_quality.get("log_marginal_likelihood"),
            "training_predictive_variance_kind": fit_quality.get(
                "predictive_variance_kind"
            ),
            "training_standardization_sigma_source": fit_quality.get(
                "standardization_sigma_source"
            ),
            "training_measurement_uncertainty_added_separately": fit_quality.get(
                "measurement_uncertainty_added_separately"
            ),
            "score_components": [reason],
            "source_outcome": outcome,
        }
    )


def score_period_independent_wavelength_model_kernel_config_quality(
    run_report: dict[str, Any],
) -> dict[str, Any]:
    """Score completed wavelength model/kernel config fits using training residual diagnostics.

    This is still advisory-only and non-selecting.  It uses actual fit-quality
    summaries recorded by the PR61 runner, but it does not install the top model
    on any Lightcurve and it does not claim to be cross-validation or evidence.
    """
    if not isinstance(run_report, dict) or run_report.get("kind") != "period_independent_wavelength_model_kernel_config_results":
        raise ValueError(
            "score_period_independent_wavelength_model_kernel_config_quality() requires a "
            "period-independent wavelength model/kernel-config results report."
        )
    outcomes = run_report.get("model_kernel_config_results")
    if outcomes is None:
        outcomes = run_report.get("outcomes")
    if not isinstance(outcomes, list):
        raise ValueError("run report must contain an 'outcomes' or 'model_kernel_config_results' list")

    scored = [
        _piwd_score_one_wavelength_fit_quality(outcome)
        for outcome in outcomes
        if isinstance(outcome, dict)
    ]
    scored.sort(
        key=lambda item: (
            0 if item.get("fit_quality_available") else 1,
            -(
                float(item.get("fit_quality_score"))
                if item.get("fit_quality_available")
                and item.get("fit_quality_score") is not None
                else 0.0
            ),
            int(item.get("rank") or 999999),
        )
    )

    n_with_fit_quality = sum(
        1 for item in scored if item.get("fit_quality_available") is True
    )
    if n_with_fit_quality >= 2:
        ranking_status = "available"
    elif n_with_fit_quality == 1:
        ranking_status = "single_valid_candidate"
    else:
        ranking_status = "unavailable"

    ranked_results = []
    quality_rank = 0
    for item in scored:
        copied = dict(item)
        if copied.get("fit_quality_available") is True:
            quality_rank += 1
            copied["quality_rank"] = quality_rank
            copied["is_top_ranked"] = (
                ranking_status == "available" and quality_rank == 1
            )
        else:
            copied["quality_rank"] = None
            copied["is_top_ranked"] = False
        ranked_results.append(copied)

    valid_rows = [
        item for item in ranked_results if item.get("fit_quality_available") is True
    ]
    top = valid_rows[0] if ranking_status == "available" else None
    only_valid = valid_rows[0] if ranking_status == "single_valid_candidate" else None

    return _clean_scalar_dict(
        {
            "kind": "period_independent_wavelength_model_kernel_config_quality_scores",
            "stage": "model_kernel_config_training_residual_quality_summary",
            "source_run_report_kind": run_report.get("kind"),
            "is_period_independent": True,
            "uses_temporal_consensus": True,
            "uses_period_or_frequency": True,
            "runs_fits": False,
            "scores_completed_fits": True,
            "score_kind": "training_residual_fit_quality",
            "scores_fit_quality": True,
            "fit_quality_metrics_used": [
                "training_nrmse_by_target_scale",
                "training_median_abs_standardized_residual",
                "training_outlier_fraction_3sigma",
                "training_reduced_chi2",
            ],
            "score_interpretation": (
                "Training-residual fit-quality score computed at the fitted training "
                "coordinates. This is not cross-validation, AIC, BIC, or model evidence."
            ),
            "applies_to_fit": False,
            "advisory_only": True,
            "automatic_model_selection_applied": False,
            "selected_model": None,
            "ranking_status": ranking_status,
            "fit_quality_ranking_available": ranking_status == "available",
            "single_valid_candidate": ranking_status == "single_valid_candidate",
            "top_ranked_model": top.get("model") if top else None,
            "top_ranked_fit_quality_score": top.get("fit_quality_score") if top else None,
            "only_valid_model": only_valid.get("model") if only_valid else None,
            "only_valid_fit_quality_score": (
                only_valid.get("fit_quality_score") if only_valid else None
            ),
            "hard_model_exclusions": False,
            "automatic_constraints_applied": False,
            "automatic_initialization_applied": False,
            "parameter_suggestions_applied": False,
            "n_model_kernel_configs": len(outcomes),
            "n_scored": n_with_fit_quality,
            "n_unscored": len(ranked_results) - n_with_fit_quality,
            "n_with_fit_quality": n_with_fit_quality,
            "ranked_results": ranked_results,
            "quality_ranked_results": ranked_results,
            "notes": [
                "Fit-quality scores use training residual diagnostics from completed model/kernel config fits.",
                "Candidates without valid fit-quality diagnostics remain unscored and cannot be top ranked.",
                "A single valid candidate is reported separately and is not treated as a comparative ranking.",
                "No model is selected or installed automatically.",
                "Use cross-validation or held-out diagnostics before treating this as scientific model selection.",
            ],
        }
    )


def _piwd_fit_quality_row_available(row: dict[str, Any]) -> bool:
    """Return whether a comparison row has usable fit-quality diagnostics."""
    if not isinstance(row, dict):
        return False
    if "fit_quality_available" in row:
        return row.get("fit_quality_available") is True

    fit_success = _piwd_bool_from_status(
        row.get("fit_success"), status=row.get("status")
    )
    try:
        score = float(row.get("fit_quality_score"))
    except (TypeError, ValueError):
        return False
    return fit_success and np.isfinite(score)


def _piwd_valid_fit_quality_rows(
    quality_report: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return comparison rows with usable fit-quality diagnostics."""
    quality_report = quality_report if isinstance(quality_report, dict) else {}
    rows = quality_report.get("ranked_results")
    if rows is None:
        rows = quality_report.get("quality_ranked_results")
    if not isinstance(rows, list):
        return []
    return [
        row
        for row in rows
        if isinstance(row, dict) and _piwd_fit_quality_row_available(row)
    ]


def _piwd_resolve_fit_quality_ranking_state(
    quality_report: dict[str, Any] | None,
) -> tuple[str, int]:
    """Return normalized ``(ranking_status, n_with_fit_quality)`` values.

    When comparison rows are present, their actual diagnostic availability is
    authoritative.  This prevents stale summary fields from preserving a
    comparative winner after every row has become failed or unscored.
    """
    quality_report = quality_report if isinstance(quality_report, dict) else {}
    rows = quality_report.get("ranked_results")
    if rows is None:
        rows = quality_report.get("quality_ranked_results")

    if isinstance(rows, list):
        n_with_fit_quality = len(_piwd_valid_fit_quality_rows(quality_report))
    else:
        try:
            n_with_fit_quality = max(
                int(quality_report.get("n_with_fit_quality") or 0), 0
            )
        except (TypeError, ValueError):
            n_with_fit_quality = 0

    if n_with_fit_quality >= 2:
        ranking_status = "available"
    elif n_with_fit_quality == 1:
        ranking_status = "single_valid_candidate"
    else:
        ranking_status = "unavailable"

    return ranking_status, n_with_fit_quality


# -----------------------------------------------------------------------------
# Period-independent wavelength candidate comparison presentation (Level 0h)
# -----------------------------------------------------------------------------

def _piwd_candidate_comparison_results(score_report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return ranked candidate comparison rows from a PR60/PR61 report."""
    if not isinstance(score_report, dict):
        raise ValueError("candidate comparison report must be a dict")
    kind = score_report.get("kind")
    allowed = {
        "period_independent_wavelength_model_kernel_config_scores",
        "period_independent_wavelength_model_kernel_config_quality_scores",
    }
    if kind not in allowed:
        raise ValueError(
            "candidate comparison helpers require a wavelength model/kernel-config "
            "score or quality-score report."
        )

    rows = score_report.get("ranked_results")
    if rows is None:
        rows = score_report.get("quality_ranked_results")
    if rows is None:
        rows = score_report.get("viability_ranked_results")
    if not isinstance(rows, list):
        raise ValueError("score report must contain ranked_results")
    return [row for row in rows if isinstance(row, dict)]


def _piwd_format_float(value: Any, precision: int = 4) -> str:
    """Format a scalar for compact human-readable reports."""
    try:
        val = float(value)
    except Exception:
        return "-" if value is None else str(value)
    if not np.isfinite(val):
        return "-"
    if abs(val) >= 1.0e4 or (abs(val) < 1.0e-3 and val != 0.0):
        return f"{val:.{precision}e}"
    return f"{val:.{precision}g}"


def format_period_independent_wavelength_model_kernel_config_comparison_report(
    score_report: dict[str, Any],
    *,
    max_rows: int | None = None,
) -> str:
    """Format a PR60/PR61 wavelength candidate comparison as plain text.

    The formatter is deliberately non-selecting: it reports the top-ranked
    candidate in the supplied score report, but it also repeats the advisory
    contract fields so the formatted report cannot be mistaken for an automatic
    model-selection step.
    """
    rows = _piwd_candidate_comparison_results(score_report)
    if max_rows is not None:
        max_rows = int(max_rows)
        if max_rows <= 0:
            raise ValueError("max_rows must be positive when supplied")
        rows = rows[:max_rows]

    lines: list[str] = []
    lines.append("Period-independent wavelength model/kernel-config comparison")
    lines.append("=" * 63)
    lines.append(f"kind: {score_report.get('kind')}")
    lines.append(f"score_kind: {score_report.get('score_kind')}")
    lines.append(f"scores_fit_quality: {score_report.get('scores_fit_quality')}")
    lines.append(f"runs_fits: {score_report.get('runs_fits')}")
    lines.append(f"applies_to_fit: {score_report.get('applies_to_fit')}")
    lines.append(f"advisory_only: {score_report.get('advisory_only')}")
    lines.append(
        "automatic_model_selection_applied: "
        f"{score_report.get('automatic_model_selection_applied')}"
    )
    lines.append(f"selected_model: {score_report.get('selected_model')}")
    if score_report.get("score_kind") == "training_residual_fit_quality":
        lines.append(f"ranking_status: {score_report.get('ranking_status')}")
        lines.append(
            "fit_quality_ranking_available: "
            f"{score_report.get('fit_quality_ranking_available')}"
        )
        lines.append(f"only_valid_model: {score_report.get('only_valid_model')}")
    lines.append(f"top_ranked_model: {score_report.get('top_ranked_model')}")
    if score_report.get("score_interpretation"):
        lines.append(f"score_interpretation: {score_report.get('score_interpretation')}")
    lines.append("")

    header = (
        "rank | model | score | fit_success | consensus | period | "
        "kernel_mode | nrmse | med_abs_std | red_chi2"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        if score_report.get("score_kind") == "training_residual_fit_quality":
            rank = row.get("quality_rank")
            if rank is None:
                rank = "-"
        else:
            rank = row.get("score_rank") or row.get("rank")
        score = row.get("fit_quality_score")
        if score is None:
            score = row.get("viability_score", row.get("score"))
        lines.append(
            " | ".join(
                [
                    str(rank),
                    str(row.get("model")),
                    _piwd_format_float(score),
                    str(row.get("fit_success")),
                    str(row.get("consensus_success")),
                    _piwd_format_float(row.get("consensus_period")),
                    str(row.get("consensus_time_kernel_constraint_mode")),
                    _piwd_format_float(row.get("training_nrmse_by_target_scale")),
                    _piwd_format_float(row.get("training_median_abs_standardized_residual")),
                    _piwd_format_float(row.get("training_reduced_chi2")),
                ]
            )
        )

    lines.append("")
    lines.append(
        "Note: this report summarizes an advisory comparison. It does not choose "
        "or install a winning model."
    )
    return "\n".join(lines)


def _piwd_plot_metric_bar(
    rows: list[dict[str, Any]],
    metric_key: str,
    *,
    title: str,
    ylabel: str,
):
    """Return a simple one-metric bar figure, or None if no values exist."""
    values: list[float] = []
    labels: list[str] = []
    for row in rows:
        value = row.get(metric_key)
        try:
            val = float(value)
        except Exception:
            continue
        if not np.isfinite(val):
            continue
        labels.append(str(row.get("model")))
        values.append(val)
    if not values:
        return None

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("candidate model")
    fig.tight_layout()
    return fig


def plot_period_independent_wavelength_model_kernel_config_comparison(
    score_report: dict[str, Any],
    *,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Create simple diagnostic plots for a PR60/PR61 candidate comparison.

    Each metric is plotted in its own figure.  The helper does not run fits,
    rescore candidates, or select a model.
    """
    rows = _piwd_candidate_comparison_results(score_report)
    if max_rows is not None:
        max_rows = int(max_rows)
        if max_rows <= 0:
            raise ValueError("max_rows must be positive when supplied")
        rows = rows[:max_rows]

    figures: dict[str, Any] = {}
    score_key = "fit_quality_score" if any("fit_quality_score" in row for row in rows) else "score"
    score_title = (
        "Training-residual fit-quality score"
        if score_key == "fit_quality_score"
        else "Completion/diagnostic viability score"
    )
    fig = _piwd_plot_metric_bar(
        rows,
        score_key,
        title=score_title,
        ylabel=score_key,
    )
    if fig is not None:
        figures["score"] = fig

    for key, title, ylabel in [
        (
            "training_nrmse_by_target_scale",
            "Training normalized RMSE by target scale",
            "normalized RMSE",
        ),
        (
            "training_median_abs_standardized_residual",
            "Training median absolute standardized residual",
            "median |standardized residual|",
        ),
        (
            "training_reduced_chi2",
            "Training reduced chi-square",
            "reduced chi-square",
        ),
        (
            "training_outlier_fraction_3sigma",
            "Training 3-sigma outlier fraction",
            "outlier fraction",
        ),
    ]:
        fig = _piwd_plot_metric_bar(rows, key, title=title, ylabel=ylabel)
        if fig is not None:
            figures[key] = fig

    return figures


# -----------------------------------------------------------------------------
# Period-independent wavelength advisory workflow (Level 0i)
# -----------------------------------------------------------------------------




def _piwd_count_values(values: list[Any]) -> dict[str, int]:
    """Count non-empty scalar values for JSON/report summaries."""
    counts: dict[str, int] = {}
    for value in values:
        if value is None or value == "":
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _piwd_build_advisory_workflow_fallback_summary(
    run_report: dict[str, Any] | None,
    quality_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize diagnostic fallback state when fit-quality ranking is unavailable.

    The fallback is available when no comparative training-residual ranking can
    be formed: all attempted fits failed, all completed fits lack usable
    diagnostics, or only one candidate has valid diagnostics.  It remains
    advisory-only and never converts a failed or non-comparative workflow into
    an automatically selected model.
    """
    run_report = run_report if isinstance(run_report, dict) else {}
    quality_report = quality_report if isinstance(quality_report, dict) else {}
    outcomes = (
        run_report.get("model_kernel_config_results")
        or run_report.get("outcomes")
        or []
    )
    outcomes = [item for item in outcomes if isinstance(item, dict)]
    passed = [
        item
        for item in outcomes
        if item.get("fit_success") is True or item.get("status") == "passed"
    ]
    failed = [
        item
        for item in outcomes
        if item.get("fit_failed") is True or item.get("status") == "failed"
    ]

    failure_stage_counts = _piwd_count_values(
        [item.get("failure_stage") for item in failed]
    )
    exception_type_counts = _piwd_count_values(
        [item.get("exception_type") for item in failed]
    )
    consensus_failure_models = [
        item.get("model") for item in failed if item.get("is_consensus_failure") is True
    ]
    numerical_failure_models = [
        item.get("model") for item in failed if item.get("is_numerical_failure") is True
    ]

    ranking_status, n_with_fit_quality = (
        _piwd_resolve_fit_quality_ranking_state(quality_report)
    )
    valid_quality_rows = _piwd_valid_fit_quality_rows(quality_report)
    only_valid_model = quality_report.get("only_valid_model")
    if only_valid_model is None and len(valid_quality_rows) == 1:
        only_valid_model = valid_quality_rows[0].get("model")

    fit_based_model_ranking_available = ranking_status == "available"
    all_attempted_failed = bool(outcomes) and not passed and bool(failed)

    if all_attempted_failed:
        reason = "all_model_kernel_config_fits_failed"
        recommended_next_steps = [
            "Inspect exception_type, exception_message, and failure_stage for each model/kernel config.",
            "Use the period-independent wavelength diagnostics and parameter-plan metadata as the fallback interpretation; no model is selected automatically.",
            "If failures are consensus dominated, inspect LS/ACF/consensus period diagnostics and consider rerunning with safer consensus settings or a smaller model/kernel-config set.",
            "If failures are numerical, inspect constraints, learned noise, time-centering, and SM ARD scale-ceiling diagnostics before trusting fit-quality comparisons.",
        ]
    elif ranking_status == "single_valid_candidate":
        reason = "only_one_model_kernel_config_has_fit_quality"
        recommended_next_steps = [
            "Inspect the only valid candidate as a completed fit, not as a comparative winner.",
            "Use period-independent diagnostics as the fallback interpretation until at least one additional candidate has valid fit-quality diagnostics.",
        ]
    elif bool(outcomes) and ranking_status == "unavailable":
        reason = "no_model_kernel_config_fit_quality_available"
        recommended_next_steps = [
            "Inspect why completed fits lack training-residual diagnostics before comparing candidates.",
            "Use the period-independent wavelength diagnostics and parameter-plan metadata as the fallback interpretation; no model is selected automatically.",
        ]
    elif fit_based_model_ranking_available:
        reason = "fit_quality_ranking_available"
        recommended_next_steps = []
    else:
        reason = "no_model_kernel_config_fits_were_attempted"
        recommended_next_steps = []

    fallback_available = bool(outcomes) and not fit_based_model_ranking_available
    top_ranked_model = (
        quality_report.get("top_ranked_model")
        if fit_based_model_ranking_available
        else None
    )

    return _clean_scalar_dict(
        {
            "kind": "period_independent_wavelength_advisory_fallback_summary",
            "available": fallback_available,
            "reason": reason,
            "diagnostic_only": True,
            "advisory_only": True,
            "automatic_model_selection_applied": False,
            "selected_model": None,
            "fit_quality_ranking_status": ranking_status,
            "fit_based_model_ranking_available": fit_based_model_ranking_available,
            "single_valid_candidate": ranking_status == "single_valid_candidate",
            "top_ranked_model": top_ranked_model,
            "only_valid_model": only_valid_model,
            "n_attempted": len(outcomes),
            "n_passed": len(passed),
            "n_failed": len(failed),
            "n_with_fit_quality": n_with_fit_quality,
            "failed_models": [item.get("model") for item in failed],
            "failure_stage_counts": failure_stage_counts,
            "exception_type_counts": exception_type_counts,
            "consensus_failure_models": consensus_failure_models,
            "n_consensus_failure_models": len(consensus_failure_models),
            "numerical_failure_models": numerical_failure_models,
            "n_numerical_failure_models": len(numerical_failure_models),
            "recommended_next_steps": recommended_next_steps,
        }
    )


def _piwd_format_advisory_workflow_text_report(
    workflow_report: dict[str, Any],
    comparison_text_report: str | None,
) -> str:
    """Format a workflow-level advisory report.

    The nested quality-score comparison report has ``runs_fits=False`` because
    scoring/formatting does not itself execute fits.  The workflow report may
    have ``runs_fits=True`` when it built and ran model/kernel config fits before scoring.
    Keep both scopes explicit so users do not mistake nested score-report
    metadata for the workflow execution contract.
    """
    quality_report = workflow_report.get("quality_report") or {}
    run_report = workflow_report.get("run_report") or {}

    fallback_report = workflow_report.get("fallback_report") or {}

    lines = [
        "Period-independent wavelength advisory workflow",
        "=" * 60,
        f"kind: {workflow_report.get('kind')}",
        f"advisory_only: {workflow_report.get('advisory_only')}",
        f"workflow_runs_model_kernel_config_fits: {workflow_report.get('runs_fits')}",
        f"model_kernel_config_runner_ran_fits: {run_report.get('runs_fits')}",
        f"quality_score_report_runs_fits: {quality_report.get('runs_fits')}",
        f"model_kernel_config_state_isolated: {workflow_report.get('model_kernel_config_state_isolated')}",
        f"mutates_input_lightcurve: {workflow_report.get('mutates_input_lightcurve')}",
        f"automatic_model_selection_applied: {workflow_report.get('automatic_model_selection_applied')}",
        f"selected_model: {workflow_report.get('selected_model')}",
        f"automatic_constraints_applied: {workflow_report.get('automatic_constraints_applied')}",
        f"automatic_initialization_applied: {workflow_report.get('automatic_initialization_applied')}",
        f"score_kind: {workflow_report.get('score_kind')}",
        f"fit_quality_ranking_status: {workflow_report.get('fit_quality_ranking_status')}",
        f"fit_quality_ranking_available: {workflow_report.get('fit_quality_ranking_available')}",
        f"only_valid_model: {workflow_report.get('only_valid_model')}",
        f"top_ranked_model: {workflow_report.get('top_ranked_model')}",
        f"top_ranked_fit_quality_score: {workflow_report.get('top_ranked_fit_quality_score')}",
        f"fallback_diagnostics_available: {workflow_report.get('fallback_diagnostics_available')}",
        f"fallback_reason: {fallback_report.get('reason')}",
        "",
        "Scope note: workflow_runs_model_kernel_config_fits describes this one-shot wrapper. "
        "quality_score_report_runs_fits describes the nested quality-score report; "
        "it is expected to be False because scoring summarizes already-completed fits.",
    ]

    if fallback_report.get("available"):
        lines.extend(
            [
                "",
                "Fallback diagnostics",
                "=" * 60,
                f"failure_stage_counts: {fallback_report.get('failure_stage_counts')}",
                f"exception_type_counts: {fallback_report.get('exception_type_counts')}",
                f"consensus_failure_models: {fallback_report.get('consensus_failure_models')}",
                f"numerical_failure_models: {fallback_report.get('numerical_failure_models')}",
                "recommended_next_steps:",
            ]
        )
        for step in fallback_report.get("recommended_next_steps") or []:
            lines.append(f"- {step}")

    if comparison_text_report:
        lines.extend(
            [
                "",
                "Nested model/kernel config comparison report",
                "=" * 60,
                comparison_text_report,
            ]
        )

    return "\n".join(lines)

def run_period_independent_wavelength_advisory_workflow(
    lightcurve: Any,
    *,
    model_kernel_config_report: dict[str, Any] | None = None,
    run_report: dict[str, Any] | None = None,
    quality_report: dict[str, Any] | None = None,
    include_2d_baseline: bool = True,
    base_fit_kwargs: dict[str, Any] | None = None,
    include_models: list[str] | tuple[str, ...] | None = None,
    model_kernel_config_limit: int | None = None,
    stop_on_error: bool = False,
    make_text_report: bool = True,
    make_plots: bool = False,
) -> dict[str, Any]:
    """Run the advisory wavelength-candidate workflow end to end.

    The workflow is a convenience wrapper around the PR58--PR62 helpers:

    1. build advisory model/kernel-config configs,
    2. run those model/kernel config fits in isolated Lightcurve copies,
    3. score completed candidates using training-residual diagnostics,
    4. optionally format and/or plot the comparison report.

    It is deliberately non-selecting.  It reports a top-ranked candidate only
    when at least two candidates have valid fit-quality diagnostics.  A single
    valid candidate is reported separately rather than treated as a comparative
    winner.  The workflow never installs a candidate, sets ``selected_model``,
    mutates the input Lightcurve fit state, or applies wavelength-parameter
    suggestions as constraints or initial values.
    """
    built_model_kernel_config_report = model_kernel_config_report is None
    ran_model_kernel_config_fits = run_report is None
    scored_quality = quality_report is None

    if model_kernel_config_report is None:
        model_kernel_config_report = build_period_independent_wavelength_model_kernel_configs(
            lightcurve,
            include_2d_baseline=include_2d_baseline,
            base_fit_kwargs=base_fit_kwargs,
            include_models=include_models,
            model_kernel_config_limit=model_kernel_config_limit,
        )

    if run_report is None:
        run_report = run_period_independent_wavelength_model_kernel_configs(
            lightcurve,
            model_kernel_config_report=model_kernel_config_report,
            model_kernel_config_limit=model_kernel_config_limit,
            stop_on_error=stop_on_error,
        )

    if quality_report is None:
        quality_report = score_period_independent_wavelength_model_kernel_config_quality(
            run_report
        )

    ranking_status, n_with_fit_quality = (
        _piwd_resolve_fit_quality_ranking_state(quality_report)
    )
    valid_quality_rows = _piwd_valid_fit_quality_rows(quality_report)
    only_valid_model = (
        quality_report.get("only_valid_model")
        if isinstance(quality_report, dict)
        else None
    )
    only_valid_fit_quality_score = (
        quality_report.get("only_valid_fit_quality_score")
        if isinstance(quality_report, dict)
        else None
    )
    if ranking_status == "single_valid_candidate" and len(valid_quality_rows) == 1:
        only_valid_model = only_valid_model or valid_quality_rows[0].get("model")
        if only_valid_fit_quality_score is None:
            only_valid_fit_quality_score = valid_quality_rows[0].get(
                "fit_quality_score"
            )

    fallback_report = _piwd_build_advisory_workflow_fallback_summary(
        run_report,
        quality_report,
    )

    comparison_text_report = None
    if make_text_report:
        comparison_text_report = format_period_independent_wavelength_model_kernel_config_comparison_report(
            quality_report
        )

    figures = None
    if make_plots:
        figures = plot_period_independent_wavelength_model_kernel_config_comparison(
            quality_report
        )

    workflow_report: dict[str, Any] = {
        "kind": "period_independent_wavelength_advisory_workflow",
        "stage": "model_kernel_config_build_run_quality_report",
        "advisory_only": True,
        "builds_model_kernel_configs": True,
        "runs_fits": bool(ran_model_kernel_config_fits),
        "scores_fit_quality": True,
        "formats_report": bool(make_text_report),
        "makes_plots": bool(make_plots),
        "model_kernel_config_state_isolated": bool(
            run_report.get("model_kernel_config_state_isolated", True)
        ) if isinstance(run_report, dict) else True,
        "mutates_input_lightcurve": False,
        "automatic_model_selection_applied": False,
        "selected_model": None,
        "automatic_constraints_applied": False,
        "automatic_initialization_applied": False,
        "parameter_suggestions_applied": False,
        "built_model_kernel_config_report": bool(built_model_kernel_config_report),
        "ran_model_kernel_config_fits": bool(ran_model_kernel_config_fits),
        "scored_quality": bool(scored_quality),
        "fit_quality_ranking_status": ranking_status,
        "fit_quality_ranking_available": ranking_status == "available",
        "single_valid_candidate": ranking_status == "single_valid_candidate",
        "n_with_fit_quality": n_with_fit_quality,
        "only_valid_model": only_valid_model,
        "only_valid_fit_quality_score": only_valid_fit_quality_score,
        "top_ranked_model": (
            quality_report.get("top_ranked_model")
            if isinstance(quality_report, dict) and ranking_status == "available"
            else None
        ),
        "top_ranked_fit_quality_score": (
            quality_report.get("top_ranked_fit_quality_score")
            if isinstance(quality_report, dict) and ranking_status == "available"
            else None
        ),
        "score_kind": (
            quality_report.get("score_kind")
            if isinstance(quality_report, dict)
            else None
        ),
        "fallback_diagnostics_available": fallback_report.get("available"),
        "fallback_report": fallback_report,
        "model_kernel_config_report": model_kernel_config_report,
        "run_report": run_report,
        "quality_report": quality_report,
        "comparison_text_report": comparison_text_report,
        "text_report": None,
    }
    if make_text_report:
        workflow_report["text_report"] = _piwd_format_advisory_workflow_text_report(
            workflow_report,
            comparison_text_report,
        )
    if figures is not None:
        workflow_report["figures"] = figures
    return workflow_report


def _piwd_export_json_safe(value):
    """Return a JSON-serializable copy of a wavelength-workflow payload.

    Matplotlib figures and other non-serializable objects are represented by
    strings rather than being embedded in the JSON export.  This helper is
    intentionally local to the export path so it does not change the schema of
    the in-memory advisory workflow report.
    """
    import math

    try:
        import numpy as _np
    except Exception:  # pragma: no cover - numpy is expected but not required here
        _np = None

    try:
        import torch as _torch
    except Exception:  # pragma: no cover - torch may not be importable in minimal envs
        _torch = None

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if _np is not None and isinstance(value, _np.generic):
        return _piwd_export_json_safe(value.item())
    if _np is not None and isinstance(value, _np.ndarray):
        return _piwd_export_json_safe(value.tolist())
    if _torch is not None and isinstance(value, _torch.Tensor):
        return _piwd_export_json_safe(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): _piwd_export_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_piwd_export_json_safe(v) for v in value]
    if hasattr(value, "savefig"):
        return f"<{type(value).__name__}>"
    return str(value)


def _piwd_export_sanitize_filename_component(name):
    """Make a short filesystem-safe component for exported plot names."""
    text = str(name).strip() or "figure"
    safe = []
    for char in text:
        if char.isalnum() or char in ("-", "_", "."):
            safe.append(char)
        else:
            safe.append("_")
    cleaned = "".join(safe).strip("._")
    return cleaned or "figure"


def export_period_independent_wavelength_advisory_workflow(
    workflow,
    output_dir,
    *,
    prefix="wavelength_advisory_workflow",
    save_json=True,
    save_text=True,
    save_figures=True,
    figure_format="png",
    figure_dpi=150,
    close_figures=False,
):
    """Export a period-independent wavelength advisory workflow report.

    This helper is an output/export layer only.  It writes the already-produced
    workflow dictionary, optional text reports, and optional matplotlib figures
    to ``output_dir``.  It does not run fits, score candidates, select a model,
    apply constraints, or apply initialization.

    Parameters
    ----------
    workflow : dict
        Output of ``run_period_independent_wavelength_advisory_workflow``.
    output_dir : str or pathlib.Path
        Directory where files should be written.
    prefix : str, optional
        Prefix used for all exported files.
    save_json, save_text, save_figures : bool, optional
        Control which output products are written.
    figure_format : str, optional
        Figure extension/format passed to matplotlib ``savefig``.
    figure_dpi : int or float, optional
        Resolution for saved figures.
    close_figures : bool, optional
        If true, close figures after saving them.

    Returns
    -------
    dict
        JSON-safe export manifest with paths to written files.
    """
    import json
    from pathlib import Path

    if not isinstance(workflow, dict):
        raise ValueError("workflow must be a period-independent wavelength advisory workflow dict")
    if workflow.get("kind") != "period_independent_wavelength_advisory_workflow":
        raise ValueError(
            "workflow must be a period-independent wavelength advisory workflow dict "
            "with kind='period_independent_wavelength_advisory_workflow'."
        )

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    safe_prefix = _piwd_export_sanitize_filename_component(prefix)
    fig_format = str(figure_format).lstrip(".") or "png"

    manifest = {
        "kind": "period_independent_wavelength_advisory_workflow_export",
        "source_workflow_kind": workflow.get("kind"),
        "output_dir": str(outdir),
        "prefix": safe_prefix,
        "exports_workflow_outputs": True,
        "runs_fits": False,
        "applies_to_fit": False,
        "advisory_only": True,
        "mutates_input_lightcurve": False,
        "automatic_model_selection_applied": False,
        "selected_model": None,
        "automatic_constraints_applied": False,
        "automatic_initialization_applied": False,
        "wrote_json": False,
        "wrote_text_report": False,
        "wrote_comparison_text_report": False,
        "wrote_figures": False,
        "json_path": None,
        "text_report_path": None,
        "comparison_text_report_path": None,
        "figure_paths": {},
        "exported_files": [],
    }

    if save_json:
        json_payload = {k: v for k, v in workflow.items() if k != "figures"}
        json_path = outdir / f"{safe_prefix}.json"
        json_path.write_text(
            json.dumps(_piwd_export_json_safe(json_payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        manifest["wrote_json"] = True
        manifest["json_path"] = str(json_path)
        manifest["exported_files"].append(str(json_path))

    if save_text:
        text_report = workflow.get("text_report")
        if text_report:
            text_path = outdir / f"{safe_prefix}.txt"
            text_path.write_text(str(text_report), encoding="utf-8")
            manifest["wrote_text_report"] = True
            manifest["text_report_path"] = str(text_path)
            manifest["exported_files"].append(str(text_path))

        comparison_text = workflow.get("comparison_text_report")
        if comparison_text and comparison_text != text_report:
            comparison_path = outdir / f"{safe_prefix}_comparison.txt"
            comparison_path.write_text(str(comparison_text), encoding="utf-8")
            manifest["wrote_comparison_text_report"] = True
            manifest["comparison_text_report_path"] = str(comparison_path)
            manifest["exported_files"].append(str(comparison_path))

    if save_figures:
        figures = workflow.get("figures") or {}
        if figures:
            for fig_name, fig in figures.items():
                if not hasattr(fig, "savefig"):
                    continue
                safe_name = _piwd_export_sanitize_filename_component(fig_name)
                fig_path = outdir / f"{safe_prefix}_{safe_name}.{fig_format}"
                fig.savefig(fig_path, dpi=figure_dpi)
                manifest["figure_paths"][str(fig_name)] = str(fig_path)
                manifest["exported_files"].append(str(fig_path))
                if close_figures:
                    try:
                        import matplotlib.pyplot as _plt

                        _plt.close(fig)
                    except Exception:  # pragma: no cover - best-effort cleanup only
                        pass
            manifest["wrote_figures"] = bool(manifest["figure_paths"])

    return manifest


def _piwd_batch_safe_source_id(value, index):
    """Return a stable source identifier for batch workflow rows and paths."""
    if value is None or str(value).strip() == "":
        return f"source_{index + 1:04d}"
    return str(value)


def _piwd_batch_safe_path_component(value, index):
    """Return a filesystem-safe path component for a batch source."""
    raw = _piwd_batch_safe_source_id(value, index)
    safe = []
    for char in raw:
        if char.isalnum() or char in ("-", "_", "."):
            safe.append(char)
        else:
            safe.append("_")
    cleaned = "".join(safe).strip("._")
    return cleaned or f"source_{index + 1:04d}"



def _piwd_batch_apply_positive_data_filter(
    lightcurve,
    *,
    require_positive_flux=False,
    require_positive_flux_error=False,
):
    """Return ``(lightcurve, report)`` after optional positive-row filtering.

    The batch runner uses this immediately after CSV ingestion so survey runs
    can reject rows with non-positive fluxes or flux uncertainties before any
    advisory fits are attempted.  The original light curve is returned
    unchanged when no rows need to be dropped.
    """
    require_positive_flux = bool(require_positive_flux)
    require_positive_flux_error = bool(require_positive_flux_error)

    ydata = getattr(lightcurve, "_ydata_raw", None)
    n_before = None
    if ydata is not None:
        try:
            n_before = int(ydata.reshape(-1).shape[0])
        except Exception:
            n_before = None

    report = {
        "applied": bool(require_positive_flux or require_positive_flux_error),
        "require_positive_flux": require_positive_flux,
        "require_positive_flux_error": require_positive_flux_error,
        "n_rows_before": n_before,
        "n_rows_after": n_before,
        "n_rows_dropped": 0,
        "n_dropped_nonpositive_flux": 0,
        "n_dropped_nonpositive_flux_error": 0,
    }

    if not report["applied"]:
        return lightcurve, report

    if torch is None:  # pragma: no cover - pgmuvi normally depends on torch
        raise RuntimeError("positive light-curve filtering requires torch.")

    xdata = getattr(lightcurve, "_xdata_raw", None)
    if xdata is None or ydata is None:
        raise ValueError(
            "positive light-curve filtering requires a Lightcurve-like object "
            "with _xdata_raw and _ydata_raw arrays."
        )
    if getattr(ydata, "dim", lambda: None)() != 1:
        raise ValueError(
            "positive light-curve filtering currently requires one-dimensional ydata."
        )
    if xdata.shape[0] != ydata.shape[0]:
        raise ValueError(
            "positive light-curve filtering requires matching xdata/ydata row counts."
        )

    keep = torch.ones(ydata.shape[0], dtype=torch.bool, device=ydata.device)

    if require_positive_flux:
        good_flux = torch.isfinite(ydata) & (ydata > 0.0)
        report["n_dropped_nonpositive_flux"] = int((~good_flux).sum().item())
        keep &= good_flux

    yerr = getattr(lightcurve, "_yerr_raw", None)
    if require_positive_flux_error:
        if yerr is None:
            raise ValueError(
                "positive flux-error filtering was requested, but this light curve "
                "has no yerr values."
            )
        if yerr.shape[0] != ydata.shape[0]:
            raise ValueError(
                "positive flux-error filtering requires matching ydata/yerr row counts."
            )
        good_yerr = torch.isfinite(yerr) & (yerr > 0.0)
        report["n_dropped_nonpositive_flux_error"] = int((~good_yerr).sum().item())
        keep &= good_yerr

    n_after = int(keep.sum().item())
    report["n_rows_after"] = n_after
    report["n_rows_dropped"] = int(ydata.shape[0] - n_after)

    if n_after <= 0:
        raise ValueError(
            "No rows remain after strictly positive flux/flux-error filtering."
        )
    if n_after == ydata.shape[0]:
        return lightcurve, report

    mask_np = keep.detach().cpu().numpy()
    band = getattr(lightcurve, "band", None)
    new_band = None
    if band is not None:
        band_arr = np.asarray(band, dtype=np.str_)
        if band_arr.ndim == 1 and len(band_arr) == len(mask_np):
            new_band = band_arr[mask_np]
        else:
            new_band = band_arr

    from .lightcurve import Lightcurve

    filtered = Lightcurve(
        xdata[keep].clone(),
        ydata[keep].clone(),
        yerr=yerr[keep].clone() if yerr is not None else None,
        xtransform=copy.deepcopy(getattr(lightcurve, "xtransform", None)),
        ytransform=copy.deepcopy(getattr(lightcurve, "ytransform", None)),
        name=getattr(lightcurve, "name", None),
        band=new_band,
    )
    return filtered, report

def _piwd_batch_resolve_source(
    source, index, *, from_csv_kwargs=None, positive_data_filter_kwargs=None
):
    """Resolve a batch source specification to ``(source_id, lightcurve, metadata)``.

    Supported source forms are intentionally small and explicit:

    * ``{"source_id": ..., "lightcurve": lc}`` for preconstructed lightcurves;
    * ``{"source_id": ..., "csv_path": ...}`` for CSV-backed sources;
    * ``"path/to/source.csv"`` or ``Path(...)`` for CSV-backed sources.

    The returned lightcurve object only needs to expose the advisory workflow
    methods used by the batch runner, which keeps the helper easy to test with
    lightweight fakes while supporting real ``Lightcurve`` instances.
    """
    from pathlib import Path

    common_csv_kwargs = dict(from_csv_kwargs or {})
    positive_filter_kwargs = dict(positive_data_filter_kwargs or {})

    if isinstance(source, dict):
        source_id = _piwd_batch_safe_source_id(
            source.get("source_id")
            or source.get("id")
            or source.get("name")
            or source.get("csv_path"),
            index,
        )
        metadata = {
            k: v
            for k, v in source.items()
            if k not in {
                "lightcurve",
                "csv_path",
                "from_csv_kwargs",
                "positive_data_filter_kwargs",
            }
        }
        source_positive_filter_kwargs = dict(positive_filter_kwargs)
        source_positive_filter_kwargs.update(source.get("positive_data_filter_kwargs") or {})
        if "lightcurve" in source:
            lc, filter_report = _piwd_batch_apply_positive_data_filter(
                source["lightcurve"], **source_positive_filter_kwargs
            )
            metadata["positive_data_filter"] = filter_report
            return source_id, lc, metadata
        if "csv_path" in source:
            csv_kwargs = dict(common_csv_kwargs)
            csv_kwargs.update(source.get("from_csv_kwargs") or {})
            from .lightcurve import Lightcurve

            lc = Lightcurve.from_csv(source["csv_path"], **csv_kwargs)
            lc, filter_report = _piwd_batch_apply_positive_data_filter(
                lc, **source_positive_filter_kwargs
            )
            metadata["positive_data_filter"] = filter_report
            return source_id, lc, metadata
        raise ValueError(
            "Each batch source dict must contain either 'lightcurve' or 'csv_path'."
        )

    if isinstance(source, (str, Path)):
        path = Path(source)
        source_id = _piwd_batch_safe_source_id(path.stem, index)
        from .lightcurve import Lightcurve

        lc = Lightcurve.from_csv(path, **common_csv_kwargs)
        lc, filter_report = _piwd_batch_apply_positive_data_filter(
            lc, **positive_filter_kwargs
        )
        return source_id, lc, {
            "csv_path": str(path),
            "positive_data_filter": filter_report,
        }

    raise ValueError(
        "Batch sources must be dictionaries, CSV paths, or path-like strings."
    )


def _piwd_batch_write_summary_csv(path, rows):
    """Write a compact batch summary CSV."""
    import csv

    fields = [
        "source_index",
        "source_id",
        "status",
        "fit_quality_ranking_status",
        "fit_quality_ranking_available",
        "only_valid_model",
        "top_ranked_model",
        "top_ranked_fit_quality_score",
        "score_kind",
        "n_model_kernel_configs",
        "n_successful_model_kernel_configs",
        "n_failed_model_kernel_configs",
        "n_rows_before_positive_filter",
        "n_rows_after_positive_filter",
        "n_rows_dropped_positive_filter",
        "exception_type",
        "exception_message",
        "source_output_dir",
        "source_output_prefix",
        "export_json_path",
        "export_text_report_path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})



def _piwd_batch_nonempty(value):
    """Return True for values that can safely identify or enrich a row."""
    return value is not None and value != ""


def _piwd_batch_index_model_kernel_config_row(row, *, by_id, by_model):
    """Index a merged execution row by stable identifiers.

    A model name is only used as a merge key when it is unique among the
    execution rows.  That keeps the current one-row-per-model workflow simple
    while avoiding an unsafe merge if a future workflow evaluates the same
    model with multiple kernel settings.
    """
    if not isinstance(row, dict):
        return

    config_id = row.get("model_kernel_config_id")
    if _piwd_batch_nonempty(config_id):
        by_id[config_id] = row

    model = row.get("model")
    if _piwd_batch_nonempty(model):
        if model in by_model and by_model[model] is not row:
            by_model[model] = None
        else:
            by_model[model] = row


def _piwd_batch_merge_nonempty_fields(target, source):
    """Merge source fields into target without erasing execution metadata."""
    if not isinstance(target, dict) or not isinstance(source, dict):
        return target

    for key, value in source.items():
        if not _piwd_batch_nonempty(value):
            if key not in target:
                target[key] = value
            continue

        # ``rank`` in execution rows is the model/kernel-config rank.  Ranking
        # rows may also carry rank-like information, but quality ordering must
        # be represented by ``quality_rank`` and must not overwrite the run
        # configuration rank.
        if key == "rank" and _piwd_batch_nonempty(target.get("rank")):
            continue

        # Preserve execution fit kwargs unless the execution row omitted them.
        if key == "fit_kwargs" and isinstance(target.get("fit_kwargs"), dict):
            if isinstance(value, dict):
                merged = dict(target["fit_kwargs"])
                for subkey, subvalue in value.items():
                    if _piwd_batch_nonempty(subvalue) or subkey not in merged:
                        merged[subkey] = subvalue
                target["fit_kwargs"] = merged
            continue

        target[key] = value

    return target


def _piwd_batch_merge_model_kernel_config_rows(run_rows, ranked_rows):
    """Merge execution and quality rows without duplicating configs.

    ``run_rows`` come from the candidate/model-kernel-config runner and contain
    execution metadata such as model_kernel_config_id, fit kwargs, status, and
    consensus diagnostics.  ``ranked_rows`` come from the quality scorer and may
    contain only the model name plus quality_rank/fit_quality_score.  A ranked
    row must enrich the corresponding execution row rather than becoming a
    second long-form CSV row.
    """
    merged = []
    by_id = {}
    by_model = {}

    for row in run_rows or []:
        if not isinstance(row, dict):
            continue
        copied = dict(row)
        merged.append(copied)
        _piwd_batch_index_model_kernel_config_row(copied, by_id=by_id, by_model=by_model)

    for row in ranked_rows or []:
        if not isinstance(row, dict):
            continue

        target = None
        config_id = row.get("model_kernel_config_id")
        if _piwd_batch_nonempty(config_id):
            target = by_id.get(config_id)

        if target is None:
            model = row.get("model")
            if _piwd_batch_nonempty(model):
                target = by_model.get(model)

        if target is None:
            copied = dict(row)
            merged.append(copied)
            _piwd_batch_index_model_kernel_config_row(copied, by_id=by_id, by_model=by_model)
        else:
            _piwd_batch_merge_nonempty_fields(target, row)
            _piwd_batch_index_model_kernel_config_row(target, by_id=by_id, by_model=by_model)

    return merged


def _piwd_batch_extract_model_kernel_config_rows(*, source_row, workflow):
    """Return long-form rows for a source's evaluated model/kernel configs.

    Each output row is suitable for a batch-level source-by-model CSV.  It is
    intentionally flattened so downstream inspection does not require parsing
    the full nested workflow JSON for every source.
    """
    workflow = workflow if isinstance(workflow, dict) else {}
    source_row = source_row if isinstance(source_row, dict) else {}
    run_report = workflow.get("run_report") or {}
    quality_report = workflow.get("quality_report") or {}

    run_rows = (
        run_report.get("model_kernel_config_results")
        or run_report.get("outcomes")
        or workflow.get("model_kernel_config_results")
        or workflow.get("outcomes")
        or []
    )
    ranked_rows = quality_report.get("ranked_results") or workflow.get("ranked_results") or []
    rows = _piwd_batch_merge_model_kernel_config_rows(run_rows, ranked_rows)

    ranking_status, _ = _piwd_resolve_fit_quality_ranking_state(
        quality_report if quality_report else workflow
    )

    out = []
    for item in rows:
        fit_kwargs = item.get("fit_kwargs") if isinstance(item.get("fit_kwargs"), dict) else {}
        flattened = {
            "source_index": source_row.get("source_index"),
            "source_id": source_row.get("source_id"),
            "source_status": source_row.get("status"),
            "workflow_kind": workflow.get("kind"),
            "score_kind": quality_report.get("score_kind") or workflow.get("score_kind"),
            "fit_quality_ranking_status": ranking_status,
            "fit_quality_ranking_available": ranking_status == "available",
            "model_kernel_config_id": item.get("model_kernel_config_id"),
            "model_kernel_config_rank": item.get("rank"),
            "quality_rank": item.get("quality_rank"),
            "is_top_ranked": item.get("is_top_ranked"),
            "model": item.get("model"),
            "status": item.get("status"),
            "fit_success": item.get("fit_success"),
            "fit_failed": item.get("fit_failed"),
            "failure_stage": item.get("failure_stage"),
            "failure_stage_reason": item.get("failure_stage_reason"),
            "is_consensus_failure": item.get("is_consensus_failure"),
            "is_numerical_failure": item.get("is_numerical_failure"),
            "is_input_validation_failure": item.get("is_input_validation_failure"),
            "fit_quality_score": item.get("fit_quality_score"),
            "fit_quality_available": item.get("fit_quality_available"),
            "fit_strategy": fit_kwargs.get("fit_strategy"),
            "time_kernel_type": fit_kwargs.get("time_kernel_type"),
            "wavelength_kernel_type": fit_kwargs.get("wavelength_kernel_type"),
            "learn_additional_noise": fit_kwargs.get("learn_additional_noise"),
            "training_iter": fit_kwargs.get("training_iter"),
            "miniter": fit_kwargs.get("miniter"),
            "consensus_success": item.get("consensus_success"),
            "consensus_period": item.get("consensus_period"),
            "consensus_frequency": item.get("consensus_frequency"),
            "consensus_time_kernel_constraint_mode": item.get(
                "consensus_time_kernel_constraint_mode"
            ),
            "n_accepted_bands": item.get("n_accepted_bands"),
            "n_rejected_bands": item.get("n_rejected_bands"),
            "training_rmse": item.get("training_rmse"),
            "training_mae": item.get("training_mae"),
            "training_nrmse_by_target_scale": item.get(
                "training_nrmse_by_target_scale"
            ),
            "training_median_abs_standardized_residual": item.get(
                "training_median_abs_standardized_residual"
            ),
            "training_outlier_fraction_3sigma": item.get(
                "training_outlier_fraction_3sigma"
            ),
            "training_reduced_chi2": item.get("training_reduced_chi2"),
            "training_log_marginal_likelihood": item.get(
                "training_log_marginal_likelihood"
            ),
            "training_predictive_variance_kind": item.get(
                "training_predictive_variance_kind"
            ),
            "training_standardization_sigma_source": item.get(
                "training_standardization_sigma_source"
            ),
            "training_measurement_uncertainty_added_separately": item.get(
                "training_measurement_uncertainty_added_separately"
            ),
            "n_constrained_sm_ard_components": item.get(
                "n_constrained_sm_ard_components"
            ),
            "n_constrained_sm_time_components": item.get(
                "n_constrained_sm_time_components"
            ),
            "n_constrained_sm_wavelength_components": item.get(
                "n_constrained_sm_wavelength_components"
            ),
            "constrained_sm_ard_dimension_counts": item.get(
                "constrained_sm_ard_dimension_counts"
            ),
            "constrained_sm_ard_components": item.get("constrained_sm_ard_components"),
            "exception_type": item.get("exception_type"),
            "exception_message": item.get("exception_message"),
        }
        out.append(_clean_scalar_dict(flattened))
    return out


def _piwd_batch_model_kernel_config_csv_fields():
    """Return stable field order for the long-form model/kernel-config CSV."""
    return [
        "source_index",
        "source_id",
        "source_status",
        "workflow_kind",
        "score_kind",
        "fit_quality_ranking_status",
        "fit_quality_ranking_available",
        "model_kernel_config_id",
        "model_kernel_config_rank",
        "quality_rank",
        "is_top_ranked",
        "model",
        "status",
        "fit_success",
        "fit_failed",
        "failure_stage",
        "failure_stage_reason",
        "is_consensus_failure",
        "is_numerical_failure",
        "is_input_validation_failure",
        "fit_quality_score",
        "fit_quality_available",
        "fit_strategy",
        "time_kernel_type",
        "wavelength_kernel_type",
        "learn_additional_noise",
        "training_iter",
        "miniter",
        "consensus_success",
        "consensus_period",
        "consensus_frequency",
        "consensus_time_kernel_constraint_mode",
        "n_accepted_bands",
        "n_rejected_bands",
        "training_rmse",
        "training_mae",
        "training_nrmse_by_target_scale",
        "training_median_abs_standardized_residual",
        "training_outlier_fraction_3sigma",
        "training_reduced_chi2",
        "training_log_marginal_likelihood",
        "training_predictive_variance_kind",
        "training_standardization_sigma_source",
        "training_measurement_uncertainty_added_separately",
        "n_constrained_sm_ard_components",
        "n_constrained_sm_time_components",
        "n_constrained_sm_wavelength_components",
        "constrained_sm_ard_dimension_counts",
        "constrained_sm_ard_components",
        "exception_type",
        "exception_message",
    ]


def _piwd_batch_write_model_kernel_config_csv(path, rows):
    """Write a long-form source-by-model/kernel-config batch CSV."""
    import csv

    fields = _piwd_batch_model_kernel_config_csv_fields()
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})





def _piwd_batch_to_float(value):
    """Best-effort conversion of a scalar batch CSV/report value to float."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _piwd_batch_to_bool(value):
    """Best-effort conversion of a scalar batch CSV/report value to bool."""
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _piwd_batch_median(values):
    """Return the median of a non-empty numeric sequence."""
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _piwd_batch_mean(values):
    """Return the arithmetic mean of a non-empty numeric sequence."""
    if not values:
        return None
    return sum(values) / float(len(values))


def _piwd_batch_model_kernel_config_group_key(row):
    """Return the batch-aggregate key for one evaluated model/kernel config."""
    if not isinstance(row, dict):
        return (None, None, None, None, None)
    return (
        row.get("model"),
        row.get("fit_strategy"),
        row.get("time_kernel_type"),
        row.get("wavelength_kernel_type"),
        row.get("learn_additional_noise"),
    )


def _piwd_batch_summarize_model_kernel_config_rows(rows):
    """Summarize the long-form source/model-kernel-config table by config.

    The input rows are the PR69 one-row-per-source/model-kernel-config rows.
    The output is one row per distinct model/kernel setup, suitable for asking
    questions such as which model family most often ranked first across a batch.
    """
    groups = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        key = _piwd_batch_model_kernel_config_group_key(row)
        if key not in groups:
            model, fit_strategy, time_kernel_type, wavelength_kernel_type, learn_additional_noise = key
            groups[key] = {
                "model": model,
                "fit_strategy": fit_strategy,
                "time_kernel_type": time_kernel_type,
                "wavelength_kernel_type": wavelength_kernel_type,
                "learn_additional_noise": learn_additional_noise,
                "n_sources_evaluated": 0,
                "n_successful_sources": 0,
                "n_failed_sources": 0,
                "n_sources_with_fit_quality": 0,
                "n_sources_with_comparative_ranking": 0,
                "n_top_ranked_sources": 0,
                "_fit_quality_score": [],
                "_training_nrmse_by_target_scale": [],
                "_training_median_abs_standardized_residual": [],
                "_training_outlier_fraction_3sigma": [],
                "_training_reduced_chi2": [],
            }
        group = groups[key]
        group["n_sources_evaluated"] += 1

        if _piwd_batch_to_bool(row.get("fit_success")) is True:
            group["n_successful_sources"] += 1
        elif _piwd_batch_to_bool(row.get("fit_failed")) is True or row.get("status") == "failed":
            group["n_failed_sources"] += 1

        fit_quality_available = _piwd_fit_quality_row_available(row)
        ranking_available = (
            _piwd_batch_to_bool(row.get("fit_quality_ranking_available")) is True
            or row.get("fit_quality_ranking_status") == "available"
        )
        if fit_quality_available:
            group["n_sources_with_fit_quality"] += 1
        if ranking_available:
            group["n_sources_with_comparative_ranking"] += 1

        is_top = _piwd_batch_to_bool(row.get("is_top_ranked"))
        if ranking_available and fit_quality_available and is_top is True:
            group["n_top_ranked_sources"] += 1

        if fit_quality_available:
            for metric in [
                "fit_quality_score",
                "training_nrmse_by_target_scale",
                "training_median_abs_standardized_residual",
                "training_outlier_fraction_3sigma",
                "training_reduced_chi2",
            ]:
                value = _piwd_batch_to_float(row.get(metric))
                if value is not None:
                    group[f"_{metric}"].append(value)

    out = []
    for group in groups.values():
        n_eval = group["n_sources_evaluated"]
        n_success = group["n_successful_sources"]
        n_comparative = group["n_sources_with_comparative_ranking"]
        n_top = group["n_top_ranked_sources"]
        scores = group.pop("_fit_quality_score")
        nrmse = group.pop("_training_nrmse_by_target_scale")
        med_abs_std = group.pop("_training_median_abs_standardized_residual")
        outlier = group.pop("_training_outlier_fraction_3sigma")
        red_chi2 = group.pop("_training_reduced_chi2")
        group.update(
            {
                "success_fraction": (n_success / n_eval) if n_eval else None,
                "top_ranked_fraction": (
                    n_top / n_comparative if n_comparative else None
                ),
                "mean_fit_quality_score": _piwd_batch_mean(scores),
                "median_fit_quality_score": _piwd_batch_median(scores),
                "best_fit_quality_score": max(scores) if scores else None,
                "worst_fit_quality_score": min(scores) if scores else None,
                "mean_training_nrmse_by_target_scale": _piwd_batch_mean(nrmse),
                "median_training_nrmse_by_target_scale": _piwd_batch_median(nrmse),
                "mean_training_median_abs_standardized_residual": _piwd_batch_mean(med_abs_std),
                "median_training_median_abs_standardized_residual": _piwd_batch_median(med_abs_std),
                "mean_training_outlier_fraction_3sigma": _piwd_batch_mean(outlier),
                "median_training_outlier_fraction_3sigma": _piwd_batch_median(outlier),
                "mean_training_reduced_chi2": _piwd_batch_mean(red_chi2),
                "median_training_reduced_chi2": _piwd_batch_median(red_chi2),
            }
        )
        out.append(_clean_scalar_dict(group))

    def _sort_key(row):
        median_score = _piwd_batch_to_float(row.get("median_fit_quality_score"))
        if median_score is None:
            median_score = float("-inf")
        return (
            -int(row.get("n_top_ranked_sources") or 0),
            -median_score,
            str(row.get("model") or ""),
            str(row.get("time_kernel_type") or ""),
            str(row.get("wavelength_kernel_type") or ""),
        )

    out.sort(key=_sort_key)
    return out


def _piwd_batch_model_kernel_config_summary_csv_fields():
    """Return stable field order for the aggregate model/kernel-config CSV."""
    return [
        "model",
        "fit_strategy",
        "time_kernel_type",
        "wavelength_kernel_type",
        "learn_additional_noise",
        "n_sources_evaluated",
        "n_successful_sources",
        "n_failed_sources",
        "n_sources_with_fit_quality",
        "n_sources_with_comparative_ranking",
        "n_top_ranked_sources",
        "success_fraction",
        "top_ranked_fraction",
        "mean_fit_quality_score",
        "median_fit_quality_score",
        "best_fit_quality_score",
        "worst_fit_quality_score",
        "mean_training_nrmse_by_target_scale",
        "median_training_nrmse_by_target_scale",
        "mean_training_median_abs_standardized_residual",
        "median_training_median_abs_standardized_residual",
        "mean_training_outlier_fraction_3sigma",
        "median_training_outlier_fraction_3sigma",
        "mean_training_reduced_chi2",
        "median_training_reduced_chi2",
    ]


def _piwd_batch_write_model_kernel_config_summary_csv(path, rows):
    """Write the aggregate one-row-per-model/kernel-config summary CSV."""
    import csv

    fields = _piwd_batch_model_kernel_config_summary_csv_fields()
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})



def _piwd_batch_markdown_value(value):
    """Format a scalar value for the batch Markdown report."""
    if value is None or value == "":
        return "—"
    return str(value)


def _piwd_batch_markdown_cell(value):
    """Format a scalar value for a Markdown table cell."""
    text = _piwd_batch_markdown_value(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _piwd_batch_format_markdown_report(manifest):
    """Return a human-readable Markdown report for a batch advisory run."""
    if not isinstance(manifest, dict):
        raise TypeError("manifest must be a dictionary")

    lines = [
        "# Period-independent wavelength advisory batch report",
        "",
        "## Batch contract",
        f"- kind: {_piwd_batch_markdown_value(manifest.get('kind'))}",
        f"- advisory_only: {_piwd_batch_markdown_value(manifest.get('advisory_only'))}",
        f"- runs_fits: {_piwd_batch_markdown_value(manifest.get('runs_fits'))}",
        f"- applies_to_fit: {_piwd_batch_markdown_value(manifest.get('applies_to_fit'))}",
        f"- model_kernel_config_state_isolated: {_piwd_batch_markdown_value(manifest.get('model_kernel_config_state_isolated'))}",
        f"- mutates_input_lightcurve: {_piwd_batch_markdown_value(manifest.get('mutates_input_lightcurve'))}",
        f"- automatic_model_selection_applied: {_piwd_batch_markdown_value(manifest.get('automatic_model_selection_applied'))}",
        f"- selected_model: {_piwd_batch_markdown_value(manifest.get('selected_model'))}",
        "",
        "## Source totals",
        f"- n_sources: {_piwd_batch_markdown_value(manifest.get('n_sources'))}",
        f"- n_succeeded: {_piwd_batch_markdown_value(manifest.get('n_succeeded'))}",
        f"- n_failed: {_piwd_batch_markdown_value(manifest.get('n_failed'))}",
        "",
        "## Output files",
        f"- batch_json_path: {_piwd_batch_markdown_value(manifest.get('batch_json_path'))}",
        f"- batch_csv_path: {_piwd_batch_markdown_value(manifest.get('batch_csv_path'))}",
        f"- batch_model_kernel_config_csv_path: {_piwd_batch_markdown_value(manifest.get('batch_model_kernel_config_csv_path'))}",
        f"- batch_model_kernel_config_summary_csv_path: {_piwd_batch_markdown_value(manifest.get('batch_model_kernel_config_summary_csv_path'))}",
        f"- batch_markdown_report_path: {_piwd_batch_markdown_value(manifest.get('batch_markdown_report_path'))}",
        "",
        "## Model/kernel config aggregate summary",
    ]

    summary_rows = manifest.get("model_kernel_config_summary") or []
    if summary_rows:
        fields = [
            "model",
            "fit_strategy",
            "time_kernel_type",
            "n_sources_evaluated",
            "n_successful_sources",
            "n_failed_sources",
            "n_sources_with_fit_quality",
            "n_sources_with_comparative_ranking",
            "n_top_ranked_sources",
            "success_fraction",
            "top_ranked_fraction",
            "median_fit_quality_score",
            "median_training_nrmse_by_target_scale",
        ]
        lines.append("| " + " | ".join(fields) + " |")
        lines.append("|" + "|".join(["---"] * len(fields)) + "|")
        for row in summary_rows:
            lines.append(
                "| "
                + " | ".join(_piwd_batch_markdown_cell(row.get(field)) for field in fields)
                + " |"
            )
    else:
        lines.append("No model/kernel config aggregate rows were produced.")

    lines.extend(["", "## Source summary"])
    source_rows = manifest.get("source_results") or []
    if source_rows:
        fields = [
            "source_id",
            "status",
            "fit_quality_ranking_status",
            "only_valid_model",
            "top_ranked_model",
            "top_ranked_fit_quality_score",
            "score_kind",
            "n_model_kernel_configs",
            "n_successful_model_kernel_configs",
            "n_failed_model_kernel_configs",
            "n_rows_before_positive_filter",
            "n_rows_after_positive_filter",
            "n_rows_dropped_positive_filter",
            "exception_type",
            "exception_message",
            "source_output_dir",
            "source_output_prefix",
        ]
        lines.append("| " + " | ".join(fields) + " |")
        lines.append("|" + "|".join(["---"] * len(fields)) + "|")
        for row in source_rows:
            lines.append(
                "| "
                + " | ".join(_piwd_batch_markdown_cell(row.get(field)) for field in fields)
                + " |"
            )
    else:
        lines.append("No source rows were produced.")

    return "\n".join(lines) + "\n"



def _piwd_batch_failure_text_report(row):
    """Return a compact human-readable report for one failed batch source."""
    if not isinstance(row, dict):
        row = {}
    lines = [
        "# Period-independent wavelength advisory source failure",
        "",
        f"source_index: {_piwd_batch_markdown_value(row.get('source_index'))}",
        f"source_id: {_piwd_batch_markdown_value(row.get('source_id'))}",
        f"status: {_piwd_batch_markdown_value(row.get('status'))}",
        f"exception_type: {_piwd_batch_markdown_value(row.get('exception_type'))}",
        f"exception_message: {_piwd_batch_markdown_value(row.get('exception_message'))}",
        "",
    ]
    tb = row.get("traceback")
    if tb:
        lines.extend(["## Traceback", "", "```text", str(tb).rstrip(), "```", ""])
    return "\n".join(lines)


def _piwd_batch_write_failure_artifacts(output_dir, prefix, row):
    """Write per-source JSON/text artifacts for a failed batch source."""
    import json
    from pathlib import Path

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    safe_prefix = _piwd_batch_safe_path_component(prefix, int(row.get("source_index") or 0))
    json_path = outdir / f"{safe_prefix}_failure.json"
    text_path = outdir / f"{safe_prefix}_failure.txt"
    manifest = {
        "kind": "period_independent_wavelength_advisory_workflow_failure_export",
        "source_index": row.get("source_index"),
        "source_id": row.get("source_id"),
        "status": row.get("status"),
        "exception_type": row.get("exception_type"),
        "exception_message": row.get("exception_message"),
        "traceback": row.get("traceback"),
        "json_path": str(json_path),
        "text_report_path": str(text_path),
        "exported_files": [str(json_path), str(text_path)],
    }
    json_path.write_text(
        json.dumps(_piwd_export_json_safe(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    text_path.write_text(_piwd_batch_failure_text_report(row), encoding="utf-8")
    return manifest

def _piwd_batch_write_markdown_report(path, manifest):
    """Write a human-readable batch advisory Markdown report."""
    text = _piwd_batch_format_markdown_report(manifest)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)

def run_period_independent_wavelength_advisory_workflow_batch(
    sources,
    *,
    from_csv_kwargs=None,
    positive_data_filter_kwargs=None,
    workflow_kwargs=None,
    output_dir=None,
    export=True,
    export_kwargs=None,
    batch_prefix="wavelength_advisory_batch",
    stop_on_error=False,
):
    """Run the advisory wavelength workflow over multiple sources.

    This helper orchestrates the PR63/PR64 workflow per source.  It may run
    model/kernel config fits because each per-source advisory workflow may run
    model/kernel config fits, but it still does not apply automatic model selection, install a
    winning model, apply constraints, or apply initialization.

    Parameters
    ----------
    sources : iterable
        Source specifications.  Each entry may be a dict containing a
        preconstructed ``lightcurve`` object, a dict containing ``csv_path``, or
        a path-like CSV filename.
    from_csv_kwargs : dict, optional
        Common keyword arguments passed to ``Lightcurve.from_csv`` for CSV
        sources.  Per-source dicts may override these via ``from_csv_kwargs``.
    positive_data_filter_kwargs : dict, optional
        Optional row filter applied after CSV ingestion and before advisory
        fitting.  Supported keys are ``require_positive_flux`` and
        ``require_positive_flux_error``.
    workflow_kwargs : dict, optional
        Keyword arguments passed to
        ``run_period_independent_wavelength_advisory_workflow`` for each source.
    output_dir : str or pathlib.Path, optional
        If provided and ``export`` is true, per-source workflow outputs and a
        batch summary are written here.
    export : bool, optional
        Whether to export each per-source workflow when ``output_dir`` is
        provided.
    export_kwargs : dict, optional
        Extra keyword arguments passed to the per-source export helper.
    batch_prefix : str, optional
        Prefix for batch-level summary files.
    stop_on_error : bool, optional
        If true, re-raise the first source failure instead of recording it and
        continuing.

    Returns
    -------
    dict
        Batch advisory manifest with one row per source and optional export
        paths.  The manifest remains advisory and non-selecting.
    """
    import json
    import traceback
    from pathlib import Path

    if sources is None:
        raise ValueError("sources must be a non-empty iterable of source specifications.")
    source_list = list(sources)
    if not source_list:
        raise ValueError("sources must be a non-empty iterable of source specifications.")

    workflow_kwargs = dict(workflow_kwargs or {})
    export_kwargs = dict(export_kwargs or {})

    outdir = Path(output_dir) if output_dir is not None else None
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    model_kernel_config_rows = []
    exported_files = []

    for index, source in enumerate(source_list):
        source_id = _piwd_batch_safe_source_id(None, index)
        row = {
            "source_index": index,
            "source_id": source_id,
            "status": "not_started",
            "advisory_only": True,
            "runs_fits": True,
            "applies_to_fit": True,
            "model_kernel_config_state_isolated": True,
            "mutates_input_lightcurve": False,
            "automatic_model_selection_applied": False,
            "selected_model": None,
            "automatic_constraints_applied": False,
            "automatic_initialization_applied": False,
            "fit_quality_ranking_status": None,
            "fit_quality_ranking_available": False,
            "only_valid_model": None,
            "top_ranked_model": None,
            "top_ranked_fit_quality_score": None,
            "score_kind": None,
            "n_model_kernel_configs": 0,
            "n_successful_model_kernel_configs": 0,
            "n_failed_model_kernel_configs": 0,
            "exception_type": None,
            "exception_message": None,
            "traceback": None,
            "positive_data_filter": None,
            "n_rows_before_positive_filter": None,
            "n_rows_after_positive_filter": None,
            "n_rows_dropped_positive_filter": None,
            "source_output_dir": None,
            "source_output_prefix": None,
            "export_manifest": None,
            "export_json_path": None,
            "export_text_report_path": None,
        }

        try:
            source_id, lc, metadata = _piwd_batch_resolve_source(
                source,
                index,
                from_csv_kwargs=from_csv_kwargs,
                positive_data_filter_kwargs=positive_data_filter_kwargs,
            )
            row["source_id"] = source_id
            row["source_metadata"] = metadata
            positive_filter_report = metadata.get("positive_data_filter")
            if isinstance(positive_filter_report, dict):
                row["positive_data_filter"] = positive_filter_report
                row["n_rows_before_positive_filter"] = positive_filter_report.get(
                    "n_rows_before"
                )
                row["n_rows_after_positive_filter"] = positive_filter_report.get(
                    "n_rows_after"
                )
                row["n_rows_dropped_positive_filter"] = positive_filter_report.get(
                    "n_rows_dropped"
                )

            workflow = lc.run_period_independent_wavelength_advisory_workflow(
                **workflow_kwargs
            )
            run_report = workflow.get("run_report") or {}
            quality_report = workflow.get("quality_report") or {}
            candidate_rows = (
                run_report.get("model_kernel_config_results")
                or run_report.get("outcomes")
                or workflow.get("model_kernel_config_results")
                or workflow.get("outcomes")
                or quality_report.get("ranked_results")
                or workflow.get("ranked_results")
                or []
            )
            n_model_kernel_configs = len(candidate_rows)
            n_successful_model_kernel_configs = sum(
                1 for item in candidate_rows if item.get("fit_success") is True
            )
            n_failed_model_kernel_configs = (
                n_model_kernel_configs - n_successful_model_kernel_configs
            )
            workflow_ranking_status, _ = (
                _piwd_resolve_fit_quality_ranking_state(
                    quality_report if quality_report else workflow
                )
            )
            workflow_ranking_available = workflow_ranking_status == "available"
            valid_quality_rows = _piwd_valid_fit_quality_rows(quality_report)
            workflow_only_valid_model = (
                quality_report.get("only_valid_model")
                or workflow.get("only_valid_model")
            )
            if (
                workflow_only_valid_model is None
                and workflow_ranking_status == "single_valid_candidate"
                and len(valid_quality_rows) == 1
            ):
                workflow_only_valid_model = valid_quality_rows[0].get("model")
            workflow_top_ranked_model = (
                workflow.get("top_ranked_model")
                if workflow_ranking_available
                else None
            )
            workflow_top_ranked_score = (
                workflow.get("top_ranked_fit_quality_score")
                if workflow_ranking_available
                else None
            )
            row.update(
                {
                    "status": "passed",
                    "workflow_kind": workflow.get("kind"),
                    "fit_quality_ranking_status": workflow_ranking_status,
                    "fit_quality_ranking_available": workflow_ranking_available,
                    "only_valid_model": workflow_only_valid_model,
                    "top_ranked_model": workflow_top_ranked_model,
                    "top_ranked_fit_quality_score": workflow_top_ranked_score,
                    "score_kind": workflow.get("score_kind"),
                    "n_model_kernel_configs": n_model_kernel_configs,
                    "n_successful_model_kernel_configs": n_successful_model_kernel_configs,
                    "n_failed_model_kernel_configs": n_failed_model_kernel_configs,
                    "workflow_summary": {
                        "kind": workflow.get("kind"),
                        "fit_quality_ranking_status": workflow_ranking_status,
                        "fit_quality_ranking_available": workflow_ranking_available,
                        "only_valid_model": workflow_only_valid_model,
                        "top_ranked_model": workflow_top_ranked_model,
                        "top_ranked_fit_quality_score": workflow_top_ranked_score,
                        "score_kind": workflow.get("score_kind"),
                        "automatic_model_selection_applied": workflow.get(
                            "automatic_model_selection_applied"
                        ),
                        "selected_model": workflow.get("selected_model"),
                    },
                }
            )

            source_model_kernel_config_rows = _piwd_batch_extract_model_kernel_config_rows(
                source_row=row,
                workflow=workflow,
            )
            model_kernel_config_rows.extend(source_model_kernel_config_rows)

            if export and outdir is not None:
                source_component = _piwd_batch_safe_path_component(source_id, index)
                source_outdir = outdir / source_component
                source_prefix = f"{source_component}_wavelength_advisory"
                row["source_output_dir"] = str(source_outdir)
                row["source_output_prefix"] = source_prefix
                manifest = lc.export_period_independent_wavelength_advisory_workflow(
                    workflow=workflow,
                    output_dir=source_outdir,
                    prefix=source_prefix,
                    **export_kwargs,
                )
                row["export_manifest"] = manifest
                row["export_json_path"] = manifest.get("json_path")
                row["export_text_report_path"] = manifest.get("text_report_path")
                exported_files.extend(manifest.get("exported_files") or [])

        except Exception as exc:
            if stop_on_error:
                raise
            row.update(
                {
                    "status": "failed",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            if export and outdir is not None:
                source_component = _piwd_batch_safe_path_component(
                    row.get("source_id"), index
                )
                source_outdir = outdir / source_component
                source_prefix = f"{source_component}_wavelength_advisory"
                row["source_output_dir"] = str(source_outdir)
                row["source_output_prefix"] = source_prefix
                failure_manifest = _piwd_batch_write_failure_artifacts(
                    source_outdir,
                    source_prefix,
                    row,
                )
                row["export_manifest"] = failure_manifest
                row["export_json_path"] = failure_manifest.get("json_path")
                row["export_text_report_path"] = failure_manifest.get("text_report_path")
                exported_files.extend(failure_manifest.get("exported_files") or [])

        rows.append(row)

    model_kernel_config_summary = _piwd_batch_summarize_model_kernel_config_rows(
        model_kernel_config_rows
    )

    manifest = {
        "kind": "period_independent_wavelength_advisory_workflow_batch",
        "advisory_only": True,
        "runs_fits": True,
        "applies_to_fit": True,
        "model_kernel_config_state_isolated": True,
        "mutates_input_lightcurve": False,
        "automatic_model_selection_applied": False,
        "selected_model": None,
        "automatic_constraints_applied": False,
        "automatic_initialization_applied": False,
        "n_sources": len(rows),
        "n_succeeded": sum(1 for row in rows if row.get("status") == "passed"),
        "n_failed": sum(1 for row in rows if row.get("status") == "failed"),
        "source_results": rows,
        "model_kernel_config_results": model_kernel_config_rows,
        "model_kernel_config_summary": model_kernel_config_summary,
        "exported_files": exported_files,
        "batch_json_path": None,
        "batch_csv_path": None,
        "batch_model_kernel_config_csv_path": None,
        "batch_model_kernel_config_summary_csv_path": None,
        "batch_markdown_report_path": None,
    }

    if outdir is not None:
        safe_prefix = _piwd_batch_safe_path_component(batch_prefix, 0)
        json_path = outdir / f"{safe_prefix}_summary.json"
        csv_path = outdir / f"{safe_prefix}_summary.csv"
        model_kernel_config_csv_path = outdir / f"{safe_prefix}_model_kernel_configs.csv"
        model_kernel_config_summary_csv_path = outdir / f"{safe_prefix}_model_kernel_config_summary.csv"
        markdown_report_path = outdir / f"{safe_prefix}_report.md"

        manifest["batch_json_path"] = str(json_path)
        manifest["batch_csv_path"] = str(csv_path)
        manifest["batch_model_kernel_config_csv_path"] = str(model_kernel_config_csv_path)
        manifest["batch_model_kernel_config_summary_csv_path"] = str(
            model_kernel_config_summary_csv_path
        )
        manifest["batch_markdown_report_path"] = str(markdown_report_path)
        manifest["exported_files"].extend(
            [
                str(json_path),
                str(csv_path),
                str(model_kernel_config_csv_path),
                str(model_kernel_config_summary_csv_path),
                str(markdown_report_path),
            ]
        )

        _piwd_batch_write_summary_csv(csv_path, rows)
        _piwd_batch_write_model_kernel_config_csv(
            model_kernel_config_csv_path, model_kernel_config_rows
        )
        _piwd_batch_write_model_kernel_config_summary_csv(
            model_kernel_config_summary_csv_path, model_kernel_config_summary
        )
        _piwd_batch_write_markdown_report(markdown_report_path, manifest)
        json_path.write_text(
            json.dumps(_piwd_export_json_safe(manifest), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    return manifest

