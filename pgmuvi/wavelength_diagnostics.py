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
        "n_bands_with_fixed_frequency_fit": int(len(amplitude_values)),
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
        "n_fixed_frequency_bands": int(len(fixed_rows)),
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
                entry["failure_diagnostics"] = getattr(exception, "failure_diagnostics")
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
    """Summarise scored successful candidate fits."""
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
            "best_candidate": None,
            "selection_status": "not_scored",
            "selection_basis": None,
        }

    scored.sort(key=lambda item: item[0])
    best_value, best = scored[0]
    return {
        "n_scored_successful": int(len(scored)),
        "best_candidate": {
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
    candidate set is informative, whether scores are effectively tied, and
    whether residual structure argues against trusting the provisional best
    candidate.
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
        "provisional_best_candidate": None,
        "candidate_rankings": [],
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
            "No candidate model fit succeeded, so wavelength-model comparison "
            "cannot be interpreted. Inspect failure categories before changing "
            "the candidate set."
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
            "At least one candidate fit succeeded, but predictive scores were "
            "not available. Use fit-status and residual diagnostics only."
        )
        return _clean_scalar_dict(interpretation)

    best_score, best_result = scored[0]
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
            "provisional_best_candidate": rankings[0],
            "candidate_rankings": rankings,
            "selection_basis": "lowest training-point mean negative log predictive density",
        }
    )

    if len(scored) == 1:
        interpretation["decision"] = "single_scored_candidate"
        warnings.append(
            "Only one successful candidate was scored. Treat it as a fitted "
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
            "- Fit candidates: "
            f"{_format_value(comp_summary.get('n_fit_candidates'), precision=0)}; "
            f"successful={_format_value(comp_summary.get('n_successful'), precision=0)}, "
            f"failed={_format_value(comp_summary.get('n_failed'), precision=0)}, "
            f"skipped={_format_value(comp_summary.get('n_skipped'), precision=0)}."
        )
        best = comp_summary.get("best_candidate")
        if isinstance(best, dict):
            lines.append(
                "- Best scored candidate: "
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
    pathway.  The function records which candidate fits succeeded or failed and
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
        Fit keyword arguments applied to every fit candidate, e.g.
        ``training_iter``, ``miniter``, ``lr``, or ``learn_additional_noise``.
    per_candidate_fit_kwargs : dict or None, optional
        Additional kwargs keyed by candidate ``name`` or ``model``.
    residual_diagnostic_kwargs : dict or None, optional
        Keyword arguments passed to :func:`compute_wavelength_residual_diagnostics`
        after each successful fit.
    score_successful_fits : bool, optional
        If True, compute residual and predictive diagnostics immediately after
        each successful candidate fit.
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
    n_fit_candidates = 0
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

        n_fit_candidates += 1
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
            "n_candidates": int(len(normalised)),
            "n_fit_candidates": int(n_fit_candidates),
            "n_successful": int(n_successful),
            "n_failed": int(n_failed),
            "n_skipped": int(n_skipped),
            "all_fit_candidates_succeeded": bool(
                n_fit_candidates > 0 and n_successful == n_fit_candidates
            ),
            "any_fit_candidate_succeeded": bool(n_successful > 0),
            "n_scored_successful": scoring_summary["n_scored_successful"],
            "best_candidate": scoring_summary["best_candidate"],
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
        "q05_flux": None,
        "q10_flux": None,
        "q16_flux": None,
        "q25_flux": None,
        "q50_flux": None,
        "q75_flux": None,
        "q84_flux": None,
        "q90_flux": None,
        "q95_flux": None,
        "mad_scatter": None,
        "iqr_scatter": None,
        "robust_scatter": None,
        "raw_peak_to_peak_q05_q95": None,
        "raw_half_amplitude_q05_q95": None,
        "raw_peak_to_peak_q10_q90": None,
        "raw_half_amplitude_q10_q90": None,
        "fractional_half_amplitude_q05_q95": None,
        "fractional_half_amplitude_q10_q90": None,
        "median_yerr": None,
        "mean_yerr": None,
        "noise_corrected_robust_scatter": None,
        "noise_corrected_half_amplitude_q05_q95": None,
        "noise_corrected_half_amplitude_q10_q90": None,
    }

    if yf.size < min_points:
        return result

    q05, q10, q16, q25, q50, q75, q84, q90, q95 = np.percentile(
        yf, [5.0, 10.0, 16.0, 25.0, 50.0, 75.0, 84.0, 90.0, 95.0]
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
            "q05_flux": float(q05),
            "q10_flux": float(q10),
            "q16_flux": float(q16),
            "q25_flux": float(q25),
            "q50_flux": float(q50),
            "q75_flux": float(q75),
            "q84_flux": float(q84),
            "q90_flux": float(q90),
            "q95_flux": float(q95),
            "mad_scatter": mad,
            "iqr_scatter": iqr_scatter,
            "robust_scatter": robust,
            "raw_peak_to_peak_q05_q95": float(q95 - q05),
            "raw_half_amplitude_q05_q95": amp_05_95,
            "raw_peak_to_peak_q10_q90": float(q90 - q10),
            "raw_half_amplitude_q10_q90": amp_10_90,
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
        "n_bands": int(len(wavelengths)),
        "n_usable_bands": int(len(usable_rows)),
        "usable_wavelengths": usable_wavelengths,
        "has_yerr": bool(yerr_np is not None),
        "uses_temporal_consensus": False,
        "uses_period_or_frequency": False,
        "median_flux_ratio_max_to_min_abs": _piwd_ratio([abs(v) if v is not None else None for v in medians]),
        "raw_half_amplitude_q05_q95_ratio_max_to_min": _piwd_ratio(raw_amp_05_95),
        "raw_half_amplitude_q10_q90_ratio_max_to_min": _piwd_ratio(raw_amp_10_90),
        "robust_scatter_ratio_max_to_min": _piwd_ratio(scatters),
        "noise_corrected_robust_scatter_ratio_max_to_min": _piwd_ratio(nc_scatters),
        "median_flux_monotonicity_class": _piwd_monotonic_class(
            usable_wavelengths, medians
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
        "n_bands": int(len(wavelengths)),
        "n_sampling_pass": int(n_sampling_pass),
        "n_variable": int(n_variable),
        "n_usable_for_wavelength_diagnostics": int(len(usable_wavelengths)),
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
