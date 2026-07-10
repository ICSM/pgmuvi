"""Wavelength-dependence diagnostics for multiband light curves.

This module contains cheap, pre-fit diagnostics used to decide what classes
of wavelength-dependent GP models are worth testing.  The initial API is
intentionally conservative: it does not run GP fitting and it does not try to
choose a final model.  It builds a band-by-band table that later model-
selection stages can reuse.
"""

from __future__ import annotations

from typing import Any

import numpy as np

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


def diagnose_wavelength_dependence_prefit(
    lightcurve,
    *,
    sampling_kwargs: dict[str, Any] | None = None,
    variability_kwargs: dict[str, Any] | None = None,
    frequency: float | None = None,
    period: float | None = None,
    amplitude_phase_kwargs: dict[str, Any] | None = None,
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

    return report
