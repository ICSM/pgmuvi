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


def diagnose_wavelength_dependence_prefit(
    lightcurve,
    *,
    sampling_kwargs: dict[str, Any] | None = None,
    variability_kwargs: dict[str, Any] | None = None,
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

    Returns
    -------
    dict
        JSON-safe report containing ``band_table``, ``summary``, and
        ``warnings``.  The report is diagnostics-only: no GP model is fit.

    Raises
    ------
    ValueError
        If the light curve is not standard 2-D multiband data.
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

    return {
        "kind": "wavelength_dependence_prefit_diagnostics",
        "stage": "prefit",
        "band_table": band_table,
        "summary": _clean_scalar_dict(summary),
        "warnings": warnings,
    }
