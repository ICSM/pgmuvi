"""Multi-method period diagnostics for fitted light curves.

This module combines cached Lomb--Scargle results, cached data-ACF results,
and the fitted GP period-summary machinery. It is deliberately
multi-component-aware: LS candidates, ACF recurrences, summed-PSD peaks, and
kernel components remain distinct and are never paired by array position.
"""

from __future__ import annotations

import copy
import hashlib
import math
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .lomb_scargle_plotting import (
    plot_lomb_scargle_periodogram,
)


def _as_numpy(value: Any) -> np.ndarray:
    """Return a detached NumPy array without mutating the input."""
    if value is None:
        return np.asarray([], dtype=float)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mapping_get(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def period_diagnostic_data_signature(lightcurve: Any) -> str:
    """Fingerprint observations used by cached period diagnostics."""
    digest = hashlib.sha256()
    for attribute in ("xdata", "ydata", "yerr"):
        value = getattr(lightcurve, attribute, None)
        digest.update(attribute.encode("utf-8"))
        if value is None:
            digest.update(b"<none>")
            continue
        array = np.ascontiguousarray(_as_numpy(value))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(array.tobytes())

    labels = getattr(
        lightcurve,
        "observational_channel_labels",
        getattr(lightcurve, "band", None),
    )
    digest.update(b"observational_channel_labels")
    if labels is None:
        digest.update(b"<none>")
    else:
        label_array = np.asarray(labels, dtype=str).reshape(-1)
        digest.update("\x1f".join(label_array.tolist()).encode("utf-8"))
    return digest.hexdigest()


def _normalize_fit_ls_result(
    result: Any,
    *,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize a full ``fit_LS(..., return_full=True)`` result."""
    if isinstance(result, dict):
        peak_frequencies = _mapping_get(
            result,
            "peak_frequencies",
            "frequencies",
            default=[],
        )
        significant = _mapping_get(
            result,
            "significant",
            "peak_significant",
            "flags",
            default=[],
        )
        frequency_grid = _mapping_get(
            result,
            "frequency_grid",
            "freq_grid",
            "frequency",
            default=[],
        )
        power_grid = _mapping_get(
            result,
            "power_grid",
            "power",
            "psd",
            default=[],
        )
    elif isinstance(result, (tuple, list)) and len(result) >= 4:
        peak_frequencies, significant, frequency_grid, power_grid = result[:4]
    elif (
        isinstance(result, (tuple, list))
        and len(result) == 2
        and bool((arguments or {}).get("freq_only", False))
    ):
        frequency_grid, power_grid = result
        peak_frequencies = []
        significant = []
    else:
        return {
            "available": False,
            "reason": (
                "fit_LS did not return a full frequency/power grid; "
                "call fit_LS(..., return_full=True)."
            ),
        }

    peak_frequencies = _as_numpy(peak_frequencies).reshape(-1)
    significant = _as_numpy(significant).reshape(-1)
    frequency_grid = _as_numpy(frequency_grid).reshape(-1)
    power_grid = _as_numpy(power_grid).reshape(-1)

    if frequency_grid.size == 0 or power_grid.size != frequency_grid.size:
        return {
            "available": False,
            "reason": (
                "fit_LS did not return a compatible full frequency/power grid; "
                "call fit_LS(..., return_full=True)."
            ),
        }

    candidates = []
    for rank, frequency_value in enumerate(peak_frequencies, start=1):
        frequency = _finite_float(frequency_value)
        if frequency is None or frequency <= 0.0:
            continue
        candidates.append(
            {
                "rank": rank,
                "frequency": frequency,
                "period": 1.0 / frequency,
                "significant": (
                    bool(significant[rank - 1])
                    if rank - 1 < significant.size
                    else None
                ),
            }
        )

    return {
        "available": True,
        "frequency_grid": frequency_grid,
        "power_grid": power_grid,
        "candidates": candidates,
    }


def _normalize_lomb_scargle_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {"available": False, "reason": "missing Lomb--Scargle result"}

    frequency_grid = _as_numpy(
        _mapping_get(
            value,
            "frequency_grid",
            "freq_grid",
            "frequency",
            default=[],
        )
    ).reshape(-1)
    power_grid = _as_numpy(
        _mapping_get(value, "power_grid", "power", "psd", default=[])
    ).reshape(-1)
    peak_periods = _as_numpy(
        _mapping_get(value, "peak_periods", "periods", default=[])
    ).reshape(-1)
    peak_frequencies = _as_numpy(
        _mapping_get(
            value,
            "peak_frequencies",
            "frequencies",
            default=[],
        )
    ).reshape(-1)
    significant = _as_numpy(
        _mapping_get(
            value,
            "peak_significant",
            "significant",
            "flags",
            default=[],
        )
    ).reshape(-1)
    peak_powers = _as_numpy(
        _mapping_get(value, "peak_powers", "powers", default=[])
    ).reshape(-1)

    if peak_frequencies.size == 0 and peak_periods.size:
        peak_frequencies = np.divide(
            1.0,
            peak_periods,
            out=np.full(peak_periods.shape, np.nan, dtype=float),
            where=peak_periods > 0.0,
        )
    if peak_periods.size == 0 and peak_frequencies.size:
        peak_periods = np.divide(
            1.0,
            peak_frequencies,
            out=np.full(peak_frequencies.shape, np.nan, dtype=float),
            where=peak_frequencies > 0.0,
        )

    candidates = []
    count = max(peak_periods.size, peak_frequencies.size)
    for index in range(count):
        period = (
            _finite_float(peak_periods[index])
            if index < peak_periods.size
            else None
        )
        frequency = (
            _finite_float(peak_frequencies[index])
            if index < peak_frequencies.size
            else None
        )
        if period is None and frequency is not None and frequency > 0.0:
            period = 1.0 / frequency
        if frequency is None and period is not None and period > 0.0:
            frequency = 1.0 / period
        if (
            period is None
            or frequency is None
            or period <= 0.0
            or frequency <= 0.0
        ):
            continue
        candidates.append(
            {
                "rank": index + 1,
                "period": period,
                "frequency": frequency,
                "significant": (
                    bool(significant[index])
                    if index < significant.size
                    else None
                ),
                "power": (
                    _finite_float(peak_powers[index])
                    if index < peak_powers.size
                    else None
                ),
            }
        )

    available = (
        frequency_grid.size > 0
        and power_grid.size == frequency_grid.size
    )
    return {
        "available": available,
        "reason": None if available else "missing full LS frequency/power grid",
        "frequency_grid": frequency_grid,
        "power_grid": power_grid,
        "candidates": candidates,
    }


def _normalize_acf_mapping(value: Any) -> dict[str, Any]:
    """Normalize a data-ACF result as a comparison curve only.

    No peaks, recurrence candidates, or component counts are derived from the
    ACF. Period determinations remain the responsibility of Lomb--Scargle and
    the fitted GP; the ACF is only an independent visual comparison.
    """
    if value is None:
        return {"available": False, "reason": "missing data ACF result"}

    lag = _as_numpy(_mapping_get(value, "lag", "lags", default=[])).reshape(-1)
    acf_values = _as_numpy(
        _mapping_get(
            value,
            "acf",
            "acf_values",
            "correlation",
            "value",
            "values",
            default=[],
        )
    ).reshape(-1)
    if lag.size == 0 or acf_values.size != lag.size:
        return {
            "available": False,
            "reason": "missing compatible data-ACF lag/value arrays",
        }

    return {
        "available": True,
        "lag": lag,
        "acf": acf_values,
        "interpretation": "comparison_curve_only_no_peak_identification",
    }

def cache_period_diagnostic_call(
    lightcurve: Any,
    *,
    kind: str,
    result: Any,
    arguments: dict[str, Any],
) -> None:
    """Cache successful direct calls without changing their return values."""
    cache = dict(getattr(lightcurve, "_period_diagnostic_call_cache", {}))
    signature = period_diagnostic_data_signature(lightcurve)

    if kind == "fit_LS":
        normalized = _normalize_fit_ls_result(
            result,
            arguments=arguments,
        )
        if normalized.get("available"):
            cache["fit_LS"] = {
                "kind": "fit_LS",
                "result": result,
                "normalized": normalized,
                "arguments": dict(arguments),
                "data_signature": signature,
            }
    elif kind == "acf":
        method = str(arguments.get("method", "data")).strip().lower()
        if method == "data":
            cache["acf_data"] = {
                "kind": "acf_data",
                "result": result,
                "arguments": dict(arguments),
                "data_signature": signature,
            }

    lightcurve._period_diagnostic_call_cache = cache


def register_period_diagnostic_evidence(
    lightcurve: Any,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Register structured per-channel LS/ACF evidence on a parent lightcurve."""
    if not isinstance(evidence, dict):
        raise TypeError("period diagnostic evidence must be a dictionary")
    rows = evidence.get("rows")
    if not isinstance(rows, list):
        raise ValueError("period diagnostic evidence must contain a rows list")
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("every period diagnostic row must be a dictionary")
        if not str(row.get("observational_channel", "")).strip():
            raise ValueError(
                "every period diagnostic row must identify an "
                "observational_channel"
            )

    lightcurve._period_diagnostic_evidence = {
        "evidence": copy.deepcopy(evidence),
        "data_signature": period_diagnostic_data_signature(lightcurve),
    }
    return evidence


def _direct_call_evidence(lightcurve: Any) -> dict[str, Any] | None:
    cache = getattr(lightcurve, "_period_diagnostic_call_cache", {})
    ls_record = cache.get("fit_LS")
    acf_record = cache.get("acf_data")
    if ls_record is None or acf_record is None:
        return None

    signature = period_diagnostic_data_signature(lightcurve)
    if (
        ls_record.get("data_signature") != signature
        or acf_record.get("data_signature") != signature
    ):
        return {"stale": True, "rows": []}

    ls_normalized = ls_record.get("normalized") or {}
    acf_normalized = _normalize_acf_mapping(acf_record.get("result"))

    labels = getattr(
        lightcurve,
        "observational_channel_labels",
        getattr(lightcurve, "band", None),
    )
    if labels is None:
        channel = "lightcurve"
    else:
        ordered = list(dict.fromkeys(np.asarray(labels, dtype=str).tolist()))
        channel = ordered[0] if len(ordered) == 1 else "multiband"

    wavelength_values = []
    xdata = _as_numpy(getattr(lightcurve, "xdata", None))
    if xdata.ndim == 2 and xdata.shape[1] > 1:
        wavelength_values = np.unique(xdata[:, 1]).astype(float).tolist()

    return {
        "rows": [
            {
                "observational_channel": channel,
                "physical_wavelengths": wavelength_values,
                "status": "available",
                "lomb_scargle": {
                    "frequency_grid": ls_normalized.get("frequency_grid", []),
                    "power_grid": ls_normalized.get("power_grid", []),
                    "peak_periods": [
                        candidate["period"]
                        for candidate in ls_normalized.get("candidates", [])
                    ],
                    "peak_frequencies": [
                        candidate["frequency"]
                        for candidate in ls_normalized.get("candidates", [])
                    ],
                    "peak_significant": [
                        candidate.get("significant")
                        for candidate in ls_normalized.get("candidates", [])
                    ],
                },
                "acf": {
                    "lag": acf_normalized.get("lag", []),
                    "acf": acf_normalized.get("acf", []),
                    "interpretation": (
                        "comparison_curve_only_no_peak_identification"
                    ),
                },
            }
        ],
        "all_channels_retained_for_consensus": True,
    }

def _registered_evidence(
    lightcurve: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    registered = getattr(lightcurve, "_period_diagnostic_evidence", None)
    if registered is None:
        direct = _direct_call_evidence(lightcurve)
        if direct is None:
            return None, (
                "Lomb--Scargle and data-ACF results are unavailable. "
                "Call fit_LS(..., return_full=True) and acf(method='data'), "
                "or register per-channel evidence."
            )
        if direct.get("stale"):
            return None, (
                "Cached Lomb--Scargle or data-ACF results are stale because "
                "the light-curve observations changed."
            )
        return direct, None

    signature = period_diagnostic_data_signature(lightcurve)
    if registered.get("data_signature") != signature:
        return None, (
            "Registered per-channel Lomb--Scargle/ACF evidence is stale "
            "because the light-curve observations changed."
        )
    return registered.get("evidence"), None


def _fit_is_complete(lightcurve: Any) -> bool:
    model = getattr(
        lightcurve,
        "model",
        getattr(lightcurve, "gp_model", None),
    )
    if model is None:
        return False
    if getattr(lightcurve, "likelihood", None) is None:
        return False

    fit_history = getattr(
        lightcurve,
        "fit_history",
        getattr(lightcurve, "_fit_history", None),
    )
    if fit_history is not None:
        try:
            if len(fit_history) > 0:
                return True
        except TypeError:
            if bool(fit_history):
                return True

    results = getattr(lightcurve, "results", None)
    if results is None:
        return False
    if isinstance(results, dict):
        return bool(results)
    try:
        return len(results) > 0
    except TypeError:
        return True


def _normalize_gp_peaks(summary: Any) -> list[dict[str, Any]]:
    peaks = _mapping_get(summary, "peaks", default=[]) or []
    normalized = []
    for index, peak in enumerate(peaks):
        period = _finite_float(_mapping_get(peak, "period"))
        frequency = _finite_float(_mapping_get(peak, "frequency"))
        if period is None and frequency is not None and frequency > 0.0:
            period = 1.0 / frequency
        if frequency is None and period is not None and period > 0.0:
            frequency = 1.0 / period
        if (
            period is None
            or frequency is None
            or period <= 0.0
            or frequency <= 0.0
        ):
            continue

        interval = _mapping_get(
            peak,
            "interval_period",
            "period_interval",
            default=None,
        )
        period_interval = None
        if interval is not None:
            try:
                lower, upper = interval
            except (TypeError, ValueError):
                lower, upper = None, None
            lower = _finite_float(lower)
            upper = _finite_float(upper)
            if lower is not None and upper is not None:
                period_interval = tuple(sorted((lower, upper)))

        normalized.append(
            {
                "rank": int(_mapping_get(peak, "rank", default=index + 1)),
                "period": period,
                "frequency": frequency,
                "period_interval": period_interval,
                "area_fraction": _finite_float(
                    _mapping_get(peak, "area_fraction")
                ),
                "prominence": _finite_float(_mapping_get(peak, "prominence")),
                "coherence_proxy": _finite_float(
                    _mapping_get(peak, "coherence_proxy")
                ),
            }
        )

    if not normalized:
        period = _finite_float(
            _mapping_get(summary, "dominant_period", default=None)
        )
        frequency = _finite_float(
            _mapping_get(summary, "dominant_frequency", default=None)
        )
        if period is None and frequency is not None and frequency > 0.0:
            period = 1.0 / frequency
        if frequency is None and period is not None and period > 0.0:
            frequency = 1.0 / period
        if period is not None and frequency is not None:
            normalized.append(
                {
                    "rank": 1,
                    "period": period,
                    "frequency": frequency,
                    "period_interval": None,
                    "area_fraction": None,
                    "prominence": None,
                    "coherence_proxy": None,
                }
            )
    return normalized


def _normalize_component_diagnostics(summary: Any) -> list[dict[str, Any]]:
    diagnostics = _mapping_get(
        summary,
        "component_diagnostics",
        default=None,
    )
    if diagnostics is None:
        return []

    periods = _as_numpy(
        _mapping_get(diagnostics, "component_periods", default=[])
    ).reshape(-1)
    frequencies = _as_numpy(
        _mapping_get(diagnostics, "component_frequencies", default=[])
    ).reshape(-1)
    weights = _as_numpy(
        _mapping_get(diagnostics, "component_weights", default=[])
    ).reshape(-1)

    count = max(periods.size, frequencies.size, weights.size)
    rows = []
    for index in range(count):
        period = _finite_float(periods[index]) if index < periods.size else None
        frequency = (
            _finite_float(frequencies[index])
            if index < frequencies.size
            else None
        )
        if period is None and frequency is not None and frequency > 0.0:
            period = 1.0 / frequency
        if frequency is None and period is not None and period > 0.0:
            frequency = 1.0 / period
        rows.append(
            {
                "component_index": index,
                "period": period,
                "frequency": frequency,
                "weight": (
                    _finite_float(weights[index])
                    if index < weights.size
                    else None
                ),
            }
        )
    return rows


def _candidate_match(
    frequency: float,
    anchor_frequency: float,
    *,
    tolerance: float,
    harmonic_orders: tuple[float, ...],
) -> tuple[float, float] | None:
    if frequency <= 0.0 or anchor_frequency <= 0.0:
        return None
    ratio = frequency / anchor_frequency
    best_order = None
    best_distance = math.inf
    for harmonic_order in harmonic_orders:
        distance = abs(ratio - harmonic_order) / harmonic_order
        if distance < best_distance:
            best_distance = distance
            best_order = harmonic_order
    if best_order is None or best_distance > tolerance:
        return None
    return float(best_order), float(best_distance)


def build_period_feature_matches(
    *,
    ls_candidates: list[dict[str, Any]],
    gp_psd_peaks: list[dict[str, Any]],
    tolerance: float,
    harmonic_orders: tuple[float, ...],
) -> list[dict[str, Any]]:
    """Match LS candidates to fitted-GP PSD peaks in frequency space.

    The ACF is deliberately absent: it is a comparison curve, not a period
    candidate generator. Matching never uses candidate rank or array position.
    """
    features = []

    for peak in gp_psd_peaks:
        features.append(
            {
                "feature_id": len(features),
                "anchor_frequency": peak["frequency"],
                "anchor_period": peak["period"],
                "gp_psd_peaks": [copy.deepcopy(peak)],
                "ls_candidates": [],
            }
        )

    for candidate in ls_candidates:
        best = None
        for feature in features:
            match = _candidate_match(
                candidate["frequency"],
                feature["anchor_frequency"],
                tolerance=tolerance,
                harmonic_orders=harmonic_orders,
            )
            if match is None:
                continue
            harmonic_order, distance = match
            score = (distance, feature["feature_id"])
            if best is None or score < best[0]:
                best = (score, feature, harmonic_order, distance)

        candidate_copy = copy.deepcopy(candidate)
        if best is None:
            features.append(
                {
                    "feature_id": len(features),
                    "anchor_frequency": candidate["frequency"],
                    "anchor_period": candidate["period"],
                    "gp_psd_peaks": [],
                    "ls_candidates": [candidate_copy],
                }
            )
            continue

        _, feature, harmonic_order, distance = best
        candidate_copy["matched_harmonic_order"] = harmonic_order
        candidate_copy["fractional_match_distance"] = distance
        feature["ls_candidates"].append(candidate_copy)

    return features

def _consensus_period(lightcurve: Any, evidence: dict[str, Any]) -> float | None:
    candidate_values = [
        evidence.get("consensus_period"),
        evidence.get("final_consensus_period"),
    ]
    for attribute in (
        "consensus_diagnostics",
        "fit_consensus_diagnostics",
        "_consensus_diagnostics",
    ):
        value = getattr(lightcurve, attribute, None)
        if isinstance(value, dict):
            candidate_values.extend(
                [
                    value.get("consensus_period"),
                    value.get("final_consensus_period"),
                ]
            )
            frequency = _finite_float(value.get("final_consensus_frequency"))
            if frequency is not None and frequency > 0.0:
                candidate_values.append(1.0 / frequency)

    for value in candidate_values:
        finite_value = _finite_float(value)
        if finite_value is not None and finite_value > 0.0:
            return finite_value
    return None


def _handle_unavailable(message: str, *, strict: bool) -> dict[str, Any]:
    full_message = "Period diagnostic comparison was not generated:\n" + message
    if strict:
        raise RuntimeError(full_message)
    warnings.warn(full_message, UserWarning, stacklevel=3)
    return {}


def _plot_period_markers(
    ax: Any,
    *,
    ls_candidates: list[dict[str, Any]],
    gp_psd_peaks: list[dict[str, Any]],
    consensus_period: float | None,
) -> None:
    """Overlay LS and GP period references; never derive ACF peaks."""
    for candidate in ls_candidates:
        ax.axvline(
            candidate["period"],
            linestyle="--",
            alpha=0.65,
            label=f"LS #{candidate['rank']}",
        )
    for peak in gp_psd_peaks:
        interval = peak.get("period_interval")
        if interval is not None:
            ax.axvspan(
                interval[0],
                interval[1],
                alpha=0.08,
                label=f"GP PSD #{peak['rank']} interval",
            )
        ax.axvline(
            peak["period"],
            linestyle=":",
            linewidth=1.5,
            label=f"GP PSD #{peak['rank']}",
        )
    if consensus_period is not None:
        ax.axvline(
            consensus_period,
            linewidth=1.2,
            label="Consensus period",
        )

def plot_period_diagnostic_comparison(
    lightcurve: Any,
    *,
    observational_channels: list[str] | tuple[str, ...] | None = None,
    show: bool = True,
    strict: bool = False,
    match_tolerance: float = 0.15,
    harmonic_orders: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0),
    period_summary: Any | None = None,
    period_summary_kwargs: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compare per-channel LS and data-ACF curves with one shared GP summary.

    LS candidates and fitted-GP PSD peaks are the only period features. The
    data ACF is plotted as a curve and receives LS/GP reference lines, but no
    ACF peaks or components are identified. The fitted temporal GP is shared
    by a 2-D light curve, so ``plot_period_summary`` is called exactly once and
    the same global figure is referenced by every plotted channel record.
    """
    if not 0.0 < float(match_tolerance) < 1.0:
        raise ValueError("match_tolerance must lie strictly between 0 and 1")
    if not harmonic_orders or any(order <= 0.0 for order in harmonic_orders):
        raise ValueError("harmonic_orders must contain positive values")

    evidence, evidence_error = _registered_evidence(lightcurve)
    if evidence_error is not None:
        return _handle_unavailable(evidence_error, strict=strict)

    if not _fit_is_complete(lightcurve):
        return _handle_unavailable(
            "A completed GP fit is unavailable. Call lightcurve.fit(...) "
            "before requesting the comparison.",
            strict=strict,
        )

    if period_summary is None:
        try:
            period_summary = lightcurve.get_period_summary()
        except Exception as exc:
            return _handle_unavailable(
                "The fitted model did not provide a period summary: "
                f"{type(exc).__name__}: {exc}",
                strict=strict,
            )

    gp_psd_peaks = _normalize_gp_peaks(period_summary)
    if not gp_psd_peaks:
        return _handle_unavailable(
            "The fitted model does not provide a comparable temporal "
            "period feature.",
            strict=strict,
        )
    gp_kernel_components = _normalize_component_diagnostics(period_summary)

    summary_kwargs = dict(period_summary_kwargs or {})
    summary_kwargs.setdefault("show", False)
    summary_kwargs.setdefault("x_axis", "period")
    summary_kwargs.setdefault("log_x", True)
    summary_kwargs.setdefault("show_components", True)
    summary_result = lightcurve.plot_period_summary(
        summary=period_summary,
        **summary_kwargs,
    )
    if not isinstance(summary_result, tuple) or len(summary_result) != 2:
        raise RuntimeError(
            "plot_period_summary() must return (figure, axes)."
        )
    summary_figure, summary_axes = summary_result

    requested = (
        None
        if observational_channels is None
        else {str(channel) for channel in observational_channels}
    )
    rows = evidence.get("rows", [])
    consensus_period = _consensus_period(lightcurve, evidence)

    comparisons: dict[str, dict[str, Any]] = {}
    for row in rows:
        channel = str(row.get("observational_channel", "")).strip()
        if not channel or (requested is not None and channel not in requested):
            continue

        wavelengths = row.get("physical_wavelengths")
        if wavelengths is None:
            wavelengths = []
        record: dict[str, Any] = {
            "observational_channel": channel,
            "physical_wavelengths": list(wavelengths),
            "status": "skipped",
            "reason": None,
            "comparison_figure": None,
            "comparison_axes": None,
            "period_summary_figure": None,
            "period_summary_axes": None,
            "period_summary_shared": True,
            "features": [],
            "ls_candidates": [],
            "gp_psd_peaks": copy.deepcopy(gp_psd_peaks),
            "gp_kernel_components": copy.deepcopy(gp_kernel_components),
            "acf_interpretation": (
                "comparison_curve_only_no_peak_identification"
            ),
            "gp_period_scope": (
                "shared_2d_temporal_kernel"
                if getattr(lightcurve, "ndim", 1) > 1
                else "single_lightcurve"
            ),
        }
        comparisons[channel] = record

        if row.get("status", "available") != "available":
            record["reason"] = (
                row.get("reason") or "channel diagnostics unavailable"
            )
            continue

        ls_result = _normalize_lomb_scargle_mapping(row.get("lomb_scargle"))
        acf_result = _normalize_acf_mapping(row.get("acf"))
        missing = []
        if not ls_result.get("available"):
            missing.append(ls_result.get("reason") or "Lomb--Scargle")
        if not acf_result.get("available"):
            missing.append(acf_result.get("reason") or "data ACF")
        if missing:
            record["reason"] = "; ".join(missing)
            continue

        ls_candidates = ls_result["candidates"]
        features = build_period_feature_matches(
            ls_candidates=ls_candidates,
            gp_psd_peaks=gp_psd_peaks,
            tolerance=float(match_tolerance),
            harmonic_orders=tuple(float(value) for value in harmonic_orders),
        )

        lag = np.asarray(acf_result["lag"], dtype=float)
        acf_values = np.asarray(acf_result["acf"], dtype=float)
        finite_acf = np.isfinite(lag) & np.isfinite(acf_values)

        figure, axes = plt.subplots(1, 2, figsize=(14, 4.8))
        plot_lomb_scargle_periodogram(
            ls_result["frequency_grid"],
            ls_result["power_grid"],
            candidates=ls_candidates,
            ax=axes[0],
            x_axis="period",
            x_scale="log",
            y_scale="linear",
            title=f"{channel}: Lomb--Scargle candidates",
            show=False,
            legend=False,
        )
        _plot_period_markers(
            axes[0],
            ls_candidates=[],
            gp_psd_peaks=gp_psd_peaks,
            consensus_period=consensus_period,
        )
        axes[0].legend(fontsize=7)

        axes[1].plot(
            lag[finite_acf],
            acf_values[finite_acf],
            marker=".",
            label="Data ACF",
        )
        axes[1].axhline(0.0, linestyle=":", linewidth=1)
        axes[1].set_xlabel("Lag")
        axes[1].set_ylabel("ACF")
        axes[1].set_title(f"{channel}: data ACF comparison curve")
        _plot_period_markers(
            axes[1],
            ls_candidates=ls_candidates,
            gp_psd_peaks=gp_psd_peaks,
            consensus_period=consensus_period,
        )
        axes[1].legend(fontsize=7)

        scope_text = (
            "GP period features come from the shared temporal kernel "
            "of the fitted 2-D model."
            if getattr(lightcurve, "ndim", 1) > 1
            else "GP period features come from this fitted 1-D light curve."
        )
        figure.suptitle(
            f"{channel}: LS / data-ACF / fitted-GP period comparison"
            + chr(10)
            + f"{scope_text}",
            y=1.04,
        )
        figure.tight_layout()

        record.update(
            {
                "status": "plotted",
                "comparison_figure": figure,
                "comparison_axes": axes,
                "period_summary_figure": summary_figure,
                "period_summary_axes": summary_axes,
                "features": features,
                "ls_candidates": copy.deepcopy(ls_candidates),
            }
        )

    if not comparisons:
        return _handle_unavailable(
            "No requested observational channels were present in the "
            "registered period evidence.",
            strict=strict,
        )
    if show:
        plt.show()
    return comparisons
