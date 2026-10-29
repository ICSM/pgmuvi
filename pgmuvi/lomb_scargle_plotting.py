"""Maintained Lomb--Scargle plotting utilities.

The plotting contract defaults to period on a logarithmic x-axis and
Lomb--Scargle power on a linear y-axis.  Multiple LS candidates and reference
periods are retained explicitly; no single-component assumption is made.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np


_VALID_AXES = {"period", "frequency"}
_VALID_SCALES = {"linear", "log"}


def _as_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=float)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _finite_positive(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result <= 0.0:
        return None
    return result


def _normalize_candidates(
    candidates: Sequence[Mapping[str, Any]] | None,
    *,
    candidate_frequencies: Any = None,
    candidate_periods: Any = None,
    candidate_significant: Any = None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates or (), start=1):
        period = _finite_positive(candidate.get("period"))
        frequency = _finite_positive(candidate.get("frequency"))
        if period is None and frequency is not None:
            period = 1.0 / frequency
        if frequency is None and period is not None:
            frequency = 1.0 / period
        if period is None or frequency is None:
            continue
        normalized.append(
            {
                "rank": int(candidate.get("rank", index)),
                "period": period,
                "frequency": frequency,
                "significant": candidate.get("significant"),
                "label": candidate.get("label"),
            }
        )

    frequencies = _as_numpy(candidate_frequencies).reshape(-1)
    periods = _as_numpy(candidate_periods).reshape(-1)
    significant = _as_numpy(candidate_significant).reshape(-1)
    count = max(frequencies.size, periods.size)
    for index in range(count):
        frequency = (
            _finite_positive(frequencies[index])
            if index < frequencies.size
            else None
        )
        period = (
            _finite_positive(periods[index])
            if index < periods.size
            else None
        )
        if period is None and frequency is not None:
            period = 1.0 / frequency
        if frequency is None and period is not None:
            frequency = 1.0 / period
        if period is None or frequency is None:
            continue
        normalized.append(
            {
                "rank": len(normalized) + 1,
                "period": period,
                "frequency": frequency,
                "significant": (
                    bool(significant[index])
                    if index < significant.size
                    else None
                ),
                "label": None,
            }
        )
    return normalized


def _normalize_reference_periods(
    reference_periods: Mapping[str, Any]
    | Sequence[Mapping[str, Any] | tuple[str, Any]]
    | None,
) -> list[dict[str, Any]]:
    if reference_periods is None:
        return []

    if isinstance(reference_periods, Mapping):
        items = list(reference_periods.items())
    else:
        items = []
        for entry in reference_periods:
            if isinstance(entry, Mapping):
                items.append((entry.get("label"), entry.get("period")))
            else:
                label, period = entry
                items.append((label, period))

    normalized = []
    for label, period_value in items:
        period = _finite_positive(period_value)
        if period is None:
            continue
        normalized.append(
            {
                "label": str(label) if label is not None else "Reference period",
                "period": period,
                "frequency": 1.0 / period,
            }
        )
    return normalized


def _deduplicate_legend(ax: Any) -> None:
    handles, labels = ax.get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels, strict=True):
        if label and label not in unique:
            unique[label] = handle
    if unique:
        ax.legend(unique.values(), unique.keys())


def plot_lomb_scargle_periodogram(
    frequency: Any,
    power: Any,
    *,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    candidate_frequencies: Any = None,
    candidate_periods: Any = None,
    candidate_significant: Any = None,
    reference_periods: Mapping[str, Any]
    | Sequence[Mapping[str, Any] | tuple[str, Any]]
    | None = None,
    ax: Any = None,
    x_axis: str = "period",
    x_scale: str | None = None,
    y_scale: str = "linear",
    label: str | None = None,
    title: str | None = None,
    show: bool = True,
    candidate_label_prefix: str = "LS",
    mark_candidates: bool = True,
    legend: bool = True,
    **plot_kwargs: Any,
):
    """Plot a full Lomb--Scargle periodogram using the maintained axis policy.

    Parameters
    ----------
    frequency, power
        Full frequency and Lomb--Scargle power grids.  They are plotted without
        recomputation.
    candidates
        Structured candidate records.  Every valid candidate is marked; the
        function never silently reduces a multi-component result to one peak.
    candidate_frequencies, candidate_periods, candidate_significant
        Array-style alternative to ``candidates``.
    reference_periods
        Labeled reference periods, including multiple injected or fitted
        components.
    ax
        Existing axes for overlays.  A new figure and axes are created when
        omitted.
    x_axis
        ``"period"`` by default; ``"frequency"`` is available for specialized
        diagnostics.
    x_scale
        Defaults to ``"log"`` for period and ``"linear"`` for frequency.
    y_scale
        Defaults to ``"linear"``.
    show
        Call ``plt.show()`` when true.

    Returns
    -------
    fig, ax
        The figure and axes containing the plot.
    """
    x_axis = str(x_axis).strip().lower()
    if x_axis not in _VALID_AXES:
        raise ValueError(f"x_axis must be one of {sorted(_VALID_AXES)}")

    if x_scale is None:
        x_scale = "log" if x_axis == "period" else "linear"
    x_scale = str(x_scale).strip().lower()
    y_scale = str(y_scale).strip().lower()
    if x_scale not in _VALID_SCALES:
        raise ValueError(f"x_scale must be one of {sorted(_VALID_SCALES)}")
    if y_scale not in _VALID_SCALES:
        raise ValueError(f"y_scale must be one of {sorted(_VALID_SCALES)}")

    frequency_array = _as_numpy(frequency).astype(float, copy=False).reshape(-1)
    power_array = _as_numpy(power).astype(float, copy=False).reshape(-1)
    if frequency_array.size != power_array.size:
        raise ValueError("frequency and power must have the same number of values")

    finite = (
        np.isfinite(frequency_array)
        & np.isfinite(power_array)
        & (frequency_array > 0.0)
    )
    if y_scale == "log":
        finite &= power_array > 0.0
    if not np.any(finite):
        raise ValueError("no finite positive-frequency periodogram values remain")

    frequency_array = frequency_array[finite]
    power_array = power_array[finite]
    x_values = (
        1.0 / frequency_array
        if x_axis == "period"
        else frequency_array
    )
    order = np.argsort(x_values)
    x_values = x_values[order]
    power_array = power_array[order]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure

    ax.plot(x_values, power_array, label=label, **plot_kwargs)
    normalized_candidates = _normalize_candidates(
        candidates,
        candidate_frequencies=candidate_frequencies,
        candidate_periods=candidate_periods,
        candidate_significant=candidate_significant,
    )
    if mark_candidates:
        for candidate in normalized_candidates:
            x_value = (
                candidate["period"]
                if x_axis == "period"
                else candidate["frequency"]
            )
            candidate_label = candidate.get("label") or (
                f"{candidate_label_prefix} #{candidate['rank']}"
            )
            significant = candidate.get("significant")
            ax.axvline(
                x_value,
                linestyle="--" if significant is not False else ":",
                alpha=0.75 if significant is not False else 0.45,
                label=candidate_label,
            )

    for reference in _normalize_reference_periods(reference_periods):
        x_value = (
            reference["period"]
            if x_axis == "period"
            else reference["frequency"]
        )
        ax.axvline(
            x_value,
            linestyle=":",
            linewidth=1.2,
            label=reference["label"],
        )

    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    ax.set_xlabel("Period" if x_axis == "period" else "Frequency")
    ax.set_ylabel("Lomb--Scargle power")
    if title is not None:
        ax.set_title(title)
    if legend:
        _deduplicate_legend(ax)
    if show:
        plt.show()
    return fig, ax


def plot_lightcurve_lomb_scargle_periodogram(
    lightcurve: Any,
    frequency: Any = None,
    power: Any = None,
    *,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    strict: bool = False,
    **kwargs: Any,
):
    """Plot explicit or cached full LS output from a ``Lightcurve``.

    This function never calls ``fit_LS`` itself.  When arrays are omitted, it
    uses the most recent compatible full-grid cache created by
    ``fit_LS(freq_only=True)`` or ``fit_LS(return_full=True)``.
    """
    if (frequency is None) != (power is None):
        raise ValueError("frequency and power must be supplied together")

    if frequency is None:
        cache = getattr(lightcurve, "_period_diagnostic_call_cache", {})
        record = cache.get("fit_LS")
        normalized = record.get("normalized", {}) if record else {}
        if not normalized.get("available"):
            message = (
                "Lomb--Scargle periodogram was not generated: no cached full "
                "frequency/power grid is available. Call "
                "fit_LS(freq_only=True) or fit_LS(return_full=True), or pass "
                "frequency and power explicitly."
            )
            if strict:
                raise RuntimeError(message)
            warnings.warn(message, UserWarning, stacklevel=2)
            return None, None
        frequency = normalized["frequency_grid"]
        power = normalized["power_grid"]
        if candidates is None:
            candidates = normalized.get("candidates", [])

    return plot_lomb_scargle_periodogram(
        frequency,
        power,
        candidates=candidates,
        **kwargs,
    )
