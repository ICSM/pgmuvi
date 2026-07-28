"""Reusable helpers for one complete real-source PGMUVI analysis.

The public notebook built on this module keeps period evidence, temporal
consensus, wavelength-constraint derivation, GP fitting, predictions, plots,
and residual diagnostics in an explicit scientific order.  The helpers here
normalise those outputs without changing model-selection policy or silently
calibrating observational channels that share a physical wavelength.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from .lightcurve import Lightcurve
from .wavelength_validation_real_lpv import (
    build_representative_lpv_runtime_environment,
)


SINGLE_SOURCE_ANALYSIS_SCHEMA_VERSION = "1.0"
SINGLE_SOURCE_ANALYSIS_STAGE_ORDER = (
    "load_and_validate",
    "sampling_and_variability",
    "per_observational_channel_period_evidence",
    "period_consensus",
    "wavelength_evidence",
    "apply_and_verify_constraints",
    "gp_fit",
    "predictions",
    "plots",
    "residuals",
    "structured_outputs",
)


def single_source_json_safe(value: Any) -> Any:
    """Return a recursively JSON-safe representation without NaN values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return single_source_json_safe(value.item())
    if isinstance(value, np.ndarray):
        return single_source_json_safe(value.tolist())
    if isinstance(value, torch.Tensor):
        return single_source_json_safe(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {
            str(key): single_source_json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [single_source_json_safe(item) for item in value]
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return single_source_json_safe(as_dict())
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return single_source_json_safe(to_dict())
    return str(value)


@contextmanager
def preserve_single_source_analysis_state(
    *,
    seed: int | None = None,
    default_dtype: torch.dtype | None = None,
    working_directory: str | Path | None = None,
) -> Iterator[None]:
    """Temporarily set reproducibility controls and restore caller state.

    Python, NumPy, Torch CPU/CUDA RNG states, Torch's default dtype, the working
    directory, process environment, and Matplotlib rc/interactivity state are
    restored even when analysis raises.  The context does not suppress errors.
    """
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
    ):
        raise TypeError("seed must be an integer or None.")
    if default_dtype is not None and not isinstance(default_dtype, torch.dtype):
        raise TypeError("default_dtype must be a torch.dtype or None.")

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    cuda_states = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )
    original_dtype = torch.get_default_dtype()
    original_cwd = Path.cwd()
    original_environment = dict(os.environ)

    pyplot = None
    rc_params = None
    interactive = None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pass
    else:
        pyplot = plt
        rc_params = copy.deepcopy(dict(plt.rcParams))
        interactive = bool(plt.isinteractive())

    if seed is not None:
        resolved_seed = int(seed)
        random.seed(resolved_seed)
        np.random.seed(resolved_seed % (2**32))
        torch.manual_seed(resolved_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(resolved_seed)
    if default_dtype is not None:
        torch.set_default_dtype(default_dtype)
    if working_directory is not None:
        os.chdir(Path(working_directory))

    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.set_default_dtype(original_dtype)
        os.chdir(original_cwd)
        os.environ.clear()
        os.environ.update(original_environment)
        if pyplot is not None and rc_params is not None:
            pyplot.rcParams.update(rc_params)
            if interactive:
                pyplot.ion()
            else:
                pyplot.ioff()


def _numpy_1d(value: Any, *, name: str, dtype: Any = float) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array


def _raw_single_source_arrays(
    lightcurve: Lightcurve,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    raw_x = getattr(lightcurve, "_xdata_raw", None)
    raw_y = getattr(lightcurve, "_ydata_raw", None)
    if raw_x is None or raw_y is None:
        raise ValueError("Lightcurve raw input arrays are required.")

    if isinstance(raw_x, torch.Tensor):
        raw_x = raw_x.detach().cpu().numpy()
    x = np.asarray(raw_x, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError(
            "A single-source multiwavelength analysis requires 2-D inputs "
            "with time in dimension 0 and physical wavelength in dimension 1."
        )
    y = _numpy_1d(raw_y, name="flux")
    if x.shape[0] != y.size:
        raise ValueError("Lightcurve time/wavelength and flux rows differ.")

    raw_yerr = getattr(lightcurve, "_yerr_raw", None)
    yerr = None
    if raw_yerr is not None:
        yerr = _numpy_1d(raw_yerr, name="flux uncertainty")
        if yerr.size != y.size:
            raise ValueError("Flux and uncertainty rows differ.")

    labels = getattr(lightcurve, "band", None)
    if labels is None:
        raise ValueError(
            "One observational-channel label per row is required."
        )
    channels = _numpy_1d(
        labels,
        name="observational-channel labels",
        dtype=str,
    )
    if channels.size != y.size:
        raise ValueError("Flux and observational-channel rows differ.")
    if np.any(np.char.strip(channels) == ""):
        raise ValueError("Observational-channel labels must be non-empty.")
    return x, y, yerr, channels


def _positive_time_spacing_summary(times: np.ndarray) -> dict[str, Any]:
    unique = np.unique(np.asarray(times, dtype=float))
    if unique.size < 2:
        return {
            "n_unique_times": int(unique.size),
            "median_positive_cadence": None,
            "largest_positive_gap": None,
        }
    spacing = np.diff(unique)
    positive = spacing[spacing > 0.0]
    return {
        "n_unique_times": int(unique.size),
        "median_positive_cadence": (
            float(np.median(positive)) if positive.size else None
        ),
        "largest_positive_gap": (
            float(np.max(positive)) if positive.size else None
        ),
    }


def summarize_single_source_observational_channels(
    lightcurve: Lightcurve,
) -> dict[str, Any]:
    """Summarize sampling and variability without merging channel identity."""
    x, flux, flux_error, channels = _raw_single_source_arrays(lightcurve)
    times = x[:, 0]
    wavelengths = x[:, 1]
    ordered_channels = list(dict.fromkeys(channels.tolist()))

    rows = []
    for channel in ordered_channels:
        mask = channels == channel
        channel_time = times[mask]
        channel_flux = flux[mask]
        channel_wavelengths = np.unique(wavelengths[mask])
        cadence = _positive_time_spacing_summary(channel_time)
        flux_median = float(np.median(channel_flux))
        median_absolute_deviation = float(
            np.median(np.abs(channel_flux - flux_median))
        )
        row = {
            "observational_channel": channel,
            "n_points": int(np.count_nonzero(mask)),
            "physical_wavelengths": channel_wavelengths.tolist(),
            "n_physical_wavelengths": int(channel_wavelengths.size),
            "time_start": float(np.min(channel_time)),
            "time_end": float(np.max(channel_time)),
            "time_baseline": float(
                np.max(channel_time) - np.min(channel_time)
            ),
            "flux_median": flux_median,
            "flux_mad": median_absolute_deviation,
            **cadence,
        }
        if flux_error is not None:
            row["median_flux_error"] = float(np.median(flux_error[mask]))
        else:
            row["median_flux_error"] = None
        rows.append(row)

    wavelength_groups = []
    for wavelength in np.unique(wavelengths):
        group_channels = list(
            dict.fromkeys(channels[wavelengths == wavelength].tolist())
        )
        if len(group_channels) > 1:
            wavelength_groups.append(
                {
                    "physical_wavelength": float(wavelength),
                    "observational_channels": group_channels,
                    "multiplet_size": len(group_channels),
                }
            )

    return {
        "dimension_order": ["time", "physical_wavelength"],
        "flux_domain": "linear",
        "n_rows": int(flux.size),
        "n_observational_channels": len(ordered_channels),
        "n_physical_wavelengths": int(np.unique(wavelengths).size),
        "observational_channel_order": ordered_channels,
        "observational_channel_rows": rows,
        "duplicate_physical_wavelength_groups": wavelength_groups,
        "global_time_start": float(np.min(times)),
        "global_time_end": float(np.max(times)),
        "global_time_baseline": float(np.max(times) - np.min(times)),
        "global_sampling": _positive_time_spacing_summary(times),
    }


def build_per_observational_channel_period_evidence(
    lightcurve: Lightcurve,
    *,
    num_peaks: int = 3,
    single_threshold: float = 0.05,
    nyquist_factor: int = 5,
    acf_n_lags: int = 50,
    minimum_points: int = 5,
) -> dict[str, Any]:
    """Compute independent LS and data-ACF evidence for every channel.

    Failures are retained per observational channel so one unsuitable channel
    does not erase valid evidence from the rest of the source.
    """
    if isinstance(num_peaks, bool) or not isinstance(num_peaks, int):
        raise TypeError("num_peaks must be an integer.")
    if num_peaks < 1:
        raise ValueError("num_peaks must be positive.")
    if isinstance(minimum_points, bool) or not isinstance(minimum_points, int):
        raise TypeError("minimum_points must be an integer.")
    if minimum_points < 3:
        raise ValueError("minimum_points must be at least 3.")

    x, flux, flux_error, channels = _raw_single_source_arrays(lightcurve)
    times = x[:, 0]
    wavelengths = x[:, 1]
    ordered_channels = list(dict.fromkeys(channels.tolist()))
    rows = []

    for channel in ordered_channels:
        mask = channels == channel
        channel_time = times[mask]
        channel_flux = flux[mask]
        channel_error = flux_error[mask] if flux_error is not None else None
        wavelength_values = np.unique(wavelengths[mask])
        row = {
            "observational_channel": channel,
            "physical_wavelengths": wavelength_values.tolist(),
            "n_points": int(channel_time.size),
            "status": "unavailable",
            "reason": None,
            "lomb_scargle": None,
            "acf": None,
        }

        if channel_time.size < minimum_points:
            row["reason"] = "insufficient_points"
            rows.append(row)
            continue
        if np.unique(channel_time).size < 3:
            row["reason"] = "insufficient_unique_times"
            rows.append(row)
            continue
        if float(np.max(channel_time) - np.min(channel_time)) <= 0.0:
            row["reason"] = "nonpositive_time_baseline"
            rows.append(row)
            continue

        try:
            channel_lightcurve = Lightcurve(
                torch.as_tensor(
                    channel_time,
                    dtype=torch.get_default_dtype(),
                ),
                torch.as_tensor(
                    channel_flux,
                    dtype=torch.get_default_dtype(),
                ),
                yerr=(
                    torch.as_tensor(
                        channel_error,
                        dtype=torch.get_default_dtype(),
                    )
                    if channel_error is not None
                    else None
                ),
                name=f"{getattr(lightcurve, 'name', 'source')}:{channel}",
                check_sampling=False,
                check_variability=False,
                max_samples=None,
                max_samples_per_band=None,
                center_time="auto",
            )
            peak_frequency, significant, frequency, power = (
                channel_lightcurve.fit_LS(
                    num_peaks=num_peaks,
                    single_threshold=single_threshold,
                    Nyquist_factor=nyquist_factor,
                    return_full=True,
                )
            )
            acf_result = channel_lightcurve.acf(
                method="data",
                n_lags=acf_n_lags,
            )

            peak_frequency_values = _numpy_1d(
                peak_frequency,
                name="peak frequencies",
            )
            significant_values = _numpy_1d(
                significant,
                name="significance mask",
                dtype=bool,
            )
            frequency_values = _numpy_1d(
                frequency,
                name="frequency grid",
            )
            power_values = _numpy_1d(power, name="power grid")
            periods = np.full(
                peak_frequency_values.shape,
                np.nan,
                dtype=float,
            )
            positive_frequency = peak_frequency_values > 0.0
            periods[positive_frequency] = (
                1.0 / peak_frequency_values[positive_frequency]
            )

            acf_lag = _numpy_1d(acf_result.lag, name="ACF lag")
            acf_value = _numpy_1d(acf_result.acf, name="ACF value")
            acf_count = (
                _numpy_1d(acf_result.counts, name="ACF count")
                if acf_result.counts is not None
                else None
            )
            valid_acf_peak = (
                (acf_lag > 0.0)
                & np.isfinite(acf_value)
                & ((acf_count > 0.0) if acf_count is not None else True)
            )
            acf_peak_lag = None
            if np.any(valid_acf_peak):
                valid_indices = np.flatnonzero(valid_acf_peak)
                best_index = valid_indices[
                    int(np.argmax(acf_value[valid_acf_peak]))
                ]
                acf_peak_lag = float(acf_lag[best_index])

            row.update(
                {
                    "status": "available",
                    "lomb_scargle": {
                        "peak_frequencies": peak_frequency_values,
                        "peak_periods": periods,
                        "significant": significant_values,
                        "frequency_grid": frequency_values,
                        "power_grid": power_values,
                        "dominant_grid_frequency": (
                            float(frequency_values[np.argmax(power_values)])
                            if frequency_values.size and power_values.size
                            else None
                        ),
                    },
                    "acf": {
                        "lag": acf_lag,
                        "value": acf_value,
                        "counts": acf_count,
                        "strongest_positive_lag": acf_peak_lag,
                        "interpretation": (
                            "diagnostic_only_not_a_consensus_period"
                        ),
                    },
                }
            )
        except Exception as exc:
            row["reason"] = "period_evidence_failed"
            row["failure"] = {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }
        rows.append(row)

    return single_source_json_safe(
        {
            "scope": "all_observational_channels_independently",
            "all_channels_retained_for_consensus": True,
            "n_observational_channels": len(ordered_channels),
            "n_available": sum(row["status"] == "available" for row in rows),
            "n_unavailable": sum(
                row["status"] != "available" for row in rows
            ),
            "rows": rows,
        }
    )


def summarize_single_source_noise_provenance(
    lightcurve: Lightcurve,
) -> dict[str, Any]:
    """Separate supplied measurement variance from learned additional noise."""
    likelihood = getattr(lightcurve, "likelihood", None)
    if likelihood is None:
        return {
            "available": False,
            "reason": "likelihood_not_configured",
        }

    def covariance_noise(covariance: Any) -> np.ndarray | None:
        if covariance is None or not hasattr(covariance, "noise"):
            return None
        return _numpy_1d(covariance.noise, name="likelihood noise")

    fixed = covariance_noise(getattr(likelihood, "noise_covar", None))
    additional = covariance_noise(
        getattr(likelihood, "second_noise_covar", None)
    )

    def summary(values: np.ndarray | None) -> dict[str, Any] | None:
        if values is None or values.size == 0:
            return None
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {
                "n_values": int(values.size),
                "n_finite": 0,
                "minimum": None,
                "median": None,
                "maximum": None,
            }
        return {
            "n_values": int(values.size),
            "n_finite": int(finite.size),
            "minimum": float(np.min(finite)),
            "median": float(np.median(finite)),
            "maximum": float(np.max(finite)),
        }

    return single_source_json_safe(
        {
            "available": True,
            "likelihood_class": type(likelihood).__name__,
            "measurement_noise_semantics": (
                "supplied flux_error values interpreted as standard "
                "deviations and squared to fixed variances"
            ),
            "fixed_measurement_variance": summary(fixed),
            "learn_additional_noise": bool(
                getattr(lightcurve, "_learn_additional_noise", False)
            ),
            "initial_additional_noise_variance": getattr(
                lightcurve,
                "_initial_additional_noise_variance",
                None,
            ),
            "fitted_additional_noise_variance": summary(additional),
            "additional_noise_is_not_measurement_error_replacement": True,
        }
    )


def build_phase_folded_prediction_summary(
    predictions: Mapping[str, Any],
    *,
    period: float,
    reference_time: float | None = None,
) -> dict[str, Any]:
    """Build row-aligned phase and residual diagnostics for a positive period."""
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("period must be finite and strictly positive.")

    time = _numpy_1d(predictions["time"], name="prediction time")
    channel = _numpy_1d(
        predictions["observational_channel"],
        name="prediction observational channel",
        dtype=str,
    )
    wavelength = _numpy_1d(
        predictions["physical_wavelength"],
        name="prediction physical wavelength",
    )
    residual = _numpy_1d(predictions["residual"], name="prediction residual")
    if not (
        time.size == channel.size == wavelength.size == residual.size
    ):
        raise ValueError("Prediction arrays must have equal row counts.")

    resolved_reference_time = (
        float(np.min(time)) if reference_time is None else float(reference_time)
    )
    phase = np.mod(time - resolved_reference_time, period) / period
    ordered_channels = list(dict.fromkeys(channel.tolist()))
    channel_rows = []
    for label in ordered_channels:
        mask = channel == label
        finite = np.isfinite(residual[mask])
        channel_rows.append(
            {
                "observational_channel": label,
                "n_points": int(np.count_nonzero(mask)),
                "phase_minimum": float(np.min(phase[mask])),
                "phase_maximum": float(np.max(phase[mask])),
                "phase_span": float(
                    np.max(phase[mask]) - np.min(phase[mask])
                ),
                "residual_rmse": (
                    float(np.sqrt(np.mean(residual[mask][finite] ** 2)))
                    if np.any(finite)
                    else None
                ),
            }
        )

    return single_source_json_safe(
        {
            "period": float(period),
            "reference_time": resolved_reference_time,
            "phase": phase,
            "time": time,
            "physical_wavelength": wavelength,
            "observational_channel": channel,
            "residual": residual,
            "observational_channel_rows": channel_rows,
        }
    )


def summarize_single_source_fit_quality(
    residual_rows: Sequence[Mapping[str, Any]],
    *,
    warning_threshold: float = 2.0,
    severe_threshold: float = 3.0,
) -> dict[str, Any]:
    """Screen channel-level standardized residual RMS values for review.

    A standardized residual RMS near one is the reference expectation when the
    fitted mean and predictive variance describe the training data adequately.
    Values above ``warning_threshold`` are surfaced for scientific review, and
    values above ``severe_threshold`` are labelled severe.  This is a diagnostic
    screen, not an automatic model-rejection rule.
    """
    if (
        not np.isfinite(warning_threshold)
        or not np.isfinite(severe_threshold)
        or warning_threshold <= 0.0
        or severe_threshold <= warning_threshold
    ):
        raise ValueError(
            "Require finite thresholds with "
            "0 < warning_threshold < severe_threshold."
        )
    if not isinstance(residual_rows, Sequence) or isinstance(
        residual_rows,
        (str, bytes),
    ):
        raise TypeError("residual_rows must be a sequence of mappings.")

    normalized_rows = []
    warning_rows = []
    severe_rows = []
    for residual_row in residual_rows:
        if not isinstance(residual_row, Mapping):
            raise TypeError("Every residual row must be a mapping.")
        channel = str(
            residual_row.get("observational_channel", "")
        ).strip()
        if not channel:
            raise ValueError(
                "Every residual row requires an observational channel."
            )
        value = residual_row.get("standardized_residual_rms")
        try:
            standardized_rms = float(value)
        except (TypeError, ValueError):
            standardized_rms = float("nan")

        if not np.isfinite(standardized_rms):
            status = "unavailable"
        elif standardized_rms > severe_threshold:
            status = "severe"
        elif standardized_rms > warning_threshold:
            status = "warning"
        else:
            status = "within_screening_threshold"

        row = {
            "observational_channel": channel,
            "physical_wavelength": residual_row.get(
                "physical_wavelength"
            ),
            "n_points": residual_row.get("n_points"),
            "standardized_residual_rms": (
                standardized_rms
                if np.isfinite(standardized_rms)
                else None
            ),
            "status": status,
        }
        normalized_rows.append(row)
        if status in {"warning", "severe"}:
            warning_rows.append(row)
        if status == "severe":
            severe_rows.append(row)

    finite_values = [
        row["standardized_residual_rms"]
        for row in normalized_rows
        if row["standardized_residual_rms"] is not None
    ]
    overall_status = (
        "requires_review"
        if warning_rows
        else "no_large_standardized_residual_rms_detected"
    )
    warning_messages = []
    if warning_rows:
        channel_text = ", ".join(
            (
                f"{row['observational_channel']}="
                f"{row['standardized_residual_rms']:.3g}"
                f" ({row['status']})"
            )
            for row in warning_rows
        )
        warning_messages.append(
            "Channel-level standardized training-residual RMS exceeds "
            f"the review threshold {warning_threshold:g}: {channel_text}. "
            "Inspect residual structure, predictive uncertainty, sampling, "
            "and model adequacy before scientific interpretation."
        )

    return single_source_json_safe(
        {
            "status": overall_status,
            "diagnostic_only": True,
            "automatic_model_rejection": False,
            "reference_standardized_residual_rms": 1.0,
            "warning_threshold": float(warning_threshold),
            "severe_threshold": float(severe_threshold),
            "n_observational_channels": len(normalized_rows),
            "n_channels_requiring_review": len(warning_rows),
            "n_severe_channels": len(severe_rows),
            "maximum_standardized_residual_rms": (
                max(finite_values) if finite_values else None
            ),
            "channels": normalized_rows,
            "channels_requiring_review": warning_rows,
            "severe_channels": severe_rows,
            "warning_messages": warning_messages,
            "interpretation": (
                "This screen identifies channel-level mismatch between "
                "training residuals and predictive uncertainty. It does not "
                "by itself validate or reject the GP model."
            ),
        }
    )


def validate_single_source_stage_records(
    stage_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate unique, monotonic stage records against the public workflow."""
    if not isinstance(stage_records, Sequence) or isinstance(
        stage_records,
        (str, bytes),
    ):
        raise TypeError("stage_records must be a sequence of mappings.")

    expected_index = {
        name: index
        for index, name in enumerate(SINGLE_SOURCE_ANALYSIS_STAGE_ORDER)
    }
    normalized = []
    seen = set()
    previous_index = -1
    for record in stage_records:
        if not isinstance(record, Mapping):
            raise TypeError("Every stage record must be a mapping.")
        stage = str(record.get("stage", "")).strip()
        if stage not in expected_index:
            raise ValueError(f"Unknown single-source analysis stage: {stage!r}.")
        if stage in seen:
            raise ValueError(f"Duplicate single-source analysis stage: {stage!r}.")
        current_index = expected_index[stage]
        if current_index <= previous_index:
            raise ValueError(
                "Single-source analysis stages are not in scientific order."
            )
        status = str(record.get("status", "")).strip().lower()
        if status not in {"completed", "failed", "unavailable", "skipped"}:
            raise ValueError(
                f"Invalid status {status!r} for stage {stage!r}."
            )
        normalized.append(
            {
                **dict(record),
                "stage": stage,
                "status": status,
            }
        )
        seen.add(stage)
        previous_index = current_index
    return single_source_json_safe(normalized)


def build_single_source_analysis_report(
    *,
    source_id: str,
    source_path: str | Path,
    stage_records: Sequence[Mapping[str, Any]],
    input_summary: Mapping[str, Any],
    period_evidence: Mapping[str, Any],
    consensus: Mapping[str, Any],
    wavelength_evidence: Mapping[str, Any],
    constraint_diagnostics: Mapping[str, Any],
    fit_diagnostics: Mapping[str, Any],
    prediction_diagnostics: Mapping[str, Any],
    residual_diagnostics: Mapping[str, Any],
    phase_diagnostics: Mapping[str, Any] | None,
    noise_provenance: Mapping[str, Any],
    duplicate_wavelength_resolution: Mapping[str, Any],
    warnings: Sequence[Any] = (),
    failures: Sequence[Any] = (),
    runtime_environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one strict JSON-safe report for the maintained notebook."""
    normalized_stages = validate_single_source_stage_records(stage_records)
    report = {
        "schema": "pgmuvi.single_source_analysis",
        "schema_version": SINGLE_SOURCE_ANALYSIS_SCHEMA_VERSION,
        "source_id": str(source_id),
        "source_path": str(source_path),
        "stage_order": list(SINGLE_SOURCE_ANALYSIS_STAGE_ORDER),
        "stage_records": normalized_stages,
        "input_summary": input_summary,
        "period_evidence": period_evidence,
        "consensus": consensus,
        "wavelength_evidence": wavelength_evidence,
        "constraint_diagnostics": constraint_diagnostics,
        "fit_diagnostics": fit_diagnostics,
        "prediction_diagnostics": prediction_diagnostics,
        "residual_diagnostics": residual_diagnostics,
        "phase_diagnostics": phase_diagnostics,
        "noise_provenance": noise_provenance,
        "duplicate_wavelength_resolution": duplicate_wavelength_resolution,
        "warnings": list(warnings),
        "failures": list(failures),
        "runtime_environment": (
            dict(runtime_environment)
            if runtime_environment is not None
            else build_representative_lpv_runtime_environment()
        ),
    }
    safe = single_source_json_safe(report)
    json.dumps(safe, allow_nan=False)
    return safe


def write_single_source_analysis_report(
    path: str | Path,
    report: Mapping[str, Any],
) -> Path:
    """Write a validated single-source report as deterministic JSON text."""
    output_path = Path(path)
    payload = single_source_json_safe(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path
