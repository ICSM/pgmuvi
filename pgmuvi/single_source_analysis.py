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
from itertools import combinations
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



def resolve_single_source_period_component_configuration(
    *,
    ls_num_components: int,
    gp_num_components: int,
) -> dict[str, Any]:
    """Resolve independent LS-candidate and fitted-GP component controls.

    ``ls_num_components`` controls how many Lomb--Scargle candidates are
    requested per observational channel. ``gp_num_components`` controls the
    fitted temporal GP. A single GP component uses the explicit-period
    quasi-periodic consensus path. More than one GP component uses the
    spectral-mixture multi-component consensus path with exactly the requested
    number of mixtures. The LS and GP counts are intentionally independent.
    """

    def positive_integer(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value < 1:
            raise ValueError(f"{name} must be positive.")
        return int(value)

    ls_count = positive_integer(
        "ls_num_components",
        ls_num_components,
    )
    gp_count = positive_integer(
        "gp_num_components",
        gp_num_components,
    )

    if gp_count == 1:
        return {
            "ls_num_components": ls_count,
            "gp_num_components": gp_count,
            "fit_strategy": "consensus",
            "time_kernel_type": "quasi_periodic",
            "fit_kwargs": {},
        }

    return {
        "ls_num_components": ls_count,
        "gp_num_components": gp_count,
        "fit_strategy": "consensus_multicomp",
        "time_kernel_type": "spectral_mixture",
        "fit_kwargs": {
            "num_mixtures": gp_count,
            "max_components_per_band": ls_count,
        },
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
                        "interpretation": (
                            "comparison_curve_only_no_peak_identification"
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
    """Build row-aligned phase and scale-aware residual diagnostics."""
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
    residual = _numpy_1d(
        predictions["residual"],
        name="prediction residual",
    )
    if "standardized_residual" in predictions:
        standardized_residual = _numpy_1d(
            predictions["standardized_residual"],
            name="prediction standardized residual",
        )
        standardized_residual_source = "provided"
    elif "predictive_standard_deviation" in predictions:
        predictive_standard_deviation = _numpy_1d(
            predictions["predictive_standard_deviation"],
            name="prediction predictive standard deviation",
        )
        standardized_residual = np.full(
            residual.shape,
            np.nan,
            dtype=float,
        )
        usable = (
            np.isfinite(residual)
            & np.isfinite(predictive_standard_deviation)
            & (predictive_standard_deviation > 0.0)
        )
        standardized_residual[usable] = (
            residual[usable] / predictive_standard_deviation[usable]
        )
        standardized_residual_source = (
            "derived_from_predictive_standard_deviation"
        )
    else:
        standardized_residual = np.full(
            residual.shape,
            np.nan,
            dtype=float,
        )
        standardized_residual_source = "unavailable"
    if not (
        time.size
        == channel.size
        == wavelength.size
        == residual.size
        == standardized_residual.size
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
        finite_residual = np.isfinite(residual[mask])
        finite_standardized = np.isfinite(standardized_residual[mask])
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
                    float(
                        np.sqrt(
                            np.mean(residual[mask][finite_residual] ** 2)
                        )
                    )
                    if np.any(finite_residual)
                    else None
                ),
                "standardized_residual_rms": (
                    float(
                        np.sqrt(
                            np.mean(
                                standardized_residual[mask][
                                    finite_standardized
                                ]
                                ** 2
                            )
                        )
                    )
                    if np.any(finite_standardized)
                    else None
                ),
                "empirical_95_percent_coverage": (
                    float(
                        np.mean(
                            np.abs(
                                standardized_residual[mask][
                                    finite_standardized
                                ]
                            )
                            <= 1.959963984540054
                        )
                    )
                    if np.any(finite_standardized)
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
            "standardized_residual": standardized_residual,
            "standardized_residual_source": (
                standardized_residual_source
            ),
            "observational_channel_rows": channel_rows,
            "combined_signed_residual_metric": "standardized_residual",
            "raw_signed_residual_presentation": (
                "per_channel_independent_y_ranges"
            ),
        }
    )


def _robust_scale(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return None
    centered = finite - np.median(finite)
    scale = 1.4826 * float(np.median(np.abs(centered)))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.std(finite))
    return scale if np.isfinite(scale) and scale > 0.0 else None


def _circular_phase_coverage(phases: np.ndarray) -> float | None:
    finite = np.mod(np.asarray(phases, dtype=float), 1.0)
    finite = np.unique(finite[np.isfinite(finite)])
    if finite.size < 2:
        return None
    wrapped = np.concatenate([finite, finite[:1] + 1.0])
    largest_gap = float(np.max(np.diff(wrapped)))
    return float(np.clip(1.0 - largest_gap, 0.0, 1.0))


def _normalized_kernel_correlation(
    kernel: Any,
    left: torch.Tensor,
    right: torch.Tensor,
) -> np.ndarray:
    with torch.no_grad():
        cross = kernel(left, right).to_dense()
        left_variance = kernel(left).to_dense().diagonal()
        right_variance = kernel(right).to_dense().diagonal()
    denominator = torch.sqrt(
        torch.clamp(left_variance, min=0.0)[:, None]
        * torch.clamp(right_variance, min=0.0)[None, :]
    )
    correlation = torch.full_like(cross, float('nan'))
    usable = torch.isfinite(denominator) & (denominator > 0.0)
    correlation[usable] = cross[usable] / denominator[usable]
    return correlation.detach().cpu().numpy()


def summarize_single_source_fit_explanations(
    predictions: Mapping[str, Any],
    *,
    period: float,
    lightcurve: Lightcurve | None = None,
    phase_bins: int = 12,
    minimum_shared_phase_bins: int = 4,
    shape_similarity_threshold: float = 0.8,
    shape_difference_threshold: float = 0.4,
    large_flux_difference_dex: float = 0.5,
    comparable_wavelength_fraction: float = 0.15,
) -> dict[str, Any]:
    """Explain unequal channel fits without treating pivot wavelength as a filter.

    The result separates sampling, phase coverage, predictive adequacy,
    normalized phase-shape agreement, flux-level disagreement, and fitted
    kernel support.  A large flux mismatch between channels with comparable
    scalar wavelengths is deliberately labelled as passband/SED *or*
    calibration incompatibility: the current pivot-wavelength model does not
    contain full filter-response profiles and cannot distinguish those causes.
    """
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError('period must be finite and strictly positive.')
    if isinstance(phase_bins, bool) or not isinstance(
        phase_bins, (int, np.integer)
    ):
        raise TypeError('phase_bins must be an integer.')
    if phase_bins < 4:
        raise ValueError('phase_bins must be at least 4.')
    if (
        isinstance(minimum_shared_phase_bins, bool)
        or not isinstance(minimum_shared_phase_bins, (int, np.integer))
        or minimum_shared_phase_bins < 2
        or minimum_shared_phase_bins > phase_bins
    ):
        raise ValueError(
            'minimum_shared_phase_bins must be between 2 and phase_bins.'
        )
    if not (
        0.0 <= shape_difference_threshold < shape_similarity_threshold <= 1.0
    ):
        raise ValueError(
            'Require 0 <= shape_difference_threshold '
            '< shape_similarity_threshold <= 1.'
        )
    if (
        not np.isfinite(large_flux_difference_dex)
        or large_flux_difference_dex <= 0.0
    ):
        raise ValueError(
            'large_flux_difference_dex must be finite and positive.'
        )
    if (
        not np.isfinite(comparable_wavelength_fraction)
        or comparable_wavelength_fraction <= 0.0
    ):
        raise ValueError(
            'comparable_wavelength_fraction must be finite and positive.'
        )

    keys = (
        'time',
        'physical_wavelength',
        'observational_channel',
        'observed',
        'predictive_mean',
        'predictive_standard_deviation',
        'residual',
        'standardized_residual',
    )
    arrays = {
        key: _numpy_1d(
            predictions[key],
            name=f'prediction {key}',
            dtype=str if key == 'observational_channel' else float,
        )
        for key in keys
    }
    row_count = arrays['time'].size
    measurement_sd_value = predictions.get(
        'measurement_standard_deviation'
    )
    measurement_sd = (
        np.full(row_count, np.nan, dtype=float)
        if measurement_sd_value is None
        else _numpy_1d(
            measurement_sd_value,
            name='measurement standard deviation',
        )
    )
    if (
        row_count == 0
        or any(array.size != row_count for array in arrays.values())
        or measurement_sd.size != row_count
    ):
        raise ValueError('Prediction arrays must be non-empty and row aligned.')

    time = arrays['time']
    wavelength = arrays['physical_wavelength']
    channel = arrays['observational_channel']
    observed = arrays['observed']
    predictive_sd = arrays['predictive_standard_deviation']
    residual = arrays['residual']
    standardized = arrays['standardized_residual']
    reference_time = float(np.min(time))
    phase = np.mod(time - reference_time, period) / period
    phase_bin = np.minimum((phase * phase_bins).astype(int), phase_bins - 1)
    phase_bin_centers = (np.arange(phase_bins, dtype=float) + 0.5) / phase_bins
    ordered_channels = list(dict.fromkeys(channel.tolist()))

    channel_rows: list[dict[str, Any]] = []
    channel_profiles: dict[str, dict[str, Any]] = {}
    for label in ordered_channels:
        mask = channel == label
        channel_time = time[mask]
        channel_wavelengths = np.unique(wavelength[mask])
        if channel_wavelengths.size != 1:
            raise ValueError(
                'Every GP-training observational channel must map to one '
                'physical wavelength.'
            )
        channel_observed = observed[mask]
        channel_predictive_sd = predictive_sd[mask]
        channel_measurement_sd = measurement_sd[mask]
        channel_residual = residual[mask]
        channel_standardized = standardized[mask]
        unique_times = np.unique(channel_time)
        positive_spacing = np.diff(unique_times)
        positive_spacing = positive_spacing[positive_spacing > 0.0]
        baseline = float(np.max(channel_time) - np.min(channel_time))
        largest_gap = (
            float(np.max(positive_spacing)) if positive_spacing.size else None
        )
        max_gap_fraction = (
            largest_gap / baseline
            if largest_gap is not None and baseline > 0.0
            else None
        )
        occupied_bins = np.unique(phase_bin[mask])
        binned_observed: dict[int, float] = {}
        for bin_index in occupied_bins:
            bin_values = channel_observed[phase_bin[mask] == bin_index]
            finite_bin_values = bin_values[np.isfinite(bin_values)]
            if finite_bin_values.size:
                binned_observed[int(bin_index)] = float(
                    np.median(finite_bin_values)
                )
        profile_bins = np.asarray(sorted(binned_observed), dtype=int)
        profile_flux = np.asarray(
            [binned_observed[index] for index in profile_bins],
            dtype=float,
        )
        profile_scale = _robust_scale(profile_flux)
        if profile_scale is None:
            normalized_profile = np.full(profile_flux.shape, np.nan)
        else:
            normalized_profile = (
                profile_flux - float(np.median(profile_flux))
            ) / profile_scale
        profile_map = {
            int(index): float(value)
            for index, value in zip(
                profile_bins,
                normalized_profile,
                strict=True,
            )
            if np.isfinite(value)
        }
        channel_profiles[label] = {
            'bins': set(profile_map),
            'normalized': profile_map,
            'time_min': float(np.min(channel_time)),
            'time_max': float(np.max(channel_time)),
            'time_median': float(np.median(channel_time)),
            'baseline': baseline,
            'times': channel_time,
            'phases': phase[mask],
            'wavelength': float(channel_wavelengths[0]),
        }

        finite_residual = channel_residual[np.isfinite(channel_residual)]
        finite_standardized = channel_standardized[
            np.isfinite(channel_standardized)
        ]
        usable_coverage = (
            np.isfinite(channel_residual)
            & np.isfinite(channel_predictive_sd)
            & (channel_predictive_sd > 0.0)
        )
        coverage_95 = (
            float(
                np.mean(
                    np.abs(channel_residual[usable_coverage])
                    <= 1.96 * channel_predictive_sd[usable_coverage]
                )
            )
            if np.any(usable_coverage)
            else None
        )
        observed_scale = _robust_scale(channel_observed)
        finite_measurement_sd = channel_measurement_sd[
            np.isfinite(channel_measurement_sd)
        ]
        median_measurement_sd = (
            float(np.median(finite_measurement_sd))
            if finite_measurement_sd.size
            else None
        )
        median_predictive_sd = (
            float(np.median(channel_predictive_sd[np.isfinite(channel_predictive_sd)]))
            if np.any(np.isfinite(channel_predictive_sd))
            else None
        )
        bias = (
            float(np.mean(finite_residual)) if finite_residual.size else None
        )
        rmse = (
            float(np.sqrt(np.mean(np.square(finite_residual))))
            if finite_residual.size
            else None
        )
        standardized_rms = (
            float(np.sqrt(np.mean(np.square(finite_standardized))))
            if finite_standardized.size
            else None
        )
        uncertainty_scale_ratio = (
            median_predictive_sd / observed_scale
            if median_predictive_sd is not None
            and observed_scale is not None
            and observed_scale > 0.0
            else None
        )

        issues: list[str] = []
        phase_coverage = len(occupied_bins) / phase_bins
        if phase_coverage < 0.5:
            issues.append('limited_phase_bin_coverage')
        if standardized_rms is not None and standardized_rms > 3.0:
            issues.append('severe_standardized_residual_mismatch')
        elif standardized_rms is not None and standardized_rms > 2.0:
            issues.append('standardized_residual_mismatch')
        if coverage_95 is not None and coverage_95 < 0.8:
            issues.append('predictive_interval_undercoverage')
        if (
            uncertainty_scale_ratio is not None
            and uncertainty_scale_ratio > 1.0
            and standardized_rms is not None
            and standardized_rms < 0.7
        ):
            issues.append('predictive_uncertainty_large_relative_to_variability')
        if (
            bias is not None
            and median_predictive_sd is not None
            and abs(bias) > max(median_predictive_sd, 0.25 * (observed_scale or 0.0))
        ):
            issues.append('predictive_mean_bias')
        fit_interpretation = (
            'adequate_under_current_training_diagnostics'
            if not issues
            else ';'.join(issues)
        )

        channel_rows.append(
            {
                'observational_channel': label,
                'physical_wavelength': float(channel_wavelengths[0]),
                'n_points': int(np.count_nonzero(mask)),
                'time_start': float(np.min(channel_time)),
                'time_end': float(np.max(channel_time)),
                'time_baseline': baseline,
                'median_positive_cadence': (
                    float(np.median(positive_spacing))
                    if positive_spacing.size
                    else None
                ),
                'largest_positive_gap': largest_gap,
                'maximum_gap_fraction': max_gap_fraction,
                'occupied_phase_bins': len(occupied_bins),
                'phase_bin_count': int(phase_bins),
                'phase_bin_coverage_fraction': float(phase_coverage),
                'circular_phase_coverage_fraction': (
                    _circular_phase_coverage(phase[mask])
                ),
                'median_observed_flux': float(np.median(channel_observed)),
                'observed_robust_scale': observed_scale,
                'predictive_mean_bias': bias,
                'rmse': rmse,
                'standardized_residual_rms': standardized_rms,
                'median_measurement_standard_deviation': (
                    median_measurement_sd
                ),
                'median_predictive_standard_deviation': median_predictive_sd,
                'predictive_uncertainty_to_observed_scale': (
                    uncertainty_scale_ratio
                ),
                'empirical_95_percent_coverage': coverage_95,
                'fit_interpretation': fit_interpretation,
            }
        )

    model = getattr(lightcurve, 'model', None) if lightcurve is not None else None
    kernels = getattr(getattr(model, 'covar_module', None), 'kernels', None)
    kernel_available = kernels is not None and len(kernels) >= 2
    kernel_reason = None
    transformed_by_channel: dict[str, torch.Tensor] = {}
    if kernel_available:
        try:
            dtype = lightcurve.xdata.dtype
            device = lightcurve.xdata.device
            for label, profile in channel_profiles.items():
                raw = torch.as_tensor(
                    np.column_stack(
                        [
                            profile['times'],
                            np.full(
                                profile['times'].shape,
                                profile['wavelength'],
                            ),
                        ]
                    ),
                    dtype=dtype,
                    device=device,
                )
                transformed_by_channel[label] = lightcurve.transform_x(raw)
        except Exception as error:  # diagnostic fallback with provenance
            kernel_available = False
            kernel_reason = f'{type(error).__name__}: {error}'
    elif lightcurve is None:
        kernel_reason = 'lightcurve_not_supplied'
    else:
        kernel_reason = 'separable_fitted_kernel_components_not_available'

    pair_rows: list[dict[str, Any]] = []
    for label_a, label_b in combinations(ordered_channels, 2):
        profile_a = channel_profiles[label_a]
        profile_b = channel_profiles[label_b]
        bins_a = profile_a['bins']
        bins_b = profile_b['bins']
        shared_bins = sorted(bins_a & bins_b)
        union_bins = bins_a | bins_b
        shape_correlation = None
        if len(shared_bins) >= minimum_shared_phase_bins:
            values_a = np.asarray(
                [profile_a['normalized'][index] for index in shared_bins],
                dtype=float,
            )
            values_b = np.asarray(
                [profile_b['normalized'][index] for index in shared_bins],
                dtype=float,
            )
            if (
                _robust_scale(values_a) is not None
                and _robust_scale(values_b) is not None
            ):
                shape_correlation = float(np.corrcoef(values_a, values_b)[0, 1])
                if not np.isfinite(shape_correlation):
                    shape_correlation = None

        row_a = next(
            row for row in channel_rows
            if row['observational_channel'] == label_a
        )
        row_b = next(
            row for row in channel_rows
            if row['observational_channel'] == label_b
        )
        median_a = row_a['median_observed_flux']
        median_b = row_b['median_observed_flux']
        flux_ratio = (
            median_a / median_b
            if median_a > 0.0 and median_b > 0.0
            else None
        )
        flux_difference_dex = (
            float(np.log10(flux_ratio))
            if flux_ratio is not None and flux_ratio > 0.0
            else None
        )
        scale_a = row_a['observed_robust_scale']
        scale_b = row_b['observed_robust_scale']
        amplitude_ratio = (
            scale_a / scale_b
            if scale_a is not None and scale_b is not None and scale_b > 0.0
            else None
        )
        overlap = max(
            0.0,
            min(profile_a['time_max'], profile_b['time_max'])
            - max(profile_a['time_min'], profile_b['time_min']),
        )
        shorter_baseline = min(profile_a['baseline'], profile_b['baseline'])
        temporal_overlap_fraction = (
            overlap / shorter_baseline if shorter_baseline > 0.0 else None
        )

        wavelength_correlation = None
        total_phase_aligned_kernel_support = None
        temporal_median_absolute_correlation = None
        temporal_maximum_absolute_correlation = None
        phase_aligned_temporal_median_absolute_correlation = None
        pair_kernel_reason = kernel_reason
        if kernel_available:
            try:
                transformed_a = transformed_by_channel[label_a]
                transformed_b = transformed_by_channel[label_b]
                temporal_correlation = _normalized_kernel_correlation(
                    kernels[0],
                    transformed_a,
                    transformed_b,
                )
                finite_absolute = np.abs(
                    temporal_correlation[np.isfinite(temporal_correlation)]
                )
                if finite_absolute.size:
                    temporal_median_absolute_correlation = float(
                        np.median(finite_absolute)
                    )
                    temporal_maximum_absolute_correlation = float(
                        np.max(finite_absolute)
                    )
                phase_distance = np.abs(
                    profile_a['phases'][:, None]
                    - profile_b['phases'][None, :]
                )
                phase_distance = np.minimum(phase_distance, 1.0 - phase_distance)
                aligned = (
                    phase_distance <= (0.5 / phase_bins)
                ) & np.isfinite(temporal_correlation)
                if np.any(aligned):
                    phase_aligned_temporal_median_absolute_correlation = float(
                        np.median(np.abs(temporal_correlation[aligned]))
                    )

                reference = float(np.median(time))
                raw_wavelength_pair = torch.as_tensor(
                    [
                        [reference, profile_a['wavelength']],
                        [reference, profile_b['wavelength']],
                    ],
                    dtype=lightcurve.xdata.dtype,
                    device=lightcurve.xdata.device,
                )
                transformed_wavelength_pair = lightcurve.transform_x(
                    raw_wavelength_pair
                )
                wavelength_matrix = _normalized_kernel_correlation(
                    kernels[1],
                    transformed_wavelength_pair[:1],
                    transformed_wavelength_pair[1:],
                )
                if np.isfinite(wavelength_matrix[0, 0]):
                    wavelength_correlation = float(wavelength_matrix[0, 0])
                if (
                    wavelength_correlation is not None
                    and phase_aligned_temporal_median_absolute_correlation
                    is not None
                ):
                    total_phase_aligned_kernel_support = abs(
                        wavelength_correlation
                    ) * phase_aligned_temporal_median_absolute_correlation
                pair_kernel_reason = None
            except Exception as error:  # preserve remaining diagnostics
                pair_kernel_reason = f'{type(error).__name__}: {error}'

        wavelength_separation = abs(
            profile_a['wavelength'] - profile_b['wavelength']
        )
        relative_wavelength_separation = wavelength_separation / min(
            profile_a['wavelength'], profile_b['wavelength']
        )
        comparable_scalar_wavelengths = (
            relative_wavelength_separation
            <= comparable_wavelength_fraction
        )
        absolute_dex = (
            abs(flux_difference_dex)
            if flux_difference_dex is not None
            else None
        )
        if len(shared_bins) < minimum_shared_phase_bins:
            interpretation = 'insufficient_shared_phase_support'
        elif (
            shape_correlation is not None
            and shape_correlation >= shape_similarity_threshold
            and absolute_dex is not None
            and absolute_dex >= large_flux_difference_dex
            and comparable_scalar_wavelengths
        ):
            interpretation = (
                'passband_sed_or_calibration_incompatibility_candidate'
            )
        elif (
            shape_correlation is not None
            and shape_correlation <= shape_difference_threshold
        ):
            interpretation = (
                'possible_chromatic_shape_or_model_family_mismatch'
            )
        elif (
            len(union_bins) > 0
            and len(shared_bins) / len(union_bins) < 0.25
        ):
            interpretation = 'limited_phase_support'
        else:
            interpretation = 'no_decisive_pairwise_failure_identified'

        pair_rows.append(
            {
                'observational_channel_a': label_a,
                'observational_channel_b': label_b,
                'physical_wavelength_a': profile_a['wavelength'],
                'physical_wavelength_b': profile_b['wavelength'],
                'wavelength_separation': wavelength_separation,
                'relative_wavelength_separation': (
                    relative_wavelength_separation
                ),
                'comparable_scalar_wavelengths': (
                    comparable_scalar_wavelengths
                ),
                'median_flux_ratio_a_over_b': flux_ratio,
                'median_flux_difference_dex_a_minus_b': flux_difference_dex,
                'absolute_median_flux_difference_dex': absolute_dex,
                'robust_amplitude_ratio_a_over_b': amplitude_ratio,
                'shared_phase_bins': len(shared_bins),
                'phase_bin_union': len(union_bins),
                'phase_overlap_fraction': (
                    len(shared_bins) / len(union_bins) if union_bins else None
                ),
                'normalized_phase_shape_correlation': shape_correlation,
                'temporal_overlap_fraction': temporal_overlap_fraction,
                'median_epoch_separation_in_periods': abs(
                    profile_a['time_median'] - profile_b['time_median']
                ) / period,
                'fitted_wavelength_kernel_correlation': wavelength_correlation,
                'fitted_temporal_kernel_median_absolute_correlation': (
                    temporal_median_absolute_correlation
                ),
                'fitted_temporal_kernel_maximum_absolute_correlation': (
                    temporal_maximum_absolute_correlation
                ),
                'phase_aligned_temporal_kernel_median_absolute_correlation': (
                    phase_aligned_temporal_median_absolute_correlation
                ),
                'phase_aligned_total_kernel_support': (
                    total_phase_aligned_kernel_support
                ),
                'kernel_support_available': pair_kernel_reason is None,
                'kernel_support_unavailable_reason': pair_kernel_reason,
                'interpretation': interpretation,
            }
        )

    for channel_row in channel_rows:
        label = channel_row['observational_channel']
        candidates = [
            row for row in pair_rows
            if label in {
                row['observational_channel_a'],
                row['observational_channel_b'],
            }
        ]
        nearest = min(
            candidates,
            key=lambda row: row['wavelength_separation'],
            default=None,
        )
        if nearest is None:
            channel_row.update(
                {
                    'nearest_observational_channel': None,
                    'nearest_wavelength_separation': None,
                    'nearest_pair_interpretation': None,
                }
            )
        else:
            neighbour = (
                nearest['observational_channel_b']
                if nearest['observational_channel_a'] == label
                else nearest['observational_channel_a']
            )
            channel_row.update(
                {
                    'nearest_observational_channel': neighbour,
                    'nearest_wavelength_separation': nearest[
                        'wavelength_separation'
                    ],
                    'nearest_pair_interpretation': nearest['interpretation'],
                }
            )

        fit_is_adequate = (
            channel_row['fit_interpretation']
            == 'adequate_under_current_training_diagnostics'
        )
        nearest_interpretation = channel_row[
            'nearest_pair_interpretation'
        ]
        if fit_is_adequate:
            final_interpretation = (
                'fit_adequate_under_training_point_diagnostics'
            )
        elif nearest_interpretation == (
            'passband_sed_or_calibration_incompatibility_candidate'
        ):
            final_interpretation = (
                'fit_quality_issue_with_nearby_passband_sed_or_'
                'calibration_ambiguity'
            )
        elif nearest_interpretation in {
            'limited_phase_support',
            'insufficient_shared_phase_support',
        }:
            final_interpretation = (
                'fit_quality_issue_with_limited_pairwise_phase_support'
            )
        elif nearest_interpretation == (
            'possible_chromatic_shape_or_model_family_mismatch'
        ):
            final_interpretation = (
                'fit_quality_issue_with_possible_chromatic_shape_or_'
                'model_family_mismatch'
            )
        else:
            final_interpretation = (
                'fit_quality_issue_without_decisive_pairwise_cause'
            )
        channel_row['final_diagnostic_interpretation'] = (
            final_interpretation
        )

    channel_medians = np.asarray(
        [row['median_observed_flux'] for row in channel_rows],
        dtype=float,
    )
    positive_medians = channel_medians[channel_medians > 0.0]
    flux_dynamic_range_dex = (
        float(
            np.log10(np.max(positive_medians))
            - np.log10(np.min(positive_medians))
        )
        if positive_medians.size
        else None
    )
    wavelengths = np.asarray(
        [row['physical_wavelength'] for row in channel_rows],
        dtype=float,
    )
    review_pairs = [
        row for row in pair_rows
        if row['interpretation']
        != 'no_decisive_pairwise_failure_identified'
    ]
    review_pairs.sort(
        key=lambda row: (
            row['interpretation']
            != 'passband_sed_or_calibration_incompatibility_candidate',
            -(
                row['absolute_median_flux_difference_dex']
                if row['absolute_median_flux_difference_dex'] is not None
                else -1.0
            ),
            row['wavelength_separation'],
        )
    )
    warnings = []
    passband_candidates = [
        row for row in review_pairs
        if row['interpretation']
        == 'passband_sed_or_calibration_incompatibility_candidate'
    ]
    if passband_candidates:
        pair_text = ', '.join(
            f"{row['observational_channel_a']} vs "
            f"{row['observational_channel_b']}"
            for row in passband_candidates
        )
        warnings.append(
            'Similar normalized phase shapes coexist with large flux-level '
            f'differences for {pair_text}. The current scalar-wavelength '
            'model cannot distinguish full passband/SED effects from unit, '
            'zero-point, or calibration incompatibility.'
        )

    profile_rows = []
    for label in ordered_channels:
        profile = channel_profiles[label]
        for bin_index in sorted(profile['normalized']):
            profile_rows.append(
                {
                    'observational_channel': label,
                    'physical_wavelength': profile['wavelength'],
                    'phase_bin': int(bin_index),
                    'phase_bin_center': float(phase_bin_centers[bin_index]),
                    'normalized_median_observed_flux': profile['normalized'][
                        bin_index
                    ],
                }
            )

    return single_source_json_safe(
        {
            'diagnostic_only': True,
            'automatic_model_rejection': False,
            'prediction_scope': 'training_points_not_held_out',
            'period': float(period),
            'reference_time': reference_time,
            'phase_bin_count': int(phase_bins),
            'minimum_shared_phase_bins': int(minimum_shared_phase_bins),
            'shape_similarity_threshold': float(
                shape_similarity_threshold
            ),
            'shape_difference_threshold': float(shape_difference_threshold),
            'large_flux_difference_dex': float(large_flux_difference_dex),
            'comparable_wavelength_fraction': float(
                comparable_wavelength_fraction
            ),
            'n_observational_channels': len(channel_rows),
            'n_channel_pairs': len(pair_rows),
            'n_pairs_requiring_interpretation': len(review_pairs),
            'physical_wavelength_minimum': float(np.min(wavelengths)),
            'physical_wavelength_maximum': float(np.max(wavelengths)),
            'retained_wavelength_range_is_approximately_optical': bool(
                np.min(wavelengths) >= 0.3 and np.max(wavelengths) <= 1.0
            ),
            'median_flux_dynamic_range_dex': flux_dynamic_range_dex,
            'temporal_kernel_class': (
                type(kernels[0]).__name__ if kernel_available else None
            ),
            'wavelength_kernel_class': (
                type(kernels[1]).__name__ if kernel_available else None
            ),
            'kernel_support_semantics': (
                'Direct normalized covariance from the fitted kernel. '
                'For spectral-mixture kernels this is not a single temporal '
                'coherence scale.'
            ),
            'pivot_wavelength_limitation': (
                'Observational channels are represented by scalar physical '
                'wavelengths. Full response profiles and source-SED '
                'integration are absent, so passband/SED effects cannot be '
                'separated from calibration or unit incompatibility.'
            ),
            'observational_channel_rows': channel_rows,
            'pair_rows': pair_rows,
            'pairs_requiring_interpretation': review_pairs,
            'normalized_phase_profile_rows': profile_rows,
            'warning_messages': warnings,
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

def summarize_single_source_constraint_diagnostics(
    constraint_rows: Sequence[Mapping[str, Any]],
    *,
    parameter_workflow_report: Mapping[str, Any] | None = None,
    near_bound_fraction: float = 0.1,
    at_bound_fraction: float = 0.01,
) -> dict[str, Any]:
    (
        "Interpret live parameter constraints without overclaiming "
        "statistical identification."
    )
    if (
        not np.isfinite(at_bound_fraction)
        or not np.isfinite(near_bound_fraction)
        or at_bound_fraction < 0.0
        or near_bound_fraction <= at_bound_fraction
        or near_bound_fraction >= 0.5
    ):
        raise ValueError(
            "Require finite thresholds with "
            "0 <= at_bound_fraction < near_bound_fraction < 0.5."
        )
    if not isinstance(constraint_rows, Sequence) or isinstance(
        constraint_rows,
        (str, bytes),
    ):
        raise TypeError("constraint_rows must be a sequence of mappings.")

    report = parameter_workflow_report or {}
    applied_entries = report.get("applied", [])
    if not isinstance(applied_entries, Sequence) or isinstance(
        applied_entries,
        (str, bytes),
    ):
        applied_entries = []

    def normalize_parameter_name(value: Any) -> str:
        return ".".join(
            component.removeprefix("raw_")
            for component in str(value).split(".")
        )

    workflow_entries = {}
    for entry in applied_entries:
        if not isinstance(entry, Mapping):
            continue
        parameter = entry.get("parameter")
        if parameter is None:
            continue
        workflow_entries[
            normalize_parameter_name(parameter)
        ] = entry

    def first_nonempty(mapping: Mapping[str, Any], keys) -> Any:
        for key in keys:
            value = mapping.get(key)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, (list, tuple, set, dict)) and not value:
                continue
            return value
        return None

    def provenance_coordinate(entry: Mapping[str, Any]) -> Any:
        direct = first_nonempty(
            entry,
            (
                "coordinate_space",
                "model_coordinate_space",
                "units",
            ),
        )
        if direct is not None:
            return direct
        for key in (
            "wavelength_estimate_provenance",
            "wavelength_mean_estimate_provenance",
            "spectral_mixture_ard_provenance",
        ):
            provenance = entry.get(key)
            if not isinstance(provenance, Mapping):
                continue
            value = first_nonempty(
                provenance,
                (
                    "model_coordinate_space",
                    "coordinate_space",
                    "transformed_coordinate_space",
                    "units",
                ),
            )
            if value is not None:
                return value
        return None

    def parameter_role(parameter: str, applies_to: str) -> str:
        lowered = parameter.lower()
        if "lengthscale" in lowered:
            return "wavelength covariance lengthscale"
        if "mean_module" in lowered and "bias" in lowered:
            return "wavelength-dependent mean bias"
        if "mean_module" in lowered:
            return "wavelength-dependent mean coefficient"
        if "mixture_means" in lowered:
            return "spectral-mixture temporal frequency"
        if "mixture_scales" in lowered:
            return "spectral-mixture temporal width"
        if "mixture_weights" in lowered:
            return "spectral-mixture covariance weight"
        if "noise" in lowered:
            return "additional likelihood-noise parameter"
        if applies_to == "covariance":
            return "covariance parameter"
        if applies_to == "mean":
            return "mean-function parameter"
        if applies_to == "likelihood":
            return "likelihood parameter"
        return "model parameter"

    interpreted_rows = []
    for constraint_row in constraint_rows:
        if not isinstance(constraint_row, Mapping):
            raise TypeError(
                "Every constraint diagnostic row must be a mapping."
            )
        parameter = str(
            constraint_row.get("parameter", "")
        ).strip()
        if not parameter:
            raise ValueError(
                "Every constraint diagnostic row requires a parameter."
            )
        normalized_parameter = normalize_parameter_name(parameter)
        applies_to = str(
            constraint_row.get("applies_to", "model")
        ).strip() or "model"
        workflow_entry = workflow_entries.get(
            normalized_parameter,
            {},
        )

        constraint_registered = bool(
            constraint_row.get("constraint_registered")
        )
        value_initialized = bool(
            constraint_row.get("value_initialized")
        )
        initial_inside = bool(
            constraint_row.get("initial_inside_constraint")
        )
        fitted_inside = bool(
            constraint_row.get("fitted_inside_constraint")
        )

        fractional_position_value = constraint_row.get(
            "fractional_position_within_bounds"
        )
        try:
            fractional_position = float(
                fractional_position_value
            )
        except (TypeError, ValueError):
            fractional_position = None
        if (
            fractional_position is not None
            and not np.isfinite(fractional_position)
        ):
            fractional_position = None

        if not constraint_registered:
            technical_status = "constraint_not_registered"
        elif not value_initialized:
            technical_status = "value_not_initialized"
        elif not initial_inside:
            technical_status = "initialization_outside_constraint"
        elif not fitted_inside:
            technical_status = "fitted_value_outside_constraint"
        else:
            technical_status = "constraint_respected"

        nearest_fractional_bound = None
        boundary_side = None
        if (
            technical_status == "constraint_respected"
            and fractional_position is not None
        ):
            nearest_fractional_bound = min(
                fractional_position,
                1.0 - fractional_position,
            )
            boundary_side = (
                "lower"
                if fractional_position <= 0.5
                else "upper"
            )
            if nearest_fractional_bound <= at_bound_fraction:
                boundary_status = "at_bound"
            elif nearest_fractional_bound <= near_bound_fraction:
                boundary_status = "near_bound"
            else:
                boundary_status = "interior"
        elif technical_status == "constraint_respected":
            boundary_status = "position_unavailable"
        else:
            boundary_status = "technical_check_failed"

        technically_satisfactory = (
            technical_status == "constraint_respected"
            and boundary_status == "interior"
        )
        review_required = not technically_satisfactory

        value_origin = (
            workflow_entry.get("value_reason")
            or constraint_row.get("value_reason")
            or "recorded model initialization"
        )
        interval_origin = (
            workflow_entry.get("constraint_reason")
            or constraint_row.get("constraint_reason")
            or "registered live model constraint"
        )
        coordinate_system = (
            provenance_coordinate(workflow_entry)
            or first_nonempty(
                constraint_row,
                (
                    "model_coordinate_space",
                    "coordinate_space",
                    "units",
                ),
            )
            or (
                "model wavelength coordinate"
                if applies_to in {"covariance", "mean"}
                else "native model parameter coordinate"
            )
        )

        if technically_satisfactory:
            interpretation = (
                "The live constraint was registered, initialization and the "
                "fitted value are inside it, and the fitted value is not near "
                "either boundary. This is technically satisfactory, but it "
                "does not establish statistical identification or scientific "
                "adequacy."
            )
        elif (
            technical_status == "constraint_respected"
            and boundary_status == "near_bound"
        ):
            interpretation = (
                f"The live constraint was respected, but the fitted value is "
                f"near the {boundary_side} boundary. Treat the parameter as "
                "requiring sensitivity checks rather than as well identified."
            )
        elif (
            technical_status == "constraint_respected"
            and boundary_status == "at_bound"
        ):
            interpretation = (
                f"The fitted value is effectively at the {boundary_side} "
                "boundary of its live interval. The result is technically "
                "valid but boundary-limited and requires review."
            )
        elif (
            technical_status == "constraint_respected"
            and boundary_status == "position_unavailable"
        ):
            interpretation = (
                "The live constraint was respected, but no finite fractional "
                "position is available. Boundary proximity and parameter "
                "identification remain undetermined."
            )
        else:
            interpretation = (
                "The technical constraint check failed with status "
                f"{technical_status!r}; do not interpret this fitted parameter "
                "scientifically until the registration or value issue is "
                "resolved."
            )

        interpreted_rows.append(
            {
                "parameter": parameter,
                "applies_to": applies_to,
                "role": parameter_role(parameter, applies_to),
                "coordinate_system": coordinate_system,
                "value_origin": value_origin,
                "interval_origin": interval_origin,
                "lower_bound": constraint_row.get("lower_bound"),
                "initial_value": constraint_row.get("initial_value"),
                "fitted_value": constraint_row.get("fitted_value"),
                "upper_bound": constraint_row.get("upper_bound"),
                "fractional_position_within_bounds": (
                    fractional_position
                ),
                "nearest_fractional_bound": (
                    nearest_fractional_bound
                ),
                "technical_status": technical_status,
                "boundary_status": boundary_status,
                "technically_satisfactory": technically_satisfactory,
                "review_required": review_required,
                "identification_status": (
                    "not_established_by_constraint_diagnostics"
                ),
                "interpretation": interpretation,
            }
        )

    satisfactory_parameters = list(
        dict.fromkeys(
            row["parameter"]
            for row in interpreted_rows
            if row["technically_satisfactory"]
        )
    )
    review_parameters = list(
        dict.fromkeys(
            row["parameter"]
            for row in interpreted_rows
            if row["review_required"]
        )
    )
    near_bound_parameters = list(
        dict.fromkeys(
            row["parameter"]
            for row in interpreted_rows
            if row["boundary_status"] == "near_bound"
        )
    )
    at_bound_parameters = list(
        dict.fromkeys(
            row["parameter"]
            for row in interpreted_rows
            if row["boundary_status"] == "at_bound"
        )
    )

    summary = {
        "status": (
            "technical_constraints_satisfactory"
            if not review_parameters
            else "technical_constraint_review_required"
        ),
        "n_rows": len(interpreted_rows),
        "n_technically_satisfactory": sum(
            row["technically_satisfactory"]
            for row in interpreted_rows
        ),
        "n_requiring_review": sum(
            row["review_required"]
            for row in interpreted_rows
        ),
        "technically_satisfactory_parameters": satisfactory_parameters,
        "parameters_requiring_review": review_parameters,
        "near_bound_parameters": near_bound_parameters,
        "at_bound_parameters": at_bound_parameters,
        "identification_established": False,
        "interpretation": (
            "A technically satisfactory row confirms registration, valid "
            "initialization, a fitted value inside the live interval, and an "
            "interior boundary position. It does not prove that the parameter "
            "is statistically well identified or that the GP is scientifically "
            "adequate."
        ),
    }
    return single_source_json_safe(
        {
            "rows": interpreted_rows,
            "summary": summary,
            "thresholds": {
                "near_bound_fraction": float(near_bound_fraction),
                "at_bound_fraction": float(at_bound_fraction),
            },
            "diagnostic_scope": (
                "technical constraint registration, value validity, and "
                "boundary proximity only"
            ),
        }
    )
