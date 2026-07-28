"""Reusable helpers for the real-data wavelength-constraint tutorial.

The helpers in this module keep the documentation notebook focused on the
scientific workflow while preserving deterministic, inspectable data selection.
They do not perform instrument-channel calibration and they never merge
observational channels that share one physical wavelength.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import gpytorch
import numpy as np
import torch

from .lightcurve import Lightcurve
from .wavelength_validation_real_lpv import build_representative_lpv_source_summary


_REQUIRED_COLUMNS = ("time", "flux", "flux_error", "wavelength", "band")


def deterministic_time_stratified_observational_channel_indices(
    times: Any,
    observational_channel_labels: Any,
    *,
    max_samples_per_observational_channel: int = 100,
) -> np.ndarray:
    """Return deterministic time-stratified row indices per channel.

    Channels at or below the quota are retained in full.  Larger channels are
    sorted by time and sampled at evenly spaced order statistics, including the
    first and last observation.  Returned indices are sorted in original row
    order so all row-aligned arrays can be indexed directly.
    """
    if isinstance(max_samples_per_observational_channel, bool) or not isinstance(
        max_samples_per_observational_channel, (int, np.integer)
    ):
        raise TypeError(
            "max_samples_per_observational_channel must be an integer."
        )
    quota = int(max_samples_per_observational_channel)
    if quota < 2:
        raise ValueError(
            "max_samples_per_observational_channel must be at least 2."
        )

    time_values = np.asarray(times, dtype=float)
    labels = np.asarray(observational_channel_labels, dtype=str)
    if time_values.ndim != 1 or labels.ndim != 1:
        raise ValueError("times and observational_channel_labels must be 1-D.")
    if time_values.size != labels.size:
        raise ValueError(
            "times and observational_channel_labels must have equal length."
        )
    if not np.all(np.isfinite(time_values)):
        raise ValueError("times must be finite.")
    if np.any(np.char.strip(labels) == ""):
        raise ValueError("observational_channel_labels must be non-empty.")

    selected: list[int] = []
    for label in dict.fromkeys(labels.tolist()):
        channel_indices = np.flatnonzero(labels == label)
        if channel_indices.size <= quota:
            selected.extend(channel_indices.tolist())
            continue

        time_order = channel_indices[
            np.argsort(time_values[channel_indices], kind="mergesort")
        ]
        positions = np.rint(
            np.linspace(0, time_order.size - 1, quota)
        ).astype(int)
        if np.unique(positions).size != quota:
            raise RuntimeError(
                "Time-stratified sampling did not produce the requested quota."
            )
        selected.extend(time_order[positions].tolist())

    return np.asarray(sorted(selected), dtype=int)


def load_wavelength_constraint_tutorial_lightcurve(
    source_path: str | Path,
    *,
    max_samples_per_observational_channel: int = 100,
    name: str | None = None,
) -> tuple[Lightcurve, dict[str, Any]]:
    """Load a bounded real multiwavelength CSV for the tutorial.

    The input must use the maintained representative-data columns.  Rows with
    non-finite values, non-positive flux, non-positive uncertainty, or empty
    observational-channel labels are excluded before deterministic sampling.
    """
    csv_path = Path(source_path)
    data = np.genfromtxt(
        csv_path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding=None,
    )
    column_names = tuple(data.dtype.names or ())
    missing = [column for column in _REQUIRED_COLUMNS if column not in column_names]
    if missing:
        raise ValueError(
            "Tutorial CSV is missing required columns: " + ", ".join(missing)
        )

    times = np.asarray(data["time"], dtype=float)
    flux = np.asarray(data["flux"], dtype=float)
    flux_error = np.asarray(data["flux_error"], dtype=float)
    wavelength = np.asarray(data["wavelength"], dtype=float)
    channels = np.asarray(data["band"], dtype=str)

    valid = (
        np.isfinite(times)
        & np.isfinite(flux)
        & np.isfinite(flux_error)
        & np.isfinite(wavelength)
        & (flux > 0.0)
        & (flux_error > 0.0)
        & (np.char.strip(channels) != "")
    )
    if not np.any(valid):
        raise ValueError("No valid positive-flux rows remain in the tutorial CSV.")

    eligible_times = times[valid]
    eligible_flux = flux[valid]
    eligible_flux_error = flux_error[valid]
    eligible_wavelength = wavelength[valid]
    eligible_channels = channels[valid]

    retained_local = deterministic_time_stratified_observational_channel_indices(
        eligible_times,
        eligible_channels,
        max_samples_per_observational_channel=(
            max_samples_per_observational_channel
        ),
    )

    retained_times = eligible_times[retained_local]
    retained_flux = eligible_flux[retained_local]
    retained_flux_error = eligible_flux_error[retained_local]
    retained_wavelength = eligible_wavelength[retained_local]
    retained_channels = eligible_channels[retained_local]

    model_inputs = torch.as_tensor(
        np.column_stack((retained_times, retained_wavelength)),
        dtype=torch.get_default_dtype(),
    )
    lightcurve = Lightcurve(
        model_inputs,
        torch.as_tensor(retained_flux, dtype=torch.get_default_dtype()),
        yerr=torch.as_tensor(retained_flux_error, dtype=torch.get_default_dtype()),
        band=retained_channels,
        name=(name or csv_path.stem),
        check_sampling=False,
        max_samples=None,
        max_samples_per_band=None,
    )

    original_channel_counts = Counter(channels[valid].tolist())
    retained_channel_counts = Counter(retained_channels.tolist())
    original_wavelengths = np.unique(eligible_wavelength)
    retained_wavelengths = np.unique(retained_wavelength)
    source_summary = build_representative_lpv_source_summary(lightcurve)

    summary = {
        "source_path": str(csv_path),
        "sampling_method": (
            "deterministic time-stratified observational-channel sampling"
        ),
        "max_samples_per_observational_channel": int(
            max_samples_per_observational_channel
        ),
        "n_rows_original": int(times.size),
        "n_rows_eligible": int(np.count_nonzero(valid)),
        "n_rows_retained": int(retained_local.size),
        "n_rows_excluded_by_validity_policy": int(np.count_nonzero(~valid)),
        "strictly_positive_flux_and_uncertainty_required": True,
        "n_observational_channels_original": len(original_channel_counts),
        "n_observational_channels_retained": len(retained_channel_counts),
        "n_physical_wavelengths_original": int(original_wavelengths.size),
        "n_physical_wavelengths_retained": int(retained_wavelengths.size),
        "observational_channel_counts_original": dict(
            sorted(original_channel_counts.items())
        ),
        "observational_channel_counts_retained": dict(
            sorted(retained_channel_counts.items())
        ),
        "physical_wavelengths_original": original_wavelengths.tolist(),
        "physical_wavelengths_retained": retained_wavelengths.tolist(),
        "observational_channels_by_shared_wavelength": source_summary[
            "observational_channels_by_shared_wavelength"
        ],
        "instrument_calibration_status": source_summary[
            "instrument_calibration_status"
        ],
        "instrument_calibration_tbd": source_summary[
            "instrument_calibration_tbd"
        ],
    }

    if summary["n_observational_channels_retained"] != summary[
        "n_observational_channels_original"
    ]:
        raise AssertionError("Tutorial sampling dropped an observational channel.")
    if summary["n_physical_wavelengths_retained"] != summary[
        "n_physical_wavelengths_original"
    ]:
        raise AssertionError("Tutorial sampling dropped a physical wavelength.")

    return lightcurve, summary



def _numpy_1d(value: Any, *, name: str) -> np.ndarray:
    """Return a finite-friendly one-dimensional NumPy view."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array


def training_point_prediction_summary(lightcurve: Lightcurve) -> dict[str, Any]:
    """Evaluate predictive means, variances, and residuals at training rows.

    Values remain in the model's training-target coordinate.  The maintained
    tutorial uses no y-axis transform, so that coordinate is the input flux
    coordinate.  This function deliberately preserves row-aligned
    observational-channel identity.
    """
    if not bool(getattr(lightcurve, "is_fitted", False)):
        raise RuntimeError("A successful Lightcurve.fit() call is required.")

    model = getattr(lightcurve, "model", None)
    likelihood = getattr(lightcurve, "likelihood", None)
    train_x = getattr(lightcurve, "_xdata_transformed", None)
    train_y = getattr(lightcurve, "_ydata_transformed", None)
    raw_x = getattr(lightcurve, "_xdata_raw", None)
    if any(item is None for item in (model, likelihood, train_x, train_y, raw_x)):
        raise RuntimeError(
            "Fitted model, likelihood, and training arrays are required."
        )

    evaluator = getattr(lightcurve, "_eval", None)
    if callable(evaluator):
        evaluator()
    else:
        model.eval()
        likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictive = likelihood(model(train_x))
        mean = predictive.mean.detach().cpu().numpy().reshape(-1)
        variance = predictive.variance.detach().cpu().numpy().reshape(-1)

    observed = train_y.detach().cpu().numpy().reshape(-1)
    coordinates = raw_x.detach().cpu().numpy()
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("Training predictions require time/wavelength coordinates.")
    if not (mean.size == variance.size == observed.size == coordinates.shape[0]):
        raise RuntimeError("Training prediction arrays are not row aligned.")

    residual = observed - mean
    standard_deviation = np.sqrt(np.clip(variance, 0.0, None))
    standardized_residual = np.full_like(residual, np.nan, dtype=float)
    usable = np.isfinite(standard_deviation) & (standard_deviation > 0.0)
    standardized_residual[usable] = residual[usable] / standard_deviation[usable]

    labels = np.asarray(
        lightcurve.observational_channel_labels,
        dtype=str,
    )
    if labels.shape != residual.shape:
        raise RuntimeError("Observational-channel labels are not row aligned.")

    return {
        "time": np.asarray(coordinates[:, 0], dtype=float),
        "physical_wavelength": np.asarray(coordinates[:, 1], dtype=float),
        "observational_channel": labels,
        "observed": observed,
        "predictive_mean": mean,
        "predictive_variance": variance,
        "predictive_standard_deviation": standard_deviation,
        "residual": residual,
        "standardized_residual": standardized_residual,
        "prediction_space": "training_target_space",
    }


def summarize_observational_channel_residuals(
    predictions: dict[str, Any],
) -> list[dict[str, Any]]:
    """Summarize fitted residuals without merging observational channels."""
    times = _numpy_1d(predictions["time"], name="time")
    wavelengths = _numpy_1d(
        predictions["physical_wavelength"],
        name="physical_wavelength",
    )
    residual = _numpy_1d(predictions["residual"], name="residual")
    standardized = _numpy_1d(
        predictions["standardized_residual"],
        name="standardized_residual",
    )
    labels = np.asarray(predictions["observational_channel"], dtype=str).reshape(-1)
    row_count = times.size
    if not all(
        array.size == row_count
        for array in (wavelengths, residual, standardized, labels)
    ):
        raise ValueError("Prediction arrays must be row aligned.")

    rows: list[dict[str, Any]] = []
    for label in dict.fromkeys(labels.tolist()):
        mask = labels == label
        channel_wavelengths = np.unique(wavelengths[mask])
        finite_residual = residual[mask][np.isfinite(residual[mask])]
        finite_standardized = standardized[mask][np.isfinite(standardized[mask])]
        if finite_residual.size == 0:
            bias = rmse = mae = float("nan")
        else:
            bias = float(np.mean(finite_residual))
            rmse = float(np.sqrt(np.mean(np.square(finite_residual))))
            mae = float(np.mean(np.abs(finite_residual)))
        standardized_rms = (
            float(np.sqrt(np.mean(np.square(finite_standardized))))
            if finite_standardized.size
            else float("nan")
        )
        rows.append(
            {
                "observational_channel": str(label),
                "physical_wavelength": (
                    float(channel_wavelengths[0])
                    if channel_wavelengths.size == 1
                    else channel_wavelengths.tolist()
                ),
                "n_points": int(np.count_nonzero(mask)),
                "time_min": float(np.min(times[mask])),
                "time_max": float(np.max(times[mask])),
                "bias": bias,
                "mae": mae,
                "rmse": rmse,
                "standardized_residual_rms": standardized_rms,
            }
        )
    return rows


def _resolve_dotted_module(root: Any, dotted_path: str) -> Any:
    target = root
    for component in dotted_path.split(".") if dotted_path else ():
        if component.isdigit():
            target = target[int(component)]
        else:
            target = getattr(target, component)
    return target


def _registered_parameter_constraint(
    model: Any,
    parameter_name: str,
) -> tuple[float, float] | None:
    module_path, local_name = parameter_name.rsplit(".", 1)
    target = _resolve_dotted_module(model, module_path)
    constraint = getattr(target, f"raw_{local_name}_constraint", None)
    if constraint is None:
        return None
    lower = _numpy_1d(constraint.lower_bound, name="lower_bound")
    upper = _numpy_1d(constraint.upper_bound, name="upper_bound")
    if lower.size == 1 and upper.size > 1:
        lower = np.repeat(lower, upper.size)
    if upper.size == 1 and lower.size > 1:
        upper = np.repeat(upper, lower.size)
    if lower.size != upper.size:
        raise RuntimeError("Registered constraint bounds are not broadcastable.")
    return lower, upper


def build_wavelength_constraint_position_rows(
    lightcurve: Lightcurve,
    fit_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return initial/fitted positions for wavelength-derived constraints."""
    report = lightcurve.get_parameter_workflow_report()
    parameters = lightcurve.get_parameters(raw=False, transform=False)
    rows: list[dict[str, Any]] = []

    for entry in report.get("applied", []):
        provenance_key = None
        applies_to = None
        if "wavelength_estimate_provenance" in entry:
            provenance_key = "wavelength_estimate_provenance"
            applies_to = "covariance"
        elif "wavelength_mean_estimate_provenance" in entry:
            provenance_key = "wavelength_mean_estimate_provenance"
            applies_to = "mean"
        if provenance_key is None:
            continue

        parameter_name = str(entry["parameter"])
        if parameter_name not in parameters:
            raise RuntimeError(
                f"Fitted parameter {parameter_name!r} is unavailable."
            )
        fitted_values = _numpy_1d(
            parameters[parameter_name],
            name=f"fitted {parameter_name}",
        )
        history = fit_result.get(parameter_name)
        if not isinstance(history, list) or not history:
            raise RuntimeError(
                f"Trainer history for {parameter_name!r} is unavailable."
            )
        initial_values = _numpy_1d(
            history[0],
            name=f"initial {parameter_name}",
        )
        bounds = _registered_parameter_constraint(
            lightcurve.model,
            parameter_name,
        )
        if bounds is None:
            raise RuntimeError(
                f"No registered interval constraint for {parameter_name!r}."
            )
        lower_values, upper_values = bounds
        target_size = max(
            fitted_values.size,
            initial_values.size,
            lower_values.size,
            upper_values.size,
        )

        def broadcast(
            array: np.ndarray,
            *,
            _target_size: int = target_size,
            _parameter_name: str = parameter_name,
        ) -> np.ndarray:
            if array.size == _target_size:
                return array
            if array.size == 1:
                return np.repeat(array, _target_size)
            raise RuntimeError(
                f"Values for {_parameter_name!r} are not broadcastable."
            )

        fitted_values = broadcast(fitted_values)
        initial_values = broadcast(initial_values)
        lower_values = broadcast(lower_values)
        upper_values = broadcast(upper_values)
        provenance = entry[provenance_key]

        for component_index in range(target_size):
            lower = float(lower_values[component_index])
            upper = float(upper_values[component_index])
            initial = float(initial_values[component_index])
            fitted = float(fitted_values[component_index])
            width = upper - lower
            if not (
                np.isfinite(lower)
                and np.isfinite(upper)
                and lower < upper
            ):
                raise ValueError(
                    "Wavelength constraint bounds must be finite and ordered."
                )
            fractional_position = (fitted - lower) / width
            rows.append(
                {
                    "parameter": parameter_name,
                    "component_index": component_index,
                    "applies_to": applies_to,
                    "application_order": "constraint_then_value",
                    "constraint_action": entry.get("constraint_action"),
                    "constraint_registered": bool(entry.get("constraint_applied")),
                    "value_initialized": bool(entry.get("value_applied")),
                    "lower_bound": lower,
                    "initial_value": initial,
                    "fitted_value": fitted,
                    "upper_bound": upper,
                    "initial_inside_constraint": bool(lower <= initial <= upper),
                    "fitted_inside_constraint": bool(lower <= fitted <= upper),
                    "absolute_parameter_change": abs(fitted - initial),
                    "fractional_position_within_bounds": float(fractional_position),
                    "distance_to_lower_bound": float(fitted - lower),
                    "distance_to_upper_bound": float(upper - fitted),
                    "minimum_distance_to_bound": float(
                        min(fitted - lower, upper - fitted)
                    ),
                    "estimated_value": provenance.get("estimated_value"),
                    "estimated_constraint": provenance.get("estimated_constraint"),
                    "effective_constraint": provenance.get("effective_constraint"),
                    "diagnostics": provenance.get("diagnostics", {}),
                }
            )
    return rows


def build_tutorial_fit_summary(
    *,
    model_name: str,
    lightcurve: Lightcurve,
    fit_result: dict[str, Any],
    sampling_summary: dict[str, Any],
    constraint_rows: list[dict[str, Any]],
    predictions: dict[str, Any],
    warning_messages: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build the tutorial's compact numerical fit-success record."""
    losses = _numpy_1d(fit_result.get("loss", []), name="loss")
    if losses.size == 0:
        raise ValueError("The fit result contains no loss history.")
    residual = _numpy_1d(predictions["residual"], name="residual")
    predictive_mean = _numpy_1d(
        predictions["predictive_mean"],
        name="predictive_mean",
    )
    predictive_variance = _numpy_1d(
        predictions["predictive_variance"],
        name="predictive_variance",
    )
    training_channels = np.asarray(
        predictions["observational_channel"],
        dtype=str,
    ).reshape(-1)
    training_wavelengths = _numpy_1d(
        predictions["physical_wavelength"],
        name="physical_wavelength",
    )
    prediction_row_count = residual.size
    if not all(
        array.size == prediction_row_count
        for array in (
            predictive_mean,
            predictive_variance,
            training_channels,
            training_wavelengths,
        )
    ):
        raise ValueError("Prediction arrays must be row aligned.")
    covariance_rows = [
        row for row in constraint_rows if row["applies_to"] == "covariance"
    ]
    wavelength_row = covariance_rows[0] if covariance_rows else None

    return {
        "model_name": str(model_name),
        "resolved_model_class": lightcurve.model.__class__.__name__,
        "fit_executed": bool(getattr(lightcurve, "is_fitted", False)),
        "optimizer_status": (
            "recovered_best_finite_state"
            if fit_result.get("training_recovered_from_failure")
            else "completed"
        ),
        "n_iterations": int(losses.size),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "best_loss": float(np.min(losses)),
        "objective_improved": bool(losses[-1] < losses[0]),
        "n_observations": int(prediction_row_count),
        "n_observational_channels": int(
            np.unique(training_channels).size
        ),
        "n_physical_wavelengths": int(
            np.unique(training_wavelengths).size
        ),
        "gp_training_scope": {
            "n_observations": int(prediction_row_count),
            "n_observational_channels": int(
                np.unique(training_channels).size
            ),
            "n_physical_wavelengths": int(
                np.unique(training_wavelengths).size
            ),
            "observational_channels": list(
                dict.fromkeys(training_channels.tolist())
            ),
        },
        "consensus_input_scope": {
            "n_observations": int(sampling_summary["n_rows_retained"]),
            "n_observational_channels": int(
                sampling_summary["n_observational_channels_retained"]
            ),
            "n_physical_wavelengths": int(
                sampling_summary["n_physical_wavelengths_retained"]
            ),
        },
        "wavelength_parameter": (
            None if wavelength_row is None else wavelength_row["parameter"]
        ),
        "fitted_wavelength_parameter": (
            None if wavelength_row is None else wavelength_row["fitted_value"]
        ),
        "wavelength_lower_bound": (
            None if wavelength_row is None else wavelength_row["lower_bound"]
        ),
        "wavelength_upper_bound": (
            None if wavelength_row is None else wavelength_row["upper_bound"]
        ),
        "fractional_position_within_bounds": (
            None
            if wavelength_row is None
            else wavelength_row["fractional_position_within_bounds"]
        ),
        "minimum_distance_to_bound": (
            None
            if wavelength_row is None
            else wavelength_row["minimum_distance_to_bound"]
        ),
        "nonfinite_prediction_count": int(
            np.count_nonzero(~np.isfinite(predictive_mean))
        ),
        "nonfinite_variance_count": int(
            np.count_nonzero(~np.isfinite(predictive_variance))
        ),
        "negative_variance_count": int(
            np.count_nonzero(predictive_variance < 0.0)
        ),
        "residual_rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "warning_count": len(warning_messages),
        "warnings": list(warning_messages),
    }
