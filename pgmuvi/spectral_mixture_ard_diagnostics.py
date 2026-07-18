"""Diagnostics for fitted two-dimensional spectral-mixture ARD parameters.

The full non-separable ``2D`` model stores temporal-frequency and
wavelength-frequency entries in the last axis of ``mixture_means`` and
``mixture_scales``.  This module inspects the fitted constrained values, the
registered GPyTorch bounds, and the unconstrained raw parameters without
changing the model.

Boundary proximity is reported separately for each mixture component,
parameter, ARD dimension, and bound side.  A boundary hit is descriptive: it
means the fitted value is close to a registered bound under the requested
fractional tolerance.  It is not evidence that a physical correlation scale is
infinite, absent, or otherwise scientifically preferred.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .spectral_mixture_ard import ARD_COORDINATE_ORDER


SPECTRAL_MIXTURE_ARD_DIAGNOSTIC_SCHEMA_VERSION = (
    "pgmuvi-spectral-mixture-ard-diagnostics-v1"
)

_PARAMETER_NAMES = ("mixture_means", "mixture_scales")


def _to_numpy(value: Any) -> np.ndarray | None:
    """Return ``value`` as a floating NumPy array when possible."""
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        array = np.asarray(value, dtype=float)
    except Exception:
        return None
    if array.size == 0:
        return None
    return array


def _json_safe(value: Any) -> Any:
    """Return nested diagnostics using plain JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _iter_kernel_like_objects(obj: Any):
    """Yield kernel-like objects reachable from a light curve or model."""
    seen: set[int] = set()
    stack: list[Any] = []

    model = getattr(obj, "model", None)
    if model is not None:
        stack.extend(
            candidate
            for candidate in (
                getattr(model, "sci_kernel", None),
                getattr(model, "covar_module", None),
                model,
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
            "sci_kernel",
        ):
            child = getattr(item, attr, None)
            if child is not None:
                stack.append(child)

        kernels = getattr(item, "kernels", None)
        if kernels is not None:
            try:
                stack.extend(kernel for kernel in kernels if kernel is not None)
            except TypeError:
                pass


def _find_spectral_mixture_kernel(obj: Any) -> Any | None:
    """Return the first kernel exposing at least two SM ARD dimensions."""
    for candidate in _iter_kernel_like_objects(obj):
        if not all(hasattr(candidate, name) for name in _PARAMETER_NAMES):
            continue
        means = _component_dimension_array(
            getattr(candidate, "mixture_means", None)
        )
        scales = _component_dimension_array(
            getattr(candidate, "mixture_scales", None)
        )
        if (
            means is not None
            and scales is not None
            and means.shape[1] >= 2
            and scales.shape[1] >= 2
        ):
            return candidate
    return None


def _component_dimension_array(value: Any) -> np.ndarray | None:
    """Normalize an SM tensor to ``(component, ard_dimension)``."""
    array = _to_numpy(value)
    if array is None or array.ndim == 0:
        return None
    if array.ndim == 1:
        return array.reshape(array.shape[0], 1)
    return array.reshape(array.shape[0], int(np.prod(array.shape[1:])))


def _dimension_names(n_dimensions: int) -> list[str]:
    names = []
    for index in range(int(n_dimensions)):
        if index < len(ARD_COORDINATE_ORDER):
            names.append(str(ARD_COORDINATE_ORDER[index]))
        else:
            names.append(f"ard_dimension_{index}")
    return names


def _parameter_workflow_entry(obj: Any, parameter_name: str) -> dict[str, Any]:
    """Return ARD provenance for ``parameter_name`` when retained."""
    result = getattr(obj, "parameter_workflow_result", None)
    if not isinstance(result, dict):
        return {}

    suffix = f".{parameter_name}"
    for full_name, entry in result.items():
        if not isinstance(entry, dict):
            continue
        if full_name == parameter_name or str(full_name).endswith(suffix):
            provenance = entry.get("spectral_mixture_ard_provenance")
            if isinstance(provenance, dict):
                return provenance
    return {}


def _coordinate_factors(
    obj: Any,
    parameter_name: str,
    n_dimensions: int,
) -> tuple[np.ndarray | None, str | None]:
    """Return model/raw-input frequency factors from PR124 provenance."""
    provenance = _parameter_workflow_entry(obj, parameter_name)
    diagnostics = provenance.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return None, None

    raw = _component_dimension_array(diagnostics.get("raw_initial_value"))
    model = _component_dimension_array(diagnostics.get("model_initial_value"))
    if raw is None or model is None or raw.shape != model.shape:
        return None, None

    factors = []
    for index in range(n_dimensions):
        ratios = model[:, index] / raw[:, index]
        valid = ratios[np.isfinite(ratios) & (ratios > 0)]
        if valid.size == 0:
            return None, None
        factors.append(float(np.median(valid)))
    return np.asarray(factors, dtype=float), "parameter_workflow_provenance"


def _constraint_bounds(
    kernel: Any,
    parameter_name: str,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return registered model-coordinate bounds broadcast to ``target_shape``."""
    constraint = getattr(kernel, f"raw_{parameter_name}_constraint", None)
    if constraint is None:
        lower = np.full(target_shape, -np.inf, dtype=float)
        upper = np.full(target_shape, np.inf, dtype=float)
        return lower, upper, False

    lower = _component_dimension_array(
        getattr(constraint, "lower_bound", None)
    )
    upper = _component_dimension_array(
        getattr(constraint, "upper_bound", None)
    )
    if lower is None or upper is None:
        lower = np.full(target_shape, -np.inf, dtype=float)
        upper = np.full(target_shape, np.inf, dtype=float)
        return lower, upper, False

    try:
        lower = np.broadcast_to(lower, target_shape).astype(float, copy=False)
        upper = np.broadcast_to(upper, target_shape).astype(float, copy=False)
    except ValueError:
        lower = np.full(target_shape, -np.inf, dtype=float)
        upper = np.full(target_shape, np.inf, dtype=float)
        return lower, upper, False
    return lower, upper, True


def _boundary_status(
    value: float,
    lower: float,
    upper: float,
    *,
    boundary_tolerance_fraction: float,
    at_bound_tolerance_fraction: float,
    absolute_tolerance: float,
) -> dict[str, Any]:
    """Return distances and boundary classification for one scalar value."""
    finite_lower = math.isfinite(lower)
    finite_upper = math.isfinite(upper)
    if not math.isfinite(value):
        return {
            "status": "invalid_value",
            "near_lower": False,
            "near_upper": False,
            "at_lower": False,
            "at_upper": False,
            "outside_bounds": False,
            "distance_to_lower": None,
            "distance_to_upper": None,
            "distance_to_nearest_bound": None,
            "normalized_distance_to_lower": None,
            "normalized_distance_to_upper": None,
            "fraction_across_interval": None,
        }

    distance_lower = value - lower if finite_lower else None
    distance_upper = upper - value if finite_upper else None
    outside = bool(
        (finite_lower and value < lower) or (finite_upper and value > upper)
    )

    width = upper - lower if finite_lower and finite_upper else None
    valid_width = width is not None and math.isfinite(width) and width > 0
    if valid_width:
        normalized_lower = distance_lower / width
        normalized_upper = distance_upper / width
        fraction = normalized_lower
        near_margin = max(
            absolute_tolerance,
            boundary_tolerance_fraction * width,
        )
        at_margin = max(
            absolute_tolerance,
            at_bound_tolerance_fraction * width,
        )
    else:
        normalized_lower = None
        normalized_upper = None
        fraction = None
        finite_scale = max(
            abs(value),
            abs(lower) if finite_lower else 0.0,
            abs(upper) if finite_upper else 0.0,
            1.0,
        )
        near_margin = max(
            absolute_tolerance,
            boundary_tolerance_fraction * finite_scale,
        )
        at_margin = max(
            absolute_tolerance,
            at_bound_tolerance_fraction * finite_scale,
        )

    numeric_scale = max(
        abs(value),
        abs(lower) if finite_lower else 0.0,
        abs(upper) if finite_upper else 0.0,
        abs(width) if valid_width else 0.0,
        1.0,
    )
    numeric_slack = max(absolute_tolerance, 1.0e-7 * numeric_scale)
    at_lower = bool(
        finite_lower
        and distance_lower is not None
        and abs(distance_lower) <= at_margin + numeric_slack
    )
    at_upper = bool(
        finite_upper
        and distance_upper is not None
        and abs(distance_upper) <= at_margin + numeric_slack
    )
    near_lower = bool(
        finite_lower
        and distance_lower is not None
        and distance_lower >= -at_margin - numeric_slack
        and distance_lower <= near_margin + numeric_slack
    )
    near_upper = bool(
        finite_upper
        and distance_upper is not None
        and distance_upper >= -at_margin - numeric_slack
        and distance_upper <= near_margin + numeric_slack
    )

    if outside:
        status = "outside_bounds"
    elif at_lower and at_upper:
        status = "at_both_bounds"
    elif at_lower:
        status = "at_lower"
    elif at_upper:
        status = "at_upper"
    elif near_lower and near_upper:
        status = "near_both_bounds"
    elif near_lower:
        status = "near_lower"
    elif near_upper:
        status = "near_upper"
    elif finite_lower or finite_upper:
        status = "interior"
    else:
        status = "unbounded"

    nearest = None
    finite_distances = [
        distance
        for distance in (distance_lower, distance_upper)
        if distance is not None and math.isfinite(distance)
    ]
    if finite_distances:
        nearest = min(abs(distance) for distance in finite_distances)

    return {
        "status": status,
        "near_lower": near_lower,
        "near_upper": near_upper,
        "at_lower": at_lower,
        "at_upper": at_upper,
        "outside_bounds": outside,
        "distance_to_lower": distance_lower,
        "distance_to_upper": distance_upper,
        "distance_to_nearest_bound": nearest,
        "normalized_distance_to_lower": normalized_lower,
        "normalized_distance_to_upper": normalized_upper,
        "fraction_across_interval": fraction,
    }


def _empty_parameter_record(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "value_shape": None,
        "constraint_registered": False,
        "model_coordinate_values": None,
        "model_coordinate_lower_bounds": None,
        "model_coordinate_upper_bounds": None,
        "raw_input_coordinate_available": False,
        "raw_input_coordinate_values": None,
        "raw_input_coordinate_lower_bounds": None,
        "raw_input_coordinate_upper_bounds": None,
        "coordinate_transform_factors": None,
        "coordinate_transform_source": None,
        "gpytorch_raw_parameter_values": None,
        "component_diagnostics": [],
        "boundary_hits": [],
        "n_boundary_hits": 0,
        "boundary_hit_counts_by_dimension": {},
        "boundary_hit_counts_by_side": {},
    }


def _diagnose_parameter(
    obj: Any,
    kernel: Any,
    parameter_name: str,
    *,
    boundary_tolerance_fraction: float,
    at_bound_tolerance_fraction: float,
    absolute_tolerance: float,
) -> dict[str, Any]:
    values = _component_dimension_array(getattr(kernel, parameter_name, None))
    if values is None:
        return _empty_parameter_record(
            f"spectral-mixture kernel does not expose {parameter_name}"
        )

    n_components, n_dimensions = values.shape
    dimension_names = _dimension_names(n_dimensions)
    lower, upper, registered = _constraint_bounds(
        kernel,
        parameter_name,
        values.shape,
    )
    raw_parameter = _component_dimension_array(
        getattr(kernel, f"raw_{parameter_name}", None)
    )

    factors, factor_source = _coordinate_factors(
        obj,
        parameter_name,
        n_dimensions,
    )
    if factors is not None:
        raw_values = values / factors.reshape(1, -1)
        raw_lower = lower / factors.reshape(1, -1)
        raw_upper = upper / factors.reshape(1, -1)
    else:
        raw_values = None
        raw_lower = None
        raw_upper = None

    rows = []
    hits = []
    dimension_counts: dict[str, int] = {}
    side_counts: dict[str, int] = {}
    for component_index in range(n_components):
        for dimension_index, dimension_name in enumerate(dimension_names):
            value = float(values[component_index, dimension_index])
            lo = float(lower[component_index, dimension_index])
            hi = float(upper[component_index, dimension_index])
            classification = _boundary_status(
                value,
                lo,
                hi,
                boundary_tolerance_fraction=boundary_tolerance_fraction,
                at_bound_tolerance_fraction=at_bound_tolerance_fraction,
                absolute_tolerance=absolute_tolerance,
            )
            row = {
                "parameter": parameter_name,
                "component_index": int(component_index),
                "dimension_index": int(dimension_index),
                "dimension_name": dimension_name,
                "model_coordinate_value": value,
                "model_coordinate_lower_bound": lo,
                "model_coordinate_upper_bound": hi,
                "gpytorch_raw_parameter_value": (
                    float(raw_parameter[component_index, dimension_index])
                    if raw_parameter is not None
                    and raw_parameter.shape == values.shape
                    else None
                ),
                "raw_input_coordinate_value": (
                    float(raw_values[component_index, dimension_index])
                    if raw_values is not None
                    else None
                ),
                "raw_input_coordinate_lower_bound": (
                    float(raw_lower[component_index, dimension_index])
                    if raw_lower is not None
                    else None
                ),
                "raw_input_coordinate_upper_bound": (
                    float(raw_upper[component_index, dimension_index])
                    if raw_upper is not None
                    else None
                ),
                **classification,
            }
            rows.append(row)

            sides = []
            if classification["near_lower"] or classification["at_lower"]:
                sides.append("lower")
            if classification["near_upper"] or classification["at_upper"]:
                sides.append("upper")
            for side in sides:
                hit = dict(row)
                hit["bound_side"] = side
                hit["at_bound"] = bool(classification[f"at_{side}"])
                hit["near_bound"] = bool(classification[f"near_{side}"])
                hits.append(hit)
                dimension_counts[dimension_name] = (
                    dimension_counts.get(dimension_name, 0) + 1
                )
                side_counts[side] = side_counts.get(side, 0) + 1

    return {
        "available": True,
        "reason": None,
        "value_shape": list(getattr(kernel, parameter_name).shape),
        "constraint_registered": registered,
        "model_coordinate_values": values.tolist(),
        "model_coordinate_lower_bounds": lower.tolist(),
        "model_coordinate_upper_bounds": upper.tolist(),
        "raw_input_coordinate_available": factors is not None,
        "raw_input_coordinate_values": (
            raw_values.tolist() if raw_values is not None else None
        ),
        "raw_input_coordinate_lower_bounds": (
            raw_lower.tolist() if raw_lower is not None else None
        ),
        "raw_input_coordinate_upper_bounds": (
            raw_upper.tolist() if raw_upper is not None else None
        ),
        "coordinate_transform_factors": (
            factors.tolist() if factors is not None else None
        ),
        "coordinate_transform_source": factor_source,
        "gpytorch_raw_parameter_values": (
            raw_parameter.tolist() if raw_parameter is not None else None
        ),
        "component_diagnostics": rows,
        "boundary_hits": hits,
        "n_boundary_hits": len(hits),
        "boundary_hit_counts_by_dimension": dimension_counts,
        "boundary_hit_counts_by_side": side_counts,
    }


def diagnose_spectral_mixture_ard(
    fitted_object: Any,
    *,
    boundary_tolerance_fraction: float = 0.05,
    at_bound_tolerance_fraction: float = 1.0e-6,
    absolute_tolerance: float = 1.0e-12,
    requested_num_mixtures: int | None = None,
) -> dict[str, Any]:
    """Return component- and dimension-specific fitted ARD diagnostics.

    Parameters
    ----------
    fitted_object
        A fitted :class:`~pgmuvi.lightcurve.Lightcurve`, GP model, or kernel
        reachable through the standard PGMUVI model attributes.
    boundary_tolerance_fraction
        Fraction of a finite registered interval used to classify ``near``
        lower/upper bounds.  The default is 0.05.
    at_bound_tolerance_fraction
        Smaller interval fraction used to distinguish ``at`` from merely
        ``near`` a bound.  The default is ``1e-6``.
    absolute_tolerance
        Absolute floor applied to both boundary tolerances.
    requested_num_mixtures
        Explicit component count requested by the caller, when known.  This is
        used to distinguish a model that happened to fit one component from one
        whose component count was explicitly fixed at one.

    Returns
    -------
    dict
        JSON-safe diagnostics for both ``mixture_means`` and
        ``mixture_scales``.  Boundary hits are descriptive and do not perform
        model selection or alter the fitted object.
    """
    for name, value in (
        ("boundary_tolerance_fraction", boundary_tolerance_fraction),
        ("at_bound_tolerance_fraction", at_bound_tolerance_fraction),
        ("absolute_tolerance", absolute_tolerance),
    ):
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite non-negative number") from exc
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(f"{name} must be a finite non-negative number")

    boundary_tolerance_fraction = float(boundary_tolerance_fraction)
    at_bound_tolerance_fraction = float(at_bound_tolerance_fraction)
    absolute_tolerance = float(absolute_tolerance)
    if boundary_tolerance_fraction >= 0.5:
        raise ValueError("boundary_tolerance_fraction must be less than 0.5")
    if at_bound_tolerance_fraction > boundary_tolerance_fraction:
        raise ValueError(
            "at_bound_tolerance_fraction must not exceed "
            "boundary_tolerance_fraction"
        )

    kernel = _find_spectral_mixture_kernel(fitted_object)
    if kernel is None:
        return {
            "schema_version": SPECTRAL_MIXTURE_ARD_DIAGNOSTIC_SCHEMA_VERSION,
            "available": False,
            "reason": "fitted object does not expose a spectral-mixture kernel",
            "parameterization": "component_by_ard_dimension",
            "coordinate_order": [],
            "ard_index": {},
            "num_mixtures": None,
            "requested_num_mixtures": requested_num_mixtures,
            "num_mixtures_is_one": False,
            "num_mixtures_fixed_at_one": requested_num_mixtures == 1,
            "num_mixtures_request_source": (
                "explicit_fit_kwargs"
                if requested_num_mixtures is not None
                else "not_recorded"
            ),
            "boundary_tolerance_fraction": boundary_tolerance_fraction,
            "at_bound_tolerance_fraction": at_bound_tolerance_fraction,
            "absolute_tolerance": absolute_tolerance,
            "parameters": {},
            "boundary_hits": [],
            "n_boundary_hits": 0,
            "boundary_pressure_scope": "none",
            "boundary_hit_counts_by_parameter": {},
            "boundary_hit_counts_by_dimension": {},
            "boundary_component_counts_by_dimension": {},
            "boundary_hit_counts_by_side": {},
        }

    parameter_records = {
        name: _diagnose_parameter(
            fitted_object,
            kernel,
            name,
            boundary_tolerance_fraction=boundary_tolerance_fraction,
            at_bound_tolerance_fraction=at_bound_tolerance_fraction,
            absolute_tolerance=absolute_tolerance,
        )
        for name in _PARAMETER_NAMES
    }

    available_records = [
        record for record in parameter_records.values() if record["available"]
    ]
    n_dimensions = 0
    num_mixtures = None
    for parameter_name in _PARAMETER_NAMES:
        values = _component_dimension_array(getattr(kernel, parameter_name, None))
        if values is not None:
            num_mixtures = int(values.shape[0])
            n_dimensions = max(n_dimensions, int(values.shape[1]))

    dimension_names = _dimension_names(n_dimensions)
    all_hits = []
    parameter_counts: dict[str, int] = {}
    dimension_counts: dict[str, int] = {}
    side_counts: dict[str, int] = {}
    component_sets: dict[str, set[int]] = {
        name: set() for name in dimension_names
    }
    for parameter_name, record in parameter_records.items():
        hits = list(record.get("boundary_hits") or [])
        if hits:
            parameter_counts[parameter_name] = len(hits)
        for hit in hits:
            all_hits.append(hit)
            dimension_name = str(hit["dimension_name"])
            side = str(hit["bound_side"])
            dimension_counts[dimension_name] = (
                dimension_counts.get(dimension_name, 0) + 1
            )
            side_counts[side] = side_counts.get(side, 0) + 1
            component_sets.setdefault(dimension_name, set()).add(
                int(hit["component_index"])
            )

    component_counts = {
        name: len(indices)
        for name, indices in component_sets.items()
        if indices
    }
    time_hits = component_counts.get("temporal_frequency", 0)
    wavelength_hits = component_counts.get("wavelength_frequency", 0)
    if time_hits and wavelength_hits:
        pressure_scope = "both"
    elif time_hits:
        pressure_scope = "temporal_only"
    elif wavelength_hits:
        pressure_scope = "wavelength_only"
    else:
        pressure_scope = "none"

    actual_is_one = num_mixtures == 1
    fixed_at_one = requested_num_mixtures == 1
    request_source = (
        "explicit_fit_kwargs"
        if requested_num_mixtures is not None
        else "fitted_kernel_only"
    )

    diagnostics = {
        "schema_version": SPECTRAL_MIXTURE_ARD_DIAGNOSTIC_SCHEMA_VERSION,
        "available": bool(available_records),
        "reason": None if available_records else "ARD parameters are unavailable",
        "parameterization": "component_by_ard_dimension",
        "coordinate_order": dimension_names,
        "ard_index": {
            name: index for index, name in enumerate(dimension_names)
        },
        "num_mixtures": num_mixtures,
        "requested_num_mixtures": requested_num_mixtures,
        "num_mixtures_is_one": actual_is_one,
        "num_mixtures_fixed_at_one": fixed_at_one,
        "num_mixtures_request_source": request_source,
        "single_component_interpretation_warning": bool(actual_is_one),
        "boundary_tolerance_fraction": boundary_tolerance_fraction,
        "at_bound_tolerance_fraction": at_bound_tolerance_fraction,
        "absolute_tolerance": absolute_tolerance,
        "parameters": parameter_records,
        "boundary_hits": all_hits,
        "n_boundary_hits": len(all_hits),
        "boundary_pressure_scope": pressure_scope,
        "boundary_hit_counts_by_parameter": parameter_counts,
        "boundary_hit_counts_by_dimension": dimension_counts,
        "boundary_component_counts_by_dimension": component_counts,
        "boundary_hit_counts_by_side": side_counts,
        "n_temporal_boundary_components": time_hits,
        "n_wavelength_boundary_components": wavelength_hits,
    }
    return _json_safe(diagnostics)
