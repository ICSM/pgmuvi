"""Utilities for safe GPyTorch constraint registration.

GPyTorch stores constrained parameters through raw parameters plus constraint
transforms. Replacing a constraint without updating the raw value changes the
reported constrained value. The helpers here centralise constraint inspection,
interior clipping, and value-preserving re-registration.
"""

from __future__ import annotations

import math
from typing import Any

import gpytorch
import torch


_RELATIVE_INTERIOR_EPS = 1.0e-6
_ABSOLUTE_INTERIOR_EPS = 1.0e-12


def _as_tensor_like(value: Any, reference: torch.Tensor) -> torch.Tensor:
    """Return ``value`` as a tensor matching ``reference`` dtype and device."""
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def get_bounds(constraint: Any) -> tuple[Any, Any]:
    """Return lower and upper bounds for a GPyTorch constraint.

    Missing one-sided bounds are represented by ``-inf``/``inf``. The function
    deliberately returns the native scalar/tensor objects when available so
    callers can preserve dtype, device, and shape where needed.
    """
    if constraint is None:
        return -math.inf, math.inf

    lower = getattr(constraint, "lower_bound", -math.inf)
    upper = getattr(constraint, "upper_bound", math.inf)

    if lower is None:
        lower = -math.inf
    if upper is None:
        upper = math.inf

    return lower, upper


def clamp_to_constraint_interior(
    value: Any,
    constraint: Any,
    *,
    relative_eps: float = _RELATIVE_INTERIOR_EPS,
    absolute_eps: float = _ABSOLUTE_INTERIOR_EPS,
) -> torch.Tensor:
    """Clamp a value to the numerically safe interior of a constraint.

    GPyTorch interval constraints are open under the inverse-logit transform:
    values exactly on an interval endpoint are not valid initialisation targets.
    This helper moves values just inside finite bounds before calling
    ``initialize``.
    """
    value_tensor = torch.as_tensor(value).clone()
    lower, upper = get_bounds(constraint)

    lower_tensor = _as_tensor_like(lower, value_tensor)
    upper_tensor = _as_tensor_like(upper, value_tensor)

    finite_lower = torch.isfinite(lower_tensor)
    finite_upper = torch.isfinite(upper_tensor)

    if torch.any(finite_lower & finite_upper):
        width = upper_tensor - lower_tensor
        margin = torch.maximum(
            torch.abs(width) * relative_eps,
            torch.as_tensor(
                absolute_eps,
                dtype=value_tensor.dtype,
                device=value_tensor.device,
            ),
        )
        lower_safe = lower_tensor + margin
        upper_safe = upper_tensor - margin
        bounded = finite_lower & finite_upper
        value_tensor = torch.where(
            bounded,
            torch.minimum(torch.maximum(value_tensor, lower_safe), upper_safe),
            value_tensor,
        )

    lower_only = finite_lower & ~finite_upper
    if torch.any(lower_only):
        margin = torch.maximum(
            torch.abs(lower_tensor) * relative_eps,
            torch.as_tensor(
                absolute_eps,
                dtype=value_tensor.dtype,
                device=value_tensor.device,
            ),
        )
        lower_safe = lower_tensor + margin
        value_tensor = torch.where(
            lower_only,
            torch.maximum(value_tensor, lower_safe),
            value_tensor,
        )

    upper_only = ~finite_lower & finite_upper
    if torch.any(upper_only):
        margin = torch.maximum(
            torch.abs(upper_tensor) * relative_eps,
            torch.as_tensor(
                absolute_eps,
                dtype=value_tensor.dtype,
                device=value_tensor.device,
            ),
        )
        upper_safe = upper_tensor - margin
        value_tensor = torch.where(
            upper_only,
            torch.minimum(value_tensor, upper_safe),
            value_tensor,
        )

    return value_tensor


def make_interval_constraint(lower: Any, upper: Any) -> gpytorch.constraints.Interval:
    """Create a GPyTorch interval-style constraint from finite or one-sided bounds."""
    lower_tensor = torch.as_tensor(lower)
    upper_tensor = torch.as_tensor(upper)

    lower_finite = torch.all(torch.isfinite(lower_tensor)).item()
    upper_finite = torch.all(torch.isfinite(upper_tensor)).item()

    if lower_finite and upper_finite:
        return gpytorch.constraints.Interval(lower, upper)

    if lower_finite and not upper_finite:
        return gpytorch.constraints.GreaterThan(lower)

    if not lower_finite and upper_finite:
        return gpytorch.constraints.LessThan(upper)

    return gpytorch.constraints.Interval(-math.inf, math.inf)


def intersect_constraint_bounds(existing: Any, proposed: Any) -> tuple[Any, Any]:
    """Return the intersection bounds of two constraints.

    Existing bounds are never loosened: callers can decide whether to register
    the returned bounds or keep the existing constraint when the intersection is
    identical.
    """
    existing_lower, existing_upper = get_bounds(existing)
    proposed_lower, proposed_upper = get_bounds(proposed)

    existing_lower_tensor = torch.as_tensor(existing_lower)
    proposed_lower_tensor = torch.as_tensor(proposed_lower)
    existing_upper_tensor = torch.as_tensor(existing_upper)
    proposed_upper_tensor = torch.as_tensor(proposed_upper)

    lower = torch.maximum(existing_lower_tensor, proposed_lower_tensor)
    upper = torch.minimum(existing_upper_tensor, proposed_upper_tensor)

    return lower, upper


def bounds_are_equivalent(first: tuple[Any, Any], second: tuple[Any, Any]) -> bool:
    """Return True when two lower/upper bound pairs are numerically equal."""
    first_lower, first_upper = first
    second_lower, second_upper = second

    return bool(
        torch.allclose(torch.as_tensor(first_lower), torch.as_tensor(second_lower))
        and torch.allclose(torch.as_tensor(first_upper), torch.as_tensor(second_upper))
    )


def bounds_are_valid(bounds: tuple[Any, Any]) -> bool:
    """Return True when every lower bound is strictly below its upper bound."""
    lower, upper = bounds
    return bool(torch.all(torch.as_tensor(lower) < torch.as_tensor(upper)))


def _constrained_name_from_raw(raw_parameter_name: str) -> str:
    if raw_parameter_name.startswith("raw_"):
        return raw_parameter_name.removeprefix("raw_")
    return raw_parameter_name


def register_constraint_preserving_value(
    module: gpytorch.Module,
    raw_parameter_name: str,
    constraint: gpytorch.constraints.Interval,
) -> None:
    """Register a constraint while preserving the constrained value.

    GPyTorch does not transform raw parameter values when a constraint object is
    replaced. Without an explicit restore, the constrained value is reinterpreted
    under the new transform. This helper reads the current constrained property,
    registers the new constraint, and reinitialises the property to the same
    value clipped into the new constraint's safe interior.
    """
    constrained_name = _constrained_name_from_raw(raw_parameter_name)
    current_value = None
    can_restore = isinstance(module, gpytorch.Module) and hasattr(
        module,
        constrained_name,
    )

    if can_restore:
        current_value = getattr(module, constrained_name)
        if torch.is_tensor(current_value):
            current_value = current_value.detach().clone()

    module.register_constraint(raw_parameter_name, constraint)

    if can_restore and current_value is not None:
        safe_value = clamp_to_constraint_interior(current_value, constraint)
        module.initialize(**{constrained_name: safe_value})
