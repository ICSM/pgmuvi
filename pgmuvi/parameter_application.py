"""Application layer for parameter estimates.

This module defines utilities for applying physical-space parameter
estimates to GPyTorch model parameters, including parameter-path
resolution and parameter-space transformations.
"""

from __future__ import annotations

from pgmuvi.constraint_utils import bounds_are_equivalent
from pgmuvi.constraint_utils import bounds_are_valid
from pgmuvi.constraint_utils import clamp_to_constraint_interior
from pgmuvi.constraint_utils import get_bounds
from pgmuvi.constraint_utils import intersect_constraint_bounds
from pgmuvi.constraint_utils import make_interval_constraint
from pgmuvi.constraint_utils import register_constraint_preserving_value
from pgmuvi.parameter_estimates import ParameterEstimateCollection
from pgmuvi.parameter_specs import ConstraintStrategy
from pgmuvi.parameter_specs import GuessStrategy
from pgmuvi.parameter_specs import ParameterScale
import math
import torch
import gpytorch


class ParameterEstimateApplicator:
    """Apply parameter estimates to a model."""

    def apply(
        self,
        model,
        estimates: ParameterEstimateCollection,
    ):
        """Apply available parameter estimate values and constraints to a model."""
        results = {}

        for estimate in estimates:
            module_path, parameter_name = self._split_parameter_path(
                estimate.name
            )

            target_module = (
                model
                if not module_path
                else self._resolve_parameter(model, module_path)
            )

            constraint_applied = self._apply_constraint(model, estimate)
            value_applied = self._apply_value(
                target_module,
                parameter_name,
                estimate,
            )

            result = {
                "value": value_applied,
                "constraint": constraint_applied,
                "value_reason": (
                    None
                    if value_applied
                    else estimate.metadata.get(
                        "value_reason",
                        "value_unavailable",
                    )
                    or "value_unavailable"
                ),
                "constraint_reason": (
                    None
                    if constraint_applied
                    else estimate.metadata.get(
                        "constraint_reason",
                        "constraint_unavailable",
                    )
                ),
            }

            constraint_action = estimate.metadata.get("constraint_action")
            if constraint_action is not None:
                result["constraint_action"] = constraint_action

            if self._is_wavelength_range_estimate(estimate):
                result["wavelength_estimate_provenance"] = {
                    "value_source": estimate.value_source,
                    "constraint_source": estimate.constraint_source,
                    "estimated_value": self._to_python(estimate.value),
                    "estimated_constraint": self._to_python(
                        estimate.constraint
                    ),
                    "effective_constraint": self._to_python(
                        self._effective_constraint_bounds(
                            target_module,
                            parameter_name,
                        )
                    ),
                    "diagnostics": self._to_python(estimate.diagnostics),
                }

            if self._is_wavelength_mean_estimate(estimate):
                result["wavelength_mean_estimate_provenance"] = {
                    "value_source": estimate.value_source,
                    "constraint_source": estimate.constraint_source,
                    "estimated_value": self._to_python(estimate.value),
                    "estimated_constraint": self._to_python(
                        estimate.constraint
                    ),
                    "effective_constraint": self._to_python(
                        self._effective_constraint_bounds(
                            target_module,
                            parameter_name,
                        )
                    ),
                    "diagnostics": self._to_python(estimate.diagnostics),
                }

            results[estimate.name] = result

        return results

    @staticmethod
    def _is_wavelength_range_estimate(estimate) -> bool:
        """Return whether an estimate comes from wavelength-range diagnostics."""
        return (
            estimate.spec.guess_strategy is GuessStrategy.WAVELENGTH_RANGE
            or estimate.spec.constraint_strategy
            is ConstraintStrategy.WAVELENGTH_RANGE
        )

    @staticmethod
    def _is_wavelength_mean_estimate(estimate) -> bool:
        """Return whether an estimate comes from wavelength-mean diagnostics."""
        return (
            estimate.spec.guess_strategy is GuessStrategy.WAVELENGTH_MEAN
            or estimate.spec.constraint_strategy
            is ConstraintStrategy.WAVELENGTH_MEAN
        )

    @staticmethod
    def _to_python(value):
        """Return a JSON-safe scalar/list representation when possible."""
        if value is None:
            return None
        if isinstance(value, dict):
            return {
                key: ParameterEstimateApplicator._to_python(item)
                for key, item in value.items()
            }
        if isinstance(value, tuple):
            return [
                ParameterEstimateApplicator._to_python(item)
                for item in value
            ]
        if isinstance(value, list):
            return [
                ParameterEstimateApplicator._to_python(item)
                for item in value
            ]
        if isinstance(value, torch.Tensor):
            detached = value.detach().cpu()
            if detached.numel() == 1:
                return float(detached.item())
            return detached.tolist()
        if isinstance(value, (bool, int, float, str)):
            return value
        return str(value)

    @staticmethod
    def _effective_constraint_bounds(target_module, parameter_name):
        """Return the bounds registered on a constrained GPyTorch parameter."""
        raw_parameter_name = (
            parameter_name
            if parameter_name.startswith("raw_")
            else f"raw_{parameter_name}"
        )
        constraint = getattr(
            target_module,
            f"{raw_parameter_name}_constraint",
            None,
        )
        if constraint is None:
            return None
        return get_bounds(constraint)

    def _resolve_parameter(self, model, parameter_name: str):
        """Resolve a dotted parameter path on a model."""
        obj = model

        for part in parameter_name.split("."):
            obj = getattr(obj, part)

        return obj

    @staticmethod
    def _is_explicit_log_parameter(parameter_name: str) -> bool:
        """Return True when the model parameter itself stores log(value)."""
        local_name = parameter_name.split(".")[-1]
        return local_name.startswith("log_")

    @staticmethod
    def _split_parameter_path(parameter_name: str):
        """Split a parameter path into module path and local parameter name."""
        parts = parameter_name.split(".")

        if len(parts) == 1:
            return "", parts[0]

        return ".".join(parts[:-1]), parts[-1]

    def _apply_value(self, target_module, parameter_name, estimate):
        """Apply one estimated value to a resolved parameter."""
        transformed_value = self._transform_value(estimate)

        if transformed_value is None:
            return False

        current = getattr(target_module, parameter_name)

        dtype = getattr(current, "dtype", None)
        device = getattr(current, "device", None)

        if isinstance(current, torch.nn.Parameter):
            dtype = current.data.dtype
            device = current.data.device

        value_tensor = torch.as_tensor(
            transformed_value,
            dtype=dtype,
            device=device,
        )

        if hasattr(current, "shape"):
            if value_tensor.numel() != current.numel():
                estimate.metadata["value_reason"] = "shape_mismatch"
                estimate.metadata["expected_shape"] = tuple(current.shape)
                estimate.metadata["actual_shape"] = tuple(value_tensor.shape)
                return False

            value_tensor = value_tensor.reshape_as(current)

        if isinstance(target_module, gpytorch.Module):
            raw_parameter_name = (
                parameter_name
                if parameter_name.startswith("raw_")
                else f"raw_{parameter_name}"
            )
            constraint = getattr(
                target_module,
                f"{raw_parameter_name}_constraint",
                None,
            )
            if constraint is not None:
                value_tensor = clamp_to_constraint_interior(
                    value_tensor,
                    constraint,
                )
        elif isinstance(current, torch.nn.Parameter):
            transformed_constraint = self._transform_constraint(estimate)
            if transformed_constraint is not None:
                value_tensor = clamp_to_constraint_interior(
                    value_tensor,
                    make_interval_constraint(*transformed_constraint),
                )

        with torch.no_grad():
            if isinstance(target_module, gpytorch.Module):
                target_module.initialize(**{parameter_name: value_tensor})
            elif isinstance(current, torch.nn.Parameter):
                current.copy_(value_tensor)
            else:
                setattr(target_module, parameter_name, value_tensor)

        return True

    def _apply_constraint(self, model, estimate):
        """Apply one estimated constraint to a model parameter."""
        transformed_constraint = self._transform_constraint(
            estimate
        )

        if transformed_constraint is None:
            return False

        lower, upper = transformed_constraint

        module_path, parameter_name = self._split_parameter_path(
            estimate.name
        )

        target_module = (
            model
            if not module_path
            else self._resolve_parameter(model, module_path)
        )

        raw_parameter_name = (
            parameter_name
            if parameter_name.startswith("raw_")
            else f"raw_{parameter_name}"
        )

        constraint_target = (
            raw_parameter_name
            if hasattr(target_module, raw_parameter_name)
            else parameter_name
        )

        current = getattr(target_module, parameter_name, None)
        if (
            isinstance(current, torch.nn.Parameter)
            and constraint_target == parameter_name
        ):
            estimate.metadata["constraint_action"] = (
                "constraint_not_enforceable_plain_parameter"
            )
            estimate.metadata["constraint_reason"] = (
                "constraint_not_enforceable_plain_parameter"
            )
            return False

        proposed_constraint = gpytorch.constraints.Interval(
            lower,
            upper,
        )

        if (
            isinstance(target_module, gpytorch.Module)
            and constraint_target.startswith("raw_")
        ):
            existing_constraint = getattr(
                target_module,
                f"{constraint_target}_constraint",
                None,
            )

            constraint = proposed_constraint
            if (
                estimate.spec.constraint_strategy
                in {
                    ConstraintStrategy.DEFAULT,
                    ConstraintStrategy.WAVELENGTH_RANGE,
                    ConstraintStrategy.WAVELENGTH_MEAN,
                }
                and existing_constraint is not None
            ):
                existing_bounds = get_bounds(existing_constraint)
                proposed_bounds = get_bounds(proposed_constraint)
                intersected_bounds = intersect_constraint_bounds(
                    existing_constraint,
                    proposed_constraint,
                )

                if not bounds_are_valid(intersected_bounds):
                    estimate.metadata["constraint_action"] = (
                        "conflict_kept_existing"
                    )
                    return True

                if bounds_are_equivalent(intersected_bounds, existing_bounds):
                    estimate.metadata["constraint_action"] = "kept_existing"
                    return True

                if bounds_are_equivalent(intersected_bounds, proposed_bounds):
                    estimate.metadata["constraint_action"] = "applied"
                else:
                    estimate.metadata["constraint_action"] = "tightened"

                constraint = make_interval_constraint(*intersected_bounds)
            else:
                estimate.metadata["constraint_action"] = "applied"

            register_constraint_preserving_value(
                target_module,
                constraint_target,
                constraint,
            )
        else:
            target_module.register_constraint(
                constraint_target,
                proposed_constraint,
            )
            estimate.metadata["constraint_action"] = "applied"

        return True

    def _transform_value(self, estimate):
        """Transform a physical-space estimate value into model parameter space."""
        if estimate.value is None:
            return None

        if estimate.spec.scale is ParameterScale.LINEAR:
            return estimate.value

        if estimate.spec.scale is ParameterScale.LOG:
            if not self._is_explicit_log_parameter(estimate.name):
                return estimate.value

            value_tensor = torch.as_tensor(
                estimate.value,
                dtype=torch.get_default_dtype(),
            )

            if value_tensor.numel() == 1:
                scalar = float(value_tensor.item())
                if scalar <= 0:
                    raise ValueError(
                        f"Cannot apply log transform to non-positive value for "
                        f"{estimate.name!r}."
                    )
                return math.log(scalar)

            if torch.any(value_tensor <= 0):
                raise ValueError(
                    f"Cannot apply log transform to non-positive value for "
                    f"{estimate.name!r}."
                )

            return torch.log(value_tensor)

        raise NotImplementedError(
            f"Value transformation for scale {estimate.spec.scale.value!r} "
            "is not implemented yet."
        )

    def _transform_constraint(self, estimate):
        """Transform a physical-space constraint into parameter space."""
        if estimate.constraint is None:
            return None

        lower, upper = estimate.constraint

        if estimate.spec.scale is ParameterScale.LINEAR:
            return (lower, upper)

        if estimate.spec.scale is ParameterScale.LOG:
            if not self._is_explicit_log_parameter(estimate.name):
                return (lower, upper)

            lower_tensor = torch.as_tensor(
                lower,
                dtype=torch.get_default_dtype(),
            )
            upper_tensor = torch.as_tensor(
                upper,
                dtype=torch.get_default_dtype(),
            )

            if lower_tensor.numel() == 1 and upper_tensor.numel() == 1:
                lower_scalar = float(lower_tensor.item())
                upper_scalar = float(upper_tensor.item())

                if lower_scalar <= 0 or upper_scalar <= 0:
                    raise ValueError(
                        f"Cannot apply log transform to non-positive constraint "
                        f"for {estimate.name!r}."
                    )

                return (
                    math.log(lower_scalar),
                    math.log(upper_scalar),
                )

            if torch.any(lower_tensor <= 0) or torch.any(upper_tensor <= 0):
                raise ValueError(
                    f"Cannot apply log transform to non-positive constraint "
                    f"for {estimate.name!r}."
                )

            return (
                torch.log(lower_tensor),
                torch.log(upper_tensor),
            )

        raise NotImplementedError(
            f"Constraint transformation for scale "
            f"{estimate.spec.scale.value!r} is not implemented yet."
        )
