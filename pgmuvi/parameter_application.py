"""Application layer for parameter estimates.

This module defines utilities for applying physical-space parameter
estimates to GPyTorch model parameters, including parameter-path
resolution and parameter-space transformations.
"""

from __future__ import annotations

from pgmuvi.parameter_estimates import ParameterEstimateCollection
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

            results[estimate.name] = {
                "value": self._apply_value(
                    target_module,
                    parameter_name,
                    estimate,
                ),
                "constraint": self._apply_constraint(model, estimate),
            }

        return results

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
            value_tensor = value_tensor.reshape_as(current)

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

        target_module.register_constraint(
            constraint_target,
            gpytorch.constraints.Interval(
                lower,
                upper,
            ),
        )

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

            value_tensor = torch.as_tensor(estimate.value)
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
