"""Application layer for parameter estimates.

This module defines the interface for applying physical-space parameter
estimates to model parameters. Concrete application logic is added in
later commits.
"""

from __future__ import annotations

from pgmuvi.parameter_estimates import ParameterEstimateCollection
from pgmuvi.parameter_specs import ParameterScale
import math
import torch


class ParameterEstimateApplicator:
    """Apply parameter estimates to a model."""

    def apply(
        self,
        model,
        estimates: ParameterEstimateCollection,
    ):
        """Apply available parameter estimate values to a model."""
        results = {}

        for estimate in estimates:
            parameter = self._resolve_parameter(model, estimate.name)
            results[estimate.name] = self._apply_value(parameter, estimate)

        return results

    def _resolve_parameter(self, model, parameter_name: str):
        """Resolve a dotted parameter path on a model."""
        obj = model

        for part in parameter_name.split("."):
            obj = getattr(obj, part)

        return obj

    @staticmethod
    def _split_parameter_path(parameter_name: str):
        """Split a parameter path into module path and local parameter name."""
        parts = parameter_name.split(".")

        if len(parts) == 1:
            return "", parts[0]

        return ".".join(parts[:-1]), parts[-1]

    def _apply_value(self, parameter, estimate):
        """Apply one estimated value to a resolved parameter."""
        transformed_value = self._transform_value(estimate)

        if transformed_value is None:
            return False

        value_tensor = torch.as_tensor(
            transformed_value,
            dtype=parameter.data.dtype,
            device=parameter.data.device,
        )

        value_tensor = value_tensor.reshape_as(parameter.data)

        with torch.no_grad():
            parameter.copy_(value_tensor)

        return True

    def _apply_constraint(self, model, estimate):
        """Apply one estimated constraint to a model parameter."""
        raise NotImplementedError(
            "Parameter constraint application is not implemented yet."
        )

    def _transform_value(self, estimate):
        """Transform a physical-space estimate value into model parameter space."""
        if estimate.value is None:
            return None

        if estimate.spec.scale is ParameterScale.LINEAR:
            return estimate.value

        if estimate.spec.scale is ParameterScale.LOG:
            if estimate.value <= 0:
                raise ValueError(
                    f"Cannot apply log transform to non-positive value for "
                    f"{estimate.name!r}."
                )
            return math.log(estimate.value)

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
            if lower <= 0 or upper <= 0:
                raise ValueError(
                    f"Cannot apply log transform to non-positive constraint "
                    f"for {estimate.name!r}."
                )

            return (
                math.log(lower),
                math.log(upper),
            )

        raise NotImplementedError(
            f"Constraint transformation for scale "
            f"{estimate.spec.scale.value!r} is not implemented yet."
        )
