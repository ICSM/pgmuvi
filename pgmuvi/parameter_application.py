"""Application layer for parameter estimates.

This module defines the interface for applying physical-space parameter
estimates to model parameters. Concrete application logic is added in
later commits.
"""

from __future__ import annotations

from pgmuvi.parameter_estimates import ParameterEstimateCollection
from pgmuvi.parameter_specs import ParameterScale
import math


class ParameterEstimateApplicator:
    """Apply parameter estimates to a model."""

    def apply(
        self,
        model,
        estimates: ParameterEstimateCollection,
    ):
        """Apply a collection of parameter estimates to a model."""
        raise NotImplementedError(
            "Parameter estimate application is not implemented yet."
        )

    def _resolve_parameter(self, model, parameter_name: str):
        """Resolve a dotted parameter path on a model."""
        obj = model

        for part in parameter_name.split("."):
            obj = getattr(obj, part)

        return obj

    def _apply_value(self, parameter, estimate):
        """Apply one estimated value to a resolved parameter."""
        raise NotImplementedError(
            "Parameter value application is not implemented yet."
        )

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
