"""Application layer for parameter estimates.

This module defines the interface for applying physical-space parameter
estimates to model parameters. Concrete application logic is added in
later commits.
"""

from __future__ import annotations

from pgmuvi.parameter_estimates import ParameterEstimateCollection


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
        """Resolve a dotted parameter name on a model."""
        raise NotImplementedError(
            "Parameter resolution is not implemented yet."
        )

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