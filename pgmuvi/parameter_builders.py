"""Parameter-estimate builder infrastructure.

This module defines builder classes that convert parameter schemas and
diagnostic contexts into parameter estimates. Scientific estimation
rules are intentionally deferred to later commits.
"""

from __future__ import annotations

from pgmuvi.parameter_context import ParameterEstimationContext
from pgmuvi.parameter_estimates import (
    ParameterEstimate,
    ParameterEstimateCollection,
)
from pgmuvi.parameter_specs import ParameterSpecCollection


class ParameterEstimateBuilder:
    """Construct parameter estimates from schemas and diagnostics."""

    def build(
        self,
        schema: ParameterSpecCollection,
        context: ParameterEstimationContext,
    ) -> ParameterEstimateCollection:
        """Build parameter estimates.

        The initial implementation only creates empty estimates for each
        parameter specification. Scientific estimation rules are added in
        later commits.
        """
        estimates = ParameterEstimateCollection()

        for spec in schema:
            estimates.add(
                ParameterEstimate(
                    spec=spec,
                )
            )

        return estimates