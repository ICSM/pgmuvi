"""Parameter-estimate builder infrastructure.

This module defines builder classes that convert parameter schemas and
diagnostic contexts into parameter estimates.
"""

from __future__ import annotations

from pgmuvi.parameter_context import ParameterEstimationContext
from pgmuvi.parameter_estimates import (
    ParameterEstimate,
    ParameterEstimateCollection,
)
from pgmuvi.parameter_specs import (
    ConstraintStrategy,
    GuessStrategy,
    ParameterSpec,
    ParameterSpecCollection,
)


class ParameterEstimateBuilder:
    """Construct parameter estimates from schemas and diagnostics."""

    def build(
        self,
        schema: ParameterSpecCollection,
        context: ParameterEstimationContext,
    ) -> ParameterEstimateCollection:
        """Build parameter estimates from a schema and diagnostic context."""
        estimates = ParameterEstimateCollection()

        for spec in schema:
            estimates.add(self.build_one(spec, context))

        return estimates

    def build_one(
        self,
        spec: ParameterSpec,
        context: ParameterEstimationContext,
    ) -> ParameterEstimate:
        """Build one parameter estimate."""
        value = self._estimate_value(spec, context)
        constraint = self._estimate_constraint(spec, context)

        return ParameterEstimate(
            spec=spec,
            value=value,
            constraint=constraint,
            value_source=spec.guess_strategy.value if spec.guess_strategy else None,
            constraint_source=(
                spec.constraint_strategy.value if spec.constraint_strategy else None
            ),
        )

    def _estimate_value(
        self,
        spec: ParameterSpec,
        context: ParameterEstimationContext,
    ):
        """Estimate an initial value for one parameter specification."""
        if spec.guess_strategy is GuessStrategy.MEDIAN_FLUX:
            return self._estimate_median_flux(context)

        return None

    def _estimate_constraint(
        self,
        spec: ParameterSpec,
        context: ParameterEstimationContext,
    ):
        """Estimate a constraint for one parameter specification."""
        if spec.constraint_strategy is ConstraintStrategy.ROBUST_FLUX_RANGE:
            return self._estimate_robust_flux_range(context)

        return None

    @staticmethod
    def _estimate_median_flux(context: ParameterEstimationContext):
        """Estimate a flux-level value from global median-flux diagnostics."""
        if context.global_diagnostics is None:
            return None

        return context.global_diagnostics.median_flux

    @staticmethod
    def _estimate_robust_flux_range(context: ParameterEstimationContext):
        """Estimate a robust global flux range using p2.5 and p97.5."""
        if context.global_diagnostics is None:
            return None

        percentiles = context.global_diagnostics.flux_percentiles

        if 2.5 not in percentiles or 97.5 not in percentiles:
            return None

        return (percentiles[2.5], percentiles[97.5])

    @staticmethod
    def _get_flux_percentile(
        context: ParameterEstimationContext,
        percentile: float,
    ):
        """Return a global flux percentile value if available."""
        if context.global_diagnostics is None:
            return None

        return context.global_diagnostics.flux_percentiles.get(percentile)

    def _estimate_robust_flux_interval(
        self,
        context: ParameterEstimationContext,
    ):
        """Return (p2.5, p97.5) if available."""
        p025 = self._get_flux_percentile(context, 2.5)
        p975 = self._get_flux_percentile(context, 97.5)

        if p025 is None or p975 is None:
            return None

        return (p025, p975)

    def _estimate_robust_flux_span(
        self,
        context: ParameterEstimationContext,
    ):
        """Return p97.5 - p2.5 if available."""
        interval = self._estimate_robust_flux_interval(context)

        if interval is None:
            return None

        lower, upper = interval
        return upper - lower
