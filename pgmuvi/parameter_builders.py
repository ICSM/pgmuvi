"""Parameter-estimate builder infrastructure.

This module defines builder classes that convert parameter schemas and
diagnostic contexts into parameter estimates.
"""

from __future__ import annotations
import math
import torch

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
        value_reason = self._estimate_value_reason(
            spec,
            context,
            value,
        )

        return ParameterEstimate(
            spec=spec,
            value=value,
            constraint=constraint,
            value_source=spec.guess_strategy.value if spec.guess_strategy else None,
            constraint_source=(
                spec.constraint_strategy.value if spec.constraint_strategy else None
            ),
            metadata={
                "value_reason": value_reason,
            },
        )

    def _estimate_value_reason(
        self,
        spec: ParameterSpec,
        context: ParameterEstimationContext,
        value,
    ):
        """Return a reason string when value estimation did not produce a value."""
        if value is not None:
            return None

        if spec.guess_strategy is GuessStrategy.CONSENSUS_FREQUENCY:
            return "consensus_frequency_unavailable"

        diagnostics = context.global_diagnostics

        if spec.guess_strategy is GuessStrategy.MEDIAN_FLUX:
            if diagnostics is None:
                return "global_diagnostics_unavailable"

            return "median_flux_unavailable"

        if spec.guess_strategy is GuessStrategy.ROBUST_FLUX_SPAN:
            if diagnostics is None:
                return "global_diagnostics_unavailable"

            return "robust_flux_span_unavailable"

        return None

    def _reshape_initial_value(self, value, shape):
        """Return an initial value reshaped to the declared parameter shape."""
        if value is None or shape is None:
            return value

        value_tensor = torch.as_tensor(value)

        if tuple(value_tensor.shape) == tuple(shape):
            return value

        if value_tensor.numel() == 1:
            return value_tensor.expand(shape).clone()

        if value_tensor.numel() == shape[0]:
            view_shape = (shape[0],) + (1,) * (len(shape) - 1)
            return value_tensor.reshape(view_shape).expand(shape).clone()

        return value

    def _expand_component_values(self, values, shape):
        """Expand per-component values to the declared parameter shape."""
        if values is None or shape is None:
            return values

        value_tensor = torch.as_tensor(values)

        if tuple(value_tensor.shape) == tuple(shape):
            return value_tensor

        if len(shape) == 1:
            if value_tensor.numel() < shape[0]:
                return None

            return value_tensor[: shape[0]]

        if value_tensor.numel() < shape[0]:
            return None

        component_values = value_tensor[: shape[0]]
        view_shape = (shape[0],) + (1,) * (len(shape) - 1)

        return component_values.reshape(view_shape).expand(shape).clone()

    def _estimate_value(
        self,
        spec: ParameterSpec,
        context: ParameterEstimationContext,
    ):
        """Estimate an initial value for one parameter specification."""
        if spec.guess_strategy is GuessStrategy.DEFAULT:
            return self._reshape_initial_value(
                spec.initial_value,
                spec.shape,
            )

        if spec.guess_strategy is GuessStrategy.MEDIAN_FLUX:
            return self._estimate_median_flux(context)

        if spec.guess_strategy is GuessStrategy.ROBUST_FLUX_SPAN:
            return self._estimate_robust_flux_span(context)

        if spec.guess_strategy is GuessStrategy.GEOMETRIC_SAMPLING_TIMESCALE:
            return self._estimate_geometric_sampling_timescale(context)

        if spec.guess_strategy is GuessStrategy.BASELINE_FREQUENCY:
            return self._estimate_baseline_frequency(context)

        if spec.guess_strategy is GuessStrategy.CONSENSUS_FREQUENCY:
            return self._estimate_consensus_frequency(spec, context)

        if spec.guess_strategy is GuessStrategy.CONSENSUS_MULTICOMP_PERIOD:
            return self._estimate_consensus_multicomp_period(spec, context)

        return None

    @staticmethod
    def _estimate_consensus_multicomp_period(
        spec: ParameterSpec,
        context: ParameterEstimationContext,
    ):
        """Estimate period(s) from multicomponent consensus diagnostics."""
        diagnostics = context.consensus_diagnostics

        if diagnostics is None:
            return None

        periods = diagnostics.periods

        if periods is None:
            return None

        periods = list(periods)

        if spec.shape is None:
            return periods[0] if periods else None

        if len(spec.shape) != 1:
            return None

        n_components = spec.shape[0]

        if len(periods) < n_components:
            return None

        return periods[:n_components]

    def _estimate_frequency_fallback(
        self,
        spec: ParameterSpec,
        context: ParameterEstimationContext,
    ):
        """Fallback frequency estimate based on data baseline."""
        baseline_frequency = self._estimate_baseline_frequency(context)
        lower = 1.0e-6
        upper = None

        if spec.constraint is not None:
            lower = float(spec.constraint[0])
            upper = float(spec.constraint[1])

        min_frequency = max(10.0 * lower, 1.0e-5)

        if baseline_frequency is None:
            return None

        if spec.shape is None:
            return baseline_frequency

        n_components = spec.shape[0]

        frequencies = torch.tensor(
            [
                max(baseline_frequency * (i + 1), min_frequency)
                for i in range(n_components)
            ],
            dtype=torch.float32,
        )

        if upper is not None:
            frequencies = torch.clamp(
                frequencies,
                min=min_frequency,
                max=0.5 * upper,
            )

        if len(spec.shape) == 1:
            return frequencies

        lower = 1.0e-6

        if spec.constraint is not None:
            lower = float(spec.constraint[0])

        neutral_frequency = max(10.0 * lower, 1.0e-5)

        fallback = torch.full(
            spec.shape,
            min_frequency,
            dtype=torch.float32,
        )

        fallback[:, 0, 0] = frequencies

        return fallback

    def _estimate_consensus_frequency(
        self,
        spec: ParameterSpec,
        context: ParameterEstimationContext,
    ):
        """Estimate frequency from consensus diagnostics."""
        diagnostics = context.consensus_diagnostics

        if diagnostics is None:
            return self._estimate_frequency_fallback(
                spec,
                context,
            )

        frequencies = diagnostics.frequencies

        if frequencies is None and diagnostics.periods is not None:
            frequencies = []

            for period in diagnostics.periods:
                if period is None or period <= 0:
                    continue

                frequencies.append(1.0 / period)

        if frequencies is None:
            return self._estimate_frequency_fallback(
                spec,
                context,
            )

        frequencies = list(frequencies)

        if not frequencies:
            return self._estimate_frequency_fallback(
                spec,
                context,
            )

        if spec.shape is None:
            return frequencies[0]

        return self._expand_component_values(
            frequencies,
            spec.shape,
        )

    def _estimate_constraint(
        self,
        spec: ParameterSpec,
        context: ParameterEstimationContext,
    ):
        """Estimate a constraint for one parameter specification."""
        if spec.constraint_strategy is ConstraintStrategy.DEFAULT:
            return spec.constraint

        if spec.constraint_strategy is ConstraintStrategy.ROBUST_FLUX_RANGE:
            return self._estimate_robust_flux_range(context)

        if spec.constraint_strategy is ConstraintStrategy.ROBUST_POSITIVE_FLUX_SPAN:
            return self._estimate_robust_positive_flux_span_constraint(context)

        return None

    def _estimate_robust_positive_flux_span_constraint(
        self,
        context: ParameterEstimationContext,
    ):
        """Return a positive amplitude constraint based on robust flux span."""
        span = self._estimate_robust_flux_span(context)

        if span is None or span <= 0:
            return None

        lower = max(1e-12, 1e-6 * span)
        upper = 5.0 * span

        return (lower, upper)

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

    @staticmethod
    def _estimate_geometric_sampling_timescale(
        context: ParameterEstimationContext,
    ):
        """Estimate a timescale from sampling diagnostics."""
        diagnostics = context.global_diagnostics

        if diagnostics is None:
            return None

        baseline = diagnostics.baseline_duration
        cadence = diagnostics.median_cadence

        if baseline is None or cadence is None:
            return None

        if baseline <= 0 or cadence <= 0:
            return None

        return math.sqrt(baseline * cadence)

    @staticmethod
    def _estimate_baseline_frequency(
        context: ParameterEstimationContext,
    ):
        """Estimate the lowest baseline-resolved frequency."""
        diagnostics = context.global_diagnostics

        if diagnostics is None:
            return None

        baseline = diagnostics.baseline_duration

        if baseline is None or baseline <= 0:
            return None

        return 1.0 / baseline
