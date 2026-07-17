"""Parameter-estimate builder infrastructure.

This module defines builder classes that convert parameter schemas and
diagnostic contexts into parameter estimates.
"""

from __future__ import annotations
import math
import torch

from pgmuvi.dtypes import DEFAULT_DTYPE
from pgmuvi.parameter_context import ParameterEstimationContext
from pgmuvi.parameter_estimates import (
    ParameterEstimate,
    ParameterEstimateCollection,
)
from pgmuvi.parameter_specs import (
    ConstraintStrategy,
    GuessStrategy,
    ParameterDomain,
    ParameterRole,
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

        metadata = {
            "value_reason": value_reason,
        }
        diagnostics = {}

        if (
            spec.guess_strategy is GuessStrategy.WAVELENGTH_RANGE
            or spec.constraint_strategy is ConstraintStrategy.WAVELENGTH_RANGE
        ):
            wavelength_diagnostics = context.wavelength_diagnostics
            if wavelength_diagnostics is not None:
                diagnostics = {
                    "schema_version": getattr(
                        wavelength_diagnostics,
                        "schema_version",
                        None,
                    ),
                    "coordinate_space": getattr(
                        wavelength_diagnostics,
                        "coordinate_space",
                        None,
                    ),
                    "n_usable_bands": getattr(
                        wavelength_diagnostics,
                        "n_usable_bands",
                        None,
                    ),
                    "wavelength_span": getattr(
                        wavelength_diagnostics,
                        "wavelength_span",
                        None,
                    ),
                    "median_adjacent_spacing": getattr(
                        wavelength_diagnostics,
                        "median_adjacent_spacing",
                        None,
                    ),
                    "largest_gap": getattr(
                        wavelength_diagnostics,
                        "largest_gap",
                        None,
                    ),
                    "recommendation_method": getattr(
                        wavelength_diagnostics,
                        "recommendation_method",
                        None,
                    ),
                }

        if spec.name.startswith("mean_module.") and constraint is not None:
            metadata["constraint_reason"] = (
                "constraint_not_enforceable_plain_parameter"
            )
            metadata["constraint_enforceability"] = "plain_parameter"

        return ParameterEstimate(
            spec=spec,
            value=value,
            constraint=constraint,
            value_source=spec.guess_strategy.value if spec.guess_strategy else None,
            constraint_source=(
                spec.constraint_strategy.value if spec.constraint_strategy else None
            ),
            diagnostics=diagnostics,
            metadata=metadata,
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

        if spec.guess_strategy is GuessStrategy.WAVELENGTH_RANGE:
            if context.wavelength_diagnostics is None:
                return "wavelength_diagnostics_unavailable"

            return "wavelength_lengthscale_recommendation_unavailable"

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

        if spec.guess_strategy is GuessStrategy.WAVELENGTH_RANGE:
            return self._estimate_wavelength_range_value(spec, context)

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
            dtype=DEFAULT_DTYPE,
        )

        if upper is not None:
            frequencies = torch.clamp(
                frequencies,
                min=min_frequency,
                max=0.5 * upper,
            )

        if len(spec.shape) == 1:
            return frequencies

        fallback = torch.full(
            spec.shape,
            min_frequency,
            dtype=DEFAULT_DTYPE,
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
            return self._estimate_default_constraint(spec, context)

        if spec.constraint_strategy is ConstraintStrategy.ROBUST_FLUX_RANGE:
            return self._estimate_robust_flux_range(context)

        if spec.constraint_strategy is ConstraintStrategy.ROBUST_POSITIVE_FLUX_SPAN:
            return self._estimate_robust_positive_flux_span_constraint(context)

        if spec.constraint_strategy is ConstraintStrategy.WAVELENGTH_RANGE:
            return self._estimate_wavelength_range_constraint(spec, context)

        return None

    def _estimate_wavelength_range_value(
        self,
        spec: ParameterSpec,
        context: ParameterEstimationContext,
    ):
        """Return the raw-coordinate wavelength length-scale recommendation."""
        diagnostics = context.wavelength_diagnostics
        if diagnostics is None:
            return None

        value = getattr(
            diagnostics,
            "recommended_lengthscale_initial",
            None,
        )
        return self._reshape_initial_value(value, spec.shape)

    def _estimate_wavelength_range_constraint(
        self,
        spec: ParameterSpec,
        context: ParameterEstimationContext,
    ):
        """Return raw-coordinate wavelength length-scale bounds."""
        diagnostics = context.wavelength_diagnostics
        if diagnostics is None:
            return None

        bounds = getattr(
            diagnostics,
            "recommended_lengthscale_bounds",
            None,
        )
        if bounds is None:
            return None

        lower, upper = bounds
        if spec.shape is None:
            return (lower, upper)

        lower_tensor = torch.full(
            spec.shape,
            float(lower),
            dtype=DEFAULT_DTYPE,
        )
        upper_tensor = torch.full(
            spec.shape,
            float(upper),
            dtype=DEFAULT_DTYPE,
        )
        return (lower_tensor, upper_tensor)

    def _estimate_default_constraint(
        self,
        spec: ParameterSpec,
        context: ParameterEstimationContext,
    ):
        """Return the default constraint, expanding variance guards from data.

        Spectral-mixture weights and ScaleKernel outputscale are variances in
        the GP training target space.  Fixed schema caps are therefore only
        safe as last-resort guards: bright linear-flux light curves can have
        robust variances far above those constants.  When flux percentiles are
        available, expand the upper bound to a multiple of the robust flux
        variance while keeping the schema value as a conservative fallback.
        """
        if not self._uses_data_scaled_variance_guard(spec):
            return spec.constraint

        return self._estimate_data_scaled_variance_constraint(spec, context)

    @staticmethod
    def _uses_data_scaled_variance_guard(spec: ParameterSpec) -> bool:
        """Return whether a DEFAULT variance constraint should scale with data."""
        if spec.domain is not ParameterDomain.VARIANCE:
            return False

        if spec.role is not ParameterRole.WEIGHT:
            return False

        return (
            spec.name.endswith("mixture_weights")
            or spec.name.endswith("outputscale")
        )

    def _estimate_data_scaled_variance_constraint(
        self,
        spec: ParameterSpec,
        context: ParameterEstimationContext,
    ):
        """Return a variance guard expanded from robust flux diagnostics."""
        fallback = spec.constraint
        robust_variance = self._estimate_robust_flux_variance(context)

        if robust_variance is None:
            return fallback

        lower = 1.0e-12
        fallback_upper = None

        if fallback is not None:
            lower = float(fallback[0])
            fallback_upper = float(fallback[1])

        data_upper = 10.0 * robust_variance

        if spec.initial_value is not None:
            initial = torch.as_tensor(spec.initial_value, dtype=DEFAULT_DTYPE)
            if initial.numel():
                data_upper = max(data_upper, float(torch.max(initial)))

        if fallback_upper is not None and math.isfinite(fallback_upper):
            data_upper = max(data_upper, fallback_upper)

        if not math.isfinite(data_upper) or data_upper <= lower:
            return fallback

        return (lower, data_upper)

    def _estimate_robust_flux_variance(
        self,
        context: ParameterEstimationContext,
    ):
        """Estimate target variance from p2.5--p97.5 robust flux span."""
        span = self._estimate_robust_flux_span(context)

        if span is None or span <= 0:
            return None

        # For a Gaussian distribution, p97.5 - p2.5 is approximately
        # 3.92 sigma.  The exact constant is unnecessary here; this is a
        # guard scale, not a statistical estimator used in the likelihood.
        robust_sigma = span / 4.0
        robust_variance = robust_sigma * robust_sigma

        if not math.isfinite(robust_variance) or robust_variance <= 0:
            return None

        return robust_variance

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
