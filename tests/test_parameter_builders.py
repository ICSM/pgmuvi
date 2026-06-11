import unittest

from pgmuvi.parameter_builders import ParameterEstimateBuilder
from pgmuvi.parameter_context import (
    ConsensusDiagnostics,
    LightcurveDiagnostics,
    ParameterEstimationContext,
)
from pgmuvi.parameter_specs import (
    ConstraintStrategy,
    GuessStrategy,
    ParameterDomain,
    ParameterScale,
    ParameterRole,
    ParameterSpec,
    ParameterSpecCollection,
)


class TestParameterEstimateBuilder(unittest.TestCase):

    def test_build_returns_estimate_collection(self):
        schema = ParameterSpecCollection(
            [
                ParameterSpec(
                    name="mean_module.offset",
                    role=ParameterRole.OFFSET,
                    domain=ParameterDomain.FLUX,
                ),
                ParameterSpec(
                    name="mean_module.log_amplitude",
                    role=ParameterRole.AMPLITUDE,
                    domain=ParameterDomain.FLUX,
                ),
            ]
        )

        context = ParameterEstimationContext(
            is_multiband=False,
        )

        builder = ParameterEstimateBuilder()
        estimates = builder.build(schema=schema, context=context)

        self.assertEqual(
            estimates.names(),
            [
                "mean_module.offset",
                "mean_module.log_amplitude",
            ],
        )

        self.assertIsNone(estimates["mean_module.offset"].value)
        self.assertIsNone(estimates["mean_module.offset"].constraint)
        self.assertIsNone(estimates["mean_module.log_amplitude"].value)
        self.assertIsNone(estimates["mean_module.log_amplitude"].constraint)

    def test_median_flux_guess_strategy_uses_global_diagnostics(self):
        spec = ParameterSpec(
            name="mean_module.offset",
            role=ParameterRole.OFFSET,
            domain=ParameterDomain.FLUX,
            guess_strategy=GuessStrategy.MEDIAN_FLUX,
        )

        context = ParameterEstimationContext(
            is_multiband=False,
            global_diagnostics=LightcurveDiagnostics(
                median_flux=123.4,
            ),
        )

        builder = ParameterEstimateBuilder()
        estimate = builder.build_one(spec=spec, context=context)

        self.assertEqual(estimate.value, 123.4)
        self.assertIsNone(estimate.constraint)
        self.assertEqual(estimate.value_source, "median_flux")
        self.assertIsNone(estimate.constraint_source)

    def test_median_flux_guess_strategy_returns_none_when_unavailable(self):
        spec = ParameterSpec(
            name="mean_module.offset",
            role=ParameterRole.OFFSET,
            domain=ParameterDomain.FLUX,
            guess_strategy=GuessStrategy.MEDIAN_FLUX,
        )

        context = ParameterEstimationContext(
            is_multiband=False,
        )

        builder = ParameterEstimateBuilder()
        estimate = builder.build_one(spec=spec, context=context)

        self.assertIsNone(estimate.value)
        self.assertEqual(estimate.value_source, "median_flux")

    def test_robust_flux_range_constraint_uses_global_percentiles(self):
        spec = ParameterSpec(
            name="mean_module.offset",
            role=ParameterRole.OFFSET,
            domain=ParameterDomain.FLUX,
            constraint_strategy=ConstraintStrategy.ROBUST_FLUX_RANGE,
        )

        context = ParameterEstimationContext(
            is_multiband=False,
            global_diagnostics=LightcurveDiagnostics(
                flux_percentiles={
                    2.5: 10.0,
                    50.0: 55.0,
                    97.5: 100.0,
                },
            ),
        )

        builder = ParameterEstimateBuilder()
        estimate = builder.build_one(spec=spec, context=context)

        self.assertIsNone(estimate.value)
        self.assertEqual(estimate.constraint, (10.0, 100.0))
        self.assertEqual(estimate.constraint_source, "robust_flux_range")

    def test_robust_flux_range_constraint_returns_none_when_percentiles_missing(self):
        spec = ParameterSpec(
            name="mean_module.offset",
            role=ParameterRole.OFFSET,
            domain=ParameterDomain.FLUX,
            constraint_strategy=ConstraintStrategy.ROBUST_FLUX_RANGE,
        )

        context = ParameterEstimationContext(
            is_multiband=False,
            global_diagnostics=LightcurveDiagnostics(
                flux_percentiles={
                    50.0: 55.0,
                },
            ),
        )

        builder = ParameterEstimateBuilder()
        estimate = builder.build_one(spec=spec, context=context)

        self.assertIsNone(estimate.constraint)
        self.assertEqual(estimate.constraint_source, "robust_flux_range")

    def test_robust_flux_span_guess_strategy(self):
        spec = ParameterSpec(
            name="mean_module.log_amplitude",
            role=ParameterRole.AMPLITUDE,
            domain=ParameterDomain.FLUX,
            guess_strategy=GuessStrategy.ROBUST_FLUX_SPAN,
        )

        context = ParameterEstimationContext(
            is_multiband=False,
            global_diagnostics=LightcurveDiagnostics(
                flux_percentiles={
                    2.5: 10.0,
                    97.5: 100.0,
                },
            ),
        )

        builder = ParameterEstimateBuilder()
        estimate = builder.build_one(spec=spec, context=context)

        self.assertEqual(estimate.value, 90.0)
        self.assertEqual(
            estimate.value_source,
            "robust_flux_span",
        )

    def test_robust_flux_span_guess_strategy_returns_none_when_unavailable(self):
        spec = ParameterSpec(
            name="mean_module.log_amplitude",
            role=ParameterRole.AMPLITUDE,
            domain=ParameterDomain.FLUX,
            guess_strategy=GuessStrategy.ROBUST_FLUX_SPAN,
        )

        context = ParameterEstimationContext(
            is_multiband=False,
        )

        builder = ParameterEstimateBuilder()
        estimate = builder.build_one(spec=spec, context=context)

        self.assertIsNone(estimate.value)

    def test_offset_estimate_can_use_median_and_robust_range_together(self):
        spec = ParameterSpec(
            name="mean_module.offset",
            role=ParameterRole.OFFSET,
            domain=ParameterDomain.FLUX,
            guess_strategy=GuessStrategy.MEDIAN_FLUX,
            constraint_strategy=ConstraintStrategy.ROBUST_FLUX_RANGE,
        )

        context = ParameterEstimationContext(
            is_multiband=False,
            global_diagnostics=LightcurveDiagnostics(
                median_flux=55.0,
                flux_percentiles={
                    2.5: 10.0,
                    50.0: 55.0,
                    97.5: 100.0,
                },
            ),
        )

        builder = ParameterEstimateBuilder()
        estimate = builder.build_one(spec=spec, context=context)

        self.assertEqual(estimate.value, 55.0)
        self.assertEqual(estimate.constraint, (10.0, 100.0))
        self.assertEqual(estimate.value_source, "median_flux")
        self.assertEqual(estimate.constraint_source, "robust_flux_range")

    def test_robust_flux_interval_helper(self):
        context = ParameterEstimationContext(
            is_multiband=False,
            global_diagnostics=LightcurveDiagnostics(
                flux_percentiles={
                    2.5: 10.0,
                    50.0: 55.0,
                    97.5: 100.0,
                },
            ),
        )

        builder = ParameterEstimateBuilder()

        self.assertEqual(
            builder._estimate_robust_flux_interval(context),
            (10.0, 100.0),
        )

    def test_robust_flux_span_helper(self):
        context = ParameterEstimationContext(
            is_multiband=False,
            global_diagnostics=LightcurveDiagnostics(
                flux_percentiles={
                    2.5: 10.0,
                    97.5: 100.0,
                },
            ),
        )

        builder = ParameterEstimateBuilder()

        self.assertEqual(
            builder._estimate_robust_flux_span(context),
            90.0,
        )

    def test_robust_flux_span_helper_returns_none_when_unavailable(self):
        context = ParameterEstimationContext(
            is_multiband=False,
        )

        builder = ParameterEstimateBuilder()

        self.assertIsNone(
            builder._estimate_robust_flux_span(context)
        )

    def test_robust_positive_flux_span_constraint_strategy(self):
        spec = ParameterSpec(
            name="mean_module.log_amplitude",
            role=ParameterRole.AMPLITUDE,
            domain=ParameterDomain.FLUX,
            constraint_strategy=ConstraintStrategy.ROBUST_POSITIVE_FLUX_SPAN,
        )

        context = ParameterEstimationContext(
            is_multiband=False,
            global_diagnostics=LightcurveDiagnostics(
                flux_percentiles={
                    2.5: 10.0,
                    97.5: 100.0,
                },
            ),
        )

        builder = ParameterEstimateBuilder()
        estimate = builder.build_one(spec=spec, context=context)

        self.assertAlmostEqual(estimate.constraint[0], 9.0e-5)
        self.assertAlmostEqual(estimate.constraint[1], 450.0)
        self.assertEqual(
            estimate.constraint_source,
            "robust_positive_flux_span",
        )

    def test_robust_positive_flux_span_constraint_returns_none_for_zero_span(self):
        spec = ParameterSpec(
            name="mean_module.log_amplitude",
            role=ParameterRole.AMPLITUDE,
            domain=ParameterDomain.FLUX,
            constraint_strategy=ConstraintStrategy.ROBUST_POSITIVE_FLUX_SPAN,
        )

        context = ParameterEstimationContext(
            is_multiband=False,
            global_diagnostics=LightcurveDiagnostics(
                flux_percentiles={
                    2.5: 10.0,
                    97.5: 10.0,
                },
            ),
        )

        builder = ParameterEstimateBuilder()
        estimate = builder.build_one(spec=spec, context=context)

        self.assertIsNone(estimate.constraint)

    def test_default_guess_strategy_uses_spec_initial_value(self):
        spec = ParameterSpec(
            name="mean_module.log_tau",
            role=ParameterRole.SHAPE,
            domain=ParameterDomain.DIMENSIONLESS,
            scale=ParameterScale.LOG,
            initial_value=1.0,
            guess_strategy=GuessStrategy.DEFAULT,
        )

        context = ParameterEstimationContext(
            is_multiband=True,
        )

        estimate = ParameterEstimateBuilder().build_one(
            spec=spec,
            context=context,
        )

        self.assertEqual(estimate.value, 1.0)
        self.assertEqual(estimate.value_source, "default")

    def test_default_constraint_strategy_uses_spec_constraint(self):
        spec = ParameterSpec(
            name="mean_module.log_tau",
            role=ParameterRole.SHAPE,
            domain=ParameterDomain.DIMENSIONLESS,
            scale=ParameterScale.LOG,
            constraint=(1.0e-3, 1.0e3),
            constraint_strategy=ConstraintStrategy.DEFAULT,
        )

        context = ParameterEstimationContext(
            is_multiband=True,
        )

        estimate = ParameterEstimateBuilder().build_one(
            spec=spec,
            context=context,
        )

        self.assertEqual(estimate.constraint, (1.0e-3, 1.0e3))
        self.assertEqual(estimate.constraint_source, "default")

    def test_geometric_sampling_timescale_guess(self):
        spec = ParameterSpec(
            name="covar_module.lengthscale",
            role=ParameterRole.LENGTHSCALE,
            domain=ParameterDomain.TIME,
            scale=ParameterScale.LOG,
            guess_strategy=GuessStrategy.GEOMETRIC_SAMPLING_TIMESCALE,
        )

        context = ParameterEstimationContext(
            is_multiband=False,
            global_diagnostics=LightcurveDiagnostics(
                baseline_duration=1000.0,
                median_cadence=10.0,
            ),
        )

        estimate = ParameterEstimateBuilder().build_one(
            spec=spec,
            context=context,
        )

        self.assertAlmostEqual(
            estimate.value,
            100.0,
        )

    def test_geometric_sampling_timescale_requires_sampling_diagnostics(self):
        spec = ParameterSpec(
            name="covar_module.lengthscale",
            role=ParameterRole.LENGTHSCALE,
            domain=ParameterDomain.TIME,
            scale=ParameterScale.LOG,
            guess_strategy=GuessStrategy.GEOMETRIC_SAMPLING_TIMESCALE,
        )

        context = ParameterEstimationContext(
                is_multiband=False,
                )

        estimate = ParameterEstimateBuilder().build_one(
            spec=spec,
            context=context,
        )

        self.assertIsNone(estimate.value)

    def test_baseline_frequency_guess(self):
        spec = ParameterSpec(
            name="covar_module.mixture_means",
            role=ParameterRole.FREQUENCY,
            domain=ParameterDomain.FREQUENCY,
            scale=ParameterScale.LINEAR,
            guess_strategy=GuessStrategy.BASELINE_FREQUENCY,
        )

        context = ParameterEstimationContext(
            is_multiband=False,
            global_diagnostics=LightcurveDiagnostics(
                baseline_duration=1000.0,
            ),
        )

        estimate = ParameterEstimateBuilder().build_one(
            spec=spec,
            context=context,
        )

        self.assertAlmostEqual(
            estimate.value,
            0.001,
        )

    def test_baseline_frequency_requires_positive_baseline(self):
        spec = ParameterSpec(
            name="covar_module.mixture_means",
            role=ParameterRole.FREQUENCY,
            domain=ParameterDomain.FREQUENCY,
            scale=ParameterScale.LINEAR,
            guess_strategy=GuessStrategy.BASELINE_FREQUENCY,
        )

        context = ParameterEstimationContext(
            is_multiband=False,
            global_diagnostics=LightcurveDiagnostics(
                baseline_duration=0.0,
            ),
        )

        estimate = ParameterEstimateBuilder().build_one(
            spec=spec,
            context=context,
        )

        self.assertIsNone(estimate.value)

    def test_baseline_frequency_can_initialize_spectral_mixture_means(self):
        spec = ParameterSpec(
            name="covar_module.mixture_means",
            role=ParameterRole.FREQUENCY,
            domain=ParameterDomain.FREQUENCY,
            scale=ParameterScale.LINEAR,
            guess_strategy=GuessStrategy.BASELINE_FREQUENCY,
        )

        context = ParameterEstimationContext(
            is_multiband=False,
            global_diagnostics=LightcurveDiagnostics(
                baseline_duration=500.0,
            ),
        )

        estimate = ParameterEstimateBuilder().build_one(
            spec=spec,
            context=context,
        )

        self.assertAlmostEqual(
            estimate.value,
            1.0 / 500.0,
        )

    def test_consensus_frequency_from_frequency_diagnostics(self):
        spec = ParameterSpec(
            name="covar_module.mixture_means",
            role=ParameterRole.FREQUENCY,
            domain=ParameterDomain.FREQUENCY,
            scale=ParameterScale.LINEAR,
            guess_strategy=GuessStrategy.CONSENSUS_FREQUENCY,
        )

        context = ParameterEstimationContext(
            is_multiband=False,
            consensus_diagnostics=ConsensusDiagnostics(
                method="consensus",
                frequencies=[0.05],
            ),
        )

        estimate = ParameterEstimateBuilder().build_one(
            spec=spec,
            context=context,
        )

        self.assertAlmostEqual(estimate.value, 0.05)

    def test_consensus_frequency_from_period_diagnostics(self):
        spec = ParameterSpec(
            name="covar_module.mixture_means",
            role=ParameterRole.FREQUENCY,
            domain=ParameterDomain.FREQUENCY,
            scale=ParameterScale.LINEAR,
            guess_strategy=GuessStrategy.CONSENSUS_FREQUENCY,
        )

        context = ParameterEstimationContext(
            is_multiband=False,
            consensus_diagnostics=ConsensusDiagnostics(
                method="consensus",
                periods=[20.0],
            ),
        )

        estimate = ParameterEstimateBuilder().build_one(
            spec=spec,
            context=context,
        )

        self.assertAlmostEqual(estimate.value, 0.05)

    def test_consensus_frequency_preferred_over_baseline_frequency(self):
        spec = ParameterSpec(
            name="covar_module.mixture_means",
            role=ParameterRole.FREQUENCY,
            domain=ParameterDomain.FREQUENCY,
            scale=ParameterScale.LINEAR,
            guess_strategy=GuessStrategy.CONSENSUS_FREQUENCY,
        )

        context = ParameterEstimationContext(
            is_multiband=False,
            global_diagnostics=LightcurveDiagnostics(
                baseline_duration=1000.0,
            ),
            consensus_diagnostics=ConsensusDiagnostics(
                method="consensus",
                frequencies=[0.05],
            ),
        )

        estimate = ParameterEstimateBuilder().build_one(
            spec=spec,
            context=context,
        )

        self.assertAlmostEqual(
            estimate.value,
            0.05,
        )

    def test_consensus_frequency_returns_multiple_components(self):
        spec = ParameterSpec(
            name="covar_module.mixture_means",
            role=ParameterRole.FREQUENCY,
            domain=ParameterDomain.FREQUENCY,
            scale=ParameterScale.LINEAR,
            shape=(3,),
            guess_strategy=GuessStrategy.CONSENSUS_FREQUENCY,
        )

        context = ParameterEstimationContext(
            is_multiband=False,
            consensus_diagnostics=ConsensusDiagnostics(
                method="consensus",
                frequencies=[0.10, 0.05, 0.02],
            ),
        )

        estimate = ParameterEstimateBuilder().build_one(
            spec=spec,
            context=context,
        )

        self.assertEqual(
            estimate.value,
            [0.10, 0.05, 0.02],
        )

    def test_consensus_frequency_requires_enough_components(self):
        spec = ParameterSpec(
            name="covar_module.mixture_means",
            role=ParameterRole.FREQUENCY,
            domain=ParameterDomain.FREQUENCY,
            scale=ParameterScale.LINEAR,
            shape=(3,),
            guess_strategy=GuessStrategy.CONSENSUS_FREQUENCY,
        )

        context = ParameterEstimationContext(
            is_multiband=False,
            consensus_diagnostics=ConsensusDiagnostics(
                method="consensus",
                frequencies=[0.10, 0.05],
            ),
        )

        estimate = ParameterEstimateBuilder().build_one(
            spec=spec,
            context=context,
        )

        self.assertIsNone(estimate.value)

    def test_consensus_multicomp_period_returns_multiple_components(self):
        spec = ParameterSpec(
            name="periods",
            role=ParameterRole.PERIOD,
            domain=ParameterDomain.TIME,
            scale=ParameterScale.LINEAR,
            shape=(3,),
            guess_strategy=GuessStrategy.CONSENSUS_MULTICOMP_PERIOD,
        )

        context = ParameterEstimationContext(
            is_multiband=False,
            consensus_diagnostics=ConsensusDiagnostics(
                method="consensus_multicomp",
                periods=[300.0, 600.0, 1200.0],
            ),
        )

        estimate = ParameterEstimateBuilder().build_one(
            spec=spec,
            context=context,
        )

        self.assertEqual(
            estimate.value,
            [300.0, 600.0, 1200.0],
        )

    def test_consensus_multicomp_period_requires_enough_components(self):
        spec = ParameterSpec(
            name="periods",
            role=ParameterRole.PERIOD,
            domain=ParameterDomain.TIME,
            scale=ParameterScale.LINEAR,
            shape=(3,),
            guess_strategy=GuessStrategy.CONSENSUS_MULTICOMP_PERIOD,
        )

        context = ParameterEstimationContext(
            is_multiband=False,
            consensus_diagnostics=ConsensusDiagnostics(
                method="consensus_multicomp",
                periods=[300.0, 600.0],
            ),
        )

        estimate = ParameterEstimateBuilder().build_one(
            spec=spec,
            context=context,
        )

        self.assertIsNone(estimate.value)


if __name__ == "__main__":
    unittest.main()
