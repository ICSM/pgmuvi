import unittest

from pgmuvi.gps import DustMean
from pgmuvi.parameter_specs import (
    ConstraintStrategy,
    GuessStrategy,
    ParameterDomain,
    ParameterRole,
    ParameterScale,
)
from pgmuvi.parameter_builders import ParameterEstimateBuilder
from pgmuvi.parameter_context import (
    LightcurveDiagnostics,
    ParameterEstimationContext,
)


class TestDustMeanParameterSchema(unittest.TestCase):
    def test_dust_mean_parameter_schema_names(self):
        schema = DustMean().parameter_schema()

        self.assertEqual(
            schema.names(),
            [
                "mean_module.offset",
                "mean_module.log_amplitude",
                "mean_module.log_tau",
                "mean_module.log_alpha",
            ],
        )

    def test_dust_mean_parameter_schema_semantics(self):
        schema = DustMean().parameter_schema()

        offset = schema["mean_module.offset"]
        amplitude = schema["mean_module.log_amplitude"]
        tau = schema["mean_module.log_tau"]
        alpha = schema["mean_module.log_alpha"]

        self.assertIs(offset.role, ParameterRole.OFFSET)
        self.assertIs(offset.domain, ParameterDomain.FLUX)
        self.assertIs(offset.scale, ParameterScale.LINEAR)

        self.assertIs(amplitude.role, ParameterRole.AMPLITUDE)
        self.assertIs(amplitude.domain, ParameterDomain.FLUX)
        self.assertIs(amplitude.scale, ParameterScale.LOG)

        self.assertIs(tau.role, ParameterRole.SHAPE)
        self.assertIs(tau.domain, ParameterDomain.DIMENSIONLESS)
        self.assertIs(tau.scale, ParameterScale.LOG)

        self.assertIs(alpha.role, ParameterRole.SHAPE)
        self.assertIs(alpha.domain, ParameterDomain.DIMENSIONLESS)
        self.assertIs(alpha.scale, ParameterScale.LOG)

    def test_dust_mean_parameter_schema_accepts_empty_prefix(self):
        schema = DustMean().parameter_schema(prefix="")

        self.assertEqual(
            schema.names(),
            [
                "offset",
                "log_amplitude",
                "log_tau",
                "log_alpha",
            ],
        )

    def test_dust_mean_schema_estimation_strategies(self):
        schema = DustMean().parameter_schema()

        self.assertEqual(
            schema["mean_module.offset"].guess_strategy,
            GuessStrategy.MEDIAN_FLUX,
        )

        self.assertEqual(
            schema["mean_module.offset"].constraint_strategy,
            ConstraintStrategy.ROBUST_FLUX_RANGE,
        )

        self.assertEqual(
            schema["mean_module.log_amplitude"].guess_strategy,
            GuessStrategy.ROBUST_FLUX_SPAN,
        )

        self.assertEqual(
            schema["mean_module.log_amplitude"].constraint_strategy,
            ConstraintStrategy.ROBUST_POSITIVE_FLUX_SPAN,
        )

        self.assertEqual(
            schema["mean_module.log_tau"].guess_strategy,
            GuessStrategy.DEFAULT,
        )

        self.assertEqual(
            schema["mean_module.log_tau"].constraint_strategy,
            ConstraintStrategy.DEFAULT,
        )

        self.assertEqual(
            schema["mean_module.log_alpha"].guess_strategy,
            GuessStrategy.DEFAULT,
        )

        self.assertEqual(
            schema["mean_module.log_alpha"].constraint_strategy,
            ConstraintStrategy.DEFAULT,
        )

    def test_dust_mean_schema_builds_estimates_from_diagnostics(self):
        schema = DustMean().parameter_schema()

        context = ParameterEstimationContext(
            is_multiband=True,
            global_diagnostics=LightcurveDiagnostics(
                median_flux=55.0,
                flux_percentiles={
                    2.5: 10.0,
                    50.0: 55.0,
                    97.5: 100.0,
                },
            ),
        )

        estimates = ParameterEstimateBuilder().build(
            schema=schema,
            context=context,
        )

        offset = estimates["mean_module.offset"]
        amplitude = estimates["mean_module.log_amplitude"]

        tau = estimates["mean_module.log_tau"]
        alpha = estimates["mean_module.log_alpha"]

        self.assertEqual(offset.value, 55.0)
        self.assertEqual(offset.constraint, (10.0, 100.0))

        self.assertEqual(amplitude.value, 90.0)

        self.assertAlmostEqual(
            amplitude.constraint[0],
            9.0e-5,
        )
        self.assertAlmostEqual(
            amplitude.constraint[1],
            450.0,
        )

        self.assertEqual(tau.value, 1.0)
        self.assertEqual(tau.constraint, (1.0e-3, 1.0e3))

        self.assertEqual(alpha.value, 1.7)
        self.assertEqual(alpha.constraint, (0.1, 10.0))
