import unittest

from pgmuvi.gps import spectral_mixture_parameter_schema
from pgmuvi.parameter_specs import (
    ConstraintStrategy,
    GuessStrategy,
    ParameterDomain,
    ParameterRole,
    ParameterScale,
)


class TestSpectralMixtureParameterSchema(unittest.TestCase):

    def test_names(self):
        schema = spectral_mixture_parameter_schema()

        self.assertEqual(
            schema.names(),
            [
                "covar_module.mixture_means",
                "covar_module.mixture_scales",
                "covar_module.mixture_weights",
            ],
        )

    def test_semantics(self):
        schema = spectral_mixture_parameter_schema(num_mixtures=3)

        means = schema["covar_module.mixture_means"]
        scales = schema["covar_module.mixture_scales"]
        weights = schema["covar_module.mixture_weights"]

        self.assertIs(means.role, ParameterRole.FREQUENCY)
        self.assertIs(means.domain, ParameterDomain.FREQUENCY)
        self.assertIs(means.scale, ParameterScale.LINEAR)
        self.assertEqual(means.shape, (3,))
        self.assertIsNone(means.initial_value)
        self.assertEqual(means.constraint, (1.0e-6, 1.0e6))
        self.assertEqual(
            means.guess_strategy,
            GuessStrategy.BASELINE_FREQUENCY,
        )
        self.assertEqual(
            means.constraint_strategy,
            ConstraintStrategy.DEFAULT,
        )

        self.assertIs(scales.role, ParameterRole.LENGTHSCALE)
        self.assertIs(scales.domain, ParameterDomain.FREQUENCY)
        self.assertIs(scales.scale, ParameterScale.LOG)
        self.assertEqual(scales.shape, (3,))
        self.assertEqual(scales.initial_value, 1.0)
        self.assertEqual(scales.constraint, (1.0e-6, 1.0e6))
        self.assertEqual(scales.guess_strategy, GuessStrategy.DEFAULT)
        self.assertEqual(
            scales.constraint_strategy,
            ConstraintStrategy.DEFAULT,
        )

        self.assertIs(weights.role, ParameterRole.WEIGHT)
        self.assertIs(weights.domain, ParameterDomain.VARIANCE)
        self.assertIs(weights.scale, ParameterScale.LOG)
        self.assertEqual(weights.shape, (3,))
        self.assertEqual(weights.initial_value, 1.0)
        self.assertEqual(weights.constraint, (1.0e-12, 1.0e12))
        self.assertEqual(weights.guess_strategy, GuessStrategy.DEFAULT)
        self.assertEqual(
            weights.constraint_strategy,
            ConstraintStrategy.DEFAULT,
        )


if __name__ == "__main__":
    unittest.main()
