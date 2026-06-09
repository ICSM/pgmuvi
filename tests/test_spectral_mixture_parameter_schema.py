import unittest

from pgmuvi.gps import spectral_mixture_parameter_schema
from pgmuvi.parameter_specs import (
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

        self.assertIs(scales.role, ParameterRole.LENGTHSCALE)
        self.assertIs(scales.domain, ParameterDomain.FREQUENCY)
        self.assertIs(scales.scale, ParameterScale.LOG)
        self.assertEqual(scales.shape, (3,))

        self.assertIs(weights.role, ParameterRole.WEIGHT)
        self.assertIs(weights.domain, ParameterDomain.VARIANCE)
        self.assertIs(weights.scale, ParameterScale.LOG)
        self.assertEqual(weights.shape, (3,))


if __name__ == "__main__":
    unittest.main()