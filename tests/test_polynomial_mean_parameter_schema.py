import unittest

from pgmuvi.gps import CustomLinearConstantMean, CustomQuadConstantMean
from pgmuvi.parameter_specs import ParameterDomain, ParameterRole, ParameterScale


class TestPolynomialMeanParameterSchema(unittest.TestCase):
    def test_custom_linear_constant_mean_schema_names(self):
        schema = CustomLinearConstantMean().parameter_schema()

        self.assertEqual(
            schema.names(),
            [
                "mean_module.wavelength_slope",
                "mean_module.bias",
            ],
        )

    def test_custom_linear_constant_mean_schema_semantics(self):
        schema = CustomLinearConstantMean().parameter_schema()

        slope = schema["mean_module.wavelength_slope"]
        bias = schema["mean_module.bias"]

        self.assertIs(slope.role, ParameterRole.WAVELENGTH_SCALE)
        self.assertIs(slope.domain, ParameterDomain.FLUX)
        self.assertIs(slope.scale, ParameterScale.LINEAR)

        self.assertIs(bias.role, ParameterRole.OFFSET)
        self.assertIs(bias.domain, ParameterDomain.FLUX)
        self.assertIs(bias.scale, ParameterScale.LINEAR)

    def test_custom_quad_constant_mean_schema_names(self):
        schema = CustomQuadConstantMean().parameter_schema()

        self.assertEqual(
            schema.names(),
            [
                "mean_module.weights",
                "mean_module.bias",
            ],
        )

    def test_custom_quad_constant_mean_schema_semantics(self):
        schema = CustomQuadConstantMean().parameter_schema()

        weights = schema["mean_module.weights"]
        bias = schema["mean_module.bias"]

        self.assertIs(weights.role, ParameterRole.WAVELENGTH_SCALE)
        self.assertIs(weights.domain, ParameterDomain.FLUX)
        self.assertIs(weights.scale, ParameterScale.LINEAR)
        self.assertEqual(weights.shape, (2,))

        self.assertIs(bias.role, ParameterRole.OFFSET)
        self.assertIs(bias.domain, ParameterDomain.FLUX)
        self.assertIs(bias.scale, ParameterScale.LINEAR)

    def test_polynomial_mean_schemas_accept_empty_prefix(self):
        linear = CustomLinearConstantMean().parameter_schema(prefix="")
        quadratic = CustomQuadConstantMean().parameter_schema(prefix="")

        self.assertEqual(linear.names(), ["wavelength_slope", "bias"])
        self.assertEqual(quadratic.names(), ["weights", "bias"])
