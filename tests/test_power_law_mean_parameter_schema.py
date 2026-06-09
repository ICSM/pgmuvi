import unittest

from pgmuvi.gps import PowerLawMean
from pgmuvi.parameter_specs import ParameterDomain, ParameterRole, ParameterScale


class TestPowerLawMeanParameterSchema(unittest.TestCase):
    def test_power_law_mean_parameter_schema_names(self):
        schema = PowerLawMean().parameter_schema()

        self.assertEqual(
            schema.names(),
            [
                "mean_module.offset",
                "mean_module.weight",
                "mean_module.exponent",
            ],
        )

    def test_power_law_mean_parameter_schema_semantics(self):
        schema = PowerLawMean().parameter_schema()

        offset = schema["mean_module.offset"]
        weight = schema["mean_module.weight"]
        exponent = schema["mean_module.exponent"]

        self.assertIs(offset.role, ParameterRole.OFFSET)
        self.assertIs(offset.domain, ParameterDomain.FLUX)
        self.assertIs(offset.scale, ParameterScale.LINEAR)

        self.assertIs(weight.role, ParameterRole.AMPLITUDE)
        self.assertIs(weight.domain, ParameterDomain.FLUX)
        self.assertIs(weight.scale, ParameterScale.LINEAR)

        self.assertIs(exponent.role, ParameterRole.SHAPE)
        self.assertIs(exponent.domain, ParameterDomain.DIMENSIONLESS)
        self.assertIs(exponent.scale, ParameterScale.LINEAR)

    def test_power_law_mean_parameter_schema_accepts_empty_prefix(self):
        schema = PowerLawMean().parameter_schema(prefix="")

        self.assertEqual(
            schema.names(),
            [
                "offset",
                "weight",
                "exponent",
            ],
        )
