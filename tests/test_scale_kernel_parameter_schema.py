import unittest

from pgmuvi.gps import scale_kernel_parameter_schema
from pgmuvi.parameter_specs import (
    ParameterDomain,
    ParameterRole,
    ParameterScale,
)


class TestScaleKernelParameterSchema(unittest.TestCase):

    def test_names(self):
        schema = scale_kernel_parameter_schema()

        self.assertEqual(
            schema.names(),
            ["covar_module.outputscale"],
        )

    def test_semantics(self):
        schema = scale_kernel_parameter_schema()

        outputscale = schema["covar_module.outputscale"]

        self.assertIs(outputscale.role, ParameterRole.WEIGHT)
        self.assertIs(outputscale.domain, ParameterDomain.VARIANCE)
        self.assertIs(outputscale.scale, ParameterScale.LOG)

    def test_accepts_empty_prefix(self):
        schema = scale_kernel_parameter_schema(prefix="")

        self.assertEqual(schema.names(), ["outputscale"])


if __name__ == "__main__":
    unittest.main()