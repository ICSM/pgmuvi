import unittest

from pgmuvi.gps import scale_kernel_parameter_schema
from pgmuvi.parameter_specs import (
    ConstraintStrategy,
    GuessStrategy,
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
        self.assertEqual(outputscale.initial_value, 1.0)
        self.assertEqual(outputscale.constraint, (1.0e-6, 1.0e6))
        self.assertEqual(outputscale.guess_strategy, GuessStrategy.DEFAULT)
        self.assertEqual(
            outputscale.constraint_strategy,
            ConstraintStrategy.DEFAULT,
        )

    def test_accepts_empty_prefix(self):
        schema = scale_kernel_parameter_schema(prefix="")

        self.assertEqual(schema.names(), ["outputscale"])


if __name__ == "__main__":
    unittest.main()
