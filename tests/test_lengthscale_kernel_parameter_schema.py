import unittest

from pgmuvi.gps import lengthscale_kernel_parameter_schema
from pgmuvi.parameter_specs import (
    ConstraintStrategy,
    GuessStrategy,
    ParameterDomain,
    ParameterRole,
    ParameterScale,
)


class TestLengthscaleKernelParameterSchema(unittest.TestCase):

    def test_time_lengthscale_schema(self):
        schema = lengthscale_kernel_parameter_schema(
            prefix="time_kernel",
            domain=ParameterDomain.TIME,
        )

        self.assertEqual(schema.names(), ["time_kernel.lengthscale"])

        lengthscale = schema["time_kernel.lengthscale"]

        self.assertIs(lengthscale.role, ParameterRole.LENGTHSCALE)
        self.assertIs(lengthscale.domain, ParameterDomain.TIME)
        self.assertIs(lengthscale.scale, ParameterScale.LOG)
        self.assertEqual(lengthscale.initial_value, 1.0)
        self.assertEqual(lengthscale.constraint, (1.0e-3, 1.0e3))
        self.assertEqual(
                lengthscale.guess_strategy,
                GuessStrategy.GEOMETRIC_SAMPLING_TIMESCALE
                )
        self.assertEqual(
            lengthscale.constraint_strategy,
            ConstraintStrategy.DEFAULT,
        )

    def test_wavelength_lengthscale_schema(self):
        schema = lengthscale_kernel_parameter_schema(
            prefix="wavelength_kernel",
            domain=ParameterDomain.WAVELENGTH,
        )

        self.assertEqual(schema.names(), ["wavelength_kernel.lengthscale"])

        lengthscale = schema["wavelength_kernel.lengthscale"]

        self.assertIs(lengthscale.role, ParameterRole.LENGTHSCALE)
        self.assertIs(lengthscale.domain, ParameterDomain.WAVELENGTH)
        self.assertIs(lengthscale.scale, ParameterScale.LOG)

    def test_accepts_empty_prefix(self):
        schema = lengthscale_kernel_parameter_schema(
            prefix="",
            domain=ParameterDomain.TIME,
        )

        self.assertEqual(schema.names(), ["lengthscale"])

    def test_rejects_invalid_domain(self):
        with self.assertRaisesRegex(ValueError, "requires domain"):
            lengthscale_kernel_parameter_schema(
                domain=ParameterDomain.FLUX,
            )


if __name__ == "__main__":
    unittest.main()
