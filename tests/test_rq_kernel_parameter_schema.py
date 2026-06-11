import unittest

from pgmuvi.gps import rq_kernel_parameter_schema
from pgmuvi.parameter_specs import (
    ConstraintStrategy,
    GuessStrategy,
    ParameterDomain,
    ParameterRole,
    ParameterScale,
)


class TestRQKernelParameterSchema(unittest.TestCase):

    def test_time_domain_schema(self):
        schema = rq_kernel_parameter_schema(
            prefix="time_kernel",
            domain=ParameterDomain.TIME,
        )

        self.assertEqual(
            schema.names(),
            [
                "time_kernel.lengthscale",
                "time_kernel.alpha",
            ],
        )

        lengthscale = schema["time_kernel.lengthscale"]
        alpha = schema["time_kernel.alpha"]

        self.assertIs(lengthscale.role, ParameterRole.LENGTHSCALE)
        self.assertIs(lengthscale.domain, ParameterDomain.TIME)
        self.assertIs(lengthscale.scale, ParameterScale.LOG)

        self.assertIs(alpha.role, ParameterRole.SHAPE)
        self.assertIs(alpha.domain, ParameterDomain.DIMENSIONLESS)
        self.assertIs(alpha.scale, ParameterScale.LOG)
        self.assertEqual(alpha.initial_value, 1.0)
        self.assertEqual(alpha.constraint, (1.0e-3, 1.0e3))
        self.assertEqual(alpha.guess_strategy, GuessStrategy.DEFAULT)
        self.assertEqual(
            alpha.constraint_strategy,
            ConstraintStrategy.DEFAULT,
        )

    def test_wavelength_domain_schema(self):
        schema = rq_kernel_parameter_schema(
            prefix="wavelength_kernel",
            domain=ParameterDomain.WAVELENGTH,
        )

        lengthscale = schema["wavelength_kernel.lengthscale"]

        self.assertIs(lengthscale.domain, ParameterDomain.WAVELENGTH)

    def test_accepts_empty_prefix(self):
        schema = rq_kernel_parameter_schema(
            prefix="",
            domain=ParameterDomain.TIME,
        )

        self.assertEqual(
            schema.names(),
            [
                "lengthscale",
                "alpha",
            ],
        )

    def test_rejects_invalid_domain(self):
        with self.assertRaisesRegex(ValueError, "requires domain"):
            rq_kernel_parameter_schema(
                domain=ParameterDomain.FLUX,
            )


if __name__ == "__main__":
    unittest.main()
