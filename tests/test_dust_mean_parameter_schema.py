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
    ParameterEstimationContext,
    WavelengthMeanEstimationDiagnostics,
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
            GuessStrategy.WAVELENGTH_MEAN,
        )

        self.assertEqual(
            schema["mean_module.offset"].constraint_strategy,
            ConstraintStrategy.WAVELENGTH_MEAN,
        )

        self.assertEqual(
            schema["mean_module.log_amplitude"].guess_strategy,
            GuessStrategy.WAVELENGTH_MEAN,
        )

        self.assertEqual(
            schema["mean_module.log_amplitude"].constraint_strategy,
            ConstraintStrategy.WAVELENGTH_MEAN,
        )

        self.assertEqual(
            schema["mean_module.log_tau"].guess_strategy,
            GuessStrategy.WAVELENGTH_MEAN,
        )

        self.assertEqual(
            schema["mean_module.log_tau"].constraint_strategy,
            ConstraintStrategy.WAVELENGTH_MEAN,
        )

        self.assertEqual(
            schema["mean_module.log_alpha"].guess_strategy,
            GuessStrategy.WAVELENGTH_MEAN,
        )

        self.assertEqual(
            schema["mean_module.log_alpha"].constraint_strategy,
            ConstraintStrategy.WAVELENGTH_MEAN,
        )

    def test_dust_mean_schema_builds_estimates_from_diagnostics(self):
        schema = DustMean().parameter_schema()

        context = ParameterEstimationContext(
            is_multiband=True,
            wavelength_mean_diagnostics=WavelengthMeanEstimationDiagnostics(
                available=True,
                n_usable_bands=4,
                recommendations={
                    "2DDustMean": {
                        "available": True,
                        "coordinate_basis": (
                            "physical_wavelength_and_model_flux"
                        ),
                        "initial_values": {
                            "mean_module.offset": 0.2,
                            "mean_module.log_amplitude": 3.0,
                            "mean_module.log_tau": 1.4,
                            "mean_module.log_alpha": 1.8,
                        },
                        "constraints": {
                            "mean_module.offset": [-1.0, 4.0],
                            "mean_module.log_amplitude": [1.0e-4, 10.0],
                            "mean_module.log_tau": [1.0e-3, 1.0e3],
                            "mean_module.log_alpha": [0.1, 10.0],
                        },
                        "fit_rmse": 0.01,
                        "reason": None,
                    }
                },
            ),
        )

        estimates = ParameterEstimateBuilder().build(
            schema=schema,
            context=context,
        )

        self.assertEqual(estimates["mean_module.offset"].value, 0.2)
        self.assertEqual(
            estimates["mean_module.offset"].constraint,
            (-1.0, 4.0),
        )
        self.assertEqual(
            estimates["mean_module.log_amplitude"].value,
            3.0,
        )
        self.assertEqual(
            estimates["mean_module.log_tau"].value,
            1.4,
        )
        self.assertEqual(
            estimates["mean_module.log_alpha"].value,
            1.8,
        )
        self.assertEqual(
            estimates["mean_module.log_amplitude"].constraint,
            (1.0e-4, 10.0),
        )
        self.assertEqual(
            estimates["mean_module.log_tau"].constraint,
            (1.0e-3, 1.0e3),
        )
        self.assertEqual(
            estimates["mean_module.log_alpha"].constraint,
            (0.1, 10.0),
        )


if __name__ == "__main__":
    unittest.main()
