import numpy as np
import unittest

from pgmuvi.parameter_specs import (
    ConstraintStrategy,
    GuessStrategy,
    ParameterDomain,
    ParameterRole,
    ParameterScale,
    ParameterSpec,
    ParameterSpecCollection,
)


class TestParameterSpecs(unittest.TestCase):
    def test_parameter_spec_accepts_physical_space_values(self):
        spec = ParameterSpec(
            name="mean_module.log_tau",
            role=ParameterRole.TIMESCALE,
            domain=ParameterDomain.TIME,
            scale=ParameterScale.LOG,
            initial_value=250.0,
            constraint=(50.0, 2000.0),
            units="days",
            description="Characteristic variability timescale in physical time units.",
        )

        spec.validate()

        self.assertEqual(spec.initial_value, 250.0)
        self.assertEqual(spec.constraint, (50.0, 2000.0))
        self.assertIs(spec.scale, ParameterScale.LOG)
        self.assertIsNotNone(spec.description)

    def test_parameter_spec_rejects_initial_value_outside_constraint(self):
        spec = ParameterSpec(
            name="mean_module.log_amplitude",
            role=ParameterRole.AMPLITUDE,
            domain=ParameterDomain.FLUX,
            scale=ParameterScale.LOG,
            initial_value=7.3,
            constraint=(10.0, 100.0),
        )

        with self.assertRaisesRegex(ValueError, "outside its constraint"):
            spec.validate()

    def test_parameter_spec_rejects_invalid_constraint(self):
        spec = ParameterSpec(
            name="bad_parameter",
            role=ParameterRole.OTHER,
            domain=ParameterDomain.OTHER,
            initial_value=1.0,
            constraint=(2.0, 1.0),
        )

        with self.assertRaisesRegex(ValueError, "lower bound"):
            spec.validate()

    def test_parameter_spec_validates_array_shape(self):
        spec = ParameterSpec(
            name="covar_module.mixture_means",
            role=ParameterRole.FREQUENCY,
            domain=ParameterDomain.FREQUENCY,
            initial_value=np.array([0.01, 0.02, 0.03]),
            constraint=(
                np.array([0.005, 0.005, 0.005]),
                np.array([0.05, 0.05, 0.05]),
            ),
            shape=(3,),
        )

        spec.validate()

    def test_parameter_spec_rejects_wrong_array_shape(self):
        spec = ParameterSpec(
            name="covar_module.mixture_means",
            role=ParameterRole.FREQUENCY,
            domain=ParameterDomain.FREQUENCY,
            initial_value=np.array([0.01, 0.02]),
            shape=(3,),
        )

        with self.assertRaisesRegex(ValueError, "Shape mismatch"):
            spec.validate()

    def test_parameter_spec_collection_rejects_duplicate_names(self):
        spec = ParameterSpec(
            name="same",
            role=ParameterRole.OTHER,
            domain=ParameterDomain.OTHER,
        )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            ParameterSpecCollection([spec, spec])

    def test_parameter_spec_collection_lookup_and_add(self):
        collection = ParameterSpecCollection()

        spec = ParameterSpec(
            name="mean_module.offset",
            role=ParameterRole.OFFSET,
            domain=ParameterDomain.FLUX,
            initial_value=100.0,
            constraint=(-1000.0, 1000.0),
        )

        collection.add(spec)

        self.assertIn("mean_module.offset", collection)
        self.assertIs(collection["mean_module.offset"], spec)
        self.assertEqual(collection.names(), ["mean_module.offset"])

    def test_parameter_spec_accepts_strategy_metadata(self):
        spec = ParameterSpec(
            name="mean_module.log_tau",
            role=ParameterRole.TIMESCALE,
            domain=ParameterDomain.TIME,
            scale=ParameterScale.LOG,
            guess_strategy=GuessStrategy.VARIABILITY_TIMESCALE,
            constraint_strategy=ConstraintStrategy.VARIABILITY_TIMESCALE,
        )

        self.assertIs(spec.guess_strategy, GuessStrategy.VARIABILITY_TIMESCALE)
        self.assertIs(
            spec.constraint_strategy,
            ConstraintStrategy.VARIABILITY_TIMESCALE,
        )

    def test_parameter_spec_supports_fixed_parameters(self):
        spec = ParameterSpec(
            name="mean_module.log_alpha",
            role=ParameterRole.SHAPE,
            domain=ParameterDomain.DIMENSIONLESS,
            scale=ParameterScale.LOG,
            required=True,
            trainable=False,
        )

        self.assertIs(spec.required, True)
        self.assertIs(spec.trainable, False)
