import math
import unittest

import torch

from pgmuvi.parameter_builders import ParameterEstimateBuilder
from pgmuvi.parameter_context import ParameterEstimationContext
from pgmuvi.parameter_specs import (
    ConstraintStrategy,
    GuessStrategy,
    ParameterDomain,
    ParameterRole,
    ParameterScale,
    ParameterSpec,
)
from pgmuvi.wavelength_estimation import build_wavelength_estimation_context


class TestWavelengthRangeParameterBuilder(unittest.TestCase):
    @staticmethod
    def _context():
        diagnostics, band_diagnostics = build_wavelength_estimation_context(
            wavelengths=[1.0] * 3 + [2.0] * 3 + [5.0] * 3,
            fluxes=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            band_labels=["a"] * 3 + ["b"] * 3 + ["c"] * 3,
        )
        return ParameterEstimationContext(
            is_multiband=True,
            band_diagnostics=band_diagnostics,
            wavelength_diagnostics=diagnostics,
        )

    @staticmethod
    def _spec(shape=None):
        return ParameterSpec(
            name="covar_module.wavelength_kernel.lengthscale",
            role=ParameterRole.WAVELENGTH_SCALE,
            domain=ParameterDomain.WAVELENGTH,
            scale=ParameterScale.LINEAR,
            shape=shape,
            guess_strategy=GuessStrategy.WAVELENGTH_RANGE,
            constraint_strategy=ConstraintStrategy.WAVELENGTH_RANGE,
        )

    def test_builds_scalar_value_and_constraint_from_wavelength_context(self):
        estimate = ParameterEstimateBuilder().build_one(
            self._spec(),
            self._context(),
        )

        self.assertAlmostEqual(estimate.value, math.sqrt(8.0))
        self.assertEqual(estimate.constraint, (0.25, 20.0))
        self.assertEqual(estimate.value_source, "wavelength_range")
        self.assertEqual(estimate.constraint_source, "wavelength_range")
        self.assertIsNone(estimate.metadata["value_reason"])
        self.assertEqual(estimate.diagnostics["coordinate_space"], "raw_input")
        self.assertEqual(estimate.diagnostics["n_usable_bands"], 3)
        self.assertAlmostEqual(estimate.diagnostics["wavelength_span"], 4.0)
        self.assertAlmostEqual(
            estimate.diagnostics["median_adjacent_spacing"],
            2.0,
        )
        self.assertAlmostEqual(estimate.diagnostics["largest_gap"], 3.0)

    def test_expands_value_and_bounds_to_declared_shape(self):
        estimate = ParameterEstimateBuilder().build_one(
            self._spec(shape=(2, 1, 1)),
            self._context(),
        )

        self.assertEqual(tuple(estimate.value.shape), (2, 1, 1))
        self.assertTrue(
            torch.allclose(
                estimate.value,
                torch.full((2, 1, 1), math.sqrt(8.0)),
            )
        )
        lower, upper = estimate.constraint
        self.assertEqual(tuple(lower.shape), (2, 1, 1))
        self.assertEqual(tuple(upper.shape), (2, 1, 1))
        self.assertTrue(torch.all(lower == 0.25))
        self.assertTrue(torch.all(upper == 20.0))

    def test_reports_unavailable_wavelength_context(self):
        estimate = ParameterEstimateBuilder().build_one(
            self._spec(),
            ParameterEstimationContext(is_multiband=True),
        )

        self.assertIsNone(estimate.value)
        self.assertIsNone(estimate.constraint)
        self.assertEqual(
            estimate.metadata["value_reason"],
            "wavelength_diagnostics_unavailable",
        )
        self.assertEqual(estimate.diagnostics, {})

    def test_reports_context_without_two_usable_wavelengths(self):
        diagnostics, band_diagnostics = build_wavelength_estimation_context(
            wavelengths=[1.0, 1.0, 1.0],
            fluxes=[1.0, 2.0, 3.0],
        )
        context = ParameterEstimationContext(
            is_multiband=True,
            band_diagnostics=band_diagnostics,
            wavelength_diagnostics=diagnostics,
        )

        estimate = ParameterEstimateBuilder().build_one(
            self._spec(),
            context,
        )

        self.assertIsNone(estimate.value)
        self.assertIsNone(estimate.constraint)
        self.assertEqual(
            estimate.metadata["value_reason"],
            "wavelength_lengthscale_recommendation_unavailable",
        )
        self.assertEqual(estimate.diagnostics["n_usable_bands"], 1)


if __name__ == "__main__":
    unittest.main()
