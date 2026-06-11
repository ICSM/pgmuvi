import unittest

import torch

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.parameter_context import (
    LightcurveDiagnostics,
    ParameterEstimationContext,
)


class TestLightcurveParameterEstimationContext(unittest.TestCase):

    def test_build_parameter_estimation_context_for_1d_lightcurve(self):
        lc = Lightcurve(
            torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
            torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]),
        )

        context = lc._build_parameter_estimation_context()

        self.assertIsInstance(context, ParameterEstimationContext)
        self.assertIsInstance(context.global_diagnostics, LightcurveDiagnostics)

        self.assertFalse(context.is_multiband)
        self.assertEqual(context.global_diagnostics.n_points, 5)
        self.assertAlmostEqual(context.global_diagnostics.median_flux, 30.0)

        self.assertAlmostEqual(
            context.global_diagnostics.flux_percentiles[50.0],
            30.0,
        )
        self.assertAlmostEqual(
            context.global_diagnostics.flux_percentiles[2.5],
            11.0,
        )
        self.assertAlmostEqual(
            context.global_diagnostics.flux_percentiles[97.5],
            49.0,
        )

    def test_build_parameter_estimation_context_for_2d_lightcurve(self):
        xdata = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 2.0],
                [1.0, 2.0],
            ]
        )
        ydata = torch.tensor([10.0, 20.0, 100.0, 200.0])

        lc = Lightcurve(xdata, ydata)

        context = lc._build_parameter_estimation_context()

        self.assertIsInstance(context, ParameterEstimationContext)
        self.assertTrue(context.is_multiband)
        self.assertEqual(context.global_diagnostics.n_points, 4)
        self.assertAlmostEqual(context.global_diagnostics.median_flux, 60.0)

    def test_build_parameter_estimation_context_ignores_nonfinite_flux_values(self):
        lc = Lightcurve(
            torch.tensor([0.0, 1.0, 2.0]),
            torch.tensor([10.0, float("nan"), 30.0]),
        )

        # Lightcurve currently rejects NaNs on construction, so this test is
        # only useful if non-finite values can exist internally later.
        # Do not add this test if constructor validation rejects the input.