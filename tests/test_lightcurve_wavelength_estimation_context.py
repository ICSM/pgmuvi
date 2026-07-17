import unittest

import torch

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.wavelength_estimation import WAVELENGTH_ESTIMATION_SCHEMA_VERSION


class TestLightcurveWavelengthEstimationContext(unittest.TestCase):
    def test_multiband_context_contains_wavelength_and_band_diagnostics(self):
        xdata = torch.tensor(
            [
                [0.0, 0.5],
                [1.0, 0.5],
                [2.0, 0.5],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [0.0, 2.0],
                [1.0, 2.0],
                [2.0, 2.0],
            ]
        )
        ydata = torch.tensor(
            [8.0, 10.0, 12.0, 18.0, 20.0, 22.0, 28.0, 30.0, 32.0]
        )
        yerr = torch.full((9,), 0.5)
        bands = ["g"] * 3 + ["r"] * 3 + ["j"] * 3
        lc = Lightcurve(xdata, ydata, yerr=yerr, band=bands)

        context = lc._build_parameter_estimation_context()

        self.assertEqual(
            context.wavelength_diagnostics.schema_version,
            WAVELENGTH_ESTIMATION_SCHEMA_VERSION,
        )
        self.assertTrue(context.wavelength_diagnostics.available)
        self.assertEqual(context.wavelength_diagnostics.n_usable_bands, 3)
        self.assertEqual(set(context.band_diagnostics), {"g", "r", "j"})
        self.assertAlmostEqual(
            context.band_diagnostics["r"].median_flux,
            20.0,
        )
        self.assertAlmostEqual(
            context.band_diagnostics["r"].median_uncertainty,
            0.5,
        )

    def test_one_dimensional_context_has_no_wavelength_diagnostics(self):
        lc = Lightcurve(
            torch.tensor([0.0, 1.0, 2.0]),
            torch.tensor([1.0, 2.0, 3.0]),
        )

        context = lc._build_parameter_estimation_context()

        self.assertIsNone(context.wavelength_diagnostics)
        self.assertEqual(context.band_diagnostics, {})


if __name__ == "__main__":
    unittest.main()
