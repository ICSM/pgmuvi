"""Regression tests for consensus per-band transform isolation."""

import unittest

import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve, MinMax


class TestConsensusBandTransformInheritance(unittest.TestCase):
    def _make_multiband_lightcurve(self):
        times = torch.arange(36.0, dtype=torch.float64)
        time = torch.cat((times, times))
        wavelength = torch.cat(
            (
                torch.full_like(times, 1.25),
                torch.full_like(times, 3.50),
            )
        )
        xdata = torch.stack((time, wavelength), dim=1)
        ydata = 20.0 + torch.sin(2.0 * torch.pi * time / 12.0)
        yerr = torch.full_like(ydata, 0.1)
        band = np.asarray(["short"] * 36 + ["long"] * 36, dtype=np.str_)
        return Lightcurve(
            xdata,
            ydata,
            yerr=yerr,
            band=band,
            xtransform="minmax",
            ytransform="minmax",
        )

    def test_per_band_consensus_lightcurves_do_not_inherit_2d_transform(self):
        lc = self._make_multiband_lightcurve()

        self.assertIsInstance(lc.xtransform, MinMax)
        self.assertEqual(tuple(lc.xtransform.min.shape), (1, 2))
        parent_x_transformed = lc._xdata_transformed.clone()
        parent_y_transformed = lc._ydata_transformed.clone()

        per_band = dict(lc._consensus_iter_band_lightcurves())

        self.assertEqual(set(per_band), {"short", "long"})
        for band_lc in per_band.values():
            self.assertEqual(band_lc.ndim, 1)
            self.assertIsNone(band_lc.xtransform)
            self.assertIsNone(band_lc.ytransform)
            torch.testing.assert_close(
                band_lc._xdata_transformed,
                band_lc._xdata_raw,
            )
            torch.testing.assert_close(
                band_lc._ydata_transformed,
                band_lc._ydata_raw,
            )

        self.assertIsInstance(lc.xtransform, MinMax)
        self.assertEqual(tuple(lc.xtransform.min.shape), (1, 2))
        torch.testing.assert_close(lc._xdata_transformed, parent_x_transformed)
        torch.testing.assert_close(lc._ydata_transformed, parent_y_transformed)

    def test_per_band_sampling_preparation_accepts_parent_minmax_transform(self):
        lc = self._make_multiband_lightcurve()

        prepared = lc._consensus_prepare_band_consensus_inputs(
            min_points_per_band=8,
            max_gap_fraction=1.0,
            min_duty_cycle=0.0,
        )

        self.assertEqual(set(prepared["per_band_lc"]), {"short", "long"})
        self.assertEqual(
            prepared["controls"]["min_points_per_band"],
            8,
        )
        for metrics in prepared["metrics_by_band"].values():
            self.assertEqual(metrics["n_points"], 36)


if __name__ == "__main__":
    unittest.main()
