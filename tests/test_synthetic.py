"""Tests for the pgmuvi.synthetic module."""

import math
import unittest

import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.synthetic import (
    make_chromatic_sinusoid_2d,
    make_multi_sinusoid_1d,
    make_multi_sinusoid_chromatic_2d,
    make_simple_sinusoid_1d,
)


class TestMakeSimpleSinusoid1D(unittest.TestCase):
    """Tests for make_simple_sinusoid_1d."""

    def test_returns_lightcurve(self):
        lc = make_simple_sinusoid_1d(seed=0)
        self.assertIsInstance(lc, Lightcurve)

    def test_ndim_is_1(self):
        lc = make_simple_sinusoid_1d(seed=0)
        self.assertEqual(lc.ndim, 1)

    def test_output_shape(self):
        lc = make_simple_sinusoid_1d(n_obs=50, seed=0)
        self.assertEqual(lc.xdata.shape, torch.Size([50]))
        self.assertEqual(lc.ydata.shape, torch.Size([50]))

    def test_reproducible_with_seed(self):
        lc1 = make_simple_sinusoid_1d(seed=42)
        lc2 = make_simple_sinusoid_1d(seed=42)
        self.assertTrue(torch.allclose(lc1.ydata, lc2.ydata))

    def test_different_seeds_differ(self):
        lc1 = make_simple_sinusoid_1d(seed=0)
        lc2 = make_simple_sinusoid_1d(seed=1)
        self.assertFalse(torch.allclose(lc1.ydata, lc2.ydata))

    def test_noise_free(self):
        """Without noise the signal is a pure sinusoid."""
        lc = make_simple_sinusoid_1d(
            n_obs=100,
            period=5.0,
            amplitude=1.0,
            phase=0.0,
            noise_level=0.0,
            t_span=20.0,
            irregular=False,
            seed=0,
        )
        t = lc.xdata.numpy()
        expected = np.sin(2 * math.pi * t / 5.0).astype(np.float32)
        np.testing.assert_allclose(lc.ydata.numpy(), expected, atol=1e-5)

    def test_irregular_sampling(self):
        lc_reg = make_simple_sinusoid_1d(n_obs=50, irregular=False, seed=0)
        lc_irr = make_simple_sinusoid_1d(n_obs=50, irregular=True, seed=0)
        # Regular sampling should be exactly equally spaced
        diffs = torch.diff(lc_reg.xdata)
        self.assertAlmostEqual(diffs.std().item(), 0.0, places=4)
        # Irregular sampling should NOT be exactly equally spaced (with high probability)
        diffs_irr = torch.diff(lc_irr.xdata)
        self.assertGreater(diffs_irr.std().item(), 1e-4)

    def test_t_span_and_t_min(self):
        lc = make_simple_sinusoid_1d(t_min=100.0, t_span=50.0, seed=0)
        self.assertGreaterEqual(lc.xdata.min().item(), 100.0)
        self.assertLessEqual(lc.xdata.max().item(), 150.0)


class TestMakeMultiSinusoid1D(unittest.TestCase):
    """Tests for make_multi_sinusoid_1d."""

    def test_returns_lightcurve(self):
        lc = make_multi_sinusoid_1d(seed=0)
        self.assertIsInstance(lc, Lightcurve)

    def test_ndim_is_1(self):
        lc = make_multi_sinusoid_1d(seed=0)
        self.assertEqual(lc.ndim, 1)

    def test_output_shape(self):
        lc = make_multi_sinusoid_1d(n_obs=60, seed=0)
        self.assertEqual(lc.xdata.shape, torch.Size([60]))
        self.assertEqual(lc.ydata.shape, torch.Size([60]))

    def test_reproducible_with_seed(self):
        lc1 = make_multi_sinusoid_1d(seed=7)
        lc2 = make_multi_sinusoid_1d(seed=7)
        self.assertTrue(torch.allclose(lc1.ydata, lc2.ydata))

    def test_custom_components(self):
        components = [
            {"period": 2.0, "amplitude": 1.0, "phase": 0.0},
            {"period": 4.0, "amplitude": 0.5, "phase": math.pi / 2},
        ]
        lc = make_multi_sinusoid_1d(components=components, noise_level=0.0, seed=0)
        self.assertIsInstance(lc, Lightcurve)

    def test_noise_free_sum_of_sinusoids(self):
        """Without noise the signal equals the sum of component sinusoids."""
        components = [
            {"period": 5.0, "amplitude": 1.0, "phase": 0.0},
            {"period": 3.0, "amplitude": 0.5, "phase": math.pi / 3},
        ]
        lc = make_multi_sinusoid_1d(
            n_obs=100,
            components=components,
            noise_level=0.0,
            t_span=20.0,
            irregular=False,
            seed=0,
        )
        t = lc.xdata.numpy()
        expected = (
            np.sin(2 * math.pi * t / 5.0)
            + 0.5 * np.sin(2 * math.pi * t / 3.0 + math.pi / 3)
        ).astype(np.float32)
        np.testing.assert_allclose(lc.ydata.numpy(), expected, atol=1e-5)


class TestMakeChromaticSinusoid2D(unittest.TestCase):
    """Tests for make_chromatic_sinusoid_2d."""

    def test_returns_lightcurve(self):
        lc = make_chromatic_sinusoid_2d(seed=0)
        self.assertIsInstance(lc, Lightcurve)

    def test_ndim_is_2(self):
        lc = make_chromatic_sinusoid_2d(seed=0)
        self.assertEqual(lc.ndim, 2)

    def test_output_shape_default(self):
        """Default: 3 bands x 50 obs = 150 rows, 2 columns."""
        lc = make_chromatic_sinusoid_2d(n_per_band=50, seed=0)
        self.assertEqual(lc.xdata.shape, torch.Size([150, 2]))
        self.assertEqual(lc.ydata.shape, torch.Size([150]))

    def test_output_shape_per_band_list(self):
        lc = make_chromatic_sinusoid_2d(
            n_per_band=[10, 20, 30],
            wavelengths=[450.0, 600.0, 750.0],
            seed=0,
        )
        self.assertEqual(lc.xdata.shape[0], 60)

    def test_mismatched_n_per_band_raises(self):
        with self.assertRaises(ValueError):
            make_chromatic_sinusoid_2d(
                n_per_band=[10, 20],
                wavelengths=[450.0, 600.0, 750.0],
            )

    def test_linear_amplitude_law(self):
        lc = make_chromatic_sinusoid_2d(
            amplitude_law="linear", amplitude_slope=0.3, seed=0
        )
        self.assertIsInstance(lc, Lightcurve)

    def test_extinction_amplitude_law(self):
        lc = make_chromatic_sinusoid_2d(
            wavelengths=[0.8, 1.2, 2.2],
            amplitude_law="extinction",
            tau=2.0,
            alpha=1.7,
            seed=0,
        )
        self.assertIsInstance(lc, Lightcurve)

    def test_unknown_amplitude_law_raises(self):
        with self.assertRaises(ValueError):
            make_chromatic_sinusoid_2d(amplitude_law="unknown")

    def test_linear_phase_law(self):
        lc = make_chromatic_sinusoid_2d(phase_law="linear", phase_slope=0.1, seed=0)
        self.assertIsInstance(lc, Lightcurve)

    def test_unknown_phase_law_raises(self):
        with self.assertRaises(ValueError):
            make_chromatic_sinusoid_2d(phase_law="unknown")

    def test_reproducible_with_seed(self):
        lc1 = make_chromatic_sinusoid_2d(seed=3)
        lc2 = make_chromatic_sinusoid_2d(seed=3)
        self.assertTrue(torch.allclose(lc1.ydata, lc2.ydata))

    def test_wavelength_column_values(self):
        """Second column of xdata must only contain the specified wavelengths."""
        wls = [450.0, 600.0, 750.0]
        lc = make_chromatic_sinusoid_2d(wavelengths=wls, n_per_band=20, seed=0)
        wl_col = lc.xdata[:, 1].numpy()
        unique_wls = set(np.round(wl_col, 3))
        self.assertEqual(unique_wls, set(wls))

    def test_amplitude_varies_with_wavelength(self):
        """With linear law, higher slope => larger amplitude spread across bands."""
        lc_flat = make_chromatic_sinusoid_2d(
            amplitude_slope=0.0, noise_level=0.0, seed=0
        )
        lc_steep = make_chromatic_sinusoid_2d(
            amplitude_slope=1.0, noise_level=0.0, seed=0
        )
        # The steep-slope curve should have a larger range of y values
        self.assertGreater(
            lc_steep.ydata.abs().max().item(),
            lc_flat.ydata.abs().max().item(),
        )


class TestMakeMultiSinusoidChromatic2D(unittest.TestCase):
    """Tests for make_multi_sinusoid_chromatic_2d."""

    def test_returns_lightcurve(self):
        lc = make_multi_sinusoid_chromatic_2d(seed=0)
        self.assertIsInstance(lc, Lightcurve)

    def test_ndim_is_2(self):
        lc = make_multi_sinusoid_chromatic_2d(seed=0)
        self.assertEqual(lc.ndim, 2)

    def test_output_shape_default(self):
        """Default: 3 bands x 50 obs = 150 rows."""
        lc = make_multi_sinusoid_chromatic_2d(n_per_band=50, seed=0)
        self.assertEqual(lc.xdata.shape[0], 150)

    def test_custom_components(self):
        components = [
            {"period": 5.0, "amplitude_fraction": 0.4, "phase": 0.0},
            {"period": 2.5, "amplitude_fraction": 0.1, "phase": math.pi / 2},
            {"period": 10.0, "amplitude_fraction": 0.05, "phase": math.pi},
        ]
        lc = make_multi_sinusoid_chromatic_2d(components=components, seed=0)
        self.assertIsInstance(lc, Lightcurve)

    def test_reproducible_with_seed(self):
        lc1 = make_multi_sinusoid_chromatic_2d(seed=99)
        lc2 = make_multi_sinusoid_chromatic_2d(seed=99)
        self.assertTrue(torch.allclose(lc1.ydata, lc2.ydata))

    def test_extinction_law(self):
        lc = make_multi_sinusoid_chromatic_2d(
            wavelengths=[0.8, 1.2, 2.2],
            amplitude_law="extinction",
            tau=2.0,
            alpha=1.7,
            overall_amplitude=5.0,
            offset=0.2,
            seed=0,
        )
        self.assertIsInstance(lc, Lightcurve)
        self.assertEqual(lc.ndim, 2)

    def test_linear_amplitude_law(self):
        lc = make_multi_sinusoid_chromatic_2d(
            amplitude_law="linear",
            amplitude_slope=0.3,
            wl_ref=600.0,
            seed=0,
        )
        self.assertIsInstance(lc, Lightcurve)

    def test_unknown_amplitude_law_raises(self):
        with self.assertRaises(ValueError):
            make_multi_sinusoid_chromatic_2d(amplitude_law="bad")

    def test_mismatched_n_per_band_raises(self):
        with self.assertRaises(ValueError):
            make_multi_sinusoid_chromatic_2d(
                n_per_band=[10, 20],
                wavelengths=[0.8, 1.2, 2.2],
            )


if __name__ == "__main__":
    unittest.main()
