"""Tests for fit_LS() return_full parameter."""

import unittest

import torch

from pgmuvi.synthetic import make_chromatic_sinusoid_2d, make_simple_sinusoid_1d


def _make_1d_lc(n_obs=80, period=5.0, noise_level=0.0, seed=42):
    return make_simple_sinusoid_1d(
        n_obs=n_obs, period=period, noise_level=noise_level, seed=seed
    )


def _make_2d_lc(n_per_band=50, period=5.0, seed=42):
    return make_chromatic_sinusoid_2d(
        n_per_band=n_per_band,
        period=period,
        wavelengths=[500.0, 700.0],
        amplitude_slope=0.0,
        noise_level=0.0,
        seed=seed,
    )


class TestFitLSReturnFullDefault(unittest.TestCase):
    """Verify that default behavior (return_full=False) is unchanged."""

    def test_1d_default_returns_2tuple(self):
        lc = _make_1d_lc()
        result = lc.fit_LS(num_peaks=3)
        self.assertEqual(len(result), 2)
        freqs, mask = result
        self.assertIsInstance(freqs, torch.Tensor)
        self.assertIsInstance(mask, torch.Tensor)
        self.assertEqual(mask.dtype, torch.bool)

    def test_2d_default_returns_2tuple(self):
        lc = _make_2d_lc()
        result = lc.fit_LS(num_peaks=3)
        self.assertEqual(len(result), 2)
        freqs, mask = result
        self.assertIsInstance(freqs, torch.Tensor)
        self.assertIsInstance(mask, torch.Tensor)
        self.assertEqual(mask.dtype, torch.bool)

    def test_1d_return_full_false_explicit_returns_2tuple(self):
        lc = _make_1d_lc()
        result = lc.fit_LS(num_peaks=3, return_full=False)
        self.assertEqual(len(result), 2)

    def test_2d_return_full_false_explicit_returns_2tuple(self):
        lc = _make_2d_lc()
        result = lc.fit_LS(num_peaks=3, return_full=False)
        self.assertEqual(len(result), 2)


class TestFitLSReturnFull1D(unittest.TestCase):
    """Tests for return_full=True on 1D lightcurves."""

    def setUp(self):
        self.lc = _make_1d_lc(n_obs=80, period=5.0, noise_level=0.0, seed=42)

    def test_returns_4tuple(self):
        result = self.lc.fit_LS(num_peaks=3, return_full=True)
        self.assertEqual(len(result), 4)

    def test_return_types(self):
        peak_freqs, sig_mask, freq_grid, power_grid = self.lc.fit_LS(
            num_peaks=3, return_full=True
        )
        self.assertIsInstance(peak_freqs, torch.Tensor)
        self.assertIsInstance(sig_mask, torch.Tensor)
        self.assertIsInstance(freq_grid, torch.Tensor)
        self.assertIsInstance(power_grid, torch.Tensor)
        self.assertEqual(sig_mask.dtype, torch.bool)

    def test_freq_grid_and_power_grid_shapes_match(self):
        _, _, freq_grid, power_grid = self.lc.fit_LS(
            num_peaks=3, return_full=True
        )
        self.assertEqual(freq_grid.shape, power_grid.shape)
        self.assertGreater(len(freq_grid), 0)

    def test_freq_grid_is_positive(self):
        _, _, freq_grid, _ = self.lc.fit_LS(num_peaks=3, return_full=True)
        self.assertTrue(torch.all(freq_grid > 0))

    def test_peak_freqs_subset_of_freq_grid(self):
        """Each returned peak frequency should lie within the grid bounds."""
        peak_freqs, _, freq_grid, _ = self.lc.fit_LS(
            num_peaks=3, return_full=True
        )
        if len(peak_freqs) > 0:
            self.assertTrue(
                torch.all(peak_freqs >= freq_grid.min())
                and torch.all(peak_freqs <= freq_grid.max())
            )

    def test_dtype_matches_xdata(self):
        _, _, freq_grid, power_grid = self.lc.fit_LS(
            num_peaks=3, return_full=True
        )
        self.assertEqual(freq_grid.dtype, self.lc.xdata.dtype)
        self.assertEqual(power_grid.dtype, self.lc.xdata.dtype)

    def test_device_matches_xdata(self):
        _, _, freq_grid, power_grid = self.lc.fit_LS(
            num_peaks=3, return_full=True
        )
        self.assertEqual(freq_grid.device, self.lc.xdata.device)
        self.assertEqual(power_grid.device, self.lc.xdata.device)

    def test_return_full_consistent_with_freq_only(self):
        """freq_grid and power_grid must match what freq_only=True returns."""
        fg_full, pg_full = self.lc.fit_LS(freq_only=True)
        _, _, fg_rf, pg_rf = self.lc.fit_LS(num_peaks=3, return_full=True)
        torch.testing.assert_close(fg_full, fg_rf)
        torch.testing.assert_close(pg_full, pg_rf)

    def test_insignificant_peaks_still_returns_4tuple(self):
        """return_full=True works even when the highest peak is insignificant."""
        # Very short, noisy lightcurve — highest peak likely insignificant
        lc = _make_1d_lc(n_obs=20, period=5.0, noise_level=2.0, seed=7)
        result = lc.fit_LS(
            num_peaks=3, return_full=True, single_threshold=1e-10
        )
        self.assertEqual(len(result), 4)
        _, sig_mask, freq_grid, power_grid = result
        self.assertEqual(freq_grid.shape, power_grid.shape)


class TestFitLSReturnFull2D(unittest.TestCase):
    """Tests for return_full=True on multiband (2D) lightcurves."""

    def setUp(self):
        self.lc = _make_2d_lc(n_per_band=55, period=5.0, seed=42)

    def test_returns_4tuple(self):
        result = self.lc.fit_LS(num_peaks=3, return_full=True)
        self.assertEqual(len(result), 4)

    def test_return_types(self):
        peak_freqs, sig_mask, freq_grid, power_grid = self.lc.fit_LS(
            num_peaks=3, return_full=True
        )
        self.assertIsInstance(peak_freqs, torch.Tensor)
        self.assertIsInstance(sig_mask, torch.Tensor)
        self.assertIsInstance(freq_grid, torch.Tensor)
        self.assertIsInstance(power_grid, torch.Tensor)
        self.assertEqual(sig_mask.dtype, torch.bool)

    def test_freq_grid_and_power_grid_shapes_match(self):
        _, _, freq_grid, power_grid = self.lc.fit_LS(
            num_peaks=3, return_full=True
        )
        self.assertEqual(freq_grid.shape, power_grid.shape)
        self.assertGreater(len(freq_grid), 0)

    def test_dtype_matches_xdata(self):
        _, _, freq_grid, power_grid = self.lc.fit_LS(
            num_peaks=3, return_full=True
        )
        self.assertEqual(freq_grid.dtype, self.lc.xdata.dtype)
        self.assertEqual(power_grid.dtype, self.lc.xdata.dtype)

    def test_device_matches_xdata(self):
        _, _, freq_grid, power_grid = self.lc.fit_LS(
            num_peaks=3, return_full=True
        )
        self.assertEqual(freq_grid.device, self.lc.xdata.device)
        self.assertEqual(power_grid.device, self.lc.xdata.device)

    def test_return_full_consistent_with_freq_only(self):
        """freq_grid and power_grid must match what freq_only=True returns."""
        fg_full, pg_full = self.lc.fit_LS(freq_only=True)
        _, _, fg_rf, pg_rf = self.lc.fit_LS(num_peaks=3, return_full=True)
        torch.testing.assert_close(fg_full, fg_rf)
        torch.testing.assert_close(pg_full, pg_rf)

    def test_insignificant_peaks_2d_still_returns_4tuple(self):
        """return_full=True works on 2D even when peak is insignificant."""
        result = self.lc.fit_LS(
            num_peaks=3, return_full=True, single_threshold=1e-20
        )
        self.assertEqual(len(result), 4)
        _, _, freq_grid, power_grid = result
        self.assertEqual(freq_grid.shape, power_grid.shape)

    def test_freq_only_ignores_return_full(self):
        """freq_only=True returns 2-tuple regardless of return_full."""
        result = self.lc.fit_LS(freq_only=True, return_full=True)
        self.assertEqual(len(result), 2)


class TestFitLSReturnFullNoPeaks(unittest.TestCase):
    """Tests for the no-peaks code path with return_full=True."""

    def test_1d_no_peaks_returns_4tuple_with_return_full(self):
        """When find_peaks returns nothing, return_full=True still gives 4-tuple."""
        from unittest.mock import patch
        import numpy as np
        lc = _make_1d_lc()
        # Patch find_peaks to simulate a periodogram with no detectable peaks.
        with patch(
            "scipy.signal.find_peaks",
            return_value=(np.array([], dtype=int), {}),
        ):
            result = lc.fit_LS(num_peaks=3, return_full=True)
        self.assertEqual(len(result), 4)
        peak_freqs, sig_mask, freq_grid, power_grid = result
        self.assertEqual(len(peak_freqs), 0)
        self.assertEqual(len(sig_mask), 0)
        self.assertGreater(len(freq_grid), 0)
        self.assertEqual(freq_grid.shape, power_grid.shape)

    def test_1d_no_peaks_default_still_returns_2tuple(self):
        """When find_peaks returns nothing and return_full=False, gives 2-tuple."""
        from unittest.mock import patch
        import numpy as np
        lc = _make_1d_lc()
        with patch(
            "scipy.signal.find_peaks",
            return_value=(np.array([], dtype=int), {}),
        ):
            result = lc.fit_LS(num_peaks=3, return_full=False)
        self.assertEqual(len(result), 2)

    def test_2d_no_peaks_returns_4tuple_with_return_full(self):
        """Multiband: no-peaks path also returns 4-tuple with return_full=True."""
        from unittest.mock import patch
        import numpy as np
        lc = _make_2d_lc()
        with patch(
            "scipy.signal.find_peaks",
            return_value=(np.array([], dtype=int), {}),
        ):
            result = lc.fit_LS(num_peaks=3, return_full=True)
        self.assertEqual(len(result), 4)
        peak_freqs, sig_mask, freq_grid, power_grid = result
        self.assertEqual(len(peak_freqs), 0)
        self.assertEqual(len(sig_mask), 0)
        self.assertGreater(len(freq_grid), 0)
        self.assertEqual(freq_grid.shape, power_grid.shape)


if __name__ == "__main__":
    unittest.main()
