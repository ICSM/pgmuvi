"""Tests for subsample_lightcurve and fit_LS subsampling."""

import unittest
import warnings

import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.preprocess import subsample_lightcurve
from pgmuvi.preprocess.quality import subsample_lightcurve as subsample_from_quality
from pgmuvi.synthetic import make_chromatic_sinusoid_2d, make_simple_sinusoid_1d


class TestSubsampleLightcurveSmall(unittest.TestCase):
    """When N <= max_samples the full index range should be returned."""

    def test_small_array_returned_unchanged(self):
        """Arrays smaller than max_samples should be returned as-is."""
        t = np.linspace(0, 100, 50)
        idx = subsample_lightcurve(t, max_samples=100)
        np.testing.assert_array_equal(idx, np.arange(50))

    def test_exact_size_returned_unchanged(self):
        """Arrays exactly equal to max_samples should be returned as-is."""
        t = np.linspace(0, 100, 100)
        idx = subsample_lightcurve(t, max_samples=100)
        np.testing.assert_array_equal(idx, np.arange(100))


class TestSubsampleLightcurveLarge(unittest.TestCase):
    """When N > max_samples the result should satisfy size and gap constraints."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.t_uniform = np.sort(rng.uniform(0, 100, 5000))
        self.max_samples = 500
        self.max_gap_fraction = 0.3

    def test_output_size_at_most_max_samples(self):
        """Result length should be <= max_samples (budget strictly enforced)."""
        idx = subsample_lightcurve(
            self.t_uniform,
            max_samples=self.max_samples,
            max_gap_fraction=self.max_gap_fraction,
            random_seed=42,
        )
        self.assertLessEqual(len(idx), self.max_samples)

    def test_indices_are_valid(self):
        """All returned indices should be valid indices into the original array."""
        idx = subsample_lightcurve(
            self.t_uniform,
            max_samples=self.max_samples,
            random_seed=42,
        )
        self.assertTrue(np.all(idx >= 0))
        self.assertTrue(np.all(idx < len(self.t_uniform)))

    def test_indices_sorted_by_time(self):
        """Returned indices should be sorted in ascending time order."""
        idx = subsample_lightcurve(
            self.t_uniform,
            max_samples=self.max_samples,
            random_seed=42,
        )
        t_sub = self.t_uniform[idx]
        self.assertTrue(np.all(np.diff(t_sub) >= 0))

    def test_first_and_last_included(self):
        """The first and last time points (by time) must always be included."""
        idx = subsample_lightcurve(
            self.t_uniform,
            max_samples=self.max_samples,
            random_seed=42,
        )
        sort = np.argsort(self.t_uniform)
        self.assertIn(int(sort[0]), idx.tolist())
        self.assertIn(int(sort[-1]), idx.tolist())

    def test_gap_constraint_satisfied(self):
        """All gaps in the subsample must be <= max_gap_fraction * baseline."""
        idx = subsample_lightcurve(
            self.t_uniform,
            max_samples=self.max_samples,
            max_gap_fraction=self.max_gap_fraction,
            random_seed=42,
        )
        t_sub = self.t_uniform[idx]
        baseline = t_sub[-1] - t_sub[0]
        gaps = np.diff(t_sub)
        self.assertTrue(
            np.all(gaps <= self.max_gap_fraction * baseline + 1e-10),
            msg=f"Max gap fraction {gaps.max() / baseline:.4f} exceeds threshold "
            f"{self.max_gap_fraction}",
        )

    def test_no_duplicate_indices(self):
        """Returned indices should be unique."""
        idx = subsample_lightcurve(
            self.t_uniform,
            max_samples=self.max_samples,
            random_seed=42,
        )
        self.assertEqual(len(idx), len(np.unique(idx)))

    def test_reproducible_with_seed(self):
        """Same seed should yield identical results."""
        idx1 = subsample_lightcurve(self.t_uniform, max_samples=200, random_seed=7)
        idx2 = subsample_lightcurve(self.t_uniform, max_samples=200, random_seed=7)
        np.testing.assert_array_equal(idx1, idx2)

    def test_different_seeds_differ(self):
        """Different seeds should (very likely) yield different results."""
        idx1 = subsample_lightcurve(self.t_uniform, max_samples=200, random_seed=1)
        idx2 = subsample_lightcurve(self.t_uniform, max_samples=200, random_seed=2)
        self.assertFalse(np.array_equal(idx1, idx2))


class TestSubsampleLightcurveValidation(unittest.TestCase):
    """Input validation tests for subsample_lightcurve."""

    def test_max_samples_less_than_2_raises(self):
        """max_samples values less than 2 should raise ValueError."""
        t = np.linspace(0, 100, 50)
        for bad_value in (1, 0, -1, -100):
            with self.subTest(max_samples=bad_value):
                with self.assertRaises(ValueError):
                    subsample_lightcurve(t, max_samples=bad_value)

    def test_max_samples_non_integer_raises(self):
        """Non-integer max_samples should raise ValueError."""
        t = np.linspace(0, 100, 50)
        with self.assertRaises(ValueError):
            subsample_lightcurve(t, max_samples=2.5)


class TestSubsampleLightcurveEdgeCases(unittest.TestCase):
    """Edge-case behaviour of subsample_lightcurve."""

    def test_degenerate_zero_baseline(self):
        """All-identical times: return at most max_samples indices."""
        t = np.zeros(200)
        idx = subsample_lightcurve(t, max_samples=50)
        self.assertLessEqual(len(idx), 50)

    def test_unsorted_input(self):
        """Input does not need to be sorted; output is sorted by time."""
        rng = np.random.default_rng(99)
        t = rng.uniform(0, 100, 1000)  # deliberately unsorted
        idx = subsample_lightcurve(t, max_samples=100, random_seed=0)
        t_sub = t[idx]
        self.assertTrue(np.all(np.diff(t_sub) >= 0))

    def test_large_gap_in_original_data(self):
        """When the original data has a gap, the subsample should still respect it."""
        # Build data with a large gap in the middle
        t_part1 = np.linspace(0, 30, 2000)
        t_part2 = np.linspace(70, 100, 2000)
        t = np.concatenate([t_part1, t_part2])
        # max_gap_fraction=0.5 allows a gap of up to 50% of baseline (50 days).
        # The actual gap is 40 days = 40% which is under the 50% threshold.
        idx = subsample_lightcurve(
            t, max_samples=200, max_gap_fraction=0.5, random_seed=0
        )
        t_sub = t[idx]
        baseline = t_sub[-1] - t_sub[0]
        gaps = np.diff(t_sub)
        self.assertTrue(np.all(gaps <= 0.5 * baseline + 1e-10))

    def test_tight_gap_constraint_repaired(self):
        """A very tight max_gap_fraction should force extra points to be added."""
        rng = np.random.default_rng(0)
        t = rng.uniform(0, 100, 5000)
        # Use a very tight gap constraint.
        idx = subsample_lightcurve(
            t, max_samples=100, max_gap_fraction=0.05, random_seed=0
        )
        t_sub = t[idx]
        baseline = t_sub[-1] - t_sub[0]
        gaps = np.diff(t_sub)
        self.assertTrue(np.all(gaps <= 0.05 * baseline + 1e-10))


class TestSubsampleExportedFromPreprocess(unittest.TestCase):
    """subsample_lightcurve should be importable from pgmuvi.preprocess."""

    def test_import_from_preprocess(self):
        """Both import paths should return the same function."""
        self.assertIs(subsample_lightcurve, subsample_from_quality)

    def test_basic_functionality_via_preprocess(self):
        """Function imported from preprocess should work correctly."""
        t = np.linspace(0, 100, 500)
        idx = subsample_lightcurve(t, max_samples=100, random_seed=0)
        self.assertLessEqual(len(idx), 100)
        self.assertTrue(np.all(np.diff(t[idx]) >= 0))


class TestFitLSSubsampling1D(unittest.TestCase):
    """Lightcurve with max_samples set at __init__ should produce correct fit_LS."""

    def setUp(self):
        lc = make_simple_sinusoid_1d(
            n_obs=200, period=5.0, amplitude=2.0, noise_level=0.1,
            t_span=50.0, seed=42,
        )
        self.lc = lc

    def test_no_warning_below_limit(self):
        """No subsampling warning when N <= max_samples at init."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Lightcurve(
                xdata=self.lc.xdata,
                ydata=self.lc.ydata,
                max_samples=5000,
            )
        sub_warns = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "max_samples" in str(w.message)
        ]
        self.assertEqual(len(sub_warns), 0)

    def test_warning_when_above_default_limit(self):
        """A UserWarning should be issued when N exceeds max_samples at init."""
        rng = np.random.default_rng(42)
        n = 4000
        t = np.sort(rng.uniform(0, 100, n))
        y = np.sin(2 * np.pi * t / 5)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Lightcurve(
                xdata=torch.tensor(t, dtype=torch.float32),
                ydata=torch.tensor(y, dtype=torch.float32),
                max_samples=3000,
            )
        sub_warns = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "max_samples" in str(w.message)
        ]
        self.assertGreater(len(sub_warns), 0)

    def test_warning_when_above_limit(self):
        """A UserWarning about subsampling should be issued when N > max_samples."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Lightcurve(
                xdata=self.lc.xdata,
                ydata=self.lc.ydata,
                max_samples=50,
            )
        sub_warns = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "max_samples" in str(w.message)
        ]
        self.assertGreater(len(sub_warns), 0)
        self.assertIn("max_samples", str(sub_warns[0].message))

    def test_no_warning_when_disabled(self):
        """max_samples=None (default) should not trigger a subsampling warning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Lightcurve(
                xdata=self.lc.xdata,
                ydata=self.lc.ydata,
                max_samples=None,
            )
        sub_warns = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "max_samples" in str(w.message)
        ]
        self.assertEqual(len(sub_warns), 0)

    def test_reproducible_with_seed(self):
        """Same subsample_seed should produce identical periodogram peaks."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            lc1 = Lightcurve(
                xdata=self.lc.xdata,
                ydata=self.lc.ydata,
                max_samples=50,
                subsample_seed=7,
            )
            lc2 = Lightcurve(
                xdata=self.lc.xdata,
                ydata=self.lc.ydata,
                max_samples=50,
                subsample_seed=7,
            )
            freq1, mask1 = lc1.fit_LS()
            freq2, mask2 = lc2.fit_LS()
        self.assertTrue(torch.allclose(freq1, freq2))
        self.assertTrue(torch.equal(mask1, mask2))

    def test_return_shapes_unchanged(self):
        """Subsampling should not change the shape/type of fit_LS return values."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            lc_sub = Lightcurve(
                xdata=self.lc.xdata,
                ydata=self.lc.ydata,
                max_samples=50,
                subsample_seed=0,
            )
            freq_sub, mask_sub = lc_sub.fit_LS()
        freq_full, mask_full = self.lc.fit_LS()
        # Both should return 1D tensors
        self.assertEqual(freq_sub.ndim, 1)
        self.assertEqual(mask_sub.ndim, 1)
        self.assertEqual(freq_full.ndim, 1)
        self.assertEqual(mask_full.ndim, 1)

    def test_data_permanently_subsampled(self):
        """Lightcurve data is permanently subsampled when max_samples is set."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            lc_sub = Lightcurve(
                xdata=self.lc.xdata,
                ydata=self.lc.ydata,
                max_samples=50,
                subsample_seed=0,
            )
        self.assertLessEqual(lc_sub.xdata.shape[0], 50)
        self.assertLessEqual(lc_sub.ydata.shape[0], 50)

    def test_freq_only_with_subsampling(self):
        """freq_only=True should work with a subsampled lightcurve."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            lc_sub = Lightcurve(
                xdata=self.lc.xdata,
                ydata=self.lc.ydata,
                max_samples=50,
                subsample_seed=0,
            )
            freq, power = lc_sub.fit_LS(freq_only=True)
        self.assertEqual(freq.ndim, 1)
        self.assertEqual(power.ndim, 1)
        self.assertEqual(freq.shape, power.shape)


class TestFitLSSubsampling2D(unittest.TestCase):
    """Multiband Lightcurve with max_samples set at __init__ should work."""

    def setUp(self):
        lc_2d = make_chromatic_sinusoid_2d(
            n_per_band=120,
            period=2.0,
            wavelengths=[0.5, 1.5],
            amplitude_slope=0.0,
            noise_level=0.1,
            t_span=10.0,
            seed=42,
        )
        self.lc_2d = lc_2d

    def test_warning_when_above_limit(self):
        """A UserWarning should be issued when the 2D lightcurve is too large."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Lightcurve(
                xdata=self.lc_2d.xdata,
                ydata=self.lc_2d.ydata,
                max_samples=50,
            )
        sub_warns = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "max_samples" in str(w.message)
        ]
        self.assertGreater(len(sub_warns), 0)

    def test_reproducible_with_seed(self):
        """Same seed should produce identical results for 2D lightcurves."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            lc1 = Lightcurve(
                xdata=self.lc_2d.xdata,
                ydata=self.lc_2d.ydata,
                max_samples=50,
                subsample_seed=3,
            )
            lc2 = Lightcurve(
                xdata=self.lc_2d.xdata,
                ydata=self.lc_2d.ydata,
                max_samples=50,
                subsample_seed=3,
            )
            freq1, mask1 = lc1.fit_LS()
            freq2, mask2 = lc2.fit_LS()
        self.assertTrue(torch.allclose(freq1, freq2))

    def test_data_permanently_subsampled(self):
        """Each band is subsampled independently to at most max_samples points."""
        orig_n = self.lc_2d.xdata.shape[0]
        max_samples = 50
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            lc_sub = Lightcurve(
                xdata=self.lc_2d.xdata,
                ydata=self.lc_2d.ydata,
                max_samples=max_samples,
                subsample_seed=0,
            )
        # Overall size is reduced
        self.assertLess(lc_sub.xdata.shape[0], orig_n)
        # Each band should have at most max_samples points
        unique_bands = torch.unique(lc_sub.xdata[:, 1])
        for band in unique_bands:
            band_count = (lc_sub.xdata[:, 1] == band).sum().item()
            self.assertLessEqual(band_count, max_samples)

    def test_band_below_limit_not_reduced(self):
        """A band whose count is already <= max_samples must not be subsampled."""
        # Create a 2D lightcurve with two bands of very different sizes:
        # band A has 30 points (below limit), band B has 120 points (above).
        lc_big = make_chromatic_sinusoid_2d(
            n_per_band=120,
            period=2.0,
            wavelengths=[0.5, 1.5],
            amplitude_slope=0.0,
            noise_level=0.1,
            t_span=10.0,
            seed=99,
        )
        lc_small = make_chromatic_sinusoid_2d(
            n_per_band=30,
            period=2.0,
            wavelengths=[2.5],
            amplitude_slope=0.0,
            noise_level=0.1,
            t_span=10.0,
            seed=99,
        )
        # Combine: band 0.5 and 1.5 have 120 pts; band 2.5 has 30 pts.
        xdata_combined = torch.cat(
            [lc_big.xdata, lc_small.xdata], dim=0
        )
        ydata_combined = torch.cat(
            [lc_big.ydata, lc_small.ydata], dim=0
        )
        max_samples = 50
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lc_sub = Lightcurve(
                xdata=xdata_combined,
                ydata=ydata_combined,
                max_samples=max_samples,
                subsample_seed=0,
            )
        # Band 2.5 had 30 points (< 50), so it must remain at 30.
        band_val = torch.tensor(2.5, dtype=lc_sub.xdata.dtype)
        small_band_count = (
            torch.abs(lc_sub.xdata[:, 1] - band_val) < 1e-5
        ).sum().item()
        self.assertEqual(small_band_count, 30)
        # Only the two large bands should produce subsampling warnings.
        sub_warns = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "max_samples" in str(w.message)
        ]
        self.assertEqual(len(sub_warns), 2)


def _make_lightcurve(n):
    """Create a simple 1-D Lightcurve with *n* uniformly-spaced points."""
    rng = np.random.default_rng(42)
    t = np.sort(rng.uniform(0, 100, n))
    y = np.sin(2 * np.pi * t / 10) + rng.normal(0, 0.1, n)
    yerr = np.full(n, 0.1)
    return Lightcurve(
        xdata=torch.tensor(t, dtype=torch.float32),
        ydata=torch.tensor(y, dtype=torch.float32),
        yerr=torch.tensor(yerr, dtype=torch.float32),
    )


class TestSubsampleLightcurveWarning(unittest.TestCase):
    """Lightcurve.__init__ should warn and subsample when N > max_samples."""

    def test_no_warning_when_below_limit(self):
        """No UserWarning should be issued when N <= max_samples."""
        lc = _make_lightcurve(50)
        max_samples = 3000  # Larger than N, so no subsampling should occur.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            Lightcurve(
                xdata=lc.xdata,
                ydata=lc.ydata,
                yerr=lc.yerr,
                max_samples=max_samples,
            )
        subsample_warnings = [
            w for w in caught if issubclass(w.category, UserWarning)
        ]
        self.assertEqual(len(subsample_warnings), 0)

    def test_warning_issued_when_above_limit(self):
        """A UserWarning about subsampling should be issued when N > max_samples."""
        lc = _make_lightcurve(200)
        max_samples = 100  # Smaller than N, so subsampling should occur.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            lc_sub = Lightcurve(
                xdata=lc.xdata,
                ydata=lc.ydata,
                yerr=lc.yerr,
                max_samples=max_samples,
                subsample_seed=0,
            )
        subsample_warnings = [
            w for w in caught if issubclass(w.category, UserWarning)
        ]
        self.assertGreater(len(subsample_warnings), 0)
        first_msg = str(subsample_warnings[0].message)
        self.assertIn("max_samples", first_msg)
        self.assertIn("subsample", first_msg.lower())
        # Data is permanently subsampled.
        self.assertLessEqual(lc_sub._xdata_raw.shape[0], max_samples)

    def test_data_permanently_subsampled(self):
        """After __init__ with subsampling, data size is permanently reduced."""
        lc = _make_lightcurve(200)
        orig_n = lc._xdata_raw.shape[0]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            lc_sub = Lightcurve(
                xdata=lc.xdata,
                ydata=lc.ydata,
                yerr=lc.yerr,
                max_samples=50,
                subsample_seed=0,
            )
        # Data should be permanently reduced.
        self.assertLess(lc_sub._xdata_raw.shape[0], orig_n)
        self.assertLessEqual(lc_sub._xdata_raw.shape[0], 50)


if __name__ == "__main__":
    unittest.main()
