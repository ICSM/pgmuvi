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
        """Result length should be <= max_samples (ignoring gap-repair additions)."""
        idx = subsample_lightcurve(
            self.t_uniform,
            max_samples=self.max_samples,
            max_gap_fraction=self.max_gap_fraction,
            random_seed=42,
        )
        # Allow a small buffer for gap-repair additions
        self.assertLessEqual(len(idx), self.max_samples + 10)

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
        self.assertLessEqual(len(idx), 100 + 5)  # allow gap-repair headroom
        self.assertTrue(np.all(np.diff(t[idx]) >= 0))


class TestFitLSSubsampling1D(unittest.TestCase):
    """fit_LS should auto-subsample oversized 1-D lightcurves."""

    def setUp(self):
        lc = make_simple_sinusoid_1d(
            n_obs=200, period=5.0, amplitude=2.0, noise_level=0.1,
            t_span=50.0, seed=42,
        )
        self.lc = lc

    def test_no_warning_below_limit(self):
        """No subsampling warning when N <= max_samples."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.lc.fit_LS(max_samples=10000)
        sub_warns = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "max_samples" in str(w.message)
        ]
        self.assertEqual(len(sub_warns), 0)

    def test_warning_when_above_default_limit(self):
        """A UserWarning should be issued when N exceeds the default max_samples."""
        # Build a lightcurve large enough to trigger the default limit (10000)
        rng = np.random.default_rng(42)
        n = 12000
        t = np.sort(rng.uniform(0, 100, n))
        y = np.sin(2 * np.pi * t / 5)
        lc_large = Lightcurve(
            xdata=torch.tensor(t, dtype=torch.float32),
            ydata=torch.tensor(y, dtype=torch.float32),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lc_large.fit_LS()  # default max_samples=10000
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
            self.lc.fit_LS(max_samples=50)
        sub_warns = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "max_samples" in str(w.message)
        ]
        self.assertGreater(len(sub_warns), 0)
        self.assertIn("max_samples", str(sub_warns[0].message))

    def test_no_warning_when_disabled(self):
        """max_samples=None should disable subsampling entirely."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.lc.fit_LS(max_samples=None)
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
            freq1, mask1 = self.lc.fit_LS(max_samples=50, subsample_seed=7)
            freq2, mask2 = self.lc.fit_LS(max_samples=50, subsample_seed=7)
        self.assertTrue(torch.allclose(freq1, freq2))
        self.assertTrue(torch.equal(mask1, mask2))

    def test_return_shapes_unchanged(self):
        """Subsampling should not change the shape/type of fit_LS return values."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            freq_sub, mask_sub = self.lc.fit_LS(max_samples=50, subsample_seed=0)
        freq_full, mask_full = self.lc.fit_LS(max_samples=None)
        # Both should return 1D tensors
        self.assertEqual(freq_sub.ndim, 1)
        self.assertEqual(mask_sub.ndim, 1)
        self.assertEqual(freq_full.ndim, 1)
        self.assertEqual(mask_full.ndim, 1)

    def test_original_data_unaffected(self):
        """Lightcurve data must be unchanged after fit_LS with subsampling."""
        orig_n = self.lc.xdata.shape[0]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            self.lc.fit_LS(max_samples=50, subsample_seed=0)
        self.assertEqual(self.lc.xdata.shape[0], orig_n)
        self.assertEqual(self.lc.ydata.shape[0], orig_n)

    def test_freq_only_with_subsampling(self):
        """freq_only=True should work together with max_samples."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            freq, power = self.lc.fit_LS(freq_only=True, max_samples=50,
                                          subsample_seed=0)
        self.assertEqual(freq.ndim, 1)
        self.assertEqual(power.ndim, 1)
        self.assertEqual(freq.shape, power.shape)


class TestFitLSSubsampling2D(unittest.TestCase):
    """fit_LS should auto-subsample oversized multiband lightcurves."""

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
            self.lc_2d.fit_LS(max_samples=50)
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
            freq1, mask1 = self.lc_2d.fit_LS(max_samples=50, subsample_seed=3)
            freq2, mask2 = self.lc_2d.fit_LS(max_samples=50, subsample_seed=3)
        self.assertTrue(torch.allclose(freq1, freq2))

    def test_original_data_unaffected(self):
        """Lightcurve data must be unchanged after fit_LS with subsampling."""
        orig_n = self.lc_2d.xdata.shape[0]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            self.lc_2d.fit_LS(max_samples=50, subsample_seed=0)
        self.assertEqual(self.lc_2d.xdata.shape[0], orig_n)


if __name__ == "__main__":
    unittest.main()
