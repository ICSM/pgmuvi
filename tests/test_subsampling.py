"""Tests for subsample_lightcurve in pgmuvi/preprocess/quality.py."""

import unittest
import warnings

import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.preprocess.quality import subsample_lightcurve


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
        # Use a very tight gap constraint.
        idx = subsample_lightcurve(
            t, max_samples=100, max_gap_fraction=0.05, random_seed=0
        )
        t_sub = t[idx]
        baseline = t_sub[-1] - t_sub[0]
        gaps = np.diff(t_sub)
        self.assertTrue(np.all(gaps <= 0.05 * baseline + 1e-10))


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
    """Lightcurve.fit should warn and subsample when N > max_samples."""

    def test_no_warning_when_below_limit(self):
        """No UserWarning should be issued when N <= max_samples."""
        lc = _make_lightcurve(50)
        max_samples = 3000  # Larger than N, so no subsampling should occur.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            lc.fit(
                training_iter=1,
                miniter=1,
                max_samples=max_samples,
                subsample_seed=0,
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
            lc.fit(
                training_iter=1,
                miniter=1,
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

    def test_buffers_restored_after_subsampling(self):
        """After fit() with subsampling, original data buffers must be restored."""
        lc = _make_lightcurve(200)
        orig_n = lc._xdata_raw.shape[0]

        # Call fit() with subsampling enabled; internal buffers should be restored.
        lc.fit(
            training_iter=1,
            miniter=1,
            max_samples=50,
            subsample_seed=0,
        )

        # Buffers should be back to the original size after fit().
        self.assertEqual(lc._xdata_raw.shape[0], orig_n)


if __name__ == "__main__":
    unittest.main()
