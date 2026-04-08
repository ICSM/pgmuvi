"""Tests for subsample_lightcurve and fit_LS subsampling."""

import unittest

import numpy as np

from pgmuvi.preprocess import subsample_lightcurve
from pgmuvi.preprocess.quality import subsample_lightcurve as subsample_from_quality


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


if __name__ == "__main__":
    unittest.main()
