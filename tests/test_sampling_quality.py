"""Tests for compute_sampling_metrics and assess_sampling_quality."""

import unittest
import warnings

import numpy as np

from pgmuvi.preprocess.quality import (
    assess_sampling_quality,
    compute_sampling_metrics,
)


def _user_warnings(caught):
    """Return only UserWarning entries from a catch_warnings record list."""
    return [w for w in caught if issubclass(w.category, UserWarning)]


class TestComputeSamplingMetricsNormal(unittest.TestCase):
    """compute_sampling_metrics with regular, well-sampled data."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.t = np.sort(rng.uniform(0, 100, 200))

    def test_returns_expected_keys(self):
        """All expected metric keys must be present."""
        metrics = compute_sampling_metrics(self.t)
        required = {
            "n_points",
            "baseline",
            "max_gap",
            "max_gap_fraction",
            "median_cadence",
            "mean_cadence",
            "cadence_std",
            "nyquist_period",
            "nyquist_frequency",
            "longest_detectable_period",
            "duty_cycle",
            "sampling_uniformity",
        }
        self.assertTrue(required.issubset(metrics.keys()))

    def test_nyquist_period_is_positive(self):
        metrics = compute_sampling_metrics(self.t)
        self.assertGreater(metrics["nyquist_period"], 0)

    def test_nyquist_frequency_is_finite(self):
        metrics = compute_sampling_metrics(self.t)
        self.assertTrue(np.isfinite(metrics["nyquist_frequency"]))

    def test_duty_cycle_in_range(self):
        metrics = compute_sampling_metrics(self.t)
        self.assertGreaterEqual(metrics["duty_cycle"], 0.0)
        self.assertLessEqual(metrics["duty_cycle"], 1.0)

    def test_no_warning_for_regular_data(self):
        """No UserWarning should be issued for normally-sampled data."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_sampling_metrics(self.t)
        self.assertEqual(len(_user_warnings(caught)), 0)

    def test_nyquist_uses_median_for_regular_data(self):
        """For regular data, nyquist_period == 2 * median_cadence."""
        metrics = compute_sampling_metrics(self.t)
        expected = 2.0 * metrics["median_cadence"]
        self.assertAlmostEqual(metrics["nyquist_period"], expected)


class TestComputeSamplingMetricsDuplicateTimestamps(unittest.TestCase):
    """compute_sampling_metrics when median cadence collapses to zero."""

    def _make_duplicate_heavy_times(self, n_unique=10, n_duplicates=20, seed=0):
        """Return times where majority of entries are duplicated."""
        rng = np.random.default_rng(seed)
        unique_times = np.sort(rng.uniform(0, 100, n_unique))
        # Repeat each unique time many times so that > half of gaps are zero.
        return np.repeat(unique_times, n_duplicates)

    def test_nyquist_period_is_nonzero(self):
        """nyquist_period must be > 0 even when median_cadence == 0."""
        t = self._make_duplicate_heavy_times()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            metrics = compute_sampling_metrics(t)
        self.assertGreater(metrics["nyquist_period"], 0.0)

    def test_nyquist_frequency_is_finite(self):
        """nyquist_frequency must be finite even when median_cadence == 0."""
        t = self._make_duplicate_heavy_times()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            metrics = compute_sampling_metrics(t)
        self.assertTrue(np.isfinite(metrics["nyquist_frequency"]))

    def test_duty_cycle_is_nonzero(self):
        """duty_cycle must be > 0 even when median_cadence == 0."""
        t = self._make_duplicate_heavy_times()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            metrics = compute_sampling_metrics(t)
        self.assertGreater(metrics["duty_cycle"], 0.0)

    def test_median_cadence_is_zero(self):
        """Confirm the test data actually produces median_cadence == 0."""
        t = self._make_duplicate_heavy_times()
        gaps = np.diff(np.sort(t))
        self.assertEqual(float(np.median(gaps)), 0.0)

    def test_nyquist_period_uses_positive_mean_cadence(self):
        """nyquist_period should equal 2 * mean(positive gaps) when median is zero."""
        t = self._make_duplicate_heavy_times()
        gaps = np.diff(np.sort(t))
        positive_mean = float(np.mean(gaps[gaps > 0]))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            metrics = compute_sampling_metrics(t)
        self.assertEqual(metrics["median_cadence"], 0.0)
        expected = 2.0 * positive_mean
        self.assertAlmostEqual(metrics["nyquist_period"], expected)

    def test_userwarning_emitted_when_median_zero(self):
        """A UserWarning must be issued when median_cadence == 0."""
        t = self._make_duplicate_heavy_times()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_sampling_metrics(t)
        user_warns = _user_warnings(caught)
        self.assertGreater(len(user_warns), 0)
        msg = str(user_warns[0].message)
        self.assertIn("median_cadence", msg)
        self.assertIn("positive gaps", msg)


class TestComputeSamplingMetricsTightlyClustered(unittest.TestCase):
    """compute_sampling_metrics with tightly clustered (non-zero) cadence."""

    def _make_clustered_times(self, n_clusters=10, n_per_cluster=5, seed=0):
        """Return times with tight within-cluster spacing but nonzero gaps."""
        rng = np.random.default_rng(seed)
        cluster_centres = np.sort(rng.uniform(0, 100, n_clusters))
        offsets = rng.uniform(0.01, 0.1, (n_clusters, n_per_cluster))
        t = (cluster_centres[:, None] + offsets).ravel()
        return np.sort(t)

    def test_no_fallback_warning_for_clustered_data(self):
        """No UserWarning for tightly clustered data (median_cad > 0)."""
        t = self._make_clustered_times()
        # Confirm median_cadence > 0 for this data.
        gaps = np.diff(t)
        self.assertGreater(float(np.median(gaps)), 0.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_sampling_metrics(t)
        self.assertEqual(len(_user_warnings(caught)), 0)

    def test_nyquist_period_uses_median_for_clustered_data(self):
        """When median_cad > 0, nyquist_period must equal 2 * median_cad."""
        t = self._make_clustered_times()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            metrics = compute_sampling_metrics(t)
        expected = 2.0 * metrics["median_cadence"]
        self.assertAlmostEqual(metrics["nyquist_period"], expected)


class TestAssessSamplingQualityConfiguredSnrThreshold(unittest.TestCase):
    """The SNR fraction gate must follow the caller's threshold."""

    def test_fraction_gate_uses_min_snr_not_legacy_threshold_three(self):
        t = np.linspace(0.0, 100.0, 100)
        y = np.concatenate([np.full(50, 4.0), np.full(50, 6.0)])
        yerr = np.ones_like(y)

        passes, diag = assess_sampling_quality(
            t,
            y,
            yerr,
            min_snr=5.0,
            min_fraction_good_snr=0.75,
        )

        self.assertFalse(passes)
        self.assertFalse(diag["gates"]["min_snr"])
        self.assertEqual(diag["metrics"]["snr_threshold"], 5.0)
        self.assertAlmostEqual(
            diag["metrics"]["fraction_snr_gt_min"],
            0.5,
        )
        self.assertEqual(diag["metrics"]["fraction_snr_gt_3"], 1.0)
        self.assertTrue(
            any("SNR > 5.0" in warning for warning in diag["warnings"])
        )


class TestAssessSamplingQualityDuplicateWarning(unittest.TestCase):
    """assess_sampling_quality diagnostic warnings with duplicate timestamps."""

    def _make_duplicate_heavy_times(self, n_unique=10, n_duplicates=20, seed=0):
        rng = np.random.default_rng(seed)
        unique_times = np.sort(rng.uniform(0, 100, n_unique))
        return np.repeat(unique_times, n_duplicates)

    def test_diagnostic_warning_mentions_identical_timestamps(self):
        """The diagnostics warnings list should mention duplicate timestamps."""
        t = self._make_duplicate_heavy_times()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _, diag = assess_sampling_quality(t)
        combined = " ".join(diag["warnings"])
        self.assertIn("identical", combined.lower())

    def test_assess_uses_mean_cadence_for_baseline_factor(self):
        """Duplicate timestamps should trigger the mean-cadence fallback."""
        t = self._make_duplicate_heavy_times()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _, diag = assess_sampling_quality(t)
        self.assertEqual(diag["metrics"]["median_cadence"], 0.0)
        combined = " ".join(diag["warnings"]).lower()
        self.assertIn("identical", combined)
        self.assertIn("mean cadence", combined)


if __name__ == "__main__":
    unittest.main()
