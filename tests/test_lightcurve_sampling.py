"""Tests for robust_scale, sampling metrics and Lightcurve sampling methods."""
import unittest
import warnings

import gpytorch
import numpy as np
import torch

from pgmuvi.lightcurve import (
    Lightcurve,
)


class TestRobustScale(unittest.TestCase):
    """Tests for robust_scale function."""

    def test_gaussian_scale(self):
        """Standard normal should give scale ≈ 1."""
        from pgmuvi.preprocess.quality import robust_scale
        np.random.seed(42)
        y = np.random.normal(0, 1, 1000)
        scale = robust_scale(y)
        self.assertAlmostEqual(scale, 1.0, delta=0.05)

    def test_constant_array(self):
        """Constant array should give scale = 0."""
        from pgmuvi.preprocess.quality import robust_scale
        y_const = np.ones(100)
        self.assertEqual(robust_scale(y_const), 0.0)

    def test_empty_after_filtering(self):
        """All non-finite values should give scale = 0."""
        from pgmuvi.preprocess.quality import robust_scale
        y_nan = np.array([np.nan, np.inf, -np.inf])
        self.assertEqual(robust_scale(y_nan), 0.0)


class TestComputeSamplingMetrics(unittest.TestCase):
    """Tests for compute_sampling_metrics function."""

    def test_basic_uniform(self):
        """Test basic metrics for uniform sampling."""
        from pgmuvi.preprocess.quality import compute_sampling_metrics
        t = np.linspace(0, 100, 101)
        metrics = compute_sampling_metrics(t)
        self.assertEqual(metrics['n_points'], 101)
        self.assertAlmostEqual(metrics['baseline'], 100.0, places=3)
        self.assertAlmostEqual(metrics['median_cadence'], 1.0, delta=0.01)
        self.assertAlmostEqual(metrics['nyquist_period'], 2.0, delta=0.01)
        self.assertGreater(metrics['sampling_uniformity'], 0.99)

    def test_with_gap(self):
        """Test metrics with large gap."""
        from pgmuvi.preprocess.quality import compute_sampling_metrics
        t = np.concatenate([np.linspace(0, 10, 50), np.linspace(90, 100, 50)])
        metrics = compute_sampling_metrics(t)
        self.assertAlmostEqual(metrics['baseline'], 100.0, places=3)
        self.assertGreater(metrics['max_gap'], 75)
        self.assertGreater(metrics['max_gap_fraction'], 0.7)

    def test_with_snr(self):
        """Test SNR metrics."""
        from pgmuvi.preprocess.quality import compute_sampling_metrics
        t = np.linspace(0, 100, 100)
        y = np.ones(100)
        yerr = np.full(100, 0.1)
        metrics = compute_sampling_metrics(t, y, yerr)
        self.assertIn('median_snr', metrics)
        self.assertAlmostEqual(metrics['median_snr'], 10.0, delta=0.1)
        self.assertEqual(metrics['fraction_snr_gt_3'], 1.0)
        self.assertEqual(metrics['fraction_snr_gt_5'], 1.0)

    def test_too_few_points(self):
        """Test error for too few points."""
        from pgmuvi.preprocess.quality import compute_sampling_metrics
        metrics = compute_sampling_metrics(np.array([1.0]))
        self.assertIn('error', metrics)

    def test_zero_baseline(self):
        """Test error for zero baseline."""
        from pgmuvi.preprocess.quality import compute_sampling_metrics
        metrics = compute_sampling_metrics(np.array([1.0, 1.0, 1.0]))
        self.assertIn('error', metrics)


class TestAssessSamplingQuality(unittest.TestCase):
    """Tests for assess_sampling_quality function."""

    def test_good_sampling(self):
        """Good sampling should pass all gates."""
        from pgmuvi.preprocess.quality import assess_sampling_quality
        np.random.seed(42)
        t = np.linspace(0, 100, 100)
        y = np.ones(100) + np.random.normal(0, 0.01, 100)
        yerr = np.full(100, 0.01)
        passes, diag = assess_sampling_quality(t, y, yerr, verbose=False)
        self.assertTrue(passes)
        self.assertEqual(diag['recommendation'], 'PROCEED')
        self.assertEqual(len(diag['warnings']), 0)

    def test_too_few_points(self):
        """Too few points should fail min_points gate."""
        from pgmuvi.preprocess.quality import assess_sampling_quality
        t = np.array([0, 1, 2, 3], dtype=float)
        y = np.ones(4)
        yerr = np.full(4, 0.1)
        passes, diag = assess_sampling_quality(
            t, y, yerr, min_points=6, verbose=False
        )
        self.assertFalse(passes)
        self.assertEqual(diag['recommendation'], 'DO NOT FIT')
        self.assertFalse(diag['gates']['min_points'])
        self.assertTrue(any('Too few points' in w for w in diag['warnings']))

    def test_large_gap(self):
        """Large gap should fail max_gap gate."""
        from pgmuvi.preprocess.quality import assess_sampling_quality
        t = np.concatenate([np.linspace(0, 10, 50), np.linspace(60, 70, 50)])
        y = np.ones(100)
        yerr = np.full(100, 0.1)
        passes, diag = assess_sampling_quality(
            t, y, yerr, max_gap_fraction=0.3, verbose=False
        )
        self.assertFalse(passes)
        self.assertFalse(diag['gates']['max_gap'])
        self.assertTrue(any('Large gap' in w for w in diag['warnings']))

    def test_poor_snr(self):
        """Poor SNR should fail min_snr gate."""
        from pgmuvi.preprocess.quality import assess_sampling_quality
        t = np.linspace(0, 100, 100)
        y = np.ones(100)
        yerr = np.full(100, 1.0)  # SNR = 1
        passes, diag = assess_sampling_quality(
            t, y, yerr, min_snr=3.0, verbose=False
        )
        self.assertFalse(passes)
        self.assertFalse(diag['gates']['min_snr'])
        self.assertTrue(any('Poor SNR' in w for w in diag['warnings']))

    def test_error_propagation(self):
        """Error in metrics should propagate to DO NOT FIT."""
        from pgmuvi.preprocess.quality import assess_sampling_quality
        t = np.array([1.0])  # Only 1 point
        passes, diag = assess_sampling_quality(t, verbose=False)
        self.assertFalse(passes)
        self.assertEqual(diag['recommendation'], 'DO NOT FIT')


class TestLightcurveSamplingMethods(unittest.TestCase):
    """Tests for sampling quality methods on Lightcurve class."""

    def setUp(self):
        np.random.seed(42)
        t = np.linspace(0, 100, 100)
        y = np.ones(100) + 0.1 * np.sin(2 * np.pi * t / 10)
        yerr = np.full(100, 0.01)
        self.lc = Lightcurve(
            torch.as_tensor(t, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
            yerr=torch.as_tensor(yerr, dtype=torch.float32),
        )

    def test_compute_sampling_metrics(self):
        """Lightcurve.compute_sampling_metrics returns expected keys."""
        metrics = self.lc.compute_sampling_metrics()
        self.assertIn('n_points', metrics)
        self.assertEqual(metrics['n_points'], 100)
        self.assertIn('nyquist_period', metrics)
        self.assertNotIn('snr_values', metrics)

    def test_assess_sampling_quality(self):
        """Lightcurve.assess_sampling_quality returns passes and diagnostics."""
        passes, diag = self.lc.assess_sampling_quality(verbose=False)
        self.assertTrue(passes)
        self.assertEqual(diag['recommendation'], 'PROCEED')
        self.assertNotIn('snr_values', diag.get('metrics', {}))

    def test_init_check_sampling_raises(self):
        """Lightcurve(..., check_sampling=True) raises for poor sampling."""
        t_few = torch.as_tensor([0, 1, 2, 3], dtype=torch.float32)
        y_few = torch.as_tensor([1, 1, 1, 1], dtype=torch.float32)
        yerr_few = torch.as_tensor([0.1, 0.1, 0.1, 0.1], dtype=torch.float32)
        with self.assertRaises(ValueError):
            Lightcurve(t_few, y_few, yerr=yerr_few, check_sampling=True)

    def test_init_no_check_sampling_allows_fit(self):
        """Without check_sampling, fit() raises for missing model, not sampling."""
        t_few = torch.as_tensor([0, 1, 2, 3], dtype=torch.float32)
        y_few = torch.as_tensor([1, 1, 1, 1], dtype=torch.float32)
        yerr_few = torch.as_tensor([0.1, 0.1, 0.1, 0.1], dtype=torch.float32)
        lc_few = Lightcurve(t_few, y_few, yerr=yerr_few)
        # Should raise for missing model, not for sampling quality
        with self.assertRaises(ValueError) as ctx:
            lc_few.fit()
        self.assertNotIn('temporal sampling', str(ctx.exception))


class TestLightcurve2DSamplingMethods(unittest.TestCase):
    """Tests for multiband sampling quality methods."""

    def setUp(self):
        np.random.seed(0)
        t = np.linspace(0, 100, 50)
        wavelengths = [3.6, 4.5, 5.8]
        t_all, wl_all, y_all, ye_all = [], [], [], []
        for wl in wavelengths:
            t_all.extend(t)
            wl_all.extend([wl] * len(t))
            y_all.extend(np.ones(len(t)))
            ye_all.extend([0.01] * len(t))

        xdata = np.column_stack([t_all, wl_all])
        self.lc2d = Lightcurve(
            torch.as_tensor(xdata, dtype=torch.float32),
            torch.as_tensor(y_all, dtype=torch.float32),
            yerr=torch.as_tensor(ye_all, dtype=torch.float32),
        )

    def test_compute_sampling_metrics_per_band(self):
        """Should return metrics dict with 3 bands and summary."""
        results = self.lc2d.compute_sampling_metrics_per_band()
        self.assertIn('summary', results)
        self.assertEqual(results['summary']['n_bands'], 3)
        # Check via summary rather than exact float32 keys
        band_keys = [k for k in results if k != 'summary']
        self.assertEqual(len(band_keys), 3)
        for metrics in [results[k] for k in band_keys]:
            self.assertIn('n_points', metrics)
            self.assertEqual(metrics['n_points'], 50)
            self.assertNotIn('snr_values', metrics)

    def test_assess_sampling_quality_per_band(self):
        """All bands should pass quality checks."""
        results = self.lc2d.assess_sampling_quality_per_band(verbose=False)
        self.assertEqual(results['summary']['n_passing'], 3)
        self.assertEqual(len(results['summary']['failing_wavelengths']), 0)
        for wl, diag in results.items():
            if wl == 'summary':
                continue
            self.assertNotIn('snr_values', diag)
            self.assertNotIn('snr_values', diag.get('metrics', {}))

    def test_filter_well_sampled_bands(self):
        """filter_well_sampled_bands returns Lightcurve with passing bands only."""
        filtered = self.lc2d.filter_well_sampled_bands()
        self.assertIsInstance(filtered, Lightcurve)
        self.assertGreater(filtered.ndim, 1)

    def test_per_band_methods_raise_for_1d(self):
        """Per-band methods should raise for 1D lightcurves."""
        t = torch.as_tensor(np.linspace(0, 10, 20), dtype=torch.float32)
        y = torch.ones(20)
        lc1d = Lightcurve(t, y)
        with self.assertRaises(ValueError):
            lc1d.compute_sampling_metrics_per_band()
        with self.assertRaises(ValueError):
            lc1d.assess_sampling_quality_per_band()
        with self.assertRaises(ValueError):
            lc1d.filter_well_sampled_bands()

    def test_init_check_sampling_2d_all_bands_fail_raises(self):
        """Lightcurve(..., check_sampling=True) raises when ALL 2D bands fail."""
        # 3 bands, each with only 4 points (below min_points=6 default)
        wavelengths = [3.6, 4.5, 5.8]
        t_all, wl_all, y_all, ye_all = [], [], [], []
        for wl in wavelengths:
            t_all.extend([0.0, 1.0, 2.0, 3.0])
            wl_all.extend([wl] * 4)
            y_all.extend([1.0] * 4)
            ye_all.extend([0.1] * 4)
        xdata = np.column_stack([t_all, wl_all])
        with self.assertRaises(ValueError) as ctx:
            Lightcurve(
                torch.as_tensor(xdata, dtype=torch.float32),
                torch.as_tensor(y_all, dtype=torch.float32),
                yerr=torch.as_tensor(ye_all, dtype=torch.float32),
                check_sampling=True,
            )
        self.assertIn('sampling quality checks', str(ctx.exception))

    def test_init_check_sampling_2d_some_bands_fail_filters(self):
        """Lightcurve(..., check_sampling=True) filters out poorly-sampled 2D bands."""
        # Band 3.6: 50 good points (passes); band 4.5: 4 points (fails)
        t_good = np.linspace(0, 100, 50)
        t_bad = [0.0, 1.0, 2.0, 3.0]
        t_all = list(t_good) + t_bad
        wl_all = [3.6] * 50 + [4.5] * 4
        y_all = [1.0] * 54
        ye_all = [0.01] * 54
        xdata = np.column_stack([t_all, wl_all])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lc2d_mixed = Lightcurve(
                torch.as_tensor(xdata, dtype=torch.float32),
                torch.as_tensor(y_all, dtype=torch.float32),
                yerr=torch.as_tensor(ye_all, dtype=torch.float32),
                check_sampling=True,
            )
        # The poorly-sampled band should have been filtered at init time.
        remaining_wls = torch.unique(lc2d_mixed._xdata_raw[:, 1]).tolist()
        self.assertEqual(len(remaining_wls), 1)
        self.assertAlmostEqual(remaining_wls[0], 3.6, places=4)
        # fit() should then raise for missing model, not for sampling.
        with self.assertRaises(ValueError) as ctx:
            lc2d_mixed.fit()
        self.assertIn('must provide a model', str(ctx.exception))

    def test_init_band_filter_fit_works_on_filtered_data(self):
        """fit() works correctly after band filtering at __init__ time."""
        # Band 3.6: 50 good points (passes); band 4.5: 4 points (fails)
        t_good = np.linspace(0, 100, 50)
        t_bad = [0.0, 1.0, 2.0, 3.0]
        t_all = list(t_good) + t_bad
        wl_all = [3.6] * 50 + [4.5] * 4
        y_all = (np.sin(2 * np.pi * np.array(t_all) / 10.0) + 1.0).tolist()
        ye_all = [0.01] * 54
        xdata = np.column_stack([t_all, wl_all])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lc = Lightcurve(
                torch.as_tensor(xdata, dtype=torch.float32),
                torch.as_tensor(y_all, dtype=torch.float32),
                yerr=torch.as_tensor(ye_all, dtype=torch.float32),
                check_sampling=True,
            )
        # Only the well-sampled band should remain after init.
        remaining_wls = torch.unique(lc._xdata_raw[:, 1]).tolist()
        self.assertEqual(len(remaining_wls), 1)
        self.assertAlmostEqual(remaining_wls[0], 3.6, places=4)

        # fit() on the filtered data should complete successfully.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = lc.fit(
                model='2D',
                use_mls_init=False,
                training_iter=5,
                miniter=2,
                lr=0.05,
            )
        self.assertIsNotNone(results)
        self.assertIsNotNone(lc.model)
        # Likelihood is built for the 50 remaining points.
        self.assertIsInstance(
            lc.likelihood,
            gpytorch.likelihoods.FixedNoiseGaussianLikelihood,
        )
        self.assertEqual(lc.likelihood.noise.numel(), 50)

    def test_init_band_filter_model_instance_bound_to_filtered_data(self):
        """fit() with a pre-set GP instance works on init-filtered data."""
        from pgmuvi.gps import TwoDSpectralMixtureGPModel
        # Band 3.6: 50 good points; band 4.5: 4 points (fails)
        t_good = np.linspace(0, 100, 50)
        t_bad = [0.0, 1.0, 2.0, 3.0]
        t_all = list(t_good) + t_bad
        wl_all = [3.6] * 50 + [4.5] * 4
        y_all = (np.sin(2 * np.pi * np.array(t_all) / 10.0) + 1.0).tolist()
        ye_all = [0.01] * 54
        xdata = np.column_stack([t_all, wl_all])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lc = Lightcurve(
                torch.as_tensor(xdata, dtype=torch.float32),
                torch.as_tensor(y_all, dtype=torch.float32),
                yerr=torch.as_tensor(ye_all, dtype=torch.float32),
                check_sampling=True,
            )
        # After init, only 50 points from band 3.6 remain.
        n_filtered = lc._xdata_raw.shape[0]
        self.assertEqual(n_filtered, 50)

        # Build a GP instance on the filtered data.
        lik = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            lc.yerr ** 2
        )
        gp_instance = TwoDSpectralMixtureGPModel(
            lc._xdata_transformed,
            lc._ydata_transformed,
            lik,
            num_mixtures=2,
        )
        lc.set_model(gp_instance, likelihood=None, num_mixtures=2)
        self.assertIs(lc.model, gp_instance)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = lc.fit(
                use_mls_init=False,
                training_iter=5,
                miniter=2,
                lr=0.05,
            )
        self.assertIs(lc.model, gp_instance)
        self.assertEqual(lc.model.train_inputs[0].shape[0], 50)
        self.assertIsNotNone(results)
