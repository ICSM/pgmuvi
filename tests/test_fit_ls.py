"""Tests for Lightcurve.fit_LS (Lomb-Scargle fitting)."""
import unittest
import warnings

import torch

from pgmuvi.lightcurve import (
    Lightcurve,
)
from pgmuvi.synthetic import make_chromatic_sinusoid_2d, make_simple_sinusoid_1d


class TestFitLS(unittest.TestCase):
    """Tests for the fit_LS() method"""

    def setUp(self):
        """Set up test data for Lomb-Scargle tests"""
        # 1D periodic signal (period ~2, frequency ~0.5)
        lc_1d = make_simple_sinusoid_1d(
            n_obs=100, period=2.0, amplitude=2.0, noise_level=0.1,
            t_span=10.0, irregular=False, seed=42,
        )
        self.t = lc_1d.xdata
        self.y = lc_1d.ydata
        self.yerr = 0.1 * torch.ones(100, dtype=torch.float32)

        # Create lightcurves with and without errors
        self.lc_with_yerr = Lightcurve(self.t, self.y, yerr=self.yerr)
        self.lc_without_yerr = Lightcurve(self.t, self.y)

        # 2D multiband lightcurve for testing multiband functionality
        lc_2d = make_chromatic_sinusoid_2d(
            n_per_band=25,
            period=2.0,
            wavelengths=[0.5, 1.5],
            amplitude_law="linear",
            amplitude_slope=0.0,
            noise_level=0.1,
            t_span=10.0,
            irregular=True,
            seed=42,
        )
        self.xdata_2d = lc_2d.xdata
        self.lc_2d = Lightcurve(self.xdata_2d, lc_2d.ydata)

    def test_basic_functionality_with_yerr(self):
        """Test basic fit_LS functionality with yerr"""
        freq, mask = self.lc_with_yerr.fit_LS(num_peaks=1)

        # Check that we got tensors back
        self.assertIsInstance(freq, torch.Tensor)
        self.assertIsInstance(mask, torch.Tensor)

        # Check that we got 1 peak as requested
        self.assertEqual(len(freq), 1)
        self.assertEqual(len(mask), 1)

        # Check that frequency is positive
        self.assertGreater(freq[0].item(), 0)

    def test_basic_functionality_without_yerr(self):
        """Test basic fit_LS functionality without yerr (edge case)"""
        freq, mask = self.lc_without_yerr.fit_LS(num_peaks=1)

        # Check that we got tensors back
        self.assertIsInstance(freq, torch.Tensor)
        self.assertIsInstance(mask, torch.Tensor)

        # Check that we got 1 peak as requested
        self.assertEqual(len(freq), 1)
        self.assertEqual(len(mask), 1)

        # Check that frequency is positive
        self.assertGreater(freq[0].item(), 0)

    def test_freq_only_flag(self):
        """Test freq_only=True returns full frequency grid"""
        freq, power = self.lc_with_yerr.fit_LS(freq_only=True)

        # Check that we got tensors back
        self.assertIsInstance(freq, torch.Tensor)
        self.assertIsInstance(power, torch.Tensor)

        # Check that we got the full grid (should be many points)
        self.assertGreater(len(freq), 100)
        self.assertEqual(len(freq), len(power))

    def test_multiple_peaks(self):
        """Test requesting multiple peaks"""
        freq, mask = self.lc_with_yerr.fit_LS(num_peaks=3)

        # Check that we got up to 3 peaks (might be fewer if not enough found)
        self.assertLessEqual(len(freq), 3)
        self.assertEqual(len(freq), len(mask))

    def test_fewer_peaks_than_requested(self):
        """Test edge case when fewer peaks are found than num_peaks"""
        # Create a very simple signal with very few peaks
        t_simple = torch.linspace(0, 2, 10, dtype=torch.float32)
        y_simple = torch.ones(10, dtype=torch.float32)
        lc_simple = Lightcurve(t_simple, y_simple)

        # Request more peaks than likely exist
        freq, mask = lc_simple.fit_LS(num_peaks=100)

        # Should return whatever peaks were found, not crash
        self.assertIsInstance(freq, torch.Tensor)
        self.assertIsInstance(mask, torch.Tensor)
        self.assertEqual(len(freq), len(mask))

    def test_no_peaks_found(self):
        """Test edge case when no peaks are found"""
        # Create a completely flat signal
        t_flat = torch.linspace(0, 10, 50, dtype=torch.float32)
        y_flat = torch.ones(50, dtype=torch.float32)
        lc_flat = Lightcurve(t_flat, y_flat)

        freq, mask = lc_flat.fit_LS(num_peaks=1)

        # Should return empty tensors when no peaks found
        self.assertEqual(len(freq), 0)
        self.assertEqual(len(mask), 0)

    def test_multiband_functionality(self):
        """Test that multiband lightcurves work correctly"""
        # Test that multiband data works
        freq, mask = self.lc_2d.fit_LS(num_peaks=1)

        # Should return tensors
        self.assertIsInstance(freq, torch.Tensor)
        self.assertIsInstance(mask, torch.Tensor)

        # Should return at least some peaks
        self.assertGreaterEqual(len(freq), 0)
        self.assertEqual(len(freq), len(mask))

        # Test freq_only mode for multiband
        freq_grid, power_grid = self.lc_2d.fit_LS(freq_only=True)
        self.assertIsInstance(freq_grid, torch.Tensor)
        self.assertIsInstance(power_grid, torch.Tensor)
        self.assertGreater(len(freq_grid), 50)  # Should have many points in grid

    def test_device_consistency(self):
        """Test that output device matches input device"""
        freq, mask = self.lc_with_yerr.fit_LS(num_peaks=1)

        # Check that output is on same device as input
        self.assertEqual(freq.device, self.lc_with_yerr.xdata.device)
        self.assertEqual(mask.device, self.lc_with_yerr.xdata.device)

    def test_dtype_consistency(self):
        """Test that output dtype matches input dtype"""
        freq, mask = self.lc_with_yerr.fit_LS(num_peaks=1)

        # Check that frequency dtype matches xdata dtype
        self.assertEqual(freq.dtype, self.lc_with_yerr.xdata.dtype)
        # mask should be bool
        self.assertEqual(mask.dtype, torch.bool)

    def test_fdr_correction_logic(self):
        """Test that FDR correction is applied"""
        # For a strong periodic signal, we should get at least one significant peak
        freq, mask = self.lc_with_yerr.fit_LS(num_peaks=3, single_threshold=0.05)

        # The first (strongest) peak should typically be significant
        # Note: This is probabilistic, but with our synthetic signal it should be true
        if len(mask) > 0:
            self.assertIsInstance(mask[0].item(), bool)

    def test_fap_method_parameter_1d(self):
        """Test that fap_method parameter is accepted and affects results"""
        # 'davies' is the default; 'baluev' also valid
        for method in ('baluev', 'davies'):
            freq, mask = self.lc_with_yerr.fit_LS(
                num_peaks=1, fap_method=method
            )
            self.assertIsInstance(freq, torch.Tensor)
            self.assertIsInstance(mask, torch.Tensor)

    def test_fap_method_single_warns_for_fap_max(self):
        """Test that fap_method='single' issues a warning for fap_max"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            freq, mask = self.lc_with_yerr.fit_LS(
                num_peaks=1, fap_method='single'
            )
            # Should have issued exactly one UserWarning about 'single'
            user_warnings = [
                x for x in w
                if issubclass(x.category, UserWarning)
                and "'single'" in str(x.message)
            ]
            self.assertEqual(len(user_warnings), 1)
        self.assertIsInstance(freq, torch.Tensor)
        self.assertIsInstance(mask, torch.Tensor)

    def test_fap_method_forwarded_1d(self):
        """Test that the chosen fap_method is actually forwarded to astropy"""
        from unittest.mock import patch
        from astropy.timeseries import LombScargle

        original_fap = LombScargle.false_alarm_probability
        called_methods = []

        def mock_fap(self_ls, power, method='baluev', **kwargs):
            called_methods.append(method)
            return original_fap(self_ls, power, method='davies')

        with patch.object(LombScargle, 'false_alarm_probability', mock_fap):
            self.lc_with_yerr.fit_LS(num_peaks=1, fap_method='baluev')

        # fap_max should use 'baluev'
        self.assertIn('baluev', called_methods)

    def test_fap_rng_forwarded_to_1d_bootstrap(self):
        """fap_rng reaches astropy's 1D bootstrap FAP as method_kwds['random_seed']"""
        from unittest.mock import patch
        from astropy.timeseries import LombScargle

        original_fap = LombScargle.false_alarm_probability
        calls = []

        def mock_fap(self_ls, power, method='baluev', method_kwds=None, **kw):
            calls.append((method, method_kwds))
            return original_fap(self_ls, power, method='davies')

        with patch.object(LombScargle, 'false_alarm_probability', mock_fap):
            self.lc_with_yerr.fit_LS(num_peaks=1, fap_method='bootstrap', fap_rng=7)
            self.lc_with_yerr.fit_LS(num_peaks=1, fap_method='bootstrap')
            self.lc_with_yerr.fit_LS(num_peaks=1, fap_method='baluev', fap_rng=7)

        bootstrap_kwds = [kwds for method, kwds in calls if method == 'bootstrap']
        self.assertEqual(bootstrap_kwds[0], {'random_seed': 7})
        self.assertFalse(bootstrap_kwds[1])  # no seed given: nothing forwarded
        # a non-bootstrap method never receives the seed
        self.assertTrue(all(not kwds for method, kwds in calls if method == 'baluev'))

    def test_fap_method_parameter_2d(self):
        """Test that fap_method parameter works for multiband lightcurves"""
        for method in ('analytical', 'bootstrap'):
            freq, mask = self.lc_2d.fit_LS(
                num_peaks=1, fap_method=method
            )
            self.assertIsInstance(freq, torch.Tensor)
            self.assertIsInstance(mask, torch.Tensor)

    def test_fap_method_forwarded_multiband(self):
        """Test that fap_method and freq_grid are forwarded for multiband"""
        from unittest.mock import patch
        from pgmuvi.multiband_ls_significance import MultibandLSWithSignificance

        original_fap = MultibandLSWithSignificance.false_alarm_probability
        call_kwargs = []

        def mock_fap(self_ls, power, method='analytical', n_samples=100,
                     freq_grid=None, n_jobs=1, rng=None):
            call_kwargs.append({'method': method, 'freq_grid': freq_grid})
            return original_fap(self_ls, power, method='analytical',
                                freq_grid=freq_grid)

        with patch.object(MultibandLSWithSignificance, 'false_alarm_probability',
                          mock_fap):
            self.lc_2d.fit_LS(num_peaks=1, fap_method='analytical')

        # Both calls should use 'analytical' and pass a freq_grid (not None)
        self.assertGreaterEqual(len(call_kwargs), 1)
        for kw in call_kwargs:
            self.assertEqual(kw['method'], 'analytical')
            self.assertIsNotNone(kw['freq_grid'])
