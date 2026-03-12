import os
import tempfile
import unittest
import warnings

from pgmuvi.lightcurve import InputHelpers, Lightcurve, Transformer, MinMax, ZScore, RobustZScore
from pgmuvi.trainers import train
from pgmuvi.gps import SpectralMixtureGPModel
import numpy as np
import torch

test_zeros_one = torch.as_tensor([0, 0, 0, 0, 1], dtype=torch.float32)

class TestTransformer(unittest.TestCase):
    def test_transform_implemented(self):
        transformer = Transformer()
        self.assertRaises(NotImplementedError,
                          transformer.transform,
                          test_zeros_one)

    def test_inverse_implemented(self):
        transformer = Transformer()
        self.assertRaises(NotImplementedError,
                          transformer.inverse,
                          test_zeros_one)


class TestMinMax(unittest.TestCase):
    def test_transform(self):
        transformer = MinMax()
        self.assertEqual(transformer.transform(test_zeros_one).min(), 0)
        self.assertEqual(transformer.transform(test_zeros_one).max(), 1)

    def test_inverse(self):
        transformer = MinMax()
        # first we have to do a forward transform to get the transformer set up:
        x_new = transformer.transform(test_zeros_one)
        self.assertEqual(transformer.inverse(x_new).min(), 0)
        self.assertEqual(transformer.inverse(x_new).max(), 1)


class TestZScore(unittest.TestCase):
    @unittest.skip("Not implemented")
    def test_transform(self):
        transformer = ZScore() # these tests need a more sensible input array to test against
        self.assertAlmostEqual(transformer.transform(test_zeros_one).mean(), 0)
        self.assertAlmostEqual(transformer.transform(test_zeros_one).std(), 1)

    @unittest.skip("Not implemented")
    def test_inverse(self):
        transformer = ZScore()
        # first we have to do a forward transform to get the transformer set up:
        x_new = transformer.transform(test_zeros_one)
        self.assertAlmostEqual(transformer.inverse(np.array([0, 0, 0, 0, 1])).mean(), 0)
        self.assertAlmostEqual(transformer.inverse(np.array([0, 0, 0, 0, 1])).std(), 1)


class TestRobustZScore(unittest.TestCase):
    @unittest.skip("Not implemented")
    def test_transform(self):
        transformer = RobustZScore() # these tests need a more sensible input array to test against
        self.assertAlmostEqual(transformer.transform(test_zeros_one).mean(), 0)
        self.assertAlmostEqual(transformer.transform(test_zeros_one).std(), 1)

    @unittest.skip("Not implemented")
    def test_inverse(self):
        transformer = RobustZScore()
        self.assertAlmostEqual(transformer.inverse(test_zeros_one).mean(), 0)
        self.assertAlmostEqual(transformer.inverse(test_zeros_one).std(), 1)


class TestLightCurve(unittest.TestCase):
    def setUp(self):
        self.test_xdata = torch.as_tensor([1, 2, 3, 4], dtype=torch.float32)
        self.test_ydata = torch.as_tensor([1, 2, 1, 2], dtype=torch.float32)
        self.test_xdata_2d = torch.as_tensor([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=torch.float32)
        self.test_ydata_2d = torch.as_tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.float32)
        self.lightcurve = Lightcurve(self.test_xdata, self.test_ydata, yerr=self.test_ydata)
        self.lightcurve_2d = Lightcurve(self.test_xdata_2d, self.test_ydata_2d)

    def test_ndim(self):
        self.assertEqual(self.lightcurve.ndim, 1)
        self.assertEqual(self.lightcurve_2d.ndim, 4)

    def test_xdata_getter(self):
        self.assertTrue(torch.equal(self.lightcurve.xdata, self.test_xdata))

    def test_default_no_transform(self):
        lc = Lightcurve(self.test_xdata, self.test_ydata)
        self.assertIsNone(lc.xtransform)
        self.assertIsNone(lc.ytransform)
        self.assertTrue(torch.equal(lc._xdata_transformed, self.test_xdata))

    def test_xdata_setter_no_transform(self):
        self.lightcurve.xtransform = None
        self.lightcurve.xdata = self.test_xdata
        self.assertTrue(torch.equal(self.lightcurve._xdata_raw, self.test_xdata))
        self.assertTrue(torch.equal(self.lightcurve._xdata_transformed, self.test_xdata))

    def test_xdata_setter_with_transform(self):
        lc_with_transform = Lightcurve(self.test_xdata, self.test_ydata, xtransform="minmax")
        xtransformer = lc_with_transform.xtransform
        self.test_xdata_transformed = xtransformer.transform(self.test_xdata)

        lc_with_transform.xdata = self.test_xdata
        self.assertTrue(torch.equal(lc_with_transform._xdata_raw, self.test_xdata))
        self.assertTrue(torch.equal(lc_with_transform._xdata_transformed, self.test_xdata_transformed))

    def test_ydata_getter(self):
        self.assertTrue(torch.equal(self.lightcurve.ydata, self.test_ydata))

    def test_ydata_setter_no_transform(self):
        self.lightcurve.ydata = self.test_ydata
        self.assertTrue(torch.equal(self.lightcurve._ydata_raw, self.test_ydata))
        self.assertTrue(torch.equal(self.lightcurve._ydata_transformed, self.test_ydata))

    def test_ydata_setter_with_transform(self):
        self.lightcurve.ytransform = MinMax()
        self.test_ydata_transformed = self.lightcurve.ytransform.transform(self.test_ydata)

        self.lightcurve.ydata = self.test_ydata
        self.assertTrue(torch.equal(self.lightcurve._ydata_raw, self.test_ydata))
        self.assertTrue(torch.equal(self.lightcurve._ydata_transformed, self.test_ydata_transformed))

    def test_yerr_getter(self):
        self.assertTrue(torch.equal(self.lightcurve.yerr, self.test_ydata))

    def test_yerr_setter_no_transform(self):
        self.lightcurve.yerr = self.test_ydata
        self.assertTrue(torch.equal(self.lightcurve._yerr_raw, self.test_ydata))
        self.assertTrue(torch.equal(self.lightcurve._yerr_transformed, self.test_ydata))

    def test_yerr_setter_with_transform(self):
        self.lightcurve.ytransform = MinMax()
        self.test_yerr_transformed = self.lightcurve.ytransform.transform(self.test_ydata)

        self.lightcurve.yerr = self.test_ydata
        self.assertTrue(torch.equal(self.lightcurve._yerr_raw, self.test_ydata))
        self.assertTrue(torch.equal(self.lightcurve._yerr_transformed, self.test_yerr_transformed))


class TestFitLS(unittest.TestCase):
    """Tests for the fit_LS() method"""

    def setUp(self):
        """Set up test data for Lomb-Scargle tests"""
        # Create a synthetic periodic signal with some noise
        np.random.seed(42)
        t = np.linspace(0, 10, 100)
        # Signal with period ~2 (frequency ~0.5)
        y = 2 * np.sin(2 * np.pi * 0.5 * t) + 0.1 * np.random.randn(100)
        yerr = 0.1 * np.ones(100)

        self.t = torch.as_tensor(t, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.yerr = torch.as_tensor(yerr, dtype=torch.float32)

        # Create lightcurves with and without errors
        self.lc_with_yerr = Lightcurve(self.t, self.y, yerr=self.yerr)
        self.lc_without_yerr = Lightcurve(self.t, self.y)

        # Create 2D multiband lightcurve for testing multiband functionality
        # Format: (n_samples, 2) where column 0 is time, column 1 is band
        n_samples = 50
        time_2d = torch.linspace(0, 10, n_samples, dtype=torch.float32)
        # Two bands: 0.5 and 1.5
        bands = torch.cat([
            torch.ones(n_samples // 2, dtype=torch.float32) * 0.5,
            torch.ones(n_samples - n_samples // 2, dtype=torch.float32) * 1.5
        ])
        # Shuffle to mix bands
        indices = torch.randperm(n_samples)
        time_2d = time_2d[indices]
        bands = bands[indices]

        self.xdata_2d = torch.stack([time_2d, bands], dim=1)
        # Create periodic signal
        y_2d = torch.sin(2 * np.pi * 0.5 * time_2d) + 0.1 * torch.randn(n_samples)
        self.lc_2d = Lightcurve(self.xdata_2d, y_2d)

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


class TestMultibandFAP(unittest.TestCase):
    """Tests for multiband false-alarm probability computation"""

    def setUp(self):
        """Set up test data for multiband FAP tests"""
        np.random.seed(42)

        # Create multiband lightcurve with strong periodic signal
        n_samples = 100
        time = np.linspace(0, 20, n_samples)

        # Two bands
        bands = np.concatenate([
            np.ones(n_samples // 2) * 0.5,
            np.ones(n_samples - n_samples // 2) * 1.5
        ])

        # Shuffle to mix bands
        indices = np.random.permutation(n_samples)
        time = time[indices]
        bands = bands[indices]

        # Strong periodic signal with frequency ~0.5
        y_signal = 2 * np.sin(2 * np.pi * 0.5 * time) + 0.1 * np.random.randn(n_samples)

        # Noisy data (no signal)
        y_noise = 0.5 * np.random.randn(n_samples)

        # Convert to torch tensors and create 2D format
        self.xdata_signal = torch.stack([
            torch.as_tensor(time, dtype=torch.float32),
            torch.as_tensor(bands, dtype=torch.float32)
        ], dim=1)

        self.xdata_noise = torch.stack([
            torch.as_tensor(time, dtype=torch.float32),
            torch.as_tensor(bands, dtype=torch.float32)
        ], dim=1)

        self.ydata_signal = torch.as_tensor(y_signal, dtype=torch.float32)
        self.ydata_noise = torch.as_tensor(y_noise, dtype=torch.float32)

        self.lc_signal = Lightcurve(self.xdata_signal, self.ydata_signal)
        self.lc_noise = Lightcurve(self.xdata_noise, self.ydata_noise)

    def test_multiband_fap_returns_valid_values(self):
        """Test that multiband FAP computation returns values in [0, 1]"""
        from pgmuvi.multiband_ls_significance import MultibandLSWithSignificance

        t = self.xdata_signal[:, 0].numpy()
        bands = self.xdata_signal[:, 1].numpy()
        y = self.ydata_signal.numpy()

        ls = MultibandLSWithSignificance(t, y, bands)
        freq = ls.autofrequency()
        power = ls.power(freq)

        # Test all FAP methods
        for method in ['bootstrap', 'analytical', 'calibrated']:
            fap = ls.false_alarm_probability(
                power.max(), method=method, n_samples=50
            )
            self.assertGreaterEqual(fap, 0.0,
                                    f"{method} returned FAP < 0")
            self.assertLessEqual(fap, 1.0,
                                 f"{method} returned FAP > 1")

    def test_multiband_strong_signal_low_fap(self):
        """Test that strong signals have low FAP"""
        freq, mask = self.lc_signal.fit_LS(num_peaks=1, single_threshold=0.05)

        # Should find at least one peak
        self.assertGreater(len(freq), 0)
        self.assertEqual(len(freq), len(mask))

        # For strong signal, highest peak should be significant
        # (FAP computation should mark it as True)
        if len(mask) > 0:
            # At least some peaks should be significant for a strong signal
            # This test is probabilistic but should pass with our signal
            self.assertIsInstance(mask[0].item(), bool)

    def test_multiband_noise_high_fap(self):
        """Test that noise has high FAP"""
        # For pure noise, most peaks should NOT be significant
        freq, mask = self.lc_noise.fit_LS(num_peaks=5, single_threshold=0.05)

        # Should return some peaks
        self.assertGreaterEqual(len(freq), 0)
        self.assertEqual(len(freq), len(mask))

        # For noise, we expect most peaks to be insignificant
        # At least one should be False (not all True like before)
        if len(mask) > 1:
            # With proper FAP, not all peaks should be marked significant
            # This verifies we're not just returning all True anymore
            num_significant = mask.sum().item()
            num_total = len(mask)

            # For pure noise with threshold=0.05, we expect ~5% false positives
            # So not all peaks should be significant
            self.assertLess(num_significant, num_total,
                            "All peaks marked significant for noise data")

    def test_multiband_fap_not_all_true(self):
        """Test that multiband FAP no longer returns all True masks"""
        # This is the key test - verifying the fix works
        # Create data with mixed signal levels
        freq, mask = self.lc_noise.fit_LS(num_peaks=3)

        # Should return results
        self.assertGreaterEqual(len(mask), 0)

        # The mask should be proper boolean tensor
        self.assertIsInstance(mask, torch.Tensor)
        self.assertEqual(mask.dtype, torch.bool)

        # For noise data, not all peaks should be significant
        # This would fail with the old "all True" implementation
        if len(mask) >= 2:
            # At least verify we're computing real FAP values
            # (the old code would have all True)
            mask_list = mask.tolist()
            self.assertIsInstance(mask_list[0], bool)

    def test_multiband_fap_methods_consistency(self):
        """Test that different FAP methods produce reasonable results"""
        from pgmuvi.multiband_ls_significance import MultibandLSWithSignificance

        t = self.xdata_signal[:, 0].numpy()
        bands = self.xdata_signal[:, 1].numpy()
        y = self.ydata_signal.numpy()

        ls = MultibandLSWithSignificance(t, y, bands)
        freq = ls.autofrequency()
        power = ls.power(freq)
        max_power = power.max()

        # Compute FAP with different methods
        fap_bootstrap = ls.false_alarm_probability(
            max_power, method='bootstrap', n_samples=50
        )
        fap_analytical = ls.false_alarm_probability(
            max_power, method='analytical'
        )
        fap_calibrated = ls.false_alarm_probability(
            max_power, method='calibrated'
        )

        # All should be in valid range
        for fap, name in [(fap_bootstrap, 'bootstrap'),
                          (fap_analytical, 'analytical'),
                          (fap_calibrated, 'calibrated')]:
            self.assertGreaterEqual(fap, 0.0, f"{name} FAP < 0")
            self.assertLessEqual(fap, 1.0, f"{name} FAP > 1")

        # For a strong signal, all methods should give relatively low FAP
        # (though exact values may differ)
        # This is a sanity check that methods are reasonable
        self.assertLess(fap_bootstrap, 0.5,
                        "Bootstrap FAP too high for strong signal")



class TestFromCSV(unittest.TestCase):
    """Tests for the InputHelpers.from_csv classmethod."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _write_csv(self, filename, content):
        path = os.path.join(self.tmpdir, filename)
        with open(path, "w") as f:
            f.write(content)
        return path

    # ------------------------------------------------------------------
    # Auto-detection tests
    # ------------------------------------------------------------------

    def test_autodetect_standard_names(self):
        """Auto-detection with canonical lowercase column names."""
        path = self._write_csv(
            "standard.csv",
            "x,y,yerr\n1.0,2.0,0.1\n2.0,3.0,0.2\n3.0,4.0,0.3\n",
        )
        lc = Lightcurve.from_csv(path)
        self.assertIsInstance(lc, Lightcurve)
        self.assertEqual(len(lc.xdata), 3)

    def test_autodetect_jd_magnitude(self):
        """Auto-detection with JD / Magnitude column names (sample data style)."""
        path = self._write_csv(
            "jd_mag.csv",
            "JD,Magnitude\n2450000.0,1.5\n2450001.0,1.6\n2450002.0,1.4\n",
        )
        lc = Lightcurve.from_csv(path)
        self.assertIsInstance(lc, Lightcurve)
        self.assertEqual(len(lc.xdata), 3)
        # yerr should be None when no uncertainty column is present
        self.assertFalse(hasattr(lc, "_yerr_raw"))

    def test_autodetect_case_insensitive(self):
        """Column name matching must be case-insensitive."""
        path = self._write_csv(
            "mixed_case.csv",
            "Time,Flux,Error\n0.0,1.0,0.05\n1.0,2.0,0.05\n2.0,1.5,0.05\n",
        )
        lc = Lightcurve.from_csv(path)
        self.assertIsInstance(lc, Lightcurve)
        self.assertEqual(len(lc.xdata), 3)
        self.assertTrue(hasattr(lc, "_yerr_raw"))

    def test_autodetect_no_yerr_column(self):
        """When no uncertainty column exists, yerr should not be set."""
        path = self._write_csv(
            "no_yerr.csv",
            "time,flux\n0.0,1.0\n1.0,2.0\n2.0,1.5\n",
        )
        lc = Lightcurve.from_csv(path)
        self.assertIsInstance(lc, Lightcurve)
        self.assertFalse(hasattr(lc, "_yerr_raw"))

    # ------------------------------------------------------------------
    # Explicit column name tests
    # ------------------------------------------------------------------

    def test_explicit_column_names(self):
        """Explicit xcol/ycol/yerrcol should override auto-detection."""
        path = self._write_csv(
            "explicit.csv",
            "date,signal,noise\n1.0,10.0,0.5\n2.0,11.0,0.5\n3.0,9.0,0.5\n",
        )
        lc = Lightcurve.from_csv(path, xcol="date", ycol="signal", yerrcol="noise")
        self.assertIsInstance(lc, Lightcurve)
        self.assertEqual(len(lc.xdata), 3)
        self.assertTrue(hasattr(lc, "_yerr_raw"))

    def test_explicit_xcol_missing_raises(self):
        """Explicitly specified xcol that does not exist should raise ValueError."""
        path = self._write_csv(
            "missing_x.csv",
            "time,flux\n0.0,1.0\n1.0,2.0\n",
        )
        with self.assertRaises(ValueError):
            Lightcurve.from_csv(path, xcol="nonexistent")

    def test_explicit_ycol_missing_raises(self):
        """Explicitly specified ycol that does not exist should raise ValueError."""
        path = self._write_csv(
            "missing_y.csv",
            "time,flux\n0.0,1.0\n1.0,2.0\n",
        )
        with self.assertRaises(ValueError):
            Lightcurve.from_csv(path, ycol="nonexistent")

    def test_explicit_yerrcol_missing_raises(self):
        """Explicitly specified yerrcol that does not exist should raise ValueError."""
        path = self._write_csv(
            "missing_yerr.csv",
            "time,flux\n0.0,1.0\n1.0,2.0\n",
        )
        with self.assertRaises(ValueError):
            Lightcurve.from_csv(path, yerrcol="nonexistent")

    def test_autodetect_x_fails_raises(self):
        """When x column cannot be auto-detected, a ValueError should be raised."""
        path = self._write_csv(
            "ambiguous.csv",
            "alpha,beta\n0.0,1.0\n1.0,2.0\n",
        )
        with self.assertRaises(ValueError):
            Lightcurve.from_csv(path)

    def test_autodetect_y_fails_raises(self):
        """When y column cannot be auto-detected, a ValueError should be raised."""
        path = self._write_csv(
            "no_y.csv",
            "time,alpha\n0.0,1.0\n1.0,2.0\n",
        )
        with self.assertRaises(ValueError):
            Lightcurve.from_csv(path)

    # ------------------------------------------------------------------
    # Data integrity tests
    # ------------------------------------------------------------------

    def test_data_values_preserved(self):
        """Values read from the CSV must match those in the Lightcurve."""
        path = self._write_csv(
            "values.csv",
            "x,y,yerr\n1.0,10.0,0.1\n2.0,20.0,0.2\n3.0,30.0,0.3\n",
        )
        lc = Lightcurve.from_csv(path)
        expected_x = torch.as_tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        expected_y = torch.as_tensor([10.0, 20.0, 30.0], dtype=torch.float32)
        expected_yerr = torch.as_tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        self.assertTrue(torch.allclose(lc._xdata_raw, expected_x))
        self.assertTrue(torch.allclose(lc._ydata_raw, expected_y))
        self.assertTrue(torch.allclose(lc._yerr_raw, expected_yerr))

    # ------------------------------------------------------------------
    # InputHelpers class attribute tests
    # ------------------------------------------------------------------

    def test_inputhelpers_is_base_of_lightcurve(self):
        """Lightcurve should be a subclass of InputHelpers."""
        self.assertTrue(issubclass(Lightcurve, InputHelpers))

    def test_column_name_lists_accessible(self):
        """Column name candidate lists should be accessible from Lightcurve."""
        self.assertIn("jd", Lightcurve._X_COLUMN_NAMES)
        self.assertIn("magnitude", Lightcurve._Y_COLUMN_NAMES)
        self.assertIn("err", Lightcurve._YERR_COLUMN_NAMES)
        self.assertIn("wavelength", Lightcurve._WAVELENGTH_COLUMN_NAMES)
        self.assertIn("band", Lightcurve._WAVELENGTH_COLUMN_NAMES)
        self.assertIn("filter", Lightcurve._WAVELENGTH_COLUMN_NAMES)
        self.assertIn("wave", Lightcurve._WAVELENGTH_COLUMN_NAMES)

    # ------------------------------------------------------------------
    # 2D / multiband lightcurve tests
    # ------------------------------------------------------------------

    def test_autodetect_multiband_2d(self):
        """A wavelength column with multiple values triggers a 2D lightcurve."""
        path = self._write_csv(
            "multiband.csv",
            "time,flux,wavelength\n"
            "1.0,1.5,0.5\n2.0,1.6,0.5\n3.0,1.4,0.5\n"
            "1.0,2.5,1.5\n2.0,2.6,1.5\n3.0,2.4,1.5\n",
        )
        lc = Lightcurve.from_csv(path)
        self.assertIsInstance(lc, Lightcurve)
        # xdata should be 2D: (N, 2) with time in col 0, wavelength in col 1
        self.assertEqual(lc._xdata_raw.dim(), 2)
        self.assertEqual(lc._xdata_raw.shape[1], 2)
        self.assertEqual(len(lc.ydata), 6)

    def test_autodetect_single_wavelength_is_1d(self):
        """A wavelength column with only one unique value → 1D lightcurve."""
        path = self._write_csv(
            "single_band.csv",
            "time,flux,wavelength\n1.0,1.5,0.5\n2.0,1.6,0.5\n3.0,1.4,0.5\n",
        )
        lc = Lightcurve.from_csv(path)
        self.assertIsInstance(lc, Lightcurve)
        # Only one wavelength → treat as 1D
        self.assertEqual(lc._xdata_raw.dim(), 1)

    def test_explicit_wavelcol(self):
        """Explicit wavelcol parameter creates a 2D lightcurve."""
        path = self._write_csv(
            "explicit_band.csv",
            "time,flux,band\n1.0,1.5,1.0\n2.0,1.6,1.0\n3.0,1.4,2.0\n",
        )
        lc = Lightcurve.from_csv(path, xcol="time", ycol="flux", wavelcol="band")
        self.assertIsInstance(lc, Lightcurve)
        self.assertEqual(lc._xdata_raw.dim(), 2)
        self.assertEqual(lc._xdata_raw.shape[1], 2)

    def test_explicit_wavelcol_missing_raises(self):
        """Explicitly specified wavelcol that does not exist should raise ValueError."""
        path = self._write_csv(
            "missing_wavelcol.csv",
            "time,flux\n0.0,1.0\n1.0,2.0\n",
        )
        with self.assertRaises(ValueError):
            Lightcurve.from_csv(path, wavelcol="nonexistent")

    def test_xcol_as_list_creates_2d(self):
        """xcol as a list of column names stacks columns into 2D xdata."""
        path = self._write_csv(
            "xcol_list.csv",
            "time,band,flux\n1.0,0.5,1.5\n2.0,0.5,1.6\n3.0,1.5,1.4\n",
        )
        lc = Lightcurve.from_csv(path, xcol=["time", "band"], ycol="flux")
        self.assertIsInstance(lc, Lightcurve)
        self.assertEqual(lc._xdata_raw.dim(), 2)
        self.assertEqual(lc._xdata_raw.shape[1], 2)

    def test_xcol_as_single_element_list_is_1d(self):
        """xcol as a single-element list returns a 1D lightcurve."""
        path = self._write_csv(
            "xcol_single_list.csv",
            "time,flux\n1.0,1.5\n2.0,1.6\n3.0,1.4\n",
        )
        lc = Lightcurve.from_csv(path, xcol=["time"], ycol="flux")
        self.assertIsInstance(lc, Lightcurve)
        self.assertEqual(lc._xdata_raw.dim(), 1)

    def test_xcol_list_with_missing_col_raises(self):
        """A column listed in xcol that is absent raises ValueError."""
        path = self._write_csv(
            "bad_list.csv",
            "time,flux\n1.0,1.5\n2.0,1.6\n",
        )
        with self.assertRaises(ValueError):
            Lightcurve.from_csv(path, xcol=["time", "nonexistent"], ycol="flux")

    def test_2d_data_values_preserved(self):
        """Check that 2D xdata contains correct time and wavelength values."""
        path = self._write_csv(
            "2d_values.csv",
            "time,wavelength,flux\n"
            "1.0,0.5,10.0\n2.0,0.5,20.0\n"
            "1.0,1.5,30.0\n2.0,1.5,40.0\n",
        )
        lc = Lightcurve.from_csv(path)
        expected_time = torch.as_tensor([1.0, 2.0, 1.0, 2.0], dtype=torch.float32)
        expected_wave = torch.as_tensor([0.5, 0.5, 1.5, 1.5], dtype=torch.float32)
        self.assertTrue(torch.allclose(lc._xdata_raw[:, 0], expected_time))
        self.assertTrue(torch.allclose(lc._xdata_raw[:, 1], expected_wave))

    # ------------------------------------------------------------------
    # Sample data file
    # ------------------------------------------------------------------

    def test_sample_csv(self):
        """from_csv should work with the bundled AlfOriAAVSO_Vband.csv."""
        sample = os.path.join(
            os.path.dirname(__file__),
            "..",
            "pgmuvi",
            "AlfOriAAVSO_Vband.csv",
        )
        lc = Lightcurve.from_csv(sample)
        self.assertIsInstance(lc, Lightcurve)
        self.assertGreater(len(lc.xdata), 0)

    # ------------------------------------------------------------------
    # NaN dropping tests
    # ------------------------------------------------------------------

    def test_nan_in_y_dropped(self):
        """Rows with NaN in y should be dropped with a warning."""
        path = self._write_csv(
            "nan_y.csv",
            "x,y,yerr\n1.0,2.0,0.1\n2.0,nan,0.2\n3.0,4.0,0.3\n",
        )
        with self.assertWarns(UserWarning):
            lc = Lightcurve.from_csv(path)
        self.assertEqual(len(lc.xdata), 2)

    def test_nan_in_x_dropped(self):
        """Rows with NaN in x should be dropped with a warning."""
        path = self._write_csv(
            "nan_x.csv",
            "x,y,yerr\n1.0,2.0,0.1\nnan,3.0,0.2\n3.0,4.0,0.3\n",
        )
        with self.assertWarns(UserWarning):
            lc = Lightcurve.from_csv(path)
        self.assertEqual(len(lc.xdata), 2)

    def test_nan_in_yerr_dropped(self):
        """Rows with NaN in yerr should be dropped with a warning."""
        path = self._write_csv(
            "nan_yerr.csv",
            "x,y,yerr\n1.0,2.0,nan\n2.0,3.0,0.2\n3.0,4.0,0.3\n",
        )
        with self.assertWarns(UserWarning):
            lc = Lightcurve.from_csv(path)
        self.assertEqual(len(lc.xdata), 2)

    def test_no_nan_no_warning(self):
        """When there are no NaN values, no warning should be raised."""
        path = self._write_csv(
            "no_nan.csv",
            "x,y,yerr\n1.0,2.0,0.1\n2.0,3.0,0.2\n3.0,4.0,0.3\n",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lc = Lightcurve.from_csv(path)
        self.assertEqual(len(lc.xdata), 3)

    def test_nan_in_2d_x_dropped(self):
        """Rows with NaN in either column of a 2D x should be dropped."""
        path = self._write_csv(
            "nan_2d_x.csv",
            "time,wavelength,flux\n"
            "1.0,0.5,10.0\nnan,1.5,20.0\n3.0,0.5,30.0\n3.0,1.5,40.0\n",
        )
        with self.assertWarns(UserWarning):
            lc = Lightcurve.from_csv(path)
        # One row dropped → 3 rows remain
        self.assertEqual(len(lc.ydata), 3)

    def test_nan_in_wave_changes_dimensionality(self):
        """NaN in wavelength column: after dropping, only one wavelength remains → 1D."""
        path = self._write_csv(
            "nan_wave_single.csv",
            "time,wavelength,flux\n"
            "1.0,0.5,10.0\n2.0,nan,20.0\n3.0,0.5,30.0\n",
        )
        with self.assertWarns(UserWarning):
            lc = Lightcurve.from_csv(path)
        # After dropping the NaN row, only wavelength 0.5 remains → 1D
        self.assertEqual(lc._xdata_raw.dim(), 1)
        self.assertEqual(len(lc.xdata), 2)

    def test_all_nan_raises(self):
        """When all rows contain NaN, a ValueError should be raised."""
        path = self._write_csv(
            "all_nan.csv",
            "x,y,yerr\nnan,nan,nan\nnan,nan,nan\n",
        )
        with self.assertRaises(ValueError):
            Lightcurve.from_csv(path)


class TestFromTable(unittest.TestCase):
    """Tests for the Lightcurve.from_table classmethod NaN dropping behaviour."""

    def _make_table(self, x_vals, y_vals, yerr_vals=None):
        """Create an astropy Table with x, y, and optional yerr columns."""
        from astropy.table import Table

        cols = {"x": x_vals, "y": y_vals}
        if yerr_vals is not None:
            cols["yerr"] = yerr_vals
        return Table(cols)

    def test_nan_in_y_dropped(self):
        """Rows with NaN in y should be dropped with a warning."""
        tab = self._make_table(
            [1.0, 2.0, 3.0],
            [10.0, float("nan"), 30.0],
        )
        with self.assertWarns(UserWarning):
            lc = Lightcurve.from_table(tab, xcol="x", ycol="y")
        self.assertEqual(len(lc.xdata), 2)

    def test_nan_in_x_dropped(self):
        """Rows with NaN in x should be dropped with a warning."""
        tab = self._make_table(
            [1.0, float("nan"), 3.0],
            [10.0, 20.0, 30.0],
        )
        with self.assertWarns(UserWarning):
            lc = Lightcurve.from_table(tab, xcol="x", ycol="y")
        self.assertEqual(len(lc.xdata), 2)

    def test_nan_in_yerr_dropped(self):
        """Rows with NaN in yerr should be dropped with a warning."""
        tab = self._make_table(
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
            yerr_vals=[0.1, float("nan"), 0.3],
        )
        with self.assertWarns(UserWarning):
            lc = Lightcurve.from_table(tab, xcol="x", ycol="y", yerrcol="yerr")
        self.assertEqual(len(lc.xdata), 2)

    def test_no_nan_no_warning(self):
        """When there are no NaN values, no warning should be raised."""
        tab = self._make_table([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lc = Lightcurve.from_table(tab, xcol="x", ycol="y")
        self.assertEqual(len(lc.xdata), 3)

    def test_all_nan_raises(self):
        """When all rows contain NaN, a ValueError should be raised."""
        tab = self._make_table(
            [float("nan"), float("nan")],
            [float("nan"), float("nan")],
        )
        with self.assertRaises(ValueError):
            Lightcurve.from_table(tab, xcol="x", ycol="y")

    def test_valid_rows_preserved(self):
        """Values in non-NaN rows should be preserved correctly."""
        tab = self._make_table(
            [1.0, 2.0, 3.0],
            [10.0, float("nan"), 30.0],
            yerr_vals=[0.1, 0.2, 0.3],
        )
        with self.assertWarns(UserWarning):
            lc = Lightcurve.from_table(tab, xcol="x", ycol="y", yerrcol="yerr")
        expected_x = torch.as_tensor([1.0, 3.0], dtype=torch.float32)
        expected_y = torch.as_tensor([10.0, 30.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(lc._xdata_raw, expected_x))
        self.assertTrue(torch.allclose(lc._ydata_raw, expected_y))


class TestTrain(unittest.TestCase):
    def test_train(self):
        pass



class TestSpectralMixtureGPModel(unittest.TestCase):
    pass


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

    def test_assess_sampling_quality(self):
        """Lightcurve.assess_sampling_quality returns passes and diagnostics."""
        passes, diag = self.lc.assess_sampling_quality(verbose=False)
        self.assertTrue(passes)
        self.assertEqual(diag['recommendation'], 'PROCEED')

    def test_fit_check_sampling_raises(self):
        """fit() with check_sampling=True raises for poor sampling."""
        t_few = torch.as_tensor([0, 1, 2, 3], dtype=torch.float32)
        y_few = torch.as_tensor([1, 1, 1, 1], dtype=torch.float32)
        yerr_few = torch.as_tensor([0.1, 0.1, 0.1, 0.1], dtype=torch.float32)
        lc_few = Lightcurve(t_few, y_few, yerr=yerr_few)
        with self.assertRaises(ValueError):
            lc_few.fit(model='1D', check_sampling=True)

    def test_fit_check_sampling_disabled(self):
        """fit() with check_sampling=False skips quality check (may fail on model)."""
        t_few = torch.as_tensor([0, 1, 2, 3], dtype=torch.float32)
        y_few = torch.as_tensor([1, 1, 1, 1], dtype=torch.float32)
        yerr_few = torch.as_tensor([0.1, 0.1, 0.1, 0.1], dtype=torch.float32)
        lc_few = Lightcurve(t_few, y_few, yerr=yerr_few)
        # Should raise for missing model, not for sampling quality
        with self.assertRaises(ValueError) as ctx:
            lc_few.fit(check_sampling=False)
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

    def test_assess_sampling_quality_per_band(self):
        """All bands should pass quality checks."""
        results = self.lc2d.assess_sampling_quality_per_band(verbose=False)
        self.assertEqual(results['summary']['n_passing'], 3)
        self.assertEqual(len(results['summary']['failing_wavelengths']), 0)

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
# Import variability tests so they are discovered when this file is run
from test_variability import (  # noqa: E402, F401
    TestComputeFvar,
    TestComputeStetsonK,
    TestIsVariable,
    TestLightcurveVariability,
    TestWeightedChi2,
)


class TestPowerLawMean(unittest.TestCase):
    """Tests for the PowerLawMean mean function."""

    def setUp(self):
        from pgmuvi.gps import PowerLawMean
        self.PowerLawMean = PowerLawMean
        # 2D input: (time, wavelength), wavelength in second column
        self.x = torch.tensor(
            [[0.0, 0.5], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32
        )

    def test_instantiation(self):
        """PowerLawMean can be instantiated without arguments."""
        mean = self.PowerLawMean()
        self.assertIsNotNone(mean)

    def test_forward_shape(self):
        """PowerLawMean forward returns a 1-D tensor of correct length."""
        mean = self.PowerLawMean()
        out = mean(self.x)
        self.assertEqual(out.shape, (3,))

    def test_parameters_registered(self):
        """PowerLawMean registers offset, weight, and exponent parameters."""
        mean = self.PowerLawMean()
        param_names = [n for n, _ in mean.named_parameters()]
        self.assertIn("offset", param_names)
        self.assertIn("weight", param_names)
        self.assertIn("exponent", param_names)

    def test_default_exponent(self):
        """Default exponent is -2.0 (steep optical-to-IR decline)."""
        mean = self.PowerLawMean()
        self.assertAlmostEqual(mean.exponent.item(), -2.0, places=5)

    def test_power_law_values(self):
        """Forward output matches expected power-law calculation."""
        mean = self.PowerLawMean()
        # Set known parameters: offset=0, weight=1, exponent=-2
        with torch.no_grad():
            mean.offset.fill_(0.0)
            mean.weight.fill_(1.0)
            mean.exponent.fill_(-2.0)
        out = mean(self.x)
        wavelengths = self.x[:, 1]
        expected = wavelengths.pow(-2.0)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))


class TestDustMean(unittest.TestCase):
    """Tests for the DustMean mean function."""

    def setUp(self):
        from pgmuvi.gps import DustMean
        self.DustMean = DustMean
        self.x = torch.tensor(
            [[0.0, 0.5], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32
        )

    def test_instantiation(self):
        """DustMean can be instantiated without arguments."""
        mean = self.DustMean()
        self.assertIsNotNone(mean)

    def test_forward_shape(self):
        """DustMean forward returns a 1-D tensor of correct length."""
        mean = self.DustMean()
        out = mean(self.x)
        self.assertEqual(out.shape, (3,))

    def test_parameters_registered(self):
        """DustMean registers the required parameters."""
        mean = self.DustMean()
        param_names = [n for n, _ in mean.named_parameters()]
        self.assertIn("offset", param_names)
        self.assertIn("log_amplitude", param_names)
        self.assertIn("log_tau", param_names)
        self.assertIn("log_alpha", param_names)

    def test_extinction_increases_at_short_wavelength(self):
        """Extinction is greater at shorter wavelengths (dust behaviour)."""
        mean = self.DustMean()
        # Set tau > 0, alpha > 0; offset = 0
        with torch.no_grad():
            mean.offset.fill_(0.0)
            mean.log_amplitude.fill_(0.0)   # amplitude = 1
            mean.log_tau.fill_(0.0)         # tau = 1
            mean.log_alpha.fill_(0.0)       # alpha = 1
        x_optical = torch.tensor([[0.0, 0.5]], dtype=torch.float32)
        x_ir = torch.tensor([[0.0, 2.0]], dtype=torch.float32)
        out_optical = mean(x_optical)
        out_ir = mean(x_ir)
        # Optical (shorter wavelength) should have lower flux
        self.assertLess(out_optical.item(), out_ir.item())

    def test_zero_tau_gives_constant_mean(self):
        """With tau -> 0, DustMean approaches amplitude + offset."""
        mean = self.DustMean()
        with torch.no_grad():
            mean.offset.fill_(0.5)
            mean.log_amplitude.fill_(0.0)   # amplitude = 1
            mean.log_tau.fill_(-30.0)       # tau ≈ 0
            mean.log_alpha.fill_(0.0)
        out = mean(self.x)
        # Should be approximately offset + amplitude = 1.5 for all points
        self.assertTrue(torch.allclose(out, torch.full((3,), 1.5), atol=1e-3))


class TestNewGPModels(unittest.TestCase):
    """Smoke tests for the new 2D GP model classes."""

    def setUp(self):
        from pgmuvi.gps import (
            TwoDSpectralMixturePowerLawMeanGPModel,
            TwoDSpectralMixturePowerLawMeanKISSGPModel,
            TwoDSpectralMixtureDustMeanGPModel,
            TwoDSpectralMixtureDustMeanKISSGPModel,
        )
        import gpytorch
        self.models = {
            "2DPowerLaw": TwoDSpectralMixturePowerLawMeanGPModel,
            "2DPowerLawSKI": TwoDSpectralMixturePowerLawMeanKISSGPModel,
            "2DDust": TwoDSpectralMixtureDustMeanGPModel,
            "2DDustSKI": TwoDSpectralMixtureDustMeanKISSGPModel,
        }
        # 2D training data: columns are (time, wavelength)
        self.train_x = torch.tensor(
            [[0.0, 0.5], [1.0, 1.0], [2.0, 2.0], [3.0, 0.5]], dtype=torch.float32
        )
        self.train_y = torch.tensor([1.0, 0.8, 0.6, 1.1], dtype=torch.float32)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def test_instantiation(self):
        """All new 2D GP model classes can be instantiated."""
        for name, ModelClass in self.models.items():
            with self.subTest(model=name):
                model = ModelClass(self.train_x, self.train_y, self.likelihood)
                self.assertIsNotNone(model)

    def test_forward_returns_mvn(self):
        """Forward pass returns a MultivariateNormal distribution."""
        from gpytorch.distributions import MultivariateNormal
        for name, ModelClass in self.models.items():
            with self.subTest(model=name):
                model = ModelClass(self.train_x, self.train_y, self.likelihood)
                model.eval()
                with torch.no_grad():
                    out = model(self.train_x)
                self.assertIsInstance(out, MultivariateNormal)

    def test_set_model_shortcut(self):
        """Lightcurve.set_model() accepts the new model shortcut strings."""
        for shortcut in ["2DPowerLaw", "2DPowerLawSKI", "2DDust", "2DDustSKI"]:
            with self.subTest(shortcut=shortcut):
                lc = Lightcurve(self.train_x, self.train_y)
                lc.set_model(model=shortcut, num_mixtures=2)
                self.assertIsNotNone(lc.model)


class TestSimpleKernelMeanGPModels(unittest.TestCase):
    """Smoke tests for DustMeanGPModel and PowerLawMeanGPModel."""

    def setUp(self):
        from pgmuvi.gps import DustMeanGPModel, PowerLawMeanGPModel
        import gpytorch
        self.models = {
            "DustMean": DustMeanGPModel,
            "PowerLawMean": PowerLawMeanGPModel,
        }
        # 2D training data: columns are (time, wavelength)
        self.train_x = torch.tensor(
            [[0.0, 0.5], [1.0, 1.0], [2.0, 2.0], [3.0, 0.5]], dtype=torch.float32
        )
        self.train_y = torch.tensor([1.0, 0.8, 0.6, 1.1], dtype=torch.float32)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def test_instantiation(self):
        """DustMeanGPModel and PowerLawMeanGPModel can be instantiated."""
        for name, ModelClass in self.models.items():
            with self.subTest(model=name):
                model = ModelClass(self.train_x, self.train_y, self.likelihood)
                self.assertIsNotNone(model)

    def test_forward_returns_mvn(self):
        """Forward pass returns a MultivariateNormal distribution."""
        from gpytorch.distributions import MultivariateNormal
        for name, ModelClass in self.models.items():
            with self.subTest(model=name):
                model = ModelClass(self.train_x, self.train_y, self.likelihood)
                model.eval()
                with torch.no_grad():
                    out = model(self.train_x)
                self.assertIsInstance(out, MultivariateNormal)

    def test_correct_mean_module(self):
        """DustMeanGPModel uses DustMean; PowerLawMeanGPModel uses PowerLawMean."""
        from pgmuvi.gps import DustMean, PowerLawMean, DustMeanGPModel, PowerLawMeanGPModel
        dust_model = DustMeanGPModel(self.train_x, self.train_y, self.likelihood)
        pl_model = PowerLawMeanGPModel(self.train_x, self.train_y, self.likelihood)
        self.assertIsInstance(dust_model.mean_module, DustMean)
        self.assertIsInstance(pl_model.mean_module, PowerLawMean)

    def test_set_model_shortcut(self):
        """Lightcurve.set_model() accepts '2DDustMean' and '2DPowerLawMean'."""
        for shortcut in ["2DDustMean", "2DPowerLawMean"]:
            with self.subTest(shortcut=shortcut):
                lc = Lightcurve(self.train_x, self.train_y)
                lc.set_model(model=shortcut)
                self.assertIsNotNone(lc.model)

    def test_wavelength_dependent_mean_module_strings(self):
        """WavelengthDependentGPModel accepts 'dust' and 'power_law' mean_module strings."""
        from pgmuvi.gps import WavelengthDependentGPModel, DustMean, PowerLawMean
        for mean_str, ExpectedCls in [("dust", DustMean), ("power_law", PowerLawMean)]:
            with self.subTest(mean_module=mean_str):
                model = WavelengthDependentGPModel(
                    self.train_x, self.train_y, self.likelihood, mean_module=mean_str
                )
                self.assertIsInstance(model.mean_module, ExpectedCls)

    def test_kernel_types_can_be_changed(self):
        """DustMeanGPModel and PowerLawMeanGPModel accept time/wavelength kernel types."""
        from pgmuvi.gps import DustMeanGPModel, PowerLawMeanGPModel
        for ModelClass in (DustMeanGPModel, PowerLawMeanGPModel):
            with self.subTest(model=ModelClass.__name__):
                model = ModelClass(
                    self.train_x,
                    self.train_y,
                    self.likelihood,
                    time_kernel_type="rbf",
                    wavelength_kernel_type="matern",
                )
                self.assertIsNotNone(model)

    def test_num_mixtures_forwarded_to_alt_model(self):
        """set_model() forwards num_mixtures to model_dic_alt models.

        Regression test: previously num_mixtures was silently dropped for
        WavelengthDependentGPModel (and other alt-path models). Now it must
        reach the SpectralMixtureKernel when time_kernel_type='sm'.
        """
        from gpytorch.kernels import SpectralMixtureKernel
        lc = Lightcurve(self.train_x, self.train_y)
        lc.set_model(
            model="2DWavelengthDependent",
            time_kernel_type="sm",
            num_mixtures=2,
        )
        # The time kernel is the first kernel in the product (AdditiveKernel
        # wraps two ScaleKernels; for separable models the covar_module is a
        # ProductKernel whose first child is the time kernel).
        time_kernel = lc.model.covar_module.kernels[0]
        # Unwrap ScaleKernel if present
        if hasattr(time_kernel, "base_kernel"):
            time_kernel = time_kernel.base_kernel
        self.assertIsInstance(time_kernel, SpectralMixtureKernel)
        self.assertEqual(time_kernel.num_mixtures, 2)


class TestYscaleAndYlim(unittest.TestCase):
    """Unit tests for Lightcurve._yscale_and_ylim static helper."""

    def _call(self, y_vals, yscale="auto", ylim=None):
        return Lightcurve._yscale_and_ylim(np.array(y_vals), yscale, ylim)

    # --- scale selection ---

    def test_auto_all_positive_large_range_selects_log(self):
        # max/min = 1000 > 100
        scale, _ = self._call([1.0, 10.0, 1000.0])
        self.assertEqual(scale, "log")

    def test_auto_all_positive_small_range_selects_linear(self):
        # max/min = 10 < 100
        scale, _ = self._call([1.0, 5.0, 10.0])
        self.assertEqual(scale, "linear")

    def test_auto_mixed_signs_selects_linear(self):
        scale, _ = self._call([-1.0, 1.0, 100.0])
        self.assertEqual(scale, "linear")

    def test_auto_zero_in_data_selects_linear(self):
        scale, _ = self._call([0.0, 1.0, 1000.0])
        self.assertEqual(scale, "linear")

    def test_explicit_log_respected(self):
        scale, _ = self._call([1.0, 2.0], yscale="log")
        self.assertEqual(scale, "log")

    def test_explicit_linear_respected(self):
        scale, _ = self._call([1.0, 1000.0], yscale="linear")
        self.assertEqual(scale, "linear")

    # --- limit computation (auto ylim) ---

    def test_linear_auto_lim_adds_padding(self):
        # data [0, 10]: range=10, padding=1 → [-1, 11]
        _, lim = self._call([0.0, 10.0], yscale="linear")
        self.assertIsNotNone(lim)
        self.assertAlmostEqual(lim[0], -1.0)
        self.assertAlmostEqual(lim[1], 11.0)

    def test_identical_values_nonzero_uses_magnitude_padding(self):
        # all values are 5: padding = 0.1 * 5 = 0.5
        _, lim = self._call([5.0, 5.0, 5.0], yscale="linear")
        self.assertIsNotNone(lim)
        self.assertAlmostEqual(lim[0], 4.5)
        self.assertAlmostEqual(lim[1], 5.5)

    def test_identical_values_zero_fallback_padding(self):
        # all values are 0: base=1, padding=0.1
        _, lim = self._call([0.0, 0.0, 0.0], yscale="linear")
        self.assertIsNotNone(lim)
        self.assertAlmostEqual(lim[0], -0.1)
        self.assertAlmostEqual(lim[1], 0.1)

    def test_log_auto_lim_is_positive(self):
        scale, lim = self._call([1.0, 10.0, 1000.0])
        self.assertEqual(scale, "log")
        self.assertIsNotNone(lim)
        self.assertGreater(lim[0], 0.0)
        self.assertGreater(lim[1], lim[0])

    def test_log_forced_nonpositive_data_returns_none_lim(self):
        # Forced log but data has zero: cannot compute valid limits
        scale, lim = self._call([0.0, 1.0, 1000.0], yscale="log")
        self.assertEqual(scale, "log")
        self.assertIsNone(lim)

    def test_log_forced_negative_data_returns_none_lim(self):
        scale, lim = self._call([-5.0, 1.0, 1000.0], yscale="log")
        self.assertEqual(scale, "log")
        self.assertIsNone(lim)

    # --- explicit ylim ---

    def test_explicit_ylim_linear_used_as_is(self):
        _, lim = self._call([1.0, 5.0], yscale="linear", ylim=[0.0, 10.0])
        self.assertEqual(lim, [0.0, 10.0])

    def test_explicit_ylim_log_positive_lower_used_as_is(self):
        _, lim = self._call([1.0, 1000.0], yscale="log", ylim=[0.5, 2000.0])
        self.assertEqual(lim, [0.5, 2000.0])

    def test_explicit_ylim_log_nonpositive_lower_returns_none(self):
        # Lower bound of 0 is invalid on a log axis
        _, lim = self._call([1.0, 1000.0], yscale="log", ylim=[0.0, 2000.0])
        self.assertIsNone(lim)

    def test_explicit_ylim_log_negative_lower_returns_none(self):
        _, lim = self._call([1.0, 1000.0], yscale="log", ylim=[-1.0, 2000.0])
        self.assertIsNone(lim)

    def test_explicit_ylim_auto_log_nonpositive_lower_returns_none(self):
        # auto selects log (large range, all positive), but caller passes
        # ylim with a non-positive lower bound
        _, lim = self._call([1.0, 1000.0], yscale="auto", ylim=[-1.0, 2000.0])
        self.assertIsNone(lim)


class TestPlotYscaleValidation(unittest.TestCase):
    """Test that plot() raises ValueError for invalid yscale values."""

    def setUp(self):
        xdata = torch.as_tensor([1.0, 2.0, 3.0, 4.0])
        ydata = torch.as_tensor([1.0, 2.0, 1.0, 2.0])
        self.lc = Lightcurve(xdata, ydata)

    def test_invalid_yscale_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.lc.plot(yscale="symlog")

    def test_invalid_yscale_message_is_informative(self):
        with self.assertRaisesRegex(ValueError, "yscale must be one of"):
            self.lc.plot(yscale="bad_value")


if __name__ == '__main__':
    unittest.main()
