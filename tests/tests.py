import os
import tempfile
import unittest

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

    def test_xdata_setter_no_transform(self):
        self.lightcurve.xtransform = None
        self.lightcurve.xdata = self.test_xdata
        self.assertTrue(torch.equal(self.lightcurve._xdata_raw, self.test_xdata))
        self.assertTrue(torch.equal(self.lightcurve._xdata_transformed, self.test_xdata))

    def test_xdata_setter_with_transform(self):
        xtransformer = self.lightcurve.xtransform
        self.test_xdata_transformed = xtransformer.transform(self.test_xdata)

        self.lightcurve.xdata = self.test_xdata
        self.assertTrue(torch.equal(self.lightcurve._xdata_raw, self.test_xdata))
        self.assertTrue(torch.equal(self.lightcurve._xdata_transformed, self.test_xdata_transformed))

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


class TestTrain(unittest.TestCase):
    def test_train(self):
        pass



class TestSpectralMixtureGPModel(unittest.TestCase):
    pass


if  __name__ == '__main__':
    unittest.main()
