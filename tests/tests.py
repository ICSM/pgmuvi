import unittest

from pgmuvi.lightcurve import Lightcurve, Transformer, MinMax, ZScore, RobustZScore
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

    def test_fit_LS_basic_functionality_with_yerr(self):
        """Test fit_LS basic functionality with valid inputs including yerr"""
        # Create a synthetic periodic signal
        t = torch.linspace(0, 10, 100, dtype=torch.float32)
        freq_true = 1.0  # 1 Hz signal
        y = torch.sin(2 * np.pi * freq_true * t) + 0.1 * torch.randn(100)
        yerr = torch.ones(100, dtype=torch.float32) * 0.1
        lc = Lightcurve(t, y, yerr=yerr)

        # Test basic functionality
        freq_peaks, significance = lc.fit_LS(num_peaks=3)

        # Check that we get tensors back
        self.assertIsInstance(freq_peaks, torch.Tensor)
        self.assertIsInstance(significance, torch.Tensor)
        # Check that we get the expected number of peaks (or fewer)
        self.assertLessEqual(len(freq_peaks), 3)
        self.assertEqual(len(freq_peaks), len(significance))
        # Check device and dtype consistency
        self.assertEqual(freq_peaks.device, lc.xdata.device)
        self.assertEqual(freq_peaks.dtype, lc.xdata.dtype)
        self.assertEqual(significance.dtype, torch.bool)

    def test_fit_LS_without_yerr(self):
        """Test fit_LS when yerr is None"""
        # Create lightcurve without yerr
        t = torch.linspace(0, 10, 100, dtype=torch.float32)
        y = torch.sin(2 * np.pi * t) + 0.1 * torch.randn(100)
        lc = Lightcurve(t, y)  # No yerr provided

        # Should still work without yerr
        freq_peaks, significance = lc.fit_LS(num_peaks=2)

        self.assertIsInstance(freq_peaks, torch.Tensor)
        self.assertIsInstance(significance, torch.Tensor)
        self.assertLessEqual(len(freq_peaks), 2)

    def test_fit_LS_freq_only_flag(self):
        """Test fit_LS with freq_only=True"""
        t = torch.linspace(0, 10, 50, dtype=torch.float32)
        y = torch.sin(2 * np.pi * t)
        lc = Lightcurve(t, y)

        freq, power = lc.fit_LS(freq_only=True)

        # Should return frequency grid and power, not peaks
        self.assertIsInstance(freq, torch.Tensor)
        self.assertIsInstance(power, torch.Tensor)
        self.assertEqual(len(freq), len(power))
        # freq grid should be longer than num_peaks
        self.assertGreater(len(freq), 10)

    def test_fit_LS_num_peaks_greater_than_available(self):
        """Test fit_LS when num_peaks is greater than available peaks"""
        # Create a simple signal that may have few peaks
        t = torch.linspace(0, 2, 20, dtype=torch.float32)
        y = torch.ones(20) + 0.01 * torch.randn(20)
        lc = Lightcurve(t, y)

        # Request more peaks than likely available
        freq_peaks, significance = lc.fit_LS(num_peaks=10)

        # Should return only available peaks, not fail
        self.assertIsInstance(freq_peaks, torch.Tensor)
        self.assertIsInstance(significance, torch.Tensor)
        # Should have fewer peaks than requested
        self.assertLessEqual(len(freq_peaks), 10)
        self.assertEqual(len(freq_peaks), len(significance))

    def test_fit_LS_no_peaks_found(self):
        """Test fit_LS when no clear peaks are found"""
        # Create very noisy signal with minimal structure
        t = torch.linspace(0, 10, 50, dtype=torch.float32)
        y = torch.randn(50) * 0.1  # Pure noise
        lc = Lightcurve(t, y)

        freq_peaks, significance = lc.fit_LS(num_peaks=5)

        # Should return some peaks (LS finds peaks even in noise)
        # but they should not be significant
        self.assertLessEqual(len(freq_peaks), 5)
        self.assertEqual(len(freq_peaks), len(significance))

    def test_fit_LS_multiband_raises_not_implemented(self):
        """Test that multiband lightcurves raise NotImplementedError"""
        # Create multiband lightcurve
        t = torch.as_tensor([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=torch.float32)
        y = torch.as_tensor([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=torch.float32)
        lc = Lightcurve(t, y)

        # Should raise NotImplementedError
        with self.assertRaises(NotImplementedError):
            lc.fit_LS()

    def test_fit_LS_device_consistency(self):
        """Test that fit_LS preserves device and dtype"""
        t = torch.linspace(0, 10, 100, dtype=torch.float32)
        y = torch.sin(2 * np.pi * t) + 0.1 * torch.randn(100)
        lc = Lightcurve(t, y)

        freq_peaks, significance = lc.fit_LS(num_peaks=3)

        # Check device consistency
        self.assertEqual(freq_peaks.device, lc.xdata.device)
        self.assertEqual(significance.device, lc.xdata.device)
        # Check dtype
        self.assertEqual(freq_peaks.dtype, lc.xdata.dtype)
        self.assertEqual(significance.dtype, torch.bool)



class TestTrain(unittest.TestCase):
    def test_train(self):
        pass



class TestSpectralMixtureGPModel(unittest.TestCase):
    pass


if  __name__ == '__main__':
    unittest.main()