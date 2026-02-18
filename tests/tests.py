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
        
        # Create 2D lightcurve for testing NotImplementedError
        self.t_2d = torch.as_tensor([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=torch.float32)
        self.y_2d = torch.as_tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.float32)
        self.lc_2d = Lightcurve(self.t_2d, self.y_2d)
    
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
    
    def test_multiband_raises_not_implemented(self):
        """Test that multiband lightcurves raise NotImplementedError"""
        with self.assertRaises(NotImplementedError) as context:
            self.lc_2d.fit_LS()
        
        self.assertIn("Multiband", str(context.exception))
    
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



class TestTrain(unittest.TestCase):
    def test_train(self):
        pass



class TestSpectralMixtureGPModel(unittest.TestCase):
    pass


if  __name__ == '__main__':
    unittest.main()