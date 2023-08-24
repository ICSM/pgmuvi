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
        self.test_xdata = torch.astensor([1, 2, 3, 4], dtype=torch.float32)
        self.test_ydata = torch.astensor([1, 1, 1, 1], dtype=torch.float32)
        self.test_xdata_2d = torch.astensor([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=torch.float32)
        self.test_ydata_2d = torch.astensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.float32)
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
        self.lightcurve.ytransform = 'minmax'
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
        self.lightcurve.ytransform = 'minmax'
        self.test_yerr_transformed = self.lightcurve.ytransform.transform(self.test_ydata)
        
        self.lightcurve.yerr = self.test_ydata
        self.assertTrue(torch.equal(self.lightcurve._yerr_raw, self.test_ydata))
        self.assertTrue(torch.equal(self.lightcurve._yerr_transformed, self.test_yerr_transformed))
    
class TestTrain(unittest.TestCase):
    def test_train(self):
        pass



class TestSpectralMixtureGPModel(unittest.TestCase):
    pass


if  __name__ == '__main__':
    unittest.main()