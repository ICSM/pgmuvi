import unittest

from pgmuvi.lightcurve import Lightcurve, Transformer, MinMax, ZScore, RobustZScore
from pgmuvi.trainers import train
from pgmuvi.gps import SpectralMixtureGPModel
import numpy as np
import torch

test_zeros_one = torch.as_tensor([0, 0, 0, 0, 1], dtype=torch.float32)
test_zeros_one_2d = torch.as_tensor([[0, 0, 0, 1], [0, 0, 0, 1]],
                                    dtype=torch.float32)

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
    def test_ndim(self):
        lightcurve = Lightcurve(test_zeros_one, test_zeros_one)
        lightcurves = Lightcurve(test_zeros_one_2d, test_zeros_one_2d)
        self.assertEqual(lightcurve.ndim(), 1)
        self.assertEqual(lightcurves.ndim(), 4)


class TestTrain(unittest.TestCase):
    def test_train(self):
        pass



class TestSpectralMixtureGPModel(unittest.TestCase):
    pass


if  __name__ == '__main__':
    unittest.main()