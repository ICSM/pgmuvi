"""Tests for the Transformer, MinMax and related transforms."""
import unittest

import torch


from pgmuvi.lightcurve import (
    MinMax,
    Transformer,
)


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
