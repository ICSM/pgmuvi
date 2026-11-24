"""Tests for Lightcurve data getters, setters and likelihood."""
import unittest

import gpytorch
import torch

from pgmuvi.lightcurve import (
    Lightcurve,
    MinMax,
)


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

    def test_set_likelihood_squares_errors_by_default(self):
        """set_likelihood should pass squared errors (variances) to
        FixedNoiseGaussianLikelihood by default (variance=False)."""
        lc = Lightcurve(self.test_xdata, self.test_ydata, yerr=self.test_ydata)
        lc.set_likelihood()
        self.assertIsInstance(
            lc.likelihood,
            gpytorch.likelihoods.FixedNoiseGaussianLikelihood,
        )
        expected_noise = lc._yerr_transformed ** 2
        self.assertTrue(torch.allclose(lc.likelihood.noise, expected_noise))

    def test_set_likelihood_uses_errors_as_variances_when_variance_true(self):
        """When variance=True, set_likelihood should pass errors unchanged to
        FixedNoiseGaussianLikelihood (i.e. treat them as already-squared)."""
        lc = Lightcurve(self.test_xdata, self.test_ydata, yerr=self.test_ydata)
        lc.set_likelihood(variance=True)
        self.assertIsInstance(
            lc.likelihood,
            gpytorch.likelihoods.FixedNoiseGaussianLikelihood,
        )
        expected_noise = lc._yerr_transformed
        self.assertTrue(torch.allclose(lc.likelihood.noise, expected_noise))
