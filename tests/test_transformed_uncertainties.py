"""Regression tests for scale-only dependent-variable uncertainty transforms."""

import unittest

import torch

from pgmuvi.lightcurve import Lightcurve, MinMax, RobustZScore, Shift, ZScore


class TestTransformerUncertaintyScaling(unittest.TestCase):
    def setUp(self):
        self.y = torch.tensor([10.0, 12.0, 14.0, 16.0], dtype=torch.float64)
        self.yerr = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)

    def test_minmax_uses_fitted_range_without_minimum_shift(self):
        transformer = MinMax()
        transformer.transform(self.y)

        transformed = transformer.transform_uncertainty(self.yerr)

        self.assertTrue(torch.allclose(transformed, self.yerr / 6.0))
        self.assertTrue(torch.all(transformed > 0))

    def test_zscore_uses_fitted_standard_deviation_without_mean_shift(self):
        transformer = ZScore()
        transformer.transform(self.y)

        transformed = transformer.transform_uncertainty(self.yerr)

        self.assertTrue(torch.allclose(transformed, self.yerr / torch.std(self.y)))
        self.assertTrue(torch.all(transformed > 0))

    def test_robust_zscore_uses_fitted_mad_without_median_shift(self):
        transformer = RobustZScore()
        transformer.transform(self.y)

        transformed = transformer.transform_uncertainty(self.yerr)

        expected = self.yerr / transformer.mad
        self.assertTrue(torch.allclose(transformed, expected))
        self.assertTrue(torch.all(transformed > 0))

    def test_shift_leaves_uncertainty_scales_unchanged(self):
        transformer = Shift(method="midpoint")
        transformer.transform(self.y)

        transformed = transformer.transform_uncertainty(self.yerr)

        self.assertTrue(torch.equal(transformed, self.yerr))

    def test_variances_use_the_square_of_the_fitted_scale(self):
        transformer = MinMax()
        transformer.transform(self.y)
        variances = self.yerr**2

        transformed = transformer.transform_variance(variances)

        self.assertTrue(torch.allclose(transformed, variances / 36.0))

    def test_unfitted_scale_transform_rejects_uncertainties(self):
        with self.assertRaisesRegex(RuntimeError, "fitted range"):
            MinMax().transform_uncertainty(self.yerr)


class TestLightcurveTransformedUncertainties(unittest.TestCase):
    def setUp(self):
        self.x = torch.arange(4, dtype=torch.float64)
        self.y = torch.tensor([10.0, 12.0, 14.0, 16.0], dtype=torch.float64)
        self.yerr = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)

    def test_minmax_constructor_scales_errors_without_shifting_them(self):
        lc = Lightcurve(
            self.x,
            self.y,
            yerr=self.yerr,
            ytransform="minmax",
            center_time=False,
        )

        self.assertTrue(torch.allclose(lc._yerr_transformed, self.yerr / 6.0))
        self.assertTrue(torch.all(lc._yerr_transformed > 0))

    def test_zscore_constructor_scales_errors_without_shifting_them(self):
        lc = Lightcurve(
            self.x,
            self.y,
            yerr=self.yerr,
            ytransform="zscore",
            center_time=False,
        )

        expected = self.yerr / torch.std(self.y)
        self.assertTrue(torch.allclose(lc._yerr_transformed, expected))

    def test_robust_zscore_constructor_scales_errors_without_shifting_them(self):
        lc = Lightcurve(
            self.x,
            self.y,
            yerr=self.yerr,
            ytransform="robust_zscore",
            center_time=False,
        )

        expected = self.yerr / lc.ytransform.mad
        self.assertTrue(torch.allclose(lc._yerr_transformed, expected))

    def test_transform_y_uncertainty_matches_stored_errors(self):
        lc = Lightcurve(
            self.x,
            self.y,
            yerr=self.yerr,
            ytransform="zscore",
            center_time=False,
        )

        transformed = lc.transform_y_uncertainty(self.yerr)

        self.assertTrue(torch.allclose(transformed, lc._yerr_transformed))

    def test_no_ytransform_preserves_uncertainties(self):
        lc = Lightcurve(
            self.x,
            self.y,
            yerr=self.yerr,
            center_time=False,
        )

        self.assertTrue(torch.equal(lc._yerr_transformed, self.yerr))
        self.assertTrue(torch.equal(lc.transform_y_uncertainty(self.yerr), self.yerr))

    def test_variance_true_uses_squared_transform_scale(self):
        variances = self.yerr**2
        lc = Lightcurve(
            self.x,
            self.y,
            yerr=variances,
            ytransform="minmax",
            center_time=False,
        )

        lc.set_likelihood(variance=True)

        self.assertTrue(torch.allclose(lc.likelihood.noise, variances / 36.0))


if __name__ == "__main__":
    unittest.main()
