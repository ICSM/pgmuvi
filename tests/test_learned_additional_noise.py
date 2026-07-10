"""Tests for learned additional-noise likelihood handling."""

import unittest

import gpytorch
import torch

from pgmuvi.lightcurve import Lightcurve


class TestLearnedAdditionalNoise(unittest.TestCase):
    def setUp(self):
        self.x = torch.linspace(0.0, 9.0, 10)
        self.y = torch.sin(self.x)
        self.yerr = torch.full_like(self.y, 0.1)

    def _assert_has_learned_additional_noise(self, likelihood):
        self.assertIsInstance(
            likelihood,
            gpytorch.likelihoods.FixedNoiseGaussianLikelihood,
        )
        self.assertTrue(hasattr(likelihood, "second_noise_covar"))
        named_parameters = dict(likelihood.named_parameters())
        self.assertTrue(
            any(name.startswith("second_noise_covar") for name in named_parameters),
            msg=(
                "FixedNoiseGaussianLikelihood should expose learned additional "
                "noise parameters under second_noise_covar."
            ),
        )

    def test_fixed_noise_can_learn_additional_noise(self):
        lc = Lightcurve(self.x, self.y, yerr=self.yerr)

        lc.set_likelihood(learn_additional_noise=True)

        self._assert_has_learned_additional_noise(lc.likelihood)
        self.assertTrue(lc._learn_additional_noise)

    def test_likelihood_learn_alias_enables_additional_noise(self):
        lc = Lightcurve(self.x, self.y, yerr=self.yerr)

        lc.set_likelihood(likelihood="learn")

        self._assert_has_learned_additional_noise(lc.likelihood)
        self.assertTrue(lc._learn_additional_noise)

    def test_likelihood_learn_without_yerr_uses_gaussian_likelihood(self):
        lc = Lightcurve(self.x, self.y)

        lc.set_likelihood(likelihood="learn")

        self.assertIsInstance(lc.likelihood, gpytorch.likelihoods.GaussianLikelihood)
        self.assertTrue(lc._learn_additional_noise)

    def test_set_model_forwards_learn_additional_noise(self):
        lc = Lightcurve(self.x, self.y, yerr=self.yerr)

        lc.set_model("1D", num_mixtures=1, learn_additional_noise=True)

        self._assert_has_learned_additional_noise(lc.likelihood)
        self.assertTrue(lc._learn_additional_noise)

    def test_reconfigures_existing_automatic_likelihood(self):
        lc = Lightcurve(self.x, self.y, yerr=self.yerr)
        lc.set_likelihood()
        self.assertFalse(lc._learn_additional_noise)

        lc.set_likelihood(learn_additional_noise=True)

        self._assert_has_learned_additional_noise(lc.likelihood)
        self.assertTrue(lc._learn_additional_noise)

    def test_rejects_explicit_likelihood_with_learn_additional_noise_flag(self):
        lc = Lightcurve(self.x, self.y, yerr=self.yerr)
        explicit_likelihood = gpytorch.likelihoods.GaussianLikelihood()

        with self.assertRaisesRegex(ValueError, "learn_additional_noise"):
            lc.set_likelihood(
                likelihood=explicit_likelihood,
                learn_additional_noise=True,
            )


if __name__ == "__main__":
    unittest.main()
