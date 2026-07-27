"""Tests for learned additional-noise likelihood handling."""

import unittest

import gpytorch
import torch
from gpytorch.constraints import Interval

from pgmuvi.lightcurve import Lightcurve


class TestLearnedAdditionalNoise(unittest.TestCase):
    def setUp(self):
        self.x = torch.linspace(0.0, 10.0, 32, dtype=torch.float64)
        self.y = 2.0 + 0.2 * torch.sin(self.x)
        self.yerr = torch.full_like(self.y, 0.2)

    def _make_lightcurve(self, *, with_yerr=True):
        return Lightcurve(
            self.x,
            self.y,
            yerr=self.yerr if with_yerr else None,
        )

    def _assert_has_learned_additional_noise(self, likelihood):
        self.assertIsInstance(
            likelihood,
            gpytorch.likelihoods.FixedNoiseGaussianLikelihood,
        )
        self.assertTrue(hasattr(likelihood, "second_noise_covar"))
        named_parameters = dict(likelihood.named_parameters())
        self.assertTrue(
            any(
                name.startswith("second_noise_covar")
                for name in named_parameters
            ),
            "FixedNoiseGaussianLikelihood should expose learned additional "
            "noise parameters under second_noise_covar.",
        )

    def _assert_has_no_learned_additional_noise(self, likelihood):
        self.assertIsInstance(
            likelihood,
            gpytorch.likelihoods.FixedNoiseGaussianLikelihood,
        )
        named_parameters = dict(likelihood.named_parameters())
        self.assertFalse(
            any(
                name.startswith("second_noise_covar")
                for name in named_parameters
            )
        )

    def test_default_with_yerr_learns_additional_noise(self):
        lc = self._make_lightcurve()
        lc.set_likelihood()
        self._assert_has_learned_additional_noise(lc.likelihood)
        self.assertTrue(lc._learn_additional_noise)

    def test_fixed_noise_can_explicitly_learn_additional_noise(self):
        lc = self._make_lightcurve()
        lc.set_likelihood(learn_additional_noise=True)
        self._assert_has_learned_additional_noise(lc.likelihood)
        self.assertTrue(lc._learn_additional_noise)

    def test_likelihood_learn_alias_enables_additional_noise(self):
        lc = self._make_lightcurve()
        lc.set_likelihood(likelihood="learn")
        self._assert_has_learned_additional_noise(lc.likelihood)
        self.assertTrue(lc._learn_additional_noise)

    def test_likelihood_fixed_alias_restores_fixed_noise_only(self):
        lc = self._make_lightcurve()
        lc.set_likelihood(likelihood="fixed")
        self._assert_has_no_learned_additional_noise(lc.likelihood)
        self.assertFalse(lc._learn_additional_noise)
        self.assertIsNone(lc._initial_additional_noise_variance)

    def test_explicit_false_restores_fixed_noise_only(self):
        lc = self._make_lightcurve()
        lc.set_likelihood(learn_additional_noise=False)
        self._assert_has_no_learned_additional_noise(lc.likelihood)
        self.assertFalse(lc._learn_additional_noise)

    def test_default_initializes_additional_variance_from_fixed_noise(self):
        lc = self._make_lightcurve()
        lc.set_likelihood()
        fixed_noise = lc.likelihood.noise_covar.noise.detach()
        expected = 0.1 * torch.median(fixed_noise)
        actual = (
            lc.likelihood.second_noise_covar.noise.detach().reshape(-1)[0]
        )
        self.assertTrue(torch.isfinite(actual))
        torch.testing.assert_close(
            actual,
            expected,
            rtol=1.0e-6,
            atol=1.0e-12,
        )
        self.assertAlmostEqual(
            lc._initial_additional_noise_variance,
            float(actual.cpu().item()),
            places=12,
        )

    def test_default_noise_receives_live_interval_constraint(self):
        lc = self._make_lightcurve()
        lc.set_model("1DMatern")
        lc.set_default_constraints()
        constraint = lc.likelihood.second_noise_covar.raw_noise_constraint
        self.assertIsInstance(constraint, Interval)
        learned_noise = (
            lc.likelihood.second_noise_covar.noise.detach().reshape(-1)[0]
        )
        lower = torch.as_tensor(
            constraint.lower_bound,
            dtype=learned_noise.dtype,
            device=learned_noise.device,
        ).max()
        upper = torch.as_tensor(
            constraint.upper_bound,
            dtype=learned_noise.dtype,
            device=learned_noise.device,
        ).min()
        self.assertGreaterEqual(float(learned_noise), float(lower))
        self.assertLessEqual(float(learned_noise), float(upper))

    def test_default_without_yerr_uses_gaussian_likelihood(self):
        lc = self._make_lightcurve(with_yerr=False)
        lc.set_likelihood()
        self.assertIsInstance(
            lc.likelihood,
            gpytorch.likelihoods.GaussianLikelihood,
        )
        self.assertTrue(lc._learn_additional_noise)
        self.assertIsNone(lc._initial_additional_noise_variance)

    def test_likelihood_learn_without_yerr_uses_gaussian_likelihood(self):
        lc = self._make_lightcurve(with_yerr=False)
        lc.set_likelihood(likelihood="learn")
        self.assertIsInstance(
            lc.likelihood,
            gpytorch.likelihoods.GaussianLikelihood,
        )
        self.assertTrue(lc._learn_additional_noise)

    def test_set_model_defaults_to_learned_additional_noise(self):
        lc = self._make_lightcurve()
        lc.set_model("1DMatern")
        self._assert_has_learned_additional_noise(lc.likelihood)
        self.assertTrue(lc._learn_additional_noise)

    def test_set_model_preserves_explicit_fixed_configuration(self):
        lc = self._make_lightcurve()
        lc.set_likelihood(likelihood="fixed")
        lc.set_model("1DMatern")
        self._assert_has_no_learned_additional_noise(lc.likelihood)
        self.assertFalse(lc._learn_additional_noise)

    def test_fit_defaults_to_learned_additional_noise(self):
        lc = self._make_lightcurve()
        lc.fit(
            model="1DMatern",
            training_iter=1,
            miniter=1,
            lr=0.01,
            use_parameter_workflow=False,
        )
        self._assert_has_learned_additional_noise(lc.likelihood)
        self.assertTrue(lc._learn_additional_noise)
        learned_noise = lc.likelihood.second_noise_covar.noise.detach()
        self.assertTrue(torch.isfinite(learned_noise).all())

    def test_fit_fixed_alias_opts_out(self):
        lc = self._make_lightcurve()
        lc.fit(
            model="1DMatern",
            likelihood="fixed",
            training_iter=1,
            miniter=1,
            lr=0.01,
            use_parameter_workflow=False,
        )
        self._assert_has_no_learned_additional_noise(lc.likelihood)
        self.assertFalse(lc._learn_additional_noise)

    def test_reconfigures_existing_automatic_likelihood(self):
        lc = self._make_lightcurve()
        lc.set_likelihood()
        self.assertTrue(lc._learn_additional_noise)
        lc.set_likelihood(learn_additional_noise=False)
        self._assert_has_no_learned_additional_noise(lc.likelihood)
        self.assertFalse(lc._learn_additional_noise)
        lc.set_likelihood()
        self._assert_has_learned_additional_noise(lc.likelihood)
        self.assertTrue(lc._learn_additional_noise)

    def test_custom_likelihood_is_accepted_when_flag_is_omitted(self):
        lc = self._make_lightcurve()
        explicit_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        lc.set_likelihood(likelihood=explicit_likelihood)
        self.assertIs(lc.likelihood, explicit_likelihood)
        self.assertFalse(lc._learn_additional_noise)

    def test_rejects_explicit_likelihood_with_learn_additional_noise_flag(
        self,
    ):
        lc = self._make_lightcurve()
        explicit_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        with self.assertRaisesRegex(ValueError, "learn_additional_noise"):
            lc.set_likelihood(
                likelihood=explicit_likelihood,
                learn_additional_noise=True,
            )

    def test_rejects_conflicting_fixed_alias_and_true_flag(self):
        lc = self._make_lightcurve()
        with self.assertRaisesRegex(ValueError, "conflicts"):
            lc.set_likelihood(
                likelihood="fixed",
                learn_additional_noise=True,
            )

    def test_rejects_unknown_string_likelihood(self):
        lc = self._make_lightcurve()
        with self.assertRaisesRegex(ValueError, "'learn' or 'fixed'"):
            lc.set_likelihood(likelihood="mystery")


if __name__ == "__main__":
    unittest.main()
