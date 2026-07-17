"""Tests for retained-state exact marginal-likelihood diagnostics."""

import unittest

import gpytorch
import torch

from pgmuvi.wavelength_diagnostics import (
    _piwd_compute_retained_state_marginal_likelihood,
    _piwd_compute_training_fit_quality,
)


class _TinyExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covariance = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covariance)


class _FittedLightcurve:
    def __init__(self, *, with_prior=False):
        self._xdata_transformed = torch.linspace(
            0.0, 1.0, 6, dtype=torch.float64
        )
        self._ydata_transformed = torch.tensor(
            [0.1, 0.5, 0.8, 0.4, -0.1, -0.3], dtype=torch.float64
        )
        self._yerr_transformed = torch.full(
            (6,), 0.2, dtype=torch.float64
        )
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
        self.likelihood.noise = 0.2
        self.model = _TinyExactGP(
            self._xdata_transformed,
            self._ydata_transformed,
            self.likelihood,
        ).double()
        self.model.covar_module.outputscale = 1.3
        self.model.covar_module.base_kernel.lengthscale = 0.4
        self.outputscale_prior = None
        if with_prior:
            self.outputscale_prior = gpytorch.priors.LogNormalPrior(
                0.0, 0.5
            )
            self.model.covar_module.register_prior(
                "outputscale_prior",
                self.outputscale_prior,
                "outputscale",
            )

    def _eval(self):
        self.model.eval()
        self.likelihood.eval()


class TestRetainedStateMarginalLikelihood(unittest.TestCase):
    def _direct_values(self, fitted):
        fitted.model.train()
        fitted.likelihood.train()
        with torch.no_grad():
            latent = fitted.model(fitted._xdata_transformed)
            observed = fitted.likelihood(latent)
            data_total = observed.log_prob(
                fitted._ydata_transformed
            ).item()
            objective = gpytorch.mlls.ExactMarginalLogLikelihood(
                fitted.likelihood, fitted.model
            )(latent, fitted._ydata_transformed).item()
        fitted.model.eval()
        fitted.likelihood.eval()
        return data_total, objective

    def test_evaluates_training_mode_and_restores_original_modes(self):
        fitted = _FittedLightcurve()
        fitted.model.eval()
        fitted.likelihood.eval()
        expected_data_total, expected_objective = self._direct_values(fitted)

        report = _piwd_compute_retained_state_marginal_likelihood(fitted)

        self.assertTrue(report["available"])
        self.assertEqual(report["evaluation_mode"], "train")
        self.assertEqual(report["parameter_state"], "retained_current_state")
        self.assertFalse(fitted.model.training)
        self.assertFalse(fitted.likelihood.training)
        self.assertAlmostEqual(
            report["data_log_marginal_likelihood_total"],
            expected_data_total,
        )
        self.assertAlmostEqual(
            report["map_objective_per_observation"],
            expected_objective,
        )
        self.assertAlmostEqual(
            report["data_log_marginal_likelihood_per_observation"],
            expected_data_total / report["n_data"],
        )
        self.assertEqual(report["registered_prior_count"], 0)
        self.assertFalse(report["map_objective_includes_registered_priors"])
        self.assertAlmostEqual(
            report["map_objective_total"],
            report["data_log_marginal_likelihood_total"]
            + report["additional_objective_terms_total"],
        )

    def test_registered_prior_is_reported_separately(self):
        fitted = _FittedLightcurve(with_prior=True)
        fitted.model.eval()
        fitted.likelihood.eval()

        report = _piwd_compute_retained_state_marginal_likelihood(fitted)

        expected_prior_total = fitted.outputscale_prior.log_prob(
            fitted.model.covar_module.outputscale
        ).sum().item()
        self.assertTrue(report["available"])
        self.assertGreaterEqual(report["registered_prior_count"], 1)
        self.assertTrue(report["map_objective_includes_registered_priors"])
        self.assertAlmostEqual(
            report["registered_log_prior_total"], expected_prior_total
        )
        self.assertAlmostEqual(
            report["registered_log_prior_per_observation"],
            expected_prior_total / report["n_data"],
        )
        self.assertAlmostEqual(
            report["map_objective_total"],
            report["data_log_marginal_likelihood_total"]
            + report["registered_log_prior_total"]
            + report["additional_objective_terms_total"],
        )

    def test_uses_current_retained_parameters_not_a_stale_loss(self):
        fitted = _FittedLightcurve()
        before = _piwd_compute_retained_state_marginal_likelihood(fitted)

        fitted.model.mean_module.constant = 4.0
        after = _piwd_compute_retained_state_marginal_likelihood(fitted)

        self.assertTrue(before["available"])
        self.assertTrue(after["available"])
        self.assertNotAlmostEqual(
            before["data_log_marginal_likelihood_total"],
            after["data_log_marginal_likelihood_total"],
        )

    def test_training_fit_quality_exposes_precise_mll_provenance(self):
        fitted = _FittedLightcurve(with_prior=True)

        report = _piwd_compute_training_fit_quality(fitted)

        self.assertTrue(report["available"])
        self.assertTrue(report["marginal_likelihood_available"])
        self.assertEqual(
            report["marginal_likelihood_evaluation_mode"], "train"
        )
        self.assertEqual(
            report["marginal_likelihood_parameter_state"],
            "retained_current_state",
        )
        self.assertAlmostEqual(
            report["log_marginal_likelihood"],
            report["log_marginal_likelihood_total"] / 6,
        )
        self.assertTrue(report["map_objective_includes_registered_priors"])
        self.assertGreaterEqual(report["registered_prior_count"], 1)


if __name__ == "__main__":
    unittest.main()
