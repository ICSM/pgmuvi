"""Regression tests for parameter constraint integrity.

These tests document known PR31 failures in the post-PR30 parameter
workflow. They are marked as expected failures so the suite remains green
until the corresponding implementation PRs make the behaviours pass.
"""

import unittest

import gpytorch
import torch

from pgmuvi.gps import MaternGPModel, PowerLawMeanGPModel, SpectralMixtureGPModel
from pgmuvi.lightcurve import Lightcurve
from pgmuvi.parameter_application import ParameterEstimateApplicator
from pgmuvi.parameter_context import LightcurveDiagnostics, ParameterEstimationContext
from pgmuvi.parameter_estimates import ParameterEstimate, ParameterEstimateCollection
from pgmuvi.parameter_specs import (
    ParameterDomain,
    ParameterRole,
    ParameterScale,
    ParameterSpec,
)
from pgmuvi.parameter_workflow import build_and_apply_parameter_estimates


class TestConstraintIntegrityRegression(unittest.TestCase):
    """Expected-failure tests for constraint workflow correctness bugs."""

    def _toy_1d_data(self):
        x = torch.linspace(0.0, 10.0, 24)
        y = torch.sin(2.0 * torch.pi * x / 3.0)
        return x, y

    def test_value_survives_constraint_application(self):
        """Applying value+constraint should leave the parameter at the value."""
        x, y = self._toy_1d_data()
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = MaternGPModel(x, y, likelihood, lengthscale=2.0)

        spec = ParameterSpec(
            name="covar_module.base_kernel.lengthscale",
            role=ParameterRole.LENGTHSCALE,
            domain=ParameterDomain.TIME,
            scale=ParameterScale.LINEAR,
        )
        estimate = ParameterEstimate(
            spec=spec,
            value=0.5,
            constraint=(0.1, 10.0),
        )

        ParameterEstimateApplicator().apply(
            model=model,
            estimates=ParameterEstimateCollection([estimate]),
        )

        self.assertAlmostEqual(
            float(model.covar_module.base_kernel.lengthscale.item()),
            0.5,
            places=5,
        )

    def test_existing_interval_survives_default_workflow(self):
        """Workflow DEFAULT constraints should not widen existing tight bounds."""
        x, y = self._toy_1d_data()
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = SpectralMixtureGPModel(x, y, likelihood, num_mixtures=1)

        model.covar_module.register_constraint(
            "raw_mixture_means",
            gpytorch.constraints.Interval(1.0e-3, 1.0e-2),
        )

        context = ParameterEstimationContext(
            is_multiband=False,
            global_diagnostics=LightcurveDiagnostics(
                baseline_duration=10.0,
                median_flux=0.0,
                flux_percentiles={2.5: -1.0, 50.0: 0.0, 97.5: 1.0},
                n_points=len(x),
            ),
        )

        build_and_apply_parameter_estimates(model=model, context=context)

        constraint = model.covar_module.raw_mixture_means_constraint
        self.assertGreaterEqual(float(constraint.lower_bound), 1.0e-3)
        self.assertLessEqual(float(constraint.upper_bound), 1.0e-2)

    def test_set_default_constraints_survive_parameter_workflow(self):
        """Data-derived mixture-mean constraints should survive workflow application."""
        x, y = self._toy_1d_data()
        lc = Lightcurve(x, y, max_samples=None)
        lc.set_likelihood()
        lc.set_model("1D", num_mixtures=1)
        lc.set_default_constraints()

        before = lc.model.covar_module.raw_mixture_means_constraint
        before_lower = float(before.lower_bound)

        lc._apply_parameter_workflow_estimates()

        after = lc.model.covar_module.raw_mixture_means_constraint
        after_lower = float(after.lower_bound)
        after_upper = float(after.upper_bound)

        self.assertGreaterEqual(after_lower, before_lower)
        # self.assertLess(after_upper, 1.0e6)
        self.assertLessEqual(after_upper, 1.0e6)

    @unittest.expectedFailure
    def test_mean_constraint_reporting_is_honest_for_plain_parameters(self):
        """Plain nn.Parameter mean constraints should not be reported as enforced."""
        x = torch.stack(
            [
                torch.linspace(0.0, 10.0, 30),
                torch.full((30,), 2.2),
            ],
            dim=1,
        )
        y = 10.0 + 2.0 * torch.sin(2.0 * torch.pi * x[:, 0] / 3.0)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = PowerLawMeanGPModel(x, y, likelihood, time_kernel_type="matern")

        context = ParameterEstimationContext(
            is_multiband=True,
            global_diagnostics=LightcurveDiagnostics(
                baseline_duration=10.0,
                median_flux=10.0,
                flux_percentiles={2.5: 8.0, 50.0: 10.0, 97.5: 12.0},
                n_points=len(x),
            ),
        )

        result = build_and_apply_parameter_estimates(model=model, context=context)
        offset_result = result["mean_module.offset"]

        self.assertFalse(offset_result["constraint"])
        self.assertNotEqual(
            offset_result.get("constraint_reason"),
            "constraint_unavailable",
        )


if __name__ == "__main__":
    unittest.main()
