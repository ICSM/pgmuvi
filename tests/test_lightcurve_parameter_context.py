import unittest

import torch

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.parameter_specs import (
    ConstraintStrategy,
    GuessStrategy,
    ParameterDomain,
    ParameterRole,
    ParameterScale,
    ParameterSpec,
    ParameterSpecCollection,
)
from pgmuvi.parameter_context import (
    LightcurveDiagnostics,
    ParameterEstimationContext,
)


class DummyWorkflowMeanModule:
    def __init__(self):
        self.offset = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def register_constraint(self, parameter_name, constraint):
        self.calls.append((parameter_name, constraint))


class DummyWorkflowModel:
    def __init__(self):
        self.mean_module = DummyWorkflowMeanModule()

    def parameter_schema(self):
        return ParameterSpecCollection(
            [
                ParameterSpec(
                    name="mean_module.offset",
                    role=ParameterRole.OFFSET,
                    domain=ParameterDomain.FLUX,
                    scale=ParameterScale.LINEAR,
                    guess_strategy=GuessStrategy.MEDIAN_FLUX,
                    constraint_strategy=ConstraintStrategy.ROBUST_FLUX_RANGE,
                ),
            ]
        )


class DummyNoWorkflowModel:
    pass


class TestLightcurveParameterEstimationContext(unittest.TestCase):

    def test_build_parameter_estimation_context_for_1d_lightcurve(self):
        lc = Lightcurve(
            torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
            torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]),
        )

        context = lc._build_parameter_estimation_context()

        self.assertIsInstance(context, ParameterEstimationContext)
        self.assertIsInstance(context.global_diagnostics, LightcurveDiagnostics)

        self.assertFalse(context.is_multiband)
        self.assertEqual(context.global_diagnostics.n_points, 5)
        self.assertAlmostEqual(context.global_diagnostics.median_flux, 30.0)

        self.assertAlmostEqual(
            context.global_diagnostics.flux_percentiles[50.0],
            30.0,
        )
        self.assertAlmostEqual(
            context.global_diagnostics.flux_percentiles[2.5],
            11.0,
        )
        self.assertAlmostEqual(
            context.global_diagnostics.flux_percentiles[97.5],
            49.0,
        )

    def test_build_parameter_estimation_context_for_2d_lightcurve(self):
        xdata = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 2.0],
                [1.0, 2.0],
            ]
        )
        ydata = torch.tensor([10.0, 20.0, 100.0, 200.0])

        lc = Lightcurve(xdata, ydata)

        context = lc._build_parameter_estimation_context()

        self.assertIsInstance(context, ParameterEstimationContext)
        self.assertTrue(context.is_multiband)
        self.assertEqual(context.global_diagnostics.n_points, 4)
        self.assertAlmostEqual(context.global_diagnostics.median_flux, 60.0)
        self.assertAlmostEqual(
            context.global_diagnostics.baseline_duration,
            1.0,
        )
        self.assertAlmostEqual(
            context.global_diagnostics.median_cadence,
            1.0,
        )

    def test_build_parameter_estimation_context_ignores_nonfinite_flux_values(self):
        lc = Lightcurve(
            torch.tensor([0.0, 1.0, 2.0]),
            torch.tensor([10.0, float("nan"), 30.0]),
        )

        context = lc._build_parameter_estimation_context()
        self.assertEqual(context.global_diagnostics.n_points, 2)
        self.assertAlmostEqual(context.global_diagnostics.median_flux, 20.0)

    def test_apply_parameter_workflow_estimates_returns_none_without_schema(self):
        lc = Lightcurve(
            torch.tensor([0.0, 1.0, 2.0]),
            torch.tensor([10.0, 20.0, 30.0]),
        )
        lc.model = DummyNoWorkflowModel()

        result = lc._apply_parameter_workflow_estimates()

        self.assertIsNone(result)
        self.assertIsNone(lc.parameter_workflow_result)

    def test_apply_parameter_workflow_estimates_applies_supported_schema(self):
        lc = Lightcurve(
            torch.tensor([0.0, 1.0, 2.0]),
            torch.tensor([10.0, 20.0, 30.0]),
        )
        lc.model = DummyWorkflowModel()

        result = lc._apply_parameter_workflow_estimates()

        self.assertEqual(
            result,
            {
                "mean_module.offset": {
                    "value": True,
                    "constraint": True,
                },
            },
        )

        self.assertEqual(
            lc.parameter_workflow_result,
            result,
        )

        self.assertAlmostEqual(
            float(lc.model.mean_module.offset.item()),
            20.0,
        )

        self.assertEqual(
            lc.model.mean_module.calls[0][0],
            "offset",
        )

    def test_build_parameter_estimation_context_computes_1d_sampling_diagnostics(self):
        lc = Lightcurve(
            torch.tensor([0.0, 10.0, 20.0, 40.0]),
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
        )

        context = lc._build_parameter_estimation_context()

        self.assertAlmostEqual(
            context.global_diagnostics.baseline_duration,
            40.0,
        )
        self.assertAlmostEqual(
            context.global_diagnostics.median_cadence,
            10.0,
        )

    def test_parameter_workflow_result_initialized_to_none(self):
        lc = Lightcurve(
            torch.tensor([0.0, 1.0, 2.0]),
            torch.tensor([10.0, 20.0, 30.0]),
        )

        self.assertIsNone(
            lc.parameter_workflow_result,
        )

    def test_parameter_workflow_summary_without_results(self):
        lc = Lightcurve(
            torch.tensor([0.0, 1.0, 2.0]),
            torch.tensor([10.0, 20.0, 30.0]),
        )

        self.assertEqual(
            lc.get_parameter_workflow_summary(),
            {
                "available": False,
                "applied": 0,
                "skipped": 0,
                "applied_parameters": [],
                "skipped_parameters": [],
            },
        )

    def test_parameter_workflow_summary_counts_results(self):
        lc = Lightcurve(
            torch.tensor([0.0, 1.0, 2.0]),
            torch.tensor([10.0, 20.0, 30.0]),
        )

        lc.parameter_workflow_result = {
            "a": True,
            "b": True,
            "c": False,
        }

        self.assertEqual(
            lc.get_parameter_workflow_summary(),
            {
                "available": True,
                "applied": 2,
                "skipped": 1,
                "applied_parameters": ["a", "b"],
                "skipped_parameters": ["c"],
            },
        )

    def test_parameter_workflow_applies_real_matern_model_schema(self):
        import gpytorch

        from pgmuvi.gps import MaternGPModel

        lc = Lightcurve(
            torch.tensor([0.0, 1.0, 2.0, 3.0]),
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
        )

        lc.model = MaternGPModel(
            lc.xdata,
            lc.ydata,
            gpytorch.likelihoods.GaussianLikelihood(),
        )

        result = lc._apply_parameter_workflow_estimates()

        self.assertEqual(
            result,
            {
                "covar_module.outputscale": {
                    "value": True,
                    "constraint": True,
                },
                "covar_module.base_kernel.lengthscale": {
                    "value": True,
                    "constraint": True,
                },
            },
        )

        self.assertEqual(
            lc.parameter_workflow_result,
            result,
        )

        self.assertEqual(
            lc.get_parameter_workflow_summary(),
            {
                "available": True,
                "applied": 2,
                "skipped": 0,
                "applied_parameters": [
                    "covar_module.outputscale",
                    "covar_module.base_kernel.lengthscale",
                ],
                "skipped_parameters": [],
            },
        )

    def test_parameter_workflow_applies_real_spectral_mixture_model_schema(self):
        import gpytorch

        from pgmuvi.gps import SpectralMixtureGPModel

        lc = Lightcurve(
            torch.tensor([0.0, 1.0, 2.0, 3.0]),
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
        )

        lc.model = SpectralMixtureGPModel(
            lc.xdata,
            lc.ydata,
            gpytorch.likelihoods.GaussianLikelihood(),
            num_mixtures=2,
        )

        result = lc._apply_parameter_workflow_estimates()

        self.assertEqual(
            result,
            {
                "covar_module.mixture_means": {
                    "value": False,
                    "constraint": True,
                },
                "covar_module.mixture_scales": {
                    "value": True,
                    "constraint": True,
                },
                "covar_module.mixture_weights": {
                    "value": True,
                    "constraint": True,
                },
            },
        )

        self.assertEqual(
            lc.parameter_workflow_result,
            result,
        )
