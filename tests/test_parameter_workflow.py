import unittest

import torch

from pgmuvi.parameter_context import (
    LightcurveDiagnostics,
    ParameterEstimationContext,
)
from pgmuvi.parameter_specs import (
    ConstraintStrategy,
    GuessStrategy,
    ParameterDomain,
    ParameterRole,
    ParameterScale,
    ParameterSpec,
    ParameterSpecCollection,
)
from pgmuvi.parameter_workflow import (
    apply_parameter_estimates,
    build_and_apply_parameter_estimates,
    build_parameter_estimates,
    get_parameter_schema,
    model_supports_parameter_workflow,
)


class ModelWithWrongSchemaType:
    def parameter_schema(self):
        return "not-a-schema"


class ModelWithSchemaRequiringArguments:
    def parameter_schema(self, required_argument):
        return required_argument


class ModelWithSchema:
    def parameter_schema(self):
        return ParameterSpecCollection()


class ModelWithoutSchema:
    pass


class DummyMeanModule:
    def __init__(self):
        self.offset = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def register_constraint(self, parameter_name, constraint):
        self.calls.append((parameter_name, constraint))


class ModelWithRealSchema:
    def __init__(self):
        self.mean_module = DummyMeanModule()

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


class TestParameterWorkflow(unittest.TestCase):

    def test_returns_schema_when_available(self):
        schema = get_parameter_schema(
            ModelWithSchema()
        )

        self.assertIsInstance(
            schema,
            ParameterSpecCollection,
        )

        self.assertEqual(
            len(schema),
            0,
        )

    def test_returns_none_when_unavailable(self):
        self.assertIsNone(
            get_parameter_schema(ModelWithoutSchema())
        )

    def test_build_parameter_estimates_returns_none_without_schema(self):
        context = ParameterEstimationContext(is_multiband=False)

        self.assertIsNone(
            build_parameter_estimates(
                model=ModelWithoutSchema(),
                context=context,
            )
        )

    def test_build_parameter_estimates_uses_schema_and_context(self):
        model = ModelWithRealSchema()
        context = ParameterEstimationContext(
            is_multiband=False,
            global_diagnostics=LightcurveDiagnostics(
                median_flux=55.0,
                flux_percentiles={
                    2.5: 10.0,
                    97.5: 100.0,
                },
            ),
        )

        estimates = build_parameter_estimates(
            model=model,
            context=context,
        )

        estimate = estimates["mean_module.offset"]

        self.assertEqual(estimate.value, 55.0)
        self.assertEqual(estimate.constraint, (10.0, 100.0))

    def test_apply_parameter_estimates_returns_none_without_estimates(self):
        self.assertIsNone(
            apply_parameter_estimates(
                model=ModelWithoutSchema(),
                estimates=None,
            )
        )

    def test_apply_parameter_estimates_applies_values_and_constraints(self):
        model = ModelWithRealSchema()
        context = ParameterEstimationContext(
            is_multiband=False,
            global_diagnostics=LightcurveDiagnostics(
                median_flux=55.0,
                flux_percentiles={
                    2.5: 10.0,
                    97.5: 100.0,
                },
            ),
        )

        estimates = build_parameter_estimates(
            model=model,
            context=context,
        )

        results = apply_parameter_estimates(
            model=model,
            estimates=estimates,
        )

        self.assertEqual(
            results,
            {
                "mean_module.offset": {
                    "value": True,
                    "constraint": False,
                    "value_reason": None,
                    "constraint_reason": "constraint_not_enforceable_plain_parameter",
                    "constraint_action": "constraint_not_enforceable_plain_parameter",
                },
            },
        )

        self.assertAlmostEqual(
            float(model.mean_module.offset.item()),
            55.0,
        )

        self.assertEqual(model.mean_module.calls, [])

    def test_build_and_apply_parameter_estimates_runs_full_workflow(self):
        model = ModelWithRealSchema()
        context = ParameterEstimationContext(
            is_multiband=False,
            global_diagnostics=LightcurveDiagnostics(
                median_flux=55.0,
                flux_percentiles={
                    2.5: 10.0,
                    97.5: 100.0,
                },
            ),
        )

        results = build_and_apply_parameter_estimates(
            model=model,
            context=context,
        )

        self.assertEqual(
            results,
            {
                "mean_module.offset": {
                    "value": True,
                    "constraint": False,
                    "value_reason": None,
                    "constraint_reason": "constraint_not_enforceable_plain_parameter",
                    "constraint_action": "constraint_not_enforceable_plain_parameter",
                },
            },
        )

        self.assertAlmostEqual(
            float(model.mean_module.offset.item()),
            55.0,
        )

        self.assertEqual(model.mean_module.calls, [])

    def test_returns_none_when_schema_has_wrong_type(self):
        self.assertIsNone(
            get_parameter_schema(ModelWithWrongSchemaType())
        )


    def test_returns_none_when_schema_requires_arguments(self):
        self.assertIsNone(
            get_parameter_schema(ModelWithSchemaRequiringArguments())
        )

    def test_model_supports_parameter_workflow_when_schema_available(self):
        self.assertTrue(
            model_supports_parameter_workflow(
                ModelWithSchema()
            )
        )


    def test_model_supports_parameter_workflow_when_schema_missing(self):
        self.assertFalse(
            model_supports_parameter_workflow(
                ModelWithoutSchema()
            )
        )


    def test_model_supports_parameter_workflow_when_schema_invalid(self):
        self.assertFalse(
            model_supports_parameter_workflow(
                ModelWithWrongSchemaType()
            )
        )

    def test_matern_gp_model_exposes_parameter_schema(self):
        import gpytorch
        import torch

        from pgmuvi.gps import MaternGPModel

        train_x = torch.tensor([0.0, 1.0, 2.0])
        train_y = torch.tensor([1.0, 2.0, 3.0])

        model = MaternGPModel(
            train_x,
            train_y,
            gpytorch.likelihoods.GaussianLikelihood(),
        )

        schema = model.parameter_schema()

        self.assertIsNotNone(schema)

        names = schema.names()

        self.assertIn(
            "covar_module.outputscale",
            names,
        )

        self.assertIn(
            "covar_module.base_kernel.lengthscale",
            names,
        )

    def test_quasi_periodic_gp_model_exposes_parameter_schema(self):
        import gpytorch
        import torch

        from pgmuvi.gps import QuasiPeriodicGPModel

        train_x = torch.tensor([0.0, 1.0, 2.0])
        train_y = torch.tensor([1.0, 2.0, 3.0])

        model = QuasiPeriodicGPModel(
            train_x,
            train_y,
            gpytorch.likelihoods.GaussianLikelihood(),
            period=2.0,
        )

        schema = model.parameter_schema()
        names = schema.names()

        self.assertIn(
            "covar_module.outputscale",
            names,
        )
        self.assertIn(
            "covar_module.base_kernel.kernels.0.period_length",
            names,
        )
        self.assertIn(
            "covar_module.base_kernel.kernels.1.lengthscale",
            names,
        )

    def test_spectral_mixture_gp_model_exposes_parameter_schema(self):
        import gpytorch
        import torch

        from pgmuvi.gps import SpectralMixtureGPModel

        train_x = torch.tensor([0.0, 1.0, 2.0])
        train_y = torch.tensor([1.0, 2.0, 3.0])

        model = SpectralMixtureGPModel(
            train_x,
            train_y,
            gpytorch.likelihoods.GaussianLikelihood(),
            num_mixtures=3,
        )

        schema = model.parameter_schema()
        names = schema.names()

        self.assertIn("covar_module.mixture_means", names)
        self.assertIn("covar_module.mixture_scales", names)
        self.assertIn("covar_module.mixture_weights", names)

        self.assertEqual(
            schema["covar_module.mixture_means"].shape,
            (3,),
        )

    def test_spectral_mixture_linear_mean_gp_model_exposes_parameter_schema(self):
        import gpytorch
        import torch

        from pgmuvi.gps import SpectralMixtureLinearMeanGPModel

        train_x = torch.tensor([0.0, 1.0, 2.0])
        train_y = torch.tensor([1.0, 2.0, 3.0])

        model = SpectralMixtureLinearMeanGPModel(
            train_x,
            train_y,
            gpytorch.likelihoods.GaussianLikelihood(),
            num_mixtures=3,
        )

        schema = model.parameter_schema()
        names = schema.names()

        self.assertIn("covar_module.mixture_means", names)
        self.assertIn("covar_module.mixture_scales", names)
        self.assertIn("covar_module.mixture_weights", names)

        self.assertEqual(
            schema["covar_module.mixture_means"].shape,
            (3,),
        )

    def test_two_d_spectral_mixture_gp_model_exposes_parameter_schema(self):
        import gpytorch
        import torch

        from pgmuvi.gps import TwoDSpectralMixtureGPModel

        train_x = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ]
        )
        train_y = torch.tensor([1.0, 2.0, 3.0])

        model = TwoDSpectralMixtureGPModel(
            train_x,
            train_y,
            gpytorch.likelihoods.GaussianLikelihood(),
            num_mixtures=3,
        )

        schema = model.parameter_schema()
        names = schema.names()

        self.assertIn("covar_module.mixture_means", names)
        self.assertIn("covar_module.mixture_scales", names)
        self.assertIn("covar_module.mixture_weights", names)

        self.assertEqual(
            schema["covar_module.mixture_means"].shape,
            (3, 1, 2),
        )

    def test_two_d_spectral_mixture_linear_mean_gp_model_exposes_parameter_schema(self):
        import gpytorch
        import torch

        from pgmuvi.gps import TwoDSpectralMixtureLinearMeanGPModel

        train_x = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ]
        )
        train_y = torch.tensor([1.0, 2.0, 3.0])

        model = TwoDSpectralMixtureLinearMeanGPModel(
            train_x,
            train_y,
            gpytorch.likelihoods.GaussianLikelihood(),
            num_mixtures=3,
        )

        schema = model.parameter_schema()
        names = schema.names()

        self.assertIn("covar_module.mixture_means", names)
        self.assertIn("covar_module.mixture_scales", names)
        self.assertIn("covar_module.mixture_weights", names)

        self.assertEqual(
            schema["covar_module.mixture_means"].shape,
            (3, 1, 2),
        )

    def test_spectral_mixture_kiss_gp_model_exposes_parameter_schema(self):
        import gpytorch
        import torch

        from pgmuvi.gps import SpectralMixtureKISSGPModel

        train_x = torch.tensor([0.0, 1.0, 2.0, 3.0])
        train_y = torch.tensor([1.0, 2.0, 3.0, 4.0])

        model = SpectralMixtureKISSGPModel(
            train_x,
            train_y,
            gpytorch.likelihoods.GaussianLikelihood(),
            num_mixtures=3,
            grid_size=8,
        )

        schema = model.parameter_schema()
        names = schema.names()

        self.assertIn("covar_module.base_kernel.mixture_means", names)
        self.assertIn("covar_module.base_kernel.mixture_scales", names)
        self.assertIn("covar_module.base_kernel.mixture_weights", names)

        self.assertEqual(
            schema["covar_module.base_kernel.mixture_means"].shape,
            (3,),
        )

    def test_spectral_mixture_linear_mean_kiss_gp_model_exposes_parameter_schema(self):
        import gpytorch
        import torch

        from pgmuvi.gps import SpectralMixtureLinearMeanKISSGPModel

        train_x = torch.tensor([0.0, 1.0, 2.0, 3.0])
        train_y = torch.tensor([1.0, 2.0, 3.0, 4.0])

        model = SpectralMixtureLinearMeanKISSGPModel(
            train_x,
            train_y,
            gpytorch.likelihoods.GaussianLikelihood(),
            num_mixtures=3,
            grid_size=8,
        )

        schema = model.parameter_schema()
        names = schema.names()

        self.assertIn("covar_module.base_kernel.mixture_means", names)
        self.assertIn("covar_module.base_kernel.mixture_scales", names)
        self.assertIn("covar_module.base_kernel.mixture_weights", names)

        self.assertEqual(
            schema["covar_module.base_kernel.mixture_means"].shape,
            (3,),
        )


    def test_two_d_spectral_mixture_kiss_gp_model_exposes_parameter_schema(self):
        import gpytorch
        import torch

        from pgmuvi.gps import TwoDSpectralMixtureKISSGPModel

        train_x = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ]
        )
        train_y = torch.tensor([1.0, 2.0, 3.0])

        model = TwoDSpectralMixtureKISSGPModel(
            train_x,
            train_y,
            gpytorch.likelihoods.GaussianLikelihood(),
            num_mixtures=3,
            grid_size=[8, 4],
        )

        schema = model.parameter_schema()
        names = schema.names()

        self.assertIn("covar_module.base_kernel.mixture_means", names)
        self.assertIn("covar_module.base_kernel.mixture_scales", names)
        self.assertIn("covar_module.base_kernel.mixture_weights", names)

        self.assertEqual(
            schema["covar_module.base_kernel.mixture_means"].shape,
            (3, 1, 2),
        )

    def test_two_d_spectral_mixture_linear_mean_kiss_gp_model_exposes_parameter_schema(self):
        import gpytorch
        import torch

        from pgmuvi.gps import TwoDSpectralMixtureLinearMeanKISSGPModel

        train_x = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ]
        )
        train_y = torch.tensor([1.0, 2.0, 3.0])

        model = TwoDSpectralMixtureLinearMeanKISSGPModel(
            train_x,
            train_y,
            gpytorch.likelihoods.GaussianLikelihood(),
            num_mixtures=3,
            grid_size=[8, 4],
        )

        schema = model.parameter_schema()
        names = schema.names()

        self.assertIn("covar_module.base_kernel.mixture_means", names)
        self.assertIn("covar_module.base_kernel.mixture_scales", names)
        self.assertIn("covar_module.base_kernel.mixture_weights", names)

        self.assertEqual(
            schema["covar_module.base_kernel.mixture_means"].shape,
            (3, 1, 2),
        )

    def test_two_d_spectral_mixture_power_law_mean_gp_model_exposes_parameter_schema(self):
        import gpytorch
        import torch

        from pgmuvi.gps import TwoDSpectralMixturePowerLawMeanGPModel

        train_x = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ]
        )
        train_y = torch.tensor([1.0, 2.0, 3.0])

        model = TwoDSpectralMixturePowerLawMeanGPModel(
            train_x,
            train_y,
            gpytorch.likelihoods.GaussianLikelihood(),
            num_mixtures=3,
        )

        schema = model.parameter_schema()
        names = schema.names()

        self.assertIn("mean_module.offset", names)
        self.assertIn("mean_module.weight", names)
        self.assertIn("mean_module.exponent", names)

        self.assertIn("covar_module.mixture_means", names)
        self.assertIn("covar_module.mixture_scales", names)
        self.assertIn("covar_module.mixture_weights", names)

        self.assertEqual(
            schema["covar_module.mixture_means"].shape,
            (3, 1, 2),
        )

    def test_two_d_spectral_mixture_power_law_mean_kiss_gp_model_exposes_parameter_schema(self):
        import gpytorch
        import torch

        from pgmuvi.gps import TwoDSpectralMixturePowerLawMeanKISSGPModel

        train_x = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ]
        )
        train_y = torch.tensor([1.0, 2.0, 3.0])

        model = TwoDSpectralMixturePowerLawMeanKISSGPModel(
            train_x,
            train_y,
            gpytorch.likelihoods.GaussianLikelihood(),
            num_mixtures=3,
            grid_size=[8, 4],
        )

        schema = model.parameter_schema()
        names = schema.names()

        self.assertIn("mean_module.offset", names)
        self.assertIn("mean_module.weight", names)
        self.assertIn("mean_module.exponent", names)

        self.assertIn("covar_module.base_kernel.mixture_means", names)
        self.assertIn("covar_module.base_kernel.mixture_scales", names)
        self.assertIn("covar_module.base_kernel.mixture_weights", names)

        self.assertEqual(
            schema["covar_module.base_kernel.mixture_means"].shape,
            (3, 1, 2),
        )

    def test_two_d_spectral_mixture_dust_mean_gp_model_exposes_parameter_schema(self):
        import gpytorch
        import torch

        from pgmuvi.gps import TwoDSpectralMixtureDustMeanGPModel

        train_x = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ]
        )
        train_y = torch.tensor([1.0, 2.0, 3.0])

        model = TwoDSpectralMixtureDustMeanGPModel(
            train_x,
            train_y,
            gpytorch.likelihoods.GaussianLikelihood(),
            num_mixtures=3,
        )

        schema = model.parameter_schema()
        names = schema.names()

        self.assertIn("mean_module.offset", names)
        self.assertIn("mean_module.log_amplitude", names)
        self.assertIn("mean_module.log_tau", names)
        self.assertIn("mean_module.log_alpha", names)

        self.assertIn("covar_module.mixture_means", names)
        self.assertIn("covar_module.mixture_scales", names)
        self.assertIn("covar_module.mixture_weights", names)

        self.assertEqual(
            schema["covar_module.mixture_means"].shape,
            (3, 1, 2),
        )

    def test_two_d_spectral_mixture_dust_mean_kiss_gp_model_exposes_parameter_schema(self):
        import gpytorch
        import torch

        from pgmuvi.gps import TwoDSpectralMixtureDustMeanKISSGPModel

        train_x = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ]
        )
        train_y = torch.tensor([1.0, 2.0, 3.0])

        model = TwoDSpectralMixtureDustMeanKISSGPModel(
            train_x,
            train_y,
            gpytorch.likelihoods.GaussianLikelihood(),
            num_mixtures=3,
            grid_size=[8, 4],
        )

        schema = model.parameter_schema()
        names = schema.names()

        self.assertIn("mean_module.offset", names)
        self.assertIn("mean_module.log_amplitude", names)
        self.assertIn("mean_module.log_tau", names)
        self.assertIn("mean_module.log_alpha", names)

        self.assertIn("covar_module.base_kernel.mixture_means", names)
        self.assertIn("covar_module.base_kernel.mixture_scales", names)
        self.assertIn("covar_module.base_kernel.mixture_weights", names)

        self.assertEqual(
            schema["covar_module.base_kernel.mixture_means"].shape,
            (3, 1, 2),
        )

    def test_sparse_spectral_mixture_gp_model_exposes_parameter_schema(self):
        import gpytorch
        import torch

        from pgmuvi.gps import SparseSpectralMixtureGPModel

        train_x = torch.tensor([0.0, 1.0, 2.0])
        train_y = torch.tensor([1.0, 2.0, 3.0])
        inducing_points = torch.tensor([0.0, 1.0])

        model = SparseSpectralMixtureGPModel(
            train_x,
            train_y,
            gpytorch.likelihoods.GaussianLikelihood(),
            num_mixtures=3,
            inducing_points=inducing_points,
        )

        schema = model.parameter_schema()
        names = schema.names()

        self.assertIn("covar_module.mixture_means", names)
        self.assertIn("covar_module.mixture_scales", names)
        self.assertIn("covar_module.mixture_weights", names)

        self.assertEqual(
            schema["covar_module.mixture_means"].shape,
            (3,),
        )

    def test_periodic_plus_stochastic_gp_model_exposes_parameter_schema(self):
        import gpytorch
        import torch

        from pgmuvi.gps import PeriodicPlusStochasticGPModel

        train_x = torch.tensor([0.0, 1.0, 2.0, 3.0])
        train_y = torch.tensor([1.0, 2.0, 3.0, 4.0])

        model = PeriodicPlusStochasticGPModel(
            train_x,
            train_y,
            gpytorch.likelihoods.GaussianLikelihood(),
            period=2.0,
        )

        schema = model.parameter_schema()
        names = schema.names()

        self.assertIn("covar_module.kernels.0.outputscale", names)
        self.assertIn(
            "covar_module.kernels.0.base_kernel.kernels.0.period_length",
            names,
        )
        self.assertIn(
            "covar_module.kernels.0.base_kernel.kernels.1.lengthscale",
            names,
        )
        self.assertIn("covar_module.kernels.1.outputscale", names)
        self.assertIn("covar_module.kernels.1.base_kernel.lengthscale", names)

    def test_linear_mean_quasi_periodic_gp_model_exposes_parameter_schema(self):
        import gpytorch
        import torch

        from pgmuvi.gps import LinearMeanQuasiPeriodicGPModel

        train_x = torch.tensor([0.0, 1.0, 2.0, 3.0])
        train_y = torch.tensor([1.0, 2.0, 3.0, 4.0])

        model = LinearMeanQuasiPeriodicGPModel(
            train_x,
            train_y,
            gpytorch.likelihoods.GaussianLikelihood(),
            period=2.0,
        )

        schema = model.parameter_schema()
        names = schema.names()

        self.assertIn("covar_module.outputscale", names)
        self.assertIn(
            "covar_module.base_kernel.kernels.0.period_length",
            names,
        )
        self.assertIn(
            "covar_module.base_kernel.kernels.1.lengthscale",
            names,
        )

    def test_separable_gp_model_exposes_default_parameter_schema(self):
        import gpytorch
        import torch

        from pgmuvi.gps import SeparableGPModel

        train_x = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ]
        )
        train_y = torch.tensor([1.0, 2.0, 3.0])

        model = SeparableGPModel(
            train_x,
            train_y,
            gpytorch.likelihoods.GaussianLikelihood(),
        )

        schema = model.parameter_schema()
        names = schema.names()

        self.assertIn("covar_module.kernels.0.outputscale", names)
        self.assertIn("covar_module.kernels.0.base_kernel.lengthscale", names)
        self.assertIn("covar_module.kernels.1.outputscale", names)
        self.assertIn("covar_module.kernels.1.base_kernel.lengthscale", names)

        self.assertEqual(
            schema["covar_module.kernels.0.base_kernel.lengthscale"].domain,
            ParameterDomain.TIME,
        )
        self.assertEqual(
            schema["covar_module.kernels.1.base_kernel.lengthscale"].domain,
            ParameterDomain.WAVELENGTH,
        )

    def test_achromatic_gp_model_inherits_separable_parameter_schema(self):
        import gpytorch
        import torch

        from pgmuvi.gps import AchromaticGPModel

        train_x = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 2.0],
            ]
        )
        train_y = torch.tensor([1.0, 2.0, 3.0, 4.0])

        model = AchromaticGPModel(
            train_x,
            train_y,
            gpytorch.likelihoods.GaussianLikelihood(),
        )

        schema = model.parameter_schema()
        names = schema.names()

        self.assertIn("covar_module.kernels.0.outputscale", names)
        self.assertIn(
            "covar_module.kernels.0.base_kernel.lengthscale",
            names,
        )

        self.assertNotIn("covar_module.kernels.1.outputscale", names)
        self.assertNotIn(
            "covar_module.kernels.1.base_kernel.lengthscale",
            names,
        )

        self.assertEqual(
            schema["covar_module.kernels.0.base_kernel.lengthscale"].domain,
            ParameterDomain.TIME,
        )

    def test_wavelength_dependent_gp_model_exposes_mean_and_kernel_schema(self):
        import gpytorch
        import torch

        from pgmuvi.gps import WavelengthDependentGPModel

        train_x = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 2.0],
            ]
        )
        train_y = torch.tensor([1.0, 2.0, 3.0, 4.0])

        model = WavelengthDependentGPModel(
            train_x,
            train_y,
            gpytorch.likelihoods.GaussianLikelihood(),
        )

        schema = model.parameter_schema()
        names = schema.names()

        self.assertIn("mean_module.weights", names)
        self.assertIn("mean_module.bias", names)

        self.assertIn("covar_module.kernels.0.outputscale", names)
        self.assertIn(
            "covar_module.kernels.0.base_kernel.lengthscale",
            names,
        )
        self.assertIn("covar_module.kernels.1.outputscale", names)
        self.assertIn(
            "covar_module.kernels.1.base_kernel.lengthscale",
            names,
        )

    def test_dust_mean_gp_model_inherits_wavelength_dependent_schema(self):
        import gpytorch
        import torch

        from pgmuvi.gps import DustMeanGPModel

        train_x = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 2.0],
            ]
        )
        train_y = torch.tensor([1.0, 2.0, 3.0, 4.0])

        model = DustMeanGPModel(
            train_x,
            train_y,
            gpytorch.likelihoods.GaussianLikelihood(),
        )

        schema = model.parameter_schema()
        names = schema.names()

        self.assertIn("mean_module.offset", names)
        self.assertIn("mean_module.log_amplitude", names)
        self.assertIn("mean_module.log_tau", names)
        self.assertIn("mean_module.log_alpha", names)

        self.assertIn("covar_module.kernels.0.outputscale", names)
        self.assertIn(
            "covar_module.kernels.0.base_kernel.lengthscale",
            names,
        )
        self.assertIn("covar_module.kernels.1.outputscale", names)
        self.assertIn(
            "covar_module.kernels.1.base_kernel.lengthscale",
            names,
        )

    def test_power_law_mean_gp_model_inherits_wavelength_dependent_schema(self):
        import gpytorch
        import torch

        from pgmuvi.gps import PowerLawMeanGPModel

        train_x = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 2.0],
            ]
        )
        train_y = torch.tensor([1.0, 2.0, 3.0, 4.0])

        model = PowerLawMeanGPModel(
            train_x,
            train_y,
            gpytorch.likelihoods.GaussianLikelihood(),
        )

        schema = model.parameter_schema()
        names = schema.names()

        self.assertIn("mean_module.offset", names)
        self.assertIn("mean_module.weight", names)
        self.assertIn("mean_module.exponent", names)

        self.assertIn("covar_module.kernels.0.outputscale", names)
        self.assertIn(
            "covar_module.kernels.0.base_kernel.lengthscale",
            names,
        )
        self.assertIn("covar_module.kernels.1.outputscale", names)
        self.assertIn(
            "covar_module.kernels.1.base_kernel.lengthscale",
            names,
        )


if __name__ == "__main__":
    unittest.main()
