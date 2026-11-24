"""Tests for the alternative GP model classes."""
import unittest

import gpytorch
import torch

from pgmuvi.lightcurve import (
    Lightcurve,
)


class TestNewGPModels(unittest.TestCase):
    """Smoke tests for the new 2D GP model classes."""

    def setUp(self):
        from pgmuvi.gps import (
            TwoDSpectralMixturePowerLawMeanGPModel,
            TwoDSpectralMixturePowerLawMeanKISSGPModel,
            TwoDSpectralMixtureDustMeanGPModel,
            TwoDSpectralMixtureDustMeanKISSGPModel,
        )
        self.models = {
            "2DPowerLaw": TwoDSpectralMixturePowerLawMeanGPModel,
            "2DPowerLawSKI": TwoDSpectralMixturePowerLawMeanKISSGPModel,
            "2DDust": TwoDSpectralMixtureDustMeanGPModel,
            "2DDustSKI": TwoDSpectralMixtureDustMeanKISSGPModel,
        }
        # 2D training data: columns are (time, wavelength)
        self.train_x = torch.tensor(
            [[0.0, 0.5], [1.0, 1.0], [2.0, 2.0], [3.0, 0.5]], dtype=torch.float32
        )
        self.train_y = torch.tensor([1.0, 0.8, 0.6, 1.1], dtype=torch.float32)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def test_instantiation(self):
        """All new 2D GP model classes can be instantiated."""
        for name, ModelClass in self.models.items():
            with self.subTest(model=name):
                model = ModelClass(self.train_x, self.train_y, self.likelihood)
                self.assertIsNotNone(model)

    def test_forward_returns_mvn(self):
        """Forward pass returns a MultivariateNormal distribution."""
        from gpytorch.distributions import MultivariateNormal
        for name, ModelClass in self.models.items():
            with self.subTest(model=name):
                model = ModelClass(self.train_x, self.train_y, self.likelihood)
                model.eval()
                with torch.no_grad():
                    out = model(self.train_x)
                self.assertIsInstance(out, MultivariateNormal)

    def test_set_model_shortcut(self):
        """Lightcurve.set_model() accepts the new model shortcut strings."""
        for shortcut in ["2DPowerLaw", "2DPowerLawSKI", "2DDust", "2DDustSKI"]:
            with self.subTest(shortcut=shortcut):
                lc = Lightcurve(self.train_x, self.train_y)
                lc.set_model(model=shortcut, num_mixtures=2)
                self.assertIsNotNone(lc.model)


class TestSimpleKernelMeanGPModels(unittest.TestCase):
    """Smoke tests for DustMeanGPModel and PowerLawMeanGPModel."""

    def setUp(self):
        from pgmuvi.gps import DustMeanGPModel, PowerLawMeanGPModel
        self.models = {
            "DustMean": DustMeanGPModel,
            "PowerLawMean": PowerLawMeanGPModel,
        }
        # 2D training data: columns are (time, wavelength)
        self.train_x = torch.tensor(
            [[0.0, 0.5], [1.0, 1.0], [2.0, 2.0], [3.0, 0.5]], dtype=torch.float32
        )
        self.train_y = torch.tensor([1.0, 0.8, 0.6, 1.1], dtype=torch.float32)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def test_instantiation(self):
        """DustMeanGPModel and PowerLawMeanGPModel can be instantiated."""
        for name, ModelClass in self.models.items():
            with self.subTest(model=name):
                model = ModelClass(self.train_x, self.train_y, self.likelihood)
                self.assertIsNotNone(model)

    def test_forward_returns_mvn(self):
        """Forward pass returns a MultivariateNormal distribution."""
        from gpytorch.distributions import MultivariateNormal
        for name, ModelClass in self.models.items():
            with self.subTest(model=name):
                model = ModelClass(self.train_x, self.train_y, self.likelihood)
                model.eval()
                with torch.no_grad():
                    out = model(self.train_x)
                self.assertIsInstance(out, MultivariateNormal)

    def test_correct_mean_module(self):
        """DustMeanGPModel uses DustMean; PowerLawMeanGPModel uses PowerLawMean."""
        from pgmuvi.gps import DustMean, PowerLawMean, DustMeanGPModel, PowerLawMeanGPModel
        dust_model = DustMeanGPModel(self.train_x, self.train_y, self.likelihood)
        pl_model = PowerLawMeanGPModel(self.train_x, self.train_y, self.likelihood)
        self.assertIsInstance(dust_model.mean_module, DustMean)
        self.assertIsInstance(pl_model.mean_module, PowerLawMean)

    def test_set_model_shortcut(self):
        """Lightcurve.set_model() accepts '2DDustMean' and '2DPowerLawMean'."""
        for shortcut in ["2DDustMean", "2DPowerLawMean"]:
            with self.subTest(shortcut=shortcut):
                lc = Lightcurve(self.train_x, self.train_y)
                lc.set_model(model=shortcut)
                self.assertIsNotNone(lc.model)

    def test_wavelength_dependent_mean_module_strings(self):
        """WavelengthDependentGPModel accepts 'dust' and 'power_law' mean_module strings."""
        from pgmuvi.gps import WavelengthDependentGPModel, DustMean, PowerLawMean
        for mean_str, ExpectedCls in [("dust", DustMean), ("power_law", PowerLawMean)]:
            with self.subTest(mean_module=mean_str):
                model = WavelengthDependentGPModel(
                    self.train_x, self.train_y, self.likelihood, mean_module=mean_str
                )
                self.assertIsInstance(model.mean_module, ExpectedCls)

    def test_kernel_types_can_be_changed(self):
        """DustMeanGPModel and PowerLawMeanGPModel accept time/wavelength kernel types."""
        from pgmuvi.gps import DustMeanGPModel, PowerLawMeanGPModel
        for ModelClass in (DustMeanGPModel, PowerLawMeanGPModel):
            with self.subTest(model=ModelClass.__name__):
                model = ModelClass(
                    self.train_x,
                    self.train_y,
                    self.likelihood,
                    time_kernel_type="rbf",
                    wavelength_kernel_type="matern",
                )
                self.assertIsNotNone(model)

    def test_num_mixtures_forwarded_to_alt_model(self):
        """set_model() forwards num_mixtures to model_dic_alt models.

        Regression test: previously num_mixtures was silently dropped for
        WavelengthDependentGPModel (and other alt-path models). Now it must
        reach the SpectralMixtureKernel when time_kernel_type='sm'.
        """
        from gpytorch.kernels import SpectralMixtureKernel
        lc = Lightcurve(self.train_x, self.train_y)
        lc.set_model(
            model="2DWavelengthDependent",
            time_kernel_type="sm",
            num_mixtures=2,
        )
        # The time kernel is the first kernel in the product (AdditiveKernel
        # wraps two ScaleKernels; for separable models the covar_module is a
        # ProductKernel whose first child is the time kernel).
        time_kernel = lc.model.covar_module.kernels[0]
        # Unwrap ScaleKernel if present
        if hasattr(time_kernel, "base_kernel"):
            time_kernel = time_kernel.base_kernel
        self.assertIsInstance(time_kernel, SpectralMixtureKernel)
        self.assertEqual(time_kernel.num_mixtures, 2)
