"""Tests for alternative GP model classes in pgmuvi/gps.py."""

import unittest
import torch
import gpytorch

# All alternative models now live in gps.py
from pgmuvi.gps import (
    QuasiPeriodicGPModel,
    MaternGPModel,
    PeriodicPlusStochasticGPModel,
    SeparableGPModel,
    AchromaticGPModel,
    WavelengthDependentGPModel,
    LinearMeanQuasiPeriodicGPModel,
)


def _make_1d_data(n=50, period=5.0, noise=0.0):
    """Create simple 1D synthetic data."""
    t = torch.linspace(0, 20, n, dtype=torch.float32)
    y = torch.sin(2 * 3.14159 * t / period)
    if noise > 0:
        y = y + noise * torch.randn(n)
    return t, y


def _make_2d_data(n=30):
    """Create simple 2D synthetic multiwavelength data."""
    t = torch.linspace(0, 10, n, dtype=torch.float32)
    wl = torch.cat([
        torch.ones(n // 2, dtype=torch.float32) * 500.0,
        torch.ones(n - n // 2, dtype=torch.float32) * 700.0,
    ])
    x = torch.stack([t, wl], dim=1)
    y = torch.sin(2 * 3.14159 * t / 3.0)
    return x, y


class TestQuasiPeriodicGPModel(unittest.TestCase):
    """Tests for QuasiPeriodicGPModel."""

    def setUp(self):
        self.t, self.y = _make_1d_data(period=5.0)
        self.lik = gpytorch.likelihoods.GaussianLikelihood()
        self.model = QuasiPeriodicGPModel(self.t, self.y, self.lik, period=5.0)

    def test_forward_shape(self):
        """Forward pass returns distribution with correct mean shape."""
        self.model.eval()
        self.lik.eval()
        with torch.no_grad():
            pred = self.model(self.t)
        self.assertEqual(pred.mean.shape, self.t.shape)

    def test_has_covar_module(self):
        """Model has a covar_module attribute."""
        self.assertTrue(hasattr(self.model, "covar_module"))

    def test_has_sci_kernel(self):
        """Model has a sci_kernel alias."""
        self.assertTrue(hasattr(self.model, "sci_kernel"))

    def test_period_initialized(self):
        """Period is initialized to the provided value.

        The quasi-periodic kernel is ScaleKernel(ProductKernel(
        PeriodicKernel(), RBFKernel())).  The period lives at
        covar_module.base_kernel.kernels[0].period_length.
        """
        period = float(
            self.model.covar_module.base_kernel.kernels[0].period_length.detach()
        )
        self.assertAlmostEqual(period, 5.0, places=3)

    def test_default_period(self):
        """Default period is half the data span."""
        lik = gpytorch.likelihoods.GaussianLikelihood()
        model = QuasiPeriodicGPModel(self.t, self.y, lik)
        period = float(
            model.covar_module.base_kernel.kernels[0].period_length.detach()
        )
        expected = float((self.t.max() - self.t.min()) / 2.0)
        self.assertAlmostEqual(period, expected, places=3)

    def test_covar_is_scale_product(self):
        """Covariance module is ScaleKernel(ProductKernel(Periodic, RBF))."""
        self.assertIsInstance(self.model.covar_module, gpytorch.kernels.ScaleKernel)
        self.assertIsInstance(
            self.model.covar_module.base_kernel, gpytorch.kernels.ProductKernel
        )

    def test_named_parameters_not_empty(self):
        """Model has trainable parameters."""
        params = list(self.model.named_parameters())
        self.assertGreater(len(params), 0)


class TestMaternGPModel(unittest.TestCase):
    """Tests for MaternGPModel."""

    def setUp(self):
        self.t, self.y = _make_1d_data()
        self.lik = gpytorch.likelihoods.GaussianLikelihood()

    def test_forward_shape_nu_half(self):
        model = MaternGPModel(self.t, self.y, self.lik, nu=0.5)
        model.eval()
        with torch.no_grad():
            pred = model(self.t)
        self.assertEqual(pred.mean.shape, self.t.shape)

    def test_forward_shape_nu_one_and_half(self):
        lik = gpytorch.likelihoods.GaussianLikelihood()
        model = MaternGPModel(self.t, self.y, lik, nu=1.5)
        model.eval()
        with torch.no_grad():
            pred = model(self.t)
        self.assertEqual(pred.mean.shape, self.t.shape)

    def test_forward_shape_nu_two_and_half(self):
        lik = gpytorch.likelihoods.GaussianLikelihood()
        model = MaternGPModel(self.t, self.y, lik, nu=2.5)
        model.eval()
        with torch.no_grad():
            pred = model(self.t)
        self.assertEqual(pred.mean.shape, self.t.shape)

    def test_default_lengthscale(self):
        """Default lengthscale is quarter of data span."""
        lik = gpytorch.likelihoods.GaussianLikelihood()
        model = MaternGPModel(self.t, self.y, lik)
        ls = float(model.covar_module.base_kernel.lengthscale.detach())
        expected = float((self.t.max() - self.t.min()) / 4.0)
        self.assertAlmostEqual(ls, expected, places=3)


class TestPeriodicPlusStochasticGPModel(unittest.TestCase):
    """Tests for PeriodicPlusStochasticGPModel."""

    def setUp(self):
        self.t, self.y = _make_1d_data(period=5.0)
        self.lik = gpytorch.likelihoods.GaussianLikelihood()
        self.model = PeriodicPlusStochasticGPModel(self.t, self.y, self.lik, period=5.0)

    def test_forward_shape(self):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.t)
        self.assertEqual(pred.mean.shape, self.t.shape)

    def test_is_additive_kernel(self):
        """Model uses AdditiveKernel."""
        self.assertIsInstance(self.model.covar_module, gpytorch.kernels.AdditiveKernel)


class TestSeparableGPModel(unittest.TestCase):
    """Tests for SeparableGPModel."""

    def setUp(self):
        self.x, self.y = _make_2d_data()
        self.lik = gpytorch.likelihoods.GaussianLikelihood()
        self.model = SeparableGPModel(self.x, self.y, self.lik)

    def test_forward_shape(self):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.x)
        self.assertEqual(pred.mean.shape, self.y.shape)

    def test_covar_is_product_kernel(self):
        """SeparableGPModel uses a ProductKernel (no custom forward code)."""
        self.assertIsInstance(self.model.covar_module, gpytorch.kernels.ProductKernel)

    def test_custom_kernels(self):
        """SeparableGPModel accepts custom kernels."""
        lik = gpytorch.likelihoods.GaussianLikelihood()
        t_k = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        w_k = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        model = SeparableGPModel(self.x, self.y, lik, time_kernel=t_k, wavelength_kernel=w_k)
        self.assertIsInstance(model.covar_module, gpytorch.kernels.ProductKernel)


class TestAchromaticGPModel(unittest.TestCase):
    """Tests for AchromaticGPModel."""

    def setUp(self):
        self.x, self.y = _make_2d_data()

    def test_matern_time_kernel(self):
        lik = gpytorch.likelihoods.GaussianLikelihood()
        model = AchromaticGPModel(self.x, self.y, lik, time_kernel_type="matern")
        model.eval()
        with torch.no_grad():
            pred = model(self.x)
        self.assertEqual(pred.mean.shape, self.y.shape)

    def test_quasi_periodic_time_kernel(self):
        lik = gpytorch.likelihoods.GaussianLikelihood()
        model = AchromaticGPModel(
            self.x, self.y, lik, time_kernel_type="quasi_periodic", period=3.0
        )
        model.eval()
        with torch.no_grad():
            pred = model(self.x)
        self.assertEqual(pred.mean.shape, self.y.shape)

    def test_rbf_time_kernel(self):
        lik = gpytorch.likelihoods.GaussianLikelihood()
        model = AchromaticGPModel(self.x, self.y, lik, time_kernel_type="rbf")
        model.eval()
        with torch.no_grad():
            pred = model(self.x)
        self.assertEqual(pred.mean.shape, self.y.shape)

    def test_invalid_kernel_raises(self):
        lik = gpytorch.likelihoods.GaussianLikelihood()
        with self.assertRaises(ValueError):
            AchromaticGPModel(self.x, self.y, lik, time_kernel_type="invalid")

    def test_wavelength_kernel_is_constant(self):
        """Achromatic model uses a ConstantKernel for wavelength."""
        lik = gpytorch.likelihoods.GaussianLikelihood()
        model = AchromaticGPModel(self.x, self.y, lik)
        # covar_module is a ProductKernel with two sub-kernels
        self.assertIsInstance(model.covar_module, gpytorch.kernels.ProductKernel)
        wl_kernel = model.covar_module.kernels[1]
        self.assertIsInstance(wl_kernel, gpytorch.kernels.ConstantKernel)


class TestWavelengthDependentGPModel(unittest.TestCase):
    """Tests for WavelengthDependentGPModel."""

    def setUp(self):
        self.x, self.y = _make_2d_data()

    def test_forward_shape(self):
        lik = gpytorch.likelihoods.GaussianLikelihood()
        model = WavelengthDependentGPModel(self.x, self.y, lik)
        model.eval()
        with torch.no_grad():
            pred = model(self.x)
        self.assertEqual(pred.mean.shape, self.y.shape)

    def test_invalid_kernel_raises(self):
        lik = gpytorch.likelihoods.GaussianLikelihood()
        with self.assertRaises(ValueError):
            WavelengthDependentGPModel(self.x, self.y, lik, time_kernel_type="invalid")


class TestLinearMeanQuasiPeriodicGPModel(unittest.TestCase):
    """Tests for LinearMeanQuasiPeriodicGPModel."""

    def setUp(self):
        self.t, self.y = _make_1d_data(period=5.0)
        self.lik = gpytorch.likelihoods.GaussianLikelihood()
        self.model = LinearMeanQuasiPeriodicGPModel(self.t, self.y, self.lik, period=5.0)

    def test_forward_shape(self):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.t)
        self.assertEqual(pred.mean.shape, self.t.shape)

    def test_linear_mean(self):
        """Model uses a LinearMean."""
        self.assertIsInstance(self.model.mean_module, gpytorch.means.LinearMean)


if __name__ == "__main__":
    unittest.main()
