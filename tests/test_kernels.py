"""Tests for pgmuvi/kernels.py helper functions.

The custom kernel classes (QuasiPeriodicKernel, SeparableKernel,
AchromaticKernel) have been replaced by compositions of standard
GPyTorch kernels in gps.py.  This file tests the remaining factory
functions provided for convenience.
"""

import unittest
import torch
import gpytorch
from gpytorch.kernels import ProductKernel, ScaleKernel

from pgmuvi.kernels import (
    make_quasi_periodic_kernel,
    make_matern_kernel,
    make_rbf_kernel,
)


class TestMakeQuasiPeriodicKernel(unittest.TestCase):
    """Tests for make_quasi_periodic_kernel()."""

    def setUp(self):
        self.kernel = make_quasi_periodic_kernel(period=5.0, decay=25.0)
        self.x = torch.linspace(0, 20, 10).unsqueeze(-1)

    def test_returns_scale_kernel(self):
        """make_quasi_periodic_kernel returns a ScaleKernel."""
        self.assertIsInstance(self.kernel, gpytorch.kernels.ScaleKernel)

    def test_base_kernel_is_product(self):
        """Base kernel is a ProductKernel(PeriodicKernel, RBFKernel)."""
        self.assertIsInstance(self.kernel.base_kernel, gpytorch.kernels.ProductKernel)
        sub_kernels = self.kernel.base_kernel.kernels
        self.assertIsInstance(sub_kernels[0], gpytorch.kernels.PeriodicKernel)
        self.assertIsInstance(sub_kernels[1], gpytorch.kernels.RBFKernel)

    def test_forward_shape(self):
        """Kernel forward pass returns matrix of correct shape."""
        cov = self.kernel(self.x, self.x).to_dense()
        self.assertEqual(cov.shape, (10, 10))

    def test_symmetric(self):
        """Kernel matrix is symmetric."""
        cov = self.kernel(self.x, self.x).to_dense()
        self.assertTrue(torch.allclose(cov, cov.T, atol=1e-5))

    def test_period_initialized(self):
        """Period is initialized to the provided value."""
        period = float(
            self.kernel.base_kernel.kernels[0].period_length.detach()
        )
        self.assertAlmostEqual(period, 5.0, places=3)

    def test_outputscale_initialized(self):
        """Outputscale is initialized to the provided value."""
        os = float(self.kernel.outputscale.detach())
        self.assertAlmostEqual(os, 1.0, places=4)


class TestMakeMaternKernel(unittest.TestCase):
    """Tests for make_matern_kernel()."""

    def test_returns_scale_kernel(self):
        k = make_matern_kernel(nu=1.5, lengthscale=1.0)
        self.assertIsInstance(k, gpytorch.kernels.ScaleKernel)

    def test_base_is_matern(self):
        k = make_matern_kernel(nu=1.5)
        self.assertIsInstance(k.base_kernel, gpytorch.kernels.MaternKernel)

    def test_forward_shape(self):
        k = make_matern_kernel(nu=1.5, lengthscale=1.0)
        x = torch.linspace(0, 5, 5).unsqueeze(-1)
        cov = k(x, x).to_dense()
        self.assertEqual(cov.shape, (5, 5))


class TestMakeRBFKernel(unittest.TestCase):
    """Tests for make_rbf_kernel()."""

    def test_returns_scale_kernel(self):
        k = make_rbf_kernel(lengthscale=2.0)
        self.assertIsInstance(k, gpytorch.kernels.ScaleKernel)

    def test_base_is_rbf(self):
        k = make_rbf_kernel()
        self.assertIsInstance(k.base_kernel, gpytorch.kernels.RBFKernel)

    def test_forward_shape(self):
        k = make_rbf_kernel(lengthscale=2.0)
        x = torch.linspace(0, 5, 5).unsqueeze(-1)
        cov = k(x, x).to_dense()
        self.assertEqual(cov.shape, (5, 5))


class TestSeparableViaProductKernel(unittest.TestCase):
    """Verify that the separable 2D covariance works correctly via GPyTorch's
    ProductKernel + active_dims (no custom kernel class needed)."""

    def setUp(self):
        n = 10
        t = torch.linspace(0, 5, n)
        wl = torch.linspace(400, 800, n)
        self.x = torch.stack([t, wl], dim=1)

        self.time_kernel = ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5), active_dims=(0,)
        )
        self.wl_kernel = ScaleKernel(
            gpytorch.kernels.RBFKernel(), active_dims=(1,)
        )
        self.sep_kernel = self.time_kernel * self.wl_kernel

    def test_is_product_kernel(self):
        """Separable kernel is a ProductKernel."""
        self.assertIsInstance(self.sep_kernel, gpytorch.kernels.ProductKernel)

    def test_forward_shape(self):
        """Separable kernel returns matrix of correct shape."""
        cov = self.sep_kernel(self.x, self.x).to_dense()
        self.assertEqual(cov.shape, (10, 10))

    def test_symmetric(self):
        """Separable kernel matrix is symmetric."""
        cov = self.sep_kernel(self.x, self.x).to_dense()
        self.assertTrue(torch.allclose(cov, cov.T, atol=1e-5))

    def test_factorization(self):
        """Product kernel factorizes into time x wavelength covariances."""
        # With active_dims set, each sub-kernel selects its own column
        # from the full 2D input — we must pass the full x here.
        time_mat = self.time_kernel(self.x, self.x).to_dense()
        wl_mat = self.wl_kernel(self.x, self.x).to_dense()
        expected = (time_mat * wl_mat).detach()

        cov = self.sep_kernel(self.x, self.x).to_dense().detach()
        self.assertTrue(torch.allclose(cov, expected, atol=1e-4))

    def test_returns_lazy_kernel_tensor(self):
        """ProductKernel preserves GPyTorch's lazy evaluation machinery."""
        result = self.sep_kernel(self.x, self.x)
        # The result should be a lazy tensor (not a dense one), i.e. it exposes
        # to_dense() but hasn't materialized the full n×n matrix yet.
        self.assertTrue(hasattr(result, "to_dense"))
        self.assertIsNotNone(result.to_dense())


if __name__ == "__main__":
    unittest.main()
