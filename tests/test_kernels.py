"""Tests for custom kernel implementations in pgmuvi/kernels.py."""

import math
import unittest
import torch
import gpytorch

from pgmuvi.kernels import (
    QuasiPeriodicKernel,
    SeparableKernel,
    AchromaticKernel,
    make_quasi_periodic_kernel,
    make_matern_kernel,
    make_rbf_kernel,
)


class TestQuasiPeriodicKernel(unittest.TestCase):
    """Tests for QuasiPeriodicKernel."""

    def setUp(self):
        self.kernel = QuasiPeriodicKernel()
        self.kernel.period = 5.0
        self.kernel.lengthscale = 0.5
        self.kernel.decay = 25.0
        self.x = torch.linspace(0, 20, 10).unsqueeze(-1)

    def test_forward_shape(self):
        """Kernel forward pass returns matrix of correct shape."""
        cov = self.kernel(self.x, self.x).to_dense()
        self.assertEqual(cov.shape, (10, 10))

    def test_symmetric(self):
        """Kernel matrix is symmetric."""
        cov = self.kernel(self.x, self.x).to_dense()
        self.assertTrue(torch.allclose(cov, cov.T, atol=1e-5))

    def test_diagonal_is_one(self):
        """Diagonal of kernel matrix equals 1 (no amplitude scaling)."""
        cov = self.kernel(self.x, self.x).to_dense()
        diag = cov.diagonal()
        self.assertTrue(
            torch.allclose(diag, torch.ones_like(diag), atol=1e-5),
            f"Diagonal values: {diag}",
        )

    def test_values_in_range(self):
        """Kernel values are between 0 and 1."""
        cov = self.kernel(self.x, self.x).to_dense()
        self.assertTrue((cov >= -1e-5).all())
        self.assertTrue((cov <= 1.0 + 1e-5).all())

    def test_period_setter(self):
        """Period parameter can be set and retrieved."""
        self.kernel.period = 10.0
        self.assertAlmostEqual(float(self.kernel.period), 10.0, places=4)

    def test_lengthscale_setter(self):
        """Lengthscale parameter can be set and retrieved."""
        self.kernel.lengthscale = 1.0
        self.assertAlmostEqual(float(self.kernel.lengthscale), 1.0, places=4)

    def test_decay_setter(self):
        """Decay parameter can be set and retrieved."""
        self.kernel.decay = 50.0
        self.assertAlmostEqual(float(self.kernel.decay), 50.0, places=4)

    def test_constraints_positive(self):
        """Period, lengthscale, and decay are always positive."""
        self.assertGreater(float(self.kernel.period), 0)
        self.assertGreater(float(self.kernel.lengthscale), 0)
        self.assertGreater(float(self.kernel.decay), 0)

    def test_periodic_at_integer_multiples(self):
        """Kernel value is high at integer multiples of the period."""
        period = float(self.kernel.period)
        x1 = torch.tensor([[0.0]])
        x2 = torch.tensor([[period]])
        cov = self.kernel(x1, x2).to_dense()
        # k(0, period) should be close to 1 (same phase)
        self.assertGreater(float(cov[0, 0]), 0.9)

    def test_diag_mode(self):
        """Diag mode returns diagonal of full matrix."""
        cov_full = self.kernel(self.x, self.x).to_dense()
        cov_diag = self.kernel(self.x, self.x, diag=True)
        expected_diag = cov_full.diagonal()
        self.assertTrue(torch.allclose(cov_diag, expected_diag, atol=1e-5))

    def test_scaled_kernel_helper(self):
        """make_quasi_periodic_kernel returns a ScaleKernel."""
        k = make_quasi_periodic_kernel(period=5.0)
        self.assertIsInstance(k, gpytorch.kernels.ScaleKernel)
        x = torch.linspace(0, 10, 5).unsqueeze(-1)
        cov = k(x, x).to_dense()
        self.assertEqual(cov.shape, (5, 5))


class TestSeparableKernel(unittest.TestCase):
    """Tests for SeparableKernel."""

    def setUp(self):
        self.time_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
        self.wl_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.kernel = SeparableKernel(self.time_kernel, self.wl_kernel)

        # Create 2D test data: (n, 2) with [time, wavelength]
        n = 10
        t = torch.linspace(0, 5, n)
        wl = torch.linspace(400, 800, n)
        self.x = torch.stack([t, wl], dim=1)

    def test_forward_shape(self):
        """Separable kernel returns matrix of correct shape."""
        cov = self.kernel(self.x, self.x).to_dense()
        self.assertEqual(cov.shape, (10, 10))

    def test_symmetric(self):
        """Separable kernel matrix is symmetric."""
        cov = self.kernel(self.x, self.x).to_dense()
        self.assertTrue(torch.allclose(cov, cov.T, atol=1e-5))

    def test_factorization(self):
        """Separable kernel factorizes correctly."""
        t1 = self.x[:, 0:1]
        t2 = self.x[:, 0:1]
        w1 = self.x[:, 1:2]
        w2 = self.x[:, 1:2]

        time_mat = self.time_kernel(t1, t2).to_dense()
        wl_mat = self.wl_kernel(w1, w2).to_dense()
        expected = time_mat * wl_mat

        cov = self.kernel(self.x, self.x).to_dense()
        self.assertTrue(torch.allclose(cov, expected, atol=1e-4))

    def test_diag_mode(self):
        """Diag mode returns diagonal of full matrix."""
        cov_full = self.kernel(self.x, self.x).to_dense()
        cov_diag = self.kernel(self.x, self.x, diag=True)
        expected_diag = cov_full.diagonal()
        self.assertTrue(torch.allclose(cov_diag, expected_diag, atol=1e-5))


class TestAchromaticKernel(unittest.TestCase):
    """Tests for AchromaticKernel."""

    def setUp(self):
        self.time_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
        self.kernel = AchromaticKernel(self.time_kernel)

        n = 8
        t = torch.linspace(0, 5, n)
        wl = torch.cat([torch.ones(n // 2) * 500.0, torch.ones(n - n // 2) * 700.0])
        self.x = torch.stack([t, wl], dim=1)

    def test_forward_shape(self):
        """Achromatic kernel returns matrix of correct shape."""
        cov = self.kernel(self.x, self.x).to_dense()
        self.assertEqual(cov.shape, (8, 8))

    def test_wavelength_kernel_is_constant(self):
        """Wavelength kernel of AchromaticKernel is a ConstantKernel."""
        self.assertIsInstance(self.kernel.wavelength_kernel, gpytorch.kernels.ConstantKernel)


class TestKernelHelpers(unittest.TestCase):
    """Tests for kernel helper factory functions."""

    def test_make_matern_kernel(self):
        k = make_matern_kernel(nu=1.5, lengthscale=1.0)
        self.assertIsInstance(k, gpytorch.kernels.ScaleKernel)
        x = torch.linspace(0, 5, 5).unsqueeze(-1)
        cov = k(x, x).to_dense()
        self.assertEqual(cov.shape, (5, 5))

    def test_make_rbf_kernel(self):
        k = make_rbf_kernel(lengthscale=2.0)
        self.assertIsInstance(k, gpytorch.kernels.ScaleKernel)
        x = torch.linspace(0, 5, 5).unsqueeze(-1)
        cov = k(x, x).to_dense()
        self.assertEqual(cov.shape, (5, 5))


if __name__ == "__main__":
    unittest.main()
