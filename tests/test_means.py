"""Tests for the PowerLawMean and DustMean mean functions."""
import unittest

import torch



class TestPowerLawMean(unittest.TestCase):
    """Tests for the PowerLawMean mean function."""

    def setUp(self):
        from pgmuvi.gps import PowerLawMean
        self.PowerLawMean = PowerLawMean
        # 2D input: (time, wavelength), wavelength in second column
        self.x = torch.tensor(
            [[0.0, 0.5], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32
        )

    def test_instantiation(self):
        """PowerLawMean can be instantiated without arguments."""
        mean = self.PowerLawMean()
        self.assertIsNotNone(mean)

    def test_forward_shape(self):
        """PowerLawMean forward returns a 1-D tensor of correct length."""
        mean = self.PowerLawMean()
        out = mean(self.x)
        self.assertEqual(out.shape, (3,))

    def test_parameters_registered(self):
        """PowerLawMean registers offset, weight, and exponent parameters."""
        mean = self.PowerLawMean()
        param_names = [n for n, _ in mean.named_parameters()]
        self.assertIn("offset", param_names)
        self.assertIn("weight", param_names)
        self.assertIn("exponent", param_names)

    def test_default_exponent(self):
        """Default exponent is -2.0 (steep optical-to-IR decline)."""
        mean = self.PowerLawMean()
        self.assertAlmostEqual(mean.exponent.item(), -2.0, places=5)

    def test_power_law_values(self):
        """Forward output matches expected power-law calculation."""
        mean = self.PowerLawMean()
        # Set known parameters: offset=0, weight=1, exponent=-2
        with torch.no_grad():
            mean.offset.fill_(0.0)
            mean.weight.fill_(1.0)
            mean.exponent.fill_(-2.0)
        out = mean(self.x)
        wavelengths = self.x[:, 1]
        expected = wavelengths.pow(-2.0)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))


class TestDustMean(unittest.TestCase):
    """Tests for the DustMean mean function."""

    def setUp(self):
        from pgmuvi.gps import DustMean
        self.DustMean = DustMean
        self.x = torch.tensor(
            [[0.0, 0.5], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32
        )

    def test_instantiation(self):
        """DustMean can be instantiated without arguments."""
        mean = self.DustMean()
        self.assertIsNotNone(mean)

    def test_forward_shape(self):
        """DustMean forward returns a 1-D tensor of correct length."""
        mean = self.DustMean()
        out = mean(self.x)
        self.assertEqual(out.shape, (3,))

    def test_parameters_registered(self):
        """DustMean registers the required parameters."""
        mean = self.DustMean()
        param_names = [n for n, _ in mean.named_parameters()]
        self.assertIn("offset", param_names)
        self.assertIn("log_amplitude", param_names)
        self.assertIn("log_tau", param_names)
        self.assertIn("log_alpha", param_names)

    def test_extinction_increases_at_short_wavelength(self):
        """Extinction is greater at shorter wavelengths (dust behaviour)."""
        mean = self.DustMean()
        # Set tau > 0, alpha > 0; offset = 0
        with torch.no_grad():
            mean.offset.fill_(0.0)
            mean.log_amplitude.fill_(0.0)   # amplitude = 1
            mean.log_tau.fill_(0.0)         # tau = 1
            mean.log_alpha.fill_(0.0)       # alpha = 1
        x_optical = torch.tensor([[0.0, 0.5]], dtype=torch.float32)
        x_ir = torch.tensor([[0.0, 2.0]], dtype=torch.float32)
        out_optical = mean(x_optical)
        out_ir = mean(x_ir)
        # Optical (shorter wavelength) should have lower flux
        self.assertLess(out_optical.item(), out_ir.item())

    def test_zero_tau_gives_constant_mean(self):
        """With tau -> 0, DustMean approaches amplitude + offset."""
        mean = self.DustMean()
        with torch.no_grad():
            mean.offset.fill_(0.5)
            mean.log_amplitude.fill_(0.0)   # amplitude = 1
            mean.log_tau.fill_(-30.0)       # tau ≈ 0
            mean.log_alpha.fill_(0.0)
        out = mean(self.x)
        # Should be approximately offset + amplitude = 1.5 for all points
        self.assertTrue(torch.allclose(out, torch.full((3,), 1.5), atol=1e-3))
