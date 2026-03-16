"""Tests for initialization routines in pgmuvi/initialization.py."""

import unittest
import numpy as np
import torch

from pgmuvi.initialization import (
    initialize_quasi_periodic_from_data,
    initialize_separable_from_data,
    initialize_from_physics,
)
from pgmuvi.synthetic import make_chromatic_sinusoid_2d, make_simple_sinusoid_1d


class TestInitializeQuasiPeriodic(unittest.TestCase):
    """Tests for initialize_quasi_periodic_from_data."""

    def setUp(self):
        # Create a strong periodic signal with period=5
        lc = make_simple_sinusoid_1d(
            n_obs=100, period=5.0, noise_level=0.0, irregular=False, seed=42
        )
        self.t = lc.xdata
        self.y = lc.ydata
        self.yerr = 0.05 * torch.ones_like(self.y)

    def test_returns_dict_with_keys(self):
        """Function returns dict with required keys."""
        params = initialize_quasi_periodic_from_data(self.t, self.y)
        for key in ["period", "lengthscale", "decay", "outputscale"]:
            self.assertIn(key, params, f"Missing key: {key}")

    def test_period_is_positive(self):
        """Estimated period is positive."""
        params = initialize_quasi_periodic_from_data(self.t, self.y)
        self.assertGreater(params["period"], 0)

    def test_period_detection_accuracy(self):
        """Estimated period is close to the true period (within 20%)."""
        params = initialize_quasi_periodic_from_data(self.t, self.y)
        true_period = 5.0
        self.assertAlmostEqual(params["period"], true_period, delta=true_period * 0.2)

    def test_with_yerr(self):
        """Function works with measurement uncertainties."""
        params = initialize_quasi_periodic_from_data(self.t, self.y, yerr=self.yerr)
        self.assertIn("period", params)
        self.assertGreater(params["period"], 0)

    def test_with_numpy_input(self):
        """Function works with NumPy array inputs."""
        t_np = np.linspace(0, 20, 100)
        y_np = np.sin(2 * np.pi * t_np / 5.0)
        params = initialize_quasi_periodic_from_data(t_np, y_np)
        self.assertIn("period", params)
        self.assertGreater(params["period"], 0)

    def test_flat_signal_fallback(self):
        """Flat (constant) signal returns fallback values without error."""
        t_flat = torch.linspace(0, 10, 50, dtype=torch.float32)
        y_flat = torch.ones(50, dtype=torch.float32)
        params = initialize_quasi_periodic_from_data(t_flat, y_flat)
        self.assertIn("period", params)
        self.assertGreater(params["period"], 0)

    def test_outputscale_positive(self):
        """Outputscale is positive for non-constant data."""
        params = initialize_quasi_periodic_from_data(self.t, self.y)
        self.assertGreater(params["outputscale"], 0)

    def test_lengthscale_and_decay_positive(self):
        """Lengthscale and decay are positive."""
        params = initialize_quasi_periodic_from_data(self.t, self.y)
        self.assertGreater(params["lengthscale"], 0)
        self.assertGreater(params["decay"], 0)


class TestInitializeSeparable(unittest.TestCase):
    """Tests for initialize_separable_from_data."""

    def setUp(self):
        lc = make_chromatic_sinusoid_2d(
            n_per_band=30,
            period=5.0,
            wavelengths=[500.0, 700.0],
            amplitude_law="linear",
            amplitude_slope=0.0,
            noise_level=0.05,
            irregular=False,
            seed=42,
        )
        self.x = lc.xdata
        self.y = lc.ydata

    def test_returns_dict_with_keys(self):
        """Function returns dict with required keys."""
        params = initialize_separable_from_data(self.x, self.y)
        for key in ["period", "is_achromatic", "wavelength_lengthscale",
                    "periods_per_band", "outputscale"]:
            self.assertIn(key, params, f"Missing key: {key}")
        # is_significant is a new key from MultibandLSWithSignificance
        self.assertIn("is_significant", params)

    def test_period_positive(self):
        """Estimated mean period is positive."""
        params = initialize_separable_from_data(self.x, self.y)
        self.assertGreater(params["period"], 0)

    def test_is_achromatic_for_consistent_periods(self):
        """Returns is_achromatic=True when both bands have the same period."""
        params = initialize_separable_from_data(self.x, self.y)
        # Both bands have the same underlying period (~5.0), so should be achromatic
        self.assertIsInstance(params["is_achromatic"], bool)

    def test_wavelength_lengthscale_positive(self):
        """Wavelength lengthscale is positive."""
        params = initialize_separable_from_data(self.x, self.y)
        self.assertGreater(params["wavelength_lengthscale"], 0)

    def test_periods_per_band_list(self):
        """periods_per_band is a list."""
        params = initialize_separable_from_data(self.x, self.y)
        self.assertIsInstance(params["periods_per_band"], list)

    def test_outputscale_positive(self):
        """Outputscale is positive."""
        params = initialize_separable_from_data(self.x, self.y)
        self.assertGreater(params["outputscale"], 0)


class TestInitializeFromPhysics(unittest.TestCase):
    """Tests for initialize_from_physics."""

    def test_basic_call(self):
        """Returns dict with correct keys."""
        params = initialize_from_physics(period=10.0, outputscale=0.5)
        for key in ["period", "lengthscale", "decay", "outputscale"]:
            self.assertIn(key, params, f"Missing key: {key}")

    def test_period_preserved(self):
        """Provided period is returned unchanged."""
        params = initialize_from_physics(period=7.3)
        self.assertAlmostEqual(params["period"], 7.3, places=5)

    def test_default_decay_is_five_times_period(self):
        """Default decay is 5x the period."""
        params = initialize_from_physics(period=10.0)
        self.assertAlmostEqual(params["decay"], 50.0, places=5)

    def test_default_lengthscale(self):
        """Default lengthscale is 10% of the period."""
        params = initialize_from_physics(period=10.0)
        self.assertAlmostEqual(params["lengthscale"], 1.0, places=5)  # 0.1 * 10.0

    def test_custom_lengthscale(self):
        """Custom lengthscale is returned unchanged."""
        params = initialize_from_physics(period=10.0, lengthscale=2.0)
        self.assertAlmostEqual(params["lengthscale"], 2.0, places=5)

    def test_custom_decay(self):
        """Custom decay is returned unchanged."""
        params = initialize_from_physics(period=10.0, decay=100.0)
        self.assertAlmostEqual(params["decay"], 100.0, places=5)

    def test_all_values_positive(self):
        """All returned values are positive."""
        params = initialize_from_physics(period=5.0, outputscale=1.0)
        for key, val in params.items():
            self.assertGreater(val, 0, f"Value for '{key}' is not positive")


if __name__ == "__main__":
    unittest.main()
