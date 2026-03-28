"""Integration tests for alternative GP models via the Lightcurve interface."""

import unittest
import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.synthetic import make_chromatic_sinusoid_2d, make_simple_sinusoid_1d


def _make_lc_1d(period=5.0, n=60, noise=0.05):
    """Create a 1D Lightcurve with a strong periodic signal."""
    return make_simple_sinusoid_1d(n_obs=n, period=period, noise_level=noise, seed=0)


def _make_lc_2d(period=5.0, n=60):
    """Create a 2D multiband Lightcurve with consistent periods."""
    return make_chromatic_sinusoid_2d(
        n_per_band=n // 2,
        period=period,
        wavelengths=[500.0, 700.0],
        amplitude_law="linear",
        amplitude_slope=0.0,
        noise_level=0.0,
        irregular=False,
        seed=0,
    )


class TestSetModelAlternative(unittest.TestCase):
    """Tests for set_model() with alternative kernel model strings."""

    def test_set_quasi_periodic(self):
        lc = _make_lc_1d()
        lc.set_model("1DQuasiPeriodic")
        from pgmuvi.models import QuasiPeriodicGPModel
        self.assertIsInstance(lc.model, QuasiPeriodicGPModel)

    def test_set_matern(self):
        lc = _make_lc_1d()
        lc.set_model("1DMatern")
        from pgmuvi.models import MaternGPModel
        self.assertIsInstance(lc.model, MaternGPModel)

    def test_set_periodic_stochastic(self):
        lc = _make_lc_1d()
        lc.set_model("1DPeriodicStochastic")
        from pgmuvi.models import PeriodicPlusStochasticGPModel
        self.assertIsInstance(lc.model, PeriodicPlusStochasticGPModel)

    def test_set_linear_quasi_periodic(self):
        lc = _make_lc_1d()
        lc.set_model("1DLinearQuasiPeriodic")
        from pgmuvi.models import LinearMeanQuasiPeriodicGPModel
        self.assertIsInstance(lc.model, LinearMeanQuasiPeriodicGPModel)

    def test_set_separable_2d(self):
        lc = _make_lc_2d()
        lc.set_model("2DSeparable")
        from pgmuvi.models import SeparableGPModel
        self.assertIsInstance(lc.model, SeparableGPModel)

    def test_separable_2d_active_dims(self):
        """After set_model('2DSeparable'), time kernel uses col 0 and wavelength kernel uses col 1."""
        lc = _make_lc_2d()
        lc.set_model("2DSeparable")
        time_kernel = lc.model.covar_module.kernels[0]
        wl_kernel = lc.model.covar_module.kernels[1]
        self.assertEqual(time_kernel.active_dims.tolist(), [0])
        self.assertEqual(wl_kernel.active_dims.tolist(), [1])

    def test_set_achromatic_2d(self):
        lc = _make_lc_2d()
        lc.set_model("2DAchromatic")
        from pgmuvi.models import AchromaticGPModel
        self.assertIsInstance(lc.model, AchromaticGPModel)

    def test_set_wavelength_dependent_2d(self):
        lc = _make_lc_2d()
        lc.set_model("2DWavelengthDependent")
        from pgmuvi.models import WavelengthDependentGPModel
        self.assertIsInstance(lc.model, WavelengthDependentGPModel)

    def test_invalid_model_raises(self):
        lc = _make_lc_1d()
        with self.assertRaises(ValueError):
            lc.set_model("NotAValidModel")

    def test_existing_spectral_still_works(self):
        """Existing '1D' SpectralMixture model still works."""
        lc = _make_lc_1d()
        lc.set_model("1D", num_mixtures=2)
        from pgmuvi.gps import SpectralMixtureGPModel
        self.assertIsInstance(lc.model, SpectralMixtureGPModel)


class TestAutoSelectModel(unittest.TestCase):
    """Tests for Lightcurve.auto_select_model()."""

    def test_1d_strong_signal_gives_quasi_periodic(self):
        """Strong periodic 1D signal should recommend QuasiPeriodic."""
        lc = _make_lc_1d(noise=0.01)
        model_str, diag = lc.auto_select_model(verbose=False)
        self.assertIn(model_str, ["1DQuasiPeriodic", "1DPeriodicStochastic"])
        self.assertIn("model", diag)
        self.assertIn("reason", diag)

    def test_1d_noise_gives_matern(self):
        """Pure noise 1D signal should recommend Matern."""
        torch.manual_seed(42)
        np.random.seed(42)
        t = torch.linspace(0, 20, 80, dtype=torch.float32)
        y = torch.randn(80, dtype=torch.float32) * 0.1
        lc = Lightcurve(t, y)
        model_str, diag = lc.auto_select_model(verbose=False)
        self.assertIn(model_str, ["1DMatern", "1DPeriodicStochastic", "1DQuasiPeriodic"])
        self.assertIsInstance(diag["max_ls_power"], float)

    def test_2d_returns_2d_model(self):
        """2D data should return a 2D model string."""
        lc = _make_lc_2d()
        model_str, diag = lc.auto_select_model(verbose=False)
        self.assertIn(model_str, ["2DAchromatic", "2DWavelengthDependent"])

    def test_auto_select_verbose_false(self):
        """auto_select_model runs without printing when verbose=False."""
        lc = _make_lc_1d()
        # Should not raise
        model_str, diag = lc.auto_select_model(verbose=False)
        self.assertIsInstance(model_str, str)

    def test_auto_select_returns_valid_model_string(self):
        """Returned model string can be passed directly to set_model."""
        lc = _make_lc_1d()
        model_str, _ = lc.auto_select_model(verbose=False)
        # Should not raise
        lc.set_model(model_str)
        self.assertTrue(hasattr(lc, "model"))

    def test_auto_select_2d_valid_string(self):
        """2D auto-selected model string works with set_model."""
        lc = _make_lc_2d()
        model_str, _ = lc.auto_select_model(verbose=False)
        lc.set_model(model_str)
        self.assertTrue(hasattr(lc, "model"))

    def test_diagnostics_keys(self):
        """Diagnostics dict contains expected keys."""
        lc = _make_lc_1d()
        model_str, diag = lc.auto_select_model(verbose=False)
        self.assertIn("model", diag)
        self.assertIn("reason", diag)
        self.assertIn("max_ls_power", diag)

    def test_2d_diagnostics_keys(self):
        """2D diagnostics dict contains expected keys."""
        lc = _make_lc_2d()
        model_str, diag = lc.auto_select_model(verbose=False)
        self.assertIn("model", diag)
        self.assertIn("reason", diag)
        self.assertIn("init_params", diag)


if __name__ == "__main__":
    unittest.main()
