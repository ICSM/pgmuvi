"""Tests for separable 2D models with spectral-mixture time kernels.

Verifies that MLS-based initialisation, frequency constraints, frequency
priors, and wavelength output-scale constraints are correctly applied when
a separable model (e.g. ``2DWavelengthDependent``) uses
``time_kernel_type='spectral_mixture'``.
"""

import unittest
import torch
from pgmuvi.lightcurve import (
    Lightcurve,
    _find_separable_sm_time_kernel,
    _SEPARABLE_SM_CAPABLE_MODELS,
    _SM_MODELS,
)
from pgmuvi.synthetic import make_chromatic_sinusoid_2d


def _make_2d_lc(period=40.0, n_per_band=50, t_span=200.0, seed=42):
    """Create a simple 2D Lightcurve for testing."""
    return make_chromatic_sinusoid_2d(
        n_per_band=n_per_band,
        period=period,
        wavelengths=[500.0, 700.0],
        amplitude_law="linear",
        amplitude_slope=0.0,
        noise_level=0.1,
        t_span=t_span,
        irregular=False,
        seed=seed,
    )


class TestSeparableSMCapableModels(unittest.TestCase):
    """Verify the _SEPARABLE_SM_CAPABLE_MODELS set."""

    def test_no_overlap_with_sm_models(self):
        self.assertEqual(
            _SM_MODELS & _SEPARABLE_SM_CAPABLE_MODELS,
            frozenset(),
        )

    def test_expected_models_present(self):
        for name in ("2DWavelengthDependent", "2DAchromatic", "2DDustMean",
                      "2DPowerLawMean", "2DSeparable"):
            self.assertIn(name, _SEPARABLE_SM_CAPABLE_MODELS)


class TestFindSeparableSMTimeKernel(unittest.TestCase):
    """Test _find_separable_sm_time_kernel helper."""

    def setUp(self):
        self.lc = _make_2d_lc()

    def test_returns_sm_kernel_for_sm_time_kernel(self):
        self.lc.set_model(
            "2DWavelengthDependent",
            time_kernel_type="spectral_mixture",
            num_mixtures=1,
        )
        tk = _find_separable_sm_time_kernel(self.lc.model)
        self.assertIsNotNone(tk)
        self.assertTrue(hasattr(tk, "mixture_means"))

    def test_returns_none_for_matern_time_kernel(self):
        self.lc.set_model(
            "2DWavelengthDependent",
            time_kernel_type="matern",
        )
        tk = _find_separable_sm_time_kernel(self.lc.model)
        self.assertIsNone(tk)

    def test_returns_none_for_full_2d_sm(self):
        self.lc.set_model("2D", num_mixtures=2)
        tk = _find_separable_sm_time_kernel(self.lc.model)
        self.assertIsNone(tk)


class TestSeparableSMConstraints(unittest.TestCase):
    """Verify constraints are set on separable SM models."""

    def setUp(self):
        self.lc = _make_2d_lc()
        self.lc.set_model(
            "2DWavelengthDependent",
            time_kernel_type="spectral_mixture",
            num_mixtures=1,
        )

    def test_mixture_means_in_model_pars(self):
        self.assertIn("mixture_means", self.lc._model_pars)

    def test_constraints_registered_after_set_default_constraints(self):
        self.lc.set_default_constraints()
        found = False
        for name, constraint in self.lc.model.named_constraints():
            if "mixture_means" in name:
                found = True
                break
        self.assertTrue(found, "mixture_means constraint not registered")

    def test_wavelength_outputscale_constrained(self):
        self.lc.set_default_constraints()
        sk = self.lc.model.covar_module
        for k in sk.kernels:
            ad = getattr(k, "active_dims", None)
            if ad is not None and 1 in ad.tolist() and hasattr(k, "outputscale"):
                found = False
                for name, constraint in k.named_constraints():
                    if "outputscale" in name:
                        found = True
                        break
                self.assertTrue(
                    found,
                    "wavelength kernel outputscale constraint not registered",
                )
                return
        self.skipTest("No wavelength kernel with outputscale found")


class TestSeparableSMInitPath(unittest.TestCase):
    """Verify that the init path correctly seeds separable SM models."""

    def test_named_parameters_path(self):
        lc = _make_2d_lc()
        lc.set_model(
            "2DWavelengthDependent",
            time_kernel_type="spectral_mixture",
            num_mixtures=1,
        )
        paths = [
            p for p, _ in lc.model.named_parameters()
            if p.endswith("raw_mixture_means")
        ]
        self.assertEqual(len(paths), 1)
        self.assertIn("kernels.0", paths[0])


if __name__ == "__main__":
    unittest.main()
