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
    _find_sm_in_kernel,
    _SEPARABLE_SM_CAPABLE_MODELS,
    _SM_MODELS,
)
from pgmuvi.priors import LogNormalFrequencyPrior
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


class TestFindSMInKernel(unittest.TestCase):
    """Test the recursive _find_sm_in_kernel helper."""

    def test_finds_direct_sm(self):
        from gpytorch.kernels import SpectralMixtureKernel
        k = SpectralMixtureKernel(num_mixtures=1, ard_num_dims=1)
        self.assertIs(_find_sm_in_kernel(k), k)

    def test_finds_sm_under_scale_kernel(self):
        from gpytorch.kernels import SpectralMixtureKernel, ScaleKernel
        sm = SpectralMixtureKernel(num_mixtures=1, ard_num_dims=1)
        scaled = ScaleKernel(sm)
        self.assertIs(_find_sm_in_kernel(scaled), sm)

    def test_finds_sm_in_additive_kernel(self):
        """SM nested inside an AdditiveKernel (as created by add_flicker=True)."""
        from gpytorch.kernels import SpectralMixtureKernel, RBFKernel, AdditiveKernel
        sm = SpectralMixtureKernel(num_mixtures=1, ard_num_dims=1)
        rbf = RBFKernel()
        additive = AdditiveKernel(sm, rbf)
        self.assertIs(_find_sm_in_kernel(additive), sm)

    def test_returns_none_for_rbf(self):
        from gpytorch.kernels import RBFKernel
        k = RBFKernel()
        self.assertIsNone(_find_sm_in_kernel(k))


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

    def test_fit_seeds_mixture_means_from_periods(self):
        """fit() with explicit periods= should seed mixture_means to 1/period."""
        period = 40.0
        lc = _make_2d_lc(period=period, n_per_band=20)
        default_mm = 0.6931  # GPyTorch default (log 2)
        lc.fit(
            model="2DWavelengthDependent",
            time_kernel_type="spectral_mixture",
            num_mixtures=1,
            training_iter=1,
            periods=[period],
            use_mls_init=False,
            lr=0.0,  # zero learning rate: no training movement
        )
        tk = _find_separable_sm_time_kernel(lc.model)
        self.assertIsNotNone(tk)
        mm = float(tk.mixture_means.detach().squeeze())
        expected_freq = 1.0 / period
        # Should be seeded near 1/period, well away from the default
        self.assertAlmostEqual(mm, expected_freq, places=4)
        self.assertNotAlmostEqual(mm, default_mm, places=2)

    def test_fit_registers_lognormal_frequency_prior(self):
        """fit() should register a LogNormalFrequencyPrior on the SM time kernel."""
        lc = _make_2d_lc(n_per_band=20)
        lc.fit(
            model="2DWavelengthDependent",
            time_kernel_type="spectral_mixture",
            num_mixtures=1,
            training_iter=1,
            use_mls_init=False,
            lr=0.0,
        )
        tk = _find_separable_sm_time_kernel(lc.model)
        self.assertIsNotNone(tk)
        # named_priors yields (name, module, prior, getter_fn, setter_fn)
        priors = {item[0]: item[2] for item in tk.named_priors()}
        self.assertIn("mixture_means_prior", priors)
        self.assertIsInstance(priors["mixture_means_prior"], LogNormalFrequencyPrior)

    def test_priors_set_flag_prevents_overwrite_on_repeat_call(self):
        """Calling fit() twice should not raise RuntimeError from double prior reg."""
        lc = _make_2d_lc(n_per_band=20)
        kwargs = dict(
            model="2DWavelengthDependent",
            time_kernel_type="spectral_mixture",
            num_mixtures=1,
            training_iter=1,
            use_mls_init=False,
            lr=0.0,
        )
        lc.fit(**kwargs)
        # Second fit with same model string triggers model re-creation;
        # set_default_priors() should not attempt to re-register the
        # mixture_means_prior after fit() already marked __PRIORS_SET.
        try:
            lc.fit(**kwargs)
        except RuntimeError as exc:
            self.fail(f"Second fit() raised RuntimeError: {exc}")


if __name__ == "__main__":
    unittest.main()
