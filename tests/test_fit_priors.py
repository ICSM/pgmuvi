"""Tests for the prior-setting logic in Lightcurve.fit()."""

import math
import unittest
import warnings
from unittest.mock import patch

import gpytorch
import torch

from pgmuvi.synthetic import make_simple_sinusoid_1d

# Sentinel return value for the patched `train` function.
_DUMMY_RESULTS = {"loss": [1.0], "delta_loss": [0.0]}


def _fit_without_training(lc, **kwargs):
    """Call lc.fit() with the training loop patched out."""
    with patch("pgmuvi.lightcurve.train", return_value=_DUMMY_RESULTS):
        with patch.object(lc, "_train"):
            return lc.fit(**kwargs)


def _make_lc():
    return make_simple_sinusoid_1d(
        n_obs=80, period=150.0, noise_level=0.1, t_span=800.0, seed=7
    )


class TestFitPriorSetValidation(unittest.TestCase):
    """Test that invalid prior_set values are caught early in fit()."""

    def test_prior_set_wrong_type_raises_type_error(self):
        """Non-string prior_set raises TypeError."""
        lc = _make_lc()
        with self.assertRaises(TypeError):
            _fit_without_training(lc, model="1D", prior_set=42)

    def test_prior_set_unknown_string_raises_value_error(self):
        """Unrecognised prior_set name raises ValueError."""
        lc = _make_lc()
        with self.assertRaises(ValueError):
            _fit_without_training(lc, model="1D", prior_set="NOT_A_REAL_SET")

    def test_prior_set_none_accepted(self):
        """prior_set=None (default) is accepted without error."""
        lc = _make_lc()
        # Should not raise.
        _fit_without_training(lc, model="1D", prior_set=None, training_iter=1)

    def test_prior_set_lpv_accepted(self):
        """prior_set='LPV' (valid named set) is accepted without error."""
        lc = _make_lc()
        _fit_without_training(lc, model="1D", prior_set="LPV", training_iter=1)


class TestFitMlsDataDrivenPrior(unittest.TestCase):
    """Test that MLS-based initialisation produces a data-driven prior."""

    def setUp(self):
        self.lc = _make_lc()
        _fit_without_training(self.lc, model="1D", prior_set=None, training_iter=1)

    def _get_mm_prior(self):
        for name, _mod, prior, _cl, _scl in self.lc.model.named_priors():
            if "mixture_means_prior" in name:
                return prior
        return None

    def test_mixture_means_prior_is_set(self):
        """After MLS init, a mixture_means_prior should be registered."""
        prior = self._get_mm_prior()
        self.assertIsNotNone(prior)

    def test_mixture_means_prior_is_lognormal(self):
        """The data-driven prior should be a LogNormalPrior."""
        prior = self._get_mm_prior()
        self.assertIsInstance(prior, gpytorch.priors.LogNormalPrior)

    def test_mixture_means_prior_broad_sigma(self):
        """The data-driven prior should have sigma=1.5 (broad)."""
        prior = self._get_mm_prior()
        self.assertAlmostEqual(float(prior.scale), 1.5, places=5)

    def test_mixture_means_prior_median_matches_dominant_freq(self):
        """The prior median should equal the dominant MLS frequency."""
        prior = self._get_mm_prior()
        # median of LogNormal(mu, sigma) = exp(mu)
        median_freq = math.exp(float(prior.loc))
        # Get the actual init frequencies used
        init_freqs, _ = self.lc.fit_LS(num_peaks=10)
        dominant_freq = float(init_freqs[0])
        self.assertAlmostEqual(median_freq, dominant_freq, places=5)

    def test_priors_set_flag_is_true(self):
        """__PRIORS_SET flag should be True after fit()."""
        # Access the mangled name
        flag_attr = "_Lightcurve__PRIORS_SET"
        self.assertTrue(getattr(self.lc, flag_attr))


class TestFitPriorSetOverrideWarning(unittest.TestCase):
    """Test that specifying prior_set alongside MLS init issues a warning."""

    def test_prior_set_with_mls_warns(self):
        """A UserWarning should be issued when prior_set overrides MLS prior."""
        lc = _make_lc()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _fit_without_training(lc, model="1D", prior_set="LPV", training_iter=1)
        override_warns = [
            x for x in w
            if issubclass(x.category, UserWarning)
            and "overrides" in str(x.message).lower()
        ]
        self.assertTrue(
            len(override_warns) >= 1,
            "Expected at least one UserWarning about prior_set overriding MLS prior",
        )


class TestFitNoMlsFallsBackToLpv(unittest.TestCase):
    """When no MLS and no prior_set, default is LPV prior."""

    def test_lpv_prior_applied_when_no_mls(self):
        """Prior should use LPV set when periods are user-supplied."""
        from pgmuvi.priors import LogNormalFrequencyPrior

        lc = _make_lc()
        # Providing explicit periods disables MLS → should fall back to LPV
        _fit_without_training(
            lc, model="1D", periods=[150.0], prior_set=None, training_iter=1
        )
        # Under LPV, mixture_means_prior should be a LogNormalFrequencyPrior
        mm_prior = None
        for name, _mod, prior, _cl, _scl in lc.model.named_priors():
            if "mixture_means_prior" in name:
                mm_prior = prior
        self.assertIsNotNone(mm_prior)
        self.assertIsInstance(mm_prior, LogNormalFrequencyPrior)

    def test_use_mls_false_falls_back_to_lpv(self):
        """Prior should use LPV set when use_mls_init=False."""
        from pgmuvi.priors import LogNormalFrequencyPrior

        lc = _make_lc()
        _fit_without_training(
            lc,
            model="1D",
            use_mls_init=False,
            num_mixtures=2,
            prior_set=None,
            training_iter=1,
        )
        mm_prior = None
        for name, _mod, prior, _cl, _scl in lc.model.named_priors():
            if "mixture_means_prior" in name:
                mm_prior = prior
        self.assertIsNotNone(mm_prior)
        self.assertIsInstance(mm_prior, LogNormalFrequencyPrior)


class TestFitPreregisteredPriorsNotOverridden(unittest.TestCase):
    """Priors set before fit() should not be overridden by fit()."""

    def test_preregistered_prior_survives_fit(self):
        """set_default_priors() before fit() prevents fit() from replacing it."""
        from pgmuvi.priors import LogNormalFrequencyPrior

        lc = _make_lc()
        lc.set_model("1D", num_mixtures=2)
        lc.set_default_priors(prior_set="LPV")
        # Now we know the prior is LPV (LogNormalFrequencyPrior).
        # Calling fit() should NOT replace it.
        _fit_without_training(lc, model=None, training_iter=1)

        mm_prior = None
        for name, _mod, prior, _cl, _scl in lc.model.named_priors():
            if "mixture_means_prior" in name:
                mm_prior = prior
        self.assertIsNotNone(mm_prior)
        # Should still be the LPV prior, not a plain LogNormalPrior
        self.assertIsInstance(mm_prior, LogNormalFrequencyPrior)


if __name__ == "__main__":
    unittest.main()
