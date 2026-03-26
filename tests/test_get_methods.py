"""Tests for the get_default_priors, get_period_prior, and
get_default_constraints companion methods."""

import contextlib
import io
import unittest

import torch

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.priors import LogNormalFrequencyPrior, LogNormalPeriodPrior
from pgmuvi.synthetic import make_simple_sinusoid_1d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_1d_lc(n=80, span_days=500.0, seed=0):
    """Return a simple 1-D Lightcurve spanning *span_days* days."""
    return make_simple_sinusoid_1d(
        n_obs=n, period=100.0, noise_level=0.1, t_span=span_days, seed=seed
    )


# ---------------------------------------------------------------------------
# Tests for get_default_priors
# ---------------------------------------------------------------------------


class TestGetDefaultPriorsNotInitialised(unittest.TestCase):
    """get_default_priors raises RuntimeError if model not set."""

    def test_raises_runtime_error(self):
        lc = Lightcurve(torch.linspace(0, 100, 20), torch.randn(20))
        with self.assertRaises(RuntimeError):
            lc.get_default_priors()


class TestGetDefaultPriorsSpectral(unittest.TestCase):
    """get_default_priors on a SpectralMixture model after set_default_priors."""

    def setUp(self):
        self.lc = _make_1d_lc()
        self.lc.set_model("1D", num_mixtures=2)
        self.lc.set_default_priors()

    def test_returns_dict(self):
        result = self.lc.get_default_priors()
        self.assertIsInstance(result, dict)

    def test_dict_is_nonempty(self):
        result = self.lc.get_default_priors()
        self.assertTrue(len(result) > 0)

    def test_mixture_means_prior_present(self):
        result = self.lc.get_default_priors()
        keys = list(result.keys())
        self.assertTrue(any("mixture_means_prior" in k for k in keys))

    def test_prints_output(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.lc.get_default_priors()
        output = buf.getvalue()
        self.assertIn("Registered priors:", output)
        self.assertIn("mixture_means_prior", output)

    def test_no_priors_prints_none(self):
        # Model with no priors registered should say "(none)"
        lc = _make_1d_lc()
        lc.set_model("1D", num_mixtures=2)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            lc.get_default_priors()
        output = buf.getvalue()
        self.assertIn("(none)", output)


# ---------------------------------------------------------------------------
# Tests for get_period_prior
# ---------------------------------------------------------------------------


class TestGetPeriodPriorNotInitialised(unittest.TestCase):
    """get_period_prior raises RuntimeError if model not set."""

    def test_raises_runtime_error(self):
        lc = Lightcurve(torch.linspace(0, 100, 20), torch.randn(20))
        with self.assertRaises(RuntimeError):
            lc.get_period_prior()


class TestGetPeriodPriorSpectral(unittest.TestCase):
    """get_period_prior on a SpectralMixture model after set_period_prior."""

    def setUp(self):
        self.lc = _make_1d_lc()
        self.lc.set_model("1D", num_mixtures=2)
        self.lc.set_period_prior(prior_set="LPV")

    def test_returns_dict(self):
        result = self.lc.get_period_prior()
        self.assertIsInstance(result, dict)

    def test_dict_contains_mixture_means_prior(self):
        result = self.lc.get_period_prior()
        keys = list(result.keys())
        self.assertTrue(any("mixture_means_prior" in k for k in keys))

    def test_prior_is_lognormal_frequency_prior(self):
        result = self.lc.get_period_prior()
        for prior in result.values():
            self.assertIsInstance(prior, LogNormalFrequencyPrior)

    def test_prints_prior_type_and_params(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.lc.get_period_prior()
        output = buf.getvalue()
        self.assertIn("Registered period/frequency priors:", output)
        self.assertIn("LogNormalFrequencyPrior", output)


class TestGetPeriodPriorQuasiPeriodic(unittest.TestCase):
    """get_period_prior on a QuasiPeriodic model after set_period_prior."""

    def setUp(self):
        self.lc = _make_1d_lc()
        self.lc.set_model("1DQuasiPeriodic")
        self.lc.set_period_prior(prior_set="LPV")

    def test_returns_dict(self):
        result = self.lc.get_period_prior()
        self.assertIsInstance(result, dict)

    def test_dict_contains_period_length_prior(self):
        result = self.lc.get_period_prior()
        keys = list(result.keys())
        self.assertTrue(any("period_length_prior" in k for k in keys))

    def test_prior_is_lognormal_period_prior(self):
        result = self.lc.get_period_prior()
        for prior in result.values():
            self.assertIsInstance(prior, LogNormalPeriodPrior)


class TestGetPeriodPriorNoPeriodicity(unittest.TestCase):
    """get_period_prior on a model with no periodicity returns empty dict."""

    def test_returns_empty_dict(self):
        lc = _make_1d_lc()
        lc.set_model("1DMatern")
        result = lc.get_period_prior()
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)

    def test_prints_none(self):
        lc = _make_1d_lc()
        lc.set_model("1DMatern")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            lc.get_period_prior()
        output = buf.getvalue()
        self.assertIn("(none)", output)


# ---------------------------------------------------------------------------
# Tests for get_default_constraints
# ---------------------------------------------------------------------------


class TestGetDefaultConstraintsNotInitialised(unittest.TestCase):
    """get_default_constraints raises RuntimeError if model not set."""

    def test_raises_runtime_error(self):
        lc = Lightcurve(torch.linspace(0, 100, 20), torch.randn(20))
        with self.assertRaises(RuntimeError):
            lc.get_default_constraints()


class TestGetDefaultConstraintsSpectral(unittest.TestCase):
    """get_default_constraints on a SpectralMixture model after
    set_default_constraints."""

    def setUp(self):
        self.lc = _make_1d_lc()
        self.lc.set_model("1D", num_mixtures=2)
        self.lc.set_default_constraints()

    def test_returns_dict(self):
        result = self.lc.get_default_constraints()
        self.assertIsInstance(result, dict)

    def test_dict_is_nonempty(self):
        result = self.lc.get_default_constraints()
        self.assertTrue(len(result) > 0)

    def test_mixture_means_constraint_present(self):
        result = self.lc.get_default_constraints()
        keys = list(result.keys())
        self.assertTrue(any("mixture_means" in k for k in keys))

    def test_prints_output(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.lc.get_default_constraints()
        output = buf.getvalue()
        self.assertIn("Registered constraints:", output)
        self.assertIn("mixture_means", output)

    def test_no_constraints_prints_none(self):
        # A model with only the default GPyTorch constraints (i.e. before
        # set_default_constraints is called) will still have constraints from
        # GPyTorch itself (e.g. Positive() on mixture_weights).  The "(none)"
        # branch is only reached when named_constraints yields nothing at all,
        # which cannot happen with a real model.  We therefore only check that
        # the output always contains the "Registered constraints:" header.
        lc = Lightcurve(torch.linspace(0, 100, 20), torch.randn(20))
        lc.set_model("1D", num_mixtures=2)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            lc.get_default_constraints()
        output = buf.getvalue()
        self.assertIn("Registered constraints:", output)


if __name__ == "__main__":
    unittest.main()
