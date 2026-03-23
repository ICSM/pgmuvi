"""Tests for pgmuvi.priors and Lightcurve.set_period_prior."""

import math
import unittest
import warnings

import torch

from pgmuvi.priors import (
    PRIOR_SETS,
    LogNormalFrequencyPrior,
    LogNormalPeriodPrior,
    NormalFrequencyPrior,
    NormalPeriodPrior,
    get_prior_set,
)
from pgmuvi.synthetic import make_simple_sinusoid_1d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_1d_lc(n=80, span_days=500.0, period=150.0, seed=0):
    return make_simple_sinusoid_1d(
        n_obs=n, period=period, noise_level=0.1, t_span=span_days, seed=seed
    )


# ---------------------------------------------------------------------------
# LogNormalPeriodPrior unit tests
# ---------------------------------------------------------------------------


class TestLogNormalPeriodPrior(unittest.TestCase):
    """Tests for LogNormalPeriodPrior."""

    def test_log_prob_finite_within_bounds(self):
        """log_prob is finite for x within [lower, upper]."""
        prior = LogNormalPeriodPrior(mu=5.0, sigma=1.0, lower_bound=100.0, upper_bound=500.0)
        lp = prior.log_prob(torch.tensor([150.0, 300.0]))
        self.assertTrue(torch.all(torch.isfinite(lp)))

    def test_log_prob_neginf_below_lower(self):
        """log_prob is -inf for x below lower_bound."""
        prior = LogNormalPeriodPrior(mu=5.0, sigma=1.0, lower_bound=100.0)
        lp = prior.log_prob(torch.tensor([50.0]))
        self.assertTrue(torch.isinf(lp) and lp < 0)

    def test_log_prob_neginf_above_upper(self):
        """log_prob is -inf for x above upper_bound."""
        prior = LogNormalPeriodPrior(mu=5.0, sigma=1.0, upper_bound=400.0)
        lp = prior.log_prob(torch.tensor([600.0]))
        self.assertTrue(torch.isinf(lp) and lp < 0)

    def test_log_prob_no_bounds(self):
        """Without bounds, log_prob matches plain LogNormal."""
        prior = LogNormalPeriodPrior(mu=5.0, sigma=1.0)
        import gpytorch
        ref = gpytorch.priors.LogNormalPrior(5.0, 1.0)
        x = torch.tensor([100.0, 200.0, 500.0])
        self.assertTrue(torch.allclose(prior.log_prob(x), ref.log_prob(x)))

    def test_default_parameters(self):
        """Default mu=5, sigma=1."""
        prior = LogNormalPeriodPrior()
        self.assertAlmostEqual(prior.loc.item(), 5.0, places=5)
        self.assertAlmostEqual(prior.scale.item(), 1.0, places=5)

    def test_batch_input(self):
        """Works on a batch of values."""
        prior = LogNormalPeriodPrior(mu=5.0, sigma=1.0, lower_bound=50.0)
        x = torch.tensor([60.0, 100.0, 200.0, 40.0])
        lp = prior.log_prob(x)
        self.assertEqual(lp.shape, torch.Size([4]))
        self.assertTrue(torch.isinf(lp[3]) and lp[3] < 0)  # 40 < lower=50


# ---------------------------------------------------------------------------
# NormalPeriodPrior unit tests
# ---------------------------------------------------------------------------


class TestNormalPeriodPrior(unittest.TestCase):
    """Tests for NormalPeriodPrior."""

    def test_log_prob_finite_within_bounds(self):
        prior = NormalPeriodPrior(mean=300.0, std=75.0, lower_bound=100.0)
        lp = prior.log_prob(torch.tensor([200.0, 300.0, 400.0]))
        self.assertTrue(torch.all(torch.isfinite(lp)))

    def test_log_prob_neginf_below_lower(self):
        prior = NormalPeriodPrior(mean=300.0, std=75.0, lower_bound=100.0)
        lp = prior.log_prob(torch.tensor([50.0]))
        self.assertTrue(torch.isinf(lp) and lp < 0)

    def test_default_parameters(self):
        """Default mean=300, std=75."""
        prior = NormalPeriodPrior()
        self.assertAlmostEqual(prior.loc.item(), 300.0, places=4)
        self.assertAlmostEqual(prior.scale.item(), 75.0, places=4)

    def test_mode_at_mean(self):
        """Peak of the Normal is at the mean (no truncation)."""
        prior = NormalPeriodPrior(mean=300.0, std=75.0)
        x = torch.tensor([200.0, 300.0, 400.0])
        lp = prior.log_prob(x)
        self.assertEqual(lp.argmax().item(), 1)


# ---------------------------------------------------------------------------
# LogNormalFrequencyPrior unit tests
# ---------------------------------------------------------------------------


class TestLogNormalFrequencyPrior(unittest.TestCase):
    """Tests for LogNormalFrequencyPrior."""

    def test_log_prob_jacobian_identity(self):
        """f ~ LogNormal(-mu, sigma) is Jacobian-correct for P=1/f ~ LogNormal(mu,sigma)."""
        mu, sigma = 5.0, 1.0
        period = torch.tensor(150.0)
        freq = 1.0 / period

        import gpytorch
        period_prior = gpytorch.priors.LogNormalPrior(mu, sigma)
        freq_prior = LogNormalFrequencyPrior(mu=mu, sigma=sigma)

        # The Jacobian-corrected log_prob in frequency space should equal
        # log p_period(1/f) - 2*log(f) = LogNormal(-mu,sigma).log_prob(f)
        expected = period_prior.log_prob(period) - 2.0 * torch.log(freq)
        got = freq_prior.log_prob(freq)
        self.assertAlmostEqual(expected.item(), got.item(), places=4)

    def test_neginf_for_short_period(self):
        """Frequencies corresponding to period < lower_period get -inf."""
        prior = LogNormalFrequencyPrior(mu=5.0, sigma=1.0, lower_period=100.0)
        f_short = torch.tensor(1.0 / 50.0)  # period=50 < lower=100
        lp = prior.log_prob(f_short)
        self.assertTrue(torch.isinf(lp) and lp < 0)

    def test_finite_for_long_period(self):
        """Frequencies corresponding to period > lower_period are finite."""
        prior = LogNormalFrequencyPrior(mu=5.0, sigma=1.0, lower_period=100.0)
        f_long = torch.tensor(1.0 / 150.0)  # period=150 > lower=100
        lp = prior.log_prob(f_long)
        self.assertTrue(torch.isfinite(lp))

    def test_loc_negated(self):
        """LogNormalFrequencyPrior stores loc=-mu."""
        prior = LogNormalFrequencyPrior(mu=5.0, sigma=1.0)
        self.assertAlmostEqual(prior.loc.item(), -5.0, places=5)

    def test_scale_preserved(self):
        """LogNormalFrequencyPrior preserves sigma as scale."""
        prior = LogNormalFrequencyPrior(mu=5.0, sigma=2.0)
        self.assertAlmostEqual(prior.scale.item(), 2.0, places=5)


# ---------------------------------------------------------------------------
# NormalFrequencyPrior unit tests
# ---------------------------------------------------------------------------


class TestNormalFrequencyPrior(unittest.TestCase):
    """Tests for NormalFrequencyPrior."""

    def test_jacobian_correction(self):
        """log_prob(f) = Normal(mean,std).log_prob(1/f) - 2*log(f)."""
        mean, std = 300.0, 75.0
        f = torch.tensor(1.0 / 300.0)
        period = 1.0 / f

        import gpytorch
        period_ref = gpytorch.priors.NormalPrior(mean, std)
        lp_expected = period_ref.log_prob(period) - 2.0 * torch.log(f)

        freq_prior = NormalFrequencyPrior(mean=mean, std=std)
        lp_got = freq_prior.log_prob(f)
        self.assertAlmostEqual(lp_expected.item(), lp_got.item(), places=4)

    def test_neginf_for_short_period(self):
        prior = NormalFrequencyPrior(mean=300.0, std=75.0, lower_period=100.0)
        f_short = torch.tensor(1.0 / 50.0)
        lp = prior.log_prob(f_short)
        self.assertTrue(torch.isinf(lp) and lp < 0)

    def test_finite_for_valid_period(self):
        prior = NormalFrequencyPrior(mean=300.0, std=75.0, lower_period=100.0)
        f_ok = torch.tensor(1.0 / 300.0)
        lp = prior.log_prob(f_ok)
        self.assertTrue(torch.isfinite(lp))

    def test_default_parameters(self):
        prior = NormalFrequencyPrior()
        self.assertAlmostEqual(prior.loc.item(), 300.0, places=4)
        self.assertAlmostEqual(prior.scale.item(), 75.0, places=4)


# ---------------------------------------------------------------------------
# PRIOR_SETS / get_prior_set tests
# ---------------------------------------------------------------------------


class TestPriorSetsDefinition(unittest.TestCase):
    """PRIOR_SETS has expected structure."""

    def test_lpv_exists(self):
        self.assertIn("LPV", PRIOR_SETS)

    def test_lpv_has_lognormal(self):
        self.assertIn("lognormal", PRIOR_SETS["LPV"])

    def test_lpv_has_normal(self):
        self.assertIn("normal", PRIOR_SETS["LPV"])

    def test_lpv_has_period_bounds(self):
        self.assertIn("period_bounds", PRIOR_SETS["LPV"])

    def test_lpv_lognormal_mu(self):
        self.assertEqual(PRIOR_SETS["LPV"]["lognormal"]["mu"], 5.0)

    def test_lpv_lognormal_sigma(self):
        self.assertEqual(PRIOR_SETS["LPV"]["lognormal"]["sigma"], 1.0)

    def test_lpv_normal_mean(self):
        self.assertEqual(PRIOR_SETS["LPV"]["normal"]["mean"], 300.0)

    def test_lpv_normal_std(self):
        self.assertEqual(PRIOR_SETS["LPV"]["normal"]["std"], 75.0)

    def test_lpv_period_lower_active(self):
        lower_val, lower_active = PRIOR_SETS["LPV"]["period_bounds"]["lower"]
        self.assertTrue(lower_active)
        self.assertIsNotNone(lower_val)

    def test_lpv_period_upper_inactive(self):
        upper_val, upper_active = PRIOR_SETS["LPV"]["period_bounds"]["upper"]
        self.assertFalse(upper_active)


class TestGetPriorSet(unittest.TestCase):
    """get_prior_set returns a deep copy and raises for unknown names."""

    def test_get_lpv(self):
        ps = get_prior_set("LPV")
        self.assertIn("lognormal", ps)

    def test_get_lpv_returns_copy(self):
        ps1 = get_prior_set("LPV")
        ps2 = get_prior_set("LPV")
        ps1["lognormal"]["mu"] = 999.0
        self.assertEqual(ps2["lognormal"]["mu"], 5.0)

    def test_get_unknown_raises(self):
        with self.assertRaises(ValueError):
            get_prior_set("UNKNOWN")


# ---------------------------------------------------------------------------
# Integration tests: Lightcurve.set_period_prior
# ---------------------------------------------------------------------------


class TestSetPeriodPriorSpectralMixture(unittest.TestCase):
    """set_period_prior on SpectralMixtureGPModel (frequency-based)."""

    def setUp(self):
        self.lc = _make_1d_lc(span_days=500.0)
        self.lc.set_model("1D", num_mixtures=2)

    def test_lpv_registers_frequency_prior(self):
        """LPV prior registers a LogNormalFrequencyPrior on mixture_means."""
        self.lc.set_period_prior(prior_set="LPV")
        module = self.lc._model_pars["mixture_means"]["module"]
        prior = module._priors["mixture_means_prior"][0]
        self.assertIsInstance(prior, LogNormalFrequencyPrior)

    def test_lpv_prior_loc_is_neg_mu(self):
        """LPV LogNormalFrequencyPrior has loc = -5.0."""
        self.lc.set_period_prior(prior_set="LPV")
        module = self.lc._model_pars["mixture_means"]["module"]
        prior = module._priors["mixture_means_prior"][0]
        self.assertAlmostEqual(prior.loc.item(), -5.0, places=4)

    def test_lpv_prior_truncates_short_periods(self):
        """Frequencies corresponding to period < 100 days get -inf."""
        self.lc.set_period_prior(prior_set="LPV")
        module = self.lc._model_pars["mixture_means"]["module"]
        prior = module._priors["mixture_means_prior"][0]
        # period = 50 days < lower = 100 days -> -inf
        f_short = torch.tensor(1.0 / 50.0)
        lp = prior.log_prob(f_short)
        self.assertTrue(torch.isinf(lp) and lp < 0)

    def test_lpv_prior_allows_long_periods(self):
        """Frequencies corresponding to period >= 100 days are finite."""
        self.lc.set_period_prior(prior_set="LPV")
        module = self.lc._model_pars["mixture_means"]["module"]
        prior = module._priors["mixture_means_prior"][0]
        # period = 150 days > lower = 100 days -> finite
        f_ok = torch.tensor(1.0 / 150.0)
        lp = prior.log_prob(f_ok)
        self.assertTrue(torch.isfinite(lp))

    def test_normal_prior_type(self):
        """prior_type='normal' registers a NormalFrequencyPrior."""
        self.lc.set_period_prior(prior_type="normal", mean=300.0, std=75.0,
                                  lower_period=100.0)
        module = self.lc._model_pars["mixture_means"]["module"]
        prior = module._priors["mixture_means_prior"][0]
        self.assertIsInstance(prior, NormalFrequencyPrior)

    def test_explicit_period_bounds(self):
        """Explicit lower_period overrides prior_set bound."""
        self.lc.set_period_prior(prior_set="LPV", lower_period=200.0)
        module = self.lc._model_pars["mixture_means"]["module"]
        prior = module._priors["mixture_means_prior"][0]
        self.assertEqual(prior.lower_period, 200.0)

    def test_invalid_prior_type_raises(self):
        with self.assertRaises(ValueError):
            self.lc.set_period_prior(prior_type="invalid")

    def test_invalid_prior_set_raises(self):
        with self.assertRaises(ValueError):
            self.lc.set_period_prior(prior_set="UNKNOWN")


class TestSetPeriodPriorQuasiPeriodic(unittest.TestCase):
    """set_period_prior on QuasiPeriodicGPModel (period-based)."""

    def setUp(self):
        self.lc = _make_1d_lc(span_days=500.0)
        self.lc.set_model("1DQuasiPeriodic")

    def test_lpv_registers_period_prior(self):
        """LPV prior registers a LogNormalPeriodPrior on period_length."""
        self.lc.set_period_prior(prior_set="LPV")
        period_keys = [
            k for k in self.lc._model_pars
            if "period_length" in k and "raw_" not in k
        ]
        self.assertTrue(len(period_keys) > 0, "No period_length keys found")
        module = self.lc._model_pars[period_keys[0]]["module"]
        prior = module._priors["period_length_prior"][0]
        self.assertIsInstance(prior, LogNormalPeriodPrior)

    def test_lpv_prior_loc_equals_mu(self):
        """LPV LogNormalPeriodPrior has loc = 5.0."""
        self.lc.set_period_prior(prior_set="LPV")
        period_keys = [
            k for k in self.lc._model_pars
            if "period_length" in k and "raw_" not in k
        ]
        module = self.lc._model_pars[period_keys[0]]["module"]
        prior = module._priors["period_length_prior"][0]
        self.assertAlmostEqual(prior.loc.item(), 5.0, places=4)

    def test_lpv_prior_lower_bound(self):
        """LPV lower_bound is derived from the 100-day period limit."""
        self.lc.set_period_prior(prior_set="LPV")
        period_keys = [
            k for k in self.lc._model_pars
            if "period_length" in k and "raw_" not in k
        ]
        module = self.lc._model_pars[period_keys[0]]["module"]
        prior = module._priors["period_length_prior"][0]
        # No xtransform, so lower_bound should be 100.0
        self.assertAlmostEqual(prior.lower_bound, 100.0, places=4)

    def test_period_below_lower_gets_neginf(self):
        """Periods below lower_bound get -inf log_prob."""
        self.lc.set_period_prior(prior_set="LPV")
        period_keys = [
            k for k in self.lc._model_pars
            if "period_length" in k and "raw_" not in k
        ]
        module = self.lc._model_pars[period_keys[0]]["module"]
        prior = module._priors["period_length_prior"][0]
        lp = prior.log_prob(torch.tensor(50.0))
        self.assertTrue(torch.isinf(lp) and lp < 0)

    def test_normal_period_prior(self):
        """prior_type='normal' registers a NormalPeriodPrior."""
        self.lc.set_period_prior(prior_type="normal", mean=300.0, std=75.0)
        period_keys = [
            k for k in self.lc._model_pars
            if "period_length" in k and "raw_" not in k
        ]
        module = self.lc._model_pars[period_keys[0]]["module"]
        prior = module._priors["period_length_prior"][0]
        self.assertIsInstance(prior, NormalPeriodPrior)


class TestSetPeriodPriorNoPeriodicty(unittest.TestCase):
    """set_period_prior on MaternGPModel warns and does nothing."""

    def test_matern_warns(self):
        lc = _make_1d_lc(span_days=500.0)
        lc.set_model("1DMatern")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lc.set_period_prior(prior_set="LPV")
        self.assertTrue(any("no period or frequency" in str(warning.message).lower()
                            for warning in w))


class TestSetPeriodPriorNotInitialized(unittest.TestCase):
    """set_period_prior raises if model not yet set."""

    def test_raises_without_model(self):
        lc = _make_1d_lc()
        with self.assertRaises(RuntimeError):
            lc.set_period_prior(prior_set="LPV")


class TestSetDefaultPriorsWithPriorSet(unittest.TestCase):
    """set_default_priors(prior_set=...) calls set_period_prior internally."""

    def test_set_default_priors_lpv_spectral(self):
        """set_default_priors(prior_set='LPV') works on SpectralMixture."""
        lc = _make_1d_lc()
        lc.set_model("1D", num_mixtures=2)
        lc.set_default_priors(prior_set="LPV")
        self.assertTrue(lc._Lightcurve__PRIORS_SET)
        module = lc._model_pars["mixture_means"]["module"]
        prior = module._priors["mixture_means_prior"][0]
        self.assertIsInstance(prior, LogNormalFrequencyPrior)

    def test_set_default_priors_lpv_quasi_periodic(self):
        """set_default_priors(prior_set='LPV') works on QuasiPeriodic."""
        lc = _make_1d_lc()
        lc.set_model("1DQuasiPeriodic")
        lc.set_default_priors(prior_set="LPV")
        self.assertTrue(lc._Lightcurve__PRIORS_SET)

    def test_set_default_priors_no_prior_set(self):
        """set_default_priors() without prior_set does not crash on QP model."""
        lc = _make_1d_lc()
        lc.set_model("1DQuasiPeriodic")
        # Should not raise KeyError (pre-existing bug was fixed)
        lc.set_default_priors()
        self.assertTrue(lc._Lightcurve__PRIORS_SET)


class TestSetPeriodPriorPeriodicPlusStochastic(unittest.TestCase):
    """set_period_prior on PeriodicPlusStochastic (period-based)."""

    def test_registers_period_prior(self):
        lc = _make_1d_lc()
        lc.set_model("1DPeriodicStochastic")
        lc.set_period_prior(prior_set="LPV")
        period_keys = [
            k for k in lc._model_pars
            if "period_length" in k and "raw_" not in k
        ]
        self.assertTrue(len(period_keys) > 0)
        module = lc._model_pars[period_keys[0]]["module"]
        prior = module._priors["period_length_prior"][0]
        self.assertIsInstance(prior, LogNormalPeriodPrior)


if __name__ == "__main__":
    unittest.main()
