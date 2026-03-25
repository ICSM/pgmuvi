"""Tests for pgmuvi.priors and Lightcurve.set_period_prior."""

import math
import os
import unittest
import warnings

import torch

# Set PGMUVI_TEST_HIGH_PRECISION=1 (or "true"/"yes"/"on") to use large
# integration grids for high-accuracy normalisation tests.  In CI the default
# fast mode uses smaller grids that are still accurate to two decimal places.
_HIGH_PRECISION = os.environ.get("PGMUVI_TEST_HIGH_PRECISION", "").strip().lower() in (
    "1", "true", "yes", "on"
)

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
        """Without bounds, log_prob matches plain LogNormal (no normalizer offset)."""
        prior = LogNormalPeriodPrior(mu=5.0, sigma=1.0)
        import gpytorch
        ref = gpytorch.priors.LogNormalPrior(5.0, 1.0)
        x = torch.tensor([100.0, 200.0, 500.0])
        self.assertTrue(torch.allclose(prior.log_prob(x), ref.log_prob(x)))

    def test_normalizer_nonzero_with_bounds(self):
        """Truncated prior has a non-trivial log normalizer."""
        prior = LogNormalPeriodPrior(mu=5.0, sigma=1.0, lower_bound=100.0)
        self.assertLess(prior._log_normalizer, 0.0)  # CDF mass < 1 → log < 0

    def test_invalid_bounds_raise(self):
        """lower_bound >= upper_bound raises ValueError."""
        with self.assertRaises(ValueError):
            LogNormalPeriodPrior(mu=5.0, sigma=1.0, lower_bound=500.0, upper_bound=100.0)
        with self.assertRaises(ValueError):
            LogNormalPeriodPrior(mu=5.0, sigma=1.0, lower_bound=200.0, upper_bound=200.0)

    def test_normalization_integrates_to_one(self):
        """Truncated log_prob integrates to ~1 over the allowed range."""
        prior = LogNormalPeriodPrior(mu=5.0, sigma=1.0, lower_bound=100.0, upper_bound=800.0)
        # Numerical integration on log-uniform grid
        n_pts = 200000 if _HIGH_PRECISION else 10000
        x = torch.exp(torch.linspace(math.log(100.0), math.log(800.0), n_pts))
        dx = x[1:] - x[:-1]
        lp = prior.log_prob((x[1:] + x[:-1]) / 2)
        integral = (torch.exp(lp) * dx).sum().item()
        self.assertAlmostEqual(integral, 1.0, places=2)

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

    def test_normalizer_nonzero_with_bounds(self):
        prior = NormalPeriodPrior(mean=300.0, std=75.0, lower_bound=100.0)
        self.assertLess(prior._log_normalizer, 0.0)

    def test_invalid_bounds_raise(self):
        """lower_bound >= upper_bound raises ValueError."""
        with self.assertRaises(ValueError):
            NormalPeriodPrior(mean=300.0, std=75.0, lower_bound=600.0, upper_bound=100.0)
        with self.assertRaises(ValueError):
            NormalPeriodPrior(mean=300.0, std=75.0, lower_bound=300.0, upper_bound=300.0)

    def test_normalization_integrates_to_one(self):
        """Truncated log_prob integrates to ~1 over the allowed range."""
        prior = NormalPeriodPrior(mean=300.0, std=75.0, lower_bound=100.0, upper_bound=600.0)
        n_pts = 500000 if _HIGH_PRECISION else 20000
        x = torch.linspace(100.0, 600.0, n_pts)
        dx = x[1] - x[0]
        lp = prior.log_prob((x[1:] + x[:-1]) / 2)
        integral = (torch.exp(lp) * dx).sum().item()
        self.assertAlmostEqual(integral, 1.0, places=2)

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
        """Without bounds, log_prob == Jacobian-correct frequency transformation."""
        mu, sigma = 5.0, 1.0
        period = torch.tensor(150.0)
        freq = 1.0 / period

        import gpytorch
        period_prior = gpytorch.priors.LogNormalPrior(mu, sigma)
        freq_prior = LogNormalFrequencyPrior(mu=mu, sigma=sigma)

        # No bounds → no normalizer offset
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

    def test_normalization_integrates_to_one(self):
        """Truncated log_prob integrates to ~1 over the allowed frequency range."""
        prior = LogNormalFrequencyPrior(mu=5.0, sigma=1.0,
                                        lower_period=100.0, upper_period=800.0)
        f_low, f_high = 1.0 / 800.0, 1.0 / 100.0
        n_pts = 300000 if _HIGH_PRECISION else 10000
        f = torch.exp(torch.linspace(math.log(f_low), math.log(f_high), n_pts))
        df = f[1:] - f[:-1]
        lp = prior.log_prob((f[1:] + f[:-1]) / 2)
        integral = (torch.exp(lp) * df).sum().item()
        self.assertAlmostEqual(integral, 1.0, places=2)

    def test_loc_negated(self):
        """LogNormalFrequencyPrior stores loc=-mu."""
        prior = LogNormalFrequencyPrior(mu=5.0, sigma=1.0)
        self.assertAlmostEqual(prior.loc.item(), -5.0, places=5)

    def test_scale_preserved(self):
        """LogNormalFrequencyPrior preserves sigma as scale."""
        prior = LogNormalFrequencyPrior(mu=5.0, sigma=2.0)
        self.assertAlmostEqual(prior.scale.item(), 2.0, places=5)

    def test_period_false_equivalence(self):
        """period=False with frequency bounds gives same result as period=True with period bounds."""
        # min period = 100 days = max frequency = 0.01
        prior_p = LogNormalFrequencyPrior(mu=5.0, sigma=1.0, lower_period=100.0, period=True)
        prior_f = LogNormalFrequencyPrior(mu=5.0, sigma=1.0, upper_period=0.01, period=False)
        self.assertAlmostEqual(prior_p.lower_period, prior_f.lower_period, places=5)
        freqs = torch.tensor([1.0 / 150.0, 1.0 / 300.0])
        self.assertTrue(torch.allclose(prior_p.log_prob(freqs), prior_f.log_prob(freqs)))

    def test_period_false_blocks_high_freq(self):
        """period=False: frequencies above max_freq are blocked."""
        # max freq = 0.01 (period >= 100 days required)
        prior = LogNormalFrequencyPrior(mu=5.0, sigma=1.0, upper_period=0.01, period=False)
        f_high = torch.tensor(1.0 / 50.0)  # freq=0.02 > 0.01 → blocked
        lp = prior.log_prob(f_high)
        self.assertTrue(torch.isinf(lp) and lp < 0)

    def test_invalid_period_bounds_raise(self):
        """lower_period >= upper_period (after unit conversion) raises ValueError."""
        with self.assertRaises(ValueError):
            LogNormalFrequencyPrior(mu=5.0, sigma=1.0,
                                    lower_period=500.0, upper_period=100.0)
        with self.assertRaises(ValueError):
            LogNormalFrequencyPrior(mu=5.0, sigma=1.0,
                                    lower_period=200.0, upper_period=200.0)

    def test_invalid_freq_bounds_raise(self):
        """period=False: lower_freq >= upper_freq raises ValueError."""
        # When period=False: lower_period arg = min_freq, upper_period arg = max_freq.
        # Conversion: lower_period_stored = 1/max_freq, upper_period_stored = 1/min_freq.
        # Here min_freq=0.02 > max_freq=0.01 → lower_period_stored=50 > upper_period_stored=100 → error.
        with self.assertRaises(ValueError):
            LogNormalFrequencyPrior(mu=5.0, sigma=1.0,
                                    lower_period=0.02, upper_period=0.01, period=False)


# ---------------------------------------------------------------------------
# NormalFrequencyPrior unit tests
# ---------------------------------------------------------------------------


class TestNormalFrequencyPrior(unittest.TestCase):
    """Tests for NormalFrequencyPrior."""

    def test_jacobian_correction(self):
        """Without bounds, log_prob(f) = Normal(mean,std).log_prob(1/f) - 2*log(f)."""
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

    def test_nonpositive_freq_returns_neginf(self):
        """Non-positive frequencies return -inf without NaN."""
        prior = NormalFrequencyPrior(mean=300.0, std=75.0)
        for f_val in [0.0, -1.0, -0.001]:
            with self.subTest(f=f_val):
                lp = prior.log_prob(torch.tensor([f_val]))
                self.assertTrue(torch.isinf(lp) and lp.item() < 0,
                                f"Expected -inf for f={f_val}, got {lp}")
        # Mixed batch: non-positive entries are -inf, valid entries are finite
        mixed = torch.tensor([-1.0, 0.0, 1.0 / 300.0])
        lp = prior.log_prob(mixed)
        self.assertTrue(torch.all(torch.isinf(lp[:2]) & (lp[:2] < 0)))
        self.assertTrue(torch.isfinite(lp[2]))
        self.assertFalse(torch.any(torch.isnan(lp)))

    def test_finite_for_valid_period(self):
        prior = NormalFrequencyPrior(mean=300.0, std=75.0, lower_period=100.0)
        f_ok = torch.tensor(1.0 / 300.0)
        lp = prior.log_prob(f_ok)
        self.assertTrue(torch.isfinite(lp))

    def test_normalization_integrates_to_one(self):
        """Truncated log_prob integrates to ~1 over the allowed frequency range."""
        prior = NormalFrequencyPrior(mean=300.0, std=75.0,
                                     lower_period=100.0, upper_period=600.0)
        f_low, f_high = 1.0 / 600.0, 1.0 / 100.0
        n_pts = 500000 if _HIGH_PRECISION else 20000
        f = torch.linspace(f_low, f_high, n_pts)
        df = f[1] - f[0]
        lp = prior.log_prob((f[1:] + f[:-1]) / 2)
        integral = (torch.exp(lp) * df).sum().item()
        self.assertAlmostEqual(integral, 1.0, places=2)

    def test_default_parameters(self):
        prior = NormalFrequencyPrior()
        self.assertAlmostEqual(prior.loc.item(), 300.0, places=4)
        self.assertAlmostEqual(prior.scale.item(), 75.0, places=4)

    def test_period_false_equivalence(self):
        """period=False with frequency bounds gives same result as period=True."""
        prior_p = NormalFrequencyPrior(mean=300.0, std=75.0, lower_period=100.0, period=True)
        prior_f = NormalFrequencyPrior(mean=300.0, std=75.0, upper_period=0.01, period=False)
        freqs = torch.tensor([1.0 / 150.0, 1.0 / 300.0])
        self.assertTrue(torch.allclose(prior_p.log_prob(freqs), prior_f.log_prob(freqs)))

    def test_invalid_period_bounds_raise(self):
        """lower_period >= upper_period raises ValueError."""
        with self.assertRaises(ValueError):
            NormalFrequencyPrior(mean=300.0, std=75.0,
                                 lower_period=600.0, upper_period=100.0)
        with self.assertRaises(ValueError):
            NormalFrequencyPrior(mean=300.0, std=75.0,
                                 lower_period=300.0, upper_period=300.0)


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

    def test_lpv_no_period_bounds_in_raw_dict(self):
        """period_bounds is not stored in PRIOR_SETS (derived at runtime from CONSTRAINT_SETS)."""
        self.assertNotIn("period_bounds", PRIOR_SETS["LPV"])

    def test_lpv_lognormal_mu(self):
        self.assertEqual(PRIOR_SETS["LPV"]["lognormal"]["mu"], 5.0)

    def test_lpv_lognormal_sigma(self):
        self.assertEqual(PRIOR_SETS["LPV"]["lognormal"]["sigma"], 1.0)

    def test_lpv_normal_mean(self):
        self.assertEqual(PRIOR_SETS["LPV"]["normal"]["mean"], 300.0)

    def test_lpv_normal_std(self):
        self.assertEqual(PRIOR_SETS["LPV"]["normal"]["std"], 75.0)


class TestGetPriorSet(unittest.TestCase):
    """get_prior_set returns a deep copy with period_bounds from CONSTRAINT_SETS."""

    def test_get_lpv(self):
        ps = get_prior_set("LPV")
        self.assertIn("lognormal", ps)

    def test_get_lpv_has_period_bounds(self):
        """get_prior_set includes period_bounds derived from CONSTRAINT_SETS."""
        ps = get_prior_set("LPV")
        self.assertIn("period_bounds", ps)

    def test_get_lpv_period_lower_matches_constraint_set(self):
        """period_bounds lower matches CONSTRAINT_SETS['LPV']['period']['lower']."""
        from pgmuvi.constraints import CONSTRAINT_SETS
        ps = get_prior_set("LPV")
        self.assertEqual(ps["period_bounds"]["lower"],
                         CONSTRAINT_SETS["LPV"]["period"]["lower"])

    def test_get_lpv_period_upper_matches_constraint_set(self):
        """period_bounds upper matches CONSTRAINT_SETS['LPV']['period']['upper']."""
        from pgmuvi.constraints import CONSTRAINT_SETS
        ps = get_prior_set("LPV")
        self.assertEqual(ps["period_bounds"]["upper"],
                         CONSTRAINT_SETS["LPV"]["period"]["upper"])

    def test_get_lpv_period_lower_active(self):
        lower_val, lower_active = get_prior_set("LPV")["period_bounds"]["lower"]
        self.assertTrue(lower_active)
        self.assertIsNotNone(lower_val)

    def test_get_lpv_period_upper_inactive(self):
        upper_val, upper_active = get_prior_set("LPV")["period_bounds"]["upper"]
        self.assertFalse(upper_active)

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

    def test_prior_type_case_insensitive(self):
        """prior_type is case-insensitive (e.g. 'LogNormal' is accepted)."""
        for variant in ("LogNormal", "LOGNORMAL", "Normal", "NORMAL"):
            with self.subTest(variant=variant):
                lc = _make_1d_lc(span_days=500.0)
                lc.set_model("1D", num_mixtures=2)
                # Should not raise
                lc.set_period_prior(
                    prior_type=variant, lower_period=100.0, upper_period=500.0
                )

    def test_invalid_prior_set_raises(self):
        with self.assertRaises(ValueError):
            self.lc.set_period_prior(prior_set="UNKNOWN")

    def test_period_false_registers_prior(self):
        """period=False: bounds in frequency units are correctly converted."""
        # max freq = 1/100 = 0.01 days^-1 → lower period = 100 days
        self.lc.set_period_prior(
            prior_type="lognormal",
            lower_period=1.0 / 500.0,  # min freq = 1/500
            upper_period=1.0 / 100.0,  # max freq = 1/100
            period=False,
        )
        # Verify the prior is registered
        pars = self.lc._model_pars
        self.assertIn("mixture_means", pars)
        module = pars["mixture_means"]["module"]
        self.assertIn("mixture_means_prior", module._priors)
        prior = module._priors["mixture_means_prior"][0]
        self.assertIsInstance(prior, LogNormalFrequencyPrior)


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
        self.assertTrue(
            any(
                issubclass(warning.category, UserWarning)
                and "no period or frequency" in str(warning.message).lower()
                for warning in w
            ),
            "Expected UserWarning about missing period/frequency parameter",
        )


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
