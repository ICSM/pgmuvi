"""Tests for the constraint-set feature (CONSTRAINT_SETS / get_constraint_set)
and for set_default_constraints(constraint_set=...) in Lightcurve.
"""

import unittest

from pgmuvi.constraints import CONSTRAINT_SETS, get_constraint_set
from pgmuvi.lightcurve import Lightcurve
from pgmuvi.synthetic import make_simple_sinusoid_1d
from gpytorch.constraints import Interval, GreaterThan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_1d_lc(n=80, span_days=500.0, seed=0):
    """Return a simple 1-D Lightcurve spanning *span_days* days."""
    return make_simple_sinusoid_1d(
        n_obs=n, period=100.0, noise_level=0.1, t_span=span_days, seed=seed
    )


# ---------------------------------------------------------------------------
# Unit tests for CONSTRAINT_SETS / get_constraint_set
# ---------------------------------------------------------------------------

class TestConstraintSetsDefinition(unittest.TestCase):
    """CONSTRAINT_SETS contains the expected structure."""

    def test_lpv_exists(self):
        self.assertIn("LPV", CONSTRAINT_SETS)

    def test_lpv_has_period(self):
        self.assertIn("period", CONSTRAINT_SETS["LPV"])

    def test_lpv_period_lower_is_20_days_and_active(self):
        lower_val, lower_active = CONSTRAINT_SETS["LPV"]["period"]["lower"]
        self.assertEqual(lower_val, 20.0)
        self.assertTrue(lower_active)

    def test_lpv_period_upper_is_none_and_inactive(self):
        upper_val, upper_active = CONSTRAINT_SETS["LPV"]["period"]["upper"]
        self.assertIsNone(upper_val)
        self.assertFalse(upper_active)


class TestGetConstraintSet(unittest.TestCase):
    """get_constraint_set returns correct dict and raises for unknown sets."""

    def test_get_lpv(self):
        cs = get_constraint_set("LPV")
        self.assertIn("period", cs)

    def test_get_lpv_lower(self):
        cs = get_constraint_set("LPV")
        lower_val, lower_active = cs["period"]["lower"]
        self.assertEqual(lower_val, 20.0)
        self.assertTrue(lower_active)

    def test_get_unknown_raises(self):
        with self.assertRaises(ValueError):
            get_constraint_set("UNKNOWN_SOURCE_TYPE")


# ---------------------------------------------------------------------------
# Integration tests: set_default_constraints with constraint_set="LPV"
# ---------------------------------------------------------------------------

class TestSetDefaultConstraintsNoConstraintSet(unittest.TestCase):
    """Without constraint_set the behaviour is unchanged."""

    def setUp(self):
        self.lc = _make_1d_lc(span_days=500.0)
        self.lc.set_model("1D", likelihood=None, num_mixtures=2)

    def test_default_no_constraint_set(self):
        """set_default_constraints() without constraint_set still works."""
        self.lc.set_default_constraints()
        self.assertTrue(self.lc._Lightcurve__CONTRAINTS_SET)

    def test_constraint_is_greater_than(self):
        """Default 1-D constraint is a GreaterThan (no upper bound)."""
        self.lc.set_default_constraints()
        module = self.lc._model_pars["mixture_means"]["module"]
        # GPyTorch stores constraints with a _constraint suffix
        raw_c = module._constraints.get("raw_mixture_means_constraint")
        self.assertIsInstance(raw_c, GreaterThan)


class TestSetDefaultConstraintsLPV(unittest.TestCase):
    """With constraint_set='LPV' an upper frequency limit is applied."""

    def setUp(self):
        # 500-day dataset → LPV lower period = 20 days
        # In transformed space (MinMax, 1-D): max_freq = 500/20 = 25.0
        self.span_days = 500.0
        self.min_period = 20.0
        self.expected_max_freq = self.span_days / self.min_period  # 25.0

        self.lc = _make_1d_lc(span_days=self.span_days)
        self.lc.set_model("1D", likelihood=None, num_mixtures=2)

    def test_constraint_set_flag(self):
        self.lc.set_default_constraints(constraint_set="LPV")
        self.assertTrue(self.lc._Lightcurve__CONTRAINTS_SET)

    def test_constraint_is_interval(self):
        """LPV constraint converts GreaterThan into an Interval."""
        self.lc.set_default_constraints(constraint_set="LPV")
        module = self.lc._model_pars["mixture_means"]["module"]
        raw_c = module._constraints.get("raw_mixture_means_constraint")
        self.assertIsInstance(raw_c, Interval)

    def test_upper_bound_matches_period_limit(self):
        """Upper frequency bound equals data_span / min_period_days."""
        self.lc.set_default_constraints(constraint_set="LPV")
        module = self.lc._model_pars["mixture_means"]["module"]
        raw_c = module._constraints["raw_mixture_means_constraint"]
        self.assertAlmostEqual(
            float(raw_c.upper_bound),
            self.expected_max_freq,
            places=4,
        )

    def test_lower_bound_unchanged(self):
        """The existing lower frequency bound is preserved."""
        # Lower bound without LPV = 1 / xdata_transformed.max() ≈ 1.0 (MinMax)
        self.lc.set_default_constraints(constraint_set="LPV")
        module = self.lc._model_pars["mixture_means"]["module"]
        raw_c = module._constraints["raw_mixture_means_constraint"]
        # Lower bound should be 1.0 for a MinMax-normalised 1-D dataset
        self.assertAlmostEqual(float(raw_c.lower_bound), 1.0, places=4)

    def test_invalid_constraint_set_raises(self):
        with self.assertRaises(ValueError):
            self.lc.set_default_constraints(constraint_set="INVALID")


class TestSetDefaultConstraintsLPVShortDataset(unittest.TestCase):
    """When data span < min_period the LPV constraint is silently ignored."""

    def setUp(self):
        # 10-day dataset — shorter than the 20-day LPV minimum period.
        # The LPV max-freq would be 10/20 = 0.5, which is LESS than the
        # default lower bound of 1.0, so the constraint should be ignored
        # and the original GreaterThan is kept.
        self.lc = _make_1d_lc(span_days=10.0)
        self.lc.set_model("1D", likelihood=None, num_mixtures=2)

    def test_constraint_stays_greater_than_when_period_larger_than_span(self):
        """No Interval is created if the period lower limit exceeds data span."""
        self.lc.set_default_constraints(constraint_set="LPV")
        module = self.lc._model_pars["mixture_means"]["module"]
        raw_c = module._constraints["raw_mixture_means_constraint"]
        # max_freq_from_period = 10/20 = 0.5 < lower_bound = 1.0
        # → constraint is unchanged (remains GreaterThan)
        self.assertIsInstance(raw_c, GreaterThan)


if __name__ == "__main__":
    unittest.main()
