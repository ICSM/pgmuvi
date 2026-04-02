"""Tests for Lightcurve.get_period_summary() and plot_period_summary().

Covers:
1. 1-D spectral-mixture model – method runs and returns expected keys.
2. 2-D spectral-mixture model – time dimension is handled correctly.
3. xtransform is None – raw frequency/period extraction.
4. xtransform not None (MinMax) – inverse-transformed extraction.
5. Non-spectral-mixture model – raises documented ValueError.
"""

import unittest

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")  # non-interactive backend for tests

from pgmuvi.lightcurve import Lightcurve, MinMax
from pgmuvi.synthetic import make_chromatic_sinusoid_2d, make_simple_sinusoid_1d


# ---------------------------------------------------------------------------
# Expected keys that must be present in every get_period_summary() result
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {
    "component_periods",
    "component_weights",
    "component_period_scales",
    "component_frequencies",
    "component_frequency_scales",
    "freq_grid",
    "psd",
    "dominant_frequency",
    "dominant_period",
    "period_interval_fwhm_like",
    "q_factor",
    "peak_fraction",
    "n_significant_peaks",
    "significant_periods",
    "method",
    "notes",
}


# ---------------------------------------------------------------------------
# Helper: build a 1-D Lightcurve with a set model (no xtransform)
# ---------------------------------------------------------------------------


def _make_1d_lc_no_transform(n_obs=40, period=100.0, seed=0):
    """1-D light curve with xtransform=None."""
    lc = make_simple_sinusoid_1d(
        n_obs=n_obs, period=period, noise_level=0.05, t_span=500.0, seed=seed
    )
    # make_simple_sinusoid_1d already returns a Lightcurve; confirm no transform
    lc.xtransform = None
    lc.xdata = lc._xdata_raw  # reset transforms
    lc.set_model("1D", num_mixtures=2)
    return lc


def _make_1d_lc_with_transform(n_obs=40, period=100.0, seed=0):
    """1-D light curve with MinMax xtransform."""
    lc = make_simple_sinusoid_1d(
        n_obs=n_obs, period=period, noise_level=0.05, t_span=500.0, seed=seed
    )
    lc.xtransform = MinMax()
    lc.xdata = lc._xdata_raw  # triggers transform recalculation
    lc.set_model("1D", num_mixtures=2)
    return lc


def _make_2d_lc(seed=0):
    """2-D (multiband) light curve with xtransform=None."""
    lc = make_chromatic_sinusoid_2d(period=100.0, seed=seed)
    lc.set_model("2D", num_mixtures=2)
    return lc


# ---------------------------------------------------------------------------
# 1. 1-D spectral-mixture, xtransform=None – keys and basic sanity
# ---------------------------------------------------------------------------


class TestGetPeriodSummary1D(unittest.TestCase):
    """get_period_summary() on a 1-D SM model without xtransform."""

    def setUp(self):
        self.lc = _make_1d_lc_no_transform()
        self.summary = self.lc.get_period_summary()

    def test_returns_dict(self):
        self.assertIsInstance(self.summary, dict)

    def test_all_required_keys_present(self):
        self.assertSetEqual(_REQUIRED_KEYS, set(self.summary.keys()))

    def test_dominant_period_positive(self):
        self.assertGreater(self.summary["dominant_period"], 0.0)

    def test_dominant_frequency_positive(self):
        self.assertGreater(self.summary["dominant_frequency"], 0.0)

    def test_period_interval_ordered(self):
        lo, hi = self.summary["period_interval_fwhm_like"]
        self.assertLessEqual(lo, hi)

    def test_q_factor_positive(self):
        q = self.summary["q_factor"]
        self.assertTrue((np.isfinite(q) and q > 0) or np.isinf(q))

    def test_n_significant_peaks_at_least_one(self):
        self.assertGreaterEqual(self.summary["n_significant_peaks"], 1)

    def test_freq_grid_and_psd_same_length(self):
        self.assertEqual(
            len(self.summary["freq_grid"]),
            len(self.summary["psd"]),
        )

    def test_component_arrays_have_n_mix_elements(self):
        n = 2  # num_mixtures set above
        self.assertEqual(len(self.summary["component_periods"]), n)
        self.assertEqual(len(self.summary["component_weights"]), n)
        self.assertEqual(
            len(self.summary["component_frequencies"]), n
        )

    def test_method_field(self):
        self.assertEqual(
            self.summary["method"], "spectral_mixture_psd_peak"
        )

    def test_notes_field_is_string(self):
        self.assertIsInstance(self.summary["notes"], str)

    def test_custom_n_grid(self):
        s = self.lc.get_period_summary(n_grid=200)
        self.assertEqual(len(s["freq_grid"]), 200)


# ---------------------------------------------------------------------------
# 2. 2-D spectral-mixture – time dimension handling
# ---------------------------------------------------------------------------


class TestGetPeriodSummary2D(unittest.TestCase):
    """get_period_summary() on a 2-D SM model."""

    def setUp(self):
        self.lc = _make_2d_lc()
        self.summary = self.lc.get_period_summary()

    def test_returns_dict(self):
        self.assertIsInstance(self.summary, dict)

    def test_all_required_keys_present(self):
        self.assertSetEqual(_REQUIRED_KEYS, set(self.summary.keys()))

    def test_dominant_period_positive(self):
        self.assertGreater(self.summary["dominant_period"], 0.0)

    def test_component_arrays_have_n_mix_elements(self):
        n = 2
        self.assertEqual(len(self.summary["component_periods"]), n)

    def test_freq_grid_all_positive(self):
        self.assertTrue(np.all(self.summary["freq_grid"] > 0))


# ---------------------------------------------------------------------------
# 3. xtransform=None – _extract_sm_params raw-unit extraction
# ---------------------------------------------------------------------------


class TestExtractSmParamsNoTransform(unittest.TestCase):
    """_extract_sm_params with xtransform=None."""

    def setUp(self):
        self.lc = _make_1d_lc_no_transform()
        self.params = self.lc._extract_sm_params()

    def test_frequencies_positive(self):
        self.assertTrue(np.all(self.params["component_frequencies"] > 0))

    def test_periods_positive(self):
        self.assertTrue(np.all(self.params["component_periods"] > 0))

    def test_freq_scale_positive(self):
        self.assertTrue(
            np.all(self.params["component_frequency_scales"] > 0)
        )

    def test_period_scale_positive(self):
        self.assertTrue(
            np.all(self.params["component_period_scales"] > 0)
        )

    def test_weights_positive(self):
        self.assertTrue(np.all(self.params["component_weights"] > 0))

    def test_period_frequency_consistency(self):
        """period = 1 / frequency for each component."""
        p = self.params["component_periods"]
        f = self.params["component_frequencies"]
        np.testing.assert_allclose(p, 1.0 / f, rtol=1e-5)


# ---------------------------------------------------------------------------
# 4. xtransform not None – _extract_sm_params inverse-transform extraction
# ---------------------------------------------------------------------------


class TestExtractSmParamsWithTransform(unittest.TestCase):
    """_extract_sm_params with MinMax xtransform."""

    def setUp(self):
        self.lc = _make_1d_lc_with_transform()
        self.params = self.lc._extract_sm_params()

    def test_frequencies_positive(self):
        self.assertTrue(np.all(self.params["component_frequencies"] > 0))

    def test_periods_positive(self):
        self.assertTrue(np.all(self.params["component_periods"] > 0))

    def test_period_frequency_consistency(self):
        p = self.params["component_periods"]
        f = self.params["component_frequencies"]
        np.testing.assert_allclose(p, 1.0 / f, rtol=1e-5)

    def test_summary_runs(self):
        """get_period_summary() completes without error."""
        summary = self.lc.get_period_summary()
        self.assertIn("dominant_period", summary)
        self.assertGreater(summary["dominant_period"], 0.0)


# ---------------------------------------------------------------------------
# 5. Non-spectral-mixture model – raises ValueError
# ---------------------------------------------------------------------------


class TestGetPeriodSummaryNonSM(unittest.TestCase):
    """get_period_summary() raises ValueError for non-SM models."""

    def _lc_with_model(self, model_name):
        lc = make_simple_sinusoid_1d(n_obs=20, seed=0)
        lc.set_model(model_name)
        return lc

    def test_matern_raises_value_error(self):
        lc = self._lc_with_model("1DMatern")
        with self.assertRaises(ValueError):
            lc.get_period_summary()

    def test_quasi_periodic_raises_value_error(self):
        lc = self._lc_with_model("1DQuasiPeriodic")
        with self.assertRaises(ValueError):
            lc.get_period_summary()


# ---------------------------------------------------------------------------
# 6. Model not initialised – raises RuntimeError
# ---------------------------------------------------------------------------


class TestGetPeriodSummaryNoModel(unittest.TestCase):
    """get_period_summary() raises RuntimeError if model not set."""

    def test_raises_runtime_error(self):
        lc = Lightcurve(
            torch.linspace(0, 500, 50), torch.randn(50)
        )
        with self.assertRaises(RuntimeError):
            lc.get_period_summary()


# ---------------------------------------------------------------------------
# 7. plot_period_summary – basic smoke test
# ---------------------------------------------------------------------------


class TestPlotPeriodSummary(unittest.TestCase):
    """plot_period_summary() returns a figure when show=False."""

    def setUp(self):
        self.lc = _make_1d_lc_no_transform()

    def test_returns_fig_ax_when_show_false(self):
        import matplotlib.pyplot as plt

        result = self.lc.plot_period_summary(show=False)
        self.assertIsNotNone(result)
        fig, ax = result
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_accepts_precomputed_summary(self):
        import matplotlib.pyplot as plt

        summary = self.lc.get_period_summary()
        result = self.lc.plot_period_summary(summary=summary, show=False)
        self.assertIsNotNone(result)
        fig, ax = result
        plt.close(fig)

    def test_linear_freq_axis(self):
        import matplotlib.pyplot as plt

        result = self.lc.plot_period_summary(show=False, log_freq=False)
        fig, ax = result
        self.assertNotEqual(ax.get_xscale(), "log")
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
