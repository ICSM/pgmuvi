"""Tests for Lightcurve.get_period_summary() and plot_period_summary().

Covers:
1. 1-D spectral-mixture model - method runs and returns expected keys.
2. 2-D spectral-mixture model - time dimension is handled correctly.
3. xtransform is None - raw frequency/period extraction.
4. xtransform not None (MinMax) - inverse-transformed extraction.
5. Explicit-period models (QuasiPeriodic, LinearQuasiPeriodic) - return
   a dict with a positive dominant_period and method='explicit_period_parameter'.
6. PeriodicPlusStochastic - returns a dict with a positive dominant_period
   and method='periodic_plus_stochastic'.
7. Non-periodic models (Matern) - return a graceful dict with
   dominant_period=None and method='non_periodic_kernel'.
8. Separable 2D models - return a dict; result is non-periodic when the
   default Matern time kernel is used, explicit-period when QP time kernel.
9. Model not initialised - raises RuntimeError.
10. plot_period_summary - works for SM, explicit-period, and non-periodic.
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
# Shared factory helpers
# ---------------------------------------------------------------------------


def _make_1d_lc_no_transform(n_obs=40, period=100.0, seed=0):
    """1-D light curve with xtransform=None and a 1D SM model."""
    lc = make_simple_sinusoid_1d(
        n_obs=n_obs, period=period, noise_level=0.05, t_span=500.0, seed=seed
    )
    lc.xtransform = None
    lc.xdata = lc._xdata_raw
    lc.set_model("1D", num_mixtures=2)
    return lc


def _make_1d_lc_with_transform(n_obs=40, period=100.0, seed=0):
    """1-D light curve with MinMax xtransform and a 1D SM model."""
    lc = make_simple_sinusoid_1d(
        n_obs=n_obs, period=period, noise_level=0.05, t_span=500.0, seed=seed
    )
    lc.xtransform = MinMax()
    lc.xdata = lc._xdata_raw
    lc.set_model("1D", num_mixtures=2)
    return lc


def _make_2d_lc(seed=0):
    """2-D (multiband) light curve with a 2D SM model."""
    lc = make_chromatic_sinusoid_2d(period=100.0, seed=seed)
    lc.set_model("2D", num_mixtures=2)
    return lc


def _make_1d_lc_model(model_name, n_obs=30, period=100.0, seed=0, **kw):
    """1-D light curve with the specified model."""
    lc = make_simple_sinusoid_1d(
        n_obs=n_obs, period=period, noise_level=0.05, t_span=500.0, seed=seed
    )
    lc.set_model(model_name, **kw)
    return lc


# ---------------------------------------------------------------------------
# 1. 1-D spectral-mixture, xtransform=None
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
        n = 2
        self.assertEqual(len(self.summary["component_periods"]), n)
        self.assertEqual(len(self.summary["component_weights"]), n)
        self.assertEqual(len(self.summary["component_frequencies"]), n)

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
# 2. 2-D spectral-mixture
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
# 3. xtransform=None - _extract_sm_params raw-unit extraction
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
        p = self.params["component_periods"]
        f = self.params["component_frequencies"]
        np.testing.assert_allclose(p, 1.0 / f, rtol=1e-5)


# ---------------------------------------------------------------------------
# 4. xtransform not None - _extract_sm_params inverse-transform extraction
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
        summary = self.lc.get_period_summary()
        self.assertIn("dominant_period", summary)
        self.assertGreater(summary["dominant_period"], 0.0)


# ---------------------------------------------------------------------------
# 5. Explicit-period models (QuasiPeriodic, LinearQuasiPeriodic)
# ---------------------------------------------------------------------------


class TestGetPeriodSummaryQuasiPeriodic(unittest.TestCase):
    """get_period_summary() on QuasiPeriodic and LinearMeanQuasiPeriodic."""

    def _check_summary(self, lc, model_name):
        summary = lc.get_period_summary()
        self.assertIsInstance(summary, dict, msg=model_name)
        self.assertSetEqual(
            _REQUIRED_KEYS, set(summary.keys()), msg=model_name
        )
        self.assertIsNotNone(summary["dominant_period"], msg=model_name)
        self.assertGreater(
            summary["dominant_period"], 0.0, msg=model_name
        )
        self.assertEqual(
            summary["method"], "explicit_period_parameter", msg=model_name
        )
        self.assertIsInstance(summary["notes"], str, msg=model_name)

    def test_quasi_periodic_returns_dict(self):
        lc = _make_1d_lc_model("1DQuasiPeriodic", period=100.0)
        self._check_summary(lc, "1DQuasiPeriodic")

    def test_linear_qp_returns_dict(self):
        lc = _make_1d_lc_model("1DLinearQuasiPeriodic", period=100.0)
        self._check_summary(lc, "1DLinearQuasiPeriodic")

    def test_quasi_periodic_period_positive(self):
        lc = _make_1d_lc_model("1DQuasiPeriodic", period=100.0)
        s = lc.get_period_summary()
        self.assertGreater(s["dominant_period"], 0.0)

    def test_quasi_periodic_freq_grid_is_none(self):
        """No PSD is built for the explicit-period backend."""
        lc = _make_1d_lc_model("1DQuasiPeriodic", period=100.0)
        s = lc.get_period_summary()
        self.assertIsNone(s["freq_grid"])
        self.assertIsNone(s["psd"])

    def test_quasi_periodic_interval_ordered(self):
        lc = _make_1d_lc_model("1DQuasiPeriodic", period=100.0)
        s = lc.get_period_summary()
        lo, hi = s["period_interval_fwhm_like"]
        self.assertLessEqual(lo, hi)

    def test_quasi_periodic_n_sig_peaks_at_least_one(self):
        lc = _make_1d_lc_model("1DQuasiPeriodic", period=100.0)
        s = lc.get_period_summary()
        self.assertGreaterEqual(s["n_significant_peaks"], 1)


# ---------------------------------------------------------------------------
# 6. Periodic-plus-stochastic
# ---------------------------------------------------------------------------


class TestGetPeriodSummaryPeriodicStochastic(unittest.TestCase):
    """get_period_summary() on PeriodicPlusStochasticGPModel."""

    def setUp(self):
        self.lc = _make_1d_lc_model("1DPeriodicStochastic", period=100.0)
        self.summary = self.lc.get_period_summary()

    def test_returns_dict(self):
        self.assertIsInstance(self.summary, dict)

    def test_all_required_keys_present(self):
        self.assertSetEqual(_REQUIRED_KEYS, set(self.summary.keys()))

    def test_dominant_period_positive(self):
        self.assertIsNotNone(self.summary["dominant_period"])
        self.assertGreater(self.summary["dominant_period"], 0.0)

    def test_method_is_periodic_plus_stochastic(self):
        self.assertEqual(
            self.summary["method"], "periodic_plus_stochastic"
        )

    def test_notes_mention_stochastic(self):
        self.assertIn("stochastic", self.summary["notes"].lower())

    def test_freq_grid_is_none(self):
        self.assertIsNone(self.summary["freq_grid"])


# ---------------------------------------------------------------------------
# 7. Non-periodic models (Matern)
# ---------------------------------------------------------------------------


class TestGetPeriodSummaryNonPeriodic(unittest.TestCase):
    """get_period_summary() on Matern model returns graceful summary."""

    def setUp(self):
        self.lc = _make_1d_lc_model("1DMatern")
        self.summary = self.lc.get_period_summary()

    def test_returns_dict(self):
        self.assertIsInstance(self.summary, dict)

    def test_all_required_keys_present(self):
        self.assertSetEqual(_REQUIRED_KEYS, set(self.summary.keys()))

    def test_dominant_period_is_none(self):
        self.assertIsNone(self.summary["dominant_period"])

    def test_dominant_frequency_is_none(self):
        self.assertIsNone(self.summary["dominant_frequency"])

    def test_method_is_non_periodic(self):
        self.assertEqual(self.summary["method"], "non_periodic_kernel")

    def test_freq_grid_is_none(self):
        self.assertIsNone(self.summary["freq_grid"])

    def test_psd_is_none(self):
        self.assertIsNone(self.summary["psd"])

    def test_n_significant_peaks_zero(self):
        self.assertEqual(self.summary["n_significant_peaks"], 0)

    def test_notes_field_is_string(self):
        self.assertIsInstance(self.summary["notes"], str)


# ---------------------------------------------------------------------------
# 8. Separable 2D models
# ---------------------------------------------------------------------------


class TestGetPeriodSummarySeparable2D(unittest.TestCase):
    """get_period_summary() on separable 2D models."""

    def _lc_2d(self, model_name, **kw):
        lc = make_chromatic_sinusoid_2d(period=100.0, seed=0)
        lc.set_model(model_name, **kw)
        return lc

    def _check_base(self, lc, model_name):
        s = lc.get_period_summary()
        self.assertIsInstance(s, dict, msg=model_name)
        self.assertSetEqual(
            _REQUIRED_KEYS, set(s.keys()), msg=model_name
        )
        return s

    def test_2d_separable_default_returns_dict(self):
        lc = self._lc_2d("2DSeparable")
        self._check_base(lc, "2DSeparable")

    def test_2d_separable_default_is_non_periodic(self):
        """Default Matern time kernel => non-periodic."""
        lc = self._lc_2d("2DSeparable")
        s = lc.get_period_summary()
        self.assertIsNone(s["dominant_period"])
        self.assertEqual(s["method"], "non_periodic_kernel")

    def test_2d_achromatic_default_returns_dict(self):
        lc = self._lc_2d("2DAchromatic")
        self._check_base(lc, "2DAchromatic")

    def test_2d_achromatic_with_qp_returns_period(self):
        """QP time kernel => explicit_period_parameter."""
        lc = self._lc_2d(
            "2DAchromatic",
            time_kernel_type="quasi_periodic",
            period=100.0,
        )
        s = lc.get_period_summary()
        self.assertIsNotNone(s["dominant_period"])
        self.assertGreater(s["dominant_period"], 0.0)
        self.assertEqual(s["method"], "explicit_period_parameter")

    def test_2d_wavelength_dependent_default_returns_dict(self):
        lc = self._lc_2d("2DWavelengthDependent")
        self._check_base(lc, "2DWavelengthDependent")

    def test_2d_dust_mean_returns_dict(self):
        lc = self._lc_2d("2DDustMean")
        self._check_base(lc, "2DDustMean")

    def test_2d_power_law_mean_returns_dict(self):
        lc = self._lc_2d("2DPowerLawMean")
        self._check_base(lc, "2DPowerLawMean")


# ---------------------------------------------------------------------------
# 9. Model not initialised - raises RuntimeError
# ---------------------------------------------------------------------------


class TestGetPeriodSummaryNoModel(unittest.TestCase):
    """get_period_summary() raises RuntimeError if model not set."""

    def test_raises_runtime_error(self):
        lc = Lightcurve(torch.linspace(0, 500, 50), torch.randn(50))
        with self.assertRaises(RuntimeError):
            lc.get_period_summary()


# ---------------------------------------------------------------------------
# 10. plot_period_summary - works for all summary types
# ---------------------------------------------------------------------------


class TestPlotPeriodSummary(unittest.TestCase):
    """plot_period_summary() returns a figure for all summary types."""

    def _check_fig_ax(self, result, label):
        import matplotlib.pyplot as plt

        self.assertIsNotNone(result, msg=label)
        fig, ax = result
        self.assertIsInstance(fig, plt.Figure, msg=label)
        plt.close(fig)

    def test_sm_returns_fig_ax(self):
        lc = _make_1d_lc_no_transform()
        self._check_fig_ax(
            lc.plot_period_summary(show=False), "SM 1D"
        )

    def test_sm_accepts_precomputed_summary(self):
        import matplotlib.pyplot as plt

        lc = _make_1d_lc_no_transform()
        summary = lc.get_period_summary()
        result = lc.plot_period_summary(summary=summary, show=False)
        self._check_fig_ax(result, "SM precomputed")

    def test_sm_linear_freq_axis(self):
        import matplotlib.pyplot as plt

        lc = _make_1d_lc_no_transform()
        fig, ax = lc.plot_period_summary(show=False, log_freq=False)
        self.assertNotEqual(ax.get_xscale(), "log")
        plt.close(fig)

    def test_qp_returns_fig_ax(self):
        lc = _make_1d_lc_model("1DQuasiPeriodic", period=100.0)
        self._check_fig_ax(
            lc.plot_period_summary(show=False), "QP explicit period"
        )

    def test_periodic_stochastic_returns_fig_ax(self):
        lc = _make_1d_lc_model("1DPeriodicStochastic", period=100.0)
        self._check_fig_ax(
            lc.plot_period_summary(show=False), "Periodic+Stochastic"
        )

    def test_matern_returns_fig_ax(self):
        lc = _make_1d_lc_model("1DMatern")
        self._check_fig_ax(
            lc.plot_period_summary(show=False), "Matern non-periodic"
        )

    def test_separable_2d_default_returns_fig_ax(self):
        lc = make_chromatic_sinusoid_2d(period=100.0, seed=0)
        lc.set_model("2DSeparable")
        self._check_fig_ax(
            lc.plot_period_summary(show=False), "2DSeparable"
        )

    def test_separable_2d_qp_returns_fig_ax(self):
        lc = make_chromatic_sinusoid_2d(period=100.0, seed=0)
        lc.set_model(
            "2DAchromatic",
            time_kernel_type="quasi_periodic",
            period=100.0,
        )
        self._check_fig_ax(
            lc.plot_period_summary(show=False), "2DAchromatic QP"
        )


if __name__ == "__main__":
    unittest.main()
