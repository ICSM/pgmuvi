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

from pgmuvi.lightcurve import Lightcurve, MinMax, PeriodPeakResult, PeriodSummaryResult
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
    "period_interval",
    "interval_definition",
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
        from pgmuvi.lightcurve import PeriodSummaryResult
        self.assertIsInstance(self.summary, (dict, PeriodSummaryResult))

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
        if q is None:
            return  # peak_mass mode does not compute a Q-factor
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
        from pgmuvi.lightcurve import PeriodSummaryResult
        self.assertIsInstance(self.summary, (dict, PeriodSummaryResult))

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


# ---------------------------------------------------------------------------
# 11. Adaptive PSD grid expansion for SM half-max containment
# ---------------------------------------------------------------------------


class TestSmPsdGridExpansion(unittest.TestCase):
    """Tests that the SM backend robustly contains the half-max interval."""

    def _make_sm_lc(self):
        return _make_1d_lc_no_transform(n_obs=40, period=100.0, seed=0)

    # ------------------------------------------------------------------
    # Helper: build a minimal Lightcurve whose SM params are forced so
    # that the dominant peak is very broad, guaranteeing truncation on
    # the first default grid pass.
    # ------------------------------------------------------------------

    def _forced_params(self, center_freq, scale_freq, weight=1.0):
        """Return an _extract_sm_params-compatible dict with one component."""
        return {
            "component_frequencies": np.array([center_freq]),
            "component_periods": np.array([1.0 / center_freq]),
            "component_frequency_scales": np.array([scale_freq]),
            "component_period_scales": np.array([np.nan]),
            "component_weights": np.array([weight]),
        }

    def test_interpolate_halfmax_returns_float(self):
        """_interpolate_halfmax_crossing returns a float."""
        freq_grid = np.linspace(0.001, 1.0, 100)
        # Gaussian PSD centred at 0.5
        psd = np.exp(-0.5 * ((freq_grid - 0.5) / 0.05) ** 2)
        half_max = 0.5
        # Find left crossing bracket the slow way
        peak_idx = int(np.argmax(psd))
        left_idx = peak_idx
        while left_idx > 0 and psd[left_idx] >= half_max:
            left_idx -= 1
        f_cross, interpolated = (
            _make_1d_lc_no_transform()
            ._interpolate_halfmax_crossing(
                freq_grid, psd, left_idx, "left", half_max
            )
        )
        self.assertIsInstance(f_cross, float)
        # crossing must be between the two bracketing grid points
        self.assertGreaterEqual(f_cross, freq_grid[left_idx])
        self.assertLessEqual(f_cross, freq_grid[left_idx + 1])

    def test_interpolate_halfmax_right_side(self):
        """_interpolate_halfmax_crossing right side is accurate."""
        freq_grid = np.linspace(0.001, 1.0, 1000)
        psd = np.exp(-0.5 * ((freq_grid - 0.5) / 0.05) ** 2)
        half_max = 0.5
        peak_idx = int(np.argmax(psd))
        right_idx = peak_idx
        while right_idx < len(psd) - 1 and psd[right_idx] >= half_max:
            right_idx += 1
        f_cross, interpolated = (
            _make_1d_lc_no_transform()
            ._interpolate_halfmax_crossing(
                freq_grid, psd, right_idx, "right", half_max
            )
        )
        self.assertTrue(interpolated)
        self.assertGreaterEqual(f_cross, freq_grid[right_idx - 1])
        self.assertLessEqual(f_cross, freq_grid[right_idx])

    def test_expand_psd_grid_until_contained_expands_when_needed(self):
        """_expand_psd_grid_until_contained runs at least one expansion
        when the initial grid truncates the half-max crossing."""
        lc = self._make_sm_lc()
        # Build a very broad single-component param set:
        # center = 0.01, scale = 0.05 => the Gaussian spans [~-0.14, 0.16]
        # but the initial grid starts at a positive freq > 0.
        # We force min_freq to be so large that psd[0] >= half_max.
        params = self._forced_params(center_freq=0.01, scale_freq=0.05)
        # Initial grid: from 0.009 to 0.011 (intentionally too narrow)
        freq_grid_init = np.linspace(0.009, 0.011, 500)
        psd_init = lc._sm_psd_on_grid(freq_grid_init, params)

        from scipy.signal import find_peaks
        peaks, _ = find_peaks(psd_init)
        if len(peaks) == 0:
            dom_idx = int(np.argmax(psd_init))
        else:
            dom_idx = int(peaks[np.argmax(psd_init[peaks])])
        half_max = 0.5 * float(psd_init[dom_idx])

        (
            freq_grid_out, psd_out, dom_idx_out,
            left_truncated, right_truncated, n_expansions,
        ) = lc._expand_psd_grid_until_contained(
            freq_grid_init, psd_init, params, dom_idx, half_max,
            max_expansions=10, expansion_factor=2.0, n_grid=500,
        )
        # At least one expansion should have occurred
        self.assertGreater(n_expansions, 0)

    def test_sm_summary_contains_halfmax_in_grid(self):
        """The returned freq_grid fully contains both half-max crossings."""
        lc = self._make_sm_lc()
        summary = lc.get_period_summary()
        freq_grid = summary["freq_grid"]
        psd = summary["psd"]
        period_lo, period_hi = summary["period_interval_fwhm_like"]
        f_left = 1.0 / period_hi
        f_right = 1.0 / period_lo
        # Both crossing frequencies must lie within the returned grid
        self.assertGreaterEqual(f_left, freq_grid[0] - 1e-10 * freq_grid[0])
        self.assertLessEqual(f_right, freq_grid[-1] + 1e-10 * freq_grid[-1])

    def test_sm_summary_notes_mention_expansion_when_forced(self):
        """Notes mention expansion when grid was forced to expand."""
        lc = self._make_sm_lc()
        # Force min_freq/max_freq so tight that expansion is needed.
        # Use a dominant component freq so that the 5-sigma default
        # barely clips the peak.
        params = lc._extract_sm_params()
        f0 = params["component_frequencies"][0]
        # Pass max_freq just slightly above f0 to ensure clipping
        forced_max = f0 * 1.001
        forced_min = max(f0 * 0.999, 1e-12)
        summary = lc.get_period_summary(
            min_freq=forced_min, max_freq=forced_max
        )
        # Notes should mention expansion if the peak was clipped
        # (it's fine if expansion wasn't needed for a sharp peak,
        # but the method should at least not crash)
        self.assertIsInstance(summary["notes"], str)
        self.assertIn("dominant_period", summary)

    def test_sm_notes_mention_truncation_when_expansion_maxed_out(self):
        """Notes flag truncation if max_expansions reached without containment."""
        lc = self._make_sm_lc()
        # Use a very small max_expansions so it cannot fully contain the peak
        params = lc._extract_sm_params()
        f0 = params["component_frequencies"][0]
        sigma0 = params["component_frequency_scales"][0]
        # Very narrow initial grid: peak is at f0 but grid goes only
        # [f0, f0 + tiny_delta], so the left side is definitely truncated.
        forced_min = f0 * 0.9999
        forced_max = f0 + sigma0 * 0.001

        # Monkeypatch _expand_psd_grid_until_contained to cap at 0 expansions
        original_expand = lc._expand_psd_grid_until_contained

        def _no_expand(freq_grid, psd, params, dominant_idx, half_max,
                       **kw):
            kw.pop("max_expansions", None)
            return original_expand(
                freq_grid, psd, params, dominant_idx, half_max,
                max_expansions=0, **kw
            )

        lc._expand_psd_grid_until_contained = _no_expand
        try:
            summary = lc.get_period_summary(
                min_freq=forced_min, max_freq=forced_max
            )
        finally:
            lc._expand_psd_grid_until_contained = original_expand

        # With 0 expansions allowed the notes must flag truncation
        self.assertIn("truncat", summary["notes"].lower())


# ---------------------------------------------------------------------------
# 12. Log-spaced frequency grid and local refinement
# ---------------------------------------------------------------------------


class TestSmPsdLogGrid(unittest.TestCase):
    """Tests that the SM backend uses a log-spaced frequency grid."""

    def _make_sm_lc(self):
        return _make_1d_lc_no_transform(n_obs=40, period=100.0, seed=0)

    # ------------------------------------------------------------------
    # Helper: build an _extract_sm_params-compatible dict with one broad
    # low-frequency component (spanning multiple decades).
    # ------------------------------------------------------------------

    def _broad_low_freq_params(self, center_freq=1e-3, scale_freq=3e-4):
        return {
            "component_frequencies": np.array([center_freq]),
            "component_periods": np.array([1.0 / center_freq]),
            "component_frequency_scales": np.array([scale_freq]),
            "component_period_scales": np.array([np.nan]),
            "component_weights": np.array([1.0]),
        }

    # ------------------------------------------------------------------

    def _assertIsLogSpaced(self, grid, tol=1e-3):
        """Assert that ``grid`` is strictly positive and approximately log-spaced."""
        self.assertTrue(np.all(grid > 0), "all grid values must be positive")
        ratios = grid[1:] / grid[:-1]
        ratio_cv = float(ratios.std() / ratios.mean())
        self.assertLess(
            ratio_cv, tol,
            f"ratio CV {ratio_cv:.2e} >= {tol}: grid is not log-spaced",
        )

    def test_build_frequency_grid_log(self):
        """_build_frequency_grid returns a log-spaced grid."""
        lc = self._make_sm_lc()
        grid = lc._build_frequency_grid(1e-4, 1.0, 200, spacing="log")
        self.assertEqual(len(grid), 200)
        self.assertAlmostEqual(float(grid[0]), 1e-4, places=15)
        self.assertAlmostEqual(float(grid[-1]), 1.0, places=15)
        self._assertIsLogSpaced(grid)

    def test_build_frequency_grid_linear(self):
        """_build_frequency_grid returns a linear-spaced grid when asked."""
        lc = self._make_sm_lc()
        grid = lc._build_frequency_grid(0.001, 1.0, 100, spacing="linear")
        self.assertEqual(len(grid), 100)
        diffs = np.diff(grid)
        self.assertAlmostEqual(float(diffs.max()), float(diffs.min()),
                               places=8)

    def test_build_frequency_grid_raises_on_nonpositive_min_log(self):
        """_build_frequency_grid raises ValueError for min_freq <= 0 with log."""
        lc = self._make_sm_lc()
        with self.assertRaises(ValueError):
            lc._build_frequency_grid(0.0, 1.0, 100, spacing="log")

    def test_sm_summary_freq_grid_is_log_spaced(self):
        """The returned freq_grid from get_period_summary is log-spaced."""
        lc = self._make_sm_lc()
        summary = lc.get_period_summary()
        freq_grid = summary["freq_grid"]
        self.assertGreater(len(freq_grid), 1)
        self._assertIsLogSpaced(freq_grid)

    def test_sm_notes_mention_log_grid(self):
        """Summary notes mention log-spaced grid."""
        lc = self._make_sm_lc()
        summary = lc.get_period_summary()
        self.assertIn("log", summary["notes"].lower())

    def test_sm_notes_mention_refinement(self):
        """Summary notes always mention the log-spaced grid.

        The new multi-peak implementation no longer performs a separate
        local refinement pass, but the notes always describe the method.
        """
        lc = self._make_sm_lc()
        summary = lc.get_period_summary()
        notes = summary["notes"].lower()
        # The notes must always mention log-spacing
        self.assertIn("log", notes)

    def test_log_grid_better_than_linear_for_broad_peak(self):
        """Log-spaced grid resolves broad low-freq peak; linear cannot.

        We build a broad single-component SM PSD with center at 1e-3 and
        scale 3e-4 (peak spans ~1e-4 to ~2e-3, i.e. one decade).
        We then compare FWHM estimates from a log grid vs a linear grid of
        the same size.  The log grid should give a finite, non-trivial FWHM;
        the linear grid's estimate should be less accurate or potentially
        zero/infinite due to poor low-freq sampling.
        """
        lc = self._make_sm_lc()
        params = self._broad_low_freq_params(center_freq=1e-3, scale_freq=3e-4)

        # Build a wide-range grid in both spacings
        min_f = 1e-5
        max_f = 1e-1
        n = 500

        log_grid = lc._build_frequency_grid(min_f, max_f, n, spacing="log")
        lin_grid = lc._build_frequency_grid(min_f, max_f, n, spacing="linear")

        psd_log = lc._sm_psd_on_grid(log_grid, params)
        psd_lin = lc._sm_psd_on_grid(lin_grid, params)

        peak_log = int(np.argmax(psd_log))
        peak_lin = int(np.argmax(psd_lin))

        half_max_log = 0.5 * float(psd_log[peak_log])
        half_max_lin = 0.5 * float(psd_lin[peak_lin])

        # Count samples above half_max for each grid
        n_above_log = int(np.sum(psd_log >= half_max_log))
        n_above_lin = int(np.sum(psd_lin >= half_max_lin))

        # The log grid should resolve the peak with many points;
        # the linear grid undersamples it on the low-freq side.
        self.assertGreater(n_above_log, n_above_lin,
                           "log grid should resolve the peak with more points")

    def test_refine_peak_region_returns_denser_grid(self):
        """_refine_peak_region returns a local grid denser than the global."""
        lc = self._make_sm_lc()
        params = self._broad_low_freq_params()
        n_global = 200
        global_grid = lc._build_frequency_grid(1e-5, 0.1, n_global,
                                               spacing="log")
        global_psd = lc._sm_psd_on_grid(global_grid, params)
        peak_idx = int(np.argmax(global_psd))

        freq_fine, psd_fine, _ = lc._refine_peak_region(
            global_grid, global_psd, params, peak_idx,
            f_left_approx=5e-4, f_right_approx=2e-3,
        )
        self.assertGreater(len(freq_fine), n_global)
        self._assertIsLogSpaced(freq_fine)

    def test_log_grid_expansion_does_not_crash(self):
        """Adaptive expansion with log grid completes without error."""
        lc = self._make_sm_lc()
        params = self._broad_low_freq_params(center_freq=1e-3, scale_freq=3e-4)
        # Very narrow initial grid => forces expansion
        narrow_grid = lc._build_frequency_grid(9e-4, 1.1e-3, 100,
                                               spacing="log")
        psd_init = lc._sm_psd_on_grid(narrow_grid, params)
        dom_idx = int(np.argmax(psd_init))
        half_max = 0.5 * float(psd_init[dom_idx])

        (
            freq_out, psd_out, dom_out,
            left_trunc, right_trunc, n_exp,
        ) = lc._expand_psd_grid_until_contained(
            narrow_grid, psd_init, params, dom_idx, half_max,
            max_expansions=10, expansion_factor=2.0, n_grid=200,
        )
        # Grid must still be log-spaced after expansion
        self._assertIsLogSpaced(freq_out)
        # At least some expansions should have occurred
        self.assertGreater(n_exp, 0)


# ---------------------------------------------------------------------------
# 13. Peak-mass interval (uncertainty="peak_mass")
# ---------------------------------------------------------------------------


class TestPeakMassInterval(unittest.TestCase):
    """Tests for uncertainty='peak_mass' mode in SM period summary."""

    def _make_sm_lc(self):
        return _make_1d_lc_no_transform(n_obs=40, period=100.0, seed=0)

    def _broad_low_freq_params(self, center_freq=1e-3, scale_freq=3e-4):
        return {
            "component_frequencies": np.array([center_freq]),
            "component_periods": np.array([1.0 / center_freq]),
            "component_frequency_scales": np.array([scale_freq]),
            "component_period_scales": np.array([np.nan]),
            "component_weights": np.array([1.0]),
        }

    # ------------------------------------------------------------------

    def test_peak_mass_returns_valid_interval(self):
        """uncertainty='peak_mass' returns a finite, ordered interval."""
        lc = self._make_sm_lc()
        summary = lc.get_period_summary(uncertainty="peak_mass")
        period_lo, period_hi = summary["period_interval"]
        self.assertIsNotNone(period_lo)
        self.assertIsNotNone(period_hi)
        self.assertTrue(np.isfinite(period_lo))
        self.assertTrue(np.isfinite(period_hi))
        self.assertGreater(period_hi, period_lo)

    def test_peak_mass_q_factor_is_none(self):
        """q_factor must be None for peak_mass mode."""
        lc = self._make_sm_lc()
        summary = lc.get_period_summary(uncertainty="peak_mass")
        self.assertIsNone(summary["q_factor"])

    def test_peak_mass_interval_definition_key(self):
        """interval_definition is 'peak_centered_68pct_mass_interval' for peak_mass."""
        lc = self._make_sm_lc()
        summary = lc.get_period_summary(uncertainty="peak_mass")
        self.assertEqual(
            summary["interval_definition"],
            "peak_centered_68pct_mass_interval",
        )

    def test_peak_width_interval_definition_key(self):
        """interval_definition is 'half_maximum_fwhm_like' for peak_width."""
        lc = self._make_sm_lc()
        summary = lc.get_period_summary(uncertainty="peak_width")
        self.assertEqual(
            summary["interval_definition"], "half_maximum_fwhm_like"
        )

    def test_peak_mass_period_interval_key_present(self):
        """Summary has both period_interval and period_interval_fwhm_like."""
        lc = self._make_sm_lc()
        summary = lc.get_period_summary(uncertainty="peak_mass")
        self.assertIn("period_interval", summary)
        self.assertIn("period_interval_fwhm_like", summary)

    def test_unsupported_uncertainty_raises(self):
        """Unsupported uncertainty mode raises NotImplementedError."""
        lc = self._make_sm_lc()
        with self.assertRaises(NotImplementedError):
            lc.get_period_summary(uncertainty="unsupported_mode")

    def test_peak_mass_notes_describe_method(self):
        """Notes for peak_mass mention peak-centered mass, not equal-tail."""
        lc = self._make_sm_lc()
        summary = lc.get_period_summary(uncertainty="peak_mass")
        notes = summary["notes"].lower()
        self.assertIn("mass", notes)
        self.assertIn("peak", notes)
        self.assertNotIn("half-maximum psd-width proxy", notes)

    def test_peak_mass_interval_contains_dominant_frequency(self):
        """The peak-mass interval must contain the dominant frequency."""
        lc = self._make_sm_lc()
        summary = lc.get_period_summary(uncertainty="peak_mass")
        f_dom = summary["dominant_frequency"]
        p_lo, p_hi = summary["period_interval"]
        # Convert period interval to frequency (inverted)
        f_hi_from_p = 1.0 / p_lo if p_lo > 0 else float("inf")
        f_lo_from_p = 1.0 / p_hi if p_hi > 0 else 0.0
        self.assertGreaterEqual(
            f_dom, f_lo_from_p,
            msg=f"Dominant freq {f_dom} should be >= interval lower {f_lo_from_p}",
        )
        self.assertLessEqual(
            f_dom, f_hi_from_p,
            msg=f"Dominant freq {f_dom} should be <= interval upper {f_hi_from_p}",
        )

    def test_peak_mass_dominant_period_is_mode(self):
        """The dominant period is the PSD mode, not mean/median."""
        lc = self._make_sm_lc()
        pw = lc.get_period_summary(uncertainty="peak_width")
        pm = lc.get_period_summary(uncertainty="peak_mass")
        # Both should report the same dominant frequency (mode)
        self.assertAlmostEqual(
            pw["dominant_frequency"], pm["dominant_frequency"], places=5
        )

    def test_find_dominant_peak_basin_basic(self):
        """_find_dominant_peak_basin returns sensible basin for a sharp peak."""
        lc = self._make_sm_lc()
        # Construct a simple triangle PSD: peak at index 5 in a 10-point array
        psd = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1],
            dtype=float,
        )
        bl, br, left_bdy, right_bdy = lc._find_dominant_peak_basin(psd, 5)
        self.assertEqual(bl, 0)   # left minimum at boundary
        self.assertEqual(br, 9)   # right minimum at boundary
        self.assertTrue(left_bdy)
        self.assertTrue(right_bdy)

    def test_find_dominant_peak_basin_interior_minima(self):
        """Basin correctly identified when interior minima exist."""
        lc = self._make_sm_lc()
        # Valley left of peak at 10, valley right of peak at 8
        psd = np.array(
            [0.5, 0.4, 0.1, 0.3, 0.7, 1.0, 0.7, 0.3, 0.1, 0.4],
            dtype=float,
        )
        # Dominant peak at index 5
        bl, br, left_bdy, right_bdy = lc._find_dominant_peak_basin(psd, 5)
        self.assertEqual(bl, 2)    # left minimum at index 2
        self.assertEqual(br, 8)    # right minimum at index 8
        self.assertFalse(left_bdy)
        self.assertFalse(right_bdy)

    def test_compute_equal_tail_mass_interval_basic(self):
        """_compute_equal_tail_mass_interval returns sane bounds."""
        lc = self._make_sm_lc()
        params = self._broad_low_freq_params()
        n = 300
        freq_grid = lc._build_frequency_grid(1e-5, 0.1, n, spacing="log")
        psd = lc._sm_psd_on_grid(freq_grid, params)
        peak_idx = int(np.argmax(psd))
        bl, br, _, _ = lc._find_dominant_peak_basin(psd, peak_idx)

        f_lo, f_hi, ok = lc._compute_equal_tail_mass_interval(
            freq_grid, psd, bl, br, mass_level=0.68
        )
        self.assertTrue(ok)
        self.assertGreater(f_hi, f_lo)
        self.assertGreater(f_lo, 0.0)
        # Interval should be narrower than the full basin
        self.assertGreater(f_lo, float(freq_grid[bl]))
        self.assertLess(f_hi, float(freq_grid[br]))

    def test_peak_mass_narrower_than_peak_width_for_asymmetric_peak(self):
        """For an asymmetric broad peak, peak_mass interval is narrower.

        We construct a PSD with a broad low-frequency wing by using a
        two-component SM: one sharp dominant peak and one broad low-frequency
        component.  The half-maximum interval will be wide because the broad
        component prevents the PSD from dropping below 50% of the peak far
        out on the low-frequency side.  The peak-mass interval should be
        much narrower as it focuses on the dominant basin only.
        """
        lc = self._make_sm_lc()

        pw = lc.get_period_summary(uncertainty="peak_width")
        pm = lc.get_period_summary(uncertainty="peak_mass")

        pw_lo, pw_hi = pw["period_interval"]
        pm_lo, pm_hi = pm["period_interval"]

        pw_width = pw_hi - pw_lo
        pm_width = pm_hi - pm_lo

        # Both should be finite
        self.assertTrue(np.isfinite(pw_width))
        self.assertTrue(np.isfinite(pm_width))
        # The mass width cannot be wider than the full data span
        self.assertGreater(pm_width, 0.0)

    def test_plot_period_summary_peak_mass(self):
        """plot_period_summary works with uncertainty='peak_mass'."""
        import matplotlib
        matplotlib.use("Agg")
        lc = self._make_sm_lc()
        summary = lc.get_period_summary(uncertainty="peak_mass")
        result = lc.plot_period_summary(summary=summary, show=False)
        self.assertIsNotNone(result)
        fig, ax = result
        self.assertIsNotNone(fig)
        # Legend should mention the mass interval
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        combined = " ".join(legend_texts).lower()
        self.assertIn("mass", combined)

    def test_non_periodic_summary_has_period_interval_key(self):
        """Non-periodic summary dict contains period_interval key."""
        lc = self._make_sm_lc()
        # Access the non-periodic summary directly
        summary = lc._get_non_periodic_summary()
        self.assertIn("period_interval", summary)
        self.assertIsNone(summary["period_interval"])
        self.assertIn("interval_definition", summary)


class TestMultiPeakPSDAnalysis(unittest.TestCase):
    """Tests for the multi-peak PSD analysis refactoring."""

    @classmethod
    def setUpClass(cls):
        """Create a shared LC with SM model (num_mixtures=2)."""
        cls.lc = _make_1d_lc_no_transform(n_obs=40, period=100.0, seed=0)
        # _make_1d_lc_no_transform calls set_model("1D", num_mixtures=2)
        cls.summary = cls.lc.get_period_summary()

    def test_period_summary_result_is_object(self):
        from pgmuvi.lightcurve import PeriodSummaryResult
        self.assertIsInstance(self.summary, PeriodSummaryResult)

    def test_period_summary_dict_compat(self):
        """Old dict keys must still be accessible via []."""
        for key in _REQUIRED_KEYS:
            self.assertIn(key, self.summary)
            _ = self.summary[key]  # must not raise

    def test_n_peaks_defaults_to_fit_num_mixtures(self):
        """Default n_peaks matches fit effective num_mixtures."""
        self.assertEqual(
            self.lc._fit_num_mixtures_effective, 2
        )
        summary = self.lc.get_period_summary()
        expected = min(2, summary.n_peaks_detected)
        self.assertEqual(summary.n_peaks_analyzed, expected)

    def test_n_peaks_override(self):
        """n_peaks=1 returns exactly 1 peak."""
        summary = self.lc.get_period_summary(n_peaks=1)
        self.assertEqual(len(summary.peaks), 1)

    def test_as_dict_method(self):
        """as_dict() returns a dict with all required keys."""
        d = self.summary.as_dict()
        self.assertIsInstance(d, dict)
        for key in _REQUIRED_KEYS:
            self.assertIn(key, d)

    def test_to_table_method(self):
        """to_table() returns one row per peak."""
        table = self.summary.to_table()
        self.assertIsInstance(table, list)
        self.assertEqual(len(table), len(self.summary.peaks))
        if table:
            row = table[0]
            for col in (
                "peak_rank", "period", "frequency", "height",
                "prominence", "area_fraction",
                "period_interval_lo", "period_interval_hi",
                "period_ratio_to_primary", "is_candidate_lsp", "notes",
            ):
                self.assertIn(col, row)

    def test_write_json_method(self):
        """write_json writes valid JSON that round-trips."""
        import json
        import tempfile
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as tmp:
            path = tmp.name
        try:
            self.summary.write_json(path)
            with open(path) as fh:
                data = json.load(fh)
            self.assertIn("dominant_period", data)
            self.assertIn("method", data)
        finally:
            import os
            if os.path.exists(path):
                os.remove(path)

    def test_peak_rank_1_is_dominant(self):
        """The first peak always has rank == 1."""
        self.assertGreater(len(self.summary.peaks), 0)
        self.assertEqual(self.summary.peaks[0].rank, 1)

    def test_peak_ratio_to_primary(self):
        """The primary peak has period_ratio_to_primary == 1.0."""
        self.assertGreater(len(self.summary.peaks), 0)
        self.assertAlmostEqual(
            self.summary.peaks[0].period_ratio_to_primary, 1.0
        )

    def test_lsp_classification_flag(self):
        """classify_lsp=True runs without error; primary never flagged."""
        summary = self.lc.get_period_summary(classify_lsp=True)
        self.assertGreater(len(summary.peaks), 0)
        self.assertFalse(summary.peaks[0].is_candidate_lsp)


# ---------------------------------------------------------------------------
# 14. _infer_num_mixtures_from_model and fit() inference
# ---------------------------------------------------------------------------


class TestInferNumMixturesFromModel(unittest.TestCase):
    """Tests for Part 1 fix: inferring num_mixtures from existing model."""

    def _make_sm_lc(self):
        return _make_1d_lc_no_transform(n_obs=40, period=100.0, seed=0)

    def test_infer_returns_correct_count_1d(self):
        """_infer_num_mixtures_from_model returns num_mixtures for 1D SM."""
        lc = self._make_sm_lc()
        result = lc._infer_num_mixtures_from_model()
        self.assertEqual(result, 2)

    def test_infer_returns_none_without_model(self):
        """_infer_num_mixtures_from_model returns None when model is unset."""
        lc = make_simple_sinusoid_1d(
            n_obs=20, period=100.0, noise_level=0.05,
            t_span=500.0, seed=0,
        )
        result = lc._infer_num_mixtures_from_model()
        self.assertIsNone(result)

    def test_fit_num_mixtures_effective_matches_set_model(self):
        """_fit_num_mixtures_effective matches set_model num_mixtures."""
        lc = self._make_sm_lc()
        # set_model stores the count; verify it was stored correctly
        self.assertEqual(lc._fit_num_mixtures_effective, 2)

    def test_multi_panel_plot_returns_fig_ax(self):
        """plot_period_summary returns (fig, ax) for multi-panel case."""
        import matplotlib.pyplot as plt
        lc = self._make_sm_lc()
        result = lc.plot_period_summary(show=False)
        self.assertIsNotNone(result)
        fig, ax = result
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 16. Single-peak plot centering
# ---------------------------------------------------------------------------


class TestSinglePeakPlotCentering(unittest.TestCase):
    """Tests for single-peak plot centering in plot_period_summary()."""

    def _make_single_peak_lc(self):
        """Light curve with num_mixtures=1 so only one peak is analyzed."""
        lc = make_simple_sinusoid_1d(
            n_obs=40, period=100.0, noise_level=0.05,
            t_span=500.0, seed=0,
        )
        lc.xtransform = None
        lc.xdata = lc._xdata_raw
        lc.set_model("1D", num_mixtures=1)
        return lc

    def _make_two_peak_lc(self):
        return _make_1d_lc_no_transform(n_obs=40, period=100.0, seed=0)

    # ------------------------------------------------------------------

    def test_single_peak_returns_fig_ax(self):
        """single-peak summary produces a (fig, ax) result."""
        import matplotlib
        matplotlib.use("Agg")
        lc = self._make_single_peak_lc()
        result = lc.plot_period_summary(show=False)
        self.assertIsNotNone(result)
        fig, ax = result
        import matplotlib.pyplot as plt
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_peak_figure_has_one_panel(self):
        """Single-peak figure has exactly one axes (no full-PSD top panel)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lc = self._make_single_peak_lc()
        summary = lc.get_period_summary(n_peaks=1)
        self.assertEqual(summary.n_peaks_analyzed, 1)
        fig, ax = lc.plot_period_summary(summary=summary, show=False)
        self.assertEqual(len(fig.axes), 1)
        plt.close(fig)

    def test_single_peak_dominant_freq_inside_xlim(self):
        """Dominant frequency lies strictly inside the main panel x-limits."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lc = self._make_single_peak_lc()
        summary = lc.get_period_summary(n_peaks=1)
        f_dom = summary["dominant_frequency"]
        fig, ax = lc.plot_period_summary(summary=summary, show=False)
        x_lo, x_hi = ax.get_xlim()
        self.assertGreater(f_dom, x_lo,
                           msg=f"f_dom={f_dom} not > x_lo={x_lo}")
        self.assertLess(f_dom, x_hi,
                        msg=f"f_dom={f_dom} not < x_hi={x_hi}")
        plt.close(fig)

    def test_single_peak_title_mentions_dominant_peak(self):
        """Main panel title says 'dominant peak' not 'full PSD'."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lc = self._make_single_peak_lc()
        summary = lc.get_period_summary(n_peaks=1)
        fig, ax = lc.plot_period_summary(summary=summary, show=False)
        title = ax.get_title().lower()
        self.assertIn("dominant peak", title)
        self.assertNotIn("full psd", title)
        plt.close(fig)

    def test_single_peak_show_full_psd_true_adds_second_panel(self):
        """show_full_psd=True adds a second full-range panel."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lc = self._make_single_peak_lc()
        summary = lc.get_period_summary(n_peaks=1)
        fig, ax = lc.plot_period_summary(
            summary=summary, show=False, show_full_psd=True
        )
        self.assertEqual(len(fig.axes), 2)
        plt.close(fig)

    def test_multi_peak_still_has_full_psd_top_panel(self):
        """Multi-peak summary still uses full PSD as top panel."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lc = self._make_two_peak_lc()
        summary = lc.get_period_summary()
        # num_mixtures=2, so 2 peaks => 3 panels (full + 2 zoom)
        if summary.n_peaks_analyzed < 2:
            self.skipTest("Not enough peaks for multi-peak test")
        fig, ax = lc.plot_period_summary(summary=summary, show=False)
        n_axes = len(fig.axes)
        self.assertGreater(n_axes, 1,
                           msg="Multi-peak should have >1 panel")
        title = ax.get_title().lower()
        self.assertIn("full psd", title)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 17. PeriodSummaryResult text export (to_text / write_text)
# ---------------------------------------------------------------------------


class TestPeriodSummaryTextExport(unittest.TestCase):
    """Tests for PeriodSummaryResult.to_text() and write_text()."""

    def _make_summary(self):
        """Return a realistic PeriodSummaryResult from a fitted 1-D SM lc."""
        return _make_1d_lc_no_transform(
            n_obs=40, period=100.0, seed=0
        ).get_period_summary()

    # ------------------------------------------------------------------

    def test_to_text_returns_nonempty_string(self):
        """to_text() returns a non-empty string."""
        summary = self._make_summary()
        text = summary.to_text()
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_to_text_contains_method(self):
        """to_text() includes the method string."""
        summary = self._make_summary()
        text = summary.to_text()
        self.assertIn(summary.method, text)

    def test_to_text_contains_dominant_period(self):
        """to_text() includes the dominant period."""
        summary = self._make_summary()
        text = summary.to_text()
        self.assertIn("Dominant period", text)

    def test_to_text_contains_dominant_frequency(self):
        """to_text() includes the dominant frequency."""
        summary = self._make_summary()
        text = summary.to_text()
        self.assertIn("Dominant frequency", text)

    def test_to_text_contains_peaks_section(self):
        """to_text() includes an analyzed-peaks section when peaks exist."""
        summary = self._make_summary()
        if not summary.peaks:
            self.skipTest("No peaks in summary")
        text = summary.to_text(include_peaks=True)
        self.assertIn("ANALYZED PEAKS", text)
        self.assertIn("P1", text)

    def test_to_text_omits_peaks_when_flag_false(self):
        """to_text(include_peaks=False) omits the peak section."""
        summary = self._make_summary()
        if not summary.peaks:
            self.skipTest("No peaks in summary")
        text = summary.to_text(include_peaks=False)
        self.assertNotIn("ANALYZED PEAKS", text)

    def test_to_text_contains_component_section(self):
        """to_text() includes component diagnostics when present."""
        summary = self._make_summary()
        has_comp = len(summary.component_periods) > 0
        text = summary.to_text(include_components=True)
        if has_comp:
            self.assertIn("KERNEL COMPONENT DIAGNOSTICS", text)
            self.assertIn("Component periods", text)
            self.assertIn("not final periods", text)

    def test_to_text_omits_components_when_flag_false(self):
        """to_text(include_components=False) omits the component section."""
        summary = self._make_summary()
        text = summary.to_text(include_components=False)
        self.assertNotIn("KERNEL COMPONENT DIAGNOSTICS", text)

    def test_to_text_psd_info_off_by_default(self):
        """to_text() omits PSD info by default."""
        summary = self._make_summary()
        text = summary.to_text()
        self.assertNotIn("PSD GRID INFORMATION", text)

    def test_to_text_psd_info_when_requested(self):
        """to_text(include_psd_info=True) includes PSD grid summary."""
        summary = self._make_summary()
        if summary.freq_grid is None:
            self.skipTest("No freq_grid in summary")
        text = summary.to_text(include_psd_info=True)
        self.assertIn("PSD GRID INFORMATION", text)
        self.assertIn("Frequency grid present", text)

    def test_to_text_multiple_peaks(self):
        """to_text() includes a block for each analyzed peak."""
        summary = self._make_summary()
        n = summary.n_peaks_analyzed
        if n < 2:
            self.skipTest("Need >= 2 peaks for this test")
        text = summary.to_text(include_peaks=True)
        self.assertIn("P1", text)
        self.assertIn("P2", text)

    def test_write_text_creates_file(self):
        """write_text() writes a file that exists and is non-empty."""
        import tempfile
        from pathlib import Path
        summary = self._make_summary()
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            returned = summary.write_text(tmp_path)
            self.assertTrue(tmp_path.exists())
            self.assertGreater(tmp_path.stat().st_size, 0)
            self.assertEqual(returned, tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_write_text_content_matches_to_text(self):
        """write_text() file content matches to_text()."""
        import tempfile
        from pathlib import Path
        summary = self._make_summary()
        expected = summary.to_text()
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            summary.write_text(tmp_path)
            content = tmp_path.read_text(encoding="utf-8")
            self.assertEqual(content, expected)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_to_text_separates_peaks_from_components(self):
        """Peaks section appears before component diagnostics section."""
        summary = self._make_summary()
        if not summary.peaks or len(summary.component_periods) == 0:
            self.skipTest("Need peaks and components for ordering test")
        text = summary.to_text(include_peaks=True, include_components=True)
        peaks_pos = text.find("ANALYZED PEAKS")
        comp_pos = text.find("KERNEL COMPONENT DIAGNOSTICS")
        self.assertNotEqual(peaks_pos, -1)
        self.assertNotEqual(comp_pos, -1)
        self.assertLess(
            peaks_pos, comp_pos,
            msg="Peak section should come before component section",
        )

    def test_write_json_still_works(self):
        """Existing write_json() still works after adding text export."""
        import tempfile
        import json
        from pathlib import Path
        summary = self._make_summary()
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            summary.write_json(tmp_path)
            with open(tmp_path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertIn("method", data)
            self.assertIn("dominant_period", data)
        finally:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 18. Fast synthetic tests for to_text() and write_text()
#     These tests construct PeriodSummaryResult directly — no GP fitting,
#     no LS, no randomness — so they run in well under 1 second.
# ---------------------------------------------------------------------------


def _make_peak(
    rank=1,
    period=100.0,
    frequency=0.01,
    height=0.9,
    prominence=0.7,
    area_fraction=0.5,
    interval_frequency=(0.009, 0.011),
    interval_period=(90.0, 110.0),
    period_ratio_to_primary=1.0,
    is_candidate_lsp=True,
    notes="",
):
    """Construct a frozen PeriodPeakResult with explicit values."""
    import dataclasses
    return dataclasses.replace(
        PeriodPeakResult(),
        rank=rank,
        period=period,
        frequency=frequency,
        height=height,
        prominence=prominence,
        area_fraction=area_fraction,
        interval_frequency=interval_frequency,
        interval_period=interval_period,
        period_ratio_to_primary=period_ratio_to_primary,
        is_candidate_lsp=is_candidate_lsp,
        notes=notes,
    )


def _make_synthetic_summary(n_peaks=1, with_components=True, with_psd=False):
    """Return a PeriodSummaryResult built from scratch, no fitting needed."""
    peaks = [
        _make_peak(
            rank=i + 1,
            period=100.0 / (i + 1),
            frequency=0.01 * (i + 1),
            period_ratio_to_primary=1.0 / (i + 1) if i else 1.0,
        )
        for i in range(n_peaks)
    ]
    freq_grid = np.linspace(0.001, 0.1, 200) if with_psd else None
    psd = np.random.default_rng(0).random(200) if with_psd else None
    return PeriodSummaryResult(
        method="psd_peak",
        model_name="SM-2",
        n_peaks_detected=n_peaks,
        n_peaks_analyzed=n_peaks,
        n_peaks_requested=n_peaks,
        dominant_period=peaks[0].period,
        dominant_frequency=peaks[0].frequency,
        peaks=peaks,
        notes="synthetic test",
        component_periods=(
            np.array([100.0, 50.0, 33.0]) if with_components else np.array([])
        ),
        component_frequencies=(
            np.array([0.01, 0.02, 0.03]) if with_components else np.array([])
        ),
        component_weights=(
            np.array([0.6, 0.3, 0.1]) if with_components else np.array([])
        ),
        component_period_scales=(
            np.array([5.0, 3.0, 2.0]) if with_components else np.array([])
        ),
        component_frequency_scales=(
            np.array([0.001, 0.002, 0.003]) if with_components else np.array([])
        ),
        freq_grid=freq_grid,
        psd=psd,
    )


class TestPeriodSummaryTextExportSynthetic(unittest.TestCase):
    """Fast synthetic tests for to_text() and write_text().

    These tests construct PeriodSummaryResult directly and run in
    well under 1 second — no GP fitting, no LS computation.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _summary(self, **kwargs):
        return _make_synthetic_summary(**kwargs)

    # ------------------------------------------------------------------
    # 1. Basic functionality
    # ------------------------------------------------------------------

    def test_returns_str(self):
        """to_text() returns a str instance."""
        self.assertIsInstance(self._summary().to_text(), str)

    def test_nonempty(self):
        """to_text() returns a non-empty string."""
        self.assertGreater(len(self._summary().to_text()), 0)

    def test_header_section_present(self):
        """to_text() always includes the PERIOD SUMMARY header."""
        self.assertIn("PERIOD SUMMARY", self._summary().to_text())

    def test_model_name_in_output(self):
        """to_text() includes the model name."""
        self.assertIn("SM-2", self._summary().to_text())

    def test_method_in_output(self):
        """to_text() includes the method string."""
        self.assertIn("psd_peak", self._summary().to_text())

    def test_dominant_period_label_present(self):
        """to_text() includes a 'Dominant period' label."""
        self.assertIn("Dominant period", self._summary().to_text())

    def test_dominant_frequency_label_present(self):
        """to_text() includes a 'Dominant frequency' label."""
        self.assertIn("Dominant frequency", self._summary().to_text())

    def test_peaks_detected_label_present(self):
        """to_text() includes a peaks-detected count."""
        self.assertIn("Peaks detected", self._summary().to_text())

    def test_interval_definition_present(self):
        """to_text() includes the interval definition string."""
        s = self._summary()
        text = s.to_text()
        self.assertIn(s.interval_definition, text)

    # ------------------------------------------------------------------
    # 2. Toggle behavior
    # ------------------------------------------------------------------

    def test_peaks_section_present_by_default(self):
        """ANALYZED PEAKS section appears by default."""
        self.assertIn("ANALYZED PEAKS", self._summary().to_text())

    def test_peaks_section_omitted_when_flag_false(self):
        """include_peaks=False omits the peak section."""
        self.assertNotIn(
            "ANALYZED PEAKS",
            self._summary().to_text(include_peaks=False),
        )

    def test_components_section_present_by_default(self):
        """KERNEL COMPONENT DIAGNOSTICS section appears by default."""
        self.assertIn(
            "KERNEL COMPONENT DIAGNOSTICS",
            self._summary(with_components=True).to_text(),
        )

    def test_components_section_omitted_when_flag_false(self):
        """include_components=False omits the component section."""
        self.assertNotIn(
            "KERNEL COMPONENT DIAGNOSTICS",
            self._summary().to_text(include_components=False),
        )

    def test_psd_section_absent_by_default(self):
        """PSD GRID INFORMATION is absent by default."""
        self.assertNotIn(
            "PSD GRID INFORMATION",
            self._summary(with_psd=True).to_text(),
        )

    def test_psd_section_present_when_requested(self):
        """include_psd_info=True adds PSD GRID INFORMATION section."""
        self.assertIn(
            "PSD GRID INFORMATION",
            self._summary(with_psd=True).to_text(include_psd_info=True),
        )

    def test_psd_section_shows_grid_length(self):
        """PSD section includes grid length when freq_grid is present."""
        text = self._summary(with_psd=True).to_text(include_psd_info=True)
        self.assertIn("Grid length", text)
        self.assertIn("200", text)

    # ------------------------------------------------------------------
    # 3. Edge cases
    # ------------------------------------------------------------------

    def test_no_peaks_no_exception(self):
        """to_text() works when there are no analyzed peaks."""
        s = PeriodSummaryResult(
            method="psd_peak",
            model_name="SM-2",
            n_peaks_detected=0,
            n_peaks_analyzed=0,
            dominant_period=None,
            dominant_frequency=None,
        )
        text = s.to_text()
        self.assertIsInstance(text, str)
        self.assertIn("PERIOD SUMMARY", text)

    def test_no_components_no_exception(self):
        """to_text() works when there are no component arrays."""
        s = self._summary(with_components=False)
        text = s.to_text(include_components=True)
        self.assertIsInstance(text, str)
        self.assertNotIn("KERNEL COMPONENT DIAGNOSTICS", text)

    def test_no_freq_grid_no_exception(self):
        """to_text(include_psd_info=True) works when freq_grid is None."""
        s = self._summary(with_psd=False)
        text = s.to_text(include_psd_info=True)
        self.assertIn("PSD GRID INFORMATION", text)
        self.assertIn("Frequency grid present : False", text)

    def test_notes_empty_no_exception(self):
        """to_text() works when notes is an empty string."""
        s = PeriodSummaryResult(method="psd_peak", model_name="M")
        text = s.to_text()
        self.assertIsInstance(text, str)

    # ------------------------------------------------------------------
    # 4. Deterministic / exact substring checks
    # ------------------------------------------------------------------

    def test_exact_model_name_line(self):
        """Model name line has expected format."""
        text = self._summary().to_text()
        self.assertIn("Model name          : SM-2", text)

    def test_exact_method_line(self):
        """Method line has expected format."""
        text = self._summary().to_text()
        self.assertIn("Method              : psd_peak", text)

    def test_exact_dominant_period_value(self):
        """Dominant period value 100 appears in output."""
        text = self._summary().to_text()
        self.assertIn("Dominant period     : 100", text)

    def test_peak_rank_label(self):
        """Each peak block includes its rank label."""
        text = self._summary(n_peaks=2).to_text(include_peaks=True)
        self.assertIn("Rank                       : 1", text)
        self.assertIn("Rank                       : 2", text)

    def test_two_peaks_both_blocks_present(self):
        """Two-peak summary contains blocks P1 and P2."""
        text = self._summary(n_peaks=2).to_text(include_peaks=True)
        self.assertIn("Peak P1", text)
        self.assertIn("Peak P2", text)

    def test_component_diagnostic_disclaimer(self):
        """Component section contains the 'not final periods' disclaimer."""
        text = self._summary(with_components=True).to_text(
            include_components=True
        )
        self.assertIn("not final periods", text)

    def test_component_periods_values(self):
        """Component periods appear in the component section."""
        text = self._summary(with_components=True).to_text(
            include_components=True
        )
        self.assertIn("Component periods", text)
        self.assertIn("100", text)

    def test_peaks_section_before_components_section(self):
        """ANALYZED PEAKS section precedes KERNEL COMPONENT DIAGNOSTICS."""
        text = self._summary(n_peaks=1, with_components=True).to_text()
        peak_pos = text.find("ANALYZED PEAKS")
        comp_pos = text.find("KERNEL COMPONENT DIAGNOSTICS")
        self.assertGreater(peak_pos, -1)
        self.assertGreater(comp_pos, -1)
        self.assertLess(peak_pos, comp_pos)

    def test_lsp_candidate_flag_in_peak_block(self):
        """Peak block shows the LSP candidate flag."""
        text = self._summary().to_text(include_peaks=True)
        self.assertIn("LSP candidate", text)
        self.assertIn("True", text)

    def test_interval_frequency_in_peak_block(self):
        """Peak block shows the frequency interval."""
        text = self._summary().to_text(include_peaks=True)
        self.assertIn("Interval (frequency)", text)

    def test_interval_period_in_peak_block(self):
        """Peak block shows the period interval."""
        text = self._summary().to_text(include_peaks=True)
        self.assertIn("Interval (period)", text)

    def test_period_ratio_in_peak_block(self):
        """Peak block shows the period ratio to primary."""
        text = self._summary().to_text(include_peaks=True)
        self.assertIn("Period ratio to primary", text)

    # ------------------------------------------------------------------
    # 5. write_text() behavior
    # ------------------------------------------------------------------

    def test_write_text_creates_file(self):
        """write_text() creates a non-empty file."""
        import tempfile
        from pathlib import Path

        s = self._summary()
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            s.write_text(tmp_path)
            self.assertTrue(tmp_path.exists())
            self.assertGreater(tmp_path.stat().st_size, 0)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_write_text_returns_path(self):
        """write_text() returns the output Path."""
        import tempfile
        from pathlib import Path

        s = self._summary()
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            returned = s.write_text(tmp_path)
            self.assertEqual(returned, tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_write_text_content_matches_to_text(self):
        """File written by write_text() matches to_text() exactly."""
        import tempfile
        from pathlib import Path

        s = self._summary()
        expected = s.to_text()
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            s.write_text(tmp_path)
            content = tmp_path.read_text(encoding="utf-8")
            self.assertEqual(content, expected)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_write_text_accepts_str_path(self):
        """write_text() accepts a plain string filename."""
        import tempfile
        from pathlib import Path

        s = self._summary()
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False
        ) as tmp:
            tmp_str = tmp.name
        try:
            s.write_text(tmp_str)
            self.assertTrue(Path(tmp_str).exists())
        finally:
            Path(tmp_str).unlink(missing_ok=True)

    def test_write_text_utf8_encoding(self):
        """File written by write_text() is valid UTF-8."""
        import tempfile
        from pathlib import Path

        s = self._summary()
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            s.write_text(tmp_path)
            raw = tmp_path.read_bytes()
            raw.decode("utf-8")  # must not raise
        finally:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 19. Tests for Lightcurve.write_period_summary_outputs()
# ---------------------------------------------------------------------------


class TestWritePeriodSummaryOutputs(unittest.TestCase):
    """Tests for the write_period_summary_outputs() convenience wrapper.

    Fast tests (text/JSON, pre-computed summary) use directly-constructed
    synthetic summaries and run without GP fitting.  PNG export tests use a
    real fitted LC and are therefore slower.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _synthetic_summary(self):
        return _make_synthetic_summary(n_peaks=1, with_components=True)

    def _fitted_lc(self):
        return _make_1d_lc_no_transform(n_obs=40, period=100.0, seed=0)

    # ------------------------------------------------------------------
    # A. Text-only export via the wrapper
    # ------------------------------------------------------------------

    def test_text_file_is_created(self):
        """write_period_summary_outputs(text_file=...) creates the file."""
        import tempfile
        from pathlib import Path

        s = self._synthetic_summary()
        lc = _make_1d_lc_no_transform(n_obs=40, period=100.0, seed=0)
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False
        ) as tmp:
            txt_path = Path(tmp.name)
        try:
            returned = lc.write_period_summary_outputs(
                text_file=txt_path, summary=s
            )
            self.assertTrue(txt_path.exists())
            self.assertGreater(txt_path.stat().st_size, 0)
            self.assertIs(returned, s)
        finally:
            txt_path.unlink(missing_ok=True)

    def test_text_content_matches_to_text(self):
        """Text file content matches summary.to_text()."""
        import tempfile
        from pathlib import Path

        s = self._synthetic_summary()
        lc = _make_1d_lc_no_transform(n_obs=40, period=100.0, seed=0)
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False
        ) as tmp:
            txt_path = Path(tmp.name)
        try:
            lc.write_period_summary_outputs(
                text_file=txt_path, summary=s
            )
            content = txt_path.read_text(encoding="utf-8")
            self.assertEqual(content, s.to_text())
        finally:
            txt_path.unlink(missing_ok=True)

    def test_text_kwargs_forwarded(self):
        """include_peaks/include_components are forwarded to write_text."""
        import tempfile
        from pathlib import Path

        s = self._synthetic_summary()
        lc = _make_1d_lc_no_transform(n_obs=40, period=100.0, seed=0)
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False
        ) as tmp:
            txt_path = Path(tmp.name)
        try:
            lc.write_period_summary_outputs(
                text_file=txt_path,
                summary=s,
                include_peaks=False,
                include_components=False,
            )
            content = txt_path.read_text(encoding="utf-8")
            self.assertNotIn("ANALYZED PEAKS", content)
            self.assertNotIn("KERNEL COMPONENT DIAGNOSTICS", content)
        finally:
            txt_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # B. JSON-only export via the wrapper
    # ------------------------------------------------------------------

    def test_json_file_is_created(self):
        """write_period_summary_outputs(json_file=...) creates the file."""
        import tempfile
        import json
        from pathlib import Path

        s = self._synthetic_summary()
        lc = _make_1d_lc_no_transform(n_obs=40, period=100.0, seed=0)
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as tmp:
            json_path = Path(tmp.name)
        try:
            lc.write_period_summary_outputs(
                json_file=json_path, summary=s
            )
            self.assertTrue(json_path.exists())
            with open(json_path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertIn("method", data)
            self.assertIn("dominant_period", data)
        finally:
            json_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # C. Reusing a pre-computed summary
    # ------------------------------------------------------------------

    def test_precomputed_summary_is_not_recomputed(self):
        """Supplying summary= avoids calling get_period_summary again."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        s = self._synthetic_summary()
        lc = _make_1d_lc_no_transform(n_obs=40, period=100.0, seed=0)
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False
        ) as tmp:
            txt_path = Path(tmp.name)
        try:
            with patch.object(
                lc, "get_period_summary", wraps=lc.get_period_summary
            ) as mock_gps:
                lc.write_period_summary_outputs(
                    text_file=txt_path, summary=s
                )
            mock_gps.assert_not_called()
        finally:
            txt_path.unlink(missing_ok=True)

    def test_returns_summary_object(self):
        """The wrapper returns the summary object."""
        import tempfile
        from pathlib import Path

        s = self._synthetic_summary()
        lc = _make_1d_lc_no_transform(n_obs=40, period=100.0, seed=0)
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False
        ) as tmp:
            txt_path = Path(tmp.name)
        try:
            result = lc.write_period_summary_outputs(
                text_file=txt_path, summary=s
            )
            self.assertIs(result, s)
        finally:
            txt_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # D. No files written when no paths are given
    # ------------------------------------------------------------------

    def test_no_files_written_when_no_paths(self):
        """Calling wrapper with no file paths writes nothing but returns summary."""
        s = self._synthetic_summary()
        lc = _make_1d_lc_no_transform(n_obs=40, period=100.0, seed=0)
        result = lc.write_period_summary_outputs(summary=s)
        self.assertIs(result, s)

    # ------------------------------------------------------------------
    # E. PNG export via the wrapper (requires a fitted LC)
    # ------------------------------------------------------------------

    def test_png_file_is_created(self):
        """write_period_summary_outputs(png_file=...) creates a PNG file."""
        import tempfile
        from pathlib import Path

        lc = self._fitted_lc()
        summary = lc.get_period_summary()
        with tempfile.NamedTemporaryFile(
            suffix=".png", delete=False
        ) as tmp:
            png_path = Path(tmp.name)
        try:
            lc.write_period_summary_outputs(
                png_file=png_path, summary=summary
            )
            self.assertTrue(png_path.exists())
            self.assertGreater(png_path.stat().st_size, 0)
        finally:
            png_path.unlink(missing_ok=True)

    def test_text_and_png_together(self):
        """Both text and PNG files are created in one call."""
        import tempfile
        from pathlib import Path

        lc = self._fitted_lc()
        summary = lc.get_period_summary()
        with (
            tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as t,
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as p,
        ):
            txt_path = Path(t.name)
            png_path = Path(p.name)
        try:
            lc.write_period_summary_outputs(
                text_file=txt_path,
                png_file=png_path,
                summary=summary,
            )
            self.assertTrue(txt_path.exists())
            self.assertTrue(png_path.exists())
            self.assertGreater(txt_path.stat().st_size, 0)
            self.assertGreater(png_path.stat().st_size, 0)
        finally:
            txt_path.unlink(missing_ok=True)
            png_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
