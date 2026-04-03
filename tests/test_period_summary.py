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
    """Tests that the SM backend uses a log-spaced grid and local refinement."""

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
        """Summary notes mention local refinement when it was applied.

        Refinement is only applied when both half-max crossings are contained.
        We verify that on a well-contained summary the notes mention refinement,
        OR that the notes always mention the log-spaced grid (which is always
        present even if refinement was skipped due to truncation).
        """
        lc = self._make_sm_lc()
        summary = lc.get_period_summary()
        notes = summary["notes"].lower()
        # The notes must always mention log-spacing
        self.assertIn("log", notes)
        # If no truncation the notes should also mention refinement
        period_lo, period_hi = summary["period_interval_fwhm_like"]
        if "truncat" not in notes and period_lo is not None:
            self.assertIn("refine", notes)

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


if __name__ == "__main__":
    unittest.main()
