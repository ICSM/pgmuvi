"""Tests for kernel-aware period-summary backend metadata.

Verifies that each backend helper returns a properly structured
PeriodSummaryResult with the new metadata fields
(backend, kernel_family, time_kernel_family, has_stochastic_background)
populated correctly and exported via as_dict() and to_text().

Backends covered:
  A. spectral_mixture
  B. explicit_period
  C. periodic_plus_stochastic
  D. separable_2d
  E. non_periodic
"""

import unittest

import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve, PeriodSummaryResult
from pgmuvi.synthetic import make_chromatic_sinusoid_2d, make_simple_sinusoid_1d


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_1d_lc(model_name, n_obs=30, period=100.0, seed=0, **kw):
    """Return a minimal 1-D Lightcurve with the given model set."""
    lc = make_simple_sinusoid_1d(
        n_obs=n_obs, period=period, noise_level=0.05, t_span=500.0, seed=seed
    )
    lc.set_model(model_name, **kw)
    return lc


def _make_2d_lc(model_name, period=100.0, seed=0, **kw):
    """Return a minimal 2-D Lightcurve with the given model set."""
    lc = make_chromatic_sinusoid_2d(period=period, seed=seed)
    lc.set_model(model_name, **kw)
    return lc


# ---------------------------------------------------------------------------
# A. Spectral-mixture backend
# ---------------------------------------------------------------------------


class TestSpectralMixtureBackend(unittest.TestCase):
    """get_period_summary() for 1D SM model returns correct backend metadata."""

    def setUp(self):
        self.lc = _make_1d_lc("1D", num_mixtures=1)
        self.summary = self.lc.get_period_summary()

    def test_returns_period_summary_result(self):
        self.assertIsInstance(self.summary, PeriodSummaryResult)

    def test_backend_is_spectral_mixture(self):
        self.assertEqual(self.summary.backend, "spectral_mixture")

    def test_kernel_family_populated(self):
        self.assertIsInstance(self.summary.kernel_family, str)
        self.assertGreater(len(self.summary.kernel_family), 0)

    def test_time_kernel_family_populated(self):
        self.assertIsInstance(self.summary.time_kernel_family, str)
        self.assertGreater(len(self.summary.time_kernel_family), 0)

    def test_has_stochastic_background_false(self):
        self.assertFalse(self.summary.has_stochastic_background)

    def test_as_dict_contains_backend(self):
        d = self.summary.as_dict()
        self.assertIn("backend", d)
        self.assertEqual(d["backend"], "spectral_mixture")

    def test_as_dict_contains_kernel_family(self):
        d = self.summary.as_dict()
        self.assertIn("kernel_family", d)
        self.assertIsInstance(d["kernel_family"], str)

    def test_as_dict_contains_time_kernel_family(self):
        d = self.summary.as_dict()
        self.assertIn("time_kernel_family", d)

    def test_as_dict_contains_has_stochastic_background(self):
        d = self.summary.as_dict()
        self.assertIn("has_stochastic_background", d)
        self.assertFalse(d["has_stochastic_background"])

    def test_to_text_contains_backend(self):
        text = self.summary.to_text()
        self.assertIn("Backend", text)
        self.assertIn("spectral_mixture", text)

    def test_to_text_contains_kernel_family(self):
        text = self.summary.to_text()
        self.assertIn("Kernel family", text)

    def test_to_text_contains_time_kernel_family(self):
        text = self.summary.to_text()
        self.assertIn("Time-kernel family", text)

    def test_notes_mention_summed_psd(self):
        """SM summary notes must mention summed PSD as comparison product."""
        notes = self.summary.notes.lower()
        self.assertIn("summed psd", notes)

    def test_notes_mention_diagnostic(self):
        """SM notes must distinguish diagnostic components from peaks."""
        notes = self.summary.notes.lower()
        self.assertIn("diagnostic", notes)

    def test_has_peaks(self):
        self.assertGreater(len(self.summary.peaks), 0)

    def test_has_psd(self):
        self.assertIsNotNone(self.summary.psd)
        self.assertIsNotNone(self.summary.freq_grid)


# ---------------------------------------------------------------------------
# B. Explicit-period backend
# ---------------------------------------------------------------------------


class TestExplicitPeriodBackend(unittest.TestCase):
    """get_period_summary() for QP model returns correct backend metadata."""

    def setUp(self):
        self.lc = _make_1d_lc("1DQuasiPeriodic", period=100.0)
        self.summary = self.lc.get_period_summary()

    def test_returns_period_summary_result(self):
        self.assertIsInstance(self.summary, PeriodSummaryResult)

    def test_backend_is_explicit_period(self):
        self.assertEqual(self.summary.backend, "explicit_period")

    def test_kernel_family_populated(self):
        self.assertIsInstance(self.summary.kernel_family, str)
        self.assertGreater(len(self.summary.kernel_family), 0)

    def test_has_stochastic_background_false(self):
        self.assertFalse(self.summary.has_stochastic_background)

    def test_dominant_period_from_kernel_parameter(self):
        """Period must be positive (extracted from period_length param)."""
        self.assertIsNotNone(self.summary.dominant_period)
        self.assertGreater(self.summary.dominant_period, 0.0)

    def test_no_psd(self):
        """Explicit-period backend has no PSD."""
        self.assertIsNone(self.summary.psd)
        self.assertIsNone(self.summary.freq_grid)

    def test_notes_mention_explicit_period(self):
        notes = self.summary.notes.lower()
        self.assertIn("period_length", notes)

    def test_notes_not_pretend_psd(self):
        """Notes must not claim a PSD-derived interval."""
        notes = self.summary.notes.lower()
        # Should explicitly say it is NOT a PSD-derived interval
        self.assertIn("not", notes)

    def test_as_dict_contains_backend(self):
        d = self.summary.as_dict()
        self.assertEqual(d["backend"], "explicit_period")

    def test_as_dict_contains_kernel_family(self):
        d = self.summary.as_dict()
        self.assertIn("kernel_family", d)
        self.assertIsInstance(d["kernel_family"], str)

    def test_to_text_contains_backend(self):
        text = self.summary.to_text()
        self.assertIn("explicit_period", text)

    def test_interval_definition_is_coherence_proxy(self):
        """With RBF lengthscale, interval_definition should say coherence."""
        self.assertIn("coherence", self.summary.interval_definition)

    def test_linear_qp_also_explicit_period(self):
        lc = _make_1d_lc("1DLinearQuasiPeriodic", period=100.0)
        s = lc.get_period_summary()
        self.assertIsInstance(s, PeriodSummaryResult)
        self.assertEqual(s.backend, "explicit_period")


# ---------------------------------------------------------------------------
# C. Periodic-plus-stochastic backend
# ---------------------------------------------------------------------------


class TestPeriodicPlusStochasticBackend(unittest.TestCase):
    """get_period_summary() for PeriodicPlusStochastic returns correct metadata."""

    def setUp(self):
        self.lc = _make_1d_lc("1DPeriodicStochastic", period=100.0)
        self.summary = self.lc.get_period_summary()

    def test_returns_period_summary_result(self):
        self.assertIsInstance(self.summary, PeriodSummaryResult)

    def test_backend_is_periodic_plus_stochastic(self):
        self.assertEqual(self.summary.backend, "periodic_plus_stochastic")

    def test_has_stochastic_background_true(self):
        self.assertTrue(self.summary.has_stochastic_background)

    def test_kernel_family_populated(self):
        self.assertIsInstance(self.summary.kernel_family, str)
        self.assertGreater(len(self.summary.kernel_family), 0)

    def test_time_kernel_family_populated(self):
        self.assertIsInstance(self.summary.time_kernel_family, str)

    def test_dominant_period_from_periodic_component(self):
        self.assertIsNotNone(self.summary.dominant_period)
        self.assertGreater(self.summary.dominant_period, 0.0)

    def test_no_psd(self):
        self.assertIsNone(self.summary.psd)
        self.assertIsNone(self.summary.freq_grid)

    def test_notes_mention_stochastic_background(self):
        notes = self.summary.notes.lower()
        self.assertIn("stochastic", notes)

    def test_notes_mention_periodic_component(self):
        notes = self.summary.notes.lower()
        self.assertIn("periodic", notes)

    def test_notes_not_independent_period_for_stochastic(self):
        """Notes must state stochastic is not an independent period."""
        notes = self.summary.notes.lower()
        # The notes should say something like "not interpreted as an
        # independent period"
        self.assertIn("not", notes)

    def test_as_dict_backend(self):
        d = self.summary.as_dict()
        self.assertEqual(d["backend"], "periodic_plus_stochastic")

    def test_as_dict_has_stochastic_background(self):
        d = self.summary.as_dict()
        self.assertTrue(d["has_stochastic_background"])

    def test_to_text_contains_backend(self):
        text = self.summary.to_text()
        self.assertIn("periodic_plus_stochastic", text)

    def test_to_text_stochastic_bg_true(self):
        text = self.summary.to_text()
        self.assertIn("True", text)


# ---------------------------------------------------------------------------
# D. Separable-2D backend
# ---------------------------------------------------------------------------


class TestSeparable2DBackend(unittest.TestCase):
    """get_period_summary() for separable 2D models returns correct metadata."""

    def test_backend_is_separable_2d_default(self):
        """Default Matern time kernel → separable_2d backend."""
        lc = _make_2d_lc("2DSeparable")
        s = lc.get_period_summary()
        self.assertIsInstance(s, PeriodSummaryResult)
        self.assertEqual(s.backend, "separable_2d")

    def test_time_kernel_family_populated(self):
        """time_kernel_family must be set for separable_2d."""
        lc = _make_2d_lc("2DSeparable")
        s = lc.get_period_summary()
        self.assertIsInstance(s.time_kernel_family, str)
        self.assertGreater(len(s.time_kernel_family), 0)

    def test_kernel_family_populated(self):
        lc = _make_2d_lc("2DSeparable")
        s = lc.get_period_summary()
        self.assertIsInstance(s.kernel_family, str)
        self.assertGreater(len(s.kernel_family), 0)

    def test_as_dict_backend_separable_2d(self):
        lc = _make_2d_lc("2DSeparable")
        s = lc.get_period_summary()
        d = s.as_dict()
        self.assertEqual(d["backend"], "separable_2d")

    def test_as_dict_time_kernel_family(self):
        lc = _make_2d_lc("2DSeparable")
        s = lc.get_period_summary()
        d = s.as_dict()
        self.assertIn("time_kernel_family", d)
        self.assertIsInstance(d["time_kernel_family"], str)

    def test_to_text_contains_separable_2d(self):
        lc = _make_2d_lc("2DSeparable")
        s = lc.get_period_summary()
        text = s.to_text()
        self.assertIn("separable_2d", text)

    def test_to_text_time_kernel_family(self):
        lc = _make_2d_lc("2DSeparable")
        s = lc.get_period_summary()
        text = s.to_text()
        self.assertIn("Time-kernel family", text)

    def test_notes_mention_time_kernel_only(self):
        """Notes must state the period is based on the time kernel only."""
        lc = _make_2d_lc("2DSeparable")
        s = lc.get_period_summary()
        notes = s.notes.lower()
        self.assertIn("time kernel", notes)

    def test_notes_mention_wavelength_kernel_excluded(self):
        """Notes must clarify wavelength doesn't determine the period."""
        # Use a 2D model with QP time kernel so there IS a period to report
        lc = _make_2d_lc(
            "2DAchromatic",
            time_kernel_type="quasi_periodic",
            period=100.0,
        )
        s = lc.get_period_summary()
        notes = s.notes.lower()
        # The separable-2D prefix explicitly says it's time-kernel-based
        self.assertIn("time kernel", notes)

    def test_qp_time_kernel_separable_2d(self):
        """QP time kernel in 2D → separable_2d with explicit_period method."""
        lc = _make_2d_lc(
            "2DAchromatic",
            time_kernel_type="quasi_periodic",
            period=100.0,
        )
        s = lc.get_period_summary()
        self.assertIsInstance(s, PeriodSummaryResult)
        self.assertEqual(s.backend, "separable_2d")
        self.assertIsNotNone(s.dominant_period)
        self.assertGreater(s.dominant_period, 0.0)

    def test_achromatic_separable_2d(self):
        lc = _make_2d_lc("2DAchromatic")
        s = lc.get_period_summary()
        self.assertIsInstance(s, PeriodSummaryResult)
        self.assertEqual(s.backend, "separable_2d")

    def test_wavelength_dependent_separable_2d(self):
        lc = _make_2d_lc("2DWavelengthDependent")
        s = lc.get_period_summary()
        self.assertIsInstance(s, PeriodSummaryResult)
        self.assertEqual(s.backend, "separable_2d")


# ---------------------------------------------------------------------------
# E. Non-periodic backend
# ---------------------------------------------------------------------------


class TestNonPeriodicBackend(unittest.TestCase):
    """get_period_summary() for non-periodic kernels returns clean summary."""

    def setUp(self):
        self.lc = _make_1d_lc("1DMatern")
        self.summary = self.lc.get_period_summary()

    def test_returns_period_summary_result(self):
        self.assertIsInstance(self.summary, PeriodSummaryResult)

    def test_backend_is_non_periodic(self):
        self.assertEqual(self.summary.backend, "non_periodic")

    def test_no_dominant_period(self):
        self.assertIsNone(self.summary.dominant_period)

    def test_no_dominant_frequency(self):
        self.assertIsNone(self.summary.dominant_frequency)

    def test_no_peaks(self):
        """Non-periodic backend must not produce fake peaks."""
        self.assertEqual(len(self.summary.peaks), 0)

    def test_no_psd(self):
        self.assertIsNone(self.summary.psd)
        self.assertIsNone(self.summary.freq_grid)

    def test_kernel_family_populated(self):
        self.assertIsInstance(self.summary.kernel_family, str)
        self.assertGreater(len(self.summary.kernel_family), 0)

    def test_has_stochastic_background_false(self):
        self.assertFalse(self.summary.has_stochastic_background)

    def test_notes_mention_no_periodic(self):
        notes = self.summary.notes.lower()
        self.assertIn("periodic", notes)

    def test_as_dict_backend(self):
        d = self.summary.as_dict()
        self.assertEqual(d["backend"], "non_periodic")

    def test_as_dict_no_peaks(self):
        d = self.summary.as_dict()
        self.assertEqual(len(d["peaks"]), 0)

    def test_as_dict_dominant_period_none(self):
        d = self.summary.as_dict()
        self.assertIsNone(d["dominant_period"])

    def test_to_text_contains_backend(self):
        text = self.summary.to_text()
        self.assertIn("non_periodic", text)

    def test_to_text_kernel_family(self):
        text = self.summary.to_text()
        self.assertIn("Kernel family", text)

    def test_n_significant_peaks_zero(self):
        d = self.summary.as_dict()
        self.assertEqual(d["n_significant_peaks"], 0)


# ---------------------------------------------------------------------------
# F. PeriodSummaryResult attribute existence
# ---------------------------------------------------------------------------


class TestPeriodSummaryResultAttributes(unittest.TestCase):
    """PeriodSummaryResult stores the new metadata as object attributes."""

    def _make_default(self):
        return PeriodSummaryResult()

    def test_backend_attribute_exists(self):
        r = self._make_default()
        self.assertTrue(hasattr(r, "backend"))
        self.assertEqual(r.backend, "")

    def test_kernel_family_attribute_exists(self):
        r = self._make_default()
        self.assertTrue(hasattr(r, "kernel_family"))
        self.assertEqual(r.kernel_family, "")

    def test_time_kernel_family_attribute_exists(self):
        r = self._make_default()
        self.assertTrue(hasattr(r, "time_kernel_family"))
        self.assertEqual(r.time_kernel_family, "")

    def test_has_stochastic_background_attribute_exists(self):
        r = self._make_default()
        self.assertTrue(hasattr(r, "has_stochastic_background"))
        self.assertFalse(r.has_stochastic_background)

    def test_backend_in_as_dict(self):
        r = PeriodSummaryResult(backend="spectral_mixture")
        self.assertEqual(r.as_dict()["backend"], "spectral_mixture")

    def test_kernel_family_in_as_dict(self):
        r = PeriodSummaryResult(kernel_family="SpectralMixtureKernel")
        self.assertEqual(r.as_dict()["kernel_family"], "SpectralMixtureKernel")

    def test_time_kernel_family_in_as_dict(self):
        r = PeriodSummaryResult(time_kernel_family="RBFKernel")
        self.assertEqual(r.as_dict()["time_kernel_family"], "RBFKernel")

    def test_has_stochastic_background_in_as_dict(self):
        r = PeriodSummaryResult(has_stochastic_background=True)
        self.assertTrue(r.as_dict()["has_stochastic_background"])

    def test_to_text_backend_line(self):
        r = PeriodSummaryResult(backend="test_backend")
        text = r.to_text()
        self.assertIn("Backend", text)
        self.assertIn("test_backend", text)

    def test_to_text_kernel_family_line(self):
        r = PeriodSummaryResult(kernel_family="MyKernel")
        text = r.to_text()
        self.assertIn("Kernel family", text)
        self.assertIn("MyKernel", text)

    def test_to_text_time_kernel_family_line(self):
        r = PeriodSummaryResult(time_kernel_family="TimeKernel")
        text = r.to_text()
        self.assertIn("Time-kernel family", text)
        self.assertIn("TimeKernel", text)

    def test_to_text_stochastic_bg_line(self):
        r = PeriodSummaryResult(has_stochastic_background=True)
        text = r.to_text()
        self.assertIn("Stochastic bg", text)
        self.assertIn("True", text)

    def test_backward_compat_dominant_period(self):
        r = PeriodSummaryResult(dominant_period=42.0)
        self.assertEqual(r.as_dict()["dominant_period"], 42.0)

    def test_backward_compat_notes(self):
        r = PeriodSummaryResult(notes="hello world")
        self.assertEqual(r.as_dict()["notes"], "hello world")


if __name__ == "__main__":
    unittest.main()
