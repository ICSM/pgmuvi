"""Minimal tests for the multi-peak support added in patch 4.

Covers:
1. Peak sorting at PeriodSummaryResult construction time.
2. get_primary_peak() — empty and non-empty cases.
3. get_top_n_peaks(n) — bounds and ordering.
4. get_significant_peaks(threshold) — correct filtering.
5. as_dict() contains 'peaks', 'n_peaks', 'n_significant_peaks'.
6. to_text() includes a PRIMARY PEAK section and ADDITIONAL PEAKS section.
7. max_peaks_to_show parameter in to_text() limits output correctly.
"""

import unittest

import numpy as np

from pgmuvi.lightcurve import PeriodPeakResult, PeriodSummaryResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_peak(rank, area_fraction, height=1.0, period=100.0):
    """Return a minimal PeriodPeakResult for testing."""
    return PeriodPeakResult(
        rank=rank,
        frequency=1.0 / period,
        period=period,
        height=height,
        prominence=0.5,
        area_fraction=area_fraction,
        interval_frequency=(0.009, 0.011),
        interval_period=(90.0, 110.0),
        period_ratio_to_primary=1.0 if rank == 1 else period / 100.0,
        is_candidate_lsp=False,
        notes="",
    )


def _make_summary(peaks=None, **kwargs):
    """Return a PeriodSummaryResult with given peaks."""
    return PeriodSummaryResult(
        method="test",
        model_name="TestModel",
        n_peaks_detected=len(peaks) if peaks else 0,
        n_peaks_analyzed=len(peaks) if peaks else 0,
        peaks=peaks,
        dominant_period=peaks[0].period if peaks else None,
        dominant_frequency=peaks[0].frequency if peaks else None,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Peak sorting
# ---------------------------------------------------------------------------


class TestPeakSorting(unittest.TestCase):
    """Peaks are sorted ascending by rank at init time."""

    def test_peaks_sorted_by_rank_asc(self):
        p3 = _make_peak(rank=3, area_fraction=0.30, period=200.0)
        p1 = _make_peak(rank=1, area_fraction=0.80, period=100.0)
        p2 = _make_peak(rank=2, area_fraction=0.55, period=150.0)
        # Pass in deliberately unsorted order
        summary = _make_summary(peaks=[p3, p2, p1])
        ranks = [p.rank for p in summary.peaks]
        self.assertEqual(ranks, sorted(ranks))

    def test_primary_is_rank_1(self):
        p2 = _make_peak(rank=2, area_fraction=0.20, period=300.0)
        p1 = _make_peak(rank=1, area_fraction=0.75, period=100.0)
        # Pass rank-2 first to ensure sorting works
        summary = _make_summary(peaks=[p2, p1])
        self.assertEqual(summary.peaks[0].rank, 1)

    def test_empty_peaks_is_empty_list(self):
        summary = _make_summary(peaks=None)
        self.assertEqual(summary.peaks, [])


# ---------------------------------------------------------------------------
# 2. get_primary_peak()
# ---------------------------------------------------------------------------


class TestGetPrimaryPeak(unittest.TestCase):
    def test_returns_none_when_no_peaks(self):
        summary = _make_summary(peaks=None)
        self.assertIsNone(summary.get_primary_peak())

    def test_returns_first_peak(self):
        pk = _make_peak(rank=1, area_fraction=0.80)
        summary = _make_summary(peaks=[pk])
        self.assertIs(summary.get_primary_peak(), summary.peaks[0])

    def test_returns_rank_1_peak(self):
        p2 = _make_peak(rank=2, area_fraction=0.75, period=200.0)
        p1 = _make_peak(rank=1, area_fraction=0.20, period=100.0)
        # Pass in reverse rank order to test sorting
        summary = _make_summary(peaks=[p2, p1])
        self.assertEqual(summary.get_primary_peak().rank, 1)
        self.assertAlmostEqual(summary.get_primary_peak().area_fraction, 0.20)


# ---------------------------------------------------------------------------
# 3. get_top_n_peaks(n)
# ---------------------------------------------------------------------------


class TestGetTopNPeaks(unittest.TestCase):
    def setUp(self):
        peaks = [
            _make_peak(rank=1, area_fraction=0.80, period=100.0),
            _make_peak(rank=2, area_fraction=0.50, period=200.0),
            _make_peak(rank=3, area_fraction=0.25, period=300.0),
        ]
        self.summary = _make_summary(peaks=peaks)

    def test_returns_correct_count(self):
        self.assertEqual(len(self.summary.get_top_n_peaks(2)), 2)

    def test_returns_all_when_n_exceeds_count(self):
        self.assertEqual(len(self.summary.get_top_n_peaks(100)), 3)

    def test_returns_empty_when_no_peaks(self):
        summary = _make_summary(peaks=None)
        self.assertEqual(summary.get_top_n_peaks(3), [])

    def test_ordering_is_ascending_by_rank(self):
        top2 = self.summary.get_top_n_peaks(2)
        self.assertLess(top2[0].rank, top2[1].rank)


# ---------------------------------------------------------------------------
# 4. get_significant_peaks(threshold)
# ---------------------------------------------------------------------------


class TestGetSignificantPeaks(unittest.TestCase):
    def setUp(self):
        peaks = [
            _make_peak(rank=1, area_fraction=0.80, period=100.0),
            _make_peak(rank=2, area_fraction=0.60, period=200.0),
            _make_peak(rank=3, area_fraction=0.30, period=300.0),
        ]
        self.summary = _make_summary(peaks=peaks)

    def test_default_threshold_0_68(self):
        sig = self.summary.get_significant_peaks()
        areas = [p.area_fraction for p in sig]
        self.assertTrue(all(a >= 0.68 for a in areas))

    def test_custom_threshold(self):
        sig = self.summary.get_significant_peaks(threshold=0.50)
        self.assertEqual(len(sig), 2)

    def test_none_significant_when_threshold_high(self):
        sig = self.summary.get_significant_peaks(threshold=0.99)
        self.assertEqual(sig, [])

    def test_all_significant_when_threshold_zero(self):
        sig = self.summary.get_significant_peaks(threshold=0.0)
        self.assertEqual(len(sig), 3)

    def test_nan_area_fraction_excluded(self):
        p_nan = _make_peak(rank=4, area_fraction=float("nan"), period=400.0)
        peaks_with_nan = self.summary.peaks + [p_nan]
        summary = _make_summary(peaks=peaks_with_nan)
        # NaN peak should not appear regardless of threshold
        sig = summary.get_significant_peaks(threshold=0.0)
        self.assertTrue(all(np.isfinite(p.area_fraction) for p in sig))


# ---------------------------------------------------------------------------
# 5. as_dict() fields
# ---------------------------------------------------------------------------


class TestAsDictMultiPeak(unittest.TestCase):
    def setUp(self):
        peaks = [
            _make_peak(rank=1, area_fraction=0.80, period=100.0),
            _make_peak(rank=2, area_fraction=0.40, period=200.0),
        ]
        self.summary = _make_summary(peaks=peaks)
        self.d = self.summary.as_dict()

    def test_peaks_key_present(self):
        self.assertIn("peaks", self.d)

    def test_peaks_is_list_of_dicts(self):
        self.assertIsInstance(self.d["peaks"], list)
        for item in self.d["peaks"]:
            self.assertIsInstance(item, dict)

    def test_n_peaks_key_present_and_correct(self):
        self.assertIn("n_peaks", self.d)
        self.assertEqual(self.d["n_peaks"], 2)

    def test_n_significant_peaks_present(self):
        self.assertIn("n_significant_peaks", self.d)

    def test_backward_compat_keys_present(self):
        for key in (
            "period_interval",
            "period_interval_fwhm_like",
            "significant_periods",
            "dominant_period",
            "dominant_frequency",
            "method",
        ):
            self.assertIn(key, self.d, msg=f"Missing key: {key}")

    def test_peaks_list_matches_count(self):
        self.assertEqual(len(self.d["peaks"]), self.d["n_peaks"])

    def test_each_peak_dict_has_required_fields(self):
        required = {"rank", "period", "frequency", "height", "area_fraction",
                    "interval_period", "interval_frequency"}
        for pk_dict in self.d["peaks"]:
            for field in required:
                self.assertIn(field, pk_dict)

    def test_empty_summary_n_peaks_is_zero(self):
        empty = _make_summary(peaks=None)
        d = empty.as_dict()
        self.assertEqual(d["n_peaks"], 0)
        self.assertEqual(d["peaks"], [])


# ---------------------------------------------------------------------------
# 6 & 7. to_text() — primary/additional sections, max_peaks_to_show
# ---------------------------------------------------------------------------


class TestToTextMultiPeak(unittest.TestCase):
    def _make_three_peak_summary(self):
        peaks = [
            _make_peak(rank=1, area_fraction=0.80, period=100.0),
            _make_peak(rank=2, area_fraction=0.55, period=200.0),
            _make_peak(rank=3, area_fraction=0.30, period=300.0),
        ]
        return _make_summary(peaks=peaks)

    def test_primary_peak_section_present(self):
        text = self._make_three_peak_summary().to_text()
        self.assertIn("PRIMARY PEAK", text)

    def test_additional_peaks_section_present_when_multiple(self):
        text = self._make_three_peak_summary().to_text()
        self.assertIn("ADDITIONAL PEAKS", text)

    def test_no_additional_section_for_single_peak(self):
        peaks = [_make_peak(rank=1, area_fraction=0.80)]
        summary = _make_summary(peaks=peaks)
        text = summary.to_text()
        self.assertNotIn("ADDITIONAL PEAKS", text)

    def test_max_peaks_limits_output(self):
        peaks = [_make_peak(rank=i, area_fraction=1.0 / i, period=i * 50.0)
                 for i in range(1, 6)]  # 5 peaks
        summary = _make_summary(peaks=peaks)
        # max_peaks_to_show=2: primary (1) + 1 additional
        text = summary.to_text(max_peaks_to_show=2)
        # Should mention "(+3 additional peaks not shown)"
        self.assertIn("+3 additional", text)

    def test_no_overflow_line_when_within_limit(self):
        text = self._make_three_peak_summary().to_text(max_peaks_to_show=5)
        self.assertNotIn("not shown", text)

    def test_include_peaks_false_skips_peak_sections(self):
        summary = self._make_three_peak_summary()
        text = summary.to_text(include_peaks=False)
        self.assertNotIn("PRIMARY PEAK", text)
        self.assertNotIn("ADDITIONAL PEAKS", text)


if __name__ == "__main__":
    unittest.main()
