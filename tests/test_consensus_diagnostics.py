"""
Unit tests for consensus diagnostics helpers and invariants.

Covers:
- Canonical band-record schema (_consensus_initialize_band_record)
- GP-validation status/reason helper (_consensus_set_gp_validation_status)
- Invalid GP status/reason combinations in the validator
- Accepted/rejected overlap detection
- Count consistency checks
- Rejection-summary consistency
- consensus_success=True invariants
- JSON-safe finalization (_consensus_make_json_safe)
- Recursive-fit protection (_consensus_prepare_gp_validation_fit_kwargs)

No real GP model training is performed.
"""

import math
import json
import unittest

import numpy as np

from pgmuvi.lightcurve import Lightcurve
from pgmuvi import lightcurve as lightcurve_module

GP_STATUS_NOT_REQUESTED = (
    lightcurve_module._CONSENSUS_GP_VALIDATION_STATUS_NOT_REQUESTED
)
GP_STATUS_SKIPPED = (
    lightcurve_module._CONSENSUS_GP_VALIDATION_STATUS_SKIPPED
)
GP_STATUS_FAILED = (
    lightcurve_module._CONSENSUS_GP_VALIDATION_STATUS_FAILED
)
GP_STATUS_SUCCESS = (
    lightcurve_module._CONSENSUS_GP_VALIDATION_STATUS_SUCCESS
)
GP_STATUS_REJECTED = (
    lightcurve_module._CONSENSUS_GP_VALIDATION_STATUS_REJECTED
)
BAND_STATUS_PENDING = lightcurve_module._CONSENSUS_BAND_STATUS_PENDING
BAND_STATUS_ACCEPTED = lightcurve_module._CONSENSUS_BAND_STATUS_ACCEPTED
BAND_STATUS_REJECTED = lightcurve_module._CONSENSUS_BAND_STATUS_REJECTED
ACF_STATUS_AGREEMENT = lightcurve_module._ACF_STATUS_AGREEMENT
ACF_STATUS_HARMONIC = lightcurve_module._ACF_STATUS_HARMONIC
ACF_STATUS_DISAGREEMENT = lightcurve_module._ACF_STATUS_DISAGREEMENT
ACF_STATUS_UNAVAILABLE = lightcurve_module._ACF_STATUS_UNAVAILABLE
REJECTION_REASON_NO_LS_PEAKS = lightcurve_module._CONSENSUS_REJECTION_REASON_NO_LS_PEAKS
REJECTION_REASON_NO_PLAUSIBLE_LS_PEAK = (
    lightcurve_module._CONSENSUS_REJECTION_REASON_NO_PLAUSIBLE_LS_PEAK
)


# ---------------------------------------------------------------------------
# Minimal Lightcurve factory (no GP training)
# ---------------------------------------------------------------------------

def _make_minimal_lc():
    """Return a tiny 1-D Lightcurve sufficient to call instance helpers."""
    t = np.linspace(0.0, 10.0, 20)
    y = np.sin(2.0 * math.pi * t)
    return Lightcurve(t, y)


# ---------------------------------------------------------------------------
# Helper: build a minimal valid finalized diagnostics dict
# ---------------------------------------------------------------------------

def _make_band_record(
    band="B",
    status=GP_STATUS_NOT_REQUESTED,
    reason=None,
    rejection_reasons=None,
    band_status="accepted",
):
    """Return a band record that satisfies _CONSENSUS_REQUIRED_BAND_KEYS."""
    record = {
        "band": band,
        "status": band_status,
        "rejection_reason": None,
        "rejection_reasons": rejection_reasons if rejection_reasons is not None else [],
        "metrics": None,
        "dominant_frequency": None,
        "dominant_period": None,
        "ls_significant": None,
        "ls_peak_power": None,
        "ls_peak_prominence": None,
        "ls_peak_area_fraction": None,
        "acf_frequency": None,
        "acf_period": None,
        "acf_supported": None,
        "acf_comparison_status": None,
        "acf_period_ratio": None,
        "acf_harmonic_order": None,
        "acf_error": None,
        "selected_from": None,
        "gp_validation_used": False,
        "gp_dominant_frequency": None,
        "gp_dominant_period": None,
        "gp_frequency_difference": None,
        "gp_fractional_frequency_difference": None,
        "gp_frequency_tolerance": None,
        "gp_validation_error": None,
        "gp_validation_status": status,
        "gp_validation_reason": reason,
    }
    if record["rejection_reasons"]:
        record["rejection_reason"] = record["rejection_reasons"][0]
        record["status"] = BAND_STATUS_REJECTED
    return record


def _make_valid_diagnostics(
    accepted_bands=None,
    rejected_bands=None,
    per_band_diagnostics=None,
    consensus_success=False,
    consensus_frequency=None,
    trusted_candidate_count=None,
    rejection_summary=None,
):
    """Return a minimal valid finalized diagnostics dict."""
    accepted = list(accepted_bands or [])
    rejected = list(rejected_bands or [])
    per_band = dict(per_band_diagnostics or {})
    for band in accepted + rejected:
        band_key = str(band)
        default_band_status = (
            BAND_STATUS_ACCEPTED if band_key in accepted else BAND_STATUS_REJECTED
        )
        default_reasons = (
            []
            if default_band_status == BAND_STATUS_ACCEPTED
            else [REJECTION_REASON_NO_LS_PEAKS]
        )
        per_band.setdefault(
            band_key,
            _make_band_record(
                band=band_key,
                band_status=default_band_status,
                rejection_reasons=default_reasons,
            ),
        )
    n_acc = len(accepted)
    n_rej = len(rejected)
    return {
        "fit_strategy": "consensus",
        "consensus_success": consensus_success,
        "consensus_frequency": consensus_frequency,
        "consensus_period": (
            1.0 / consensus_frequency if consensus_frequency else None
        ),
        "consensus_frequency_width": None,
        "consensus_frequency_scatter": None,
        "accepted_bands": accepted,
        "rejected_bands": rejected,
        "rejection_summary": rejection_summary if rejection_summary is not None else {},
        "per_band_diagnostics": per_band,
        "n_total_bands": n_acc + n_rej,
        "n_accepted_bands": n_acc,
        "n_rejected_bands": n_rej,
        "use_acf_validation": False,
        "use_gp_validation": False,
        "gp_validation_requested": False,
        "gp_validation_performed": False,
        "trusted_candidate_count": trusted_candidate_count,
        "candidate_count": None,
        "consensus_generation_method": None,
        "rejection_reasons": {},
        "per_band_dominant_periods": {},
        "per_band_dominant_frequencies": {},
        "median_frequency": None,
        "mad_frequency_scatter": None,
        "consensus_inlier_bands": [],
        "consensus_outlier_bands": [],
        "final_consensus_frequency": consensus_frequency,
        "final_consensus_period": (
            1.0 / consensus_frequency if consensus_frequency else None
        ),
        "robust_frequency_width": None,
        "final_constraint_bounds": None,
        "controls": {},
        "mode": None,
    }


# ---------------------------------------------------------------------------
# 1. Canonical band-record schema
# ---------------------------------------------------------------------------

class TestInitializeBandRecord(unittest.TestCase):
    """_consensus_initialize_band_record produces a stable canonical schema."""

    def setUp(self):
        self.lc = _make_minimal_lc()

    def test_contains_gp_validation_status(self):
        record = self.lc._consensus_initialize_band_record("R")
        self.assertIn("gp_validation_status", record)

    def test_contains_gp_validation_reason(self):
        record = self.lc._consensus_initialize_band_record("R")
        self.assertIn("gp_validation_reason", record)

    def test_contains_rejection_reasons(self):
        record = self.lc._consensus_initialize_band_record("R")
        self.assertIn("rejection_reasons", record)

    def test_default_gp_validation_status(self):
        record = self.lc._consensus_initialize_band_record("R")
        self.assertEqual(record["gp_validation_status"], GP_STATUS_NOT_REQUESTED)

    def test_default_gp_validation_reason_is_none(self):
        record = self.lc._consensus_initialize_band_record("R")
        self.assertIsNone(record["gp_validation_reason"])

    def test_default_rejection_reasons_is_empty_list(self):
        record = self.lc._consensus_initialize_band_record("R")
        self.assertEqual(record["rejection_reasons"], [])

    def test_band_label_stored_as_string(self):
        record = self.lc._consensus_initialize_band_record(42)
        self.assertEqual(record["band"], "42")


# ---------------------------------------------------------------------------
# 2. GP-validation status/reason helper
# ---------------------------------------------------------------------------

class TestSetGpValidationStatus(unittest.TestCase):
    """_consensus_set_gp_validation_status enforces allowed statuses."""

    def _fresh_record(self):
        return _make_minimal_lc()._consensus_initialize_band_record("X")

    def test_valid_statuses_accepted(self):
        for status in (
            GP_STATUS_NOT_REQUESTED,
            GP_STATUS_SKIPPED,
            GP_STATUS_FAILED,
            GP_STATUS_SUCCESS,
            GP_STATUS_REJECTED,
        ):
            record = self._fresh_record()
            Lightcurve._consensus_set_gp_validation_status(record, status)
            self.assertEqual(record["gp_validation_status"], status)

    def test_invalid_status_raises_value_error(self):
        record = self._fresh_record()
        with self.assertRaises(ValueError):
            Lightcurve._consensus_set_gp_validation_status(record, "bogus_status")

    def test_reason_is_set_when_provided(self):
        record = self._fresh_record()
        Lightcurve._consensus_set_gp_validation_status(
            record, GP_STATUS_FAILED, reason="exception"
        )
        self.assertEqual(record["gp_validation_reason"], "exception")

    def test_stale_reason_cleared_on_success(self):
        record = self._fresh_record()
        # First inject a stale reason via a failed status.
        Lightcurve._consensus_set_gp_validation_status(
            record, GP_STATUS_FAILED, reason="exception"
        )
        self.assertEqual(record["gp_validation_reason"], "exception")
        # Transition to success without a reason; stale value must be cleared.
        Lightcurve._consensus_set_gp_validation_status(record, GP_STATUS_SUCCESS)
        self.assertIsNone(record["gp_validation_reason"])

    def test_stale_reason_cleared_on_not_requested(self):
        record = self._fresh_record()
        Lightcurve._consensus_set_gp_validation_status(
            record, GP_STATUS_SKIPPED, reason="band_not_accepted"
        )
        self.assertEqual(record["gp_validation_reason"], "band_not_accepted")
        Lightcurve._consensus_set_gp_validation_status(
            record, GP_STATUS_NOT_REQUESTED
        )
        self.assertIsNone(record["gp_validation_reason"])

    def test_reason_preserved_on_other_statuses_without_explicit_reason(self):
        """If no reason is supplied and status is not success/not_requested,
        the existing reason should not be cleared."""
        record = self._fresh_record()
        Lightcurve._consensus_set_gp_validation_status(
            record, GP_STATUS_REJECTED, reason="diagnostics_failed"
        )
        # Calling again with same status and no reason preserves the reason.
        Lightcurve._consensus_set_gp_validation_status(record, GP_STATUS_REJECTED)
        self.assertEqual(record["gp_validation_reason"], "diagnostics_failed")


# ---------------------------------------------------------------------------
# 3. Band-status helper
# ---------------------------------------------------------------------------


class TestSetBandStatus(unittest.TestCase):
    """_consensus_set_band_status enforces status/rejection invariants."""

    def _fresh_record(self):
        return _make_minimal_lc()._consensus_initialize_band_record("X")

    def test_accepted_clears_stale_rejection_data(self):
        record = self._fresh_record()
        Lightcurve._consensus_set_band_status(
            record,
            BAND_STATUS_REJECTED,
            [REJECTION_REASON_NO_LS_PEAKS],
        )
        Lightcurve._consensus_set_band_status(record, BAND_STATUS_ACCEPTED, [])
        self.assertEqual(record["status"], BAND_STATUS_ACCEPTED)
        self.assertEqual(record["rejection_reasons"], [])
        self.assertIsNone(record["rejection_reason"])

    def test_rejected_requires_reasons(self):
        record = self._fresh_record()
        with self.assertRaises(ValueError):
            Lightcurve._consensus_set_band_status(
                record,
                BAND_STATUS_REJECTED,
                [],
            )

    def test_invalid_reasons_rejected(self):
        record = self._fresh_record()
        with self.assertRaises(ValueError):
            Lightcurve._consensus_set_band_status(
                record,
                BAND_STATUS_REJECTED,
                ["not_a_known_reason"],
            )

    def test_duplicate_reasons_are_deduplicated(self):
        record = self._fresh_record()
        Lightcurve._consensus_set_band_status(
            record,
            BAND_STATUS_REJECTED,
            [
                REJECTION_REASON_NO_LS_PEAKS,
                REJECTION_REASON_NO_LS_PEAKS,
            ],
        )
        self.assertEqual(
            record["rejection_reasons"],
            [REJECTION_REASON_NO_LS_PEAKS],
        )

    def test_rejection_reason_is_synchronized(self):
        record = self._fresh_record()
        Lightcurve._consensus_set_band_status(
            record,
            BAND_STATUS_REJECTED,
            [REJECTION_REASON_NO_LS_PEAKS],
        )
        self.assertEqual(record["rejection_reason"], REJECTION_REASON_NO_LS_PEAKS)

    def test_invalid_status_rejected(self):
        record = self._fresh_record()
        with self.assertRaises(ValueError):
            Lightcurve._consensus_set_band_status(
                record,
                "bad_status",
                [],
            )


# ---------------------------------------------------------------------------
# 4. ACF comparison status helper
# ---------------------------------------------------------------------------


class TestSetAcfComparisonStatus(unittest.TestCase):
    """_consensus_set_acf_comparison_status enforces metadata invariants."""

    def _fresh_record(self):
        return _make_minimal_lc()._consensus_initialize_band_record("X")

    def test_valid_combinations_pass(self):
        record = self._fresh_record()
        Lightcurve._consensus_set_acf_comparison_status(
            record,
            ACF_STATUS_AGREEMENT,
            period_ratio=1.02,
        )
        self.assertEqual(record["acf_comparison_status"], ACF_STATUS_AGREEMENT)
        self.assertAlmostEqual(record["acf_period_ratio"], 1.02)
        self.assertIsNone(record["acf_harmonic_order"])

        Lightcurve._consensus_set_acf_comparison_status(
            record,
            ACF_STATUS_HARMONIC,
            period_ratio=2.0,
            harmonic_order=2,
        )
        self.assertEqual(record["acf_comparison_status"], ACF_STATUS_HARMONIC)
        self.assertEqual(record["acf_harmonic_order"], 2)

    def test_invalid_metadata_combinations_raise(self):
        record = self._fresh_record()
        with self.assertRaises(ValueError):
            Lightcurve._consensus_set_acf_comparison_status(
                record,
                ACF_STATUS_AGREEMENT,
            )
        with self.assertRaises(ValueError):
            Lightcurve._consensus_set_acf_comparison_status(
                record,
                ACF_STATUS_UNAVAILABLE,
                period_ratio=1.0,
            )

    def test_stale_metadata_cleared(self):
        record = self._fresh_record()
        Lightcurve._consensus_set_acf_comparison_status(
            record,
            ACF_STATUS_HARMONIC,
            period_ratio=2.0,
            harmonic_order=2,
        )
        Lightcurve._consensus_set_acf_comparison_status(
            record,
            ACF_STATUS_UNAVAILABLE,
        )
        self.assertIsNone(record["acf_period_ratio"])
        self.assertIsNone(record["acf_harmonic_order"])

    def test_harmonic_requires_harmonic_order(self):
        record = self._fresh_record()
        with self.assertRaises(ValueError):
            Lightcurve._consensus_set_acf_comparison_status(
                record,
                ACF_STATUS_HARMONIC,
                period_ratio=2.0,
            )

    def test_none_status_clears_metadata(self):
        record = self._fresh_record()
        Lightcurve._consensus_set_acf_comparison_status(
            record,
            ACF_STATUS_DISAGREEMENT,
            period_ratio=1.5,
            harmonic_order=3,
        )
        Lightcurve._consensus_set_acf_comparison_status(record, None)
        self.assertIsNone(record["acf_comparison_status"])
        self.assertIsNone(record["acf_period_ratio"])
        self.assertIsNone(record["acf_harmonic_order"])


# ---------------------------------------------------------------------------
# 5. Top-level rejection-reasons helper
# ---------------------------------------------------------------------------


class TestSetTopLevelRejectionReasons(unittest.TestCase):
    """_consensus_set_top_level_rejection_reasons canonicalizes top-level map."""

    def _lc(self):
        return _make_minimal_lc()

    def test_helper_canonicalizes_duplicate_band_entries(self):
        diagnostics = {"rejection_reasons": {}}
        self._lc()._consensus_set_top_level_rejection_reasons(
            diagnostics,
            {
                REJECTION_REASON_NO_LS_PEAKS: ["A", "A", 2, "2", "B"],
            },
        )
        self.assertEqual(
            diagnostics["rejection_reasons"][REJECTION_REASON_NO_LS_PEAKS],
            ["A", "2", "B"],
        )

    def test_helper_preserves_first_occurrence_order_when_deduplicating(self):
        diagnostics = {"rejection_reasons": {}}
        self._lc()._consensus_set_top_level_rejection_reasons(
            diagnostics,
            {
                REJECTION_REASON_NO_LS_PEAKS: ["A", "B", "A", "C", "B"],
            },
        )
        self.assertEqual(
            diagnostics["rejection_reasons"][REJECTION_REASON_NO_LS_PEAKS],
            ["A", "B", "C"],
        )

    def test_helper_rejects_invalid_rejection_reason_keys(self):
        diagnostics = {"rejection_reasons": {}}
        with self.assertRaises(ValueError):
            self._lc()._consensus_set_top_level_rejection_reasons(
                diagnostics,
                {"not_a_valid_reason": ["A"]},
            )

    def test_finalization_returns_canonicalized_top_level_rejection_reasons(self):
        diag = _make_valid_diagnostics(accepted_bands=[], rejected_bands=["A", "B"])
        diag["rejection_reasons"] = {
            REJECTION_REASON_NO_LS_PEAKS: ["A", "A", "B", "B"],
        }
        result = self._lc()._consensus_finalize_result_structure(diag)
        self.assertEqual(
            result["rejection_reasons"][REJECTION_REASON_NO_LS_PEAKS],
            ["A", "B"],
        )
        self.assertIn("rejection_reasons", result)
        self.assertIn("rejection_summary", result)
        self.assertEqual(result["rejection_summary"], result["rejection_reasons"])

    def test_finalization_rejects_invalid_top_level_rejection_reason_keys(self):
        diag = _make_valid_diagnostics(accepted_bands=[], rejected_bands=["A"])
        diag["rejection_reasons"] = {"not_a_valid_reason": ["A"]}
        with self.assertRaises(ValueError):
            self._lc()._consensus_finalize_result_structure(diag)

    def test_finalization_migrates_rejection_summary_when_canonical_missing(self):
        diag = _make_valid_diagnostics(accepted_bands=[], rejected_bands=["A", "B"])
        diag["rejection_summary"] = {
            REJECTION_REASON_NO_LS_PEAKS: ["A", "A", "B", "B"],
        }
        result = self._lc()._consensus_finalize_result_structure(diag)
        self.assertEqual(
            result["rejection_reasons"],
            {REJECTION_REASON_NO_LS_PEAKS: ["A", "B"]},
        )
        self.assertEqual(result["rejection_summary"], result["rejection_reasons"])

    def test_finalization_accepts_identical_rejection_maps(self):
        diag = _make_valid_diagnostics(accepted_bands=[], rejected_bands=["A"])
        mapping = {REJECTION_REASON_NO_LS_PEAKS: ["A"]}
        diag["rejection_reasons"] = mapping.copy()
        diag["rejection_summary"] = mapping.copy()
        result = self._lc()._consensus_finalize_result_structure(diag)
        self.assertEqual(result["rejection_reasons"], mapping)
        self.assertEqual(result["rejection_summary"], mapping)

    def test_finalization_rejects_differing_rejection_maps(self):
        diag = _make_valid_diagnostics(accepted_bands=[], rejected_bands=["A"])
        diag["rejection_reasons"] = {REJECTION_REASON_NO_LS_PEAKS: ["A"]}
        diag["rejection_summary"] = {
            REJECTION_REASON_NO_PLAUSIBLE_LS_PEAK: ["A"]
        }
        with self.assertRaises(ValueError):
            self._lc()._consensus_finalize_result_structure(diag)

    def test_duplicate_band_canonicalization_works_via_rejection_summary_only(self):
        diag = _make_valid_diagnostics(accepted_bands=[], rejected_bands=["A", "B"])
        diag["rejection_summary"] = {
            REJECTION_REASON_NO_LS_PEAKS: ["A", "A", "B", "B"],
        }
        result = self._lc()._consensus_finalize_result_structure(diag)
        self.assertEqual(
            result["rejection_reasons"],
            {REJECTION_REASON_NO_LS_PEAKS: ["A", "B"]},
        )
        self.assertEqual(result["rejection_summary"], result["rejection_reasons"])


# ---------------------------------------------------------------------------
# 6. Invalid GP status/reason combinations (validator)
# ---------------------------------------------------------------------------

class TestValidateGpReasonStatusCombinations(unittest.TestCase):
    """_consensus_validate_result_structure rejects forbidden reason/status pairs."""

    def _validate(self, diag):
        _make_minimal_lc()._consensus_validate_result_structure(diag)

    def _diag_with_band_record(self, record):
        band = record["band"]
        return _make_valid_diagnostics(
            accepted_bands=[band],
            per_band_diagnostics={band: record},
        )

    def _assert_validation_raises(self, diag):
        """Assert that validation raises ValueError (reason/status invariant).

        _consensus_validate_result_structure uses ValueError specifically for
        semantic violations such as invalid status/reason combinations, while
        RuntimeError is reserved for structural violations (missing keys,
        type errors, etc.).  The per-band reason invariant raises ValueError.
        """
        with self.assertRaises(ValueError):
            self._validate(diag)

    # --- invalid combos ----

    def test_success_with_exception_reason_raises(self):
        record = _make_band_record(
            band="A", status=GP_STATUS_SUCCESS, reason="exception"
        )
        self._assert_validation_raises(self._diag_with_band_record(record))

    def test_not_requested_with_band_not_accepted_reason_raises(self):
        record = _make_band_record(
            band="A",
            status=GP_STATUS_NOT_REQUESTED,
            reason="band_not_accepted",
        )
        self._assert_validation_raises(self._diag_with_band_record(record))

    def test_rejected_with_exception_reason_raises(self):
        record = _make_band_record(
            band="A", status=GP_STATUS_REJECTED, reason="exception"
        )
        self._assert_validation_raises(self._diag_with_band_record(record))

    def test_failed_with_diagnostics_failed_reason_raises(self):
        record = _make_band_record(
            band="A", status=GP_STATUS_FAILED, reason="diagnostics_failed"
        )
        self._assert_validation_raises(self._diag_with_band_record(record))

    # --- valid combos ----

    def test_not_requested_with_none_reason_passes(self):
        record = _make_band_record(
            band="A", status=GP_STATUS_NOT_REQUESTED, reason=None
        )
        self._validate(self._diag_with_band_record(record))

    def test_success_with_none_reason_passes(self):
        record = _make_band_record(band="A", status=GP_STATUS_SUCCESS, reason=None)
        self._validate(self._diag_with_band_record(record))

    def test_skipped_with_band_not_accepted_reason_passes(self):
        record = _make_band_record(
            band="A", status=GP_STATUS_SKIPPED, reason="band_not_accepted"
        )
        self._validate(self._diag_with_band_record(record))

    def test_rejected_with_diagnostics_failed_reason_passes(self):
        record = _make_band_record(
            band="A", status=GP_STATUS_REJECTED, reason="diagnostics_failed"
        )
        self._validate(self._diag_with_band_record(record))

    def test_failed_with_exception_reason_passes(self):
        record = _make_band_record(
            band="A", status=GP_STATUS_FAILED, reason="exception"
        )
        self._validate(self._diag_with_band_record(record))

    def test_error_message_includes_band_label(self):
        record = _make_band_record(
            band="MY_BAND", status=GP_STATUS_SUCCESS, reason="exception"
        )
        try:
            self._validate(self._diag_with_band_record(record))
            self.fail("Expected ValueError or RuntimeError")
        except (ValueError, RuntimeError) as exc:
            self.assertIn("MY_BAND", str(exc))


# ---------------------------------------------------------------------------
# 4. Accepted/rejected overlap
# ---------------------------------------------------------------------------

class TestAcceptedRejectedOverlap(unittest.TestCase):
    """Validator detects when a band appears in both accepted and rejected."""

    def test_overlap_raises(self):
        # _consensus_validate_result_structure raises ValueError for overlap
        # (semantic violation: a band cannot be simultaneously accepted and
        # rejected).
        band_rec = _make_band_record(band="B")
        diag = _make_valid_diagnostics(
            accepted_bands=["A", "B"],
            rejected_bands=["B", "C"],
            per_band_diagnostics={"B": band_rec},
        )
        # Manually override the counts so this is the only violation.
        diag["n_accepted_bands"] = 2
        diag["n_rejected_bands"] = 2
        diag["n_total_bands"] = 4
        with self.assertRaises(ValueError):
            _make_minimal_lc()._consensus_validate_result_structure(diag)


# ---------------------------------------------------------------------------
# 5. Count consistency
# ---------------------------------------------------------------------------

class TestCountConsistency(unittest.TestCase):
    """Validator enforces n_accepted + n_rejected == n_total."""

    def _lc(self):
        return _make_minimal_lc()

    def _assert_structural_raises(self, diag):
        """Assert that validation raises RuntimeError for structural violations.

        Count inconsistencies (n_accepted != len(accepted_bands), etc.) are
        structural errors reported as RuntimeError by the validator.
        """
        with self.assertRaises(RuntimeError):
            self._lc()._consensus_validate_result_structure(diag)

    def test_wrong_n_accepted_raises(self):
        diag = _make_valid_diagnostics(accepted_bands=["A"], rejected_bands=[])
        diag["n_accepted_bands"] = 99  # wrong
        self._assert_structural_raises(diag)

    def test_wrong_n_rejected_raises(self):
        diag = _make_valid_diagnostics(accepted_bands=[], rejected_bands=["A"])
        diag["n_rejected_bands"] = 0  # wrong
        self._assert_structural_raises(diag)

    def test_wrong_n_total_raises(self):
        diag = _make_valid_diagnostics(accepted_bands=["A"], rejected_bands=["B"])
        diag["n_total_bands"] = 10  # should be 2
        self._assert_structural_raises(diag)

    def test_valid_counts_pass(self):
        diag = _make_valid_diagnostics(accepted_bands=["A"], rejected_bands=["B"])
        # Should not raise
        self._lc()._consensus_validate_result_structure(diag)


# ---------------------------------------------------------------------------
# 6. Rejection summary consistency
# ---------------------------------------------------------------------------

class TestRejectionSummaryConsistency(unittest.TestCase):
    """Validator treats rejection_summary as an alias of rejection_reasons."""

    def _lc(self):
        return _make_minimal_lc()

    def _assert_raises(self, diag):
        """Assert validation raises ValueError for summary-band mismatches.

        Rejection-summary violations (band not in rejected_bands, accepted
        band in summary) are semantic errors reported as ValueError.
        """
        with self.assertRaises(ValueError):
            self._lc()._consensus_validate_result_structure(diag)

    def test_band_not_in_rejected_raises(self):
        diag = _make_valid_diagnostics(
            accepted_bands=["A"],
            rejected_bands=["B"],
            rejection_summary={REJECTION_REASON_NO_LS_PEAKS: ["C"]},
        )
        diag["rejection_reasons"] = {REJECTION_REASON_NO_LS_PEAKS: ["C"]}
        self._assert_raises(diag)

    def test_accepted_band_in_summary_raises(self):
        diag = _make_valid_diagnostics(
            accepted_bands=["A"],
            rejected_bands=["B"],
            rejection_summary={REJECTION_REASON_NO_LS_PEAKS: ["A"]},
        )
        diag["rejection_reasons"] = {REJECTION_REASON_NO_LS_PEAKS: ["A"]}
        self._assert_raises(diag)

    def test_validator_rejects_mismatched_rejection_reasons_and_summary(self):
        diag = _make_valid_diagnostics(
            accepted_bands=[],
            rejected_bands=["A"],
            rejection_summary={REJECTION_REASON_NO_LS_PEAKS: ["A"]},
        )
        diag["rejection_reasons"] = {
            REJECTION_REASON_NO_PLAUSIBLE_LS_PEAK: ["A"]
        }
        self._assert_raises(diag)

    def test_valid_matching_rejection_summary_passes(self):
        diag = _make_valid_diagnostics(
            accepted_bands=["A"],
            rejected_bands=["B"],
            rejection_summary={REJECTION_REASON_NO_LS_PEAKS: ["B"]},
        )
        diag["rejection_reasons"] = {REJECTION_REASON_NO_LS_PEAKS: ["B"]}
        # Should not raise
        self._lc()._consensus_validate_result_structure(diag)

    def test_empty_rejection_summary_passes(self):
        diag = _make_valid_diagnostics(
            accepted_bands=["A"],
            rejected_bands=[],
            rejection_summary={},
        )
        self._lc()._consensus_validate_result_structure(diag)


# ---------------------------------------------------------------------------
# 7. consensus_success=True invariants
# ---------------------------------------------------------------------------

class TestConsensusSuccessInvariants(unittest.TestCase):
    """When consensus_success is True, frequency/count requirements apply."""

    def _lc(self):
        return _make_minimal_lc()

    def _assert_structural_raises(self, diag):
        """Assert validation raises RuntimeError for success-semantics violations.

        Violations of consensus_success invariants (null frequency, zero
        accepted bands, zero trusted candidates) are structural/semantic
        errors reported as RuntimeError by the validator.
        """
        with self.assertRaises(RuntimeError):
            self._lc()._consensus_validate_result_structure(diag)

    def test_success_true_with_null_frequency_raises(self):
        diag = _make_valid_diagnostics(
            accepted_bands=["A"],
            consensus_success=True,
            consensus_frequency=None,
            trusted_candidate_count=3,
        )
        self._assert_structural_raises(diag)

    def test_success_true_with_zero_accepted_raises(self):
        diag = _make_valid_diagnostics(
            accepted_bands=[],
            rejected_bands=[],
            consensus_success=True,
            consensus_frequency=0.5,
            trusted_candidate_count=2,
        )
        self._assert_structural_raises(diag)

    def test_success_true_with_zero_trusted_candidates_raises(self):
        diag = _make_valid_diagnostics(
            accepted_bands=["A"],
            consensus_success=True,
            consensus_frequency=0.5,
            trusted_candidate_count=0,
        )
        self._assert_structural_raises(diag)

    def test_valid_success_passes(self):
        diag = _make_valid_diagnostics(
            accepted_bands=["A"],
            rejected_bands=[],
            consensus_success=True,
            consensus_frequency=0.5,
            trusted_candidate_count=2,
        )
        # Should not raise
        self._lc()._consensus_validate_result_structure(diag)


# ---------------------------------------------------------------------------
# 8. JSON-safe finalization
# ---------------------------------------------------------------------------

class TestJsonSafeFinalization(unittest.TestCase):
    """_consensus_make_json_safe converts non-serializable values."""

    def test_numpy_scalar_converted(self):
        result = Lightcurve._consensus_make_json_safe(np.float64(3.14))
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 3.14)

    def test_numpy_integer_converted(self):
        result = Lightcurve._consensus_make_json_safe(np.int32(7))
        self.assertIsInstance(result, int)
        self.assertEqual(result, 7)

    def test_numpy_array_converted_to_list(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = Lightcurve._consensus_make_json_safe(arr)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [1.0, 2.0, 3.0])

    def test_nan_float_becomes_none(self):
        result = Lightcurve._consensus_make_json_safe(float("nan"))
        self.assertIsNone(result)

    def test_inf_float_becomes_none(self):
        result = Lightcurve._consensus_make_json_safe(float("inf"))
        self.assertIsNone(result)

    def test_neg_inf_float_becomes_none(self):
        result = Lightcurve._consensus_make_json_safe(float("-inf"))
        self.assertIsNone(result)

    def test_plain_float_preserved(self):
        result = Lightcurve._consensus_make_json_safe(2.71828)
        self.assertAlmostEqual(result, 2.71828)

    def test_none_preserved(self):
        self.assertIsNone(Lightcurve._consensus_make_json_safe(None))

    def test_string_preserved(self):
        self.assertEqual(Lightcurve._consensus_make_json_safe("hello"), "hello")

    def test_dict_values_recursively_converted(self):
        d = {"a": np.float32(1.5), "b": float("nan")}
        result = Lightcurve._consensus_make_json_safe(d)
        self.assertAlmostEqual(result["a"], 1.5, places=4)
        self.assertIsNone(result["b"])

    def test_list_values_recursively_converted(self):
        lst = [np.int64(1), float("inf"), "ok"]
        result = Lightcurve._consensus_make_json_safe(lst)
        self.assertEqual(result, [1, None, "ok"])

    def test_finalize_result_structure_makes_values_json_safe(self):
        lc = _make_minimal_lc()
        diagnostics = _make_valid_diagnostics(
            consensus_frequency=np.float64(0.5),
            trusted_candidate_count=np.int64(3),
        )
        diagnostics["controls"] = {
            "array_val": np.array([1.0, 2.0]),
            "non_finite": float("inf"),
        }
        diagnostics["robust_frequency_width"] = float("inf")
        diagnostics["mad_frequency_scatter"] = float("nan")
        diagnostics["per_band_dominant_frequencies"] = {
            "B": np.array([0.1, np.float64(0.2)])
        }

        result = lc._consensus_finalize_result_structure(diagnostics)

        self.assertIsInstance(result["consensus_frequency"], float)
        self.assertEqual(result["trusted_candidate_count"], 3)
        self.assertIsInstance(result["trusted_candidate_count"], int)
        self.assertEqual(result["controls"]["array_val"], [1.0, 2.0])
        self.assertIsNone(result["controls"]["non_finite"])
        self.assertIsNone(result["consensus_frequency_width"])
        self.assertIsNone(result["consensus_frequency_scatter"])
        self.assertEqual(
            result["per_band_dominant_frequencies"]["B"],
            [0.1, 0.2],
        )


# ---------------------------------------------------------------------------
# 9. Finalized diagnostics end-to-end validation
# ---------------------------------------------------------------------------


class TestFinalizeValidatePipeline(unittest.TestCase):
    """Finalize+validate pipeline invariants for consensus diagnostics."""

    def _lc(self):
        return _make_minimal_lc()

    def _finalized_valid_diag(self):
        return self._lc()._consensus_finalize_result_structure(
            _make_valid_diagnostics(
                accepted_bands=["A"],
                rejected_bands=["B"],
            )
        )

    def test_missing_required_top_level_keys_raise(self):
        required_keys = (
            "fit_strategy",
            "consensus_success",
            "accepted_bands",
            "rejected_bands",
            "per_band_diagnostics",
            "n_total_bands",
        )
        for key in required_keys:
            with self.subTest(key=key):
                diag = dict(self._finalized_valid_diag())
                del diag[key]
                with self.assertRaises(RuntimeError) as exc:
                    self._lc()._consensus_validate_result_structure(diag)
                self.assertIn(key, str(exc.exception))

    def test_missing_required_per_band_keys_raise(self):
        required_band_keys = (
            "band",
            "status",
            "gp_validation_status",
            "gp_validation_reason",
            "rejection_reasons",
        )
        for key in required_band_keys:
            with self.subTest(key=key):
                diag = self._finalized_valid_diag()
                diag["per_band_diagnostics"]["A"] = dict(
                    diag["per_band_diagnostics"]["A"]
                )
                del diag["per_band_diagnostics"]["A"][key]
                with self.assertRaises(RuntimeError) as exc:
                    self._lc()._consensus_validate_result_structure(diag)
                self.assertIn("A", str(exc.exception))
                self.assertIn(key, str(exc.exception))

    def test_finalization_preserves_count_consistency(self):
        diag = _make_valid_diagnostics(
            accepted_bands=["A", "B"],
            rejected_bands=["C", "D"],
        )
        result = self._lc()._consensus_finalize_result_structure(diag)
        self.assertIsInstance(result["accepted_bands"], list)
        self.assertIsInstance(result["rejected_bands"], list)
        self.assertIsInstance(result["n_accepted_bands"], int)
        self.assertIsInstance(result["n_rejected_bands"], int)
        self.assertIsInstance(result["n_total_bands"], int)
        self.assertEqual(result["n_accepted_bands"], len(result["accepted_bands"]))
        self.assertEqual(result["n_rejected_bands"], len(result["rejected_bands"]))
        self.assertEqual(
            result["n_total_bands"],
            result["n_accepted_bands"] + result["n_rejected_bands"],
        )

    def test_finalization_recursively_sanitizes_nested_structures(self):
        diag = _make_valid_diagnostics(
            accepted_bands=["A"],
            rejected_bands=["B"],
        )
        diag["controls"] = {
            "level1": [
                np.float32(1.25),
                {
                    "level2": np.array(
                        [
                            np.int64(2),
                            np.float64(np.nan),
                            np.float64(np.inf),
                        ]
                    ),
                    "level3": [
                        {"x": np.array([np.float64(3.5), np.float64(np.nan)])},
                        np.float64(np.inf),
                    ],
                },
            ],
            "scalar": np.int32(4),
        }

        result = self._lc()._consensus_finalize_result_structure(diag)
        controls = result["controls"]
        self.assertEqual(controls["level1"][0], 1.25)
        self.assertIsInstance(controls["level1"][0], float)
        self.assertEqual(controls["level1"][1]["level2"], [2, None, None])
        self.assertEqual(controls["level1"][1]["level3"][0]["x"], [3.5, None])
        self.assertIsNone(controls["level1"][1]["level3"][1])
        self.assertEqual(controls["scalar"], 4)
        self.assertIsInstance(controls["scalar"], int)

    def test_finalized_structure_is_strict_json_serializable(self):
        diag = _make_valid_diagnostics(
            accepted_bands=["A"],
            rejected_bands=["B"],
        )
        diag["controls"] = {
            "bad_values": [np.float64(np.nan), np.float64(np.inf), np.array([1, 2])],
        }
        result = self._lc()._consensus_finalize_result_structure(diag)
        json.dumps(result, allow_nan=False)

    def test_validator_rejects_non_list_accepted_bands(self):
        bad_values = (
            ("A",),
            {"A"},
            np.array(["A"]),
        )
        for value in bad_values:
            with self.subTest(container=type(value).__name__):
                diag = _make_valid_diagnostics(accepted_bands=["A"], rejected_bands=[])
                diag["accepted_bands"] = value
                with self.assertRaises(RuntimeError):
                    self._lc()._consensus_validate_result_structure(diag)

    def test_validator_rejects_non_list_rejected_bands(self):
        bad_values = (
            ("B",),
            {"B"},
            np.array(["B"]),
        )
        for value in bad_values:
            with self.subTest(container=type(value).__name__):
                diag = _make_valid_diagnostics(accepted_bands=[], rejected_bands=["B"])
                diag["rejected_bands"] = value
                with self.assertRaises(RuntimeError):
                    self._lc()._consensus_validate_result_structure(diag)

    def test_validator_rejects_duplicate_entries_in_accepted(self):
        diag = _make_valid_diagnostics(accepted_bands=["A", "A"], rejected_bands=[])
        with self.assertRaises(RuntimeError):
            self._lc()._consensus_validate_result_structure(diag)

    def test_validator_rejects_duplicate_entries_in_rejected(self):
        diag = _make_valid_diagnostics(accepted_bands=[], rejected_bands=["B", "B"])
        with self.assertRaises(RuntimeError):
            self._lc()._consensus_validate_result_structure(diag)

    def test_validator_rejects_missing_per_band_for_accepted(self):
        diag = _make_valid_diagnostics(accepted_bands=["A"], rejected_bands=[])
        del diag["per_band_diagnostics"]["A"]
        with self.assertRaises(RuntimeError):
            self._lc()._consensus_validate_result_structure(diag)

    def test_validator_rejects_missing_per_band_for_rejected(self):
        diag = _make_valid_diagnostics(accepted_bands=[], rejected_bands=["B"])
        del diag["per_band_diagnostics"]["B"]
        with self.assertRaises(RuntimeError):
            self._lc()._consensus_validate_result_structure(diag)

    def test_validator_rejects_extra_per_band_entries(self):
        diag = _make_valid_diagnostics(accepted_bands=["A"], rejected_bands=[])
        diag["per_band_diagnostics"]["Z"] = _make_band_record(band="Z")
        with self.assertRaises(RuntimeError):
            self._lc()._consensus_validate_result_structure(diag)

    def test_validator_rejects_invalid_band_status(self):
        diag = _make_valid_diagnostics(accepted_bands=["A"], rejected_bands=[])
        diag["per_band_diagnostics"]["A"]["status"] = "bad_status"
        with self.assertRaises(ValueError) as exc:
            self._lc()._consensus_validate_result_structure(diag)
        self.assertIn("A", str(exc.exception))
        self.assertIn("bad_status", str(exc.exception))

    def test_validator_rejects_invalid_acf_status(self):
        diag = _make_valid_diagnostics(accepted_bands=["A"], rejected_bands=[])
        diag["per_band_diagnostics"]["A"]["acf_comparison_status"] = "bad_acf_status"
        with self.assertRaises(ValueError) as exc:
            self._lc()._consensus_validate_result_structure(diag)
        self.assertIn("A", str(exc.exception))
        self.assertIn("bad_acf_status", str(exc.exception))

    def test_validator_rejects_invalid_rejection_reason_entry(self):
        diag = _make_valid_diagnostics(accepted_bands=[], rejected_bands=["A"])
        diag["per_band_diagnostics"]["A"]["rejection_reasons"] = ["unknown_reason"]
        diag["per_band_diagnostics"]["A"]["rejection_reason"] = "unknown_reason"
        with self.assertRaises(ValueError) as exc:
            self._lc()._consensus_validate_result_structure(diag)
        self.assertIn("A", str(exc.exception))
        self.assertIn("unknown_reason", str(exc.exception))

    def test_validator_rejects_accepted_status_with_rejection_reasons(self):
        diag = _make_valid_diagnostics(accepted_bands=["A"], rejected_bands=[])
        diag["per_band_diagnostics"]["A"]["status"] = BAND_STATUS_ACCEPTED
        diag["per_band_diagnostics"]["A"]["rejection_reasons"] = [
            REJECTION_REASON_NO_LS_PEAKS
        ]
        diag["per_band_diagnostics"]["A"]["rejection_reason"] = (
            REJECTION_REASON_NO_LS_PEAKS
        )
        with self.assertRaises(ValueError) as exc:
            self._lc()._consensus_validate_result_structure(diag)
        self.assertIn("A", str(exc.exception))
        self.assertIn(BAND_STATUS_ACCEPTED, str(exc.exception))

    def test_validator_rejects_rejected_status_without_reasons(self):
        diag = _make_valid_diagnostics(accepted_bands=[], rejected_bands=["A"])
        diag["per_band_diagnostics"]["A"]["status"] = BAND_STATUS_REJECTED
        diag["per_band_diagnostics"]["A"]["rejection_reasons"] = []
        diag["per_band_diagnostics"]["A"]["rejection_reason"] = None
        with self.assertRaises(ValueError) as exc:
            self._lc()._consensus_validate_result_structure(diag)
        self.assertIn("A", str(exc.exception))
        self.assertIn(BAND_STATUS_REJECTED, str(exc.exception))

    def test_validator_rejects_list_membership_status_mismatch(self):
        diag = _make_valid_diagnostics(accepted_bands=["A"], rejected_bands=[])
        diag["per_band_diagnostics"]["A"]["status"] = BAND_STATUS_REJECTED
        diag["per_band_diagnostics"]["A"]["rejection_reasons"] = [
            REJECTION_REASON_NO_LS_PEAKS
        ]
        diag["per_band_diagnostics"]["A"]["rejection_reason"] = (
            REJECTION_REASON_NO_LS_PEAKS
        )
        with self.assertRaises(ValueError) as exc:
            self._lc()._consensus_validate_result_structure(diag)
        self.assertIn("A", str(exc.exception))
        self.assertIn("accepted_bands", str(exc.exception))

    def test_validator_rejects_harmonic_without_order(self):
        diag = _make_valid_diagnostics(accepted_bands=["A"], rejected_bands=[])
        diag["per_band_diagnostics"]["A"]["acf_comparison_status"] = ACF_STATUS_HARMONIC
        diag["per_band_diagnostics"]["A"]["acf_period_ratio"] = 2.0
        diag["per_band_diagnostics"]["A"]["acf_harmonic_order"] = None
        with self.assertRaises(ValueError) as exc:
            self._lc()._consensus_validate_result_structure(diag)
        self.assertIn("acf_harmonic_order", str(exc.exception))

    def test_validator_rejects_none_status_with_ratio(self):
        diag = _make_valid_diagnostics(accepted_bands=["A"], rejected_bands=[])
        diag["per_band_diagnostics"]["A"]["acf_comparison_status"] = None
        diag["per_band_diagnostics"]["A"]["acf_period_ratio"] = 1.0
        with self.assertRaises(ValueError) as exc:
            self._lc()._consensus_validate_result_structure(diag)
        self.assertIn("acf_comparison_status=None", str(exc.exception))

    def test_validator_rejects_agreement_without_ratio(self):
        diag = _make_valid_diagnostics(accepted_bands=["A"], rejected_bands=[])
        diag["per_band_diagnostics"]["A"]["acf_comparison_status"] = (
            ACF_STATUS_AGREEMENT
        )
        diag["per_band_diagnostics"]["A"]["acf_period_ratio"] = None
        with self.assertRaises(ValueError) as exc:
            self._lc()._consensus_validate_result_structure(diag)
        self.assertIn("acf_period_ratio", str(exc.exception))

    def test_existing_valid_finalized_diagnostics_still_pass(self):
        diag = self._finalized_valid_diag()
        self._lc()._consensus_validate_result_structure(diag)


class TestConsensusCategoricalConstants(unittest.TestCase):
    """Categorical consensus constants exist and are immutable containers."""

    def test_required_categorical_constant_sets_exist(self):
        names = (
            "_CONSENSUS_ALLOWED_GP_VALIDATION_STATUSES",
            "_CONSENSUS_ALLOWED_GP_VALIDATION_REASONS",
            "_CONSENSUS_ALLOWED_ACF_COMPARISON_STATUSES",
            "_CONSENSUS_ALLOWED_BAND_STATUSES",
            "_CONSENSUS_ALLOWED_REJECTION_REASONS",
        )
        for name in names:
            with self.subTest(name=name):
                value = getattr(lightcurve_module, name, None)
                self.assertIsNotNone(value)
                self.assertIsInstance(value, frozenset)

    def test_individual_gp_validation_status_constants_exist(self):
        expected = {
            "_CONSENSUS_GP_VALIDATION_STATUS_NOT_REQUESTED": GP_STATUS_NOT_REQUESTED,
            "_CONSENSUS_GP_VALIDATION_STATUS_SKIPPED": GP_STATUS_SKIPPED,
            "_CONSENSUS_GP_VALIDATION_STATUS_FAILED": GP_STATUS_FAILED,
            "_CONSENSUS_GP_VALIDATION_STATUS_SUCCESS": GP_STATUS_SUCCESS,
            "_CONSENSUS_GP_VALIDATION_STATUS_REJECTED": GP_STATUS_REJECTED,
        }
        for name, expected_value in expected.items():
            with self.subTest(name=name):
                value = getattr(lightcurve_module, name, None)
                self.assertIsNotNone(
                    value, f"{name} constant is missing from lightcurve module"
                )
                self.assertEqual(value, expected_value)

    def test_individual_gp_status_constants_match_allowed_set(self):
        allowed = lightcurve_module._CONSENSUS_ALLOWED_GP_VALIDATION_STATUSES
        individual_names = (
            "_CONSENSUS_GP_VALIDATION_STATUS_NOT_REQUESTED",
            "_CONSENSUS_GP_VALIDATION_STATUS_SKIPPED",
            "_CONSENSUS_GP_VALIDATION_STATUS_FAILED",
            "_CONSENSUS_GP_VALIDATION_STATUS_SUCCESS",
            "_CONSENSUS_GP_VALIDATION_STATUS_REJECTED",
        )
        for name in individual_names:
            val = getattr(lightcurve_module, name)
            with self.subTest(name=name):
                self.assertIn(
                    val,
                    allowed,
                    f"{name}={val!r} is not in "
                    "_CONSENSUS_ALLOWED_GP_VALIDATION_STATUSES",
                )

    def test_allowed_acf_statuses_include_all_known_constants(self):
        allowed = lightcurve_module._CONSENSUS_ALLOWED_ACF_COMPARISON_STATUSES
        expected = {
            lightcurve_module._ACF_STATUS_AGREEMENT,
            lightcurve_module._ACF_STATUS_HARMONIC,
            lightcurve_module._ACF_STATUS_DISAGREEMENT,
            lightcurve_module._ACF_STATUS_UNAVAILABLE,
        }
        self.assertSetEqual(set(allowed), expected)

    def test_allowed_band_statuses_include_all_known_constants(self):
        allowed = lightcurve_module._CONSENSUS_ALLOWED_BAND_STATUSES
        expected = {
            lightcurve_module._CONSENSUS_BAND_STATUS_PENDING,
            lightcurve_module._CONSENSUS_BAND_STATUS_ACCEPTED,
            lightcurve_module._CONSENSUS_BAND_STATUS_REJECTED,
        }
        self.assertSetEqual(set(allowed), expected)

    def test_allowed_rejection_reasons_include_all_known_constants(self):
        allowed = lightcurve_module._CONSENSUS_ALLOWED_REJECTION_REASONS
        expected = {
            lightcurve_module._CONSENSUS_REJECTION_REASON_SAMPLING_METRICS_UNAVAILABLE,
            lightcurve_module._CONSENSUS_REJECTION_REASON_NO_LS_PEAKS,
            lightcurve_module._CONSENSUS_REJECTION_REASON_NO_PLAUSIBLE_LS_PEAK,
            lightcurve_module._CONSENSUS_REJECTION_REASON_CANDIDATE_FREQUENCY_TOO_LOW,
            lightcurve_module._CONSENSUS_REJECTION_REASON_LS_ACF_DISAGREEMENT,
            lightcurve_module._CONSENSUS_REJECTION_REASON_GP_LS_DISAGREEMENT,
            lightcurve_module._CONSENSUS_REJECTION_REASON_GP_VALIDATION_FAILED,
        }
        self.assertSetEqual(set(allowed), expected)

    def test_helpers_reject_values_outside_allowed_sets(self):
        record = _make_minimal_lc()._consensus_initialize_band_record("X")
        with self.assertRaises(ValueError):
            Lightcurve._consensus_set_band_status(record, "not_a_band_status", [])
        with self.assertRaises(ValueError):
            Lightcurve._consensus_set_band_status(
                record,
                BAND_STATUS_REJECTED,
                ["not_a_known_reason"],
            )
        with self.assertRaises(ValueError):
            Lightcurve._consensus_set_acf_comparison_status(
                record,
                "not_an_acf_status",
            )


class TestConsensusSchemaCentralization(unittest.TestCase):
    """Canonical schema definitions remain the single source of truth."""

    def _lc(self):
        return _make_minimal_lc()

    def test_initialized_band_record_contains_all_schema_required_keys(self):
        record = self._lc()._consensus_initialize_band_record("S")
        required = set(lightcurve_module._CONSENSUS_BAND_SCHEMA["required_keys"])
        self.assertSetEqual(set(record.keys()), required)

    def test_required_key_sets_derive_from_schema(self):
        self.assertEqual(
            lightcurve_module._CONSENSUS_REQUIRED_RESULT_KEYS,
            frozenset(lightcurve_module._CONSENSUS_TOP_LEVEL_SCHEMA["required_keys"]),
        )
        self.assertEqual(
            lightcurve_module._CONSENSUS_REQUIRED_BAND_KEYS,
            frozenset(lightcurve_module._CONSENSUS_BAND_SCHEMA["required_keys"]),
        )

    def test_categorical_schema_domains_match_constant_sets(self):
        band_fields = lightcurve_module._CONSENSUS_BAND_SCHEMA["fields"]
        self.assertSetEqual(
            set(band_fields["status"]["allowed_values"]),
            lightcurve_module._CONSENSUS_ALLOWED_BAND_STATUSES,
        )
        self.assertSetEqual(
            set(band_fields["gp_validation_status"]["allowed_values"]),
            lightcurve_module._CONSENSUS_ALLOWED_GP_VALIDATION_STATUSES,
        )
        self.assertSetEqual(
            set(band_fields["acf_comparison_status"]["allowed_values"]),
            lightcurve_module._CONSENSUS_ALLOWED_ACF_COMPARISON_STATUSES,
        )

    def test_deprecated_alias_fields_marked_in_schema(self):
        schema = lightcurve_module._CONSENSUS_TOP_LEVEL_SCHEMA
        self.assertIn("rejection_summary", schema["deprecated_alias_fields"])
        alias_field = schema["fields"]["rejection_summary"]
        self.assertTrue(alias_field["deprecated_alias"])
        self.assertEqual(alias_field["canonical_alias_for"], "rejection_reasons")

    def test_finalized_result_contains_all_schema_required_keys(self):
        finalized = self._lc()._consensus_finalize_result_structure(
            _make_valid_diagnostics(accepted_bands=["A"], rejected_bands=["B"])
        )
        required = set(lightcurve_module._CONSENSUS_TOP_LEVEL_SCHEMA["required_keys"])
        self.assertSetEqual(set(finalized.keys()), required)

    def test_no_duplicate_required_key_definitions_outside_schema_generation(self):
        top_required = tuple(
            lightcurve_module._CONSENSUS_TOP_LEVEL_SCHEMA["required_keys"]
        )
        top_field_order = tuple(
            lightcurve_module._CONSENSUS_TOP_LEVEL_SCHEMA["fields"].keys()
        )
        band_required = tuple(lightcurve_module._CONSENSUS_BAND_SCHEMA["required_keys"])
        band_field_order = tuple(
            lightcurve_module._CONSENSUS_BAND_SCHEMA["fields"].keys()
        )
        self.assertEqual(top_required, top_field_order)
        self.assertEqual(band_required, band_field_order)
        self.assertEqual(len(top_required), len(set(top_required)))
        self.assertEqual(len(band_required), len(set(band_required)))


class TestConsensusSchemaImmutability(unittest.TestCase):
    """Canonical schema definitions are immutable at runtime."""

    def test_top_level_schema_mapping_is_immutable(self):
        schema = lightcurve_module._CONSENSUS_TOP_LEVEL_SCHEMA
        with self.assertRaises(TypeError):
            schema["new_key"] = "bad"

    def test_nested_schema_field_mapping_is_immutable(self):
        field = lightcurve_module._CONSENSUS_BAND_SCHEMA["fields"]["status"]
        with self.assertRaises(TypeError):
            field["nullable"] = True

    def test_required_keys_tuple_is_immutable(self):
        required = lightcurve_module._CONSENSUS_TOP_LEVEL_SCHEMA["required_keys"]
        with self.assertRaises(TypeError):
            required[0] = "bad_key"


# ---------------------------------------------------------------------------
# 10. Recursive-fit protection
# ---------------------------------------------------------------------------

class TestPrepareGpValidationFitKwargs(unittest.TestCase):
    """_consensus_prepare_gp_validation_fit_kwargs always forces fit_strategy=None."""

    def test_fit_strategy_always_none_with_empty_input(self):
        result = Lightcurve._consensus_prepare_gp_validation_fit_kwargs()
        self.assertIsNone(result["fit_strategy"])

    def test_fit_strategy_overridden_even_if_user_sets_it(self):
        result = Lightcurve._consensus_prepare_gp_validation_fit_kwargs(
            {"fit_strategy": "consensus"}
        )
        self.assertIsNone(result["fit_strategy"])

    def test_consensus_only_keys_are_stripped(self):
        blocked = {
            "consensus_frequencies": [0.1, 0.2],
            "use_gp_validation": True,
            "outlier_sigma": 3.0,
            "min_points_per_band": 10,
        }
        result = Lightcurve._consensus_prepare_gp_validation_fit_kwargs(blocked)
        for key in blocked:
            self.assertNotIn(key, result)

    def test_legitimate_gp_kwargs_preserved(self):
        user_kwargs = {"training_iter": 200, "num_mixtures": 2}
        result = Lightcurve._consensus_prepare_gp_validation_fit_kwargs(user_kwargs)
        self.assertEqual(result["training_iter"], 200)
        self.assertEqual(result["num_mixtures"], 2)

    def test_defaults_present_when_no_overrides(self):
        result = Lightcurve._consensus_prepare_gp_validation_fit_kwargs()
        self.assertEqual(result["model"], "1D")
        self.assertIn("num_mixtures", result)
        self.assertIn("use_mls_init", result)
        self.assertIn("training_iter", result)

    def test_period_summary_kwargs_stripped(self):
        result = Lightcurve._consensus_prepare_gp_validation_fit_kwargs(
            {"period_summary_kwargs": {"nfreqs": 500}}
        )
        self.assertNotIn("period_summary_kwargs", result)


# ---------------------------------------------------------------------------
# 11. Execution-path-oriented transition tests
# ---------------------------------------------------------------------------


class TestExecutionPathTransitions(unittest.TestCase):
    """Sequential helper-call chains that simulate the consensus execution path.

    Tests verify that state transitions leave no stale metadata and that the
    finalized result is always structurally valid.
    """

    def _lc(self):
        return _make_minimal_lc()

    # --- rejected → accepted ---

    def test_rejected_to_accepted_clears_rejection_metadata(self):
        """Transitioning a record from rejected to accepted clears reasons."""
        record = self._lc()._consensus_initialize_band_record("X")
        # Reject first
        Lightcurve._consensus_set_band_status(
            record, BAND_STATUS_REJECTED, [REJECTION_REASON_NO_LS_PEAKS]
        )
        self.assertEqual(record["status"], BAND_STATUS_REJECTED)
        self.assertIn(REJECTION_REASON_NO_LS_PEAKS, record["rejection_reasons"])

        # Then accept — reasons must be cleared
        Lightcurve._consensus_set_band_status(record, BAND_STATUS_ACCEPTED, [])
        self.assertEqual(record["status"], BAND_STATUS_ACCEPTED)
        self.assertEqual(record["rejection_reasons"], [])
        self.assertIsNone(record["rejection_reason"])

    def test_rejected_to_accepted_then_re_rejected_is_consistent(self):
        """Record can cycle rejected → accepted → rejected without corruption."""
        record = self._lc()._consensus_initialize_band_record("Y")
        Lightcurve._consensus_set_band_status(
            record, BAND_STATUS_REJECTED, [REJECTION_REASON_NO_LS_PEAKS]
        )
        Lightcurve._consensus_set_band_status(record, BAND_STATUS_ACCEPTED, [])
        Lightcurve._consensus_set_band_status(
            record,
            BAND_STATUS_REJECTED,
            [REJECTION_REASON_NO_PLAUSIBLE_LS_PEAK],
        )
        self.assertEqual(record["status"], BAND_STATUS_REJECTED)
        self.assertEqual(
            record["rejection_reasons"], [REJECTION_REASON_NO_PLAUSIBLE_LS_PEAK]
        )
        self.assertEqual(
            record["rejection_reason"], REJECTION_REASON_NO_PLAUSIBLE_LS_PEAK
        )

    # --- GP failed → success ---

    def test_gp_failed_to_success_clears_reason(self):
        """Transitioning GP status from failed to success clears the reason."""
        record = self._lc()._consensus_initialize_band_record("A")
        Lightcurve._consensus_set_gp_validation_status(
            record,
            GP_STATUS_FAILED,
            reason="exception",
        )
        self.assertEqual(record["gp_validation_status"], GP_STATUS_FAILED)
        self.assertEqual(record["gp_validation_reason"], "exception")

        Lightcurve._consensus_set_gp_validation_status(record, GP_STATUS_SUCCESS)
        self.assertEqual(record["gp_validation_status"], GP_STATUS_SUCCESS)
        self.assertIsNone(record["gp_validation_reason"])

    def test_gp_rejected_then_failed_reason_updated(self):
        """GP status can transition from rejected to failed with new reason."""
        record = self._lc()._consensus_initialize_band_record("B")
        Lightcurve._consensus_set_gp_validation_status(
            record,
            GP_STATUS_REJECTED,
            reason="diagnostics_failed",
        )
        Lightcurve._consensus_set_gp_validation_status(
            record,
            GP_STATUS_FAILED,
            reason="exception",
        )
        self.assertEqual(record["gp_validation_status"], GP_STATUS_FAILED)
        self.assertEqual(record["gp_validation_reason"], "exception")

    # --- ACF harmonic → unavailable ---

    def test_acf_harmonic_to_unavailable_clears_metadata(self):
        """Transitioning ACF status from harmonic to unavailable clears metadata."""
        record = self._lc()._consensus_initialize_band_record("C")
        Lightcurve._consensus_set_acf_comparison_status(
            record,
            ACF_STATUS_HARMONIC,
            period_ratio=2.0,
            harmonic_order=2,
        )
        self.assertEqual(record["acf_comparison_status"], ACF_STATUS_HARMONIC)
        self.assertEqual(record["acf_harmonic_order"], 2)
        self.assertAlmostEqual(record["acf_period_ratio"], 2.0)

        Lightcurve._consensus_set_acf_comparison_status(
            record, ACF_STATUS_UNAVAILABLE
        )
        self.assertEqual(record["acf_comparison_status"], ACF_STATUS_UNAVAILABLE)
        self.assertIsNone(record["acf_harmonic_order"])
        self.assertIsNone(record["acf_period_ratio"])

    def test_acf_disagreement_to_agreement_updates_ratio(self):
        """ACF metadata is overwritten cleanly when transitioning statuses."""
        record = self._lc()._consensus_initialize_band_record("D")
        Lightcurve._consensus_set_acf_comparison_status(
            record,
            ACF_STATUS_DISAGREEMENT,
            period_ratio=5.0,
        )
        Lightcurve._consensus_set_acf_comparison_status(
            record,
            ACF_STATUS_AGREEMENT,
            period_ratio=1.05,
        )
        self.assertEqual(record["acf_comparison_status"], ACF_STATUS_AGREEMENT)
        self.assertAlmostEqual(record["acf_period_ratio"], 1.05)
        self.assertIsNone(record["acf_harmonic_order"])

    # --- _consensus_add_rejection_reasons deduplication ---

    def test_add_rejection_reasons_no_duplicates(self):
        """_consensus_add_rejection_reasons deduplicates across multiple calls."""
        lc = self._lc()
        record = lc._consensus_initialize_band_record("E")
        lc._consensus_add_rejection_reasons(
            record, [REJECTION_REASON_NO_LS_PEAKS]
        )
        lc._consensus_add_rejection_reasons(
            record, [REJECTION_REASON_NO_LS_PEAKS]
        )
        lc._consensus_add_rejection_reasons(
            record,
            [REJECTION_REASON_NO_LS_PEAKS, REJECTION_REASON_NO_PLAUSIBLE_LS_PEAK],
        )
        reasons = record["rejection_reasons"]
        self.assertEqual(
            reasons.count(REJECTION_REASON_NO_LS_PEAKS), 1
        )
        self.assertEqual(
            reasons.count(REJECTION_REASON_NO_PLAUSIBLE_LS_PEAK), 1
        )

    def test_add_rejection_reasons_sets_band_status_rejected(self):
        """Adding any rejection reason marks the band as rejected."""
        lc = self._lc()
        record = lc._consensus_initialize_band_record("F")
        lc._consensus_add_rejection_reasons(
            record, [REJECTION_REASON_NO_LS_PEAKS]
        )
        self.assertEqual(record["status"], BAND_STATUS_REJECTED)

    # --- finalized output always has both keys, synchronized ---

    def test_finalized_result_has_both_rejection_keys(self):
        """Finalized diagnostics always contain both rejection_reasons and
        rejection_summary."""
        diag = _make_valid_diagnostics(accepted_bands=[], rejected_bands=["G"])
        diag["rejection_reasons"] = {REJECTION_REASON_NO_LS_PEAKS: ["G"]}
        result = self._lc()._consensus_finalize_result_structure(diag)
        self.assertIn("rejection_reasons", result)
        self.assertIn("rejection_summary", result)
        self.assertEqual(result["rejection_reasons"], result["rejection_summary"])

    def test_finalized_both_keys_synchronized_after_migration(self):
        """Keys stay synchronized when only rejection_summary is provided."""
        diag = _make_valid_diagnostics(accepted_bands=[], rejected_bands=["H"])
        diag["rejection_summary"] = {
            REJECTION_REASON_NO_PLAUSIBLE_LS_PEAK: ["H"]
        }
        result = self._lc()._consensus_finalize_result_structure(diag)
        self.assertEqual(
            result["rejection_reasons"],
            {REJECTION_REASON_NO_PLAUSIBLE_LS_PEAK: ["H"]},
        )
        self.assertEqual(result["rejection_summary"], result["rejection_reasons"])


# ---------------------------------------------------------------------------
# 12. _consensus_build_rejection_summary format-conversion tests
# ---------------------------------------------------------------------------


class TestBuildRejectionSummary(unittest.TestCase):
    """_consensus_build_rejection_summary converts {band:[reasons]} to
    {reason:[bands]}."""

    def _lc(self):
        return _make_minimal_lc()

    def test_single_band_single_reason(self):
        result = self._lc()._consensus_build_rejection_summary(
            per_band_diagnostics={
                "A": _make_band_record(
                    band="A",
                    band_status=BAND_STATUS_REJECTED,
                    rejection_reasons=[REJECTION_REASON_NO_LS_PEAKS],
                )
            },
            rejected_bands=["A"],
        )
        self.assertEqual(result, {REJECTION_REASON_NO_LS_PEAKS: ["A"]})

    def test_multiple_bands_same_reason_grouped(self):
        """Two bands with the same reason appear in the same list."""
        result = self._lc()._consensus_build_rejection_summary(
            per_band_diagnostics={
                "A": _make_band_record(
                    band="A",
                    band_status=BAND_STATUS_REJECTED,
                    rejection_reasons=[REJECTION_REASON_NO_LS_PEAKS],
                ),
                "B": _make_band_record(
                    band="B",
                    band_status=BAND_STATUS_REJECTED,
                    rejection_reasons=[REJECTION_REASON_NO_LS_PEAKS],
                ),
            },
            rejected_bands=["A", "B"],
        )
        self.assertIn(REJECTION_REASON_NO_LS_PEAKS, result)
        self.assertEqual(sorted(result[REJECTION_REASON_NO_LS_PEAKS]), ["A", "B"])

    def test_multiple_bands_different_reasons(self):
        """Bands with different reasons produce separate mapping entries."""
        result = self._lc()._consensus_build_rejection_summary(
            per_band_diagnostics={
                "A": _make_band_record(
                    band="A",
                    band_status=BAND_STATUS_REJECTED,
                    rejection_reasons=[REJECTION_REASON_NO_LS_PEAKS],
                ),
                "B": _make_band_record(
                    band="B",
                    band_status=BAND_STATUS_REJECTED,
                    rejection_reasons=[REJECTION_REASON_NO_PLAUSIBLE_LS_PEAK],
                ),
            },
            rejected_bands=["A", "B"],
        )
        self.assertEqual(result[REJECTION_REASON_NO_LS_PEAKS], ["A"])
        self.assertEqual(
            result[REJECTION_REASON_NO_PLAUSIBLE_LS_PEAK], ["B"]
        )

    def test_band_not_in_per_band_diagnostics_uses_accumulator(self):
        """Bands in rejection_reasons accumulator but missing from
        per_band_diagnostics are still mapped."""
        result = self._lc()._consensus_build_rejection_summary(
            per_band_diagnostics={},
            rejected_bands=["C"],
            rejection_reasons={"C": [REJECTION_REASON_NO_LS_PEAKS]},
        )
        self.assertIn(REJECTION_REASON_NO_LS_PEAKS, result)
        self.assertIn("C", result[REJECTION_REASON_NO_LS_PEAKS])

    def test_empty_rejected_bands_returns_empty(self):
        result = self._lc()._consensus_build_rejection_summary(
            per_band_diagnostics={},
            rejected_bands=[],
        )
        self.assertEqual(result, {})

    def test_output_passable_to_set_top_level_rejection_reasons(self):
        """Output of build_rejection_summary is in {reason:[bands]} format
        accepted by _consensus_set_top_level_rejection_reasons."""
        lc = self._lc()
        summary = lc._consensus_build_rejection_summary(
            per_band_diagnostics={
                "A": _make_band_record(
                    band="A",
                    band_status=BAND_STATUS_REJECTED,
                    rejection_reasons=[REJECTION_REASON_NO_LS_PEAKS],
                )
            },
            rejected_bands=["A"],
        )
        holder = {}
        # Should not raise
        lc._consensus_set_top_level_rejection_reasons(holder, summary)
        self.assertEqual(
            holder["rejection_reasons"][REJECTION_REASON_NO_LS_PEAKS], ["A"]
        )


# ---------------------------------------------------------------------------
# 13. _consensus_debug_checkpoint tests
# ---------------------------------------------------------------------------


class TestConsensusDebugCheckpoint(unittest.TestCase):
    """_consensus_debug_checkpoint validates intermediate diagnostics snapshots."""

    def _lc(self):
        return _make_minimal_lc()

    def test_no_op_when_debug_flag_off(self):
        """Checkpoint is a no-op when _CONSENSUS_DEBUG_VALIDATE is False."""
        import pgmuvi.lightcurve as lc_mod

        original = lc_mod._CONSENSUS_DEBUG_VALIDATE
        try:
            lc_mod._CONSENSUS_DEBUG_VALIDATE = False
            # Even invalid diagnostics must not raise when flag is off.
            self._lc()._consensus_debug_checkpoint({}, "test-label")
        finally:
            lc_mod._CONSENSUS_DEBUG_VALIDATE = original

    def test_valid_state_passes_when_debug_flag_on(self):
        """Checkpoint passes for a valid intermediate diagnostics dict."""
        import pgmuvi.lightcurve as lc_mod

        original = lc_mod._CONSENSUS_DEBUG_VALIDATE
        try:
            lc_mod._CONSENSUS_DEBUG_VALIDATE = True
            valid_diag = _make_valid_diagnostics(
                accepted_bands=["A"], rejected_bands=[]
            )
            # Should not raise
            self._lc()._consensus_debug_checkpoint(valid_diag, "test-pass")
        finally:
            lc_mod._CONSENSUS_DEBUG_VALIDATE = original

    def test_invalid_state_raises_assertion_error_when_debug_flag_on(self):
        """Checkpoint raises AssertionError for structurally invalid state."""
        import pgmuvi.lightcurve as lc_mod

        original = lc_mod._CONSENSUS_DEBUG_VALIDATE
        try:
            lc_mod._CONSENSUS_DEBUG_VALIDATE = True
            # Build a diagnostics dict where per_band_diagnostics contains an
            # extra band not present in accepted_bands or rejected_bands.
            # Finalization does not remove extra per-band entries, so the
            # validator will raise for this structural violation.
            bad_diag = _make_valid_diagnostics(
                accepted_bands=["A"], rejected_bands=[]
            )
            bad_diag["per_band_diagnostics"]["Z"] = _make_band_record(band="Z")
            with self.assertRaises(AssertionError) as ctx:
                self._lc()._consensus_debug_checkpoint(bad_diag, "bad-label")
            self.assertIn("bad-label", str(ctx.exception))
        finally:
            lc_mod._CONSENSUS_DEBUG_VALIDATE = original

    def test_debug_flag_constant_exposed_on_module(self):
        """_CONSENSUS_DEBUG_VALIDATE is accessible as a module attribute."""
        import pgmuvi.lightcurve as lc_mod

        self.assertFalse(
            lc_mod._CONSENSUS_DEBUG_VALIDATE,
            "_CONSENSUS_DEBUG_VALIDATE should default to False",
        )


class TestStagedConsensusDebugCheckpoints(unittest.TestCase):
    """Execution-path tests for staged debug checkpoint labels."""

    def _lc(self):
        return _make_minimal_lc()

    def test_collect_band_candidates_emits_ls_and_acf_stage_labels(self):
        """Collect stage emits LS and ACF checkpoint labels when ACF runs."""
        import pgmuvi.lightcurve as lc_mod
        import torch

        class _FakeBandLightcurve:
            def compute_sampling_metrics(self):
                return {
                    "n_points": 20,
                    "baseline": 10.0,
                    "longest_detectable_period": 5.0,
                    "nyquist_frequency": 5.0,
                }

            def fit_LS(self, num_peaks=5):
                return torch.tensor([1.0]), torch.tensor([True])

            def acf(self, method="data", normalize=True):
                return {"dummy": True}

        lc = self._lc()
        labels = []
        original_flag = lc_mod._CONSENSUS_DEBUG_VALIDATE
        original_checkpoint = lc._consensus_debug_checkpoint
        try:
            lc_mod._CONSENSUS_DEBUG_VALIDATE = True

            def _recording_checkpoint(result_diagnostics, label):
                labels.append(label)
                return original_checkpoint(result_diagnostics, label)

            with unittest.mock.patch.object(
                lc,
                "_consensus_debug_checkpoint",
                side_effect=_recording_checkpoint,
            ), unittest.mock.patch.object(
                lc,
                "_consensus_iter_band_lightcurves",
                return_value=[("B", _FakeBandLightcurve())],
            ), unittest.mock.patch.object(
                lc,
                "_consensus_resolve_controls",
                return_value={
                    "min_points_per_band": 5,
                    "max_gap_fraction": 0.5,
                    "min_duty_cycle": 0.05,
                    "outlier_sigma": 3.5,
                    "consensus_width_factor": 3.0,
                },
            ), unittest.mock.patch.object(
                lc,
                "_consensus_reject_bad_bands",
                return_value=[],
            ), unittest.mock.patch.object(
                lc,
                "_consensus_compare_ls_acf",
                return_value={
                    "status": ACF_STATUS_AGREEMENT,
                    "ratio": 1.0,
                    "harmonic_order": None,
                },
            ), unittest.mock.patch.object(
                lc,
                "_consensus_extract_acf_candidate",
                return_value={"frequency": 1.0, "period": 1.0},
            ):
                lc._consensus_collect_band_candidates(
                    use_acf=True,
                    gp_validation_requested=False,
                )
        finally:
            lc_mod._CONSENSUS_DEBUG_VALIDATE = original_flag

        self.assertIn("after_per_band_ls_candidate_extraction", labels)
        self.assertIn("after_acf_validation_comparison", labels)

    def test_standard_fit_emits_gp_and_pre_finalize_stage_labels(self):
        """Standard consensus fit emits GP and pre-consensus/finalization labels."""
        import pgmuvi.lightcurve as lc_mod

        lc = self._lc()
        lc.model = object()
        lc._model_pars = {}
        labels = []
        original_flag = lc_mod._CONSENSUS_DEBUG_VALIDATE
        original_checkpoint = lc._consensus_debug_checkpoint
        try:
            lc_mod._CONSENSUS_DEBUG_VALIDATE = True

            def _recording_checkpoint(result_diagnostics, label):
                labels.append(label)
                return original_checkpoint(result_diagnostics, label)

            candidate_diag = {
                "controls": {
                    "outlier_sigma": 3.5,
                    "consensus_width_factor": 3.0,
                },
                "band_records": {
                    "B": {
                        **lc._consensus_initialize_band_record("B"),
                        "status": BAND_STATUS_ACCEPTED,
                        "dominant_frequency": 1.0,
                        "dominant_period": 1.0,
                        "gp_validation_used": True,
                    }
                },
                "accepted_bands": ["B"],
                "rejected_bands": [],
                "rejection_reasons": {},
            }
            consensus_diag = {
                "frequencies_all": [1.0],
                "inlier_bands": ["B"],
                "outlier_bands": [],
                "final_consensus_frequency": 1.0,
                "final_mad_frequency_scatter": 0.1,
                "mad_frequency_scatter": 0.1,
                "median_frequency": 1.0,
            }

            with unittest.mock.patch.object(
                lc,
                "_consensus_debug_checkpoint",
                side_effect=_recording_checkpoint,
            ), unittest.mock.patch(
                "pgmuvi.lightcurve.Lightcurve.ndim",
                new_callable=unittest.mock.PropertyMock,
                return_value=2,
            ), unittest.mock.patch.object(
                lc,
                "_consensus_collect_band_candidates",
                return_value=candidate_diag,
            ), unittest.mock.patch.object(
                lc,
                "_consensus_validate_candidates_with_1d_gp",
                return_value=candidate_diag,
            ), unittest.mock.patch.object(
                lc,
                "_consensus_build_frequency_consensus",
                return_value=consensus_diag,
            ), unittest.mock.patch.object(
                lc,
                "_consensus_build_guess",
                return_value={"dummy_param": 1.0},
            ), unittest.mock.patch.object(
                lc,
                "fit",
                return_value=None,
            ):
                lc._consensus_standard_fit(
                    use_gp_validation=True,
                    constrain_consensus=False,
                    verbose=False,
                )
        finally:
            lc_mod._CONSENSUS_DEBUG_VALIDATE = original_flag

        self.assertIn("after_gp_validation", labels)
        self.assertIn("before_consensus_frequency_generation", labels)
        self.assertIn("before_finalization", labels)


if __name__ == "__main__":
    unittest.main()
