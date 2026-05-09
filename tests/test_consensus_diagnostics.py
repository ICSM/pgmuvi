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
import unittest

import numpy as np

from pgmuvi.lightcurve import Lightcurve


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
    status="not_requested",
    reason=None,
    rejection_reasons=None,
):
    """Return a band record that satisfies _CONSENSUS_REQUIRED_BAND_KEYS."""
    record = {
        "band": band,
        "status": status,
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
        self.assertEqual(record["gp_validation_status"], "not_requested")

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
        for status in ("not_requested", "skipped", "failed", "success", "rejected"):
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
            record, "failed", reason="exception"
        )
        self.assertEqual(record["gp_validation_reason"], "exception")

    def test_stale_reason_cleared_on_success(self):
        record = self._fresh_record()
        # First inject a stale reason via a failed status.
        Lightcurve._consensus_set_gp_validation_status(
            record, "failed", reason="exception"
        )
        self.assertEqual(record["gp_validation_reason"], "exception")
        # Transition to success without a reason; stale value must be cleared.
        Lightcurve._consensus_set_gp_validation_status(record, "success")
        self.assertIsNone(record["gp_validation_reason"])

    def test_stale_reason_cleared_on_not_requested(self):
        record = self._fresh_record()
        Lightcurve._consensus_set_gp_validation_status(
            record, "skipped", reason="band_not_accepted"
        )
        self.assertEqual(record["gp_validation_reason"], "band_not_accepted")
        Lightcurve._consensus_set_gp_validation_status(record, "not_requested")
        self.assertIsNone(record["gp_validation_reason"])

    def test_reason_preserved_on_other_statuses_without_explicit_reason(self):
        """If no reason is supplied and status is not success/not_requested,
        the existing reason should not be cleared."""
        record = self._fresh_record()
        Lightcurve._consensus_set_gp_validation_status(
            record, "rejected", reason="diagnostics_failed"
        )
        # Calling again with same status and no reason preserves the reason.
        Lightcurve._consensus_set_gp_validation_status(record, "rejected")
        self.assertEqual(record["gp_validation_reason"], "diagnostics_failed")


# ---------------------------------------------------------------------------
# 3. Invalid GP status/reason combinations (validator)
# ---------------------------------------------------------------------------

class TestValidateGpReasonStatusCombinations(unittest.TestCase):
    """_consensus_validate_result_structure rejects forbidden reason/status pairs."""

    def _validate(self, diag):
        _make_minimal_lc()._consensus_validate_result_structure(diag)

    def _diag_with_band_record(self, record):
        band = record["band"]
        return _make_valid_diagnostics(
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
        record = _make_band_record(band="A", status="success", reason="exception")
        self._assert_validation_raises(self._diag_with_band_record(record))

    def test_not_requested_with_band_not_accepted_reason_raises(self):
        record = _make_band_record(
            band="A", status="not_requested", reason="band_not_accepted"
        )
        self._assert_validation_raises(self._diag_with_band_record(record))

    def test_rejected_with_exception_reason_raises(self):
        record = _make_band_record(band="A", status="rejected", reason="exception")
        self._assert_validation_raises(self._diag_with_band_record(record))

    def test_failed_with_diagnostics_failed_reason_raises(self):
        record = _make_band_record(
            band="A", status="failed", reason="diagnostics_failed"
        )
        self._assert_validation_raises(self._diag_with_band_record(record))

    # --- valid combos ----

    def test_not_requested_with_none_reason_passes(self):
        record = _make_band_record(band="A", status="not_requested", reason=None)
        self._validate(self._diag_with_band_record(record))

    def test_success_with_none_reason_passes(self):
        record = _make_band_record(band="A", status="success", reason=None)
        self._validate(self._diag_with_band_record(record))

    def test_skipped_with_band_not_accepted_reason_passes(self):
        record = _make_band_record(
            band="A", status="skipped", reason="band_not_accepted"
        )
        self._validate(self._diag_with_band_record(record))

    def test_rejected_with_diagnostics_failed_reason_passes(self):
        record = _make_band_record(
            band="A", status="rejected", reason="diagnostics_failed"
        )
        self._validate(self._diag_with_band_record(record))

    def test_failed_with_exception_reason_passes(self):
        record = _make_band_record(band="A", status="failed", reason="exception")
        self._validate(self._diag_with_band_record(record))

    def test_error_message_includes_band_label(self):
        record = _make_band_record(
            band="MY_BAND", status="success", reason="exception"
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
    """Validator enforces rejection_summary only references rejected_bands."""

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
            rejection_summary={"low_snr": ["C"]},  # C not in rejected
        )
        self._assert_raises(diag)

    def test_accepted_band_in_summary_raises(self):
        diag = _make_valid_diagnostics(
            accepted_bands=["A"],
            rejected_bands=["B"],
            rejection_summary={"low_snr": ["A"]},  # A is accepted
        )
        self._assert_raises(diag)

    def test_duplicate_band_in_summary_list_raises(self):
        diag = _make_valid_diagnostics(
            accepted_bands=[],
            rejected_bands=["B"],
            rejection_summary={"some_reason": ["B", "B"]},
        )
        self._assert_raises(diag)

    def test_valid_rejection_summary_passes(self):
        diag = _make_valid_diagnostics(
            accepted_bands=["A"],
            rejected_bands=["B"],
            rejection_summary={"low_snr": ["B"]},
        )
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
# 9. Recursive-fit protection
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


if __name__ == "__main__":
    unittest.main()
