"""Tests for canonical wavelength-advisory attempt and failure statuses."""

import unittest
import warnings
from unittest import mock

from pgmuvi.lightcurve import FitFailureSummary, Lightcurve
from pgmuvi.wavelength_diagnostics import (
    _piwd_extract_fit_outcome,
    run_period_independent_wavelength_model_kernel_configs,
)


class _CandidateLightcurve:
    def __init__(self):
        self.consensus_diagnostics = {"consensus_success": True}
        self.results = None


class TestAttemptOutcomeCompatibility(unittest.TestCase):
    def setUp(self):
        self.candidate = {
            "model_kernel_config_id": "cfg",
            "rank": 1,
            "model": "2DDustMean",
            "fit_kwargs": {"training_iter": 0},
        }

    def test_legacy_fields_remain_while_canonical_fields_are_added(self):
        outcome = _piwd_extract_fit_outcome(
            self.candidate,
            status="passed",
            fitted_lightcurve=_CandidateLightcurve(),
            fit_result={},
        )
        self.assertEqual(outcome["status"], "passed")
        self.assertTrue(outcome["fit_success"])
        self.assertFalse(outcome["fit_failed"])
        self.assertEqual(outcome["attempt_disposition"], "attempted")
        self.assertEqual(outcome["technical_outcome"], "initialized_only")
        self.assertEqual(outcome["comparison_eligibility"], "ineligible")

    def test_recovery_metadata_changes_technical_outcome(self):
        candidate = dict(self.candidate)
        candidate["fit_kwargs"] = {"training_iter": 20}
        outcome = _piwd_extract_fit_outcome(
            candidate,
            status="passed",
            fitted_lightcurve=_CandidateLightcurve(),
            fit_result={"training_recovered_from_failure": True},
        )
        self.assertEqual(outcome["technical_outcome"], "completed_with_recovery")
        self.assertTrue(outcome["training_recovered_from_failure"])

    def test_structured_failure_diagnostics_and_summary_survive(self):
        class StructuredFailure(RuntimeError):
            pass

        exc = StructuredFailure("consensus failed")
        exc.failure_diagnostics = {
            "status": "failed",
            "reason": "no_accepted_bands",
            "failure_stage": "band_quality",
        }
        exc.failure_summary = FitFailureSummary(
            reason="no_accepted_bands",
            message="consensus failed",
            failure_code="no_accepted_bands",
            stage="consensus",
            substage="band_quality",
            exception_type="StructuredFailure",
        )
        outcome = _piwd_extract_fit_outcome(
            self.candidate,
            status="failed",
            fitted_lightcurve=_CandidateLightcurve(),
            exception=exc,
        )
        self.assertEqual(outcome["technical_outcome"], "failed")
        self.assertEqual(outcome["failure_code"], "no_accepted_bands")
        self.assertEqual(outcome["failure_stage"], "consensus")
        self.assertEqual(outcome["failure_substage"], "band_quality")
        self.assertEqual(
            outcome["structured_failure_diagnostics"]["reason"],
            "no_accepted_bands",
        )
        self.assertEqual(
            outcome["failure_summary"]["failure_code"],
            "no_accepted_bands",
        )
        self.assertEqual(outcome["traceback_reference"], "inline:traceback")
        self.assertEqual(outcome["diagnostic_validity"], "unavailable")

    def test_consensus_failure_with_band_evidence_is_partial(self):
        class StructuredFailure(RuntimeError):
            pass

        exc = StructuredFailure("consensus failed")
        exc.failure_diagnostics = {
            "reason": "no_accepted_bands",
            "failure_stage": "band_quality",
            "accepted_bands": [],
            "rejected_bands": ["g", "r"],
            "per_band_diagnostics": {"g": {}, "r": {}},
        }
        outcome = _piwd_extract_fit_outcome(
            self.candidate,
            status="failed",
            fitted_lightcurve=_CandidateLightcurve(),
            exception=exc,
        )
        self.assertEqual(outcome["diagnostic_validity"], "partial")
        self.assertEqual(outcome["scientific_usability"], "unusable")
        self.assertEqual(outcome["comparison_eligibility"], "ineligible")


class TestRunnerWarningAndOutcomeCounts(unittest.TestCase):
    def test_runner_captures_warnings_and_counts_outcomes(self):
        report = {
            "kind": "period_independent_wavelength_model_kernel_configs",
            "model_kernel_configs": [
                {
                    "model_kernel_config_id": "cfg",
                    "rank": 1,
                    "model": "2DDustMean",
                    "fit_kwargs": {"model": "2DDustMean", "training_iter": 5},
                }
            ],
        }
        source = _CandidateLightcurve()

        def runner(candidate_lc, fit_kwargs, candidate):
            warnings.warn("synthetic warning", UserWarning)
            candidate_lc.consensus_diagnostics = {"consensus_success": True}
            return {}

        with self.assertWarnsRegex(UserWarning, "synthetic warning"):
            outcome_report = (
                run_period_independent_wavelength_model_kernel_configs(
                    source,
                    model_kernel_config_report=report,
                    fit_runner=runner,
                )
            )
        outcome = outcome_report["outcomes"][0]
        self.assertEqual(outcome["technical_outcome"], "completed_with_warnings")
        self.assertEqual(outcome["warning_count"], 1)
        self.assertEqual(outcome["warning_severity"], "warning")
        self.assertEqual(outcome["warning_records"][0]["category"], "UserWarning")
        self.assertIsInstance(outcome["warning_records"][0]["lineno"], int)
        self.assertEqual(
            outcome_report["technical_outcome_counts"],
            {"completed_with_warnings": 1},
        )


class TestOuterFitFailureState(unittest.TestCase):
    def test_non_consensus_fit_exception_sets_canonical_failure_state(self):
        lc = Lightcurve([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
        with mock.patch.object(
            lc,
            "_fit_core",
            side_effect=ValueError("synthetic outer failure"),
        ):
            with self.assertRaisesRegex(ValueError, "synthetic outer failure"):
                lc.fit(model="1D", training_iter=1)

        self.assertTrue(lc.fit_failed)
        self.assertFalse(lc.is_fitted)
        self.assertEqual(lc.failure_reason, "fit_exception")
        self.assertIsNotNone(lc.failure_summary)
        summary = lc.failure_summary.to_dict()
        self.assertEqual(summary["failure_code"], "fit_execution_failed")
        self.assertEqual(summary["stage"], "fit_execution")
        self.assertEqual(summary["substage"], "outer_fit")
        self.assertEqual(summary["exception_type"], "ValueError")
        self.assertEqual(
            summary["traceback_reference"],
            "failure_diagnostics.traceback",
        )
        self.assertIn("ValueError: synthetic outer failure", summary["diagnostics"]["traceback"])
        self.assertFalse(
            summary["diagnostics"].get("is_consensus_failure")
        )


if __name__ == "__main__":
    unittest.main()
