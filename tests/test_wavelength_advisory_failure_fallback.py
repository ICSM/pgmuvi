"""Regression tests for advisory failure fallback summaries."""

import unittest

from pgmuvi.wavelength_diagnostics import (
    _piwd_batch_extract_model_kernel_config_rows,
    _piwd_build_advisory_workflow_fallback_summary,
    _piwd_classify_model_kernel_config_failure,
)


class TestWavelengthAdvisoryFailureFallback(unittest.TestCase):
    def test_consensus_failure_is_classified_from_exception_text(self):
        exc = RuntimeError("ConsensusFitError: no common period consensus was found")

        info = _piwd_classify_model_kernel_config_failure(exc)

        self.assertEqual(info["failure_stage"], "consensus")
        self.assertIs(info["is_consensus_failure"], True)
        self.assertIs(info["is_numerical_failure"], False)

    def test_numerical_failure_is_classified_from_psd_text(self):
        exc = RuntimeError("NotPSDError: matrix is not positive definite")

        info = _piwd_classify_model_kernel_config_failure(exc)

        self.assertEqual(info["failure_stage"], "numerical_stability")
        self.assertIs(info["is_numerical_failure"], True)

    def test_fallback_summary_activates_when_all_model_configs_fail(self):
        run_report = {
            "outcomes": [
                {
                    "status": "failed",
                    "fit_failed": True,
                    "model": "2D",
                    "failure_stage": "consensus",
                    "is_consensus_failure": True,
                    "exception_type": "ConsensusFitError",
                },
                {
                    "status": "failed",
                    "fit_failed": True,
                    "model": "2DWavelengthDependent",
                    "failure_stage": "numerical_stability",
                    "is_numerical_failure": True,
                    "exception_type": "NotPSDError",
                },
            ]
        }

        fallback = _piwd_build_advisory_workflow_fallback_summary(
            run_report,
            {
                "ranking_status": "unavailable",
                "fit_quality_ranking_available": False,
                "n_with_fit_quality": 0,
                "top_ranked_model": "stale-model",
            },
        )

        self.assertIs(fallback["available"], True)
        self.assertEqual(fallback["reason"], "all_model_kernel_config_fits_failed")
        self.assertIsNone(fallback["selected_model"])
        self.assertIsNone(fallback["top_ranked_model"])
        self.assertEqual(fallback["fit_quality_ranking_status"], "unavailable")
        self.assertIs(fallback["fit_based_model_ranking_available"], False)
        self.assertIs(fallback["automatic_model_selection_applied"], False)
        self.assertEqual(
            fallback["failure_stage_counts"],
            {"consensus": 1, "numerical_stability": 1},
        )
        self.assertEqual(
            fallback["exception_type_counts"],
            {"ConsensusFitError": 1, "NotPSDError": 1},
        )
        self.assertEqual(fallback["consensus_failure_models"], ["2D"])
        self.assertEqual(fallback["n_numerical_failure_models"], 1)
        self.assertTrue(fallback["recommended_next_steps"])

    def test_fallback_summary_surfaces_canonical_status_counts(self):
        run_report = {
            "outcomes": [
                {
                    "status": "passed",
                    "fit_success": True,
                    "model": "2DDustMean",
                    "technical_outcome": "initialized_only",
                    "comparison_eligibility": "ineligible",
                },
                {
                    "status": "failed",
                    "fit_failed": True,
                    "model": "2D",
                    "technical_outcome": "failed",
                    "comparison_eligibility": "ineligible",
                    "failure_code": "no_accepted_bands",
                },
            ]
        }
        quality_report = {
            "ranking_status": "unavailable",
            "fit_quality_ranking_available": False,
            "n_with_fit_quality": 0,
        }

        fallback = _piwd_build_advisory_workflow_fallback_summary(
            run_report, quality_report
        )

        self.assertEqual(
            fallback["technical_outcome_counts"],
            {"initialized_only": 1, "failed": 1},
        )
        self.assertEqual(
            fallback["failure_code_counts"],
            {"no_accepted_bands": 1},
        )
        self.assertEqual(fallback["initialized_only_models"], ["2DDustMean"])
        self.assertEqual(
            fallback["comparison_ineligible_models"],
            ["2DDustMean", "2D"],
        )

    def test_fallback_summary_is_inactive_when_comparative_ranking_exists(self):
        run_report = {
            "outcomes": [
                {"status": "passed", "fit_success": True, "model": "2DDustMean"},
                {
                    "status": "passed",
                    "fit_success": True,
                    "model": "2DWavelengthDependent",
                },
            ]
        }
        quality_report = {
            "ranking_status": "available",
            "fit_quality_ranking_available": True,
            "n_with_fit_quality": 2,
            "top_ranked_model": "2DDustMean",
        }

        fallback = _piwd_build_advisory_workflow_fallback_summary(
            run_report, quality_report
        )

        self.assertIs(fallback["available"], False)
        self.assertEqual(fallback["reason"], "fit_quality_ranking_available")
        self.assertIs(fallback["fit_based_model_ranking_available"], True)
        self.assertEqual(fallback["top_ranked_model"], "2DDustMean")

    def test_single_valid_candidate_activates_noncomparative_fallback(self):
        run_report = {
            "outcomes": [
                {"status": "passed", "fit_success": True, "model": "2DDustMean"},
                {"status": "failed", "fit_failed": True, "model": "2D"},
            ]
        }
        quality_report = {
            "ranking_status": "single_valid_candidate",
            "fit_quality_ranking_available": False,
            "n_with_fit_quality": 1,
            "only_valid_model": "2DDustMean",
        }

        fallback = _piwd_build_advisory_workflow_fallback_summary(
            run_report, quality_report
        )

        self.assertIs(fallback["available"], True)
        self.assertEqual(
            fallback["reason"], "only_one_model_kernel_config_has_fit_quality"
        )
        self.assertIsNone(fallback["top_ranked_model"])
        self.assertEqual(fallback["only_valid_model"], "2DDustMean")

    def test_passed_but_unscored_candidates_activate_fallback(self):
        run_report = {
            "outcomes": [
                {"status": "passed", "fit_success": True, "model": "2DDustMean"},
                {"status": "passed", "fit_success": True, "model": "2D"},
            ]
        }
        quality_report = {
            "ranking_status": "unavailable",
            "fit_quality_ranking_available": False,
            "n_with_fit_quality": 0,
        }

        fallback = _piwd_build_advisory_workflow_fallback_summary(
            run_report, quality_report
        )

        self.assertIs(fallback["available"], True)
        self.assertEqual(
            fallback["reason"], "no_model_kernel_config_fit_quality_available"
        )
        self.assertIsNone(fallback["top_ranked_model"])

    def test_batch_model_kernel_rows_surface_failure_classification(self):
        workflow = {
            "kind": "period_independent_wavelength_advisory_workflow",
            "run_report": {
                "outcomes": [
                    {
                        "model_kernel_config_id": "rank1_2D",
                        "rank": 1,
                        "model": "2D",
                        "status": "failed",
                        "attempt_disposition": "attempted",
                        "execution_stage": "consensus",
                        "technical_outcome": "failed",
                        "diagnostic_validity": "partial",
                        "scientific_usability": "unusable",
                        "comparison_eligibility": "ineligible",
                        "warning_severity": "error",
                        "warning_count": 0,
                        "fit_failed": True,
                        "failure_code": "no_accepted_bands",
                        "failure_stage": "consensus",
                        "failure_substage": "band_quality",
                        "failure_stage_reason": "fit failed while deriving consensus",
                        "is_consensus_failure": True,
                        "is_numerical_failure": False,
                        "is_input_validation_failure": False,
                        "exception_type": "ConsensusFitError",
                        "exception_message": "no common consensus period",
                        "fit_kwargs": {"model": "2D", "fit_strategy": "consensus"},
                    }
                ]
            },
            "quality_report": {"score_kind": "training_residual_quality"},
        }

        rows = _piwd_batch_extract_model_kernel_config_rows(
            source_row={"source_index": 0, "source_id": "src", "status": "passed"},
            workflow=workflow,
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["technical_outcome"], "failed")
        self.assertEqual(row["comparison_eligibility"], "ineligible")
        self.assertEqual(row["failure_code"], "no_accepted_bands")
        self.assertEqual(row["failure_stage"], "consensus")
        self.assertEqual(row["failure_substage"], "band_quality")
        self.assertIs(row["is_consensus_failure"], True)
        self.assertEqual(row["exception_type"], "ConsensusFitError")


if __name__ == "__main__":
    unittest.main()
