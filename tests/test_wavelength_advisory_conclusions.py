"""Tests for structured wavelength-advisory conclusions."""

import json
import unittest

from pgmuvi.wavelength_conclusions import (
    WAVELENGTH_ADVISORY_CONCLUSION_SCHEMA_VERSION,
    WavelengthAdvisoryConclusion,
    WavelengthConclusionDisposition,
    WavelengthConclusionScope,
    WavelengthUnresolvedAmbiguity,
    synthesize_wavelength_advisory_conclusions,
)
from pgmuvi.wavelength_results import WavelengthAdvisoryResult
from pgmuvi.wavelength_diagnostics import (
    _piwd_batch_extract_model_kernel_config_rows,
    _piwd_batch_model_kernel_config_csv_fields,
    run_period_independent_wavelength_advisory_workflow,
)


def _outcome(model, *, fit_quality=True, failed=False):
    if failed:
        return {
            "model_kernel_config_id": f"config_{model}",
            "model": model,
            "status": "failed",
            "attempt_disposition": "attempted",
            "execution_stage": "optimization",
            "technical_outcome": "failed",
            "diagnostic_validity": "unavailable",
            "scientific_usability": "unusable",
            "comparison_eligibility": "ineligible",
            "warning_severity": "error",
            "failure_code": "numerical_stability_failed",
            "failure_stage": "optimization",
            "exception_type": "RuntimeError",
            "exception_message": "synthetic failure",
            "fit_kwargs": {"training_iter": 20},
        }
    return {
        "model_kernel_config_id": f"config_{model}",
        "model": model,
        "status": "passed",
        "attempt_disposition": "attempted",
        "execution_stage": "completed",
        "technical_outcome": "completed",
        "diagnostic_validity": "valid" if fit_quality else "unavailable",
        "scientific_usability": "usable" if fit_quality else "limited",
        "comparison_eligibility": "eligible" if fit_quality else "ineligible",
        "warning_severity": None,
        "fit_kwargs": {"training_iter": 20},
        "fit_quality": {"available": fit_quality},
        "fit_quality_available": fit_quality,
    }


def _workflow(outcomes, *, ranking_status="available", top="2DDustMean"):
    ranked = []
    for rank, outcome in enumerate(outcomes, start=1):
        if outcome.get("comparison_eligibility") != "eligible":
            continue
        ranked.append(
            {
                "model_kernel_config_id": outcome["model_kernel_config_id"],
                "model": outcome["model"],
                "quality_rank": rank,
                "fit_quality_available": True,
                "fit_quality_score": 10.0 - rank,
                "is_top_ranked": outcome["model"] == top,
            }
        )
    return {
        "kind": "period_independent_wavelength_advisory_workflow",
        "advisory_only": True,
        "automatic_model_selection_applied": False,
        "selected_model": None,
        "fit_quality_ranking_status": ranking_status,
        "top_ranked_model": top if ranking_status == "available" else None,
        "run_report": {
            "kind": "period_independent_wavelength_model_kernel_config_results",
            "outcomes": outcomes,
        },
        "quality_report": {
            "kind": "period_independent_wavelength_model_kernel_config_quality_scores",
            "ranking_status": ranking_status,
            "score_kind": "training_residual_fit_quality",
            "ranked_results": ranked,
        },
    }


class TestConclusionPrimitives(unittest.TestCase):
    def test_records_are_json_safe_and_require_explicit_scope(self):
        conclusion = WavelengthAdvisoryConclusion(
            conclusion_id="complete_configuration:2D",
            scope=WavelengthConclusionScope.COMPLETE_CONFIGURATION,
            subject="2D",
            disposition=WavelengthConclusionDisposition.REMAINS_PLAUSIBLE,
            summary="The baseline remains plausible.",
            models=("2D",),
            evidence_basis={"fit_quality_score": 3.0},
        )
        ambiguity = WavelengthUnresolvedAmbiguity(
            code="training_residual_ranking_is_heuristic",
            summary="The score is heuristic.",
            affected_scopes=(WavelengthConclusionScope.COMPLETE_CONFIGURATION,),
        )
        self.assertEqual(
            conclusion.to_dict()["disposition"], "remains_plausible"
        )
        self.assertEqual(
            ambiguity.to_dict()["affected_scopes"], ["complete_configuration"]
        )
        json.dumps(conclusion.to_dict())
        json.dumps(ambiguity.to_dict())

    def test_required_dispositions_are_stable(self):
        self.assertEqual(
            {item.value for item in WavelengthConclusionDisposition},
            {
                "remains_plausible",
                "weakened",
                "technically_unevaluable",
                "scientifically_ambiguous",
                "incomparable",
            },
        )


class TestConclusionSynthesis(unittest.TestCase):
    def test_synthesis_separates_configuration_mean_and_covariance_scopes(self):
        outcomes = [
            _outcome("2DDustMean"),
            _outcome("2DPowerLawMean"),
            _outcome("2DSeparable"),
            _outcome("2D"),
        ]
        synthesis = synthesize_wavelength_advisory_conclusions(
            _workflow(outcomes)
        )

        self.assertEqual(
            synthesis["schema_version"],
            WAVELENGTH_ADVISORY_CONCLUSION_SCHEMA_VERSION,
        )
        self.assertFalse(synthesis["automatic_model_selection_applied"])
        self.assertIsNone(synthesis["selected_model"])
        scopes = {item["scope"] for item in synthesis["conclusions"]}
        self.assertEqual(
            scopes,
            {
                "complete_configuration",
                "mean_structure",
                "covariance_structure",
            },
        )
        dust = next(
            item
            for item in synthesis["conclusions"]
            if item["conclusion_id"] == "mean_structure:dust_attenuation"
        )
        self.assertEqual(dust["disposition"], "remains_plausible")
        self.assertTrue(dust["evidence_basis"]["isolating_contrast_available"])
        codes = {
            item["code"] for item in synthesis["unresolved_ambiguities"]
        }
        self.assertIn("training_residual_ranking_is_heuristic", codes)
        json.dumps(synthesis)

    def test_failed_attempt_is_technically_unevaluable(self):
        synthesis = synthesize_wavelength_advisory_conclusions(
            _workflow(
                [_outcome("2DDustMean"), _outcome("2DWavelengthDependent", failed=True)],
                ranking_status="single_valid_candidate",
                top=None,
            )
        )
        failed = next(
            item
            for item in synthesis["conclusions"]
            if item["conclusion_id"]
            == "complete_configuration:config_2DWavelengthDependent"
        )
        self.assertEqual(failed["disposition"], "technically_unevaluable")
        codes = {
            item["code"] for item in synthesis["unresolved_ambiguities"]
        }
        self.assertIn("single_valid_candidate_not_comparative", codes)
        self.assertIn("technical_failures_limit_hypothesis_coverage", codes)

    def test_unisolated_mechanism_is_scientifically_ambiguous(self):
        synthesis = synthesize_wavelength_advisory_conclusions(
            _workflow(
                [_outcome("2DDustMean")],
                ranking_status="single_valid_candidate",
                top=None,
            )
        )
        mean = next(
            item
            for item in synthesis["conclusions"]
            if item["scope"] == "mean_structure"
        )
        covariance = next(
            item
            for item in synthesis["conclusions"]
            if item["scope"] == "covariance_structure"
        )
        self.assertEqual(mean["disposition"], "scientifically_ambiguous")
        self.assertEqual(
            covariance["disposition"], "scientifically_ambiguous"
        )



    def test_typed_result_preserves_conclusion_sections(self):
        outcomes = [_outcome("2DDustMean"), _outcome("2DPowerLawMean")]
        workflow = _workflow(outcomes)
        synthesis = synthesize_wavelength_advisory_conclusions(workflow)
        workflow["advisory_conclusions"] = synthesis["conclusions"]
        workflow["unresolved_ambiguities"] = synthesis[
            "unresolved_ambiguities"
        ]
        workflow["advisory_conclusion_summary"] = synthesis["summary"]
        typed = WavelengthAdvisoryResult.from_mapping(workflow)
        self.assertEqual(
            typed.sections["advisory_conclusions"], synthesis["conclusions"]
        )
        self.assertEqual(
            typed.sections["advisory_conclusion_summary"], synthesis["summary"]
        )

    def test_batch_rows_receive_complete_configuration_conclusion(self):
        outcomes = [_outcome("2DDustMean"), _outcome("2DPowerLawMean")]
        workflow = _workflow(outcomes)
        synthesis = synthesize_wavelength_advisory_conclusions(workflow)
        workflow["advisory_conclusion_schema_version"] = synthesis[
            "schema_version"
        ]
        workflow["advisory_conclusions"] = synthesis["conclusions"]
        rows = _piwd_batch_extract_model_kernel_config_rows(
            source_row={"source_index": 0, "source_id": "source", "status": "passed"},
            workflow=workflow,
        )
        dust = next(row for row in rows if row["model"] == "2DDustMean")
        self.assertEqual(
            dust["advisory_conclusion_disposition"], "remains_plausible"
        )
        self.assertEqual(
            dust["advisory_conclusion_schema_version"],
            WAVELENGTH_ADVISORY_CONCLUSION_SCHEMA_VERSION,
        )
        fields = _piwd_batch_model_kernel_config_csv_fields()
        self.assertIn("advisory_conclusion_disposition", fields)
        self.assertIn("advisory_conclusion_summary", fields)

    def test_workflow_adds_synthesis_without_selecting(self):
        outcomes = [_outcome("2DDustMean"), _outcome("2DPowerLawMean")]
        workflow = _workflow(outcomes)
        report = run_period_independent_wavelength_advisory_workflow(
            object(),
            model_kernel_config_report={
                "kind": "period_independent_wavelength_model_kernel_configs",
                "model_kernel_configs": [],
            },
            run_report=workflow["run_report"],
            quality_report=workflow["quality_report"],
            make_text_report=True,
        )
        self.assertEqual(
            report["advisory_conclusion_schema_version"],
            WAVELENGTH_ADVISORY_CONCLUSION_SCHEMA_VERSION,
        )
        self.assertTrue(report["advisory_conclusions"])
        self.assertTrue(report["unresolved_ambiguities"])
        self.assertFalse(report["automatic_model_selection_applied"])
        self.assertIsNone(report["selected_model"])
        self.assertIn("Advisory conclusions", report["text_report"])


if __name__ == "__main__":
    unittest.main()
