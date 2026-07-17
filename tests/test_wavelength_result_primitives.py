"""Tests for typed wavelength-advisory result primitives."""

import json
import unittest

from pgmuvi.wavelength_results import (
    WAVELENGTH_RESULT_SCHEMA_VERSION,
    WavelengthAdvisoryResult,
    WavelengthEvidenceKind,
    WavelengthEvidenceRecord,
    WavelengthModelAttemptResult,
    WavelengthProvenanceRecord,
    WavelengthWarningRecord,
    as_wavelength_advisory_result,
    as_wavelength_model_attempt_result,
)
from pgmuvi.wavelength_status import (
    ComparisonEligibility,
    TechnicalOutcome,
    WarningSeverity,
)


class TestEvidenceAndWarningPrimitives(unittest.TestCase):
    def test_evidence_kind_preserves_epistemic_role(self):
        record = WavelengthEvidenceRecord(
            name="median_flux_monotonicity",
            kind=WavelengthEvidenceKind.DERIVED_STATISTIC,
            value=0.8,
            available=True,
            units=None,
            summary="Robust rank trend across bands.",
            provenance={"method": "spearman"},
            limitations=("descriptive, not model evidence",),
        )
        payload = record.to_dict()
        self.assertEqual(payload["kind"], "derived_statistic")
        self.assertEqual(payload["value"], 0.8)
        self.assertEqual(
            payload["limitations"], ["descriptive, not model evidence"]
        )
        json.dumps(payload)

    def test_all_required_evidence_roles_are_explicit(self):
        self.assertEqual(
            {item.value for item in WavelengthEvidenceKind},
            {
                "observed_fact",
                "derived_statistic",
                "heuristic_interpretation",
                "formal_comparison_result",
                "workflow_warning",
                "future_work_limitation",
            },
        )

    def test_warning_adapts_current_captured_warning_shape(self):
        record = WavelengthWarningRecord.from_mapping(
            {
                "message": "synthetic warning",
                "category": "UserWarning",
                "filename": "runner.py",
                "lineno": 12,
            },
            default_severity="warning",
            default_stage="optimization",
        )
        self.assertEqual(record.severity, WarningSeverity.WARNING)
        self.assertEqual(record.stage.value, "optimization")
        self.assertEqual(record.lineno, 12)
        json.dumps(record.to_dict())

    def test_empty_evidence_name_and_warning_message_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "name must be non-empty"):
            WavelengthEvidenceRecord(
                name="",
                kind=WavelengthEvidenceKind.OBSERVED_FACT,
            )
        with self.assertRaisesRegex(ValueError, "message must be non-empty"):
            WavelengthWarningRecord(message="")


class TestAttemptResultAdapter(unittest.TestCase):
    def _payload(self):
        return {
            "model_kernel_config_id": "rank1_2DDustMean",
            "rank": 1,
            "model": "2DDustMean",
            "status": "passed",
            "attempt_disposition": "attempted",
            "execution_stage": "completed",
            "technical_outcome": "completed_with_warnings",
            "diagnostic_validity": "valid",
            "scientific_usability": "usable",
            "comparison_eligibility": "eligible",
            "warning_severity": "warning",
            "warning_count": 1,
            "warning_records": [
                {
                    "message": "synthetic warning",
                    "category": "UserWarning",
                    "filename": "runner.py",
                    "lineno": 12,
                }
            ],
            "fit_kwargs": {"training_iter": 20},
            "fit_quality": {"available": True, "normalized_rmse": 0.4},
            "fit_quality_score": 4.2,
            "consensus_success": True,
            "consensus_period": 600.0,
            "unknown_legacy_field": {"kept": True},
        }

    def test_canonical_status_and_unknown_legacy_fields_survive(self):
        source = self._payload()
        result = WavelengthModelAttemptResult.from_mapping(source)
        source["fit_kwargs"]["training_iter"] = 999

        self.assertEqual(
            result.status.technical_outcome,
            TechnicalOutcome.COMPLETED_WITH_WARNINGS,
        )
        self.assertEqual(
            result.status.comparison_eligibility,
            ComparisonEligibility.ELIGIBLE,
        )
        self.assertEqual(result.fit_kwargs["training_iter"], 20)
        self.assertIsNotNone(result.hypothesis)
        self.assertEqual(result.hypothesis.model, "2DDustMean")
        self.assertEqual(result.hypothesis.role.value, "mean_and_covariance")
        self.assertEqual(
            result.hypothesis.covariance_structure.value,
            "separable_smooth_wavelength",
        )
        self.assertEqual(
            result.to_legacy_dict()["unknown_legacy_field"], {"kept": True}
        )
        typed = result.to_dict(include_legacy_payload=True)
        self.assertEqual(
            typed["status"]["technical_outcome"],
            "completed_with_warnings",
        )
        self.assertEqual(typed["warnings"][0]["message"], "synthetic warning")
        self.assertEqual(typed["hypothesis"]["mean_structure"], "dust_attenuation")
        self.assertEqual(typed["diagnostics"]["consensus"]["consensus_period"], 600.0)
        json.dumps(typed)

    def test_legacy_status_is_derived_when_canonical_fields_are_absent(self):
        result = WavelengthModelAttemptResult.from_mapping(
            {
                "model": "2DPowerLawMean",
                "status": "passed",
                "fit_kwargs": {"training_iter": 0},
                "fit_quality_available": True,
            }
        )
        self.assertEqual(
            result.status.technical_outcome,
            TechnicalOutcome.INITIALIZED_ONLY,
        )
        self.assertEqual(
            result.status.comparison_eligibility,
            ComparisonEligibility.INELIGIBLE,
        )

    def test_structured_failure_is_adapted(self):
        result = WavelengthModelAttemptResult.from_mapping(
            {
                "model": "2DWavelengthDependent",
                "status": "failed",
                "attempt_disposition": "attempted",
                "execution_stage": "consensus",
                "technical_outcome": "failed",
                "diagnostic_validity": "partial",
                "scientific_usability": "unusable",
                "comparison_eligibility": "ineligible",
                "warning_severity": "error",
                "failure_code": "no_accepted_bands",
                "failure_stage": "consensus",
                "failure_substage": "band_quality",
                "exception_type": "ConsensusFitError",
                "exception_message": "no consensus",
                "structured_failure_diagnostics": {
                    "rejected_bands": ["g", "r"]
                },
                "traceback_reference": "inline:traceback",
            }
        )
        self.assertIsNotNone(result.failure)
        self.assertEqual(result.failure.failure_code, "no_accepted_bands")
        self.assertEqual(result.failure.stage.value, "consensus")
        self.assertEqual(
            result.failure.diagnostics["rejected_bands"], ["g", "r"]
        )
        json.dumps(result.to_dict())

    def test_idempotent_convenience_adapter(self):
        result = WavelengthModelAttemptResult.from_mapping(self._payload())
        self.assertIs(as_wavelength_model_attempt_result(result), result)


class TestAdvisoryResultAdapter(unittest.TestCase):
    def _workflow(self):
        return {
            "kind": "period_independent_wavelength_advisory_workflow",
            "advisory_only": True,
            "runs_fits": True,
            "scores_fit_quality": True,
            "mutates_input_lightcurve": False,
            "automatic_model_selection_applied": False,
            "selected_model": None,
            "model_kernel_config_report": {"model_kernel_configs": []},
            "run_report": {
                "outcomes": [
                    {
                        "model_kernel_config_id": "rank1_2DDustMean",
                        "rank": 1,
                        "model": "2DDustMean",
                        "status": "passed",
                        "attempt_disposition": "attempted",
                        "execution_stage": "completed",
                        "technical_outcome": "completed",
                        "diagnostic_validity": "valid",
                        "scientific_usability": "usable",
                        "comparison_eligibility": "eligible",
                        "warning_severity": None,
                        "fit_kwargs": {"training_iter": 20},
                        "fit_quality": {"available": True},
                    }
                ]
            },
            "quality_report": {"fit_quality_ranking_status": "single_valid_candidate"},
            "fallback_report": {"available": False},
            "custom_future_field": "preserved",
        }

    def test_workflow_sections_attempts_and_provenance_are_typed(self):
        source = self._workflow()
        result = WavelengthAdvisoryResult.from_mapping(source)
        source["run_report"]["outcomes"][0]["model"] = "changed"

        self.assertEqual(result.schema_version, WAVELENGTH_RESULT_SCHEMA_VERSION)
        self.assertTrue(result.advisory_only)
        self.assertEqual(len(result.attempts), 1)
        self.assertEqual(result.attempts[0].model, "2DDustMean")
        self.assertIn("quality_report", result.sections)
        self.assertIsInstance(result.provenance, WavelengthProvenanceRecord)
        self.assertFalse(
            result.provenance.configuration["mutates_input_lightcurve"]
        )
        self.assertEqual(
            result.to_legacy_dict()["custom_future_field"], "preserved"
        )
        json.dumps(result.to_dict(include_legacy_payload=True))

    def test_adapter_does_not_claim_formal_evidence_implicitly(self):
        result = WavelengthAdvisoryResult.from_mapping(self._workflow())
        self.assertEqual(result.evidence, ())
        self.assertNotIn("selected_model", result.to_dict())

    def test_idempotent_convenience_adapter(self):
        result = WavelengthAdvisoryResult.from_mapping(self._workflow())
        self.assertIs(as_wavelength_advisory_result(result), result)


if __name__ == "__main__":
    unittest.main()
