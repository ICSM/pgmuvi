"""Unit tests for canonical wavelength-advisory status types."""

import json
import unittest

from pgmuvi.wavelength_status import (
    AttemptDisposition,
    ComparisonEligibility,
    DiagnosticValidity,
    ExecutionStage,
    ScientificUsability,
    TechnicalOutcome,
    WarningSeverity,
    WavelengthAttemptStatus,
    WavelengthFailureRecord,
    derive_wavelength_attempt_status,
)


class TestWavelengthStatusEnums(unittest.TestCase):
    def test_string_enums_serialize_as_stable_values(self):
        self.assertEqual(str(TechnicalOutcome.COMPLETED), "completed")
        payload = WavelengthAttemptStatus(
            disposition=AttemptDisposition.ATTEMPTED,
            execution_stage=ExecutionStage.COMPLETED,
            technical_outcome=TechnicalOutcome.COMPLETED,
            diagnostic_validity=DiagnosticValidity.VALID,
            scientific_usability=ScientificUsability.USABLE,
            comparison_eligibility=ComparisonEligibility.ELIGIBLE,
            warning_severity=None,
        ).to_dict()
        self.assertEqual(payload["attempt_disposition"], "attempted")
        self.assertEqual(payload["comparison_eligibility"], "eligible")
        json.dumps(payload)

    def test_initialized_only_is_limited_and_comparison_ineligible(self):
        status = derive_wavelength_attempt_status(
            legacy_status="passed",
            training_iter=0,
            diagnostics_available=True,
        )
        self.assertEqual(status.technical_outcome, TechnicalOutcome.INITIALIZED_ONLY)
        self.assertEqual(status.scientific_usability, ScientificUsability.LIMITED)
        self.assertEqual(
            status.comparison_eligibility,
            ComparisonEligibility.INELIGIBLE,
        )

    def test_skipped_attempt_is_comparison_ineligible(self):
        status = derive_wavelength_attempt_status(legacy_status="skipped")
        self.assertEqual(status.disposition, AttemptDisposition.SKIPPED)
        self.assertEqual(status.execution_stage, ExecutionStage.PRECONDITION)
        self.assertEqual(status.technical_outcome, TechnicalOutcome.SKIPPED)
        self.assertEqual(
            status.comparison_eligibility,
            ComparisonEligibility.INELIGIBLE,
        )

    def test_completed_without_diagnostics_is_limited_and_ineligible(self):
        status = derive_wavelength_attempt_status(
            legacy_status="passed",
            training_iter=20,
            diagnostics_available=False,
        )
        self.assertEqual(status.technical_outcome, TechnicalOutcome.COMPLETED)
        self.assertEqual(
            status.diagnostic_validity, DiagnosticValidity.UNAVAILABLE
        )
        self.assertEqual(status.scientific_usability, ScientificUsability.LIMITED)
        self.assertEqual(
            status.comparison_eligibility,
            ComparisonEligibility.INELIGIBLE,
        )

    def test_recovered_fit_is_explicit(self):
        status = derive_wavelength_attempt_status(
            legacy_status="passed",
            training_iter=20,
            recovered_from_failure=True,
            diagnostics_available=True,
        )
        self.assertEqual(
            status.technical_outcome,
            TechnicalOutcome.COMPLETED_WITH_RECOVERY,
        )
        self.assertEqual(status.scientific_usability, ScientificUsability.LIMITED)
        self.assertEqual(
            status.comparison_eligibility,
            ComparisonEligibility.ELIGIBLE,
        )

    def test_completed_with_warnings_has_structured_severity(self):
        status = derive_wavelength_attempt_status(
            legacy_status="passed",
            training_iter=20,
            diagnostics_available=True,
            warning_count=2,
        )
        self.assertEqual(
            status.technical_outcome,
            TechnicalOutcome.COMPLETED_WITH_WARNINGS,
        )
        self.assertEqual(status.warning_severity, WarningSeverity.WARNING)

    def test_failed_attempt_uses_failure_stage(self):
        status = derive_wavelength_attempt_status(
            legacy_status="failed",
            diagnostics_partial=True,
            failure_stage="consensus",
        )
        self.assertEqual(status.disposition, AttemptDisposition.ATTEMPTED)
        self.assertEqual(status.execution_stage, ExecutionStage.CONSENSUS)
        self.assertEqual(status.technical_outcome, TechnicalOutcome.FAILED)
        self.assertEqual(status.diagnostic_validity, DiagnosticValidity.PARTIAL)
        self.assertEqual(status.scientific_usability, ScientificUsability.UNUSABLE)
        self.assertEqual(
            status.comparison_eligibility,
            ComparisonEligibility.INELIGIBLE,
        )

    def test_failure_record_is_json_safe(self):
        record = WavelengthFailureRecord(
            failure_code="no_accepted_bands",
            stage=ExecutionStage.CONSENSUS,
            substage="band_quality",
            exception_type="ConsensusFitError",
            message="no consensus",
            diagnostics={"reason": "no_accepted_bands"},
            traceback_reference="inline:traceback",
        ).to_dict()
        self.assertEqual(record["failure_stage"], "consensus")
        json.dumps(record)


if __name__ == "__main__":
    unittest.main()
