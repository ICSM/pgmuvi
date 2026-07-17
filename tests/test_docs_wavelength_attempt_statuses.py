"""Documentation contracts for canonical wavelength-attempt statuses."""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestWavelengthAttemptStatusDocumentation(unittest.TestCase):
    def test_api_reference_exposes_status_module(self):
        api = (ROOT / "docs/source/api.rst").read_text(encoding="utf-8")
        page = ROOT / "docs/source/pgmuvi.wavelength_status.rst"
        self.assertTrue(page.exists())
        self.assertIn("pgmuvi.wavelength_status", api)
        self.assertIn("automodule:: pgmuvi.wavelength_status", page.read_text())

    def test_single_source_guide_documents_orthogonal_statuses(self):
        text = (ROOT / "docs/source/howto/wavelength_advisory.rst").read_text(
            encoding="utf-8"
        )
        for field in (
            "attempt_disposition",
            "execution_stage",
            "technical_outcome",
            "diagnostic_validity",
            "scientific_usability",
            "comparison_eligibility",
            'technical_outcome="initialized_only"',
        ):
            self.assertIn(field, text)

    def test_interpretation_guide_documents_failure_and_recovery_semantics(self):
        text = (ROOT / "docs/source/howto/interpreting_results.rst").read_text(
            encoding="utf-8"
        )
        for term in (
            "completed_with_warnings",
            "completed_with_recovery",
            "failure_code",
            "failure_substage",
            "comparison-ineligible",
        ):
            self.assertIn(term, text)

    def test_batch_guide_documents_canonical_long_form_columns(self):
        text = (ROOT / "docs/source/howto/wavelength_advisory_batch.rst").read_text(
            encoding="utf-8"
        )
        for field in (
            "attempt_disposition",
            "technical_outcome",
            "diagnostic_validity",
            "comparison_eligibility",
            "warning_severity",
            "failure_code",
            "failure_substage",
        ):
            self.assertIn(field, text)


if __name__ == "__main__":
    unittest.main()
