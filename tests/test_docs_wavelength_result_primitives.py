"""Documentation-contract tests for typed wavelength result primitives."""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestWavelengthResultPrimitiveDocs(unittest.TestCase):
    def test_api_reference_lists_result_module(self):
        text = (ROOT / "docs/source/api.rst").read_text(encoding="utf-8")
        self.assertIn("pgmuvi.wavelength_results", text)
        page = ROOT / "docs/source/pgmuvi.wavelength_results.rst"
        self.assertTrue(page.exists())
        self.assertIn(
            "automodule:: pgmuvi.wavelength_results",
            page.read_text(encoding="utf-8"),
        )

    def test_howto_documents_compatibility_and_evidence_roles(self):
        text = (
            ROOT / "docs/source/howto/wavelength_advisory.rst"
        ).read_text(encoding="utf-8")
        for phrase in (
            "Typed result adapters",
            "existing dictionary-returning workflow remains unchanged",
            "WavelengthAdvisoryResult.from_mapping",
            "observed fact",
            "derived statistic",
            "heuristic interpretation",
            "formal comparison result",
            "workflow warning",
            "future-work limitation",
        ):
            self.assertIn(phrase, text)

    def test_interpretation_guide_warns_against_relabelling_heuristics(self):
        text = (
            ROOT / "docs/source/howto/interpreting_results.rst"
        ).read_text(encoding="utf-8")
        self.assertIn("WavelengthEvidenceKind", text)
        self.assertIn(
            "does not turn a\nheuristic score into formal model evidence",
            text,
        )


if __name__ == "__main__":
    unittest.main()
