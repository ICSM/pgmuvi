"""Documentation tests for legacy wavelength-candidate relabelling."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TestLegacyWavelengthCandidateDocumentation(unittest.TestCase):
    def read(self, relative_path):
        return (ROOT / relative_path).read_text(encoding="utf-8")

    def test_model_selection_page_is_workflow_map(self):
        text = self.read("docs/source/howto/model_selection.rst")
        self.assertIn("Choosing among fitting workflows", text)
        self.assertIn("consensus_fitting", text)
        self.assertIn("wavelength_advisory", text)
        self.assertIn("wavelength_advisory_batch", text)
        self.assertIn("legacy_wavelength_candidates", text)
        self.assertIn("model/kernel config", text)
        self.assertIn("selected_model = None", text)

    def test_legacy_page_is_clearly_marked(self):
        text = self.read("docs/source/howto/legacy_wavelength_candidates.rst")
        self.assertIn("Legacy wavelength-candidate diagnostics", text)
        self.assertIn("older candidate-based", text)
        self.assertIn("diagnose_wavelength_dependence", text)
        self.assertIn("compare_wavelength_models", text)
        self.assertIn("wavelength_advisory", text)
        self.assertIn("selected_model=None", text)

    def test_legacy_page_does_not_claim_bic_or_loo_are_implemented(self):
        text = self.read("docs/source/howto/legacy_wavelength_candidates.rst")
        self.assertIn("does not calculate", text)
        self.assertIn("Bayesian Information Criterion (BIC)", text)
        self.assertIn("Leave-One-Out cross-validation", text)
        self.assertIn("implementation or external analysis", text)
        self.assertNotIn(
            "can be used for formal model comparison",
            text,
        )

    def test_howto_index_links_current_and_legacy_pages(self):
        text = self.read("docs/source/howto/index.rst")
        for page in [
            "consensus_fitting",
            "model_selection",
            "wavelength_advisory",
            "wavelength_advisory_batch",
            "legacy_wavelength_candidates",
        ]:
            self.assertIn(page, text)

    def test_readme_points_to_legacy_page_without_hiding_current_workflow(self):
        text = self.read("README.md")
        self.assertIn("legacy_wavelength_candidates.rst", text)
        self.assertIn("model/kernel config", text)
        self.assertIn("run_period_independent_wavelength_advisory_workflow", text)

    def test_legacy_examples_are_marked(self):
        for path in [
            "examples/wavelength_model_selection_diagnostics.py",
            "examples/validate_wavelength_model_selection_workflow.py",
            "examples/model_selection.py",
        ]:
            text = self.read(path)
            self.assertIn("Status: LEGACY", text, path)
            self.assertIn("wavelength_advisory", text, path)

    def test_current_advisory_docs_do_not_use_legacy_candidate_framing(self):
        for path in [
            "docs/source/howto/wavelength_advisory.rst",
            "docs/source/howto/wavelength_advisory_batch.rst",
        ]:
            if not (ROOT / path).exists():
                continue
            text = self.read(path)
            self.assertIn("model/kernel config", text, path)
            self.assertIn("selected_model", text, path)
            self.assertNotIn("recommended_candidate_models", text, path)


if __name__ == "__main__":
    unittest.main()
