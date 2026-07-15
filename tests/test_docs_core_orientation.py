"""Regression checks for the core orientation documentation."""
from pathlib import Path
import py_compile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "source"
EXAMPLES = ROOT / "examples"


class TestCoreOrientationDocs(unittest.TestCase):
    def test_overview_page_exists_and_orients_users(self):
        page = (SOURCE / "overview.rst").read_text(encoding="utf-8")

        required = [
            "PGMUVI package orientation",
            "What PGMUVI is for",
            "The three questions to ask first",
            "Important model families",
            "First-choice workflow",
            "What PGMUVI does not decide automatically",
            "TBD",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, page)

    def test_overview_names_current_model_families(self):
        page = (SOURCE / "overview.rst").read_text(encoding="utf-8")

        for token in [
            "1D",
            "2D",
            "2DWavelengthDependent",
            "2DDustMean",
            "2DPowerLawMean",
            "2DSeparable",
            "fit_strategy=\"consensus\"",
            "learn_additional_noise=True",
        ]:
            with self.subTest(token=token):
                self.assertIn(token, page)

    def test_first_workflow_guide_is_self_contained(self):
        page = (SOURCE / "howto" / "first_workflow.rst").read_text(encoding="utf-8")

        required = [
            "First PGMUVI workflow",
            "Prepare the CSV",
            "Load and inspect the light curve",
            "Run descriptive wavelength diagnostics",
            "Fit a conservative baseline",
            "Compare wavelength-dependent families only when justified",
            "examples/first_pgmuvi_workflow.py",
            "TBD",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, page)

    def test_orientation_pages_are_linked_from_indices(self):
        root_index = (SOURCE / "index.rst").read_text(encoding="utf-8")
        howto_index = (SOURCE / "howto" / "index.rst").read_text(encoding="utf-8")

        self.assertIn("overview", root_index)
        self.assertIn("first_workflow", howto_index)

    def test_first_workflow_example_is_present_and_compiles(self):
        script = EXAMPLES / "first_pgmuvi_workflow.py"
        self.assertTrue(script.exists())
        text = script.read_text(encoding="utf-8")

        self.assertIn("toy_multiband_lpv.csv", text)
        self.assertIn("run_wavelength_advisory_batch.py", text)
        self.assertIn("fit_strategy='consensus'", text)
        py_compile.compile(str(script), doraise=True)


if __name__ == "__main__":
    unittest.main()
