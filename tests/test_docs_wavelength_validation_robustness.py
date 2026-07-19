import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestSyntheticWavelengthRobustnessDocumentation(unittest.TestCase):
    def test_robustness_module_is_in_public_api(self):
        api = (ROOT / "docs/source/api.rst").read_text(encoding="utf-8")
        page = ROOT / "docs/source/pgmuvi.wavelength_validation_robustness.rst"

        self.assertIn("pgmuvi.wavelength_validation_robustness", api)
        self.assertTrue(page.exists())
        page_text = page.read_text(encoding="utf-8")
        self.assertIn(
            "automodule:: pgmuvi.wavelength_validation_robustness",
            page_text,
        )
        normalized = " ".join(page_text.split())
        self.assertIn("does not reuse the D1 population gates", normalized)
        self.assertIn("expected failure remains a failed", normalized)
        self.assertIn("wavelength_validation_robustness_calibration", normalized)
        self.assertIn("truth-matched reference populations", normalized)
        self.assertIn("public", normalized)
        self.assertIn("tutorial_wavelength_constraints", normalized)
        self.assertIn("do not rank models", normalized)

    def test_package_exports_robustness_module_name(self):
        package_init = (ROOT / "pgmuvi/__init__.py").read_text(encoding="utf-8")
        self.assertIn('"wavelength_validation_robustness"', package_init)

    def test_future_work_tracks_d3_after_d2_and_notebook_completion(self):
        future = (ROOT / "docs/source/future_work.rst").read_text(
            encoding="utf-8"
        )
        normalized = " ".join(future.split())

        self.assertIn("canonical 20-seed D2 robustness calibration", normalized)
        self.assertIn("maintained wavelength-constraint notebook are complete", normalized)
        self.assertIn("D3 application to representative real LPV sources", normalized)
        self.assertNotIn("Execute and document the full D2 population", future)

    def test_notebook_status_tracks_constraint_walkthrough(self):
        status = (ROOT / "docs/source/notebook_status.rst").read_text(
            encoding="utf-8"
        )
        normalized = " ".join(status.split())

        self.assertNotIn("TBD[wavelength-constraint-notebook]", status)
        self.assertIn("tutorial_wavelength_constraints.ipynb", status)
        self.assertIn("coordinate round trips", normalized)
        self.assertIn("covariance and mean constraints", normalized)
        self.assertIn("empirical failure boundaries", normalized)


if __name__ == "__main__":
    unittest.main()
