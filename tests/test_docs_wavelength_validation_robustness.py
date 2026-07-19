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
        self.assertIn("maintained Jupyter notebook", normalized)
        self.assertIn("do not rank models", normalized)

    def test_package_exports_robustness_module_name(self):
        package_init = (ROOT / "pgmuvi/__init__.py").read_text(encoding="utf-8")
        self.assertIn('"wavelength_validation_robustness"', package_init)

    def test_future_work_tracks_d2_calibration_and_notebook(self):
        future = (ROOT / "docs/source/future_work.rst").read_text(
            encoding="utf-8"
        )
        normalized = " ".join(future.split())

        self.assertIn("core D2 scenario/runner framework", normalized)
        self.assertIn("resumable multi-seed execution", normalized)
        self.assertIn("wavelength-constraint Jupyter notebook", normalized)

    def test_notebook_status_tracks_constraint_walkthrough(self):
        status = (ROOT / "docs/source/notebook_status.rst").read_text(
            encoding="utf-8"
        )
        normalized = " ".join(status.split())

        self.assertIn("TBD[wavelength-constraint-notebook]", status)
        self.assertIn("wavelength-kernel scales and bounds", normalized)
        self.assertIn("covariance constraints", normalized)
        self.assertIn("sparse/noisy failure boundaries", normalized)


if __name__ == "__main__":
    unittest.main()
