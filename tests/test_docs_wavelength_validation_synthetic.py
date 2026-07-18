import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestSyntheticWavelengthValidationDocumentation(unittest.TestCase):
    def test_synthetic_validation_module_is_in_public_api(self):
        api = (ROOT / "docs/source/api.rst").read_text(encoding="utf-8")
        page = ROOT / "docs/source/pgmuvi.wavelength_validation_synthetic.rst"

        self.assertIn("pgmuvi.wavelength_validation_synthetic", api)
        self.assertTrue(page.exists())
        page_text = page.read_text(encoding="utf-8")
        self.assertIn(
            "automodule:: pgmuvi.wavelength_validation_synthetic", page_text
        )
        self.assertIn("does not run optimization or rank models", page_text)
        self.assertIn("strictly positive linear flux", page_text)
        normalized = " ".join(page_text.split())
        self.assertIn("shared reproducible irregular time grid", normalized)
        self.assertIn("D2 robustness matrix", normalized)
        self.assertIn("72 observations per band", normalized)

    def test_package_exports_synthetic_validation_module_name(self):
        package_init = (ROOT / "pgmuvi/__init__.py").read_text(encoding="utf-8")
        self.assertIn('"wavelength_validation_synthetic"', package_init)


if __name__ == "__main__":
    unittest.main()
