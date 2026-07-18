import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestSyntheticWavelengthRecoveryDocumentation(unittest.TestCase):
    def test_recovery_module_is_in_public_api(self):
        api = (ROOT / "docs/source/api.rst").read_text(encoding="utf-8")
        page = ROOT / "docs/source/pgmuvi.wavelength_validation_recovery.rst"

        self.assertIn("pgmuvi.wavelength_validation_recovery", api)
        self.assertTrue(page.exists())
        page_text = page.read_text(encoding="utf-8")
        self.assertIn(
            "automodule:: pgmuvi.wavelength_validation_recovery", page_text
        )
        normalized = " ".join(page_text.split())
        self.assertIn("Missing fitted quantities remain explicit", normalized)
        self.assertIn("do not rank candidates as scientific evidence", normalized)

    def test_package_exports_recovery_module_name(self):
        package_init = (ROOT / "pgmuvi/__init__.py").read_text(encoding="utf-8")
        self.assertIn('"wavelength_validation_recovery"', package_init)


if __name__ == "__main__":
    unittest.main()
