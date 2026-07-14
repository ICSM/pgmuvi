"""Regression tests for Sphinx API-reference coverage."""

from pathlib import Path
import unittest


DOCS = Path("docs/source")


class TestApiReferenceDocumentation(unittest.TestCase):
    def test_api_toctree_includes_direct_public_modules(self):
        text = (DOCS / "api.rst").read_text(encoding="utf-8")
        expected = [
            "pgmuvi.lightcurve",
            "pgmuvi.gps",
            "pgmuvi.models",
            "pgmuvi.trainers",
            "pgmuvi.priors",
            "pgmuvi.constraints",
            "pgmuvi.constraint_utils",
            "pgmuvi.kernels",
            "pgmuvi.initialization",
            "pgmuvi.synthetic",
            "pgmuvi.parameter_specs",
            "pgmuvi.parameter_context",
            "pgmuvi.parameter_estimates",
            "pgmuvi.parameter_builders",
            "pgmuvi.parameter_application",
            "pgmuvi.parameter_workflow",
            "pgmuvi.dtypes",
            "pgmuvi.preprocess",
            "pgmuvi.multiband_ls_significance",
            "pgmuvi.wavelength_diagnostics",
            "pgmuvi.upload_validation",
        ]
        for module in expected:
            with self.subTest(module=module):
                self.assertIn(module, text)

    def test_api_reference_has_workflow_groupings(self):
        text = (DOCS / "api.rst").read_text(encoding="utf-8")
        for heading in [
            "Core light-curve and GP model modules",
            "Fitting, kernels, constraints, and training",
            "Parameter workflow modules",
            "Data, diagnostics, and validation",
        ]:
            with self.subTest(heading=heading):
                self.assertIn(heading, text)

    def test_missing_module_pages_now_exist(self):
        modules = [
            "pgmuvi.constraint_utils",
            "pgmuvi.dtypes",
            "pgmuvi.models",
            "pgmuvi.parameter_application",
            "pgmuvi.parameter_builders",
            "pgmuvi.parameter_context",
            "pgmuvi.parameter_estimates",
            "pgmuvi.parameter_specs",
            "pgmuvi.parameter_workflow",
            "pgmuvi.upload_validation",
        ]
        for module in modules:
            with self.subTest(module=module):
                path = DOCS / f"{module}.rst"
                self.assertTrue(path.exists())
                text = path.read_text(encoding="utf-8")
                self.assertIn(f".. automodule:: {module}", text)
                self.assertIn(":members:", text)
                self.assertIn(":undoc-members:", text)
                self.assertIn(":show-inheritance:", text)

    def test_parameter_workflow_pages_are_not_hidden(self):
        text = (DOCS / "api.rst").read_text(encoding="utf-8")
        parameter_modules = [
            "pgmuvi.parameter_specs",
            "pgmuvi.parameter_context",
            "pgmuvi.parameter_estimates",
            "pgmuvi.parameter_builders",
            "pgmuvi.parameter_application",
            "pgmuvi.parameter_workflow",
        ]
        for module in parameter_modules:
            with self.subTest(module=module):
                self.assertIn(module, text)
                self.assertTrue((DOCS / f"{module}.rst").exists())

    def test_models_page_is_marked_as_compatibility_shim(self):
        text = (DOCS / "pgmuvi.models.rst").read_text(encoding="utf-8")
        self.assertIn("compatibility", text.lower())
        self.assertIn("pgmuvi.gps", text)


if __name__ == "__main__":
    unittest.main()
