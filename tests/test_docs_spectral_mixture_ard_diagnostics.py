from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestSpectralMixtureArdDiagnosticDocumentation(unittest.TestCase):
    def test_sphinx_prefers_repository_source_tree(self):
        conf = (ROOT / "docs/source/conf.py").read_text(encoding="utf-8")
        self.assertIn("REPOSITORY_ROOT", conf)
        self.assertIn("sys.path.insert(0, str(REPOSITORY_ROOT))", conf)

    def test_api_reference_exposes_diagnostic_module(self):
        api = (ROOT / "docs/source/api.rst").read_text(encoding="utf-8")
        page = (
            ROOT
            / "docs/source/pgmuvi.spectral_mixture_ard_diagnostics.rst"
        )
        self.assertIn("pgmuvi.spectral_mixture_ard_diagnostics", api)
        self.assertTrue(page.exists())
        self.assertIn(
            "automodule:: pgmuvi.spectral_mixture_ard_diagnostics",
            page.read_text(encoding="utf-8"),
        )

    def test_module_page_documents_required_diagnostic_semantics(self):
        text = " ".join(
            (
                ROOT
                / "docs/source/pgmuvi.spectral_mixture_ard_diagnostics.rst"
            ).read_text(encoding="utf-8").split()
        )
        for token in (
            "mixture_means",
            "mixture_scales",
            "lower and upper bounds",
            "distance to each bound",
            "raw GPyTorch parameter values",
            "raw-input coordinates",
            "index 0: temporal frequency",
            "index 1: wavelength frequency",
            "num_mixtures_fixed_at_one",
            "does not select a model",
        ):
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_interpretation_and_batch_guides_use_new_boundary_fields(self):
        interpretation = " ".join(
            (ROOT / "docs/source/howto/interpreting_results.rst")
            .read_text(encoding="utf-8")
            .split()
        )
        batch = " ".join(
            (ROOT / "docs/source/howto/wavelength_advisory_batch.rst")
            .read_text(encoding="utf-8")
            .split()
        )
        for text in (interpretation, batch):
            self.assertIn("sm_ard_boundary_hits", text)
            self.assertIn("sm_ard_boundary_pressure_scope", text)
            self.assertIn("sm_num_mixtures_fixed_at_one", text)
            self.assertIn("compatibility", text)


if __name__ == "__main__":
    unittest.main()
