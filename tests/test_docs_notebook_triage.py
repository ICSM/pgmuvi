from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "source" / "index.rst"
STATUS = ROOT / "docs" / "source" / "notebook_status.rst"
CONF = ROOT / "docs" / "source" / "conf.py"
NOTEBOOKS = ROOT / "docs" / "source" / "notebooks"


class TestNotebookDocumentationTriage(unittest.TestCase):
    def test_public_tutorial_toctree_has_no_deleted_mcmc_notebook(self):
        text = INDEX.read_text(encoding="utf-8")
        self.assertNotIn("notebooks/pgmuvi_tutorial_mcmc", text)
        self.assertIn("Unavailable future workflows", text)

    def test_public_tutorial_toctree_links_status_and_maintained_notebooks(self):
        text = INDEX.read_text(encoding="utf-8")
        self.assertIn("notebook_status", text)
        self.assertIn("notebooks/PGMUVI_QuasiPeriodic_and_Mean_Functions", text)
        self.assertIn("notebooks/pgmuvi_tutorial_2d", text)
        self.assertIn("notebooks/tutorial_preprocessing", text)
        self.assertIn("notebooks/tutorial_synthetic", text)
        self.assertIn("notebooks/tutorial_wavelength_advisory", text)
        self.assertIn("notebooks/pgmuvi_mock_data_from_gp", text)

    def test_status_page_records_notebook_refresh_completion(self):
        text = STATUS.read_text(encoding="utf-8")
        self.assertIn("current through PR108", text)
        self.assertIn("No quarantined or pending-refresh", text)
        self.assertIn("notebook files remain in the repository", text)
        self.assertIn("pgmuvi_tutorial_mcmc.ipynb", text)
        self.assertIn("deleted in PR108", text)
        self.assertIn("NotImplementedError", text)
        self.assertIn("TBD[mcmc-implementation]", text)
        self.assertNotIn("TBD[mcmc-reenable]", text)

    def test_status_page_records_refreshed_notebooks_as_public(self):
        text = STATUS.read_text(encoding="utf-8")
        self.assertIn("Maintained 2-D baseline and consensus-fitting tutorial", text)
        self.assertIn("Refreshed in PR103", text)
        self.assertIn("Maintained preprocessing and data-quality tutorial", text)
        self.assertIn("Refreshed in PR104", text)
        self.assertIn("Maintained analytic synthetic-data tutorial", text)
        self.assertIn("Refreshed in PR105", text)
        self.assertIn("Maintained period-independent wavelength-advisory tutorial", text)
        self.assertIn("Refreshed and renamed in PR106", text)
        self.assertIn("Maintained GP-prior mock-data tutorial", text)
        self.assertIn("Refreshed in PR107", text)
        self.assertNotIn("Quarantined stub", text)
        self.assertNotIn("Quarantined TODO skeleton", text)

    def test_status_page_defines_notebook_admission_rules(self):
        text = STATUS.read_text(encoding="utf-8")
        self.assertIn("Before adding or retaining a notebook", text)
        self.assertIn("model/kernel config", text)
        self.assertIn("candidate", text)
        self.assertIn("TODO-placeholder", text)
        self.assertIn("does not call unavailable APIs", text)

    def test_no_notebook_is_hidden_from_sphinx(self):
        text = CONF.read_text(encoding="utf-8")
        self.assertNotIn("notebooks/", text)
        self.assertIn('"test*"', text)
        self.assertIn('"old*"', text)

    def test_deleted_mcmc_notebook_is_absent(self):
        self.assertFalse((NOTEBOOKS / "pgmuvi_tutorial_mcmc.ipynb").exists())

    def test_multiband_page_links_refreshed_2d_notebook(self):
        text = (
            ROOT / "docs" / "source" / "howto" / "multiband.rst"
        ).read_text(encoding="utf-8")
        self.assertIn(":doc:`../notebooks/pgmuvi_tutorial_2d`", text)
        self.assertIn(":doc:`consensus_fitting`", text)
        self.assertNotIn("TBD[notebook-2d-consensus]", text)

    def test_preprocessing_pages_link_refreshed_notebook(self):
        preprocessing = (
            ROOT / "docs" / "source" / "howto" / "preprocessing.rst"
        ).read_text(encoding="utf-8")
        loading = (
            ROOT / "docs" / "source" / "howto" / "loading_data.rst"
        ).read_text(encoding="utf-8")
        for text in [preprocessing, loading]:
            self.assertIn(":doc:`../notebooks/tutorial_preprocessing`", text)
        self.assertNotIn("TBD[notebook-preprocessing]", preprocessing)
        self.assertNotIn("TBD[notebook-lightcurve-validation]", loading)

    def test_synthetic_concepts_and_api_link_refreshed_notebook(self):
        concepts = (ROOT / "docs" / "source" / "concepts.rst").read_text(
            encoding="utf-8"
        )
        api = (ROOT / "docs" / "source" / "pgmuvi.synthetic.rst").read_text(
            encoding="utf-8"
        )
        for text in [concepts, api]:
            self.assertIn(":doc:`notebooks/tutorial_synthetic`", text)

    def test_advisory_pages_link_refreshed_notebook(self):
        model_selection = (
            ROOT / "docs" / "source" / "howto" / "model_selection.rst"
        ).read_text(encoding="utf-8")
        advisory = (
            ROOT / "docs" / "source" / "howto" / "wavelength_advisory.rst"
        ).read_text(encoding="utf-8")
        for text in [model_selection, advisory]:
            self.assertIn(
                ":doc:`../notebooks/tutorial_wavelength_advisory`",
                text,
            )

    def test_gp_prior_guide_and_api_link_refreshed_notebook(self):
        guide = (
            ROOT / "docs" / "source" / "howto" / "gp_prior_sampling.rst"
        ).read_text(encoding="utf-8")
        concepts = (ROOT / "docs" / "source" / "concepts.rst").read_text(
            encoding="utf-8"
        )
        api = (ROOT / "docs" / "source" / "pgmuvi.gps.rst").read_text(
            encoding="utf-8"
        )
        for text in [guide, concepts, api]:
            self.assertIn("pgmuvi_mock_data_from_gp", text)


if __name__ == "__main__":
    unittest.main()
