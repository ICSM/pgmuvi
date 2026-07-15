from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "source" / "index.rst"
STATUS = ROOT / "docs" / "source" / "notebook_status.rst"
CONF = ROOT / "docs" / "source" / "conf.py"


class TestNotebookDocumentationTriage(unittest.TestCase):
    def test_public_tutorial_toctree_excludes_remaining_quarantined_notebooks(self):
        text = INDEX.read_text(encoding="utf-8")
        self.assertNotIn("notebooks/pgmuvi_tutorial_mcmc", text)
        self.assertNotIn("notebooks/pgmuvi_mock_data_from_gp", text)

    def test_public_tutorial_toctree_links_status_and_maintained_notebooks(self):
        text = INDEX.read_text(encoding="utf-8")
        self.assertIn("notebook_status", text)
        self.assertIn("notebooks/PGMUVI_QuasiPeriodic_and_Mean_Functions", text)
        self.assertIn("notebooks/pgmuvi_tutorial_2d", text)
        self.assertIn("notebooks/tutorial_preprocessing", text)
        self.assertIn("notebooks/tutorial_synthetic", text)
        self.assertIn("notebooks/tutorial_wavelength_advisory", text)

    def test_status_page_records_remaining_quarantined_notebooks_and_replacements(self):
        text = STATUS.read_text(encoding="utf-8")
        for notebook in [
            "pgmuvi_tutorial_mcmc.ipynb",
            "pgmuvi_mock_data_from_gp.ipynb",
        ]:
            self.assertIn(notebook, text)
        self.assertNotIn("tutorial_model_selection.ipynb", text)
        self.assertIn("NotImplementedError", text)

    def test_status_page_records_refreshed_notebooks_as_public(self):
        text = STATUS.read_text(encoding="utf-8")
        self.assertIn("current through PR106", text)
        self.assertIn("Maintained 2-D baseline and consensus-fitting tutorial", text)
        self.assertIn("Refreshed in PR103", text)
        self.assertIn("Maintained preprocessing and data-quality tutorial", text)
        self.assertIn("Refreshed in PR104", text)
        self.assertIn("Maintained analytic synthetic-data tutorial", text)
        self.assertIn("Refreshed in PR105", text)
        self.assertIn("Maintained period-independent wavelength advisory tutorial", text)
        self.assertIn("Refreshed and renamed in PR106", text)
        self.assertNotIn("TBD[notebook-2d-consensus]", text)
        self.assertNotIn("TBD[notebook-preprocessing-refresh]", text)
        self.assertNotIn("TBD[notebook-synthetic-refresh]", text)
        self.assertNotIn("TBD[notebook-advisory-workflow]", text)
        self.assertNotIn("Quarantined stub", text)
        self.assertNotIn("Quarantined TODO skeleton", text)

    def test_status_page_has_remaining_structured_maintenance_markers(self):
        text = STATUS.read_text(encoding="utf-8")
        self.assertIn("Documentation status", text)
        self.assertIn("TBD[mcmc-reenable]", text)
        self.assertIn("TBD[notebook-mock-data-refresh]", text)
        self.assertNotIn("TBD[notebook-advisory-workflow]", text)

    def test_status_page_defines_notebook_admission_rules(self):
        text = STATUS.read_text(encoding="utf-8")
        self.assertIn("Before adding a notebook to the public Tutorials toctree", text)
        self.assertIn("model/kernel config", text)
        self.assertIn("candidate", text)
        self.assertIn("TODO-placeholder", text)

    def test_remaining_quarantined_notebooks_are_excluded_from_sphinx(self):
        text = CONF.read_text(encoding="utf-8")
        for notebook in [
            "notebooks/pgmuvi_tutorial_2d.ipynb",
            "notebooks/tutorial_preprocessing.ipynb",
            "notebooks/tutorial_synthetic.ipynb",
            "notebooks/tutorial_wavelength_advisory.ipynb",
            "notebooks/tutorial_model_selection.ipynb",
        ]:
            self.assertNotIn(notebook, text)
        for notebook in [
            "notebooks/pgmuvi_tutorial_mcmc.ipynb",
            "notebooks/pgmuvi_mock_data_from_gp.ipynb",
        ]:
            self.assertIn(notebook, text)

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


if __name__ == "__main__":
    unittest.main()
