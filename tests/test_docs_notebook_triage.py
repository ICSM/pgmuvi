from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "source" / "index.rst"
STATUS = ROOT / "docs" / "source" / "notebook_status.rst"


class TestNotebookDocumentationTriage(unittest.TestCase):
    def test_public_tutorial_toctree_excludes_quarantined_notebooks(self):
        text = INDEX.read_text(encoding="utf-8")
        self.assertNotIn("notebooks/pgmuvi_tutorial_2d", text)
        self.assertNotIn("notebooks/tutorial_preprocessing", text)
        self.assertNotIn("notebooks/tutorial_synthetic", text)
        self.assertNotIn("notebooks/tutorial_model_selection", text)
        self.assertNotIn("notebooks/pgmuvi_tutorial_mcmc", text)
        self.assertNotIn("notebooks/pgmuvi_mock_data_from_gp", text)

    def test_public_tutorial_toctree_links_status_and_quasiperiodic_notebook(self):
        text = INDEX.read_text(encoding="utf-8")
        self.assertIn("notebook_status", text)
        self.assertIn("notebooks/PGMUVI_QuasiPeriodic_and_Mean_Functions", text)

    def test_status_page_records_quarantined_notebooks_and_replacements(self):
        text = STATUS.read_text(encoding="utf-8")
        for notebook in [
            "pgmuvi_tutorial_2d.ipynb",
            "tutorial_preprocessing.ipynb",
            "tutorial_synthetic.ipynb",
            "tutorial_model_selection.ipynb",
            "pgmuvi_tutorial_mcmc.ipynb",
            "pgmuvi_mock_data_from_gp.ipynb",
        ]:
            self.assertIn(notebook, text)
        self.assertIn("howto/consensus_fitting", text)
        self.assertIn("examples/consensus_multiband_fit.py", text)
        self.assertIn("period-independent wavelength advisory", text)
        self.assertIn("NotImplementedError", text)

    def test_status_page_has_structured_maintenance_markers(self):
        text = STATUS.read_text(encoding="utf-8")
        self.assertIn("Documentation status", text)
        self.assertIn("TBD[notebook-2d-consensus]", text)
        self.assertIn("TBD[notebook-advisory-workflow]", text)
        self.assertIn("TBD[mcmc-reenable]", text)

    def test_status_page_defines_notebook_admission_rules(self):
        text = STATUS.read_text(encoding="utf-8")
        self.assertIn("Before adding a notebook to the public Tutorials toctree", text)
        self.assertIn("model/kernel config", text)
        self.assertIn("candidate", text)
        self.assertIn("TODO-placeholder", text)


if __name__ == "__main__":
    unittest.main()
