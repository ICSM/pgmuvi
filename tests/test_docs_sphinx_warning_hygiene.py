import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestSphinxDocstringHygiene(unittest.TestCase):
    def test_lightcurve_api_page_uses_manual_synopsis(self):
        text = (ROOT / "docs" / "source" / "pgmuvi.lightcurve.rst").read_text(
            encoding="utf-8"
        )
        self.assertIn(".. automodule:: pgmuvi.lightcurve", text)
        self.assertNotIn(":members:", text)
        self.assertNotIn(":undoc-members:", text)
        self.assertNotIn(":show-inheritance:", text)
        self.assertNotIn(".. autoclass:: pgmuvi.lightcurve.Lightcurve", text)
        self.assertNotIn(".. autoclass:: pgmuvi.lightcurve.ConsensusFitError", text)
        self.assertIn(".. py:class:: Lightcurve", text)
        self.assertIn(".. py:exception:: ConsensusFitError", text)
        self.assertIn("manual public synopsis", text)

    def test_variability_docstrings_escape_restructuredtext_substitutions(self):
        text = (ROOT / "pgmuvi" / "preprocess" / "variability.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("|ybar|", text)
        self.assertNotIn("|delta_i|", text)

    def test_train_docstring_uses_single_line_type_fields(self):
        text = (ROOT / "pgmuvi" / "trainers.py").read_text(encoding="utf-8")
        start = text.index("def train(")
        end = text.index("if lightcurve is not None:", start)
        doc = text[start:end]
        self.assertIn('lossfn : {"mll", "elbo"} or MarginalLogLikelihood', doc)
        self.assertIn('optim : {"SGD", "Adam", "AdamW", "NUTS"} or optimizer', doc)
        self.assertNotIn("lossfn : string or instance of\n", doc)
        self.assertNotIn(
            "optim : string or instance of torch.optim.optimizer.Optimizer,\n",
            doc,
        )

    def test_preprocessing_howto_does_not_link_quarantined_notebook(self):
        text = (ROOT / "docs" / "source" / "howto" / "preprocessing.rst").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("../notebooks/tutorial_preprocessing", text)
        self.assertIn("TBD[notebook-preprocessing]", text)

    def test_lightcurve_legacy_docstrings_are_not_expanded_in_api_page(self):
        page = (ROOT / "docs" / "source" / "pgmuvi.lightcurve.rst").read_text(
            encoding="utf-8"
        )
        warning_prone_objects = [
            "InputHelpers",
            "FitFailureSummary.to_text",
            "FitFailureSummary.write_json",
            "Lightcurve.fit",
        ]
        for name in warning_prone_objects:
            self.assertNotIn(f".. autoclass:: pgmuvi.lightcurve.{name}", page)
            self.assertNotIn(f".. automethod:: {name}", page)


if __name__ == "__main__":
    unittest.main()
