"""Regression tests for public notebook Sphinx warning hygiene."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "docs" / "source" / "notebooks"


def read_notebook(name: str) -> dict:
    return json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))


def notebook_text(name: str) -> str:
    nb = read_notebook(name)
    chunks = []
    for cell in nb.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            chunks.append("".join(source))
        else:
            chunks.append(str(source))
    return "\n".join(chunks)


class TestPublicNotebookSphinxWarningHygiene(unittest.TestCase):
    def test_lightcurve_notebook_does_not_use_warning_prone_deep_headings(self) -> None:
        text = notebook_text("PGMUVI_Lightcurve.ipynb")
        self.assertNotIn("#### Core inputs", text)
        self.assertNotIn("#### Shape expectations", text)
        self.assertNotIn("##### 1D lightcurve (single band)", text)
        self.assertNotIn("##### 2D lightcurve (multi-band)", text)
        self.assertNotIn("#### 1D light curves", text)
        self.assertNotIn("#### 2D light curves", text)

    def test_quasiperiodic_notebook_import_cell_is_docs_safe(self) -> None:
        nb = read_notebook("PGMUVI_QuasiPeriodic_and_Mean_Functions.ipynb")
        first_cell = nb["cells"][0]
        self.assertEqual(first_cell.get("cell_type"), "code")
        source = "".join(first_cell.get("source", []))
        self.assertIn("import pgmuvi", source)
        self.assertNotIn("!pip install", source)
        self.assertNotIn("%pip", source)
        self.assertNotIn("copilot/implement-acf-method-lightcurve", source)


if __name__ == "__main__":
    unittest.main()
