"""Regression tests for maintained Lomb--Scargle plotting in notebooks."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


class TestLombScargleNotebookPlotting(unittest.TestCase):
    @staticmethod
    def _notebook(name):
        path = (
            Path(__file__).resolve().parents[1]
            / "docs/source/notebooks"
            / name
        )
        return json.loads(path.read_text(encoding="utf-8"))

    def test_lomb_scargle_notebook_uses_package_plot_method(self):
        notebook = self._notebook("PGMUVI_Lomb_Scargle.ipynb")
        all_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
        )
        code_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        self.assertNotIn(
            "does not currently provide a built-in plotting utility",
            all_text,
        )
        self.assertGreaterEqual(
            code_text.count(".plot_lomb_scargle_periodogram("),
            7,
        )
        self.assertNotIn("plt.plot(period", code_text)
        self.assertNotIn('plt.xscale("log")', code_text)
        self.assertIn('"Component 1: 150 d": 150.0', code_text)
        self.assertIn('"Component 2: 66 d": 66.0', code_text)

    def test_single_source_notebook_documents_axis_contract(self):
        notebook = self._notebook(
            "tutorial_single_source_analysis.ipynb"
        )
        text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
        )
        self.assertIn("plot_lomb_scargle_periodogram()", text)
        self.assertIn("period on a logarithmic x-axis", text)
        self.assertIn("power on a linear y-axis", text)
        self.assertNotIn(
            "Representative channel-specific Lomb–Scargle periodograms",
            text,
        )


if __name__ == "__main__":
    unittest.main()
