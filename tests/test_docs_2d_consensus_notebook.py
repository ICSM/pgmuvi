"""Regression tests for the refreshed public 2-D consensus notebook."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "docs/source/notebooks/pgmuvi_tutorial_2d.ipynb"
INDEX = ROOT / "docs/source/index.rst"
CONF = ROOT / "docs/source/conf.py"
CONSENSUS = ROOT / "docs/source/howto/consensus_fitting.rst"


def load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def notebook_text() -> str:
    chunks = []
    for cell in load_notebook().get("cells", []):
        source = cell.get("source", [])
        chunks.append("".join(source) if isinstance(source, list) else str(source))
    return "\n".join(chunks)


class TestRefreshed2DConsensusNotebook(unittest.TestCase):
    def test_notebook_is_public_and_not_excluded(self):
        self.assertIn(
            "notebooks/pgmuvi_tutorial_2d",
            INDEX.read_text(encoding="utf-8"),
        )
        self.assertNotIn(
            "notebooks/pgmuvi_tutorial_2d.ipynb",
            CONF.read_text(encoding="utf-8"),
        )

    def test_notebook_has_current_structure_and_metadata(self):
        nb = load_notebook()
        self.assertEqual(nb.get("nbformat"), 4)
        self.assertEqual(
            nb.get("metadata", {}).get("kernelspec", {}).get("name"),
            "python3",
        )
        self.assertGreaterEqual(len(nb.get("cells", [])), 12)
        self.assertTrue(
            any(cell.get("cell_type") == "markdown" for cell in nb["cells"])
        )
        self.assertTrue(
            any(cell.get("cell_type") == "code" for cell in nb["cells"])
        )
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])

    def test_notebook_contains_core_workflow_contracts(self):
        text = notebook_text()
        for token in [
            "RUN_FIT = False",
            "Lightcurve(",
            '"model": "2D"',
            '"fit_strategy": "consensus"',
            '"learn_additional_noise": True',
            "ConsensusFitError",
            "get_period_summary()",
            "consensus_diagnostics",
            "2DWavelengthDependent",
            "2DDustMean",
            "2DPowerLawMean",
            "2DSeparable",
            '"time_kernel_type": "quasi_periodic"',
            "automatic model selection",
        ]:
            self.assertIn(token, text)

    def test_notebook_removes_stale_stub_patterns(self):
        text = notebook_text()
        for token in [
            "%pip",
            "!pip install",
            "git+https://github.com/ICSM/pgmuvi.git",
            'xtransform="minmax"',
            "maybe drawing from a specific PSD",
            "#generate random x data here",
        ]:
            self.assertNotIn(token, text)
        self.assertNotIn("TODO", text)

    def test_all_code_cells_compile(self):
        for index, cell in enumerate(load_notebook().get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            compile(
                source,
                f"pgmuvi_tutorial_2d.ipynb:cell-{index}",
                "exec",
            )

    @unittest.skipUnless(
        importlib.util.find_spec("gpytorch") is not None,
        "gpytorch is required for the notebook preparation smoke test",
    )
    def test_preparation_path_executes_without_training_or_outputs(self):
        namespace = {"__name__": "__main__"}
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                for index, cell in enumerate(load_notebook().get("cells", [])):
                    if cell.get("cell_type") != "code":
                        continue
                    source = "".join(cell.get("source", []))
                    exec(
                        compile(
                            source,
                            f"pgmuvi_tutorial_2d.ipynb:cell-{index}",
                            "exec",
                        ),
                        namespace,
                    )
            finally:
                os.chdir(old_cwd)

            self.assertFalse(
                Path(tmpdir, "tutorial_2d_consensus_output").exists()
            )

        self.assertFalse(namespace["RUN_FIT"])
        self.assertEqual(namespace["lc"].ndim, 2)
        self.assertEqual(len(namespace["unique_wavelengths"]), 3)
        self.assertIsNone(namespace["fit_result"])
        self.assertIsNone(namespace["fit_error"])
        self.assertIn(
            "2DWavelengthDependent",
            namespace["comparison_fit_kwargs"],
        )

    def test_consensus_guide_links_notebook_and_removes_tbd(self):
        text = CONSENSUS.read_text(encoding="utf-8")
        self.assertIn(":doc:`../notebooks/pgmuvi_tutorial_2d`", text)
        self.assertNotIn("TBD: notebook tutorial", text)


if __name__ == "__main__":
    unittest.main()
