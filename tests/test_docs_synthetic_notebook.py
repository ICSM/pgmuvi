"""Regression tests for the refreshed public synthetic-data notebook."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "docs/source/notebooks/tutorial_synthetic.ipynb"
INDEX = ROOT / "docs/source/index.rst"
CONF = ROOT / "docs/source/conf.py"
CONCEPTS = ROOT / "docs/source/concepts.rst"
SYNTHETIC_API = ROOT / "docs/source/pgmuvi.synthetic.rst"


def load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def notebook_text() -> str:
    chunks = []
    for cell in load_notebook().get("cells", []):
        source = cell.get("source", [])
        chunks.append("".join(source) if isinstance(source, list) else str(source))
    return "\n".join(chunks)


class TestRefreshedSyntheticNotebook(unittest.TestCase):
    def test_notebook_is_public_and_not_excluded(self):
        self.assertIn(
            "notebooks/tutorial_synthetic",
            INDEX.read_text(encoding="utf-8"),
        )
        self.assertNotIn(
            "notebooks/tutorial_synthetic.ipynb",
            CONF.read_text(encoding="utf-8"),
        )

    def test_notebook_has_current_structure_and_metadata(self):
        nb = load_notebook()
        self.assertEqual(nb.get("nbformat"), 4)
        self.assertEqual(
            nb.get("metadata", {}).get("kernelspec", {}).get("name"),
            "python3",
        )
        self.assertGreaterEqual(len(nb.get("cells", [])), 18)
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

    def test_notebook_contains_all_public_generator_contracts(self):
        text = notebook_text()
        for token in [
            "make_simple_sinusoid_1d",
            "make_multi_sinusoid_1d",
            "make_chromatic_sinusoid_2d",
            "make_multi_sinusoid_chromatic_2d",
            '"noise_type": "poisson"',
            'noise_type="gaussian"',
            "noise_type=None",
            'amplitude_law="linear"',
            'amplitude_law="extinction"',
            'phase_law="linear"',
            "n_per_band=(45, 65)",
            "central-95% amplitudes",
            "injected periods",
            "fit_strategy",
            "advisory rather than automatic",
        ]:
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_notebook_states_gp_sampling_boundary_and_does_not_fit(self):
        text = notebook_text()
        self.assertIn("do **not** draw a realization from a Gaussian-process prior", text)
        self.assertIn("No GP model is created or trained", text)
        self.assertIn("does not\ncall `fit()`", text)
        self.assertNotIn(".fit(", text)
        self.assertNotIn("auto_select_model", text)

    def test_notebook_removes_stale_stub_patterns(self):
        text = notebook_text()
        for token in [
            "TODO",
            "%pip",
            "!pip install",
            "git+https://github.com/ICSM/pgmuvi.git",
            "Expand with pgmuvi.synthetic API calls",
            "Loop over cadences",
        ]:
            self.assertNotIn(token, text)

    def test_all_code_cells_compile(self):
        for index, cell in enumerate(load_notebook().get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            compile(
                source,
                f"tutorial_synthetic.ipynb:cell-{index}",
                "exec",
            )

    @unittest.skipUnless(
        importlib.util.find_spec("gpytorch") is not None,
        "gpytorch is required for the synthetic notebook smoke test",
    )
    def test_all_code_cells_execute_without_training_or_files(self):
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
                            f"tutorial_synthetic.ipynb:cell-{index}",
                            "exec",
                        ),
                        namespace,
                    )
            finally:
                os.chdir(old_cwd)

            self.assertEqual(list(Path(tmpdir).iterdir()), [])

        self.assertEqual(namespace["lc_simple"].ndim, 1)
        self.assertEqual(namespace["lc_chromatic"].ndim, 2)
        self.assertEqual(namespace["lc_multi_chromatic"].ndim, 2)
        self.assertTrue(
            torch.allclose(
                namespace["lc_simple"].ydata,
                namespace["lc_simple_repeat"].ydata,
            )
        )
        self.assertIsNone(
            getattr(namespace["lc_noise_free"], "yerr", None)
        )
        self.assertEqual(namespace["actual_counts"].tolist(), [48, 60, 72])
        self.assertEqual(namespace["injected_periods_1d"], [120.0, 60.0, 37.0])
        self.assertEqual(namespace["injected_periods_2d"], [400.0, 200.0])
        self.assertTrue(np.all(namespace["multi_counts"] >= 45))
        self.assertTrue(np.all(namespace["multi_counts"] <= 65))

    def test_companion_docs_are_accurate_and_link_notebook(self):
        concepts = CONCEPTS.read_text(encoding="utf-8")
        api = SYNTHETIC_API.read_text(encoding="utf-8")
        for text in [concepts, api]:
            self.assertIn(":doc:`notebooks/tutorial_synthetic`", text)
            self.assertIn("analytic", text.lower())
            self.assertIn("Gaussian-process prior", text)
        self.assertNotIn(
            "generate synthetic light curves from a GP model with known hyperparameters",
            concepts,
        )


if __name__ == "__main__":
    unittest.main()
