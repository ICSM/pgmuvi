"""Regression tests for the refreshed public preprocessing notebook."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "docs/source/notebooks/tutorial_preprocessing.ipynb"
INDEX = ROOT / "docs/source/index.rst"
CONF = ROOT / "docs/source/conf.py"
PREPROCESSING = ROOT / "docs/source/howto/preprocessing.rst"
LOADING = ROOT / "docs/source/howto/loading_data.rst"


def load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def notebook_text() -> str:
    chunks = []
    for cell in load_notebook().get("cells", []):
        source = cell.get("source", [])
        chunks.append("".join(source) if isinstance(source, list) else str(source))
    return "\n".join(chunks)


class TestRefreshedPreprocessingNotebook(unittest.TestCase):
    def test_notebook_is_public_and_not_excluded(self):
        self.assertIn(
            "notebooks/tutorial_preprocessing",
            INDEX.read_text(encoding="utf-8"),
        )
        self.assertNotIn(
            "notebooks/tutorial_preprocessing.ipynb",
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

    def test_notebook_contains_current_preprocessing_contracts(self):
        text = notebook_text()
        for token in [
            "Lightcurve.from_csv(",
            "warnings.catch_warnings",
            "check_sampling=False",
            "DEFAULT_DTYPE",
            "TimeCenter",
            "compute_sampling_metrics()",
            "assess_sampling_quality(",
            "check_variability(",
            "subsample_lightcurve(",
            "random_seed=SUBSAMPLE_SEED",
            "compute_sampling_metrics_per_band()",
            "assess_sampling_quality_per_band(",
            "check_variability_per_band(",
            "filter_well_sampled_bands(",
            "filter_variable_bands(",
            "do not select a GP model",
        ]:
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_notebook_removes_stale_stub_patterns(self):
        text = notebook_text()
        for token in [
            "%pip",
            "!pip install",
            "git+https://github.com/ICSM/pgmuvi.git",
            "Placeholder",
            "# TODO",
            "> **TODO:**",
        ]:
            self.assertNotIn(token, text)

    def test_all_code_cells_compile(self):
        for index, cell in enumerate(load_notebook().get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            compile(
                source,
                f"tutorial_preprocessing.ipynb:cell-{index}",
                "exec",
            )

    @unittest.skipUnless(
        importlib.util.find_spec("gpytorch") is not None,
        "gpytorch is required for the preprocessing notebook smoke test",
    )
    def test_all_cells_execute_without_gp_training_or_persistent_outputs(self):
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
                            f"tutorial_preprocessing.ipynb:cell-{index}",
                            "exec",
                        ),
                        namespace,
                    )
            finally:
                os.chdir(old_cwd)

            self.assertEqual(list(Path(tmpdir).iterdir()), [])

        self.assertEqual(len(namespace["lc_csv"].ydata), 24)
        self.assertTrue(namespace["csv_warning_messages"])
        self.assertTrue(namespace["sampling_passes_1d"])
        self.assertTrue(
            namespace["variability_1d"]["decision"].startswith("VARIABLE")
        )
        self.assertEqual(len(namespace["subsample_indices"]), 120)
        self.assertEqual(
            namespace["band_sampling"]["summary"]["n_passing"],
            2,
        )
        self.assertEqual(
            namespace["band_variability"]["summary"]["n_variable"],
            2,
        )
        np.testing.assert_allclose(
            namespace["well_sampled_wavelengths"],
            [0.55, 0.80],
        )
        np.testing.assert_allclose(
            namespace["variable_wavelengths"],
            [0.55, 0.80],
        )

    def test_companion_guides_link_notebook_and_remove_tbd_markers(self):
        preprocessing = PREPROCESSING.read_text(encoding="utf-8")
        loading = LOADING.read_text(encoding="utf-8")
        for text in [preprocessing, loading]:
            self.assertIn(":doc:`../notebooks/tutorial_preprocessing`", text)
        self.assertNotIn("TBD[notebook-preprocessing]", preprocessing)
        self.assertNotIn("TBD[notebook-lightcurve-validation]", loading)


if __name__ == "__main__":
    unittest.main()
