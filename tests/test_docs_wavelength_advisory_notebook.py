"""Regression tests for the public wavelength-advisory notebook."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "docs/source/notebooks/tutorial_wavelength_advisory.ipynb"
OLD_NOTEBOOK = ROOT / "docs/source/notebooks/tutorial_model_selection.ipynb"
INDEX = ROOT / "docs/source/index.rst"
CONF = ROOT / "docs/source/conf.py"
MODEL_SELECTION = ROOT / "docs/source/howto/model_selection.rst"
ADVISORY = ROOT / "docs/source/howto/wavelength_advisory.rst"


def load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def notebook_text() -> str:
    chunks = []
    for cell in load_notebook().get("cells", []):
        source = cell.get("source", [])
        chunks.append("".join(source) if isinstance(source, list) else str(source))
    return "\n".join(chunks)


class TestWavelengthAdvisoryNotebook(unittest.TestCase):
    def test_stale_model_selection_notebook_is_replaced(self):
        self.assertFalse(OLD_NOTEBOOK.exists())
        self.assertTrue(NOTEBOOK.is_file())
        self.assertIn(
            "notebooks/tutorial_wavelength_advisory",
            INDEX.read_text(encoding="utf-8"),
        )
        conf = CONF.read_text(encoding="utf-8")
        self.assertNotIn("notebooks/tutorial_model_selection.ipynb", conf)
        self.assertNotIn("notebooks/tutorial_wavelength_advisory.ipynb", conf)

    def test_notebook_has_current_structure_and_clean_outputs(self):
        nb = load_notebook()
        self.assertEqual(nb.get("nbformat"), 4)
        self.assertEqual(
            nb.get("metadata", {}).get("kernelspec", {}).get("name"),
            "python3",
        )
        self.assertGreaterEqual(len(nb.get("cells", [])), 18)
        self.assertTrue(any(c.get("cell_type") == "markdown" for c in nb["cells"]))
        self.assertTrue(any(c.get("cell_type") == "code" for c in nb["cells"]))
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])

    def test_notebook_covers_current_advisory_contract(self):
        text = notebook_text()
        for token in [
            "diagnose_period_independent_wavelength_structure",
            "build_period_independent_wavelength_parameter_plan",
            "build_period_independent_wavelength_model_kernel_configs",
            "run_period_independent_wavelength_advisory_workflow",
            "raw_half_amplitude_q02_5_q97_5",
            "2DWavelengthDependent",
            "2DDustMean",
            "2DPowerLawMean",
            "2DSeparable",
            "fit_strategy",
            "quasi_periodic",
            "spectral_mixture default",
            "advisory_only",
            "selected_model",
            "top_ranked_model",
            "fallback_report",
            "RUN_FITS = False",
        ]:
            with self.subTest(token=token):
                self.assertIn(token, text)
        self.assertNotIn("2DAchromatic", text)
        self.assertIn("workflow['fallback_report']['available']", text)
        self.assertNotIn(
            "workflow['fallback_report']['fallback_diagnostics_available']",
            text,
        )

    def test_notebook_rejects_automatic_selection_framing(self):
        text = notebook_text()
        self.assertIn("without automatically selecting a model", text)
        self.assertIn("not equivalent to scientific validation", text)
        self.assertIn("selected_model` remains `None", text)
        self.assertIn("does not infer a period", text)
        self.assertNotIn("auto_select_model", text)
        self.assertNotIn("best model", text.lower())

    def test_notebook_removes_stale_stub_patterns(self):
        text = notebook_text()
        for token in [
            "TODO",
            "%pip",
            "!pip install",
            "Recommended model:",
            "compare log-likelihood and BIC",
            "Expand with auto_select_model",
        ]:
            self.assertNotIn(token, text)

    def test_all_code_cells_compile(self):
        for index, cell in enumerate(load_notebook().get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            compile(
                source,
                f"tutorial_wavelength_advisory.ipynb:cell-{index}",
                "exec",
            )

    @unittest.skipUnless(
        importlib.util.find_spec("gpytorch") is not None,
        "gpytorch is required for the advisory notebook smoke test",
    )
    def test_all_code_cells_execute_without_fitting_or_files(self):
        from pgmuvi.lightcurve import Lightcurve

        namespace = {"__name__": "__main__"}
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            Lightcurve,
            "fit",
            side_effect=AssertionError("the preparation path must not fit"),
        ):
            os.chdir(tmpdir)
            try:
                for index, cell in enumerate(load_notebook().get("cells", [])):
                    if cell.get("cell_type") != "code":
                        continue
                    source = "".join(cell.get("source", []))
                    exec(
                        compile(
                            source,
                            f"tutorial_wavelength_advisory.ipynb:cell-{index}",
                            "exec",
                        ),
                        namespace,
                    )
            finally:
                os.chdir(old_cwd)

            self.assertEqual(list(Path(tmpdir).iterdir()), [])

        self.assertFalse(namespace["RUN_FITS"])
        self.assertIsNone(namespace["workflow"])
        self.assertEqual(
            namespace["diagnostics"]["kind"],
            "period_independent_wavelength_diagnostics",
        )
        self.assertFalse(
            namespace["diagnostics"]["summary"]["uses_temporal_consensus"]
        )
        self.assertFalse(
            namespace["diagnostics"]["summary"]["uses_period_or_frequency"]
        )
        self.assertTrue(namespace["parameter_plan"]["advisory_only"])
        self.assertFalse(
            namespace["parameter_plan"]["automatic_initialization_applied"]
        )
        self.assertFalse(
            namespace["parameter_plan"]["automatic_constraints_applied"]
        )
        self.assertFalse(namespace["config_report"]["runs_fits"])
        self.assertTrue(
            {"2DWavelengthDependent", "2DDustMean", "2DPowerLawMean", "2D"}
            .issubset(set(namespace["config_models"]))
        )

    def test_companion_docs_link_and_describe_the_notebook(self):
        for path in [MODEL_SELECTION, ADVISORY]:
            text = path.read_text(encoding="utf-8")
            self.assertIn(
                ":doc:`../notebooks/tutorial_wavelength_advisory`",
                text,
            )
            self.assertIn("no-fitting", text.lower())
            self.assertIn("advisory", text.lower())


if __name__ == "__main__":
    unittest.main()
