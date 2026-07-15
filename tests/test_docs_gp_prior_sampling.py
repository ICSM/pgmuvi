"""Regression tests for the refreshed GP-prior mock-data workflow."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "docs/source/notebooks/pgmuvi_mock_data_from_gp.ipynb"
INDEX = ROOT / "docs/source/index.rst"
CONF = ROOT / "docs/source/conf.py"
STATUS = ROOT / "docs/source/notebook_status.rst"
CONCEPTS = ROOT / "docs/source/concepts.rst"
HOWTO_INDEX = ROOT / "docs/source/howto/index.rst"
GUIDE = ROOT / "docs/source/howto/gp_prior_sampling.rst"
GPS_API = ROOT / "docs/source/pgmuvi.gps.rst"
EXAMPLE = ROOT / "examples/gp_prior_sampling.py"


def load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def notebook_text() -> str:
    chunks = []
    for cell in load_notebook().get("cells", []):
        source = cell.get("source", [])
        chunks.append("".join(source) if isinstance(source, list) else str(source))
    return "\n".join(chunks)


class TestGPriorSamplingDocumentation(unittest.TestCase):
    def test_notebook_is_public_and_not_excluded(self):
        self.assertIn(
            "notebooks/pgmuvi_mock_data_from_gp",
            INDEX.read_text(encoding="utf-8"),
        )
        self.assertNotIn(
            "notebooks/pgmuvi_mock_data_from_gp.ipynb",
            CONF.read_text(encoding="utf-8"),
        )

    def test_notebook_has_current_structure_and_clean_outputs(self):
        nb = load_notebook()
        self.assertEqual(nb.get("nbformat"), 4)
        self.assertEqual(
            nb.get("metadata", {}).get("kernelspec", {}).get("name"),
            "python3",
        )
        self.assertGreaterEqual(len(nb.get("cells", [])), 20)
        self.assertTrue(any(c.get("cell_type") == "markdown" for c in nb["cells"]))
        self.assertTrue(any(c.get("cell_type") == "code" for c in nb["cells"]))
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])

    def test_notebook_uses_current_prior_sampling_contract(self):
        text = notebook_text()
        for token in [
            "DEFAULT_DTYPE",
            "QuasiPeriodicGPModel",
            "MaternGPModel",
            "SpectralMixtureGPModel",
            "ParameterEstimateCollection",
            "apply_parameter_estimates",
            "parameter_schema()",
            "model.forward(x).sample()",
            "covar_module.outputscale",
            "period_length",
            "mixture_means",
            "mixture_scales",
            "mixture_weights",
            "frequency = 1 / period",
            "FIT_STARTED = False",
            "Multiple realizations",
            "posterior predictive",
        ]:
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_notebook_avoids_stale_or_unsafe_patterns(self):
        text = notebook_text()
        for token in [
            "TODO",
            "%pip",
            "!pip install",
            "git+https://github.com/ICSM/pgmuvi.git",
            ".fit(",
            "auto_select_model",
            ".initialize(",
            "raw_mixture",
            "raw_period",
            "_eval()",
            "training_iter=500",
        ]:
            with self.subTest(token=token):
                self.assertNotIn(token, text)
        self.assertIn("does not condition on an\nobserved light curve", text)
        self.assertIn("No fit, optimizer, posterior", text)

    def test_all_code_cells_compile(self):
        for index, cell in enumerate(load_notebook().get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            compile(
                source,
                f"pgmuvi_mock_data_from_gp.ipynb:cell-{index}",
                "exec",
            )

    @unittest.skipUnless(
        importlib.util.find_spec("gpytorch") is not None,
        "gpytorch is required for the GP-prior notebook smoke test",
    )
    def test_all_code_cells_execute_without_fitting_or_files(self):
        from pgmuvi.lightcurve import Lightcurve

        namespace = {"__name__": "__main__"}
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            Lightcurve,
            "fit",
            side_effect=AssertionError("the GP-prior tutorial must not fit"),
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
                            f"pgmuvi_mock_data_from_gp.ipynb:cell-{index}",
                            "exec",
                        ),
                        namespace,
                    )
            finally:
                os.chdir(old_cwd)

            self.assertEqual(list(Path(tmpdir).iterdir()), [])

        self.assertFalse(namespace["FIT_STARTED"])
        self.assertEqual(namespace["lc_qp"].ndim, 1)
        self.assertEqual(namespace["lc_matern"].ndim, 1)
        self.assertEqual(namespace["lc_sm"].ndim, 1)
        self.assertEqual(namespace["times"].dtype, namespace["DEFAULT_DTYPE"])
        self.assertEqual(namespace["ensemble_shape"], (3, 72))
        self.assertTrue(namespace["all_values_applied"])
        self.assertEqual(namespace["sm_periods"], [180.0, 65.0])
        self.assertTrue(torch.isfinite(namespace["qp_latent"]).all())
        self.assertTrue(torch.isfinite(namespace["matern_latent"]).all())
        self.assertTrue(torch.isfinite(namespace["sm_latent"]).all())

    def test_companion_guide_example_and_api_links_exist(self):
        self.assertTrue(GUIDE.is_file())
        self.assertTrue(EXAMPLE.is_file())
        for path in [CONCEPTS, GPS_API]:
            text = path.read_text(encoding="utf-8")
            self.assertIn(":doc:`notebooks/pgmuvi_mock_data_from_gp`", text)
            self.assertIn("GP-prior", text)
        self.assertIn("gp_prior_sampling", HOWTO_INDEX.read_text(encoding="utf-8"))
        guide = GUIDE.read_text(encoding="utf-8")
        for token in [
            "apply_parameter_estimates",
            "model.forward(time).sample()",
            "frequency = 1 / period",
            "Posterior predictive",
            "examples/gp_prior_sampling.py",
            ":doc:`consensus_fitting`",
            ":doc:`interpreting_results`",
        ]:
            with self.subTest(token=token):
                self.assertIn(token, guide)

    def test_status_page_records_pr107_and_only_mcmc_remains_quarantined(self):
        text = STATUS.read_text(encoding="utf-8")
        self.assertIn("current through PR107", text)
        self.assertIn("Maintained GP-prior mock-data tutorial", text)
        self.assertIn("Refreshed in PR107", text)
        self.assertNotIn("TBD[notebook-mock-data-refresh]", text)
        quarantined = text.split("Quarantined or pending-refresh notebooks", 1)[1]
        self.assertIn("pgmuvi_tutorial_mcmc.ipynb", quarantined)
        self.assertNotIn("pgmuvi_mock_data_from_gp.ipynb", quarantined)


if __name__ == "__main__":
    unittest.main()
