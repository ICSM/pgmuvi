"""No-training tests for ``examples/gp_prior_sampling.py``."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples/gp_prior_sampling.py"


def load_script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def load_module():
    spec = importlib.util.spec_from_file_location("gp_prior_sampling_example", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load GP-prior sampling example")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestGPriorSamplingExample(unittest.TestCase):
    def test_script_compiles_and_uses_current_parameter_layer(self):
        text = load_script_text()
        compile(text, str(SCRIPT), "exec")
        for token in [
            "ParameterEstimateCollection",
            "apply_parameter_estimates",
            "parameter_schema()",
            "model.forward(x).sample()",
            "QuasiPeriodicGPModel",
            "MaternGPModel",
            "SpectralMixtureGPModel",
            '"fit_started": False',
        ]:
            with self.subTest(token=token):
                self.assertIn(token, text)
        for token in [".fit(", ".initialize(", "raw_mixture", "auto_select_model"]:
            with self.subTest(token=token):
                self.assertNotIn(token, text)

    @unittest.skipUnless(
        importlib.util.find_spec("gpytorch") is not None,
        "gpytorch is required for GP-prior sampling example tests",
    )
    def test_all_kernel_report_is_no_training_and_complete(self):
        from pgmuvi.lightcurve import Lightcurve

        module = load_module()
        with patch.object(
            Lightcurve,
            "fit",
            side_effect=AssertionError("the GP-prior example must not fit"),
        ):
            report = module.build_report(
                "all",
                seed=1234,
                n_points=32,
                span=300.0,
                noise_sigma=0.08,
            )

        self.assertEqual(report["status"], "success")
        self.assertEqual(report["workflow"], "gp_prior_sampling")
        self.assertFalse(report["fit_started"])
        self.assertEqual(
            [item["kernel"] for item in report["kernels"]],
            list(module.SUPPORTED_KERNELS),
        )
        for item in report["kernels"]:
            self.assertFalse(item["fit_started"])
            self.assertEqual(item["n_points"], 32)
            self.assertEqual(item["xtransform"], "TimeCenter")
            self.assertTrue(
                all(
                    result["value"]
                    for result in item["parameter_application"].values()
                )
            )

    @unittest.skipUnless(
        importlib.util.find_spec("gpytorch") is not None,
        "gpytorch is required for GP-prior sampling example tests",
    )
    def test_cli_writes_optional_json_only_when_requested(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "report.json"
            status = module.main(
                [
                    "--kernel",
                    "quasi_periodic",
                    "--seed",
                    "42",
                    "--n-points",
                    "24",
                    "--json-output",
                    str(output),
                ]
            )
            self.assertEqual(status, 0)
            self.assertTrue(output.is_file())
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(payload["fit_started"])
            self.assertEqual(len(payload["kernels"]), 1)
            self.assertEqual(payload["kernels"][0]["kernel"], "quasi_periodic")
            self.assertEqual(sorted(path.name for path in root.iterdir()), ["report.json"])

    @unittest.skipUnless(
        importlib.util.find_spec("gpytorch") is not None,
        "gpytorch is required for GP-prior sampling example tests",
    )
    def test_same_seed_reproduces_summary_statistics(self):
        module = load_module()
        _, first = module.generate_prior_sample(
            "matern",
            seed=88,
            n_points=28,
            span=250.0,
            noise_sigma=0.1,
        )
        _, second = module.generate_prior_sample(
            "matern",
            seed=88,
            n_points=28,
            span=250.0,
            noise_sigma=0.1,
        )
        for key in [
            "time_span_days",
            "latent_mean",
            "latent_std",
            "observed_mean",
            "observed_std",
        ]:
            with self.subTest(key=key):
                self.assertEqual(first[key], second[key])


if __name__ == "__main__":
    unittest.main()
