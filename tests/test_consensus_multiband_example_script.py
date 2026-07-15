from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = ROOT / "examples" / "consensus_multiband_fit.py"


def _load_example_module():
    class StubConsensusFitError(RuntimeError):
        def __init__(self, message, *, failure_diagnostics=None):
            super().__init__(message)
            self.failure_diagnostics = failure_diagnostics or {"status": "failed"}
            self.failure_summary = None

    StubConsensusFitError.__name__ = "ConsensusFitError"

    class StubLightcurve:
        pass

    package = types.ModuleType("pgmuvi")
    package.__path__ = []
    lightcurve_module = types.ModuleType("pgmuvi.lightcurve")
    lightcurve_module.ConsensusFitError = StubConsensusFitError
    lightcurve_module.Lightcurve = StubLightcurve

    spec = importlib.util.spec_from_file_location(
        "consensus_multiband_fit",
        EXAMPLE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with patch.dict(
        sys.modules,
        {
            "pgmuvi": package,
            "pgmuvi.lightcurve": lightcurve_module,
        },
    ):
        spec.loader.exec_module(module)
    return module


class _FakePeak:
    rank = 1
    period = 120.0
    frequency = 1.0 / 120.0
    area_fraction = 0.8
    prominence = 0.7
    coherence_proxy = 12.0


class _FakeSummary:
    dominant_period = 120.0
    dominant_frequency = 1.0 / 120.0

    def as_dict(self):
        return {
            "method": "test",
            "dominant_period": self.dominant_period,
            "dominant_frequency": self.dominant_frequency,
            "peaks": [{"rank": 1, "period": 120.0}],
        }

    def get_primary_peak(self):
        return _FakePeak()


class _FakeLightcurve:
    def __init__(self, *, fit_exception=None):
        self.fit_exception = fit_exception
        self.fit_calls = []
        self.consensus_diagnostics = {
            "status": "passed",
            "accepted_bands": ["a", "b"],
        }
        self.fit_failed = False
        self.failure_reason = None
        self.failure_diagnostics = None
        self.failure_summary = None

    def fit(self, **kwargs):
        self.fit_calls.append(kwargs)
        if self.fit_exception is not None:
            raise self.fit_exception
        return {"ok": True}

    def get_period_summary(self):
        return _FakeSummary()

    def get_fit_history(self):
        return [{"success": True, "model_class": "2D"}]


class TestConsensusMultibandExample(unittest.TestCase):
    def setUp(self):
        self.module = _load_example_module()

    def test_lpv_consensus_defaults_to_quasi_periodic_handoff(self):
        args = self.module.parse_args(
            ["--model", "2DDustMean", "--fit-strategy", "consensus"]
        )
        self.module.validate_configuration(args)
        fit_kwargs = self.module.build_fit_kwargs(args)
        self.assertEqual(fit_kwargs["time_kernel_type"], "quasi_periodic")
        self.assertEqual(fit_kwargs["fit_strategy"], "consensus")

    def test_standard_2dseparable_omits_consensus_strategy(self):
        args = self.module.parse_args(
            ["--model", "2DSeparable", "--fit-strategy", "standard"]
        )
        self.module.validate_configuration(args)
        fit_kwargs = self.module.build_fit_kwargs(args)
        self.assertNotIn("fit_strategy", fit_kwargs)
        self.assertNotIn("time_kernel_type", fit_kwargs)

    def test_rejects_consensus_with_nonperiodic_lpv_time_kernel(self):
        args = self.module.parse_args(
            [
                "--model",
                "2DPowerLawMean",
                "--fit-strategy",
                "consensus",
                "--time-kernel-type",
                "matern",
            ]
        )
        with self.assertRaisesRegex(ValueError, "Consensus requires"):
            self.module.validate_configuration(args)

    def test_dry_run_loads_data_without_calling_fit(self):
        fake_lc = _FakeLightcurve()
        with patch.object(self.module, "load_lightcurve", return_value=fake_lc):
            rc = self.module.main(["--dry-run"])
        self.assertEqual(rc, 0)
        self.assertEqual(fake_lc.fit_calls, [])

    def test_success_writes_structured_artifacts(self):
        fake_lc = _FakeLightcurve()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "result"
            with patch.object(self.module, "load_lightcurve", return_value=fake_lc):
                rc = self.module.main(
                    ["--output-dir", str(output_dir), "--training-iter", "1", "--miniter", "0"]
                )

            self.assertEqual(rc, 0)
            self.assertEqual(len(fake_lc.fit_calls), 1)
            for name in [
                "run_configuration.json",
                "period_summary.json",
                "consensus_diagnostics.json",
                "fit_history.json",
            ]:
                self.assertTrue((output_dir / name).is_file(), name)

            payload = json.loads(
                (output_dir / "run_configuration.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["fit_kwargs"]["fit_strategy"], "consensus")

    def test_consensus_failure_returns_two_and_writes_failure_artifact(self):
        exc = self.module.ConsensusFitError(
            "not enough coherent bands",
            failure_diagnostics={"reason": "insufficient_accepted_bands"},
        )
        fake_lc = _FakeLightcurve(fit_exception=exc)
        fake_lc.fit_failed = True
        fake_lc.failure_reason = "insufficient_accepted_bands"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "failure"
            with patch.object(self.module, "load_lightcurve", return_value=fake_lc):
                rc = self.module.main(
                    ["--output-dir", str(output_dir), "--training-iter", "1", "--miniter", "0"]
                )

            self.assertEqual(rc, 2)
            payload = json.loads(
                (output_dir / "failure.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["exception_type"], "ConsensusFitError")
            self.assertEqual(
                payload["failure_diagnostics"]["reason"],
                "insufficient_accepted_bands",
            )


if __name__ == "__main__":
    unittest.main()
