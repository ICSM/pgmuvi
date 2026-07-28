"""Tests for the complete single-source analysis helper layer."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import tempfile
import unittest

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.single_source_analysis import (
    SINGLE_SOURCE_ANALYSIS_STAGE_ORDER,
    build_per_observational_channel_period_evidence,
    build_phase_folded_prediction_summary,
    build_single_source_analysis_report,
    preserve_single_source_analysis_state,
    single_source_json_safe,
    summarize_single_source_fit_quality,
    summarize_single_source_noise_provenance,
    summarize_single_source_observational_channels,
    validate_single_source_stage_records,
    write_single_source_analysis_report,
)


class TestSingleSourceInputSummary(unittest.TestCase):
    def test_preserves_channel_identity_at_one_physical_wavelength(self):
        lightcurve = Lightcurve(
            torch.tensor(
                [
                    [0.0, 0.55],
                    [1.0, 0.55],
                    [0.0, 0.65],
                    [1.0, 0.65],
                    [0.5, 0.65],
                    [1.5, 0.65],
                ],
                dtype=torch.float64,
            ),
            torch.tensor(
                [10.0, 11.0, 9.0, 10.0, 8.5, 9.5],
                dtype=torch.float64,
            ),
            yerr=torch.full((6,), 0.2, dtype=torch.float64),
            band=np.asarray(["V", "V", "R0", "R0", "R1", "R1"]),
            center_time=False,
            check_sampling=False,
            max_samples=None,
            max_samples_per_band=None,
        )

        summary = summarize_single_source_observational_channels(lightcurve)

        self.assertEqual(summary["dimension_order"], ["time", "physical_wavelength"])
        self.assertEqual(summary["flux_domain"], "linear")
        self.assertEqual(summary["n_observational_channels"], 3)
        self.assertEqual(summary["n_physical_wavelengths"], 2)
        self.assertEqual(
            summary["observational_channel_order"],
            ["V", "R0", "R1"],
        )
        groups = summary["duplicate_physical_wavelength_groups"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["observational_channels"], ["R0", "R1"])


class TestPerChannelPeriodEvidence(unittest.TestCase):
    def test_runs_each_observational_channel_independently(self):
        rng = np.random.default_rng(123)
        rows = []
        flux = []
        error = []
        labels = []
        for index, wavelength in enumerate((0.55, 1.25)):
            time = np.sort(rng.uniform(0.0, 120.0, size=36))
            signal = 10.0 + index + np.sin(2.0 * np.pi * time / 12.0)
            rows.append(np.column_stack((time, np.full(time.shape, wavelength))))
            flux.append(signal)
            error.append(np.full(time.shape, 0.1))
            labels.append(np.full(time.shape, f"channel-{index}", dtype=object))
        lightcurve = Lightcurve(
            torch.as_tensor(np.vstack(rows), dtype=torch.float64),
            torch.as_tensor(np.concatenate(flux), dtype=torch.float64),
            yerr=torch.as_tensor(np.concatenate(error), dtype=torch.float64),
            band=np.concatenate(labels),
            center_time=False,
            check_sampling=False,
            max_samples=None,
            max_samples_per_band=None,
        )

        evidence = build_per_observational_channel_period_evidence(
            lightcurve,
            num_peaks=2,
            acf_n_lags=12,
        )

        self.assertEqual(evidence["n_observational_channels"], 2)
        self.assertEqual(
            [row["observational_channel"] for row in evidence["rows"]],
            ["channel-0", "channel-1"],
        )
        self.assertEqual(evidence["n_available"], 2)
        for row in evidence["rows"]:
            self.assertEqual(row["status"], "available")
            self.assertIn("peak_periods", row["lomb_scargle"])
            self.assertIn("strongest_positive_lag", row["acf"])


class TestSingleSourceNoiseProvenance(unittest.TestCase):
    def test_fixed_and_additional_variances_are_reported_separately(self):
        x = torch.linspace(0.0, 5.0, 12, dtype=torch.float64)
        y = torch.sin(x)
        yerr = torch.full_like(y, 0.2)
        lightcurve = Lightcurve(x, y, yerr=yerr, center_time=False)
        lightcurve.set_likelihood()

        summary = summarize_single_source_noise_provenance(lightcurve)

        self.assertTrue(summary["available"])
        self.assertEqual(
            summary["likelihood_class"],
            "FixedNoiseGaussianLikelihood",
        )
        self.assertTrue(summary["learn_additional_noise"])
        self.assertIsNotNone(summary["fixed_measurement_variance"])
        self.assertIsNotNone(summary["fitted_additional_noise_variance"])
        self.assertTrue(
            summary["additional_noise_is_not_measurement_error_replacement"]
        )

    def test_explicit_fixed_path_has_no_additional_variance(self):
        x = torch.linspace(0.0, 5.0, 12, dtype=torch.float64)
        y = torch.sin(x)
        yerr = torch.full_like(y, 0.2)
        lightcurve = Lightcurve(x, y, yerr=yerr, center_time=False)
        lightcurve.set_likelihood(likelihood="fixed")

        summary = summarize_single_source_noise_provenance(lightcurve)

        self.assertFalse(summary["learn_additional_noise"])
        self.assertIsNone(summary["fitted_additional_noise_variance"])


class TestSingleSourcePhaseAndStages(unittest.TestCase):
    def test_phase_rows_preserve_observational_channel_identity(self):
        summary = build_phase_folded_prediction_summary(
            {
                "time": np.asarray([0.0, 2.5, 5.0, 7.5]),
                "physical_wavelength": np.asarray([0.65] * 4),
                "observational_channel": np.asarray(["R0", "R0", "R1", "R1"]),
                "residual": np.asarray([1.0, -1.0, 2.0, -2.0]),
            },
            period=10.0,
        )

        self.assertEqual(summary["phase"], [0.0, 0.25, 0.5, 0.75])
        self.assertEqual(
            [row["observational_channel"] for row in summary["observational_channel_rows"]],
            ["R0", "R1"],
        )

    def test_stage_order_rejects_prediction_before_fit(self):
        with self.assertRaisesRegex(ValueError, "scientific order"):
            validate_single_source_stage_records(
                [
                    {"stage": "predictions", "status": "completed"},
                    {"stage": "gp_fit", "status": "completed"},
                ]
            )


class TestSingleSourceFitQuality(unittest.TestCase):
    def test_flags_large_standardized_residual_rms_without_auto_rejection(self):
        summary = summarize_single_source_fit_quality(
            [
                {
                    "observational_channel": "V",
                    "physical_wavelength": 0.55,
                    "n_points": 20,
                    "standardized_residual_rms": 0.9,
                },
                {
                    "observational_channel": "K",
                    "physical_wavelength": 2.2,
                    "n_points": 14,
                    "standardized_residual_rms": 2.5,
                },
                {
                    "observational_channel": "L",
                    "physical_wavelength": 3.5,
                    "n_points": 14,
                    "standardized_residual_rms": 7.8,
                },
            ]
        )

        self.assertEqual(summary["status"], "requires_review")
        self.assertTrue(summary["diagnostic_only"])
        self.assertFalse(summary["automatic_model_rejection"])
        self.assertEqual(summary["n_channels_requiring_review"], 2)
        self.assertEqual(summary["n_severe_channels"], 1)
        self.assertEqual(
            [
                row["observational_channel"]
                for row in summary["channels_requiring_review"]
            ],
            ["K", "L"],
        )
        self.assertEqual(
            [row["observational_channel"] for row in summary["severe_channels"]],
            ["L"],
        )
        self.assertEqual(len(summary["warning_messages"]), 1)

    def test_rejects_invalid_threshold_order(self):
        with self.assertRaises(ValueError):
            summarize_single_source_fit_quality(
                [],
                warning_threshold=3.0,
                severe_threshold=2.0,
            )


class TestSingleSourceStateRestoration(unittest.TestCase):
    def test_rng_dtype_cwd_environment_and_plotting_state_are_restored(self):
        random.seed(321)
        np.random.seed(321)
        torch.manual_seed(321)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        original_dtype = torch.get_default_dtype()
        original_cwd = Path.cwd()
        original_value = os.environ.get("PGMUVI_SINGLE_SOURCE_TEST")
        original_linewidth = plt.rcParams["lines.linewidth"]
        original_interactive = plt.isinteractive()

        with tempfile.TemporaryDirectory() as temporary_directory:
            with preserve_single_source_analysis_state(
                seed=9,
                default_dtype=torch.float32,
                working_directory=temporary_directory,
            ):
                random.random()
                np.random.random()
                torch.rand(())
                os.environ["PGMUVI_SINGLE_SOURCE_TEST"] = "changed"
                plt.rcParams["lines.linewidth"] = 9.0
                plt.ion()
                self.assertTrue(
                    os.path.samefile(Path.cwd(), temporary_directory)
                )
                self.assertEqual(torch.get_default_dtype(), torch.float32)

        self.assertEqual(random.getstate(), python_state)
        restored_numpy_state = np.random.get_state()
        self.assertEqual(restored_numpy_state[0], numpy_state[0])
        np.testing.assert_array_equal(restored_numpy_state[1], numpy_state[1])
        self.assertEqual(restored_numpy_state[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))
        self.assertEqual(torch.get_default_dtype(), original_dtype)
        self.assertTrue(os.path.samefile(Path.cwd(), original_cwd))
        self.assertEqual(os.environ.get("PGMUVI_SINGLE_SOURCE_TEST"), original_value)
        self.assertEqual(plt.rcParams["lines.linewidth"], original_linewidth)
        self.assertEqual(plt.isinteractive(), original_interactive)


class TestSingleSourceReport(unittest.TestCase):
    def test_report_is_json_safe_and_writable(self):
        stages = [
            {"stage": stage, "status": "completed"}
            for stage in SINGLE_SOURCE_ANALYSIS_STAGE_ORDER
        ]
        report = build_single_source_analysis_report(
            source_id="10131+3049",
            source_path="examples/data/10131+3049.csv",
            stage_records=stages,
            input_summary={"n_rows": np.int64(10), "bad": np.nan},
            period_evidence={"available": True},
            consensus={"period": np.float64(251.7)},
            wavelength_evidence={"available": True},
            constraint_diagnostics={"registered": True},
            fit_diagnostics={"loss": torch.tensor([3.0, 2.0])},
            prediction_diagnostics={"finite": True},
            residual_diagnostics={"rmse": np.float64(0.2)},
            phase_diagnostics=None,
            noise_provenance={"learn_additional_noise": True},
            duplicate_wavelength_resolution={
                "policy": "select",
                "selected": "KELT/OSN_Johnson.Cousins_R3_0",
            },
            runtime_environment={"python_version": "test"},
        )

        self.assertIsNone(report["input_summary"]["bad"])
        json.dumps(report, allow_nan=False)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = write_single_source_analysis_report(
                Path(temporary_directory) / "report.json",
                report,
            )
            loaded = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(loaded["source_id"], "10131+3049")
        self.assertEqual(
            loaded["stage_order"],
            list(SINGLE_SOURCE_ANALYSIS_STAGE_ORDER),
        )

    def test_json_safe_handles_gpytorch_constraint_text(self):
        value = single_source_json_safe(
            gpytorch.constraints.Interval(0.1, 1.0)
        )
        self.assertIsInstance(value, str)


if __name__ == "__main__":
    unittest.main()
