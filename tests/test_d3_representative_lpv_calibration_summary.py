"""Contract tests for the PR136 D3 calibration summary."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = (
    ROOT
    / "examples/validation/"
    "d3_representative_lpv_calibration_summary.json"
)


class TestRepresentativeLPVCalibrationSummary(
    unittest.TestCase
):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(
            SUMMARY_PATH.read_text(encoding="utf-8")
        )

    def test_identity_and_scope(self):
        self.assertEqual(
            self.payload["kind"],
            "representative_lpv_calibration_summary",
        )
        self.assertEqual(
            self.payload["schema_version"],
            "1.0",
        )
        self.assertEqual(
            self.payload["source_id"],
            "10131+3049",
        )
        self.assertEqual(
            self.payload["execution_scope"],
            (
                "deterministically_sampled_"
                "representative_observed_lpv"
            ),
        )

    def test_sampling_is_bounded(self):
        source = self.payload["source_data"]

        self.assertEqual(
            source["n_rows_original"],
            10815,
        )
        self.assertEqual(
            source["n_rows_sampled"],
            789,
        )
        self.assertLessEqual(
            source["n_rows_sampled"],
            source["max_samples"],
        )
        self.assertLessEqual(
            source[
                "maximum_retained_samples_per_observational_channel"
            ],
            source[
                "max_samples_per_observational_channel"
            ],
        )
        self.assertEqual(
            source["n_observational_channels"],
            17,
        )
        self.assertEqual(
            source["n_distinct_physical_wavelengths"],
            16,
        )
        self.assertTrue(
            source[
                "multiple_observational_channels_per_wavelength"
            ]
        )

    def test_all_priority_models_completed(self):
        execution = self.payload["execution"]

        self.assertEqual(
            execution["models"],
            [
                "2DWavelengthDependent",
                "2DDustMean",
                "2DPowerLawMean",
                "2DSeparable",
                "2D",
            ],
        )
        self.assertEqual(execution["n_attempted"], 5)
        self.assertEqual(execution["n_passed"], 5)
        self.assertEqual(execution["n_failed"], 0)
        self.assertAlmostEqual(
            execution["consensus_period_days"],
            597.3663069387632,
            places=8,
        )

        by_model = {
            row["model"]: row
            for row in self.payload["model_results"]
        }

        for model in execution["models"][:-1]:
            self.assertEqual(
                by_model[model][
                    "fit_configuration"
                ]["time_kernel_type"],
                "quasi_periodic",
            )

        self.assertEqual(
            by_model["2D"][
                "fit_configuration"
            ]["time_kernel_type"],
            "model_default",
        )

    def test_warning_failure_and_residual_evidence(self):
        self.assertEqual(
            self.payload["warning_evidence"][
                "n_warning_records"
            ],
            5,
        )
        self.assertEqual(
            self.payload["warning_evidence"][
                "by_category"
            ],
            {"NumericalWarning": 5},
        )
        self.assertEqual(
            self.payload["failure_evidence"][
                "n_failed_attempts"
            ],
            0,
        )
        residual = self.payload[
            "residual_wavelength_structure"
        ]
        self.assertTrue(residual["available"])
        self.assertEqual(
            residual["aggregation_scope"],
            "physical_wavelength",
        )
        self.assertEqual(
            residual["n_models_with_evidence"],
            5,
        )

    def test_ranking_remains_advisory(self):
        fit_quality = self.payload["fit_quality"]
        boundaries = self.payload[
            "scientific_boundaries"
        ]

        self.assertEqual(
            fit_quality["ranked_models"],
            [
                "2DDustMean",
                "2DWavelengthDependent",
                "2DSeparable",
                "2DPowerLawMean",
                "2D",
            ],
        )
        self.assertEqual(
            fit_quality["top_ranked_model"],
            "2DDustMean",
        )
        self.assertTrue(boundaries["advisory_only"])
        self.assertFalse(
            boundaries[
                "automatic_model_selection_applied"
            ]
        )
        self.assertIsNone(boundaries["selected_model"])
        self.assertFalse(
            boundaries["full_data_execution_completed"]
        )
        self.assertFalse(
            boundaries[
                "closes_wavelength_constraint_validation_marker"
            ]
        )

    def test_2d_structural_diagnostics_are_recorded(self):
        by_model = {
            row["model"]: row
            for row in self.payload["model_results"]
        }
        baseline = by_model["2D"]

        self.assertEqual(
            baseline["constraint_diagnostics"][
                "n_sm_ard_boundary_hits"
            ],
            4,
        )
        self.assertEqual(
            baseline["constraint_diagnostics"][
                "n_constrained_sm_ard_components"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
