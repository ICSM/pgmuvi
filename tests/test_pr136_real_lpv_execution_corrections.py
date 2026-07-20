"""Regression tests for PR136 representative-LPV execution."""

import json
import unittest
from pathlib import Path

from pgmuvi.wavelength_diagnostics import (
    build_period_independent_wavelength_model_kernel_configs,
)
from pgmuvi.wavelength_validation_real_lpv import (
    DEFAULT_REPRESENTATIVE_LPV_MODELS,
    _residual_wavelength_evidence,
    _warning_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


class TestRepresentativeLPVRequestedModelCoverage(unittest.TestCase):
    def test_all_requested_models_are_built_in_requested_order(self):
        plan = {
            "kind": "period_independent_wavelength_parameter_plan",
            "ranked_candidates": [
                {
                    "rank": 1,
                    "model": "2DDustMean",
                    "recommendation_strength": "advisory",
                },
                {
                    "rank": 2,
                    "model": "2DPowerLawMean",
                    "recommendation_strength": "advisory",
                },
                {
                    "rank": 3,
                    "model": "2DWavelengthDependent",
                    "recommendation_strength": "advisory",
                },
            ],
            "model_parameter_suggestions": {},
        }

        report = (
            build_period_independent_wavelength_model_kernel_configs(
                plan,
                include_models=DEFAULT_REPRESENTATIVE_LPV_MODELS,
                include_2d_baseline=True,
                base_fit_kwargs={
                    "training_iter": 5,
                    "miniter": 1,
                    "fit_strategy": "consensus",
                    "learn_additional_noise": True,
                },
            )
        )

        candidates = report["model_kernel_configs"]

        self.assertEqual(
            [candidate["model"] for candidate in candidates],
            list(DEFAULT_REPRESENTATIVE_LPV_MODELS),
        )

        by_model = {
            candidate["model"]: candidate
            for candidate in candidates
        }

        self.assertEqual(
            by_model["2DSeparable"]["source"],
            "explicit_include_models",
        )

        for model in DEFAULT_REPRESENTATIVE_LPV_MODELS[:-1]:
            self.assertEqual(
                by_model[model]["fit_kwargs"]["time_kernel_type"],
                "quasi_periodic",
            )

        self.assertNotIn(
            "time_kernel_type",
            by_model["2D"]["fit_kwargs"],
        )


class TestRepresentativeLPVPublicWorkflowConfiguration(
    unittest.TestCase
):
    def test_shared_base_kwargs_do_not_override_2d_time_kernel(self):
        workflow = json.loads(
            (
                ROOT
                / "examples/validation/"
                "d3_representative_lpv_workflow.json"
            ).read_text(encoding="utf-8")
        )

        self.assertNotIn(
            "time_kernel_type",
            workflow["base_fit_kwargs"],
        )
        self.assertEqual(
            workflow["base_fit_kwargs"]["fit_strategy"],
            "consensus",
        )
        self.assertTrue(
            workflow["base_fit_kwargs"]["learn_additional_noise"]
        )


class TestRepresentativeLPVWarningEvidence(unittest.TestCase):
    def test_attempt_warning_records_are_collected(self):
        evidence = _warning_evidence(
            [
                {
                    "model": "2DDustMean",
                    "model_kernel_config_id": (
                        "rank2_2DDustMean"
                    ),
                    "warning_records": [
                        {
                            "category": "NumericalWarning",
                            "message": "small noise rounded",
                            "severity": "warning",
                        }
                    ],
                }
            ]
        )

        self.assertEqual(evidence["n_warning_records"], 1)
        self.assertEqual(
            evidence["by_category"],
            {"NumericalWarning": 1},
        )
        self.assertEqual(
            evidence["records"][0]["model"],
            "2DDustMean",
        )

    def test_legacy_warnings_field_remains_supported(self):
        evidence = _warning_evidence(
            [
                {
                    "model": "2DPowerLawMean",
                    "warnings": [
                        {
                            "category": "UserWarning",
                            "message": "legacy warning",
                        }
                    ],
                }
            ]
        )

        self.assertEqual(evidence["n_warning_records"], 1)
        self.assertEqual(
            evidence["by_category"],
            {"UserWarning": 1},
        )


class TestRepresentativeLPVResidualWavelengthEvidence(
    unittest.TestCase
):
    def test_fit_quality_rows_are_exposed_by_physical_wavelength(
        self,
    ):
        workflow_report = {
            "run_report": {
                "model_kernel_config_results": [
                    {
                        "model": "2DDustMean",
                        "model_kernel_config_id": (
                            "rank2_2DDustMean"
                        ),
                        "fit_quality": {
                            "space": "transformed_training_space",
                            "n_points": 30,
                            "by_band": [
                                {
                                    "wavelength": 2.2,
                                    "n_points": 10,
                                    "bias": 0.2,
                                    "mae": 0.3,
                                    "median_abs_residual": 0.25,
                                    "rmse": 0.4,
                                },
                                {
                                    "wavelength": 0.65,
                                    "n_points": 20,
                                    "bias": -0.01,
                                    "mae": 0.02,
                                    "median_abs_residual": 0.015,
                                    "rmse": 0.03,
                                },
                            ],
                        },
                    }
                ]
            }
        }
        source_summary = {
            "n_observational_channels": 3,
            "n_distinct_physical_wavelengths": 2,
            "multiple_observational_channels_per_wavelength": True,
        }

        evidence = _residual_wavelength_evidence(
            workflow_report,
            source_summary,
        )

        self.assertTrue(evidence["available"])
        self.assertEqual(
            evidence[
                "n_models_with_residual_wavelength_evidence"
            ],
            1,
        )
        self.assertEqual(
            evidence["aggregation_scope"],
            "physical_wavelength",
        )
        self.assertTrue(
            evidence[
                "shared_wavelength_observational_channels_aggregated"
            ]
        )

        dust = evidence["by_model"]["2DDustMean"]

        self.assertEqual(dust["n_physical_wavelengths"], 2)
        self.assertEqual(
            [
                row["physical_wavelength"]
                for row in dust["by_physical_wavelength"]
            ],
            [0.65, 2.2],
        )
        self.assertEqual(
            dust["by_physical_wavelength"][1]["rmse"],
            0.4,
        )

    def test_unavailable_evidence_is_explicit(self):
        evidence = _residual_wavelength_evidence(
            {
                "run_report": {
                    "model_kernel_config_results": []
                }
            },
            {
                "n_observational_channels": 0,
                "n_distinct_physical_wavelengths": 0,
                "multiple_observational_channels_per_wavelength": (
                    False
                ),
            },
        )

        self.assertFalse(evidence["available"])
        self.assertEqual(
            evidence[
                "n_models_with_residual_wavelength_evidence"
            ],
            0,
        )
        self.assertEqual(evidence["by_model"], {})


if __name__ == "__main__":
    unittest.main()
