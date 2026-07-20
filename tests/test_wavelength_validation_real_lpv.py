"""Contracts for representative real-LPV wavelength validation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.wavelength_validation import (
    WavelengthValidationPhase,
    WavelengthValidationSourceKind,
)
from pgmuvi.wavelength_validation_real_lpv import (
    DEFAULT_REPRESENTATIVE_LPV_MODELS,
    RepresentativeLPVValidationReport,
    build_representative_lpv_source_summary,
    build_representative_lpv_validation_report,
    run_representative_lpv_validation,
)


class RepresentativeLPVFixtures:
    @staticmethod
    def public_lightcurve():
        return Lightcurve.from_csv(
            Path("examples/data/10131+3049.csv"),
            check_sampling=False,
            max_samples=None,
            max_samples_per_band=None,
        )

    @staticmethod
    def workflow_report():
        return {
            "kind": "period_independent_wavelength_advisory_workflow",
            "advisory_only": True,
            "automatic_model_selection_applied": False,
            "selected_model": None,
            "fit_quality_ranking_status": "available",
            "fit_quality_ranking_available": True,
            "top_ranked_model": "2DDustMean",
            "top_ranked_fit_quality_score": 0.83,
            "model_kernel_config_report": {
                "period_independent_diagnostics": {
                    "kind": (
                        "period_independent_wavelength_structure_diagnostics"
                    ),
                    "n_usable_observational_channels": 17,
                    "n_distinct_physical_wavelengths": 16,
                    "median_flux_trend": "non_monotonic",
                    "amplitude_trend": "non_monotonic",
                },
            },
            "run_report": {
                "model_kernel_config_results": [
                    {
                        "model_kernel_config_id": "dust-qp-consensus",
                        "model": "2DDustMean",
                        "status": "passed",
                        "fit_success": True,
                        "fit_failed": False,
                        "technical_outcome": "completed_with_warnings",
                        "diagnostic_validity": "valid",
                        "scientific_usability": "usable",
                        "comparison_eligibility": "eligible",
                        "warning_count": 1,
                        "warnings": [
                            {
                                "category": "RuntimeWarning",
                                "message": "Test warning.",
                            }
                        ],
                        "consensus_success": True,
                        "consensus_period": 500.0,
                        "consensus_frequency": 0.002,
                        "n_accepted_bands": 15,
                        "n_rejected_bands": 2,
                        "fit_kwargs": {
                            "model": "2DDustMean",
                            "fit_strategy": "consensus",
                            "time_kernel_type": "quasi_periodic",
                            "learn_additional_noise": True,
                        },
                        "training_nrmse_by_target_scale": 0.12,
                        "training_median_abs_standardized_residual": 0.81,
                        "training_outlier_fraction_3sigma": 0.01,
                        "training_reduced_chi2": 1.08,
                        "parameter_workflow": {
                            "available": True,
                            "n_applied": 4,
                        },
                        "n_sm_ard_boundary_hits": 0,
                        "sm_ard_boundary_hits": [],
                        "n_constrained_sm_ard_components": 0,
                    },
                    {
                        "model_kernel_config_id": "joint-sm-consensus",
                        "model": "2D",
                        "status": "failed",
                        "fit_success": False,
                        "fit_failed": True,
                        "technical_outcome": "failed",
                        "diagnostic_validity": "invalid",
                        "scientific_usability": "unusable",
                        "comparison_eligibility": "ineligible",
                        "warning_count": 0,
                        "warnings": [],
                        "failure_code": "numerical_stability_failed",
                        "failure_stage": "numerical_stability",
                        "failure_substage": "cholesky",
                        "failure_stage_reason": (
                            "Covariance matrix was not positive definite."
                        ),
                        "exception_type": "NotPSDError",
                        "exception_message": "Matrix not positive definite.",
                        "fit_kwargs": {
                            "model": "2D",
                            "fit_strategy": "consensus",
                            "learn_additional_noise": True,
                        },
                        "n_sm_ard_boundary_hits": 2,
                        "sm_ard_boundary_hits": [
                            {
                                "parameter": "mixture_scales",
                                "dimension": "wavelength_frequency",
                                "component_index": 0,
                                "side": "upper",
                            },
                            {
                                "parameter": "mixture_means",
                                "dimension": "wavelength_frequency",
                                "component_index": 0,
                                "side": "lower",
                            },
                        ],
                        "n_constrained_sm_ard_components": 1,
                    },
                ],
            },
            "quality_report": {
                "score_kind": "training_residual_fit_quality",
                "fit_quality_ranking_status": "available",
                "fit_quality_ranking_available": True,
                "n_with_fit_quality": 1,
                "top_ranked_model": "2DDustMean",
                "top_ranked_fit_quality_score": 0.83,
                "ranked_results": [
                    {
                        "quality_rank": 1,
                        "model_kernel_config_id": "dust-qp-consensus",
                        "model": "2DDustMean",
                        "fit_quality_available": True,
                        "fit_quality_score": 0.83,
                        "is_top_ranked": True,
                    }
                ],
            },
            "advisory_conclusions": [
                {
                    "scope": "complete_configuration",
                    "subject": "2DDustMean",
                    "disposition": "remains_plausible",
                    "summary": (
                        "The fitted configuration remains plausible for this "
                        "source, subject to the recorded limitations."
                    ),
                    "models": ["2DDustMean"],
                    "model_kernel_config_ids": ["dust-qp-consensus"],
                }
            ],
            "unresolved_ambiguities": [
                {
                    "code": "training_space_ranking_only",
                    "summary": (
                        "Ranking uses training-space residual diagnostics."
                    ),
                }
            ],
        }


class TestRepresentativeLPVDefaults(unittest.TestCase):
    def test_default_model_order_prioritizes_lpv_relevant_families(self):
        self.assertEqual(
            DEFAULT_REPRESENTATIVE_LPV_MODELS,
            (
                "2DWavelengthDependent",
                "2DDustMean",
                "2DPowerLawMean",
                "2DSeparable",
                "2D",
            ),
        )
        self.assertNotIn(
            "2DAchromatic",
            DEFAULT_REPRESENTATIVE_LPV_MODELS,
        )


class TestRepresentativeLPVSourceSummary(
    RepresentativeLPVFixtures,
    unittest.TestCase,
):
    @classmethod
    def setUpClass(cls):
        cls.lightcurve = cls.public_lightcurve()
        cls.summary = build_representative_lpv_source_summary(
            cls.lightcurve
        )

    def test_public_source_counts_rows_channels_and_wavelengths(self):
        self.assertEqual(self.summary["n_rows"], 10815)
        self.assertEqual(
            self.summary["n_observational_channels"],
            17,
        )
        self.assertEqual(
            self.summary["n_distinct_physical_wavelengths"],
            16,
        )

    def test_public_source_records_shared_wavelength_channels(self):
        self.assertTrue(
            self.summary[
                "multiple_observational_channels_per_wavelength"
            ]
        )
        self.assertEqual(
            self.summary["instrument_calibration_status"],
            "not_implemented",
        )
        self.assertTrue(
            self.summary["instrument_calibration_tbd"]
        )
        self.assertEqual(
            self.summary["shared_wavelength_policy"],
            "preserve_channels_without_calibration",
        )

        shared = self.summary[
            "observational_channels_by_shared_wavelength"
        ]
        self.assertEqual(len(shared), 1)
        self.assertEqual(
            next(iter(shared.values())),
            [
                "KELT/OSN_Johnson.Cousins_R3_0",
                "KELT/OSN_Johnson.Cousins_R3_1",
            ],
        )

    def test_source_summary_is_json_safe(self):
        json.dumps(self.summary, allow_nan=False)


class TestRepresentativeLPVReport(
    RepresentativeLPVFixtures,
    unittest.TestCase,
):
    @classmethod
    def setUpClass(cls):
        cls.source_summary = build_representative_lpv_source_summary(
            cls.public_lightcurve()
        )
        cls.report = build_representative_lpv_validation_report(
            source_id="10131+3049",
            description=(
                "Representative public LPV validation source."
            ),
            source_summary=cls.source_summary,
            workflow_report=cls.workflow_report(),
        )
        cls.payload = cls.report.to_dict()

    def test_report_uses_observed_d3_scenario_without_truth(self):
        scenario = self.report.scenario
        self.assertIs(
            scenario.phase,
            WavelengthValidationPhase.D3_REPRESENTATIVE_LPV,
        )
        self.assertIs(
            scenario.source_kind,
            WavelengthValidationSourceKind.OBSERVED,
        )
        self.assertIsNone(scenario.truth)

    def test_report_is_advisory_and_never_selects_model(self):
        self.assertTrue(self.payload["advisory_only"])
        self.assertFalse(
            self.payload["automatic_model_selection_applied"]
        )
        self.assertIsNone(self.payload["selected_model"])

        self.assertEqual(
            self.payload["workflow_report"]["top_ranked_model"],
            "2DDustMean",
        )
        self.assertIsNone(self.payload["selected_model"])

    def test_report_collects_required_d3_evidence_sections(self):
        evidence = self.payload["source_evidence"]

        self.assertEqual(
            set(evidence),
            {
                "period_evidence",
                "fit_quality",
                "residual_wavelength_structure",
                "constraint_diagnostics",
                "warnings",
                "failures",
            },
        )

        self.assertEqual(
            evidence["period_evidence"]["n_models_with_period_evidence"],
            1,
        )
        self.assertEqual(
            evidence["period_evidence"]["by_model"][
                "2DDustMean"
            ]["consensus_period"],
            500.0,
        )

        self.assertEqual(
            evidence["fit_quality"]["ranking_status"],
            "available",
        )
        self.assertEqual(
            evidence["fit_quality"]["top_ranked_model"],
            "2DDustMean",
        )
        self.assertFalse(
            evidence["fit_quality"][
                "automatic_model_selection_applied"
            ]
        )

        self.assertEqual(
            evidence["residual_wavelength_structure"][
                "n_distinct_physical_wavelengths"
            ],
            16,
        )

        self.assertIn(
            "2DDustMean",
            evidence["constraint_diagnostics"]["by_model"],
        )
        self.assertIn(
            "2D",
            evidence["constraint_diagnostics"]["by_model"],
        )

        self.assertEqual(
            evidence["warnings"]["n_warning_records"],
            1,
        )
        self.assertEqual(
            evidence["failures"]["n_failed_attempts"],
            1,
        )
        self.assertEqual(
            evidence["failures"]["by_code"],
            {"numerical_stability_failed": 1},
        )

    def test_report_does_not_create_truth_recovery_gates(self):
        self.assertNotIn("truth", self.payload["source_evidence"])
        self.assertNotIn("recovery_metrics", self.payload)
        self.assertNotIn("recovery_gates", self.payload)
        self.assertFalse(
            self.payload["source_evidence"]["fit_quality"].get(
                "truth_recovery_evidence",
                False,
            )
        )

    def test_report_round_trip_and_json_serialization(self):
        rebuilt = RepresentativeLPVValidationReport.from_mapping(
            self.payload
        )
        self.assertEqual(rebuilt.to_dict(), self.payload)
        json.dumps(self.payload, allow_nan=False)


class TestRepresentativeLPVRunner(
    RepresentativeLPVFixtures,
    unittest.TestCase,
):
    def test_runner_reuses_advisory_workflow_with_explicit_models(self):
        lightcurve = self.public_lightcurve()
        captured = {}

        def fake_workflow_runner(candidate_lightcurve, **kwargs):
            captured["lightcurve"] = candidate_lightcurve
            captured["kwargs"] = dict(kwargs)
            return self.workflow_report()

        report = run_representative_lpv_validation(
            lightcurve,
            source_id="10131+3049",
            description="Representative public LPV source.",
            workflow_runner=fake_workflow_runner,
        )

        self.assertIs(captured["lightcurve"], lightcurve)
        self.assertEqual(
            tuple(captured["kwargs"]["include_models"]),
            DEFAULT_REPRESENTATIVE_LPV_MODELS,
        )
        self.assertTrue(
            captured["kwargs"]["include_2d_baseline"]
        )
        self.assertTrue(
            captured["kwargs"]["make_text_report"]
        )
        self.assertFalse(
            captured["kwargs"]["make_plots"]
        )

        payload = report.to_dict()
        self.assertTrue(payload["advisory_only"])
        self.assertFalse(
            payload["automatic_model_selection_applied"]
        )
        self.assertIsNone(payload["selected_model"])

    def test_runner_rejects_workflow_selection_claims(self):
        lightcurve = self.public_lightcurve()
        workflow = self.workflow_report()
        workflow["automatic_model_selection_applied"] = True
        workflow["selected_model"] = "2DDustMean"

        def invalid_workflow_runner(candidate_lightcurve, **kwargs):
            del candidate_lightcurve, kwargs
            return workflow

        with self.assertRaisesRegex(
            ValueError,
            "automatic model selection",
        ):
            run_representative_lpv_validation(
                lightcurve,
                source_id="invalid-selection-claim",
                workflow_runner=invalid_workflow_runner,
            )


if __name__ == "__main__":
    unittest.main()
