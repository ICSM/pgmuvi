import tempfile
import unittest
from pathlib import Path

import gpytorch
import numpy as np
import torch

from pgmuvi.wavelength_diagnostics import (
    _piwd_extract_fit_outcome,
    _piwd_extract_sm_ard_scale_diagnostics,
    run_period_independent_wavelength_advisory_workflow_batch,
)


class _KernelWithSMScales:
    def __init__(self, scales):
        self.mixture_scales = torch.as_tensor(scales, dtype=torch.float64)
        self.mixture_means = torch.full_like(self.mixture_scales, 0.5)
        self.raw_mixture_scales = torch.zeros_like(self.mixture_scales)
        self.raw_mixture_means = torch.zeros_like(self.mixture_means)
        lower = torch.tensor([[[1.0e-8, 1.0e-8]]], dtype=torch.float64)
        upper = torch.tensor([[[0.2, 0.2]]], dtype=torch.float64)
        self.raw_mixture_scales_constraint = gpytorch.constraints.Interval(
            lower, upper
        )
        self.raw_mixture_means_constraint = gpytorch.constraints.Interval(
            torch.zeros_like(lower), torch.ones_like(upper)
        )


class _ModelWithSMScales:
    def __init__(self, scales):
        self.sci_kernel = _KernelWithSMScales(scales)


class _LightcurveWithSMScales:
    def __init__(self, scales, diagnostics=None):
        self.model = _ModelWithSMScales(scales)
        self.consensus_diagnostics = diagnostics or {}


class TestWavelengthAdvisorySMARDDiagnostics(unittest.TestCase):
    def test_extracts_component_and_dimension_scale_ceiling_hits(self):
        lc = _LightcurveWithSMScales(
            scales=[[[0.10, 0.198]], [[0.190, 0.050]]],
            diagnostics={"consensus_scale_constraint_bounds": [1.0e-8, 0.2]},
        )

        diag = _piwd_extract_sm_ard_scale_diagnostics(
            lc,
            lc.consensus_diagnostics,
            ceiling_tolerance_fraction=0.05,
        )

        self.assertTrue(diag["available"])
        self.assertEqual(diag["sm_ard_dimension_names"], ["time_frequency", "wavelength_frequency"])
        self.assertEqual(diag["n_constrained_sm_ard_components"], 2)
        self.assertEqual(diag["constrained_sm_ard_dimension_counts"]["time_frequency"], 1)
        self.assertEqual(diag["constrained_sm_ard_dimension_counts"]["wavelength_frequency"], 1)
        hits = {
            (entry["component_index"], entry["dimension_name"])
            for entry in diag["constrained_sm_ard_components"]
        }
        self.assertEqual(hits, {(0, "wavelength_frequency"), (1, "time_frequency")})

    def test_fit_outcome_surfaces_sm_ard_summary_fields(self):
        lc = _LightcurveWithSMScales(
            scales=[[[0.10, 0.198]], [[0.190, 0.050]]],
            diagnostics={
                "consensus_success": True,
                "consensus_scale_constraint_bounds": [1.0e-8, 0.2],
            },
        )

        outcome = _piwd_extract_fit_outcome(
            {"model": "2D", "model_kernel_config_id": "rank4_2D_baseline"},
            status="passed",
            fitted_lightcurve=lc,
            fit_result=object(),
        )

        self.assertEqual(outcome["n_constrained_sm_ard_components"], 2)
        self.assertEqual(outcome["n_constrained_sm_time_components"], 1)
        self.assertEqual(outcome["n_constrained_sm_wavelength_components"], 1)
        self.assertIn("sm_ard_scale_diagnostics", outcome)
        self.assertIn("sm_ard_diagnostics", outcome)
        self.assertEqual(outcome["n_sm_ard_boundary_hits"], 2)
        self.assertEqual(outcome["sm_ard_boundary_pressure_scope"], "both")
        self.assertEqual(
            outcome["constrained_sm_ard_dimension_counts"]["wavelength_frequency"],
            1,
        )

    def test_fit_outcome_records_explicit_single_component_request(self):
        lc = _LightcurveWithSMScales(
            scales=[[[0.10, 0.198]]],
            diagnostics={"consensus_success": True},
        )

        outcome = _piwd_extract_fit_outcome(
            {
                "model": "2D",
                "model_kernel_config_id": "rank4_2D_baseline",
                "fit_kwargs": {"num_mixtures": 1},
            },
            status="passed",
            fitted_lightcurve=lc,
            fit_result=object(),
        )

        self.assertEqual(outcome["sm_num_mixtures"], 1)
        self.assertTrue(outcome["sm_num_mixtures_is_one"])
        self.assertTrue(outcome["sm_num_mixtures_fixed_at_one"])
        self.assertEqual(
            outcome["sm_num_mixtures_request_source"],
            "explicit_fit_kwargs",
        )

    def test_batch_long_form_csv_includes_sm_ard_columns(self):
        workflow = {
            "kind": "period_independent_wavelength_advisory_workflow",
            "fit_quality_ranking_status": "single_valid_candidate",
            "fit_quality_ranking_available": False,
            "only_valid_model": "2D",
            "top_ranked_model": None,
            "top_ranked_fit_quality_score": None,
            "score_kind": "training_residual_fit_quality",
            "run_report": {
                "kind": "period_independent_wavelength_model_kernel_config_results",
                "model_kernel_config_results": [
                    {
                        "model_kernel_config_id": "rank4_2D_baseline",
                        "rank": 4,
                        "model": "2D",
                        "status": "passed",
                        "fit_success": True,
                        "fit_kwargs": {
                            "fit_strategy": "consensus",
                            "time_kernel_type": "spectral_mixture",
                        },
                        "n_sm_ard_boundary_hits": 5,
                        "sm_ard_boundary_pressure_scope": "both",
                        "n_sm_temporal_boundary_components": 1,
                        "n_sm_wavelength_boundary_components": 2,
                        "sm_ard_boundary_hit_counts_by_parameter": {
                            "mixture_means": 3,
                            "mixture_scales": 2,
                        },
                        "sm_ard_boundary_hit_counts_by_dimension": {
                            "temporal_frequency": 2,
                            "wavelength_frequency": 3,
                        },
                        "sm_ard_boundary_component_counts_by_dimension": {
                            "temporal_frequency": 1,
                            "wavelength_frequency": 2,
                        },
                        "sm_ard_boundary_hit_counts_by_side": {
                            "lower": 3,
                            "upper": 2,
                        },
                        "sm_ard_boundary_hits": [
                            {
                                "parameter": "mixture_means",
                                "component_index": 0,
                                "dimension_name": "wavelength_frequency",
                                "bound_side": "upper",
                            }
                        ],
                        "sm_num_mixtures": 2,
                        "sm_requested_num_mixtures": 2,
                        "sm_num_mixtures_is_one": False,
                        "sm_num_mixtures_fixed_at_one": False,
                        "sm_num_mixtures_request_source": "explicit_fit_kwargs",
                        "n_constrained_sm_ard_components": 2,
                        "n_constrained_sm_time_components": 1,
                        "n_constrained_sm_wavelength_components": 1,
                        "constrained_sm_ard_dimension_counts": {
                            "time_frequency": 1,
                            "wavelength_frequency": 1,
                        },
                        "constrained_sm_ard_components": [
                            {"component_index": 0, "dimension_name": "wavelength_frequency"},
                            {"component_index": 1, "dimension_name": "time_frequency"},
                        ],
                    }
                ],
            },
            "quality_report": {
                "kind": "period_independent_wavelength_model_kernel_config_quality_scores",
                "score_kind": "training_residual_fit_quality",
                "ranking_status": "single_valid_candidate",
                "fit_quality_ranking_available": False,
                "n_with_fit_quality": 1,
                "only_valid_model": "2D",
                "ranked_results": [
                    {
                        "quality_rank": 1,
                        "is_top_ranked": False,
                        "model": "2D",
                        "fit_success": True,
                        "fit_quality_available": True,
                        "fit_quality_score": 12.0,
                    }
                ],
            },
        }

        class _FakeLightcurve:
            def run_period_independent_wavelength_advisory_workflow(self, **kwargs):
                return workflow

        with tempfile.TemporaryDirectory() as tmp:
            report = run_period_independent_wavelength_advisory_workflow_batch(
                [{"source_id": "src", "lightcurve": _FakeLightcurve()}],
                output_dir=tmp,
                export=False,
                batch_prefix="batch",
            )
            detail_path = Path(report["batch_model_kernel_config_csv_path"])
            text = detail_path.read_text(encoding="utf-8")

        self.assertIn("n_sm_ard_boundary_hits", text)
        self.assertIn("sm_ard_boundary_pressure_scope", text)
        self.assertIn("sm_ard_boundary_hit_counts_by_parameter", text)
        self.assertIn("sm_ard_boundary_hits", text)
        self.assertIn("sm_num_mixtures_fixed_at_one", text)
        self.assertIn("n_constrained_sm_ard_components", text)
        self.assertIn("n_constrained_sm_time_components", text)
        self.assertIn("n_constrained_sm_wavelength_components", text)
        self.assertEqual(
            report["model_kernel_config_results"][0]["n_sm_ard_boundary_hits"],
            5,
        )
        self.assertEqual(
            report["model_kernel_config_results"][0][
                "sm_ard_boundary_pressure_scope"
            ],
            "both",
        )
        self.assertEqual(
            report["model_kernel_config_results"][0]["n_constrained_sm_ard_components"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
