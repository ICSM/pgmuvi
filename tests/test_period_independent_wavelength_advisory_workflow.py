"""Tests for one-shot period-independent wavelength advisory workflow."""

import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.wavelength_diagnostics import (
    run_period_independent_wavelength_advisory_workflow,
)


def _model_kernel_config_report():
    return {
        "kind": "period_independent_wavelength_model_kernel_configs",
        "advisory_only": True,
        "model_kernel_configs": [
            {"rank": 1, "model": "2DDustMean", "fit_kwargs": {"model": "2DDustMean"}}
        ],
    }


def _run_report():
    return {
        "kind": "period_independent_wavelength_model_kernel_config_results",
        "runs_fits": True,
        "model_kernel_config_state_isolated": True,
        "mutates_input_lightcurve": False,
        "automatic_model_selection_applied": False,
        "selected_model": None,
        "model_kernel_config_results": [
            {
                "rank": 1,
                "model": "2DDustMean",
                "status": "passed",
                "fit_success": True,
                "consensus_success": True,
            }
        ],
    }


def _quality_report():
    return {
        "kind": "period_independent_wavelength_model_kernel_config_quality_scores",
        "score_kind": "training_residual_fit_quality",
        "scores_fit_quality": True,
        "runs_fits": False,
        "applies_to_fit": False,
        "advisory_only": True,
        "automatic_model_selection_applied": False,
        "selected_model": None,
        "ranking_status": "available",
        "fit_quality_ranking_available": True,
        "single_valid_candidate": False,
        "n_with_fit_quality": 2,
        "top_ranked_model": "2DDustMean",
        "top_ranked_fit_quality_score": 12.3,
        "only_valid_model": None,
        "only_valid_fit_quality_score": None,
        "ranked_results": [
            {
                "quality_rank": 1,
                "is_top_ranked": True,
                "model": "2DDustMean",
                "fit_quality_available": True,
                "fit_quality_score": 12.3,
                "fit_success": True,
                "consensus_success": True,
            },
            {
                "quality_rank": 2,
                "is_top_ranked": False,
                "model": "2DWavelengthDependent",
                "fit_quality_available": True,
                "fit_quality_score": 11.0,
                "fit_success": True,
                "consensus_success": True,
            },
        ],
    }


class TestPeriodIndependentWavelengthAdvisoryWorkflow(unittest.TestCase):
    def test_workflow_chains_helpers_without_selecting_model(self):
        lc = object()
        with patch(
            "pgmuvi.wavelength_diagnostics.build_period_independent_wavelength_model_kernel_configs",
            return_value=_model_kernel_config_report(),
        ) as build, patch(
            "pgmuvi.wavelength_diagnostics.run_period_independent_wavelength_model_kernel_configs",
            return_value=_run_report(),
        ) as run, patch(
            "pgmuvi.wavelength_diagnostics.score_period_independent_wavelength_model_kernel_config_quality",
            return_value=_quality_report(),
        ) as score, patch(
            "pgmuvi.wavelength_diagnostics.format_period_independent_wavelength_model_kernel_config_comparison_report",
            return_value="formatted report",
        ) as formatter:
            report = run_period_independent_wavelength_advisory_workflow(
                lc,
                base_fit_kwargs={"training_iter": 2},
                model_kernel_config_limit=1,
            )

        build.assert_called_once()
        run.assert_called_once()
        score.assert_called_once()
        formatter.assert_called_once()
        self.assertEqual(report["kind"], "period_independent_wavelength_advisory_workflow")
        self.assertTrue(report["advisory_only"])
        self.assertTrue(report["runs_fits"])
        self.assertTrue(report["built_model_kernel_config_report"])
        self.assertTrue(report["ran_model_kernel_config_fits"])
        self.assertTrue(report["scored_quality"])
        self.assertFalse(report["automatic_model_selection_applied"])
        self.assertIsNone(report["selected_model"])
        self.assertFalse(report["mutates_input_lightcurve"])
        self.assertEqual(report["fit_quality_ranking_status"], "available")
        self.assertTrue(report["fit_quality_ranking_available"])
        self.assertEqual(report["top_ranked_model"], "2DDustMean")
        self.assertIsNone(report["only_valid_model"])
        self.assertEqual(report["comparison_text_report"], "formatted report")
        self.assertIn("Period-independent wavelength advisory workflow", report["text_report"])
        self.assertIn("workflow_runs_model_kernel_config_fits: True", report["text_report"])
        self.assertIn("quality_score_report_runs_fits: False", report["text_report"])
        self.assertIn("formatted report", report["text_report"])

    def test_precomputed_reports_are_reused(self):
        lc = object()
        cand = _model_kernel_config_report()
        run_report = _run_report()
        quality = _quality_report()
        with patch(
            "pgmuvi.wavelength_diagnostics.build_period_independent_wavelength_model_kernel_configs"
        ) as build, patch(
            "pgmuvi.wavelength_diagnostics.run_period_independent_wavelength_model_kernel_configs"
        ) as run, patch(
            "pgmuvi.wavelength_diagnostics.score_period_independent_wavelength_model_kernel_config_quality"
        ) as score:
            report = run_period_independent_wavelength_advisory_workflow(
                lc,
                model_kernel_config_report=cand,
                run_report=run_report,
                quality_report=quality,
                make_text_report=False,
            )

        build.assert_not_called()
        run.assert_not_called()
        score.assert_not_called()
        self.assertFalse(report["built_model_kernel_config_report"])
        self.assertFalse(report["ran_model_kernel_config_fits"])
        self.assertFalse(report["scored_quality"])
        self.assertIs(report["model_kernel_config_report"], cand)
        self.assertIs(report["run_report"], run_report)
        self.assertIs(report["quality_report"], quality)
        self.assertIsNone(report["text_report"])


    def test_single_valid_candidate_is_not_promoted_to_top_ranked_model(self):
        quality = _quality_report()
        quality.update(
            {
                "ranking_status": "single_valid_candidate",
                "fit_quality_ranking_available": False,
                "single_valid_candidate": True,
                "n_with_fit_quality": 1,
                "top_ranked_model": None,
                "top_ranked_fit_quality_score": None,
                "only_valid_model": "2DDustMean",
                "only_valid_fit_quality_score": 12.3,
                "ranked_results": [quality["ranked_results"][0]],
            }
        )
        quality["ranked_results"][0]["is_top_ranked"] = False

        report = run_period_independent_wavelength_advisory_workflow(
            object(),
            model_kernel_config_report=_model_kernel_config_report(),
            run_report=_run_report(),
            quality_report=quality,
            make_text_report=False,
        )

        self.assertEqual(
            report["fit_quality_ranking_status"], "single_valid_candidate"
        )
        self.assertFalse(report["fit_quality_ranking_available"])
        self.assertIsNone(report["top_ranked_model"])
        self.assertEqual(report["only_valid_model"], "2DDustMean")
        self.assertTrue(report["fallback_diagnostics_available"])
        self.assertEqual(
            report["fallback_report"]["reason"],
            "only_one_model_kernel_config_has_fit_quality",
        )

    def test_unavailable_quality_diagnostics_clear_stale_top_rank(self):
        quality = {
            "kind": "period_independent_wavelength_model_kernel_config_quality_scores",
            "score_kind": "training_residual_fit_quality",
            "ranking_status": "available",
            "fit_quality_ranking_available": True,
            "n_with_fit_quality": 2,
            "top_ranked_model": "stale-model",
            "top_ranked_fit_quality_score": -1.0e9,
            "ranked_results": [],
        }

        report = run_period_independent_wavelength_advisory_workflow(
            object(),
            model_kernel_config_report=_model_kernel_config_report(),
            run_report=_run_report(),
            quality_report=quality,
            make_text_report=False,
        )

        self.assertEqual(report["fit_quality_ranking_status"], "unavailable")
        self.assertFalse(report["fit_quality_ranking_available"])
        self.assertIsNone(report["top_ranked_model"])
        self.assertIsNone(report["top_ranked_fit_quality_score"])
        self.assertTrue(report["fallback_diagnostics_available"])

    def test_lightcurve_method_delegates_to_module_function(self):
        lc = Lightcurve.__new__(Lightcurve)
        report = lc.run_period_independent_wavelength_advisory_workflow(
            model_kernel_config_report=_model_kernel_config_report(),
            run_report=_run_report(),
            quality_report=_quality_report(),
            make_text_report=False,
        )
        self.assertEqual(report["top_ranked_model"], "2DDustMean")
        self.assertFalse(report["automatic_model_selection_applied"])

    def test_can_return_figures_when_requested(self):
        fig = plt.figure()
        with patch(
            "pgmuvi.wavelength_diagnostics.plot_period_independent_wavelength_model_kernel_config_comparison",
            return_value={"score": fig},
        ):
            report = run_period_independent_wavelength_advisory_workflow(
                object(),
                model_kernel_config_report=_model_kernel_config_report(),
                run_report=_run_report(),
                quality_report=_quality_report(),
                make_text_report=False,
                make_plots=True,
            )
        self.assertTrue(report["makes_plots"])
        self.assertIn("score", report["figures"])
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
