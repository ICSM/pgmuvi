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


def _candidate_report():
    return {
        "kind": "period_independent_wavelength_fit_candidates",
        "advisory_only": True,
        "fit_candidates": [
            {"rank": 1, "model": "2DDustMean", "fit_kwargs": {"model": "2DDustMean"}}
        ],
    }


def _run_report():
    return {
        "kind": "period_independent_wavelength_fit_candidate_results",
        "runs_fits": True,
        "candidate_fit_state_isolated": True,
        "mutates_input_lightcurve": False,
        "automatic_model_selection_applied": False,
        "selected_model": None,
        "candidate_results": [
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
        "kind": "period_independent_wavelength_fit_candidate_quality_scores",
        "score_kind": "training_residual_fit_quality",
        "scores_fit_quality": True,
        "runs_fits": False,
        "applies_to_fit": False,
        "advisory_only": True,
        "automatic_model_selection_applied": False,
        "selected_model": None,
        "top_ranked_model": "2DDustMean",
        "top_ranked_fit_quality_score": 12.3,
        "ranked_results": [
            {
                "quality_rank": 1,
                "model": "2DDustMean",
                "fit_quality_score": 12.3,
                "fit_success": True,
                "consensus_success": True,
            }
        ],
    }


class TestPeriodIndependentWavelengthAdvisoryWorkflow(unittest.TestCase):
    def test_workflow_chains_helpers_without_selecting_model(self):
        lc = object()
        with patch(
            "pgmuvi.wavelength_diagnostics.build_period_independent_wavelength_fit_candidates",
            return_value=_candidate_report(),
        ) as build, patch(
            "pgmuvi.wavelength_diagnostics.run_period_independent_wavelength_fit_candidates",
            return_value=_run_report(),
        ) as run, patch(
            "pgmuvi.wavelength_diagnostics.score_period_independent_wavelength_fit_candidate_quality",
            return_value=_quality_report(),
        ) as score, patch(
            "pgmuvi.wavelength_diagnostics.format_period_independent_wavelength_fit_candidate_comparison_report",
            return_value="formatted report",
        ) as formatter:
            report = run_period_independent_wavelength_advisory_workflow(
                lc,
                base_fit_kwargs={"training_iter": 2},
                candidate_limit=1,
            )

        build.assert_called_once()
        run.assert_called_once()
        score.assert_called_once()
        formatter.assert_called_once()
        self.assertEqual(report["kind"], "period_independent_wavelength_advisory_workflow")
        self.assertTrue(report["advisory_only"])
        self.assertTrue(report["runs_fits"])
        self.assertTrue(report["built_candidate_report"])
        self.assertTrue(report["ran_candidate_fits"])
        self.assertTrue(report["scored_quality"])
        self.assertFalse(report["automatic_model_selection_applied"])
        self.assertIsNone(report["selected_model"])
        self.assertFalse(report["mutates_input_lightcurve"])
        self.assertEqual(report["top_ranked_model"], "2DDustMean")
        self.assertEqual(report["comparison_text_report"], "formatted report")
        self.assertIn("Period-independent wavelength advisory workflow", report["text_report"])
        self.assertIn("workflow_runs_candidate_fits: True", report["text_report"])
        self.assertIn("quality_score_report_runs_fits: False", report["text_report"])
        self.assertIn("formatted report", report["text_report"])

    def test_precomputed_reports_are_reused(self):
        lc = object()
        cand = _candidate_report()
        run_report = _run_report()
        quality = _quality_report()
        with patch(
            "pgmuvi.wavelength_diagnostics.build_period_independent_wavelength_fit_candidates"
        ) as build, patch(
            "pgmuvi.wavelength_diagnostics.run_period_independent_wavelength_fit_candidates"
        ) as run, patch(
            "pgmuvi.wavelength_diagnostics.score_period_independent_wavelength_fit_candidate_quality"
        ) as score:
            report = run_period_independent_wavelength_advisory_workflow(
                lc,
                candidate_report=cand,
                run_report=run_report,
                quality_report=quality,
                make_text_report=False,
            )

        build.assert_not_called()
        run.assert_not_called()
        score.assert_not_called()
        self.assertFalse(report["built_candidate_report"])
        self.assertFalse(report["ran_candidate_fits"])
        self.assertFalse(report["scored_quality"])
        self.assertIs(report["candidate_report"], cand)
        self.assertIs(report["run_report"], run_report)
        self.assertIs(report["quality_report"], quality)
        self.assertIsNone(report["text_report"])

    def test_fit_candidate_report_alias_is_accepted(self):
        report = run_period_independent_wavelength_advisory_workflow(
            object(),
            fit_candidate_report=_candidate_report(),
            run_report=_run_report(),
            quality_report=_quality_report(),
            make_text_report=False,
        )
        self.assertFalse(report["built_candidate_report"])

    def test_rejects_both_candidate_report_aliases(self):
        with self.assertRaisesRegex(ValueError, "candidate_report"):
            run_period_independent_wavelength_advisory_workflow(
                object(),
                candidate_report=_candidate_report(),
                fit_candidate_report=_candidate_report(),
            )

    def test_rejects_both_candidate_limit_aliases(self):
        with self.assertRaisesRegex(ValueError, "candidate_limit"):
            run_period_independent_wavelength_advisory_workflow(
                object(),
                run_report=_run_report(),
                quality_report=_quality_report(),
                candidate_limit=1,
                max_candidates=1,
            )

    def test_lightcurve_method_delegates_to_module_function(self):
        lc = Lightcurve.__new__(Lightcurve)
        report = lc.run_period_independent_wavelength_advisory_workflow(
            candidate_report=_candidate_report(),
            run_report=_run_report(),
            quality_report=_quality_report(),
            make_text_report=False,
        )
        self.assertEqual(report["top_ranked_model"], "2DDustMean")
        self.assertFalse(report["automatic_model_selection_applied"])

    def test_can_return_figures_when_requested(self):
        fig = plt.figure()
        with patch(
            "pgmuvi.wavelength_diagnostics.plot_period_independent_wavelength_fit_candidate_comparison",
            return_value={"score": fig},
        ):
            report = run_period_independent_wavelength_advisory_workflow(
                object(),
                candidate_report=_candidate_report(),
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
