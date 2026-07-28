"""Regression checks for the maintained result-interpretation guide."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/source/howto/interpreting_results.rst"
HOWTO_INDEX = ROOT / "docs/source/howto/index.rst"
EXAMPLE = ROOT / "examples/interpret_pgmuvi_outputs.py"


class TestResultInterpretationDocs(unittest.TestCase):
    def setUp(self):
        self.text = DOC.read_text(encoding="utf-8")
        self.normalized = " ".join(self.text.split())

    def test_page_is_current_and_linked(self):
        self.assertIn("current through PR174", self.text)
        self.assertIn("interpreting_results", HOWTO_INDEX.read_text(encoding="utf-8"))

    def test_period_interpretation_contracts(self):
        required = [
            "get_period_summary()",
            "primary_peak_rank",
            "largest_area_peak_rank",
            "period_interval",
            "interval_definition",
            "q_factor",
            "component_diagnostics",
            "internal kernel diagnostics, not independent physical periods",
            "component_summaries",
            "fitted_period_drift_flag",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.normalized)

    def test_wavelength_trend_interpretation_contracts(self):
        required = [
            "raw_half_amplitude_q05_q95",
            "raw_half_amplitude_q02_5_q97_5",
            "fractional_half_amplitude_*",
            "noise_corrected_half_amplitude_*",
            "monotonicity_class",
            "not hypothesis tests",
            "Integer band codes are not physical wavelengths",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.normalized)

    def test_period_and_phase_boundary_semantics(self):
        required = [
            "Two-band consensus requires direct agreement",
            "two_band_fractional_frequency_difference",
            "two_band_max_fractional_frequency_difference",
            "two_band_frequency_agreement",
            "does not report the arithmetic midpoint",
            "Fixed-frequency phase and lag diagnostics",
            "one common ``reference_time``",
            'fixed_frequency_reference_time_source="global_time_midpoint"',
            "lag_linear_span",
            "minimum_circular_arc",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.normalized)

    def test_fit_quality_caveats_are_explicit(self):
        required = [
            "fit_quality_score",
            "training_nrmse_by_target_scale",
            "training_median_abs_standardized_residual",
            "training_outlier_fraction_3sigma",
            "training_reduced_chi2",
            "not cross-validation",
            "not held-out predictive performance",
            "not marginal likelihood or model evidence",
            "``fit_quality_available=False`` means the candidate is unscored",
            "fit_quality_ranking_status",
            "single_valid_candidate",
            "only_valid_model",
            "``top_ranked_model`` remains ``None``",
            "selected_model=None",
            "standardization_sigma_source",
            "already includes the likelihood noise",
            "not added again",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.normalized)

    def test_retained_state_marginal_likelihood_semantics(self):
        required = [
            "Retained-state marginal-likelihood fields",
            "training_log_marginal_likelihood",
            "Data log marginal likelihood **per observation**",
            "training_log_marginal_likelihood_total",
            "training_map_objective",
            "training_registered_log_prior",
            "training_registered_prior_count",
            "training_additional_objective_terms",
            "training_marginal_likelihood_evaluation_mode",
            "retained_current_state",
            "does not reuse the last loss recorded before an optimizer step",
            "not held-out predictive scores",
            "not Bayesian model probabilities",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.normalized)

    def test_ard_ceiling_interpretation_is_coordinate_specific(self):
        required = [
            "temporal_frequency",
            "wavelength_frequency",
            "mixture_means",
            "mixture_scales",
            "sm_ard_boundary_hits",
            "sm_ard_boundary_hit_counts_by_parameter",
            "sm_ard_boundary_component_counts_by_dimension",
            "sm_ard_boundary_pressure_scope",
            "sm_num_mixtures_fixed_at_one",
            "constrained_sm_ard_components",
            "compatibility aliases",
            "raw input coordinate",
            "Boundary-limited ARD parameters",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.normalized)

    def test_failure_and_fallback_contracts(self):
        required = [
            "input_validation",
            "consensus",
            "numerical_stability",
            "fit_execution",
            "fallback_diagnostics_available=True",
            "fallback_report.available=True",
            "failure_stage_counts",
            "exception_type_counts",
            "Do not turn an all-failed or single-survivor advisory run into a winner",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.normalized)

    def test_runnable_example_and_completed_notebook_coverage(self):
        self.assertTrue(EXAMPLE.is_file())
        self.assertIn("examples/interpret_pgmuvi_outputs.py", self.text)
        self.assertNotIn(
            "TBD[result-interpretation-notebook]",
            self.text,
        )
        self.assertIn(
            "tutorial_single_source_analysis",
            self.text,
        )


if __name__ == "__main__":
    unittest.main()
