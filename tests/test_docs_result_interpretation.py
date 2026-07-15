"""Regression checks for the PR102 result-interpretation guide."""

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
        self.assertIn("current through PR102", self.text)
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
            "selected_model=None",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.normalized)

    def test_ard_ceiling_interpretation_is_coordinate_specific(self):
        required = [
            "time_frequency",
            "wavelength_frequency",
            "constrained_sm_ard_components",
            "constrained_sm_ard_dimension_counts",
            "n_constrained_sm_ard_components",
            "boundary-limited",
            "fraction_of_upper",
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
            "Do not turn an all-failed advisory run into a winner",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.normalized)

    def test_runnable_example_and_future_marker(self):
        self.assertTrue(EXAMPLE.is_file())
        self.assertIn("examples/interpret_pgmuvi_outputs.py", self.text)
        self.assertIn("TBD[result-interpretation-notebook]", self.text)


if __name__ == "__main__":
    unittest.main()
