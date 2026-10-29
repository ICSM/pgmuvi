"Tests for interpretable single-source constraint diagnostics."

from __future__ import annotations

import unittest

from pgmuvi.single_source_analysis import (
    summarize_single_source_constraint_diagnostics,
)


class TestSingleSourceConstraintDiagnostics(unittest.TestCase):
    def _row(self, **updates):
        row = {
            "parameter": "covar_module.kernels.1.base_kernel.lengthscale",
            "applies_to": "covariance",
            "lower_bound": 0.2,
            "initial_value": 0.8,
            "fitted_value": 1.0,
            "upper_bound": 2.0,
            "fractional_position_within_bounds": 0.5,
            "minimum_distance_to_bound": 0.8,
            "constraint_registered": True,
            "value_initialized": True,
            "initial_inside_constraint": True,
            "fitted_inside_constraint": True,
        }
        row.update(updates)
        return row

    def test_interior_row_is_technically_satisfactory_without_overclaiming(self):
        result = summarize_single_source_constraint_diagnostics(
            [self._row()],
            parameter_workflow_report={
                "available": True,
                "applied": [
                    {
                        "parameter": (
                            "covar_module.kernels.1.base_kernel."
                            "raw_lengthscale"
                        ),
                        "value_reason": "data-derived wavelength estimate",
                        "constraint_reason": "data-derived wavelength interval",
                        "wavelength_estimate_provenance": {
                            "model_coordinate_space": "log10 wavelength",
                        },
                    }
                ],
            },
        )

        row = result["rows"][0]
        self.assertEqual(row["technical_status"], "constraint_respected")
        self.assertEqual(row["boundary_status"], "interior")
        self.assertTrue(row["technically_satisfactory"])
        self.assertFalse(result["summary"]["identification_established"])
        self.assertEqual(
            row["identification_status"],
            "not_established_by_constraint_diagnostics",
        )
        self.assertEqual(row["coordinate_system"], "log10 wavelength")
        self.assertIn("does not establish", row["interpretation"])

    def test_near_and_at_bound_rows_require_review(self):
        result = summarize_single_source_constraint_diagnostics(
            [
                self._row(
                    parameter="mean_module.bias",
                    applies_to="mean",
                    fractional_position_within_bounds=0.05,
                ),
                self._row(
                    parameter="mean_module.weights",
                    applies_to="mean",
                    fractional_position_within_bounds=0.005,
                ),
            ],
        )

        self.assertEqual(
            [row["boundary_status"] for row in result["rows"]],
            ["near_bound", "at_bound"],
        )
        self.assertTrue(
            all(row["review_required"] for row in result["rows"])
        )
        self.assertEqual(
            result["summary"]["status"],
            "technical_constraint_review_required",
        )

    def test_failed_registration_is_not_satisfactory(self):
        result = summarize_single_source_constraint_diagnostics(
            [
                self._row(
                    constraint_registered=False,
                    value_initialized=False,
                    initial_inside_constraint=False,
                    fitted_inside_constraint=False,
                )
            ],
        )

        row = result["rows"][0]
        self.assertEqual(
            row["technical_status"],
            "constraint_not_registered",
        )
        self.assertEqual(
            row["boundary_status"],
            "technical_check_failed",
        )
        self.assertFalse(row["technically_satisfactory"])
        self.assertIn("failed", row["interpretation"])

    def test_threshold_validation(self):
        with self.assertRaises(ValueError):
            summarize_single_source_constraint_diagnostics(
                [self._row()],
                at_bound_fraction=0.2,
                near_bound_fraction=0.1,
            )


if __name__ == "__main__":
    unittest.main()
