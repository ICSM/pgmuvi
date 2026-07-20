"""Contracts for the D3 wavelength-validation documentation closeout."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = ROOT / "docs/source"
FUTURE_WORK = DOCS_ROOT / "future_work.rst"
ADVISORY = DOCS_ROOT / "howto/wavelength_advisory.rst"
REPRESENTATIVE = (
    DOCS_ROOT / "howto/representative_lpv_validation.rst"
)
SUMMARY = (
    ROOT
    / "examples/validation/"
    "d3_representative_lpv_calibration_summary.json"
)

CLOSED_MARKER = "TBD[wavelength-constraint-validation]"
OPEN_CALIBRATION_MARKER = (
    "TBD[instrument-channel-calibration]"
)


class TestD3WavelengthConstraintValidationCloseout(
    unittest.TestCase
):
    @classmethod
    def setUpClass(cls):
        cls.future_work = FUTURE_WORK.read_text(
            encoding="utf-8"
        )
        cls.advisory = ADVISORY.read_text(
            encoding="utf-8"
        )
        cls.representative = REPRESENTATIVE.read_text(
            encoding="utf-8"
        )
        cls.summary = json.loads(
            SUMMARY.read_text(encoding="utf-8")
        )

    def test_validation_marker_is_closed_across_docs(self):
        occurrences = []

        for doc_file in DOCS_ROOT.rglob("*.rst"):
            text = doc_file.read_text(encoding="utf-8")

            if CLOSED_MARKER in text:
                occurrences.append(
                    str(doc_file.relative_to(ROOT))
                )

        self.assertEqual(occurrences, [])

    def test_future_work_records_completed_status(self):
        self.assertIn(
            (
                "**Completed: representative "
                "wavelength-constraint validation**"
            ),
            self.future_work,
        )
        self.assertIn(
            (
                "d3_representative_lpv_"
                "calibration_summary.json"
            ),
            self.future_work,
        )
        self.assertIn(
            "597.37 days",
            self.future_work,
        )

    def test_representative_result_is_consolidated(self):
        required_text = [
            "D3 execution result and validation closeout",
            "789 of 10,815 observations",
            "17 observational channels",
            "16 distinct physical wavelengths",
            "2DWavelengthDependent",
            "2DDustMean",
            "2DPowerLawMean",
            "2DSeparable",
            "597.3663069387632",
            "completed_with_warnings",
            "selected_model",
            "advisory only",
            "aggregated by physical wavelength",
        ]

        normalized_representative = " ".join(
            self.representative.split()
        )

        for expected in required_text:
            with self.subTest(expected=expected):
                self.assertIn(
                    " ".join(expected.split()),
                    normalized_representative,
                )

    def test_instrument_calibration_marker_remains_open(self):
        self.assertIn(
            OPEN_CALIBRATION_MARKER,
            self.future_work,
        )
        self.assertIn(
            OPEN_CALIBRATION_MARKER,
            self.representative,
        )

    def test_advisory_document_points_to_completed_evidence(
        self,
    ):
        self.assertIn(
            (
                "completed wavelength-constraint "
                "validation evidence"
            ),
            self.advisory,
        )
        self.assertIn(
            ":doc:`representative_lpv_validation`",
            self.advisory,
        )

    def test_closeout_preserves_scientific_boundaries(self):
        boundaries = self.summary[
            "scientific_boundaries"
        ]

        self.assertTrue(boundaries["advisory_only"])
        self.assertFalse(
            boundaries[
                "automatic_model_selection_applied"
            ]
        )
        self.assertIsNone(boundaries["selected_model"])
        self.assertFalse(
            boundaries["truth_recovery_evidence"]
        )
        self.assertFalse(
            boundaries["full_data_execution_completed"]
        )
        self.assertFalse(
            boundaries[
                "instrument_channel_calibration_applied"
            ]
        )

    def test_execution_summary_remains_complete(self):
        execution = self.summary["execution"]

        self.assertEqual(execution["n_attempted"], 5)
        self.assertEqual(execution["n_passed"], 5)
        self.assertEqual(execution["n_failed"], 0)
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


if __name__ == "__main__":
    unittest.main()
