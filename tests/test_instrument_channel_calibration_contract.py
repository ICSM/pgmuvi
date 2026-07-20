"""Contracts for future observational-channel calibration."""

from __future__ import annotations

import json
import unittest

import numpy as np

from pgmuvi.instrument_channel_calibration import (
    INSTRUMENT_CHANNEL_CALIBRATION_SCHEMA_VERSION,
    INSTRUMENT_CHANNEL_CALIBRATION_TBD_MARKER,
    InstrumentChannelCalibrationStatus,
    apply_instrument_channel_calibration,
    assess_instrument_channel_calibration_requirement,
    fit_instrument_channel_calibration,
)


class TestInstrumentChannelCalibrationAssessment(unittest.TestCase):
    def test_unique_wavelengths_do_not_require_calibration(self):
        assessment = assess_instrument_channel_calibration_requirement(
            np.asarray([0.5, 1.0, 2.0]),
            np.asarray(["r", "J", "K"]),
        )

        self.assertFalse(assessment.required)
        self.assertEqual(
            assessment.status,
            InstrumentChannelCalibrationStatus.NOT_REQUIRED,
        )
        self.assertEqual(assessment.shared_wavelength_groups, ())
        self.assertEqual(assessment.policy, "not_applicable")
        self.assertFalse(assessment.implemented)
        self.assertFalse(assessment.calibration_applied)

    def test_shared_wavelength_groups_are_deterministic_and_json_safe(self):
        assessment = assess_instrument_channel_calibration_requirement(
            np.asarray([0.65, 0.65, 0.65, 2.2]),
            np.asarray(
                [
                    "instrument-b",
                    "instrument-a",
                    "instrument-b",
                    "channel-k",
                ]
            ),
        )

        self.assertTrue(assessment.required)
        self.assertEqual(
            assessment.status,
            InstrumentChannelCalibrationStatus.REQUIRED_NOT_IMPLEMENTED,
        )
        self.assertEqual(
            assessment.shared_wavelength_channels,
            {0.65: ("instrument-a", "instrument-b")},
        )

        payload = assessment.to_dict()
        self.assertEqual(
            payload["schema_version"],
            INSTRUMENT_CHANNEL_CALIBRATION_SCHEMA_VERSION,
        )
        self.assertEqual(payload["status"], "required_not_implemented")
        self.assertTrue(payload["required"])
        self.assertFalse(payload["implemented"])
        self.assertFalse(payload["calibration_applied"])
        self.assertFalse(payload["silent_calibration_permitted"])
        self.assertEqual(
            payload["marker"],
            INSTRUMENT_CHANNEL_CALIBRATION_TBD_MARKER,
        )
        self.assertEqual(
            payload["policy"],
            "preserve_channels_without_calibration",
        )
        json.dumps(payload, allow_nan=False)

    def test_repeated_rows_from_one_channel_are_not_a_shared_group(self):
        assessment = assess_instrument_channel_calibration_requirement(
            [1.0, 1.0, 1.0],
            ["channel-a", "channel-a", "channel-a"],
        )

        self.assertFalse(assessment.required)

    def test_mismatched_row_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            assess_instrument_channel_calibration_requirement(
                [1.0, 2.0],
                ["channel-a"],
            )

    def test_nonfinite_wavelengths_are_rejected(self):
        for wavelength in (np.nan, np.inf, -np.inf):
            with self.subTest(wavelength=wavelength):
                with self.assertRaisesRegex(
                    ValueError,
                    "coordinates must be finite",
                ):
                    assess_instrument_channel_calibration_requirement(
                        [wavelength],
                        ["channel-a"],
                    )

    def test_finite_zero_and_negative_coordinates_are_supported(self):
        assessment = assess_instrument_channel_calibration_requirement(
            [0.0, 0.0, -1.0],
            ["channel-a", "channel-b", "channel-c"],
        )

        self.assertTrue(assessment.required)
        self.assertEqual(
            assessment.shared_wavelength_channels,
            {0.0: ("channel-a", "channel-b")},
        )

    def test_one_channel_cannot_map_to_multiple_wavelengths(self):
        with self.assertRaisesRegex(
            ValueError,
            "exactly one physical wavelength",
        ):
            assess_instrument_channel_calibration_requirement(
                [1.0, 2.0],
                ["channel-a", "channel-a"],
            )


class TestUnsupportedCalibrationCallables(unittest.TestCase):
    def test_fit_raises_not_implemented(self):
        with self.assertRaisesRegex(
            NotImplementedError,
            r"TBD\[instrument-channel-calibration\]",
        ):
            fit_instrument_channel_calibration(
                [1.0, 1.0],
                [2.0, 2.1],
                ["channel-a", "channel-b"],
            )

    def test_apply_raises_not_implemented(self):
        with self.assertRaisesRegex(
            NotImplementedError,
            r"TBD\[instrument-channel-calibration\]",
        ):
            apply_instrument_channel_calibration(
                [2.0, 2.1],
                ["channel-a", "channel-b"],
                calibration={},
            )


if __name__ == "__main__":
    unittest.main()
