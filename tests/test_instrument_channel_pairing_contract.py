"""Contracts for caller-supplied instrument-channel pairing provenance."""

from __future__ import annotations

import json
import unittest

import numpy as np

from pgmuvi.instrument_channel_calibration import (
    INSTRUMENT_CHANNEL_PAIRING_SCHEMA_VERSION,
    InstrumentChannelPairing,
)


class TestInstrumentChannelPairing(unittest.TestCase):
    def test_current_schema_version_is_v2(self):
        self.assertEqual(
            INSTRUMENT_CHANNEL_PAIRING_SCHEMA_VERSION,
            "pgmuvi-instrument-channel-pairing-v2",
        )

    @staticmethod
    def _pairing(**overrides):
        values = {
            "schema_version": (
                INSTRUMENT_CHANNEL_PAIRING_SCHEMA_VERSION
            ),
            "reference_channel": "reference",
            "channel": "target",
            "wavelength": 0.656,
            "reference_row_indices": (1, 4, 8),
            "channel_row_indices": (2, 5, 9),
            "reference_times": (10.0, 20.2, 30.0),
            "channel_times": (10.0, 20.0, 30.4),
            "time_unit": "day",
        }
        values.update(overrides)
        return InstrumentChannelPairing(**values)

    def test_explicit_pairing_is_immutable_and_json_safe(self):
        pairing = self._pairing()

        self.assertEqual(pairing.n_pairs, 3)
        self.assertEqual(
            pairing.time_differences,
            (0.0, 0.1999999999999993, -0.3999999999999986),
        )
        self.assertTrue(pairing.usable_for_affine_calibration)

        payload = pairing.to_dict()
        self.assertEqual(
            payload["schema_version"],
            INSTRUMENT_CHANNEL_PAIRING_SCHEMA_VERSION,
        )
        self.assertTrue(payload["caller_supplied_pairing"])
        self.assertFalse(payload["automatic_pair_construction"])
        self.assertFalse(
            payload["scientific_pairing_validation_performed"]
        )
        self.assertEqual(payload["n_pairs"], 3)
        self.assertEqual(payload["n_exact_time_matches"], 1)
        json.dumps(payload, allow_nan=False)

    def test_pairing_normalizes_text_and_numpy_indices(self):
        pairing = self._pairing(
            reference_channel=" reference ",
            channel=" target ",
            reference_row_indices=(
                np.int64(1),
                np.int64(4),
                np.int64(8),
            ),
            time_unit=" day ",
            method=" explicit-user-selection ",
        )

        self.assertEqual(pairing.reference_channel, "reference")
        self.assertEqual(pairing.channel, "target")
        self.assertEqual(pairing.time_unit, "day")
        self.assertEqual(
            pairing.method,
            "explicit-user-selection",
        )
        self.assertEqual(
            pairing.reference_row_indices,
            (1, 4, 8),
        )

    def test_previous_schema_version_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported instrument-channel pairing schema version",
        ):
            self._pairing(
                schema_version="pgmuvi-instrument-channel-pairing-v1",
            )

    def test_mismatched_pairing_lengths_are_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "same length",
        ):
            self._pairing(channel_times=(10.0, 20.0))

    def test_nonfinite_times_are_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "finite",
        ):
            self._pairing(
                reference_times=(10.0, np.nan, 30.0),
            )

    def test_boolean_times_are_rejected(self):
        for value in (True, np.bool_(False)):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    TypeError,
                    "numeric times, not boolean values",
                ):
                    self._pairing(
                        channel_times=(10.0, value, 30.4),
                    )

    def test_negative_or_boolean_indices_are_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "non-negative",
        ):
            self._pairing(
                channel_row_indices=(2, -1, 9),
            )

        with self.assertRaisesRegex(
            TypeError,
            "integer row indices",
        ):
            self._pairing(
                channel_row_indices=(2, True, 9),
            )

    def test_row_reuse_requires_explicit_permission(self):
        with self.assertRaisesRegex(
            ValueError,
            "allow_reference_reuse is False",
        ):
            self._pairing(
                reference_row_indices=(1, 1, 8),
            )

        pairing = self._pairing(
            reference_row_indices=(1, 1, 8),
            allow_reference_reuse=True,
            interpolation_used=True,
        )

        self.assertTrue(pairing.allow_reference_reuse)
        self.assertTrue(pairing.interpolation_used)

    def test_fewer_than_three_pairs_are_recorded_but_not_fit_ready(self):
        pairing = self._pairing(
            reference_row_indices=(1, 4),
            channel_row_indices=(2, 5),
            reference_times=(10.0, 20.2),
            channel_times=(10.0, 20.0),
        )

        self.assertEqual(pairing.n_pairs, 2)
        self.assertFalse(pairing.usable_for_affine_calibration)
        self.assertFalse(
            pairing.to_dict()["usable_for_affine_calibration"]
        )

    def test_pairing_does_not_choose_channels_or_construct_matches(self):
        payload = self._pairing().to_dict()

        self.assertEqual(payload["reference_channel"], "reference")
        self.assertEqual(payload["channel"], "target")
        self.assertFalse(payload["automatic_pair_construction"])


if __name__ == "__main__":
    unittest.main()
