"""Deterministic instrument-channel time-pair construction tests."""

from __future__ import annotations

import json
import unittest

import numpy as np

from pgmuvi.instrument_channel_calibration import (
    InstrumentChannelPairingMethod,
    construct_instrument_channel_pairing,
)


class TestInstrumentChannelPairConstruction(unittest.TestCase):
    @staticmethod
    def _construct(**overrides):
        values = {
            "reference_times": (30.0, 10.0, 20.0),
            "channel_times": (20.0, 30.0, 10.0),
            "reference_channel": "reference",
            "channel": "target",
            "wavelength": 0.656,
            "time_unit": "day",
            "method": InstrumentChannelPairingMethod.EXACT_TIMESTAMP,
            "reference_row_indices": (30, 10, 20),
            "channel_row_indices": (200, 300, 100),
        }
        values.update(overrides)
        return construct_instrument_channel_pairing(**values)

    def test_exact_timestamp_matching_is_sorted_and_one_to_one(self):
        pairing = self._construct()

        self.assertEqual(pairing.reference_times, (10.0, 20.0, 30.0))
        self.assertEqual(pairing.channel_times, (10.0, 20.0, 30.0))
        self.assertEqual(pairing.reference_row_indices, (10, 20, 30))
        self.assertEqual(pairing.channel_row_indices, (100, 200, 300))
        self.assertEqual(
            pairing.method,
            InstrumentChannelPairingMethod.EXACT_TIMESTAMP.value,
        )
        self.assertIsNone(pairing.maximum_time_separation)
        self.assertEqual(len(set(pairing.reference_row_indices)), 3)
        self.assertEqual(len(set(pairing.channel_row_indices)), 3)

    def test_duplicate_exact_times_are_matched_deterministically(self):
        pairing = self._construct(
            reference_times=(1.0, 1.0, 2.0),
            channel_times=(1.0, 2.0, 1.0),
            reference_row_indices=(8, 4, 12),
            channel_row_indices=(9, 13, 5),
        )

        self.assertEqual(pairing.reference_row_indices, (4, 8, 12))
        self.assertEqual(pairing.channel_row_indices, (5, 9, 13))

    def test_nearest_matching_maximizes_pair_count_before_separation(self):
        pairing = self._construct(
            reference_times=(0.9, 1.1),
            channel_times=(0.0, 1.0),
            reference_row_indices=(9, 11),
            channel_row_indices=(0, 10),
            method=(
                InstrumentChannelPairingMethod.NEAREST_WITHIN_TOLERANCE
            ),
            maximum_time_separation=1.0,
        )

        self.assertEqual(pairing.n_pairs, 2)
        self.assertEqual(pairing.reference_row_indices, (9, 11))
        self.assertEqual(pairing.channel_row_indices, (0, 10))
        np.testing.assert_allclose(
            pairing.time_differences,
            (0.9, 0.1),
            rtol=0.0,
            atol=1.0e-15,
        )

    def test_equal_separation_tie_uses_earlier_sorted_observation(self):
        pairing = self._construct(
            reference_times=(1.0,),
            channel_times=(1.5, 0.5),
            reference_row_indices=(10,),
            channel_row_indices=(15, 5),
            method="nearest_within_tolerance",
            maximum_time_separation=0.5,
        )

        self.assertEqual(pairing.channel_times, (0.5,))
        self.assertEqual(pairing.channel_row_indices, (5,))

    def test_nearest_matching_requires_positive_tolerance(self):
        for value in (None, 0.0, -1.0, np.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "finite positive|requires maximum_time_separation",
                ):
                    self._construct(
                        method="nearest_within_tolerance",
                        maximum_time_separation=value,
                    )

    def test_exact_matching_rejects_nonzero_tolerance(self):
        with self.assertRaisesRegex(
            ValueError,
            "None or zero",
        ):
            self._construct(maximum_time_separation=0.1)

    def test_no_eligible_pairs_are_reported(self):
        with self.assertRaisesRegex(
            ValueError,
            "No eligible one-to-one",
        ):
            self._construct(
                reference_times=(1.0, 2.0),
                channel_times=(10.0, 20.0),
                reference_row_indices=(1, 2),
                channel_row_indices=(10, 20),
            )

    def test_duplicate_source_row_indices_are_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "same source row",
        ):
            self._construct(
                reference_row_indices=(1, 1, 2),
            )

    def test_unknown_method_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unknown instrument-channel pairing method",
        ):
            self._construct(method="automatic")

    def test_boolean_tolerance_is_rejected(self):
        with self.assertRaisesRegex(
            TypeError,
            "numeric, not boolean",
        ):
            self._construct(
                method="nearest_within_tolerance",
                maximum_time_separation=True,
            )

    def test_payload_records_construction_scope_and_unmatched_counts(self):
        pairing = self._construct(
            reference_times=(1.0, 2.0, 3.0),
            channel_times=(1.0, 3.0),
            reference_row_indices=(10, 20, 30),
            channel_row_indices=(100, 300),
        )

        payload = pairing.to_dict()
        self.assertFalse(payload["caller_supplied_pairing"])
        self.assertTrue(payload["automatic_pair_construction"])
        self.assertFalse(payload["automatic_reference_channel_selection"])
        self.assertFalse(payload["automatic_pairing_method_selection"])
        self.assertFalse(payload["automatic_time_tolerance_selection"])
        self.assertFalse(payload["allow_reference_reuse"])
        self.assertFalse(payload["allow_channel_reuse"])
        self.assertFalse(payload["interpolation_used"])
        self.assertEqual(payload["n_reference_observations"], 3)
        self.assertEqual(payload["n_channel_observations"], 2)
        self.assertEqual(payload["n_unmatched_reference_observations"], 1)
        self.assertEqual(payload["n_unmatched_channel_observations"], 0)
        json.dumps(payload, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
