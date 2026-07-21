import json
import unittest

from pgmuvi.instrument_channel_calibration import (
    INSTRUMENT_CHANNEL_CALIBRATION_MODEL_SCHEMA_VERSION,
    INSTRUMENT_CHANNEL_CALIBRATION_PLAN_SCHEMA_VERSION,
    INSTRUMENT_CHANNEL_PAIRING_SCHEMA_VERSION,
    InstrumentChannelCalibration,
    InstrumentChannelCalibrationChannelPlan,
    InstrumentChannelCalibrationDisposition,
    InstrumentChannelCalibrationGroupPlan,
    InstrumentChannelCalibrationPlan,
    InstrumentChannelPairing,
    InstrumentChannelPairingMethod,
    assess_instrument_channel_calibration_requirement,
    define_instrument_channel_calibration_plan,
)


class TestInstrumentChannelCalibrationOrchestrationContract(
    unittest.TestCase
):
    @staticmethod
    def _assessment():
        return assess_instrument_channel_calibration_requirement(
            [1.0, 1.0, 2.0, 2.0, 2.0],
            ["A", "B", "C", "D", "E"],
        )

    @staticmethod
    def _planned(channel, **overrides):
        values = {
            "channel": channel,
            "disposition": (
                InstrumentChannelCalibrationDisposition.PLANNED
            ),
            "pairing_method": (
                InstrumentChannelPairingMethod.EXACT_TIMESTAMP.value
            ),
            "time_unit": "day",
            "calibration_family": "affine",
        }
        values.update(overrides)
        return InstrumentChannelCalibrationChannelPlan(**values)

    @classmethod
    def _groups(cls):
        return (
            InstrumentChannelCalibrationGroupPlan(
                physical_wavelength=1.0,
                reference_channel="A",
                channel_plans=(cls._planned("B"),),
            ),
            InstrumentChannelCalibrationGroupPlan(
                physical_wavelength=2.0,
                reference_channel="C",
                channel_plans=(
                    cls._planned(
                        "D",
                        pairing_method=(
                            InstrumentChannelPairingMethod
                            .NEAREST_WITHIN_TOLERANCE.value
                        ),
                        maximum_time_separation=0.25,
                    ),
                    InstrumentChannelCalibrationChannelPlan(
                        channel="E",
                        disposition=(
                            InstrumentChannelCalibrationDisposition
                            .SKIPPED
                        ),
                        reason="No scientifically accepted pairing.",
                    ),
                ),
            ),
        )

    def test_plan_is_deterministic_and_json_safe(self):
        plan = define_instrument_channel_calibration_plan(
            self._assessment(),
            reversed(self._groups()),
        )

        self.assertIsInstance(
            plan,
            InstrumentChannelCalibrationPlan,
        )
        self.assertEqual(
            plan.schema_version,
            INSTRUMENT_CHANNEL_CALIBRATION_PLAN_SCHEMA_VERSION,
        )
        self.assertEqual(
            [
                group.physical_wavelength
                for group in plan.group_plans
            ],
            [1.0, 2.0],
        )

        payload = plan.to_dict()
        json.dumps(payload, allow_nan=False)
        self.assertFalse(
            payload["automatic_reference_channel_selection"]
        )
        self.assertFalse(
            payload["automatic_pair_construction"]
        )
        self.assertFalse(
            payload["automatic_calibration_fit"]
        )
        self.assertFalse(
            payload["automatic_calibration_application"]
        )
        self.assertFalse(
            payload["lightcurve_mutation_performed"]
        )

    def test_plan_requires_exact_assessment_group_coverage(self):
        with self.assertRaisesRegex(
            ValueError,
            "cover exactly",
        ) as context:
            define_instrument_channel_calibration_plan(
                self._assessment(),
                self._groups()[:1],
            )

        message = str(context.exception)
        self.assertIn(
            "expected_groups=",
            message,
        )
        self.assertIn(
            "observed_groups=",
            message,
        )
        self.assertIn(
            "(2.0, ('C', 'D', 'E'))",
            message,
        )
        self.assertIn(
            "observed_groups=((1.0, ('A', 'B')),)",
            message,
        )

    def test_reference_channel_must_belong_to_assessment_group(self):
        invalid = InstrumentChannelCalibrationGroupPlan(
            physical_wavelength=1.0,
            reference_channel="Z",
            channel_plans=(self._planned("B"),),
        )
        with self.assertRaisesRegex(
            ValueError,
            "cover exactly",
        ):
            define_instrument_channel_calibration_plan(
                self._assessment(),
                (invalid, self._groups()[1]),
            )

    def test_duplicate_non_reference_channels_are_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "appear only once",
        ):
            InstrumentChannelCalibrationGroupPlan(
                physical_wavelength=1.0,
                reference_channel="A",
                channel_plans=(
                    self._planned("B"),
                    self._planned("B"),
                ),
            )

    def test_planned_channel_requires_explicit_choices(self):
        with self.assertRaisesRegex(
            ValueError,
            "pairing_method",
        ):
            InstrumentChannelCalibrationChannelPlan(
                channel="B",
                disposition="planned",
                time_unit="day",
                calibration_family="affine",
            )

        with self.assertRaisesRegex(
            ValueError,
            "time_unit",
        ):
            InstrumentChannelCalibrationChannelPlan(
                channel="B",
                disposition="planned",
                pairing_method="exact_timestamp",
                calibration_family="affine",
            )

        with self.assertRaisesRegex(
            ValueError,
            "calibration_family",
        ):
            InstrumentChannelCalibrationChannelPlan(
                channel="B",
                disposition="planned",
                pairing_method="exact_timestamp",
                time_unit="day",
            )

    def test_unknown_pairing_method_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported instrument-channel pairing method",
        ):
            self._planned(
                "B",
                pairing_method="automatic_best_match",
            )

    def test_nearest_pairing_requires_positive_tolerance(self):
        with self.assertRaisesRegex(
            ValueError,
            "finite positive",
        ):
            self._planned(
                "B",
                pairing_method="nearest_within_tolerance",
            )

    def test_exact_pairing_rejects_nonzero_tolerance(self):
        with self.assertRaisesRegex(
            ValueError,
            "permits no non-zero",
        ):
            self._planned(
                "B",
                maximum_time_separation=0.1,
            )

    def test_skipped_and_unavailable_require_reason_only(self):
        skipped = InstrumentChannelCalibrationChannelPlan(
            channel="B",
            disposition="skipped",
            reason="Caller excluded this channel.",
        )
        self.assertEqual(
            skipped.disposition,
            InstrumentChannelCalibrationDisposition.SKIPPED,
        )

        with self.assertRaisesRegex(
            ValueError,
            "require a reason",
        ):
            InstrumentChannelCalibrationChannelPlan(
                channel="B",
                disposition="unavailable",
            )

        with self.assertRaisesRegex(
            ValueError,
            "cannot contain",
        ):
            InstrumentChannelCalibrationChannelPlan(
                channel="B",
                disposition="skipped",
                reason="No valid calibration.",
                pairing_method="exact_timestamp",
            )

    def test_pairing_and_calibration_provenance_are_preserved(self):
        pairing = InstrumentChannelPairing(
            schema_version=(
                INSTRUMENT_CHANNEL_PAIRING_SCHEMA_VERSION
            ),
            reference_channel="A",
            channel="B",
            wavelength=1.0,
            reference_row_indices=(0, 1, 2),
            channel_row_indices=(3, 4, 5),
            reference_times=(1.0, 2.0, 3.0),
            channel_times=(1.0, 2.0, 3.0),
            time_unit="day",
            method="exact_timestamp",
            pairing_source=(
                "pgmuvi_deterministic_time_matching"
            ),
            n_reference_observations=3,
            n_channel_observations=3,
        )
        calibration = InstrumentChannelCalibration(
            schema_version=(
                INSTRUMENT_CHANNEL_CALIBRATION_MODEL_SCHEMA_VERSION
            ),
            reference_channel="A",
            channel="B",
            wavelength=1.0,
            offset=0.5,
            scale=1.2,
            n_pairs=3,
            n_inliers=3,
            residual_mad_sigma=0.01,
        )
        channel_plan = self._planned(
            "B",
            pairing=pairing,
            calibration=calibration,
        )
        group = InstrumentChannelCalibrationGroupPlan(
            physical_wavelength=1.0,
            reference_channel="A",
            channel_plans=(channel_plan,),
        )

        payload = group.to_dict()
        self.assertEqual(
            payload["channel_plans"][0]["pairing"][
                "pairing_source"
            ],
            "pgmuvi_deterministic_time_matching",
        )
        self.assertEqual(
            payload["channel_plans"][0]["calibration"][
                "application_equation"
            ],
            "reference_flux = offset + scale * channel_flux",
        )
        self.assertFalse(
            payload["channel_plans"][0][
                "calibration_applied"
            ]
        )

    def test_fitted_calibration_requires_pairing_provenance(self):
        calibration = InstrumentChannelCalibration(
            schema_version=(
                INSTRUMENT_CHANNEL_CALIBRATION_MODEL_SCHEMA_VERSION
            ),
            reference_channel="A",
            channel="B",
            wavelength=1.0,
            offset=0.0,
            scale=1.0,
            n_pairs=3,
            n_inliers=3,
            residual_mad_sigma=0.0,
        )
        with self.assertRaisesRegex(
            ValueError,
            "requires pairing provenance",
        ):
            self._planned(
                "B",
                calibration=calibration,
            )

    def test_group_validates_nested_reference_and_wavelength(self):
        pairing = InstrumentChannelPairing(
            schema_version=(
                INSTRUMENT_CHANNEL_PAIRING_SCHEMA_VERSION
            ),
            reference_channel="A",
            channel="B",
            wavelength=1.0,
            reference_row_indices=(0, 1, 2),
            channel_row_indices=(3, 4, 5),
            reference_times=(1.0, 2.0, 3.0),
            channel_times=(1.0, 2.0, 3.0),
            time_unit="day",
            method="exact_timestamp",
        )
        with self.assertRaisesRegex(
            ValueError,
            "reference_channel",
        ):
            InstrumentChannelCalibrationGroupPlan(
                physical_wavelength=1.0,
                reference_channel="Z",
                channel_plans=(
                    self._planned("B", pairing=pairing),
                ),
            )

    def test_not_required_assessment_accepts_empty_plan(self):
        assessment = (
            assess_instrument_channel_calibration_requirement(
                [1.0, 2.0],
                ["A", "B"],
            )
        )
        plan = define_instrument_channel_calibration_plan(
            assessment,
            (),
        )
        self.assertEqual(plan.group_plans, ())
        self.assertEqual(
            plan.to_dict()["n_shared_wavelength_groups"],
            0,
        )

    def test_unknown_plan_schema_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported instrument-channel calibration plan",
        ):
            InstrumentChannelCalibrationPlan(
                schema_version=(
                    "pgmuvi-instrument-channel-calibration-plan-v0"
                ),
                assessment=self._assessment(),
                group_plans=self._groups(),
            )


if __name__ == "__main__":
    unittest.main()
