"""Dataset-level instrument-channel calibration execution tests."""

from __future__ import annotations

from dataclasses import replace
import json
import unittest

import numpy as np

from pgmuvi.instrument_channel_calibration import (
    INSTRUMENT_CHANNEL_CALIBRATION_EXECUTION_SCHEMA_VERSION,
    InstrumentChannelCalibrationChannelPlan,
    InstrumentChannelCalibrationDisposition,
    InstrumentChannelCalibrationExecution,
    InstrumentChannelCalibrationGroupPlan,
    InstrumentChannelPairing,
    InstrumentChannelPairingMethod,
    assess_instrument_channel_calibration_requirement,
    define_instrument_channel_calibration_plan,
    execute_instrument_channel_calibration_plan,
)


class TestInstrumentChannelCalibrationOrchestrationExecution(
    unittest.TestCase
):
    @staticmethod
    def _data(include_skipped=False):
        times = np.asarray(
            [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            dtype=float,
        )
        wavelengths = np.ones(6, dtype=float)
        channels = np.asarray(
            ["A", "A", "A", "B", "B", "B"],
            dtype=object,
        )
        flux = np.asarray(
            [3.0, 5.0, 7.0, 1.0, 2.0, 3.0],
            dtype=float,
        )
        error = np.asarray(
            [0.3, 0.3, 0.3, 0.1, 0.1, 0.1],
            dtype=float,
        )

        if include_skipped:
            times = np.concatenate(
                (times, np.asarray([0.0, 1.0, 2.0]))
            )
            wavelengths = np.concatenate(
                (wavelengths, np.ones(3))
            )
            channels = np.concatenate(
                (
                    channels,
                    np.asarray(["C", "C", "C"], dtype=object),
                )
            )
            flux = np.concatenate(
                (flux, np.asarray([9.0, 8.0, 7.0]))
            )
            error = np.concatenate(
                (error, np.asarray([0.4, 0.4, 0.4]))
            )

        return times, flux, error, wavelengths, channels

    @classmethod
    def _plan(cls, *, include_skipped=False, pairing=None):
        (
            _,
            _,
            _,
            wavelengths,
            channels,
        ) = cls._data(include_skipped=include_skipped)

        assessment = (
            assess_instrument_channel_calibration_requirement(
                wavelengths,
                channels,
            )
        )

        channel_plans = [
            InstrumentChannelCalibrationChannelPlan(
                channel="B",
                disposition=(
                    InstrumentChannelCalibrationDisposition.PLANNED
                ),
                pairing_method=(
                    InstrumentChannelPairingMethod
                    .EXACT_TIMESTAMP
                ),
                time_unit="day",
                calibration_family="affine",
                pairing=pairing,
            )
        ]

        if include_skipped:
            channel_plans.append(
                InstrumentChannelCalibrationChannelPlan(
                    channel="C",
                    disposition=(
                        InstrumentChannelCalibrationDisposition
                        .SKIPPED
                    ),
                    reason="Caller excluded channel C.",
                )
            )

        return define_instrument_channel_calibration_plan(
            assessment,
            (
                InstrumentChannelCalibrationGroupPlan(
                    physical_wavelength=1.0,
                    reference_channel="A",
                    channel_plans=tuple(channel_plans),
                ),
            ),
        )

    def test_execution_fits_applies_and_records_provenance(self):
        times, flux, error, wavelengths, channels = self._data()
        original_flux = flux.copy()
        original_error = error.copy()

        result = execute_instrument_channel_calibration_plan(
            self._plan(),
            times,
            flux,
            wavelengths,
            channels,
            flux_error=error,
            source_row_indices=(10, 20, 30, 40, 50, 60),
        )

        self.assertIsInstance(
            result,
            InstrumentChannelCalibrationExecution,
        )
        self.assertEqual(
            result.schema_version,
            INSTRUMENT_CHANNEL_CALIBRATION_EXECUTION_SCHEMA_VERSION,
        )
        completed = result.plan.group_plans[0].channel_plans[0]
        self.assertIsNotNone(completed.pairing)
        self.assertIsNotNone(completed.calibration)
        calibration = completed.calibration

        expected_flux = original_flux.copy()
        expected_flux[3:] = (
            calibration.offset
            + calibration.scale * original_flux[3:]
        )
        expected_error = original_error.copy()
        expected_error[3:] = (
            abs(calibration.scale) * original_error[3:]
        )
        np.testing.assert_allclose(
            result.calibrated_flux,
            expected_flux,
        )
        np.testing.assert_allclose(
            result.calibrated_flux_error,
            expected_error,
        )
        np.testing.assert_array_equal(flux, original_flux)
        np.testing.assert_array_equal(error, original_error)

        self.assertEqual(
            completed.pairing.reference_row_indices,
            (10, 20, 30),
        )
        self.assertEqual(
            completed.pairing.channel_row_indices,
            (40, 50, 60),
        )
        self.assertEqual(
            calibration.coefficient_uncertainty.status.value,
            "available",
        )
        self.assertEqual(
            calibration.coefficient_uncertainty.estimation_method,
            "inverse_observed_hessian_scale_dependent_full_objective",
        )
        self.assertEqual(
            calibration.fit_provenance.integration_status.value,
            "active",
        )
        self.assertEqual(result.applied_channels, ("B",))
        self.assertEqual(
            result.applied_source_row_indices,
            (40, 50, 60),
        )
        self.assertEqual(result.n_applied_observations, 3)

        payload = result.to_dict()
        json.dumps(payload, allow_nan=False)
        self.assertFalse(
            payload["automatic_reference_channel_selection"]
        )
        self.assertFalse(payload["input_mutation_performed"])
        self.assertEqual(
            payload["applied_source_row_indices"],
            [40, 50, 60],
        )
        self.assertFalse(
            payload["fitted_coefficient_uncertainty_propagated"]
        )

    def test_execution_result_rejects_inconsistent_applied_count(self):
        times, flux, error, wavelengths, channels = self._data()

        result = execute_instrument_channel_calibration_plan(
            self._plan(),
            times,
            flux,
            wavelengths,
            channels,
            flux_error=error,
        )

        with self.assertRaisesRegex(
            ValueError,
            "must equal the number",
        ):
            replace(
                result,
                applied_source_row_indices=(3,),
                n_applied_observations=3,
            )

    def test_completed_provenance_can_be_reused(self):
        times, flux, error, wavelengths, channels = self._data()

        first = execute_instrument_channel_calibration_plan(
            self._plan(),
            times,
            flux,
            wavelengths,
            channels,
            flux_error=error,
        )
        second = execute_instrument_channel_calibration_plan(
            first.plan,
            times,
            flux,
            wavelengths,
            channels,
            flux_error=error,
        )

        self.assertEqual(
            first.plan.group_plans[0].channel_plans[0].pairing,
            second.plan.group_plans[0].channel_plans[0].pairing,
        )
        self.assertEqual(
            first.plan.group_plans[0].channel_plans[0].calibration,
            second.plan.group_plans[0].channel_plans[0].calibration,
        )
        np.testing.assert_allclose(
            first.calibrated_flux,
            second.calibrated_flux,
        )

    def test_skipped_channel_is_preserved_unchanged(self):
        times, flux, error, wavelengths, channels = self._data(
            include_skipped=True
        )

        result = execute_instrument_channel_calibration_plan(
            self._plan(include_skipped=True),
            times,
            flux,
            wavelengths,
            channels,
            flux_error=error,
        )

        np.testing.assert_allclose(
            result.calibrated_flux[6:],
            flux[6:],
        )
        skipped = result.plan.group_plans[0].channel_plans[1]
        self.assertEqual(
            skipped.disposition,
            InstrumentChannelCalibrationDisposition.SKIPPED,
        )
        self.assertIsNone(skipped.pairing)
        self.assertIsNone(skipped.calibration)

    def test_execution_without_errors_returns_no_error_vector(self):
        times, flux, _, wavelengths, channels = self._data()

        result = execute_instrument_channel_calibration_plan(
            self._plan(),
            times,
            flux,
            wavelengths,
            channels,
        )

        self.assertIsNone(result.calibrated_flux_error)
        np.testing.assert_allclose(
            result.calibrated_flux,
            [3.0, 5.0, 7.0, 3.0, 5.0, 7.0],
        )

    def test_execution_rejects_booleans_in_object_vectors(self):
        times, flux, error, wavelengths, channels = self._data()

        cases = (
            (
                "flux",
                {
                    "flux": np.asarray(
                        [3.0, 5.0, 7.0, 1.0, True, 3.0],
                        dtype=object,
                    ),
                },
            ),
            (
                "flux_error",
                {
                    "flux_error": np.asarray(
                        [0.3, 0.3, 0.3, 0.1, False, 0.1],
                        dtype=object,
                    ),
                },
            ),
            (
                "physical_wavelengths",
                {
                    "physical_wavelengths": np.asarray(
                        [1.0, 1.0, 1.0, 1.0, True, 1.0],
                        dtype=object,
                    ),
                },
            ),
        )

        for expected_name, overrides in cases:
            arguments = {
                "plan": self._plan(),
                "times": times,
                "flux": flux,
                "physical_wavelengths": wavelengths,
                "observational_channel_labels": channels,
                "flux_error": error,
            }
            arguments.update(overrides)

            with self.subTest(name=expected_name):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"{expected_name} must contain numeric values, "
                    "not booleans",
                ):
                    execute_instrument_channel_calibration_plan(
                        **arguments
                    )

    def test_execution_rejects_assessment_mismatch(self):
        times, flux, error, wavelengths, channels = self._data()
        channels = channels.copy()
        channels[-1] = "D"

        with self.assertRaisesRegex(
            ValueError,
            "do not reproduce",
        ):
            execute_instrument_channel_calibration_plan(
                self._plan(),
                times,
                flux,
                wavelengths,
                channels,
                flux_error=error,
            )

    def test_execution_rejects_missing_pairing_source_rows(self):
        pairing = InstrumentChannelPairing(
            schema_version=(
                "pgmuvi-instrument-channel-pairing-v2"
            ),
            reference_channel="A",
            channel="B",
            wavelength=1.0,
            reference_row_indices=(100, 101, 102),
            channel_row_indices=(200, 201, 202),
            reference_times=(0.0, 1.0, 2.0),
            channel_times=(0.0, 1.0, 2.0),
            time_unit="day",
            method="exact_timestamp",
        )
        times, flux, error, wavelengths, channels = self._data()

        with self.assertRaisesRegex(
            ValueError,
            "absent from the execution input",
        ):
            execute_instrument_channel_calibration_plan(
                self._plan(pairing=pairing),
                times,
                flux,
                wavelengths,
                channels,
                flux_error=error,
            )

    def test_execution_rejects_stale_pairing_times(self):
        pairing = InstrumentChannelPairing(
            schema_version=(
                "pgmuvi-instrument-channel-pairing-v2"
            ),
            reference_channel="A",
            channel="B",
            wavelength=1.0,
            reference_row_indices=(0, 1, 2),
            channel_row_indices=(3, 4, 5),
            reference_times=(0.0, 1.0, 99.0),
            channel_times=(0.0, 1.0, 99.0),
            time_unit="day",
            method="exact_timestamp",
        )
        times, flux, error, wavelengths, channels = self._data()

        with self.assertRaisesRegex(
            ValueError,
            "reference_times do not match",
        ):
            execute_instrument_channel_calibration_plan(
                self._plan(pairing=pairing),
                times,
                flux,
                wavelengths,
                channels,
                flux_error=error,
            )

    def test_execution_rejects_pairing_with_too_few_pairs(self):
        times = np.asarray([0.0, 1.0, 0.0, 1.0])
        wavelengths = np.ones(4)
        channels = np.asarray(["A", "A", "B", "B"])
        flux = np.asarray([3.0, 5.0, 1.0, 2.0])

        assessment = (
            assess_instrument_channel_calibration_requirement(
                wavelengths,
                channels,
            )
        )
        plan = define_instrument_channel_calibration_plan(
            assessment,
            (
                InstrumentChannelCalibrationGroupPlan(
                    physical_wavelength=1.0,
                    reference_channel="A",
                    channel_plans=(
                        InstrumentChannelCalibrationChannelPlan(
                            channel="B",
                            disposition="planned",
                            pairing_method="exact_timestamp",
                            time_unit="day",
                            calibration_family="affine",
                        ),
                    ),
                ),
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "at least three paired observations",
        ):
            execute_instrument_channel_calibration_plan(
                plan,
                times,
                flux,
                wavelengths,
                channels,
            )


if __name__ == "__main__":
    unittest.main()
