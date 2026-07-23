"""Dataset predictive-uncertainty orchestration execution tests."""

from __future__ import annotations

import json
import unittest

import numpy as np

from pgmuvi.instrument_channel_calibration import (
    InstrumentChannelCalibrationChannelPlan,
    InstrumentChannelCalibrationDisposition,
    InstrumentChannelCalibrationGroupPlan,
    InstrumentChannelCalibrationPredictiveUncertaintyRequest,
    InstrumentChannelPairingMethod,
    assess_instrument_channel_calibration_requirement,
    define_instrument_channel_calibration_plan,
    execute_instrument_channel_calibration_plan,
)


class TestPredictiveOrchestrationExecution(unittest.TestCase):
    @staticmethod
    def _data(*, include_unique=True):
        times = np.tile(np.arange(3, dtype=float), 3)
        wavelengths = np.ones(9, dtype=float)
        channels = np.repeat(
            np.asarray(["A", "B", "C"], dtype=object),
            3,
        )
        flux = np.concatenate(
            (
                np.asarray([3.0, 5.0, 7.0]),
                np.asarray([1.0, 2.0, 3.0]),
                np.asarray([0.5, 1.5, 2.5]),
            )
        )
        error = np.concatenate(
            (
                np.full(3, 0.3),
                np.full(3, 0.1),
                np.full(3, 0.2),
            )
        )
        if include_unique:
            times = np.concatenate((times, np.asarray([0.0, 1.0])))
            wavelengths = np.concatenate(
                (wavelengths, np.asarray([2.0, 2.0]))
            )
            channels = np.concatenate(
                (
                    channels,
                    np.asarray(["D", "D"], dtype=object),
                )
            )
            flux = np.concatenate((flux, np.asarray([4.0, 5.0])))
            error = np.concatenate((error, np.asarray([0.4, 0.4])))
        return times, flux, error, wavelengths, channels

    @classmethod
    def _plan(
        cls,
        *,
        include_unique=True,
        c_disposition=InstrumentChannelCalibrationDisposition.SKIPPED,
    ):
        _, _, _, wavelengths, channels = cls._data(
            include_unique=include_unique
        )
        assessment = assess_instrument_channel_calibration_requirement(
            wavelengths,
            channels,
        )
        channel_plans = [
            InstrumentChannelCalibrationChannelPlan(
                channel="B",
                disposition=InstrumentChannelCalibrationDisposition.PLANNED,
                pairing_method=InstrumentChannelPairingMethod.EXACT_TIMESTAMP,
                time_unit="day",
                calibration_family="affine",
            ),
        ]
        if c_disposition is InstrumentChannelCalibrationDisposition.PLANNED:
            channel_plans.append(
                InstrumentChannelCalibrationChannelPlan(
                    channel="C",
                    disposition=(
                        InstrumentChannelCalibrationDisposition.PLANNED
                    ),
                    pairing_method=(
                        InstrumentChannelPairingMethod.EXACT_TIMESTAMP
                    ),
                    time_unit="day",
                    calibration_family="affine",
                )
            )
        else:
            channel_plans.append(
                InstrumentChannelCalibrationChannelPlan(
                    channel="C",
                    disposition=c_disposition,
                    reason=(
                        "Caller excluded channel C."
                        if c_disposition
                        is InstrumentChannelCalibrationDisposition.SKIPPED
                        else "Calibration is unavailable for channel C."
                    ),
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

    def test_marginal_request_returns_aligned_channel_blocks(self):
        times, flux, error, wavelengths, channels = self._data()
        source_rows = tuple(range(10, 120, 10))

        result = execute_instrument_channel_calibration_plan(
            self._plan(),
            times,
            flux,
            wavelengths,
            channels,
            flux_error=error,
            source_row_indices=source_rows,
            predictive_uncertainty_request=(
                InstrumentChannelCalibrationPredictiveUncertaintyRequest(
                    covariance_mode="marginal_variance"
                )
            ),
        )
        payload = result.to_dict()
        json.dumps(payload, allow_nan=False)

        predictive = result.predictive_uncertainty
        self.assertIsNotNone(predictive)
        self.assertEqual(predictive.n_available_channels, 1)
        self.assertEqual(predictive.n_skipped_channels, 1)
        self.assertEqual(
            predictive.unaffected_source_row_indices,
            (100, 110),
        )

        group = predictive.group_results[0]
        self.assertEqual(
            group.reference_source_row_indices,
            (10, 20, 30),
        )
        available, skipped = group.channel_results
        self.assertEqual(available.channel, "B")
        self.assertEqual(available.source_row_indices, (40, 50, 60))
        self.assertEqual(available.disposition.value, "available")
        self.assertEqual(skipped.channel, "C")
        self.assertEqual(skipped.disposition.value, "skipped")
        self.assertEqual(skipped.source_row_indices, (70, 80, 90))

        ordinary = np.asarray(result.calibrated_flux_error)[3:6]
        predictive_std = np.asarray(
            available.predictive_uncertainty
            .predictive_standard_deviation
        )
        self.assertTrue(np.all(predictive_std >= ordinary))
        self.assertIn("predictive_uncertainty", payload)
        self.assertTrue(payload["fitted_coefficient_uncertainty_propagated"])

    def test_full_covariance_remains_separate_for_same_wavelength_channels(self):
        times, flux, error, wavelengths, channels = self._data(
            include_unique=False
        )
        result = execute_instrument_channel_calibration_plan(
            self._plan(
                include_unique=False,
                c_disposition=(
                    InstrumentChannelCalibrationDisposition.PLANNED
                ),
            ),
            times,
            flux,
            wavelengths,
            channels,
            flux_error=error,
            predictive_uncertainty_request=(
                InstrumentChannelCalibrationPredictiveUncertaintyRequest(
                    covariance_mode="full_covariance"
                )
            ),
        )

        predictive = result.predictive_uncertainty
        group = predictive.group_results[0]
        self.assertEqual(
            tuple(item.channel for item in group.channel_results),
            ("B", "C"),
        )
        for item in group.channel_results:
            covariance = np.asarray(
                item.predictive_uncertainty.predictive_covariance
            )
            self.assertEqual(covariance.shape, (3, 3))

        payload = predictive.to_dict()
        self.assertTrue(payload["predictive_covariance_blocks_available"])
        self.assertFalse(payload["global_dense_predictive_covariance_emitted"])
        self.assertFalse(
            payload["cross_observational_channel_covariance_emitted"]
        )
        self.assertFalse(
            payload["cross_observational_channel_covariance_assumed_zero"]
        )

    def test_unavailable_plan_channel_has_structural_unavailable_result(self):
        times, flux, error, wavelengths, channels = self._data(
            include_unique=False
        )
        result = execute_instrument_channel_calibration_plan(
            self._plan(
                include_unique=False,
                c_disposition=(
                    InstrumentChannelCalibrationDisposition.UNAVAILABLE
                ),
            ),
            times,
            flux,
            wavelengths,
            channels,
            flux_error=error,
            predictive_uncertainty_request=(
                InstrumentChannelCalibrationPredictiveUncertaintyRequest(
                    covariance_mode="marginal_variance"
                )
            ),
        )

        channel_results = (
            result.predictive_uncertainty.group_results[0].channel_results
        )
        unavailable = next(
            item for item in channel_results if item.channel == "C"
        )
        nested = unavailable.predictive_uncertainty

        self.assertEqual(unavailable.disposition.value, "unavailable")
        self.assertIsNone(nested.measurement_variance)
        self.assertIsNone(nested.predictive_variance)
        self.assertIsNone(nested.predictive_covariance)
        np.testing.assert_allclose(result.calibrated_flux[6:9], flux[6:9])

    def test_request_without_measurement_errors_records_zero_contribution(self):
        times, flux, _, wavelengths, channels = self._data(
            include_unique=False
        )
        result = execute_instrument_channel_calibration_plan(
            self._plan(),
            times,
            flux,
            wavelengths,
            channels,
            predictive_uncertainty_request=(
                InstrumentChannelCalibrationPredictiveUncertaintyRequest(
                    covariance_mode="marginal_variance"
                )
            ),
        )

        self.assertIsNone(result.calibrated_flux_error)
        predictive = (
            result.predictive_uncertainty.group_results[0]
            .channel_results[0].predictive_uncertainty
        )
        self.assertFalse(
            predictive.input_measurement_uncertainty_supplied
        )
        np.testing.assert_allclose(
            predictive.measurement_variance,
            np.zeros(3),
        )
        self.assertFalse(
            result.predictive_uncertainty
            .ordinary_calibrated_flux_error_available
        )

    def test_request_on_dataset_without_shared_wavelengths_is_explicit(self):
        times = np.asarray([0.0, 1.0, 2.0])
        flux = np.asarray([1.0, 2.0, 3.0])
        wavelengths = np.asarray([1.0, 2.0, 3.0])
        channels = np.asarray(["A", "B", "C"], dtype=object)
        assessment = assess_instrument_channel_calibration_requirement(
            wavelengths,
            channels,
        )
        plan = define_instrument_channel_calibration_plan(assessment, ())

        result = execute_instrument_channel_calibration_plan(
            plan,
            times,
            flux,
            wavelengths,
            channels,
            source_row_indices=(10, 20, 30),
            predictive_uncertainty_request=(
                InstrumentChannelCalibrationPredictiveUncertaintyRequest(
                    covariance_mode="full_covariance"
                )
            ),
        )

        predictive = result.predictive_uncertainty
        self.assertTrue(predictive.requested)
        self.assertEqual(predictive.group_results, ())
        self.assertEqual(
            predictive.unaffected_source_row_indices,
            (10, 20, 30),
        )
        self.assertFalse(
            predictive.fitted_coefficient_uncertainty_propagated
        )

    def test_default_execution_payload_remains_unchanged(self):
        times, flux, error, wavelengths, channels = self._data(
            include_unique=False
        )
        result = execute_instrument_channel_calibration_plan(
            self._plan(),
            times,
            flux,
            wavelengths,
            channels,
            flux_error=error,
        )
        payload = result.to_dict()

        self.assertIsNone(result.predictive_uncertainty)
        self.assertNotIn("predictive_uncertainty", payload)
        self.assertFalse(payload["fitted_coefficient_uncertainty_propagated"])


if __name__ == "__main__":
    unittest.main()
