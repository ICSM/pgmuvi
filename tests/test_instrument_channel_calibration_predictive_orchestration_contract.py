"""Dataset predictive-uncertainty orchestration contract tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import unittest

import numpy as np

from pgmuvi.instrument_channel_calibration import (
    INSTRUMENT_CHANNEL_CALIBRATION_MODEL_SCHEMA_VERSION,
    INSTRUMENT_CHANNEL_CALIBRATION_PREDICTIVE_ORCHESTRATION_SCHEMA_VERSION,
    INSTRUMENT_CHANNEL_CALIBRATION_UNCERTAINTY_SCHEMA_VERSION,
    InstrumentChannelCalibration,
    InstrumentChannelCalibrationChannelPlan,
    InstrumentChannelCalibrationCoefficientUncertainty,
    InstrumentChannelCalibrationDisposition,
    InstrumentChannelCalibrationGroupPlan,
    InstrumentChannelCalibrationPredictiveCovarianceMode,
    InstrumentChannelCalibrationPredictiveUncertaintyChannelResult,
    InstrumentChannelCalibrationPredictiveUncertaintyDisposition,
    InstrumentChannelCalibrationPredictiveUncertaintyGroupResult,
    InstrumentChannelCalibrationPredictiveUncertaintyOrchestration,
    InstrumentChannelCalibrationPredictiveUncertaintyRequest,
    InstrumentChannelCalibrationUncertaintyStatus,
    InstrumentChannelPairingMethod,
    apply_instrument_channel_calibration_with_predictive_uncertainty,
    assess_instrument_channel_calibration_requirement,
    define_instrument_channel_calibration_plan,
    execute_instrument_channel_calibration_plan,
)


class TestPredictiveOrchestrationContract(unittest.TestCase):
    @staticmethod
    def _calibration(
        *,
        channel="B",
        covariance=((0.04, -0.006), (-0.006, 0.01)),
    ):
        uncertainty = InstrumentChannelCalibrationCoefficientUncertainty(
            schema_version=(
                INSTRUMENT_CHANNEL_CALIBRATION_UNCERTAINTY_SCHEMA_VERSION
            ),
            status=InstrumentChannelCalibrationUncertaintyStatus.AVAILABLE,
            uncertainty_source="caller_supplied_bootstrap",
            coefficient_covariance=covariance,
            estimation_method="paired_bootstrap",
        )
        return InstrumentChannelCalibration(
            schema_version=INSTRUMENT_CHANNEL_CALIBRATION_MODEL_SCHEMA_VERSION,
            reference_channel="A",
            channel=channel,
            wavelength=1.0,
            offset=0.25,
            scale=1.5,
            n_pairs=4,
            n_inliers=4,
            residual_mad_sigma=0.01,
            coefficient_uncertainty=uncertainty,
        )

    @classmethod
    def _available_result(
        cls,
        *,
        channel="B",
        source_row_indices=(3, 4),
        covariance_mode="marginal_variance",
        covariance=((0.04, -0.006), (-0.006, 0.01)),
    ):
        _, predictive = (
            apply_instrument_channel_calibration_with_predictive_uncertainty(
                np.asarray([1.0, 2.0]),
                cls._calibration(channel=channel, covariance=covariance),
                flux_error=np.asarray([0.1, 0.2]),
                covariance_mode=covariance_mode,
            )
        )
        return InstrumentChannelCalibrationPredictiveUncertaintyChannelResult(
            physical_wavelength=1.0,
            reference_channel="A",
            channel=channel,
            disposition=(
                InstrumentChannelCalibrationPredictiveUncertaintyDisposition
                .AVAILABLE
            ),
            source_row_indices=source_row_indices,
            predictive_uncertainty=predictive,
        )

    @staticmethod
    def _unavailable_result():
        uncertainty = InstrumentChannelCalibrationCoefficientUncertainty(
            schema_version=(
                INSTRUMENT_CHANNEL_CALIBRATION_UNCERTAINTY_SCHEMA_VERSION
            ),
            status=InstrumentChannelCalibrationUncertaintyStatus.UNAVAILABLE,
            uncertainty_source="pgmuvi_affine_fit",
            coefficient_covariance=None,
            reason="Coefficient covariance is unavailable.",
        )
        calibration = InstrumentChannelCalibration(
            schema_version=INSTRUMENT_CHANNEL_CALIBRATION_MODEL_SCHEMA_VERSION,
            reference_channel="A",
            channel="B",
            wavelength=1.0,
            offset=0.25,
            scale=1.5,
            n_pairs=4,
            n_inliers=4,
            residual_mad_sigma=0.01,
            coefficient_uncertainty=uncertainty,
        )
        _, predictive = (
            apply_instrument_channel_calibration_with_predictive_uncertainty(
                np.asarray([1.0, 2.0]),
                calibration,
                flux_error=np.asarray([0.1, 0.2]),
                covariance_mode="marginal_variance",
            )
        )
        return InstrumentChannelCalibrationPredictiveUncertaintyChannelResult(
            physical_wavelength=1.0,
            reference_channel="A",
            channel="B",
            disposition=(
                InstrumentChannelCalibrationPredictiveUncertaintyDisposition
                .UNAVAILABLE
            ),
            source_row_indices=(3, 4),
            predictive_uncertainty=predictive,
            reason=predictive.reason,
        )

    def test_request_is_explicit_immutable_and_json_safe(self):
        request = InstrumentChannelCalibrationPredictiveUncertaintyRequest(
            covariance_mode="marginal_variance"
        )
        payload = request.to_dict()
        json.dumps(payload, allow_nan=False)

        self.assertEqual(
            request.covariance_mode,
            InstrumentChannelCalibrationPredictiveCovarianceMode
            .MARGINAL_VARIANCE,
        )
        self.assertTrue(payload["requested"])
        self.assertFalse(payload["global_dense_covariance_requested"])
        with self.assertRaises(FrozenInstanceError):
            request.covariance_mode = "full_covariance"

    def test_request_rejects_boolean_and_unknown_modes(self):
        with self.assertRaisesRegex(TypeError, "not boolean"):
            InstrumentChannelCalibrationPredictiveUncertaintyRequest(
                covariance_mode=True
            )
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            InstrumentChannelCalibrationPredictiveUncertaintyRequest(
                covariance_mode="global_dense"
            )

    def test_available_channel_result_preserves_row_identity(self):
        result = self._available_result(source_row_indices=(40, 20))
        payload = result.to_dict()
        json.dumps(payload, allow_nan=False)

        self.assertEqual(result.source_row_indices, (40, 20))
        self.assertEqual(payload["source_row_indices"], [40, 20])
        self.assertEqual(
            payload["predictive_uncertainty"]["coefficient_order"],
            ["offset", "scale"],
        )
        self.assertIsNotNone(
            payload["predictive_uncertainty"]["predictive_standard_deviation"]
        )

    def test_unavailable_channel_has_no_measurement_only_fallback(self):
        result = self._unavailable_result()
        payload = result.to_dict()
        json.dumps(payload, allow_nan=False)

        predictive = payload["predictive_uncertainty"]
        self.assertEqual(result.disposition.value, "unavailable")
        self.assertIsNone(predictive["measurement_variance"])
        self.assertIsNone(predictive["predictive_variance"])
        self.assertIsNone(predictive["predictive_covariance"])

    def test_not_requested_and_skipped_states_are_non_numeric(self):
        not_requested = (
            InstrumentChannelCalibrationPredictiveUncertaintyChannelResult(
                physical_wavelength=1.0,
                reference_channel="A",
                channel="B",
                disposition="not_requested",
                source_row_indices=(3, 4),
            )
        )
        skipped = InstrumentChannelCalibrationPredictiveUncertaintyChannelResult(
            physical_wavelength=1.0,
            reference_channel="A",
            channel="C",
            disposition="skipped",
            source_row_indices=(5, 6),
            reason="Caller excluded channel C.",
        )

        self.assertIsNone(not_requested.predictive_uncertainty)
        self.assertIsNone(skipped.predictive_uncertainty)
        with self.assertRaisesRegex(ValueError, "requires a reason"):
            InstrumentChannelCalibrationPredictiveUncertaintyChannelResult(
                physical_wavelength=1.0,
                reference_channel="A",
                channel="C",
                disposition="skipped",
                source_row_indices=(5,),
            )

    def test_reference_rows_are_explicitly_not_applicable(self):
        group = InstrumentChannelCalibrationPredictiveUncertaintyGroupResult(
            physical_wavelength=1.0,
            reference_channel="A",
            reference_source_row_indices=(0, 1, 2),
            channel_results=(
                InstrumentChannelCalibrationPredictiveUncertaintyChannelResult(
                    physical_wavelength=1.0,
                    reference_channel="A",
                    channel="B",
                    disposition="not_requested",
                    source_row_indices=(3, 4),
                ),
            ),
        )
        payload = group.to_dict()

        self.assertEqual(payload["reference_source_row_indices"], [0, 1, 2])
        self.assertEqual(
            payload[
                "reference_channel_fitted_coefficient_uncertainty_disposition"
            ],
            "not_applicable",
        )

    def test_not_requested_dataset_summary_is_explicit(self):
        group = InstrumentChannelCalibrationPredictiveUncertaintyGroupResult(
            physical_wavelength=1.0,
            reference_channel="A",
            reference_source_row_indices=(0, 1),
            channel_results=(
                InstrumentChannelCalibrationPredictiveUncertaintyChannelResult(
                    physical_wavelength=1.0,
                    reference_channel="A",
                    channel="B",
                    disposition="not_requested",
                    source_row_indices=(2, 3),
                ),
                InstrumentChannelCalibrationPredictiveUncertaintyChannelResult(
                    physical_wavelength=1.0,
                    reference_channel="A",
                    channel="C",
                    disposition="skipped",
                    source_row_indices=(4,),
                    reason="Caller excluded channel C.",
                ),
            ),
        )
        result = InstrumentChannelCalibrationPredictiveUncertaintyOrchestration(
            schema_version=(
                INSTRUMENT_CHANNEL_CALIBRATION_PREDICTIVE_ORCHESTRATION_SCHEMA_VERSION
            ),
            covariance_mode=None,
            source_row_indices=(0, 1, 2, 3, 4, 9),
            group_results=(group,),
            unaffected_source_row_indices=(9,),
            ordinary_calibrated_flux_error_available=True,
        )
        payload = result.to_dict()
        json.dumps(payload, allow_nan=False)

        self.assertFalse(payload["requested"])
        self.assertEqual(payload["n_not_requested_channels"], 1)
        self.assertEqual(payload["n_skipped_channels"], 1)
        self.assertFalse(payload["fitted_coefficient_uncertainty_propagated"])
        self.assertFalse(payload["global_dense_predictive_covariance_emitted"])
        self.assertFalse(payload["missing_values_represented_by_nan"])

    def test_requested_summary_reports_partial_success(self):
        group = InstrumentChannelCalibrationPredictiveUncertaintyGroupResult(
            physical_wavelength=1.0,
            reference_channel="A",
            reference_source_row_indices=(0, 1, 2),
            channel_results=(
                self._available_result(),
                InstrumentChannelCalibrationPredictiveUncertaintyChannelResult(
                    physical_wavelength=1.0,
                    reference_channel="A",
                    channel="C",
                    disposition="skipped",
                    source_row_indices=(5, 6),
                    reason="Caller excluded channel C.",
                ),
            ),
        )
        result = InstrumentChannelCalibrationPredictiveUncertaintyOrchestration(
            schema_version=(
                INSTRUMENT_CHANNEL_CALIBRATION_PREDICTIVE_ORCHESTRATION_SCHEMA_VERSION
            ),
            covariance_mode="marginal_variance",
            source_row_indices=(0, 1, 2, 3, 4, 5, 6, 9),
            group_results=(group,),
            unaffected_source_row_indices=(9,),
            ordinary_calibrated_flux_error_available=True,
        )
        payload = result.to_dict()

        self.assertTrue(payload["requested"])
        self.assertTrue(payload["any_predictive_uncertainty_available"])
        self.assertTrue(payload["all_eligible_channels_available"])
        self.assertTrue(payload["fitted_coefficient_uncertainty_propagated"])
        self.assertTrue(
            payload["predictive_standard_deviation_blocks_available"]
        )
        self.assertFalse(payload["predictive_covariance_blocks_available"])

    def test_full_covariance_is_channel_local_at_shared_wavelength(self):
        first = self._available_result(
            channel="B",
            source_row_indices=(3, 4),
            covariance_mode="full_covariance",
            covariance=((0.04, -0.006), (-0.006, 0.01)),
        )
        second = self._available_result(
            channel="C",
            source_row_indices=(5, 6),
            covariance_mode="full_covariance",
            covariance=((0.09, 0.002), (0.002, 0.02)),
        )
        group = InstrumentChannelCalibrationPredictiveUncertaintyGroupResult(
            physical_wavelength=1.0,
            reference_channel="A",
            reference_source_row_indices=(0, 1, 2),
            channel_results=(first, second),
        )
        result = InstrumentChannelCalibrationPredictiveUncertaintyOrchestration(
            schema_version=(
                INSTRUMENT_CHANNEL_CALIBRATION_PREDICTIVE_ORCHESTRATION_SCHEMA_VERSION
            ),
            covariance_mode="full_covariance",
            source_row_indices=(0, 1, 2, 3, 4, 5, 6),
            group_results=(group,),
            unaffected_source_row_indices=(),
            ordinary_calibrated_flux_error_available=True,
        )
        payload = result.to_dict()
        serialized = payload["group_results"][0]["channel_results"]

        self.assertEqual([item["channel"] for item in serialized], ["B", "C"])
        self.assertNotEqual(
            serialized[0]["predictive_uncertainty"]["predictive_covariance"],
            serialized[1]["predictive_uncertainty"]["predictive_covariance"],
        )
        self.assertTrue(payload["predictive_covariance_blocks_available"])
        self.assertFalse(payload["global_dense_predictive_covariance_emitted"])
        self.assertFalse(
            payload["cross_observational_channel_covariance_emitted"]
        )
        self.assertFalse(
            payload["cross_observational_channel_covariance_assumed_zero"]
        )
        self.assertFalse(
            payload["group_results"][0][
                "physical_wavelength_used_as_covariance_identity"
            ]
        )

    def test_dataset_rows_must_form_exact_disjoint_partition(self):
        group = InstrumentChannelCalibrationPredictiveUncertaintyGroupResult(
            physical_wavelength=1.0,
            reference_channel="A",
            reference_source_row_indices=(0, 1),
            channel_results=(
                InstrumentChannelCalibrationPredictiveUncertaintyChannelResult(
                    physical_wavelength=1.0,
                    reference_channel="A",
                    channel="B",
                    disposition="not_requested",
                    source_row_indices=(2, 3),
                ),
            ),
        )
        common = {
            "schema_version": (
                INSTRUMENT_CHANNEL_CALIBRATION_PREDICTIVE_ORCHESTRATION_SCHEMA_VERSION
            ),
            "covariance_mode": None,
            "source_row_indices": (0, 1, 2, 3, 9),
            "group_results": (group,),
            "ordinary_calibrated_flux_error_available": True,
        }
        with self.assertRaisesRegex(ValueError, "partition"):
            InstrumentChannelCalibrationPredictiveUncertaintyOrchestration(
                **common,
                unaffected_source_row_indices=(),
            )
        with self.assertRaisesRegex(ValueError, "disjoint"):
            InstrumentChannelCalibrationPredictiveUncertaintyOrchestration(
                **common,
                unaffected_source_row_indices=(3, 9),
            )
        reordered = dict(common)
        reordered["source_row_indices"] = (0, 1, 3, 2, 9)
        with self.assertRaisesRegex(ValueError, "preserve original"):
            InstrumentChannelCalibrationPredictiveUncertaintyOrchestration(
                **reordered,
                unaffected_source_row_indices=(9,),
            )

    def test_manual_calibration_with_covariance_needs_no_fit_provenance(self):
        calibration = self._calibration()
        self.assertIsNone(calibration.fit_provenance)
        _, predictive = (
            apply_instrument_channel_calibration_with_predictive_uncertainty(
                np.asarray([1.0, 2.0]),
                calibration,
                covariance_mode="marginal_variance",
            )
        )
        channel_result = (
            InstrumentChannelCalibrationPredictiveUncertaintyChannelResult(
                physical_wavelength=1.0,
                reference_channel="A",
                channel="B",
                disposition="available",
                source_row_indices=(3, 4),
                predictive_uncertainty=predictive,
            )
        )
        self.assertEqual(channel_result.disposition.value, "available")


class TestPredictiveOrchestrationExecutionBoundary(unittest.TestCase):
    @staticmethod
    def _data_and_plan():
        times = np.asarray([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
        wavelengths = np.ones(6)
        channels = np.asarray(["A", "A", "A", "B", "B", "B"])
        flux = np.asarray([3.0, 5.0, 7.0, 1.0, 2.0, 3.0])
        error = np.asarray([0.3, 0.3, 0.3, 0.1, 0.1, 0.1])
        assessment = assess_instrument_channel_calibration_requirement(
            wavelengths,
            channels,
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
                            disposition=(
                                InstrumentChannelCalibrationDisposition.PLANNED
                            ),
                            pairing_method=(
                                InstrumentChannelPairingMethod.EXACT_TIMESTAMP
                            ),
                            time_unit="day",
                            calibration_family="affine",
                        ),
                    ),
                ),
            ),
        )
        return plan, times, flux, wavelengths, channels, error

    def test_default_execution_remains_backward_compatible(self):
        plan, times, flux, wavelengths, channels, error = self._data_and_plan()
        result = execute_instrument_channel_calibration_plan(
            plan,
            times,
            flux,
            wavelengths,
            channels,
            flux_error=error,
        )
        payload = result.to_dict()

        self.assertFalse(payload["fitted_coefficient_uncertainty_propagated"])
        self.assertNotIn("predictive_uncertainty", payload)

    def test_opt_in_request_returns_predictive_orchestration(self):
        plan, times, flux, wavelengths, channels, error = self._data_and_plan()
        request = InstrumentChannelCalibrationPredictiveUncertaintyRequest(
            covariance_mode="marginal_variance"
        )
        result = execute_instrument_channel_calibration_plan(
            plan,
            times,
            flux,
            wavelengths,
            channels,
            flux_error=error,
            predictive_uncertainty_request=request,
        )
        payload = result.to_dict()

        self.assertIsNotNone(result.predictive_uncertainty)
        self.assertTrue(payload["fitted_coefficient_uncertainty_propagated"])
        self.assertTrue(payload["predictive_uncertainty"]["requested"])
        self.assertEqual(
            payload["predictive_uncertainty"]["n_available_channels"],
            1,
        )

    def test_execution_request_input_is_strict(self):
        plan, times, flux, wavelengths, channels, error = self._data_and_plan()
        for value in (True, "marginal_variance"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "request"):
                    execute_instrument_channel_calibration_plan(
                        plan,
                        times,
                        flux,
                        wavelengths,
                        channels,
                        flux_error=error,
                        predictive_uncertainty_request=value,
                    )


if __name__ == "__main__":
    unittest.main()
