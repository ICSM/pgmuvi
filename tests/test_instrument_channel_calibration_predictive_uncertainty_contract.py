"""Predictive calibration-uncertainty propagation contract tests."""

import json
import unittest

import numpy as np

from pgmuvi.instrument_channel_calibration import (
    INSTRUMENT_CHANNEL_CALIBRATION_MODEL_SCHEMA_VERSION,
    INSTRUMENT_CHANNEL_CALIBRATION_PREDICTIVE_UNCERTAINTY_SCHEMA_VERSION,
    INSTRUMENT_CHANNEL_CALIBRATION_UNCERTAINTY_SCHEMA_VERSION,
    InstrumentChannelCalibration,
    InstrumentChannelCalibrationCoefficientUncertainty,
    InstrumentChannelCalibrationPredictiveCovarianceMode,
    InstrumentChannelCalibrationPredictiveUncertainty,
    InstrumentChannelCalibrationPredictiveUncertaintyDisposition,
    InstrumentChannelCalibrationUncertaintyStatus,
    apply_instrument_channel_calibration,
    apply_instrument_channel_calibration_with_predictive_uncertainty,
)


class TestPredictiveUncertaintyContract(unittest.TestCase):
    @staticmethod
    def _available(**overrides):
        values = {
            "schema_version": (
                INSTRUMENT_CHANNEL_CALIBRATION_PREDICTIVE_UNCERTAINTY_SCHEMA_VERSION
            ),
            "status": "available",
            "covariance_mode": "full_covariance",
            "input_shape": (2,),
            "input_measurement_uncertainty_supplied": True,
            "coefficient_uncertainty_status": "available",
            "coefficient_uncertainty_source": "caller_supplied_bootstrap",
            "measurement_variance": (0.0225, 0.09),
            "offset_variance": (0.04, 0.04),
            "scale_variance": (0.01, 0.04),
            "offset_scale_covariance_term": (-0.012, -0.024),
            "predictive_covariance": (
                (0.0605, 0.042),
                (0.042, 0.146),
            ),
        }
        values.update(overrides)
        return InstrumentChannelCalibrationPredictiveUncertainty(**values)

    @staticmethod
    def _unavailable(**overrides):
        values = {
            "schema_version": (
                INSTRUMENT_CHANNEL_CALIBRATION_PREDICTIVE_UNCERTAINTY_SCHEMA_VERSION
            ),
            "status": "unavailable",
            "covariance_mode": "full_covariance",
            "input_shape": (2,),
            "input_measurement_uncertainty_supplied": True,
            "coefficient_uncertainty_status": "unavailable",
            "coefficient_uncertainty_source": "pgmuvi_affine_fit",
            "measurement_variance": None,
            "offset_variance": None,
            "scale_variance": None,
            "offset_scale_covariance_term": None,
            "predictive_covariance": None,
            "reason": "Coefficient covariance is unavailable.",
        }
        values.update(overrides)
        return InstrumentChannelCalibrationPredictiveUncertainty(**values)

    def test_available_full_covariance_is_immutable_and_json_safe(self):
        result = self._available()
        payload = result.to_dict()
        json.dumps(payload, allow_nan=False)

        self.assertEqual(result.input_shape, (2,))
        np.testing.assert_allclose(
            result.coefficient_variance,
            (0.038, 0.056),
        )
        np.testing.assert_allclose(
            result.predictive_variance,
            (0.0605, 0.146),
        )
        np.testing.assert_allclose(
            result.predictive_standard_deviation,
            np.sqrt((0.0605, 0.146)),
        )
        self.assertEqual(payload["coefficient_order"], ["offset", "scale"])
        self.assertEqual(payload["coefficient_jacobian"], "[1, x]")
        self.assertTrue(payload["predictive_uncertainty_propagated"])
        self.assertTrue(
            payload["shared_coefficient_correlation_represented"]
        )

    def test_covariance_modes_are_explicit(self):
        marginal = self._available(
            covariance_mode="marginal_variance",
            predictive_covariance=None,
        )
        self.assertEqual(
            marginal.covariance_mode,
            InstrumentChannelCalibrationPredictiveCovarianceMode.MARGINAL_VARIANCE,
        )
        self.assertIsNone(marginal.predictive_covariance)

        with self.assertRaisesRegex(ValueError, "requires"):
            self._available(predictive_covariance=None)
        with self.assertRaisesRegex(ValueError, "must not carry"):
            self._available(covariance_mode="marginal_variance")

    def test_full_covariance_is_symmetric_psd_with_matching_diagonal(self):
        with self.assertRaisesRegex(ValueError, "diagonal"):
            self._available(
                predictive_covariance=((0.061, 0.042), (0.042, 0.146))
            )
        with self.assertRaisesRegex(ValueError, "symmetric"):
            self._available(
                predictive_covariance=((0.0605, 0.04), (0.042, 0.146))
            )
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            self._available(
                predictive_covariance=((0.0605, 0.2), (0.2, 0.146))
            )

    def test_cross_term_and_total_variance_must_remain_physical(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self._available(
                offset_scale_covariance_term=(-0.2, -0.2),
                predictive_covariance=None,
                covariance_mode="marginal_variance",
            )

    def test_scalar_shape_and_omitted_measurement_errors_are_explicit(self):
        result = self._available(
            input_shape=(),
            input_measurement_uncertainty_supplied=False,
            measurement_variance=(0.0,),
            offset_variance=(0.04,),
            scale_variance=(0.01,),
            offset_scale_covariance_term=(-0.012,),
            predictive_covariance=((0.038,),),
        )
        self.assertEqual(result.input_shape, ())
        np.testing.assert_allclose(
            result.predictive_variance,
            (0.038,),
        )

        with self.assertRaisesRegex(ValueError, "must be zero"):
            self._available(
                input_measurement_uncertainty_supplied=False
            )

    def test_shape_boolean_and_independence_inputs_are_strict(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            self._available(input_shape=(2, 2))
        with self.assertRaisesRegex(TypeError, "not booleans"):
            self._available(measurement_variance=(False, 0.09))
        with self.assertRaisesRegex(TypeError, "not booleans"):
            self._available(input_shape=(True,))
        with self.assertRaisesRegex(TypeError, "must be boolean"):
            self._available(input_measurement_uncertainty_supplied=1)
        with self.assertRaisesRegex(ValueError, "independent"):
            self._available(input_coefficient_independence_assumed=False)

    def test_unavailable_result_is_explicit_and_non_numeric(self):
        result = self._unavailable()
        payload = result.to_dict()
        json.dumps(payload, allow_nan=False)
        self.assertFalse(payload["predictive_uncertainty_propagated"])
        self.assertIsNone(result.predictive_variance)

        with self.assertRaisesRegex(ValueError, "requires a reason"):
            self._unavailable(reason=None)
        with self.assertRaisesRegex(ValueError, "cannot carry"):
            self._unavailable(measurement_variance=(0.01, 0.01))
        with self.assertRaisesRegex(ValueError, "unavailable coefficient"):
            self._unavailable(coefficient_uncertainty_status="available")

    def test_orchestration_dispositions_are_stable(self):
        self.assertEqual(
            [
                item.value
                for item in (
                    InstrumentChannelCalibrationPredictiveUncertaintyDisposition
                )
            ],
            ["not_requested", "available", "skipped", "unavailable"],
        )


class TestPredictiveApplicationBoundary(unittest.TestCase):
    @staticmethod
    def _calibration():
        uncertainty = InstrumentChannelCalibrationCoefficientUncertainty(
            schema_version=(
                INSTRUMENT_CHANNEL_CALIBRATION_UNCERTAINTY_SCHEMA_VERSION
            ),
            status=InstrumentChannelCalibrationUncertaintyStatus.AVAILABLE,
            uncertainty_source="caller_supplied_bootstrap",
            coefficient_covariance=((0.04, -0.006), (-0.006, 0.01)),
            estimation_method="paired_bootstrap",
        )
        return InstrumentChannelCalibration(
            schema_version=INSTRUMENT_CHANNEL_CALIBRATION_MODEL_SCHEMA_VERSION,
            reference_channel="reference",
            channel="target",
            wavelength=0.656,
            offset=0.25,
            scale=1.5,
            n_pairs=20,
            n_inliers=18,
            residual_mad_sigma=0.01,
            coefficient_uncertainty=uncertainty,
        )

    def test_existing_apply_return_contract_is_unchanged(self):
        calibration = self._calibration()
        flux, error = apply_instrument_channel_calibration(
            np.array([1.0, 2.0]),
            calibration,
            flux_error=np.array([0.1, 0.2]),
        )
        np.testing.assert_allclose(flux, [1.75, 3.25])
        np.testing.assert_allclose(error, [0.15, 0.30])
        self.assertFalse(
            calibration.coefficient_uncertainty.to_dict()[
                "predictive_uncertainty_propagated"
            ]
        )

    def test_dedicated_predictive_callable_raises(self):
        with self.assertRaisesRegex(NotImplementedError, "not implemented"):
            apply_instrument_channel_calibration_with_predictive_uncertainty(
                np.array([1.0, 2.0]),
                self._calibration(),
                flux_error=np.array([0.1, 0.2]),
                covariance_mode="full_covariance",
            )


if __name__ == "__main__":
    unittest.main()
