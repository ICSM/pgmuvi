"""Affine coefficient-uncertainty provenance contract tests."""

import json
import unittest

import numpy as np

from pgmuvi.instrument_channel_calibration import (
    INSTRUMENT_CHANNEL_CALIBRATION_MODEL_SCHEMA_VERSION,
    INSTRUMENT_CHANNEL_CALIBRATION_UNCERTAINTY_SCHEMA_VERSION,
    InstrumentChannelCalibration,
    InstrumentChannelCalibrationCoefficientUncertainty,
    InstrumentChannelCalibrationUncertaintyStatus,
    fit_instrument_channel_calibration,
)


class TestInstrumentChannelCalibrationCoefficientUncertainty(unittest.TestCase):
    @staticmethod
    def _available(**overrides):
        values = {
            "schema_version": (
                INSTRUMENT_CHANNEL_CALIBRATION_UNCERTAINTY_SCHEMA_VERSION
            ),
            "status": (
                InstrumentChannelCalibrationUncertaintyStatus.AVAILABLE
            ),
            "uncertainty_source": "caller_supplied_bootstrap",
            "coefficient_covariance": (
                (np.float64(0.04), np.float64(-0.006)),
                (np.float64(-0.006), np.float64(0.01)),
            ),
            "estimation_method": "paired_bootstrap",
            "degrees_of_freedom": np.int64(18),
            "residual_variance": np.float64(0.0025),
        }
        values.update(overrides)
        return InstrumentChannelCalibrationCoefficientUncertainty(**values)

    @staticmethod
    def _unavailable(**overrides):
        values = {
            "schema_version": (
                INSTRUMENT_CHANNEL_CALIBRATION_UNCERTAINTY_SCHEMA_VERSION
            ),
            "status": (
                InstrumentChannelCalibrationUncertaintyStatus.UNAVAILABLE
            ),
            "uncertainty_source": "pgmuvi_affine_fit",
            "coefficient_covariance": None,
            "reason": "Coefficient uncertainty was not estimated.",
        }
        values.update(overrides)
        return InstrumentChannelCalibrationCoefficientUncertainty(**values)

    def test_unknown_uncertainty_and_legacy_model_schemas_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "uncertainty schema"):
            self._available(schema_version="unknown")

        with self.assertRaisesRegex(ValueError, "model schema"):
            InstrumentChannelCalibration(
                schema_version="1.0",
                reference_channel="reference",
                channel="target",
                wavelength=1.0,
                offset=0.25,
                scale=1.5,
                n_pairs=20,
                n_inliers=18,
                residual_mad_sigma=0.01,
                coefficient_uncertainty=self._available(),
            )

    def test_available_covariance_is_immutable_and_json_safe(self):
        uncertainty = self._available()

        self.assertEqual(
            uncertainty.coefficient_covariance,
            ((0.04, -0.006), (-0.006, 0.01)),
        )
        self.assertAlmostEqual(uncertainty.offset_standard_error, 0.2)
        self.assertAlmostEqual(uncertainty.scale_standard_error, 0.1)

        payload = uncertainty.to_dict()
        json.dumps(payload, allow_nan=False)
        self.assertEqual(payload["coefficient_order"], ["offset", "scale"])
        self.assertEqual(
            payload["coefficient_covariance"],
            [[0.04, -0.006], [-0.006, 0.01]],
        )
        self.assertFalse(payload["predictive_uncertainty_propagated"])

    def test_available_uncertainty_requires_covariance_and_method(self):
        with self.assertRaisesRegex(ValueError, "requires a complete"):
            self._available(coefficient_covariance=None)
        with self.assertRaisesRegex(ValueError, "estimation_method"):
            self._available(estimation_method=None)
        with self.assertRaisesRegex(ValueError, "must not carry"):
            self._available(reason="Unexpected reason.")

    def test_covariance_must_be_symmetric_positive_semidefinite(self):
        with self.assertRaisesRegex(ValueError, "symmetric"):
            self._available(
                coefficient_covariance=((1.0, 0.2), (0.1, 1.0))
            )
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            self._available(
                coefficient_covariance=((1.0, 2.0), (2.0, 1.0))
            )
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            self._available(
                coefficient_covariance=(
                    (1.0e-8, 1.1e-8),
                    (1.1e-8, 1.0e-8),
                )
            )

    def test_covariance_rejects_boolean_values(self):
        boolean_covariances = (
            [[1.0, False], [False, 1.0]],
            (
                (1.0, np.bool_(False)),
                (np.bool_(False), 1.0),
            ),
            np.asarray(
                [[1.0, False], [False, 1.0]],
                dtype=object,
            ),
        )

        for covariance in boolean_covariances:
            with self.subTest(covariance=covariance):
                with self.assertRaisesRegex(TypeError, "not booleans"):
                    self._available(
                        coefficient_covariance=covariance
                    )

    def test_unavailable_uncertainty_is_explicit_and_non_numeric(self):
        uncertainty = self._unavailable()
        payload = uncertainty.to_dict()
        json.dumps(payload, allow_nan=False)

        self.assertEqual(payload["status"], "unavailable")
        self.assertIsNone(payload["coefficient_covariance"])
        self.assertIsNone(payload["offset_standard_error"])
        self.assertEqual(
            payload["reason"],
            "Coefficient uncertainty was not estimated.",
        )

        with self.assertRaisesRegex(ValueError, "requires a reason"):
            self._unavailable(reason=None)
        with self.assertRaisesRegex(ValueError, "cannot contain"):
            self._unavailable(
                coefficient_covariance=((1.0, 0.0), (0.0, 1.0))
            )

    def test_optional_estimation_metadata_is_strict(self):
        with self.assertRaisesRegex(TypeError, "integer"):
            self._available(degrees_of_freedom=True)
        with self.assertRaisesRegex(ValueError, "at least 1"):
            self._available(degrees_of_freedom=0)
        with self.assertRaisesRegex(TypeError, "not boolean"):
            self._available(residual_variance=np.bool_(False))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self._available(residual_variance=-0.1)

    def test_model_schema_requires_nested_uncertainty_record(self):
        uncertainty = self._available()
        calibration = InstrumentChannelCalibration(
            schema_version=(
                INSTRUMENT_CHANNEL_CALIBRATION_MODEL_SCHEMA_VERSION
            ),
            reference_channel="reference",
            channel="target",
            wavelength=1.0,
            offset=0.25,
            scale=1.5,
            n_pairs=20,
            n_inliers=18,
            residual_mad_sigma=0.01,
            coefficient_uncertainty=uncertainty,
        )

        payload = calibration.to_dict()
        json.dumps(payload, allow_nan=False)
        self.assertEqual(
            payload["coefficient_uncertainty"]["status"],
            "available",
        )

        with self.assertRaisesRegex(
            TypeError,
            "InstrumentChannelCalibrationCoefficientUncertainty",
        ):
            InstrumentChannelCalibration(
                schema_version=(
                    INSTRUMENT_CHANNEL_CALIBRATION_MODEL_SCHEMA_VERSION
                ),
                reference_channel="reference",
                channel="target",
                wavelength=1.0,
                offset=0.25,
                scale=1.5,
                n_pairs=20,
                n_inliers=18,
                residual_mad_sigma=0.01,
                coefficient_uncertainty=object(),
            )

    def test_unweighted_fit_estimates_scaled_normal_matrix_covariance(self):
        channel_flux = np.linspace(-1.0, 2.0, 24)
        perturbation = 0.01 * np.sin(np.arange(channel_flux.size))
        reference_flux = 0.25 + 1.5 * channel_flux + perturbation

        calibration = fit_instrument_channel_calibration(
            reference_flux,
            channel_flux,
            reference_channel="reference",
            channel="target",
            wavelength=1.0,
            sigma_clip=100.0,
        )

        uncertainty = calibration.coefficient_uncertainty
        self.assertEqual(
            uncertainty.status,
            InstrumentChannelCalibrationUncertaintyStatus.AVAILABLE,
        )
        self.assertEqual(
            uncertainty.uncertainty_source,
            "pgmuvi_affine_fit_final_inliers",
        )
        self.assertEqual(
            uncertainty.estimation_method,
            "ordinary_least_squares_residual_variance_scaled_"
            "normal_matrix_inverse",
        )
        self.assertEqual(uncertainty.degrees_of_freedom, 22)

        design = np.column_stack((np.ones(24), channel_flux))
        residual = reference_flux - (
            calibration.offset + calibration.scale * channel_flux
        )
        expected_residual_variance = float(
            np.dot(residual, residual) / 22
        )
        expected_covariance = (
            np.linalg.inv(design.T @ design)
            * expected_residual_variance
        )

        self.assertAlmostEqual(
            uncertainty.residual_variance,
            expected_residual_variance,
        )
        np.testing.assert_allclose(
            uncertainty.coefficient_covariance,
            expected_covariance,
            rtol=1.0e-12,
            atol=1.0e-15,
        )
        self.assertGreater(uncertainty.offset_standard_error, 0.0)
        self.assertGreater(uncertainty.scale_standard_error, 0.0)
        json.dumps(calibration.to_dict(), allow_nan=False)

    def test_reference_error_fit_uses_known_variance_covariance(self):
        channel_flux = np.linspace(0.0, 2.0, 20)
        reference_flux = 0.25 + 1.5 * channel_flux
        reference_error = np.linspace(0.02, 0.05, channel_flux.size)

        calibration = fit_instrument_channel_calibration(
            reference_flux,
            channel_flux,
            reference_channel="reference",
            channel="target",
            wavelength=1.0,
            reference_error=reference_error,
        )

        uncertainty = calibration.coefficient_uncertainty
        self.assertEqual(
            uncertainty.status,
            InstrumentChannelCalibrationUncertaintyStatus.AVAILABLE,
        )
        self.assertEqual(
            uncertainty.estimation_method,
            "known_variance_weighted_normal_matrix_inverse",
        )
        self.assertEqual(uncertainty.degrees_of_freedom, 18)
        self.assertIsNone(uncertainty.residual_variance)

        design = np.column_stack((np.ones(20), channel_flux))
        weights = 1.0 / reference_error**2
        expected_covariance = np.linalg.inv(
            design.T @ (weights[:, None] * design)
        )
        np.testing.assert_allclose(
            uncertainty.coefficient_covariance,
            expected_covariance,
            rtol=1.0e-12,
            atol=1.0e-15,
        )

    def test_channel_error_fit_records_uncertainty_as_unavailable(self):
        channel_flux = np.linspace(0.0, 2.0, 20)
        reference_flux = 0.25 + 1.5 * channel_flux

        calibration = fit_instrument_channel_calibration(
            reference_flux,
            channel_flux,
            reference_channel="reference",
            channel="target",
            wavelength=1.0,
            reference_error=np.full(20, 0.03),
            channel_error=np.linspace(0.01, 0.02, 20),
        )

        uncertainty = calibration.coefficient_uncertainty
        self.assertEqual(
            uncertainty.status,
            InstrumentChannelCalibrationUncertaintyStatus.UNAVAILABLE,
        )
        self.assertIsNone(uncertainty.coefficient_covariance)
        self.assertEqual(
            uncertainty.reason,
            "Coefficient covariance is unavailable when channel-axis "
            "measurement errors contribute scale-dependent effective "
            "variances; the current affine fitter does not expose a "
            "covariance estimator for the full iterative weighting procedure.",
        )
        json.dumps(calibration.to_dict(), allow_nan=False)

    def test_numerically_rank_deficient_fit_has_explicit_unavailable_reason(self):
        channel_flux = 1.0 + 1.0e-12 * np.arange(20, dtype=float)
        reference_flux = 0.25 + 1.5 * channel_flux

        calibration = fit_instrument_channel_calibration(
            reference_flux,
            channel_flux,
            reference_channel="reference",
            channel="target",
            wavelength=1.0,
        )

        uncertainty = calibration.coefficient_uncertainty
        self.assertEqual(
            uncertainty.status,
            InstrumentChannelCalibrationUncertaintyStatus.UNAVAILABLE,
        )
        self.assertIsNone(uncertainty.coefficient_covariance)
        self.assertEqual(
            uncertainty.reason,
            "Final weighted affine design is numerically rank-deficient; "
            "coefficient covariance is unavailable.",
        )
        json.dumps(calibration.to_dict(), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
