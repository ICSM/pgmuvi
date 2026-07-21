"""Tests for explicit paired instrument-channel calibration."""

import json
import unittest

import numpy as np

from pgmuvi.instrument_channel_calibration import (
    INSTRUMENT_CHANNEL_CALIBRATION_MODEL_SCHEMA_VERSION,
    InstrumentChannelCalibration,
    apply_instrument_channel_calibration,
    fit_instrument_channel_calibration,
)


class TestInstrumentChannelCalibrationFit(unittest.TestCase):
    def setUp(self):
        self.channel_flux = np.linspace(0.2, 2.0, 60)
        perturbation = 0.002 * np.sin(
            np.linspace(0.0, 4.0 * np.pi, 60)
        )
        self.reference_flux = (
            0.125 + 1.35 * self.channel_flux + perturbation
        )

    def test_affine_coefficients_are_recovered(self):
        calibration = fit_instrument_channel_calibration(
            self.reference_flux,
            self.channel_flux,
            reference_channel="reference",
            channel="target",
            wavelength=0.656,
        )

        self.assertIsInstance(
            calibration,
            InstrumentChannelCalibration,
        )
        self.assertAlmostEqual(calibration.offset, 0.125, delta=0.002)
        self.assertAlmostEqual(calibration.scale, 1.35, delta=0.002)
        self.assertEqual(calibration.n_pairs, 60)
        self.assertEqual(calibration.n_inliers, 60)

    def test_single_extreme_outlier_is_clipped(self):
        reference_flux = self.reference_flux.copy()
        reference_flux[12] += 5.0

        calibration = fit_instrument_channel_calibration(
            reference_flux,
            self.channel_flux,
            reference_channel="reference",
            channel="target",
            wavelength=0.656,
        )

        self.assertLess(calibration.n_inliers, calibration.n_pairs)
        self.assertAlmostEqual(calibration.offset, 0.125, delta=0.002)
        self.assertAlmostEqual(calibration.scale, 1.35, delta=0.002)

    def test_zero_mad_outliers_are_clipped(self):
        channel_flux = np.linspace(0.0, 1.0, 61)
        reference_flux = 0.25 + 1.4 * channel_flux
        reference_flux[[0, 30, 60]] += np.array(
            [5.0, -10.0, 5.0]
        )

        calibration = fit_instrument_channel_calibration(
            reference_flux,
            channel_flux,
            reference_channel="reference",
            channel="target",
            wavelength=0.656,
        )

        self.assertEqual(calibration.n_pairs, 61)
        self.assertEqual(calibration.n_inliers, 58)
        self.assertAlmostEqual(
            calibration.offset,
            0.25,
            places=12,
        )
        self.assertAlmostEqual(
            calibration.scale,
            1.4,
            places=12,
        )

    def test_optional_measurement_errors_are_supported(self):
        calibration = fit_instrument_channel_calibration(
            self.reference_flux,
            self.channel_flux,
            reference_channel="reference",
            channel="target",
            wavelength=0.656,
            reference_error=np.full(60, 0.01),
            channel_error=np.full(60, 0.02),
        )

        self.assertGreater(calibration.scale, 0.0)
        self.assertEqual(calibration.n_pairs, 60)

    def test_nonfinite_pairs_are_removed_before_fitting(self):
        reference_flux = self.reference_flux.copy()
        channel_flux = self.channel_flux.copy()
        reference_flux[2] = np.nan
        channel_flux[4] = np.inf

        calibration = fit_instrument_channel_calibration(
            reference_flux,
            channel_flux,
            reference_channel="reference",
            channel="target",
            wavelength=0.656,
        )

        self.assertEqual(calibration.n_pairs, 58)

    def test_constant_channel_flux_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "span more than one",
        ):
            fit_instrument_channel_calibration(
                self.reference_flux,
                np.ones(60),
                reference_channel="reference",
                channel="target",
                wavelength=0.656,
            )

    def test_mismatched_pair_shapes_are_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "same shape",
        ):
            fit_instrument_channel_calibration(
                self.reference_flux,
                self.channel_flux[:-1],
                reference_channel="reference",
                channel="target",
                wavelength=0.656,
            )

    def test_channel_names_are_normalized(self):
        calibration = fit_instrument_channel_calibration(
            self.reference_flux,
            self.channel_flux,
            reference_channel=" reference ",
            channel=" target ",
            wavelength=0.656,
        )

        self.assertEqual(calibration.reference_channel, "reference")
        self.assertEqual(calibration.channel, "target")

    def test_channel_identity_must_be_explicit_and_distinct(self):
        with self.assertRaisesRegex(
            ValueError,
            "different observational channels",
        ):
            fit_instrument_channel_calibration(
                self.reference_flux,
                self.channel_flux,
                reference_channel="same",
                channel="same",
                wavelength=0.656,
            )

    def test_model_payload_is_strict_json_safe(self):
        calibration = fit_instrument_channel_calibration(
            self.reference_flux,
            self.channel_flux,
            reference_channel="reference",
            channel="target",
            wavelength=-1.5,
        )

        payload = calibration.to_dict()
        encoded = json.dumps(payload, allow_nan=False)

        self.assertIn(
            INSTRUMENT_CHANNEL_CALIBRATION_MODEL_SCHEMA_VERSION,
            encoded,
        )
        self.assertFalse(payload["automatic_time_matching"])
        self.assertFalse(payload["automatic_model_selection"])

    def test_fit_method_is_normalized_and_type_checked(self):
        common = {
            "schema_version": (
                INSTRUMENT_CHANNEL_CALIBRATION_MODEL_SCHEMA_VERSION
            ),
            "reference_channel": "reference",
            "channel": "target",
            "wavelength": 0.656,
            "offset": 0.25,
            "scale": 1.5,
            "n_pairs": 20,
            "n_inliers": 18,
            "residual_mad_sigma": 0.01,
        }

        calibration = InstrumentChannelCalibration(
            **common,
            fit_method=" custom-affine ",
        )

        self.assertEqual(
            calibration.fit_method,
            "custom-affine",
        )
        json.dumps(calibration.to_dict(), allow_nan=False)

        with self.assertRaisesRegex(
            TypeError,
            "fit_method must be a string",
        ):
            InstrumentChannelCalibration(
                **common,
                fit_method=object(),
            )


class TestInstrumentChannelCalibrationApply(unittest.TestCase):
    def setUp(self):
        self.calibration = InstrumentChannelCalibration(
            schema_version=(
                INSTRUMENT_CHANNEL_CALIBRATION_MODEL_SCHEMA_VERSION
            ),
            reference_channel="reference",
            channel="target",
            wavelength=0.656,
            offset=0.25,
            scale=1.5,
            n_pairs=20,
            n_inliers=18,
            residual_mad_sigma=0.01,
        )

    def test_flux_is_mapped_to_reference_scale(self):
        result = apply_instrument_channel_calibration(
            np.array([1.0, 2.0, 3.0]),
            self.calibration,
        )

        np.testing.assert_allclose(
            result,
            np.array([1.75, 3.25, 4.75]),
        )

    def test_uncertainties_are_scaled(self):
        flux, error = apply_instrument_channel_calibration(
            np.array([1.0, 2.0]),
            self.calibration,
            flux_error=np.array([0.1, 0.2]),
        )

        np.testing.assert_allclose(
            flux,
            np.array([1.75, 3.25]),
        )
        np.testing.assert_allclose(
            error,
            np.array([0.15, 0.30]),
        )

    def test_calibration_type_is_checked(self):
        with self.assertRaisesRegex(
            TypeError,
            "InstrumentChannelCalibration",
        ):
            apply_instrument_channel_calibration(
                np.array([1.0]),
                object(),
            )

    def test_error_shape_is_checked(self):
        with self.assertRaisesRegex(
            ValueError,
            "same shape",
        ):
            apply_instrument_channel_calibration(
                np.array([1.0, 2.0]),
                self.calibration,
                flux_error=np.array([0.1]),
            )


if __name__ == "__main__":
    unittest.main()
