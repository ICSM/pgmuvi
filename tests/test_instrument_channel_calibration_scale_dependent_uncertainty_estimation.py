"""Scale-dependent channel-axis uncertainty estimator tests."""

import json
import math
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import OptimizeResult

from pgmuvi import instrument_channel_calibration
from pgmuvi.instrument_channel_calibration import (
    InstrumentChannelCalibrationUncertaintyStatus,
    estimate_scale_dependent_instrument_channel_calibration_coefficient_uncertainty,
)


class TestScaleDependentCalibrationUncertaintyEstimation(
    unittest.TestCase
):
    @staticmethod
    def _data():
        channel_flux = np.linspace(-1.5, 2.5, 80)
        reference_error = np.linspace(
            0.025,
            0.045,
            channel_flux.size,
        )
        channel_error = np.linspace(
            0.012,
            0.032,
            channel_flux.size,
        )
        perturbation = (
            0.028
            * np.sin(
                0.73 * np.arange(channel_flux.size)
            )
        )
        reference_flux = (
            0.35
            + 1.45 * channel_flux
            + perturbation
        )
        return (
            reference_flux,
            channel_flux,
            reference_error,
            channel_error,
        )

    def test_estimator_returns_inverse_observed_hessian(self):
        (
            reference_flux,
            channel_flux,
            reference_error,
            channel_error,
        ) = self._data()

        result = (
            estimate_scale_dependent_instrument_channel_calibration_coefficient_uncertainty(
                reference_flux,
                channel_flux,
                reference_error=reference_error,
                channel_error=channel_error,
                initial_offset=0.3,
                initial_scale=1.4,
            )
        )

        self.assertEqual(
            result.status,
            InstrumentChannelCalibrationUncertaintyStatus.AVAILABLE,
        )
        self.assertEqual(
            result.uncertainty_source,
            "pgmuvi_scale_dependent_full_objective_final_inliers",
        )
        self.assertTrue(result.optimizer_converged)
        self.assertGreater(result.scale, 0.0)
        self.assertEqual(result.n_inliers, channel_flux.size)
        self.assertEqual(
            result.gradient_method,
            "analytic_full_objective_gradient",
        )
        self.assertEqual(
            result.hessian_method,
            "analytic_observed_hessian",
        )
        self.assertTrue(result.to_dict()["implemented"])
        json.dumps(result.to_dict(), allow_nan=False)

        residual = reference_flux - (
            result.offset + result.scale * channel_flux
        )
        variance = (
            reference_error**2
            + result.scale**2 * channel_error**2
        )
        channel_variance = channel_error**2

        hessian = np.asarray(
            [
                [
                    np.sum(1.0 / variance),
                    np.sum(
                        channel_flux / variance
                        + 2.0
                        * result.scale
                        * channel_variance
                        * residual
                        / variance**2
                    ),
                ],
                [
                    np.sum(
                        channel_flux / variance
                        + 2.0
                        * result.scale
                        * channel_variance
                        * residual
                        / variance**2
                    ),
                    np.sum(
                        channel_flux**2 / variance
                        + channel_variance / variance
                        - 2.0
                        * result.scale**2
                        * channel_variance**2
                        / variance**2
                        - channel_variance
                        * residual**2
                        / variance**2
                        + 4.0
                        * result.scale
                        * channel_variance
                        * residual
                        * channel_flux
                        / variance**2
                        + 4.0
                        * result.scale**2
                        * channel_variance**2
                        * residual**2
                        / variance**3
                    ),
                ],
            ],
            dtype=float,
        )

        expected_covariance = np.linalg.inv(hessian)
        np.testing.assert_allclose(
            result.coefficient_covariance,
            expected_covariance,
            rtol=1.0e-9,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            result.hessian_eigenvalues,
            np.linalg.eigvalsh(hessian),
            rtol=1.0e-10,
            atol=1.0e-12,
        )

        gradient = np.asarray(
            [
                np.sum(-residual / variance),
                np.sum(
                    -residual * channel_flux / variance
                    + result.scale
                    * channel_variance
                    * (
                        1.0 / variance
                        - residual**2 / variance**2
                    )
                ),
            ],
            dtype=float,
        )
        self.assertAlmostEqual(
            result.gradient_norm,
            float(np.linalg.norm(gradient)),
            places=10,
        )

    def test_reference_error_may_be_absent(self):
        (
            reference_flux,
            channel_flux,
            _,
            channel_error,
        ) = self._data()

        result = (
            estimate_scale_dependent_instrument_channel_calibration_coefficient_uncertainty(
                reference_flux,
                channel_flux,
                reference_error=None,
                channel_error=channel_error,
                initial_offset=0.3,
                initial_scale=1.4,
            )
        )

        self.assertEqual(
            result.status,
            InstrumentChannelCalibrationUncertaintyStatus.AVAILABLE,
        )
        self.assertGreater(result.scale, 0.0)
        self.assertTrue(
            all(
                eigenvalue > 0.0
                for eigenvalue in result.hessian_eigenvalues
            )
        )

    def test_input_validation_is_strict(self):
        (
            reference_flux,
            channel_flux,
            reference_error,
            channel_error,
        ) = self._data()

        invalid_calls = (
            (
                {"channel_flux": channel_flux[:-1]},
                "same shape",
            ),
            (
                {"channel_error": None},
                "channel_error is required",
            ),
            (
                {
                    "channel_error": np.zeros_like(
                        channel_error
                    )
                },
                "strictly positive",
            ),
            (
                {
                    "reference_error": np.zeros_like(
                        reference_error
                    )
                },
                "strictly positive",
            ),
            (
                {
                    "reference_flux": np.full_like(
                        reference_flux,
                        np.nan,
                    )
                },
                "finite values",
            ),
            (
                {"initial_scale": 0.0},
                "strictly positive",
            ),
        )

        for overrides, message in invalid_calls:
            with self.subTest(overrides=overrides):
                values = {
                    "reference_flux": reference_flux,
                    "channel_flux": channel_flux,
                    "reference_error": reference_error,
                    "channel_error": channel_error,
                    "initial_offset": 0.3,
                    "initial_scale": 1.4,
                }
                values.update(overrides)
                with self.assertRaisesRegex(
                    ValueError,
                    message,
                ):
                    estimate_scale_dependent_instrument_channel_calibration_coefficient_uncertainty(
                        **values
                    )

        with self.assertRaisesRegex(
            TypeError,
            "not boolean",
        ):
            estimate_scale_dependent_instrument_channel_calibration_coefficient_uncertainty(
                reference_flux,
                channel_flux,
                reference_error=reference_error,
                channel_error=channel_error,
                initial_offset=False,
                initial_scale=1.4,
            )

    def test_optimizer_failure_returns_explicit_unavailable_result(self):
        (
            reference_flux,
            channel_flux,
            reference_error,
            channel_error,
        ) = self._data()

        failure = OptimizeResult(
            success=False,
            message="forced optimizer failure",
            x=np.asarray([0.3, math.log(1.4)]),
        )

        with patch.object(
            instrument_channel_calibration.optimize,
            "minimize",
            return_value=failure,
        ):
            result = (
                estimate_scale_dependent_instrument_channel_calibration_coefficient_uncertainty(
                    reference_flux,
                    channel_flux,
                    reference_error=reference_error,
                    channel_error=channel_error,
                    initial_offset=0.3,
                    initial_scale=1.4,
                )
            )

        self.assertEqual(
            result.status,
            InstrumentChannelCalibrationUncertaintyStatus.UNAVAILABLE,
        )
        self.assertFalse(result.optimizer_converged)
        self.assertIsNone(result.coefficient_covariance)
        self.assertIn(
            "forced optimizer failure",
            result.reason,
        )
        json.dumps(result.to_dict(), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
