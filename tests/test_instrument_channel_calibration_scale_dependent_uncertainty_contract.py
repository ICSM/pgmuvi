"""Scale-dependent channel-axis uncertainty contract tests."""

import json
import unittest

import numpy as np

from pgmuvi.instrument_channel_calibration import (
    INSTRUMENT_CHANNEL_CALIBRATION_SCALE_DEPENDENT_UNCERTAINTY_SCHEMA_VERSION,
    InstrumentChannelCalibrationScaleDependentUncertaintyEstimate,
    InstrumentChannelCalibrationUncertaintyStatus,
    estimate_scale_dependent_instrument_channel_calibration_coefficient_uncertainty,
    fit_instrument_channel_calibration,
)


class TestScaleDependentUncertaintyEstimateContract(unittest.TestCase):
    @staticmethod
    def _available(**overrides):
        values = {
            "schema_version": (
                INSTRUMENT_CHANNEL_CALIBRATION_SCALE_DEPENDENT_UNCERTAINTY_SCHEMA_VERSION
            ),
            "status": InstrumentChannelCalibrationUncertaintyStatus.AVAILABLE,
            "uncertainty_source": "future_full_objective_estimator",
            "n_inliers": np.int64(24),
            "coefficient_covariance": (
                (np.float64(0.04), np.float64(-0.006)),
                (np.float64(-0.006), np.float64(0.01)),
            ),
            "offset": np.float64(0.25),
            "scale": np.float64(1.5),
            "objective_value": np.float64(-12.5),
            "optimizer": "bounded_quasi_newton",
            "optimizer_converged": np.bool_(True),
            "gradient_method": "analytic",
            "gradient_norm": np.float64(1.0e-9),
            "hessian_method": "analytic_observed_hessian",
            "hessian_eigenvalues": (
                np.float64(20.0),
                np.float64(80.0),
            ),
        }
        values.update(overrides)
        return InstrumentChannelCalibrationScaleDependentUncertaintyEstimate(
            **values
        )

    @staticmethod
    def _unavailable(**overrides):
        values = {
            "schema_version": (
                INSTRUMENT_CHANNEL_CALIBRATION_SCALE_DEPENDENT_UNCERTAINTY_SCHEMA_VERSION
            ),
            "status": (
                InstrumentChannelCalibrationUncertaintyStatus.UNAVAILABLE
            ),
            "uncertainty_source": "future_full_objective_estimator",
            "n_inliers": 24,
            "optimizer": "bounded_quasi_newton",
            "optimizer_converged": False,
            "gradient_method": "analytic",
            "hessian_method": "analytic_observed_hessian",
            "reason": "Observed Hessian is not positive definite.",
        }
        values.update(overrides)
        return InstrumentChannelCalibrationScaleDependentUncertaintyEstimate(
            **values
        )

    def test_available_contract_is_immutable_and_json_safe(self):
        result = self._available()
        payload = result.to_dict()
        json.dumps(payload, allow_nan=False)

        self.assertEqual(payload["coefficient_order"], ["offset", "scale"])
        self.assertEqual(
            payload["objective"],
            "gaussian_negative_log_likelihood_"
            "scale_dependent_effective_variance",
        )
        self.assertEqual(
            payload["effective_variance_equation"],
            "reference_error_i**2 + scale**2 * channel_error_i**2",
        )
        self.assertEqual(
            payload["covariance_estimator"],
            "inverse_observed_hessian_at_converged_optimum",
        )
        self.assertTrue(payload["conditioned_on_final_inlier_set"])
        self.assertTrue(payload["positive_scale_domain_enforced"])
        self.assertTrue(payload["implemented"])
        self.assertAlmostEqual(result.offset_standard_error, 0.2)
        self.assertAlmostEqual(result.scale_standard_error, 0.1)

        with self.assertRaisesRegex(
            AttributeError,
            "cannot assign to field",
        ):
            result.scale = 2.0

    def test_available_contract_requires_complete_converged_provenance(self):
        required_fields = (
            "coefficient_covariance",
            "offset",
            "scale",
            "objective_value",
            "optimizer",
            "optimizer_converged",
            "gradient_method",
            "gradient_norm",
            "hessian_method",
            "hessian_eigenvalues",
        )

        for name in required_fields:
            with self.subTest(name=name):
                replacement = False if name == "optimizer_converged" else None
                with self.assertRaisesRegex(
                    ValueError,
                    "requires complete|requires a converged optimizer",
                ):
                    self._available(**{name: replacement})

        with self.assertRaisesRegex(ValueError, "strictly positive"):
            self._available(scale=0.0)
        with self.assertRaisesRegex(ValueError, "must not carry"):
            self._available(reason="Unexpected failure.")

    def test_hessian_and_boolean_fields_are_strict(self):
        with self.assertRaisesRegex(ValueError, "exactly two"):
            self._available(hessian_eigenvalues=(1.0,))
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            self._available(hessian_eigenvalues=(1.0, 0.0))
        with self.assertRaisesRegex(TypeError, "not boolean"):
            self._available(hessian_eigenvalues=(1.0, False))
        with self.assertRaisesRegex(TypeError, "must be boolean"):
            self._available(optimizer_converged=1)
        with self.assertRaisesRegex(ValueError, "must be true"):
            self._available(conditioned_on_final_inlier_set=False)
        with self.assertRaisesRegex(ValueError, "must be true"):
            self._available(positive_scale_domain_enforced=False)

    def test_unavailable_contract_is_explicit_and_non_numeric(self):
        result = self._unavailable()
        payload = result.to_dict()
        json.dumps(payload, allow_nan=False)

        self.assertEqual(payload["status"], "unavailable")
        self.assertIsNone(payload["coefficient_covariance"])
        self.assertIsNone(payload["offset"])
        self.assertIsNone(payload["scale"])
        self.assertEqual(
            payload["reason"],
            "Observed Hessian is not positive definite.",
        )

        with self.assertRaisesRegex(ValueError, "requires a reason"):
            self._unavailable(reason=None)
        with self.assertRaisesRegex(ValueError, "cannot carry coefficient"):
            self._unavailable(
                coefficient_covariance=((1.0, 0.0), (0.0, 1.0))
            )
        with self.assertRaisesRegex(ValueError, "numerical estimation"):
            self._unavailable(objective_value=1.0)
        with self.assertRaisesRegex(ValueError, "cannot report"):
            self._unavailable(optimizer_converged=True)

    def test_schema_and_inlier_count_are_strict(self):
        with self.assertRaisesRegex(ValueError, "schema version"):
            self._available(schema_version="unknown")
        with self.assertRaisesRegex(TypeError, "n_inliers must be an integer"):
            self._available(n_inliers=True)
        with self.assertRaisesRegex(ValueError, "at least 3"):
            self._available(n_inliers=2)

    def test_dedicated_estimator_callable_is_implemented(self):
        channel_flux = np.linspace(-1.0, 2.0, 40)
        reference_flux = (
            0.25
            + 1.5 * channel_flux
            + 0.02 * np.sin(np.arange(channel_flux.size))
        )

        result = (
            estimate_scale_dependent_instrument_channel_calibration_coefficient_uncertainty(
                reference_flux,
                channel_flux,
                reference_error=np.linspace(0.02, 0.04, channel_flux.size),
                channel_error=np.linspace(0.01, 0.03, channel_flux.size),
                initial_offset=0.2,
                initial_scale=1.4,
            )
        )

        self.assertEqual(
            result.status,
            InstrumentChannelCalibrationUncertaintyStatus.AVAILABLE,
        )
        self.assertTrue(result.optimizer_converged)
        self.assertTrue(result.to_dict()["implemented"])

    def test_current_fitter_boundary_remains_unavailable(self):
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
        self.assertIn(
            "current affine fitter does not expose a covariance estimator",
            uncertainty.reason,
        )


if __name__ == "__main__":
    unittest.main()
