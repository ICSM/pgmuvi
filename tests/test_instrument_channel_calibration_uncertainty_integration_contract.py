"""Scale-dependent fitter and orchestration integration contract tests."""

import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import OptimizeResult

from pgmuvi import instrument_channel_calibration

from pgmuvi.instrument_channel_calibration import (
    INSTRUMENT_CHANNEL_CALIBRATION_FIT_PROVENANCE_SCHEMA_VERSION,
    INSTRUMENT_CHANNEL_CALIBRATION_MODEL_SCHEMA_VERSION,
    InstrumentChannelCalibration,
    InstrumentChannelCalibrationChannelPlan,
    InstrumentChannelCalibrationDisposition,
    InstrumentChannelCalibrationFitProvenance,
    InstrumentChannelCalibrationGroupPlan,
    InstrumentChannelCalibrationPredictiveCovarianceMode,
    InstrumentChannelCalibrationPredictiveUncertaintyStatus,
    InstrumentChannelCalibrationUncertaintyEstimator,
    InstrumentChannelCalibrationUncertaintyIntegrationStatus,
    InstrumentChannelCalibrationUncertaintyStatus,
    InstrumentChannelPairingMethod,
    apply_instrument_channel_calibration_with_predictive_uncertainty,
    assess_instrument_channel_calibration_requirement,
    define_instrument_channel_calibration_plan,
    estimate_scale_dependent_instrument_channel_calibration_coefficient_uncertainty,
    execute_instrument_channel_calibration_plan,
    fit_instrument_channel_calibration,
    select_instrument_channel_calibration_uncertainty_estimator,
)


class TestCalibrationUncertaintyIntegrationContract(unittest.TestCase):
    @staticmethod
    def _provenance(**overrides):
        values = {
            "schema_version": (
                INSTRUMENT_CHANNEL_CALIBRATION_FIT_PROVENANCE_SCHEMA_VERSION
            ),
            "selected_uncertainty_estimator": (
                InstrumentChannelCalibrationUncertaintyEstimator
                .FIXED_WEIGHT_NORMAL_MATRIX
            ),
            "integration_status": (
                InstrumentChannelCalibrationUncertaintyIntegrationStatus
                .ACTIVE
            ),
            "reference_error_supplied": False,
            "channel_error_supplied": False,
            "n_input_pairs": 6,
            "finite_pair_indices": (0, 1, 2, 3, 4, 5),
            "final_inlier_indices": (0, 1, 2, 4, 5),
            "point_estimate_source": "pgmuvi_affine_fit_final_inliers",
            "point_estimate_objective": "fixed_weight_least_squares",
            "point_estimate_matches_uncertainty_objective": True,
        }
        values.update(overrides)
        return InstrumentChannelCalibrationFitProvenance(**values)

    def test_estimator_selection_is_deterministic_for_all_error_axes(self):
        fixed = (
            InstrumentChannelCalibrationUncertaintyEstimator
            .FIXED_WEIGHT_NORMAL_MATRIX
        )
        scale_dependent = (
            InstrumentChannelCalibrationUncertaintyEstimator
            .SCALE_DEPENDENT_FULL_OBJECTIVE
        )

        cases = (
            (False, False, fixed),
            (True, False, fixed),
            (False, True, scale_dependent),
            (True, True, scale_dependent),
        )
        for reference_supplied, channel_supplied, expected in cases:
            with self.subTest(
                reference_supplied=reference_supplied,
                channel_supplied=channel_supplied,
            ):
                self.assertIs(
                    select_instrument_channel_calibration_uncertainty_estimator(
                        reference_error_supplied=reference_supplied,
                        channel_error_supplied=channel_supplied,
                    ),
                    expected,
                )

        with self.assertRaisesRegex(TypeError, "must be boolean"):
            select_instrument_channel_calibration_uncertainty_estimator(
                reference_error_supplied=1,
                channel_error_supplied=False,
            )

    def test_fit_provenance_is_immutable_json_safe_and_indexed(self):
        provenance = self._provenance()
        payload = provenance.to_dict()
        json.dumps(payload, allow_nan=False)

        self.assertEqual(payload["coefficient_order"], ["offset", "scale"])
        self.assertEqual(payload["error_axes"], [])
        self.assertEqual(payload["n_input_pairs"], 6)
        self.assertEqual(payload["n_finite_pairs"], 6)
        self.assertEqual(payload["n_final_inliers"], 5)
        self.assertEqual(payload["final_inlier_indices"], [0, 1, 2, 4, 5])
        self.assertTrue(
            payload["uncertainty_conditioned_on_final_inlier_set"]
        )

        with self.assertRaisesRegex(AttributeError, "cannot assign"):
            provenance.n_input_pairs = 8
        with self.assertRaisesRegex(ValueError, "subset"):
            self._provenance(
                finite_pair_indices=(0, 1, 2, 4, 5),
                final_inlier_indices=(0, 1, 3),
            )
        with self.assertRaisesRegex(ValueError, "increasing order"):
            self._provenance(finite_pair_indices=(0, 2, 1, 3, 4, 5))

    def test_scale_dependent_success_and_fallback_forms_are_explicit(self):
        common = {
            "selected_uncertainty_estimator": (
                InstrumentChannelCalibrationUncertaintyEstimator
                .SCALE_DEPENDENT_FULL_OBJECTIVE
            ),
            "reference_error_supplied": True,
            "channel_error_supplied": True,
        }
        active = self._provenance(
            **common,
            integration_status=(
                InstrumentChannelCalibrationUncertaintyIntegrationStatus
                .ACTIVE
            ),
            point_estimate_source=(
                "pgmuvi_scale_dependent_full_objective_final_inliers"
            ),
            point_estimate_objective=(
                "gaussian_negative_log_likelihood_"
                "scale_dependent_effective_variance"
            ),
            point_estimate_matches_uncertainty_objective=True,
        )
        self.assertEqual(active.integration_status.value, "active")

        fallback = self._provenance(
            **common,
            integration_status=(
                InstrumentChannelCalibrationUncertaintyIntegrationStatus
                .ATTEMPTED_UNAVAILABLE_FALLBACK
            ),
            point_estimate_source=(
                "pgmuvi_iterative_mad_clipped_affine_fallback_"
                "final_inliers"
            ),
            point_estimate_objective=(
                "iterative_scale_frozen_weighted_least_squares"
            ),
            point_estimate_matches_uncertainty_objective=False,
            reason="Full-objective optimization did not converge.",
        )
        self.assertIn("did not converge", fallback.reason)

        with self.assertRaisesRegex(ValueError, "requires an explicit reason"):
            self._provenance(
                **common,
                integration_status=(
                    InstrumentChannelCalibrationUncertaintyIntegrationStatus
                    .ATTEMPTED_UNAVAILABLE_FALLBACK
                ),
                point_estimate_source=(
                    "pgmuvi_iterative_mad_clipped_affine_fallback_"
                    "final_inliers"
                ),
                point_estimate_objective=(
                    "iterative_scale_frozen_weighted_least_squares"
                ),
                point_estimate_matches_uncertainty_objective=False,
            )

    def test_fitter_owns_finite_filtering_and_final_inlier_identity(self):
        channel_flux = np.arange(7, dtype=float)
        reference_flux = 0.5 + 2.0 * channel_flux
        reference_flux[2] = np.nan
        reference_flux[5] = 100.0

        calibration = fit_instrument_channel_calibration(
            reference_flux,
            channel_flux,
            reference_channel="reference",
            channel="target",
            wavelength=1.0,
        )
        provenance = calibration.fit_provenance

        self.assertIsNotNone(provenance)
        self.assertEqual(provenance.n_input_pairs, 7)
        self.assertEqual(provenance.finite_pair_indices, (0, 1, 3, 4, 5, 6))
        self.assertEqual(provenance.final_inlier_indices, (0, 1, 3, 4, 6))
        self.assertEqual(provenance.n_finite_pairs, calibration.n_pairs)
        self.assertEqual(provenance.n_final_inliers, calibration.n_inliers)
        self.assertIs(
            provenance.integration_status,
            InstrumentChannelCalibrationUncertaintyIntegrationStatus.ACTIVE,
        )
        self.assertTrue(
            provenance.point_estimate_matches_uncertainty_objective
        )

    def test_fixed_weight_fitter_paths_are_active(self):
        channel_flux = np.linspace(0.0, 2.0, 20)
        reference_flux = 0.25 + 1.5 * channel_flux

        for reference_error in (
            None,
            np.linspace(0.02, 0.04, channel_flux.size),
        ):
            with self.subTest(reference_error=reference_error is not None):
                calibration = fit_instrument_channel_calibration(
                    reference_flux,
                    channel_flux,
                    reference_channel="reference",
                    channel="target",
                    wavelength=1.0,
                    reference_error=reference_error,
                )
                provenance = calibration.fit_provenance

                self.assertIs(
                    provenance.selected_uncertainty_estimator,
                    InstrumentChannelCalibrationUncertaintyEstimator
                    .FIXED_WEIGHT_NORMAL_MATRIX,
                )
                self.assertIs(
                    provenance.integration_status,
                    InstrumentChannelCalibrationUncertaintyIntegrationStatus
                    .ACTIVE,
                )
                self.assertEqual(
                    provenance.to_dict()["error_axes"],
                    [] if reference_error is None else ["reference"],
                )
                self.assertTrue(
                    provenance.point_estimate_matches_uncertainty_objective
                )
                self.assertIsNone(provenance.reason)
                self.assertIs(
                    calibration.coefficient_uncertainty.status,
                    InstrumentChannelCalibrationUncertaintyStatus.AVAILABLE,
                )

    def test_channel_errors_activate_full_objective(self):
        channel_flux = np.linspace(0.0, 2.0, 20)
        reference_flux = (
            0.25
            + 1.5 * channel_flux
            + 0.01 * np.sin(np.arange(channel_flux.size))
        )

        for reference_error in (
            None,
            np.full(channel_flux.size, 0.03),
        ):
            with self.subTest(reference_error=reference_error is not None):
                channel_error = np.linspace(
                    0.01,
                    0.02,
                    channel_flux.size,
                )
                calibration = fit_instrument_channel_calibration(
                    reference_flux,
                    channel_flux,
                    reference_channel="reference",
                    channel="target",
                    wavelength=1.0,
                    reference_error=reference_error,
                    channel_error=channel_error,
                )
                provenance = calibration.fit_provenance
                uncertainty = calibration.coefficient_uncertainty

                self.assertIs(
                    provenance.selected_uncertainty_estimator,
                    InstrumentChannelCalibrationUncertaintyEstimator
                    .SCALE_DEPENDENT_FULL_OBJECTIVE,
                )
                self.assertIs(
                    provenance.integration_status,
                    InstrumentChannelCalibrationUncertaintyIntegrationStatus
                    .ACTIVE,
                )
                self.assertTrue(
                    provenance.point_estimate_matches_uncertainty_objective
                )
                self.assertEqual(
                    provenance.point_estimate_source,
                    "pgmuvi_scale_dependent_full_objective_final_inliers",
                )
                self.assertEqual(
                    provenance.point_estimate_objective,
                    "gaussian_negative_log_likelihood_"
                    "scale_dependent_effective_variance",
                )
                self.assertIsNone(provenance.reason)
                self.assertIs(
                    uncertainty.status,
                    InstrumentChannelCalibrationUncertaintyStatus.AVAILABLE,
                )
                self.assertIsNotNone(uncertainty.coefficient_covariance)

                direct = (
                    estimate_scale_dependent_instrument_channel_calibration_coefficient_uncertainty(
                        reference_flux,
                        channel_flux,
                        reference_error=reference_error,
                        channel_error=channel_error,
                        initial_offset=0.25,
                        initial_scale=1.5,
                    )
                )
                self.assertIs(
                    direct.status,
                    InstrumentChannelCalibrationUncertaintyStatus.AVAILABLE,
                )
                np.testing.assert_allclose(
                    (calibration.offset, calibration.scale),
                    (direct.offset, direct.scale),
                    rtol=1.0e-8,
                    atol=1.0e-10,
                )
                np.testing.assert_allclose(
                    uncertainty.coefficient_covariance,
                    direct.coefficient_covariance,
                    rtol=1.0e-7,
                    atol=1.0e-12,
                )

    def test_optimizer_failure_retains_explicit_affine_fallback(self):
        channel_flux = np.linspace(0.0, 2.0, 20)
        reference_flux = 0.25 + 1.5 * channel_flux
        failure = OptimizeResult(
            success=False,
            message="forced integration failure",
            x=np.asarray([0.25, 0.0]),
        )

        with patch.object(
            instrument_channel_calibration.optimize,
            "minimize",
            return_value=failure,
        ):
            calibration = fit_instrument_channel_calibration(
                reference_flux,
                channel_flux,
                reference_channel="reference",
                channel="target",
                wavelength=1.0,
                reference_error=np.full(channel_flux.size, 0.03),
                channel_error=np.linspace(
                    0.01,
                    0.02,
                    channel_flux.size,
                ),
            )

        provenance = calibration.fit_provenance
        uncertainty = calibration.coefficient_uncertainty
        self.assertIs(
            provenance.integration_status,
            InstrumentChannelCalibrationUncertaintyIntegrationStatus
            .ATTEMPTED_UNAVAILABLE_FALLBACK,
        )
        self.assertFalse(
            provenance.point_estimate_matches_uncertainty_objective
        )
        self.assertIn("forced integration failure", provenance.reason)
        self.assertIs(
            uncertainty.status,
            InstrumentChannelCalibrationUncertaintyStatus.UNAVAILABLE,
        )
        self.assertIsNone(uncertainty.coefficient_covariance)
        self.assertIn("forced integration failure", uncertainty.reason)
        self.assertAlmostEqual(calibration.offset, 0.25, places=12)
        self.assertAlmostEqual(calibration.scale, 1.5, places=12)
        json.dumps(calibration.to_dict(), allow_nan=False)

    def test_model_count_consistency_is_enforced(self):
        fitted = fit_instrument_channel_calibration(
            np.asarray([1.0, 3.0, 5.0, 7.0]),
            np.asarray([0.0, 1.0, 2.0, 3.0]),
            reference_channel="reference",
            channel="target",
            wavelength=1.0,
        )

        with self.assertRaisesRegex(ValueError, "finite-pair count"):
            InstrumentChannelCalibration(
                schema_version=(
                    INSTRUMENT_CHANNEL_CALIBRATION_MODEL_SCHEMA_VERSION
                ),
                reference_channel=fitted.reference_channel,
                channel=fitted.channel,
                wavelength=fitted.wavelength,
                offset=fitted.offset,
                scale=fitted.scale,
                n_pairs=fitted.n_pairs + 1,
                n_inliers=fitted.n_inliers,
                residual_mad_sigma=fitted.residual_mad_sigma,
                coefficient_uncertainty=fitted.coefficient_uncertainty,
                fit_provenance=fitted.fit_provenance,
            )

    def test_manual_calibration_without_fit_provenance_remains_valid(self):
        fitted = fit_instrument_channel_calibration(
            np.asarray([1.0, 3.0, 5.0, 7.0]),
            np.asarray([0.0, 1.0, 2.0, 3.0]),
            reference_channel="reference",
            channel="target",
            wavelength=1.0,
        )
        manual = InstrumentChannelCalibration(
            schema_version=(
                INSTRUMENT_CHANNEL_CALIBRATION_MODEL_SCHEMA_VERSION
            ),
            reference_channel=fitted.reference_channel,
            channel=fitted.channel,
            wavelength=fitted.wavelength,
            offset=fitted.offset,
            scale=fitted.scale,
            n_pairs=fitted.n_pairs,
            n_inliers=fitted.n_inliers,
            residual_mad_sigma=fitted.residual_mad_sigma,
            coefficient_uncertainty=fitted.coefficient_uncertainty,
        )

        self.assertIsNone(manual.fit_provenance)
        self.assertIsNone(manual.to_dict()["fit_provenance"])
        json.dumps(manual.to_dict(), allow_nan=False)

    def test_orchestration_preserves_separate_same_wavelength_provenance(self):
        times = np.tile(np.arange(4, dtype=float), 3)
        channels = np.repeat(
            np.asarray(["A", "B", "C"], dtype=object),
            4,
        )
        wavelengths = np.ones(12, dtype=float)
        flux = np.concatenate(
            (
                np.asarray([1.0, 3.01, 4.99, 7.0]),
                np.asarray([0.0, 1.0, 2.0, 3.0]),
                np.asarray([0.5, 1.5, 2.5, 3.5]),
            )
        )
        error = np.full(12, 0.1)
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
                    channel_plans=tuple(
                        InstrumentChannelCalibrationChannelPlan(
                            channel=channel,
                            disposition=(
                                InstrumentChannelCalibrationDisposition.PLANNED
                            ),
                            pairing_method=(
                                InstrumentChannelPairingMethod.EXACT_TIMESTAMP
                            ),
                            time_unit="day",
                            calibration_family="affine",
                        )
                        for channel in ("B", "C")
                    ),
                ),
            ),
        )

        execution = execute_instrument_channel_calibration_plan(
            plan,
            times,
            flux,
            wavelengths,
            channels,
            flux_error=error,
        )
        completed = execution.plan.group_plans[0].channel_plans

        self.assertEqual(
            tuple(item.channel for item in completed),
            ("B", "C"),
        )
        self.assertTrue(
            all(item.calibration is not None for item in completed)
        )
        self.assertTrue(
            all(
                item.calibration.fit_provenance.integration_status
                is InstrumentChannelCalibrationUncertaintyIntegrationStatus
                .ACTIVE
                for item in completed
            )
        )
        self.assertTrue(
            all(
                item.calibration.coefficient_uncertainty.status
                is InstrumentChannelCalibrationUncertaintyStatus.AVAILABLE
                for item in completed
            )
        )
        payload = execution.to_dict()
        serialized_channels = payload["plan"]["group_plans"][0][
            "channel_plans"
        ]
        self.assertEqual(
            [item["channel"] for item in serialized_channels],
            ["B", "C"],
        )
        self.assertTrue(
            all(
                item["calibration"]["fit_provenance"]
                for item in serialized_channels
            )
        )
        json.dumps(payload, allow_nan=False)

    def test_predictive_boundary_uses_activated_covariance(self):
        channel_flux = np.linspace(0.0, 2.0, 20)
        calibration = fit_instrument_channel_calibration(
            (
                0.25
                + 1.5 * channel_flux
                + 0.01 * np.sin(np.arange(channel_flux.size))
            ),
            channel_flux,
            reference_channel="reference",
            channel="target",
            wavelength=1.0,
            channel_error=np.linspace(
                0.01,
                0.02,
                channel_flux.size,
            ),
        )

        _, predictive = (
            apply_instrument_channel_calibration_with_predictive_uncertainty(
                np.asarray([0.5, 1.0]),
                calibration,
                flux_error=np.asarray([0.1, 0.1]),
                covariance_mode=(
                    InstrumentChannelCalibrationPredictiveCovarianceMode
                    .MARGINAL_VARIANCE
                ),
            )
        )

        self.assertIs(
            predictive.status,
            InstrumentChannelCalibrationPredictiveUncertaintyStatus.AVAILABLE,
        )
        self.assertIsNotNone(predictive.measurement_variance)
        self.assertIsNotNone(predictive.predictive_variance)
        self.assertEqual(
            predictive.coefficient_uncertainty_source,
            calibration.coefficient_uncertainty.uncertainty_source,
        )


if __name__ == "__main__":
    unittest.main()
