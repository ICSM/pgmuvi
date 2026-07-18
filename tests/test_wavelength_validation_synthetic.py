import importlib.util
import json
import math
import unittest

import numpy as np

from pgmuvi.wavelength_validation import (
    WavelengthValidationPhase,
    WavelengthValidationSourceKind,
)
from pgmuvi.wavelength_validation_synthetic import (
    DEFAULT_VALIDATION_CANDIDATES,
    SyntheticWavelengthCovarianceKind,
    SyntheticWavelengthDependenceStrength,
    SyntheticWavelengthMeanKind,
    SyntheticWavelengthValidationCase,
    canonical_synthetic_wavelength_validation_cases,
    make_synthetic_wavelength_validation_case,
)


class SyntheticCaseMixin:
    def make_case(self, **overrides):
        kwargs = {
            "scenario_id": "d1-test-case",
            "generating_model": "2DWavelengthDependent",
            "mean_kind": "quadratic",
            "covariance_kind": "separable_quasi_periodic_rbf",
            "strength": "moderate",
            "wavelengths": (0.6, 1.0, 2.0),
            "band_labels": ("r", "J", "K"),
            "n_per_band": 8,
            "period": 100.0,
            "n_cycles": 3.0,
            "noise_sigma": 0.05,
            "seed": 7,
        }
        kwargs.update(overrides)
        return make_synthetic_wavelength_validation_case(**kwargs)


class TestSyntheticWavelengthValidationCase(SyntheticCaseMixin, unittest.TestCase):
    def test_case_contains_aligned_truth_preserving_arrays(self):
        case = self.make_case()

        self.assertEqual(len(case.time_values), 24)
        self.assertEqual(len(case.wavelength_values), 24)
        self.assertEqual(len(case.band_labels), 24)
        self.assertEqual(len(case.uncertainties), 24)
        reconstructed = np.asarray(case.noiseless_mean) + np.asarray(
            case.latent_process
        )
        np.testing.assert_allclose(reconstructed, case.noiseless_flux)
        self.assertGreater(min(case.observed_flux), 0.0)

    def test_scenario_records_d1_truth_and_advisory_candidate_set(self):
        case = self.make_case()
        scenario = case.scenario

        self.assertEqual(
            scenario.phase, WavelengthValidationPhase.D1_SYNTHETIC_RECOVERY
        )
        self.assertEqual(
            scenario.source_kind, WavelengthValidationSourceKind.SYNTHETIC
        )
        self.assertTrue(scenario.advisory_only)
        self.assertEqual(
            tuple(scenario.fit_configuration["candidate_models"]),
            DEFAULT_VALIDATION_CANDIDATES,
        )
        self.assertNotIn("selected_model", scenario.to_dict())

    def test_case_round_trip_is_json_safe_and_preserves_unknown_fields(self):
        case = self.make_case()
        payload = case.to_dict()
        payload["future_case_field"] = {"retained": True}
        rebuilt = SyntheticWavelengthValidationCase.from_mapping(payload)

        self.assertEqual(
            rebuilt.extra_fields["future_case_field"], {"retained": True}
        )
        self.assertEqual(rebuilt.to_dict(), {
            **case.to_dict(),
            "extra_fields": {"future_case_field": {"retained": True}},
        })
        json.dumps(rebuilt.to_dict(), allow_nan=False)

    def test_same_master_seed_is_reproducible(self):
        first = self.make_case(seed=31)
        second = self.make_case(seed=31)

        self.assertEqual(first.time_values, second.time_values)
        self.assertEqual(first.latent_process, second.latent_process)
        self.assertEqual(first.observed_flux, second.observed_flux)
        self.assertEqual(
            first.scenario.truth.latent_parameters,
            second.scenario.truth.latent_parameters,
        )

    def test_different_master_seed_changes_observations(self):
        first = self.make_case(seed=31)
        second = self.make_case(seed=32)

        self.assertNotEqual(first.time_values, second.time_values)
        self.assertNotEqual(first.observed_flux, second.observed_flux)

    def test_purpose_specific_seeds_are_recorded_separately(self):
        case = self.make_case(
            seed=None,
            sampling_seed=11,
            process_seed=22,
            noise_seed=33,
        )
        seeds = case.scenario.sampling_configuration["purpose_specific_seeds"]

        self.assertEqual(seeds["sampling"], 11)
        self.assertEqual(seeds["process"], 22)
        self.assertEqual(seeds["noise"], 33)

    def test_minmax_coordinate_truth_round_trips_physical_wavelength(self):
        case = self.make_case(xtransform="minmax")
        truth = case.scenario.truth
        record = truth.coordinate_transforms["wavelength"]
        model_values = truth.noiseless_summary["model_wavelength_by_band"]
        reconstructed = [
            record["origin"] + record["scale"] * item for item in model_values
        ]

        np.testing.assert_allclose(
            reconstructed, truth.physical_wavelengths, rtol=0.0, atol=1e-12
        )

    def test_identity_coordinate_truth_is_explicit(self):
        case = self.make_case(xtransform=None)
        transforms = case.scenario.truth.coordinate_transforms

        self.assertEqual(transforms["kind"], "identity")
        self.assertEqual(
            case.lightcurve_configuration["center_time"], False
        )

    def test_parameter_ownership_separates_ard_dimensions(self):
        case = self.make_case(
            generating_model="2D",
            mean_kind="constant",
            covariance_kind="joint_spectral_mixture_ard",
        )
        ownership = case.scenario.truth.parameter_ownership

        self.assertEqual(ownership["temporal_frequency_ard_index"], 0)
        self.assertEqual(ownership["wavelength_frequency_ard_index"], 1)
        self.assertIn(
            "covar_module.mixture_scales", ownership["covariance_parameters"]
        )

    def test_compact_mapping_can_omit_observation_arrays(self):
        case = self.make_case()
        payload = case.to_dict(include_observations=False)

        self.assertEqual(payload["n_observations"], 24)
        self.assertNotIn("observed_flux", payload)
        self.assertNotIn("latent_process", payload)

    @unittest.skipUnless(
        importlib.util.find_spec("gpytorch") is not None,
        "gpytorch is required to construct a Lightcurve",
    )
    def test_case_builds_double_precision_lightcurve_with_band_labels(self):
        case = self.make_case()
        lightcurve = case.to_lightcurve()

        self.assertEqual(lightcurve.ndim, 2)
        self.assertEqual(lightcurve.xdata.shape[0], 24)
        self.assertEqual(str(lightcurve.xdata.dtype), "torch.float64")
        self.assertEqual(set(lightcurve.band), {"r", "J", "K"})


class TestSyntheticMeanFamilies(SyntheticCaseMixin, unittest.TestCase):
    def test_quadratic_truth_uses_model_coordinate_parameters(self):
        case = self.make_case(
            mean_parameters={"bias": 9.0, "weights": [2.0, -1.0]},
            xtransform="minmax",
        )
        truth = case.scenario.truth.wavelength_mean_parameters

        self.assertEqual(truth["model_parameter_values"]["mean_module.bias"], 9.0)
        self.assertEqual(
            truth["model_parameter_values"]["mean_module.weights"],
            [2.0, -1.0],
        )

    def test_dust_truth_records_physical_and_logged_parameters(self):
        case = self.make_case(
            generating_model="2DDustMean",
            mean_kind=SyntheticWavelengthMeanKind.DUST,
            mean_parameters={
                "offset": 4.0,
                "amplitude": 12.0,
                "tau": 0.7,
                "alpha": 1.5,
            },
        )
        truth = case.scenario.truth.wavelength_mean_parameters

        self.assertEqual(truth["tau"], 0.7)
        self.assertAlmostEqual(
            truth["model_parameter_values"]["mean_module.log_tau"],
            math.log(0.7),
        )
        self.assertTrue(truth["linear_flux"])

    def test_power_law_truth_preserves_signed_weight(self):
        case = self.make_case(
            generating_model="2DPowerLawMean",
            mean_kind="power_law",
            mean_parameters={"offset": 20.0, "weight": -1.0, "exponent": 0.5},
        )
        truth = case.scenario.truth.wavelength_mean_parameters

        self.assertEqual(truth["weight"], -1.0)
        self.assertEqual(
            truth["model_parameter_values"]["mean_module.weight"], -1.0
        )

    def test_turning_point_case_has_interior_quadratic_extremum(self):
        case = self.make_case(turning_point=True, strength="strong")
        parameters = case.scenario.truth.wavelength_mean_parameters
        linear, quadratic = parameters["weights"]
        turning_point = -linear / (2.0 * quadratic)

        self.assertGreater(turning_point, 0.0)
        self.assertLess(turning_point, 1.0)
        self.assertTrue(case.scenario.truth.metadata["turning_point"])


class TestSyntheticCovarianceFamilies(SyntheticCaseMixin, unittest.TestCase):
    def test_separable_covariance_records_physical_lengthscale(self):
        case = self.make_case(
            covariance_parameters={
                "wavelength_lengthscale": 0.4,
                "signal_std": 0.8,
            }
        )
        truth = case.scenario.truth.wavelength_covariance_parameters

        self.assertEqual(
            truth["covariance_kind"],
            SyntheticWavelengthCovarianceKind.SEPARABLE_QUASI_PERIODIC_RBF.value,
        )
        self.assertEqual(truth["wavelength_lengthscale"], 0.4)
        self.assertEqual(truth["wavelength_lengthscale_coordinate"], "physical")
        self.assertEqual(truth["signal_std"], 0.8)

    def test_joint_sm_truth_has_gpytorch_ard_shapes(self):
        case = self.make_case(
            generating_model="2D",
            mean_kind="constant",
            covariance_kind="joint_spectral_mixture_ard",
        )
        truth = case.scenario.truth.wavelength_covariance_parameters

        self.assertEqual(truth["coordinate_order"], [
            "temporal_frequency",
            "wavelength_frequency",
        ])
        self.assertEqual(np.asarray(truth["mixture_means"]).shape, (1, 1, 2))
        self.assertEqual(np.asarray(truth["mixture_scales"]).shape, (1, 1, 2))
        self.assertEqual(np.asarray(truth["mixture_weights"]).shape, (1,))

    def test_known_harmonic_is_recorded_as_harmonic(self):
        case = self.make_case(
            temporal_components=(
                {"period": 100.0, "variance_fraction": 0.8},
                {"period": 50.0, "variance_fraction": 0.2},
            )
        )
        temporal = case.scenario.truth.temporal_parameters

        self.assertTrue(temporal["fundamental_plus_harmonic"])
        self.assertEqual(len(temporal["components"]), 2)
        self.assertAlmostEqual(
            sum(item["variance_fraction"] for item in temporal["components"]),
            1.0,
        )

    def test_strength_changes_default_wavelength_lengthscale(self):
        weak = self.make_case(strength=SyntheticWavelengthDependenceStrength.WEAK)
        strong = self.make_case(
            strength=SyntheticWavelengthDependenceStrength.STRONG
        )
        weak_scale = weak.scenario.truth.wavelength_covariance_parameters[
            "wavelength_lengthscale"
        ]
        strong_scale = strong.scenario.truth.wavelength_covariance_parameters[
            "wavelength_lengthscale"
        ]

        self.assertGreater(weak_scale, strong_scale)


class TestCanonicalSyntheticCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = canonical_synthetic_wavelength_validation_cases(seed=101)

    def test_canonical_set_has_unique_ids_and_all_target_models(self):
        identifiers = [case.scenario.scenario_id for case in self.cases]
        models = {
            case.scenario.truth.generating_model for case in self.cases
        }

        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(models, set(DEFAULT_VALIDATION_CANDIDATES))

    def test_canonical_set_contains_turning_point_and_harmonic_cases(self):
        metadata = [case.scenario.truth.metadata for case in self.cases]
        temporal = [case.scenario.truth.temporal_parameters for case in self.cases]

        self.assertTrue(any(item["turning_point"] for item in metadata))
        self.assertTrue(
            any(item["fundamental_plus_harmonic"] for item in temporal)
        )

    def test_canonical_set_is_positive_linear_flux(self):
        for case in self.cases:
            with self.subTest(case=case.scenario.scenario_id):
                self.assertGreater(min(case.observed_flux), 0.0)
                self.assertTrue(
                    case.scenario.truth.noiseless_summary["linear_flux"]
                )


class TestSyntheticInputValidation(SyntheticCaseMixin, unittest.TestCase):
    def test_invalid_enum_values_raise(self):
        with self.assertRaisesRegex(ValueError, "mean_kind"):
            self.make_case(mean_kind="not-a-mean")
        with self.assertRaisesRegex(ValueError, "covariance_kind"):
            self.make_case(covariance_kind="not-a-covariance")
        with self.assertRaisesRegex(ValueError, "strength"):
            self.make_case(strength="not-a-strength")

    def test_wavelengths_must_be_positive_unique_and_increasing(self):
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            self.make_case(wavelengths=(0.6, -1.0, 2.0))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.make_case(wavelengths=(0.6, 0.6, 2.0))

    def test_band_counts_and_labels_must_match(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            self.make_case(band_labels=("r", "J"))
        with self.assertRaisesRegex(ValueError, "match the number of bands"):
            self.make_case(n_per_band=[4, 5, 6, 7])

    def test_all_purpose_specific_seeds_are_required_without_master(self):
        with self.assertRaisesRegex(ValueError, "master seed"):
            self.make_case(seed=None, sampling_seed=1, process_seed=2)

    def test_only_supported_pr127_transforms_are_accepted(self):
        with self.assertRaisesRegex(ValueError, "support xtransform"):
            self.make_case(xtransform="zscore")


if __name__ == "__main__":
    unittest.main()
