"""Tests for explicit wavelength-model scientific-hypothesis metadata."""

import json
import unittest

from pgmuvi.wavelength_hypotheses import (
    LPV_WAVELENGTH_MODEL_PRIORITY,
    WAVELENGTH_HYPOTHESIS_SCHEMA_VERSION,
    WavelengthCovarianceStructure,
    WavelengthHypothesisRole,
    WavelengthMeanStructure,
    WavelengthModelHypothesis,
    describe_wavelength_model_hypothesis,
)


class TestWavelengthModelHypothesisTaxonomy(unittest.TestCase):
    def test_lpv_priority_is_explicit_and_excludes_achromatic_control(self):
        self.assertEqual(
            LPV_WAVELENGTH_MODEL_PRIORITY,
            (
                "2DWavelengthDependent",
                "2DDustMean",
                "2DPowerLawMean",
                "2DSeparable",
                "2D",
            ),
        )
        self.assertNotIn("2DAchromatic", LPV_WAVELENGTH_MODEL_PRIORITY)

    def test_2d_is_joint_nonseparable_baseline(self):
        hypothesis = describe_wavelength_model_hypothesis("2D")
        self.assertEqual(hypothesis.role, WavelengthHypothesisRole.JOINT_BASELINE)
        self.assertEqual(
            hypothesis.covariance_structure,
            WavelengthCovarianceStructure.JOINT_2D_SPECTRAL_MIXTURE,
        )
        self.assertEqual(hypothesis.mean_structure, WavelengthMeanStructure.CONSTANT)
        self.assertFalse(hypothesis.covariance_separable)
        self.assertTrue(hypothesis.baseline)

    def test_dust_and_power_law_are_not_mean_only_hypotheses(self):
        expected = {
            "2DDustMean": WavelengthMeanStructure.DUST_ATTENUATION,
            "2DPowerLawMean": WavelengthMeanStructure.POWER_LAW,
        }
        for model, mean_structure in expected.items():
            with self.subTest(model=model):
                hypothesis = describe_wavelength_model_hypothesis(model)
                self.assertEqual(
                    hypothesis.role,
                    WavelengthHypothesisRole.MEAN_AND_COVARIANCE,
                )
                self.assertEqual(hypothesis.mean_structure, mean_structure)
                self.assertEqual(
                    hypothesis.covariance_structure,
                    WavelengthCovarianceStructure.SEPARABLE_SMOOTH_WAVELENGTH,
                )
                self.assertTrue(hypothesis.mean_wavelength_dependent)
                self.assertTrue(hypothesis.covariance_wavelength_dependent)
                self.assertTrue(
                    any(
                        "not a mean-only hypothesis" in caution.lower()
                        for caution in hypothesis.comparison_cautions
                    )
                )

    def test_wavelength_dependent_model_records_both_axes(self):
        hypothesis = describe_wavelength_model_hypothesis(
            "2DWavelengthDependent"
        )
        self.assertEqual(hypothesis.mean_structure, WavelengthMeanStructure.QUADRATIC)
        self.assertEqual(
            hypothesis.covariance_structure,
            WavelengthCovarianceStructure.SEPARABLE_SMOOTH_WAVELENGTH,
        )
        self.assertTrue(hypothesis.mean_wavelength_dependent)
        self.assertTrue(hypothesis.covariance_wavelength_dependent)

    def test_mean_override_refines_configurable_model_without_mutating_input(self):
        fit_kwargs = {"mean_module": "constant", "training_iter": 20}
        hypothesis = describe_wavelength_model_hypothesis(
            "2DWavelengthDependent", fit_kwargs=fit_kwargs
        )
        self.assertEqual(hypothesis.mean_structure, WavelengthMeanStructure.CONSTANT)
        self.assertFalse(hypothesis.mean_wavelength_dependent)
        self.assertEqual(fit_kwargs, {"mean_module": "constant", "training_iter": 20})
        self.assertEqual(hypothesis.metadata["mean_module_override"], "constant")

    def test_fixed_model_means_are_not_overridden_by_generic_fit_metadata(self):
        expected = {
            "2DAchromatic": WavelengthMeanStructure.CONSTANT,
            "2DDustMean": WavelengthMeanStructure.DUST_ATTENUATION,
            "2DPowerLawMean": WavelengthMeanStructure.POWER_LAW,
        }
        for model, mean_structure in expected.items():
            with self.subTest(model=model):
                hypothesis = describe_wavelength_model_hypothesis(
                    model, fit_kwargs={"mean_module": "linear"}
                )
                self.assertEqual(hypothesis.mean_structure, mean_structure)

    def test_achromatic_is_control_without_lpv_priority(self):
        hypothesis = describe_wavelength_model_hypothesis("2DAchromatic")
        self.assertEqual(
            hypothesis.role,
            WavelengthHypothesisRole.ACHROMATIC_CONTROL,
        )
        self.assertFalse(hypothesis.covariance_wavelength_dependent)
        self.assertIsNone(hypothesis.lpv_advisory_priority)

    def test_unknown_model_is_explicitly_unknown(self):
        hypothesis = describe_wavelength_model_hypothesis("FutureModel")
        self.assertEqual(hypothesis.role, WavelengthHypothesisRole.UNKNOWN)
        self.assertEqual(
            hypothesis.mean_structure,
            WavelengthMeanStructure.UNKNOWN,
        )
        self.assertIsNone(hypothesis.mean_wavelength_dependent)
        self.assertIn("unknown", hypothesis.comparison_cautions[0].lower())

    def test_mapping_round_trip_is_versioned_and_json_safe(self):
        source = describe_wavelength_model_hypothesis("2DDustMean")
        payload = source.to_dict()
        self.assertEqual(
            payload["schema_version"], WAVELENGTH_HYPOTHESIS_SCHEMA_VERSION
        )
        restored = WavelengthModelHypothesis.from_mapping(payload)
        self.assertEqual(restored, source)
        json.dumps(payload)

    def test_arbitrary_metadata_is_sanitized_for_json(self):
        source = WavelengthModelHypothesis(
            model="FutureModel",
            role=WavelengthHypothesisRole.UNKNOWN,
            mean_structure=WavelengthMeanStructure.UNKNOWN,
            covariance_structure=WavelengthCovarianceStructure.UNKNOWN,
            mean_wavelength_dependent=None,
            covariance_wavelength_dependent=None,
            covariance_separable=None,
            temporal_kernel_configurable=None,
            metadata={
                "enum": WavelengthMeanStructure.LINEAR,
                "nonfinite": float("nan"),
                "nested": (1, object()),
            },
        )
        payload = source.to_dict()
        self.assertEqual(payload["metadata"]["enum"], "linear")
        self.assertIsNone(payload["metadata"]["nonfinite"])
        self.assertIsInstance(payload["metadata"]["nested"], list)
        json.dumps(payload, allow_nan=False)

    def test_blank_model_name_uses_explicit_unknown_label(self):
        hypothesis = describe_wavelength_model_hypothesis("")
        self.assertEqual(hypothesis.model, "unknown")
        self.assertEqual(hypothesis.role, WavelengthHypothesisRole.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
