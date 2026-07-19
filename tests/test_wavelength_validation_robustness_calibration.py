import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from pgmuvi.wavelength_validation_robustness import (
    SyntheticWavelengthRobustnessAxis,
    SyntheticWavelengthRobustnessSeverity,
    canonical_synthetic_wavelength_robustness_specifications,
    make_synthetic_wavelength_robustness_case,
    run_synthetic_wavelength_robustness,
)
from pgmuvi.wavelength_validation_robustness_calibration import (
    SyntheticWavelengthRobustnessBoundaryClass,
    SyntheticWavelengthRobustnessBoundaryRecord,
    SyntheticWavelengthRobustnessCalibrationReport,
    SyntheticWavelengthRobustnessCalibrationThresholds,
    calibrate_synthetic_wavelength_robustness_runs,
    run_synthetic_wavelength_robustness_population,
    summarize_synthetic_wavelength_robustness_population,
    validate_synthetic_wavelength_robustness_reference_pairs,
)


class CalibrationMixin:
    @staticmethod
    def lightcurve_factory(case):
        return object()

    @staticmethod
    def successful_output(lightcurve, configuration, case):
        truth = case.scenario.truth
        return {
            "fitted_period": truth.temporal_parameters["fundamental_period"],
            "fitted_wavelength_lengthscale": (
                truth.wavelength_covariance_parameters["wavelength_lengthscale"]
            ),
            "fitted_mean_by_band": truth.noiseless_summary["mean_by_band"],
            "boundary_hits": [],
            "sm_ard_diagnostics": {
                "available": False,
                "parameters": {},
                "boundary_hits": [],
            },
            "diagnostics": {"source": "calibration-test-hook"},
            "parameter_workflow": {},
        }

    @staticmethod
    def reference_and_independent_specifications():
        specifications = canonical_synthetic_wavelength_robustness_specifications()
        identifiers = {
            "d2-reference-separable-moderate",
            "d2-independent-time-grids",
        }
        return tuple(
            item for item in specifications if item.scenario_id in identifiers
        )


class TestReferencePairContracts(CalibrationMixin, unittest.TestCase):
    def test_canonical_pairs_are_truth_matched_and_use_multiple_references(self):
        specifications = canonical_synthetic_wavelength_robustness_specifications()
        validation = validate_synthetic_wavelength_robustness_reference_pairs(
            specifications
        )

        self.assertTrue(validation["valid"], validation["issues"])
        self.assertGreater(validation["n_references"], 1)
        by_id = {item.scenario_id: item for item in specifications}
        for item in specifications:
            if item.axis is SyntheticWavelengthRobustnessAxis.REFERENCE:
                self.assertIsNone(item.reference_scenario_id)
            else:
                self.assertIn(item.reference_scenario_id, by_id)
                self.assertIs(
                    by_id[item.reference_scenario_id].axis,
                    SyntheticWavelengthRobustnessAxis.REFERENCE,
                )

    def test_mismatched_model_family_is_rejected(self):
        specifications = canonical_synthetic_wavelength_robustness_specifications()
        dust = next(
            item
            for item in specifications
            if item.scenario_id == "d2-uneven-band-counts"
        )
        mismatched = replace(
            dust,
            reference_scenario_id="d2-reference-separable-moderate",
        )
        selected = tuple(
            item
            for item in specifications
            if item.scenario_id
            in {
                "d2-reference-separable-moderate",
                mismatched.scenario_id,
            }
        )
        selected = tuple(
            mismatched if item.scenario_id == mismatched.scenario_id else item
            for item in selected
        )

        validation = validate_synthetic_wavelength_robustness_reference_pairs(
            selected
        )

        self.assertFalse(validation["valid"])
        self.assertEqual(validation["issues"][0]["code"], "truth_family_mismatch")
        self.assertIn(
            "generating_model", validation["issues"][0]["differences"]
        )

    def test_strength_difference_is_allowed_for_strength_axis(self):
        specifications = canonical_synthetic_wavelength_robustness_specifications()
        selected = tuple(
            item
            for item in specifications
            if item.scenario_id
            in {
                "d2-reference-dust-moderate",
                "d2-strong-wavelength-dependence",
            }
        )

        validation = validate_synthetic_wavelength_robustness_reference_pairs(
            selected
        )

        self.assertTrue(validation["valid"], validation["issues"])


class TestCalibrationThresholds(unittest.TestCase):
    def test_round_trip_is_json_safe(self):
        thresholds = SyntheticWavelengthRobustnessCalibrationThresholds(
            minimum_seed_count=4,
            robust_completion_drop_maximum=0.12,
        )
        rebuilt = SyntheticWavelengthRobustnessCalibrationThresholds.from_mapping(
            thresholds.to_dict()
        )

        self.assertEqual(rebuilt, thresholds)
        json.dumps(rebuilt.to_dict(), allow_nan=False)

    def test_robust_envelope_cannot_exceed_degraded_envelope(self):
        with self.assertRaisesRegex(ValueError, "degraded limit"):
            SyntheticWavelengthRobustnessCalibrationThresholds(
                robust_completion_drop_maximum=0.30,
                degraded_completion_drop_maximum=0.20,
            )


class TestBoundaryPressureExtraction(CalibrationMixin, unittest.TestCase):
    def test_separable_physical_lengthscale_is_compared_with_effective_bounds(self):
        specification = self.reference_and_independent_specifications()[0]
        case = make_synthetic_wavelength_robustness_case(specification, seed=5)

        def fit_runner(lightcurve, configuration, current_case):
            output = self.successful_output(lightcurve, configuration, current_case)
            output["fitted_wavelength_lengthscale"] = 7.8
            output["parameter_workflow"] = {
                "report": {
                    "applied": [
                        {
                            "parameter": (
                                "covar_module.kernels.1.base_kernel.lengthscale"
                            ),
                            "wavelength_estimate_provenance": {
                                "effective_constraint": [0.10, 2.0],
                                "diagnostics": {
                                    "raw_recommended_lengthscale_initial": 4.0,
                                    "model_recommended_lengthscale_initial": 1.0,
                                },
                            },
                        }
                    ]
                }
            }
            return output

        run = run_synthetic_wavelength_robustness(
            case,
            lightcurve_factory=self.lightcurve_factory,
            fit_runner=fit_runner,
        )
        summary = summarize_synthetic_wavelength_robustness_population(
            [run],
            thresholds={"minimum_seed_count": 1},
        )[case.scenario.scenario_id]
        boundary = summary["boundary_summary"]

        self.assertEqual(boundary["n_available"], 1)
        self.assertEqual(boundary["n_wavelength_near_bound"], 1)
        self.assertEqual(boundary["n_wavelength_at_bound"], 0)
        self.assertAlmostEqual(
            boundary["minimum_wavelength_normalized_distance"],
            (8.0 - 7.8) / (8.0 - 0.4),
        )

    def test_joint_sm_dimensions_are_summarized_separately(self):
        specification = next(
            item
            for item in canonical_synthetic_wavelength_robustness_specifications()
            if item.scenario_id == "d2-reference-joint-sm-ard-moderate"
        )
        case = make_synthetic_wavelength_robustness_case(specification, seed=6)

        def component(dimension_name, value):
            return {
                "component_index": 0,
                "dimension_name": dimension_name,
                "raw_input_coordinate_value": value,
                "raw_input_coordinate_lower_bound": 0.0,
                "raw_input_coordinate_upper_bound": 1.0,
            }

        def fit_runner(lightcurve, configuration, current_case):
            output = self.successful_output(lightcurve, configuration, current_case)
            output["sm_ard_diagnostics"] = {
                "available": True,
                "parameters": {
                    "mixture_means": {
                        "component_diagnostics": [
                            component("temporal_frequency", 0.50),
                            component("wavelength_frequency", 0.01),
                        ]
                    }
                },
                "boundary_hits": [],
            }
            return output

        run = run_synthetic_wavelength_robustness(
            case,
            lightcurve_factory=self.lightcurve_factory,
            fit_runner=fit_runner,
        )
        summary = summarize_synthetic_wavelength_robustness_population(
            [run],
            thresholds={"minimum_seed_count": 1},
        )[case.scenario.scenario_id]
        boundary = summary["boundary_summary"]

        self.assertEqual(boundary["n_wavelength_near_bound"], 1)
        self.assertEqual(boundary["n_temporal_near_bound"], 0)
        self.assertAlmostEqual(
            boundary["minimum_wavelength_normalized_distance"], 0.01
        )
        self.assertAlmostEqual(
            boundary["minimum_temporal_normalized_distance"], 0.50
        )


class TestPopulationCalibration(CalibrationMixin, unittest.TestCase):
    def test_population_reuses_each_base_seed_across_reference_pair(self):
        seen = []

        def fit_runner(lightcurve, configuration, case):
            seen.append(
                (
                    case.scenario.scenario_id,
                    case.scenario.sampling_configuration[
                        "purpose_specific_seeds"
                    ]["master"],
                )
            )
            return self.successful_output(lightcurve, configuration, case)

        report = run_synthetic_wavelength_robustness_population(
            base_seeds=(2, 5),
            specifications=self.reference_and_independent_specifications(),
            lightcurve_factory=self.lightcurve_factory,
            fit_runner=fit_runner,
            calibration_thresholds={"minimum_seed_count": 2},
            optimizer_seed_base=None,
            report_id="paired-test",
        )

        self.assertEqual(len(report.runs), 4)
        self.assertEqual(
            sorted(seen),
            sorted(
                [
                    ("d2-reference-separable-moderate", 2),
                    ("d2-independent-time-grids", 2),
                    ("d2-reference-separable-moderate", 5),
                    ("d2-independent-time-grids", 5),
                ]
            ),
        )
        classifications = {
            item.scenario_id: item.classification
            for item in report.boundary_records
        }
        self.assertIs(
            classifications["d2-reference-separable-moderate"],
            SyntheticWavelengthRobustnessBoundaryClass.REFERENCE,
        )
        self.assertIs(
            classifications["d2-independent-time-grids"],
            SyntheticWavelengthRobustnessBoundaryClass.ROBUST,
        )
        boundary_summary = report.aggregate.extra_fields["d2_boundary_summary"]
        self.assertTrue(boundary_summary["empirical_boundaries_calibrated"])
        self.assertFalse(boundary_summary["d1_gates_reused_as_d2_gates"])
        self.assertFalse(report.automatic_model_selection_applied)

    def test_unexpected_failures_define_failure_boundary(self):
        def fit_runner(lightcurve, configuration, case):
            seed = case.scenario.sampling_configuration[
                "purpose_specific_seeds"
            ]["master"]
            if (
                case.scenario.scenario_id == "d2-independent-time-grids"
                and seed in {1, 2}
            ):
                raise RuntimeError("controlled D2 failure")
            return self.successful_output(lightcurve, configuration, case)

        report = run_synthetic_wavelength_robustness_population(
            base_seeds=(0, 1, 2),
            specifications=self.reference_and_independent_specifications(),
            lightcurve_factory=self.lightcurve_factory,
            fit_runner=fit_runner,
            calibration_thresholds={"minimum_seed_count": 3},
            optimizer_seed_base=None,
        )
        record = next(
            item
            for item in report.boundary_records
            if item.scenario_id == "d2-independent-time-grids"
        )

        self.assertIs(
            record.classification,
            SyntheticWavelengthRobustnessBoundaryClass.FAILURE_BOUNDARY,
        )
        self.assertAlmostEqual(
            record.summary["unexpected_failure_fraction"], 2.0 / 3.0
        )

    def test_repeatable_expected_failure_has_distinct_class(self):
        specifications = canonical_synthetic_wavelength_robustness_specifications()
        identifiers = {
            "d2-reference-separable-moderate",
            "d2-insufficient-per-band-sampling",
        }
        selected = tuple(
            item for item in specifications if item.scenario_id in identifiers
        )

        def fit_runner(lightcurve, configuration, case):
            if case.scenario.expected_failure is not None:
                raise ValueError("insufficient sampling")
            return self.successful_output(lightcurve, configuration, case)

        report = run_synthetic_wavelength_robustness_population(
            base_seeds=(0, 1),
            specifications=selected,
            lightcurve_factory=self.lightcurve_factory,
            fit_runner=fit_runner,
            calibration_thresholds={"minimum_seed_count": 2},
            optimizer_seed_base=None,
        )
        record = next(
            item
            for item in report.boundary_records
            if item.scenario_id == "d2-insufficient-per-band-sampling"
        )

        self.assertIs(
            record.classification,
            SyntheticWavelengthRobustnessBoundaryClass.EXPECTED_FAILURE_BOUNDARY,
        )
        self.assertEqual(record.summary["n_expected_failures_matched"], 2)

    def test_incomplete_population_is_inconclusive(self):
        specifications = self.reference_and_independent_specifications()
        runs = []
        for specification in specifications:
            case = make_synthetic_wavelength_robustness_case(specification, seed=0)
            runs.append(
                run_synthetic_wavelength_robustness(
                    case,
                    lightcurve_factory=self.lightcurve_factory,
                    fit_runner=self.successful_output,
                )
            )

        summaries, records = calibrate_synthetic_wavelength_robustness_runs(
            runs,
            thresholds={"minimum_seed_count": 2},
        )
        record = next(
            item
            for item in records
            if item.scenario_id == "d2-independent-time-grids"
        )

        self.assertIn("d2-independent-time-grids", summaries)
        self.assertIs(
            record.classification,
            SyntheticWavelengthRobustnessBoundaryClass.INCONCLUSIVE,
        )

    def test_population_output_is_resumable_per_run(self):
        calls = []

        def fit_runner(lightcurve, configuration, case):
            calls.append(case.scenario.scenario_id)
            return self.successful_output(lightcurve, configuration, case)

        with tempfile.TemporaryDirectory() as tmpdir:
            first = run_synthetic_wavelength_robustness_population(
                base_seeds=(0,),
                specifications=self.reference_and_independent_specifications(),
                lightcurve_factory=self.lightcurve_factory,
                fit_runner=fit_runner,
                calibration_thresholds={"minimum_seed_count": 1},
                optimizer_seed_base=None,
                output_dir=tmpdir,
            )
            self.assertEqual(first.extra_fields["n_executed_runs"], 2)
            self.assertEqual(first.extra_fields["n_resumed_runs"], 0)
            self.assertEqual(len(calls), 2)
            self.assertTrue((Path(tmpdir) / "report.json").is_file())
            self.assertEqual(
                len(list((Path(tmpdir) / "runs").glob("*.json"))), 2
            )

            second = run_synthetic_wavelength_robustness_population(
                base_seeds=(0,),
                specifications=self.reference_and_independent_specifications(),
                lightcurve_factory=self.lightcurve_factory,
                fit_runner=fit_runner,
                calibration_thresholds={"minimum_seed_count": 1},
                optimizer_seed_base=None,
                output_dir=tmpdir,
                resume=True,
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(second.extra_fields["n_executed_runs"], 0)
        self.assertEqual(second.extra_fields["n_resumed_runs"], 2)
        self.assertEqual(
            [run.run_id for run in second.runs],
            [run.run_id for run in first.runs],
        )

    def test_report_round_trip_preserves_unknown_fields(self):
        report = run_synthetic_wavelength_robustness_population(
            base_seeds=(0,),
            specifications=self.reference_and_independent_specifications(),
            lightcurve_factory=self.lightcurve_factory,
            fit_runner=self.successful_output,
            calibration_thresholds={"minimum_seed_count": 1},
            optimizer_seed_base=None,
        )
        payload = report.to_dict()
        payload["future_field"] = {"retained": True}
        rebuilt = SyntheticWavelengthRobustnessCalibrationReport.from_mapping(
            payload
        )

        self.assertEqual(rebuilt.extra_fields["future_field"], {"retained": True})
        self.assertEqual(len(rebuilt.runs), len(report.runs))
        json.dumps(rebuilt.to_dict(), allow_nan=False)


class TestBoundaryRecord(unittest.TestCase):
    def test_boundary_record_enforces_advisory_semantics(self):
        record = SyntheticWavelengthRobustnessBoundaryRecord(
            scenario_id="d2-test",
            classification=SyntheticWavelengthRobustnessBoundaryClass.DEGRADED,
            axis="high_noise",
            severity=SyntheticWavelengthRobustnessSeverity.BOUNDARY.value,
            reference_scenario_id="d2-reference",
            n_runs=20,
            summary={"completion_fraction": 0.9},
        )
        rebuilt = SyntheticWavelengthRobustnessBoundaryRecord.from_mapping(
            record.to_dict()
        )

        self.assertEqual(rebuilt, record)
        with self.assertRaisesRegex(ValueError, "cannot select"):
            replace(record, automatic_model_selection_applied=True)


if __name__ == "__main__":
    unittest.main()
