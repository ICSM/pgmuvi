import json
import unittest

from pgmuvi.wavelength_status import (
    AttemptDisposition,
    ComparisonEligibility,
    DiagnosticValidity,
    ExecutionStage,
    ScientificUsability,
    TechnicalOutcome,
    WavelengthAttemptStatus,
)
from pgmuvi.wavelength_validation import WavelengthValidationPhase
from pgmuvi.wavelength_validation_recovery import (
    evaluate_synthetic_wavelength_recovery,
)
from pgmuvi.wavelength_validation_robustness import (
    SYNTHETIC_WAVELENGTH_ROBUSTNESS_SCHEMA_VERSION,
    SyntheticWavelengthRobustnessAxis,
    SyntheticWavelengthRobustnessReport,
    SyntheticWavelengthRobustnessSeverity,
    SyntheticWavelengthRobustnessSpecification,
    aggregate_synthetic_wavelength_robustness_runs,
    canonical_synthetic_wavelength_robustness_cases,
    canonical_synthetic_wavelength_robustness_specifications,
    make_synthetic_wavelength_robustness_case,
    run_synthetic_wavelength_robustness,
    run_synthetic_wavelength_robustness_matrix,
)
from pgmuvi.wavelength_validation_synthetic import (
    DEFAULT_VALIDATION_CANDIDATES,
    make_synthetic_wavelength_validation_case,
)


class RobustnessCaseMixin:
    def make_specification(self, **overrides):
        kwargs = {
            "scenario_id": "d2-test-sparse",
            "axis": "sparse_sampling",
            "severity": "challenging",
            "description": "Small deterministic D2 test case.",
            "generator_configuration": {
                "generating_model": "2DWavelengthDependent",
                "mean_kind": "quadratic",
                "covariance_kind": "separable_quasi_periodic_rbf",
                "strength": "strong",
                "wavelengths": (0.6, 1.0, 2.0),
                "band_labels": ("r", "J", "K"),
                "n_per_band": 6,
                "period": 100.0,
                "n_cycles": 3.0,
                "noise_sigma": 0.05,
                "shared_time_grid": True,
            },
            "reference_scenario_id": "d2-test-reference",
        }
        kwargs.update(overrides)
        return SyntheticWavelengthRobustnessSpecification(**kwargs)

    def make_case(self, **overrides):
        seed = overrides.pop("seed", 17)
        specification = self.make_specification(**overrides)
        return make_synthetic_wavelength_robustness_case(
            specification,
            seed=seed,
        )

    def truth_outputs(self, case):
        truth = case.scenario.truth
        return {
            "fitted_period": truth.temporal_parameters["fundamental_period"],
            "fitted_wavelength_lengthscale": (
                truth.wavelength_covariance_parameters[
                    "wavelength_lengthscale"
                ]
            ),
            "fitted_mean_by_band": truth.noiseless_summary["mean_by_band"],
            "boundary_hits": [],
            "diagnostics": {"source": "test_hook"},
            "parameter_workflow": {"available": True},
        }


class TestRobustnessSpecifications(RobustnessCaseMixin, unittest.TestCase):
    def test_specification_round_trip_is_json_safe(self):
        specification = self.make_specification()
        payload = specification.to_dict()
        payload["future_field"] = {"retained": True}
        rebuilt = SyntheticWavelengthRobustnessSpecification.from_mapping(
            payload
        )

        self.assertEqual(rebuilt.extra_fields["future_field"], {"retained": True})
        json.dumps(rebuilt.to_dict(), allow_nan=False)

    def test_specification_requires_generator_model_and_families(self):
        with self.assertRaisesRegex(ValueError, "generating_model"):
            self.make_specification(generator_configuration={})

    def test_case_is_promoted_to_d2_without_changing_truth_arrays(self):
        specification = self.make_specification()
        case = make_synthetic_wavelength_robustness_case(specification, seed=3)
        direct = make_synthetic_wavelength_validation_case(
            scenario_id=specification.scenario_id,
            description=specification.description,
            seed=3,
            **dict(specification.generator_configuration),
        )

        self.assertEqual(case.time_values, direct.time_values)
        self.assertEqual(case.observed_flux, direct.observed_flux)
        self.assertEqual(
            case.scenario.phase,
            WavelengthValidationPhase.D2_ROBUSTNESS_FAILURE_BOUNDARY,
        )
        self.assertEqual(
            case.scenario.metadata["robustness_axis"], "sparse_sampling"
        )
        self.assertEqual(
            case.scenario.metadata["reference_scenario_id"],
            "d2-test-reference",
        )
        self.assertIn("d2", case.scenario.tags)
        self.assertNotIn("d1", case.scenario.tags)
        self.assertFalse(case.scenario.fit_configuration["selection_performed"])

    def test_case_round_trip_preserves_d2_phase_and_expectation(self):
        specifications = canonical_synthetic_wavelength_robustness_specifications()
        failure_specification = next(
            item for item in specifications if item.expected_failure is not None
        )
        case = make_synthetic_wavelength_robustness_case(
            failure_specification,
            seed=11,
        )
        rebuilt = type(case).from_mapping(case.to_dict())

        self.assertEqual(
            rebuilt.scenario.phase,
            WavelengthValidationPhase.D2_ROBUSTNESS_FAILURE_BOUNDARY,
        )
        self.assertEqual(
            rebuilt.scenario.expected_failure.failure_code,
            "synthetic_recovery_fit_failed",
        )
        json.dumps(rebuilt.to_dict(), allow_nan=False)


class TestCanonicalRobustnessCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specifications = (
            canonical_synthetic_wavelength_robustness_specifications()
        )
        cls.cases = canonical_synthetic_wavelength_robustness_cases(seed=101)

    def test_canonical_set_has_unique_ids_and_all_declared_axes(self):
        identifiers = [item.scenario_id for item in self.specifications]
        axes = {item.axis for item in self.specifications}

        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertIn(SyntheticWavelengthRobustnessAxis.REFERENCE, axes)
        self.assertIn(
            SyntheticWavelengthRobustnessAxis.INSUFFICIENT_PER_BAND_SAMPLING,
            axes,
        )
        self.assertIn(SyntheticWavelengthRobustnessAxis.LARGE_WAVELENGTH_GAP, axes)
        self.assertIn(SyntheticWavelengthRobustnessAxis.JOINT_SM_SPARSE_ARD, axes)

    def test_canonical_set_covers_maintained_models_only(self):
        models = {
            case.scenario.truth.generating_model for case in self.cases
        }

        self.assertEqual(models, set(DEFAULT_VALIDATION_CANDIDATES))
        self.assertNotIn("2DAchromatic", models)

    def test_canonical_cases_are_d2_positive_linear_flux(self):
        for case in self.cases:
            with self.subTest(case=case.scenario.scenario_id):
                self.assertEqual(
                    case.scenario.phase,
                    WavelengthValidationPhase.D2_ROBUSTNESS_FAILURE_BOUNDARY,
                )
                self.assertGreater(min(case.observed_flux), 0.0)
                self.assertTrue(
                    case.scenario.truth.noiseless_summary["linear_flux"]
                )

    def test_expected_failure_is_reserved_for_invalid_boundary(self):
        expected = [
            case for case in self.cases if case.scenario.expected_failure is not None
        ]

        self.assertEqual(len(expected), 1)
        case = expected[0]
        self.assertEqual(
            case.scenario.metadata["robustness_severity"],
            SyntheticWavelengthRobustnessSeverity.INVALID.value,
        )
        self.assertEqual(
            case.scenario.sampling_configuration["n_per_band"], [2, 2]
        )


class TestRobustnessRun(RobustnessCaseMixin, unittest.TestCase):
    def test_successful_nonfailure_case_gets_d2_annotation(self):
        case = self.make_case()
        run = run_synthetic_wavelength_robustness(
            case,
            lightcurve_factory=lambda _: object(),
            fit_runner=lambda _lc, _kwargs, item: self.truth_outputs(item),
        )
        by_name = {metric.name: metric for metric in run.metrics}
        evaluation = run.extra_fields["expected_failure_evaluation"]

        self.assertEqual(run.model, "2DWavelengthDependent")
        self.assertFalse(evaluation["expected"])
        self.assertFalse(evaluation["unexpected_failure"])
        self.assertTrue(
            by_name["unexpected_technical_failure_absent"].passed
        )
        self.assertEqual(
            run.extra_fields["validation_phase"],
            WavelengthValidationPhase.D2_ROBUSTNESS_FAILURE_BOUNDARY.value,
        )

    def test_matching_expected_failure_remains_failed_and_is_classified(self):
        specification = next(
            item
            for item in canonical_synthetic_wavelength_robustness_specifications()
            if item.expected_failure is not None
        )
        case = make_synthetic_wavelength_robustness_case(specification, seed=5)

        def fail(_lightcurve, _fit_kwargs, _case):
            raise RuntimeError("deliberate consensus failure")

        run = run_synthetic_wavelength_robustness(
            case,
            lightcurve_factory=lambda _: object(),
            fit_runner=fail,
        )
        evaluation = run.extra_fields["expected_failure_evaluation"]
        metric = {item.name: item for item in run.metrics}[
            "expected_failure_contract_matched"
        ]

        self.assertEqual(
            run.status.technical_outcome,
            TechnicalOutcome.FAILED,
        )
        self.assertTrue(evaluation["matched"])
        self.assertTrue(metric.passed)

    def test_mismatched_failure_stage_is_preserved(self):
        specification = next(
            item
            for item in canonical_synthetic_wavelength_robustness_specifications()
            if item.expected_failure is not None
        )
        case = make_synthetic_wavelength_robustness_case(specification, seed=5)

        def fail_factory(_case):
            raise RuntimeError("construction failed")

        run = run_synthetic_wavelength_robustness(
            case,
            lightcurve_factory=fail_factory,
        )
        evaluation = run.extra_fields["expected_failure_evaluation"]

        self.assertFalse(evaluation["matched"])
        self.assertFalse(evaluation["checks"]["stage"])
        self.assertEqual(run.failure.stage, ExecutionStage.SETUP)

    def test_d1_case_is_rejected_by_d2_runner(self):
        d1_case = make_synthetic_wavelength_validation_case(
            scenario_id="d1-not-d2",
            generating_model="2DSeparable",
            mean_kind="constant",
            covariance_kind="separable_quasi_periodic_rbf",
            wavelengths=(0.6, 1.0, 2.0),
            band_labels=("r", "J", "K"),
            n_per_band=4,
            period=100.0,
            n_cycles=3.0,
            seed=1,
        )

        with self.assertRaisesRegex(ValueError, "require a D2 scenario"):
            run_synthetic_wavelength_robustness(d1_case)

    def test_explicit_nonapplicable_failure_contract_uses_ordinary_semantics(self):
        specification = next(
            item
            for item in canonical_synthetic_wavelength_robustness_specifications()
            if item.expected_failure is not None
        )
        case = make_synthetic_wavelength_robustness_case(specification, seed=5)
        status = WavelengthAttemptStatus(
            disposition=AttemptDisposition.ATTEMPTED,
            execution_stage=ExecutionStage.COMPLETED,
            technical_outcome=TechnicalOutcome.COMPLETED,
            diagnostic_validity=DiagnosticValidity.VALID,
            scientific_usability=ScientificUsability.USABLE,
            comparison_eligibility=ComparisonEligibility.ELIGIBLE,
        )

        run = run_synthetic_wavelength_robustness(
            case,
            model="2D",
            lightcurve_factory=lambda _: object(),
            fit_runner=lambda _lc, _kwargs, _case: {
                **self.truth_outputs(case),
                "status": status,
            },
        )
        evaluation = run.extra_fields["expected_failure_evaluation"]

        self.assertTrue(evaluation["expected"])
        self.assertFalse(evaluation["applicable"])
        self.assertIsNone(evaluation["matched"])
        self.assertFalse(evaluation["unexpected_failure"])


class TestRobustnessAggregate(RobustnessCaseMixin, unittest.TestCase):
    def make_success(self, case):
        return run_synthetic_wavelength_robustness(
            case,
            lightcurve_factory=lambda _: object(),
            fit_runner=lambda _lc, _kwargs, item: self.truth_outputs(item),
        )

    def test_aggregate_rejects_unannotated_d1_run(self):
        d1_case = make_synthetic_wavelength_validation_case(
            scenario_id="d1-aggregate-rejection",
            generating_model="2DWavelengthDependent",
            mean_kind="quadratic",
            covariance_kind="separable_quasi_periodic_rbf",
            strength="strong",
            wavelengths=(0.6, 1.0, 2.0),
            band_labels=("r", "J", "K"),
            n_per_band=6,
            period=100.0,
            n_cycles=3.0,
            noise_sigma=0.05,
            seed=4,
        )
        run = evaluate_synthetic_wavelength_recovery(
            d1_case,
            "2DWavelengthDependent",
            **self.truth_outputs(d1_case),
        )

        with self.assertRaisesRegex(ValueError, "D2-annotated"):
            aggregate_synthetic_wavelength_robustness_runs([run])

    def test_aggregate_reports_axes_without_d1_gate_summary(self):
        first = self.make_case()
        second = self.make_case(
            scenario_id="d2-test-noise",
            axis="high_noise",
            severity="boundary",
        )
        aggregate = aggregate_synthetic_wavelength_robustness_runs(
            [self.make_success(first), self.make_success(second)]
        )
        extras = aggregate.extra_fields

        self.assertEqual(
            aggregate.phase,
            WavelengthValidationPhase.D2_ROBUSTNESS_FAILURE_BOUNDARY,
        )
        self.assertIn("sparse_sampling", extras["axis_summaries"])
        self.assertIn("high_noise", extras["axis_summaries"])
        self.assertFalse(
            extras["d2_boundary_summary"]["empirical_boundaries_calibrated"]
        )
        self.assertFalse(
            extras["d2_boundary_summary"]["d1_gates_reused_as_d2_gates"]
        )
        self.assertNotIn("d1_gate_summary", extras)
        self.assertFalse(aggregate.automatic_model_selection_applied)

    def test_aggregate_separates_expected_and_unexpected_failures(self):
        ordinary = self.make_case()

        def fail(_lightcurve, _fit_kwargs, _case):
            raise RuntimeError("unexpected")

        unexpected = run_synthetic_wavelength_robustness(
            ordinary,
            lightcurve_factory=lambda _: object(),
            fit_runner=fail,
        )
        specification = next(
            item
            for item in canonical_synthetic_wavelength_robustness_specifications()
            if item.expected_failure is not None
        )
        expected_case = make_synthetic_wavelength_robustness_case(
            specification,
            seed=8,
        )
        expected = run_synthetic_wavelength_robustness(
            expected_case,
            lightcurve_factory=lambda _: object(),
            fit_runner=fail,
        )
        aggregate = aggregate_synthetic_wavelength_robustness_runs(
            [unexpected, expected]
        )

        self.assertEqual(
            aggregate.failure_summaries["n_expected_failures_matched"], 1
        )
        self.assertEqual(aggregate.failure_summaries["n_unexpected_failures"], 1)
        self.assertEqual(aggregate.failure_summaries["n_failures"], 2)

    def test_matrix_defaults_to_each_generating_model(self):
        first = self.make_case()
        second = self.make_case(
            scenario_id="d2-test-dust",
            axis="uneven_band_counts",
            generator_configuration={
                **dict(self.make_specification().generator_configuration),
                "generating_model": "2DDustMean",
                "mean_kind": "dust",
            },
        )

        report = run_synthetic_wavelength_robustness_matrix(
            [first, second],
            lightcurve_factory=lambda _: object(),
            fit_runner=lambda _lc, _kwargs, case: self.truth_outputs(case),
            report_id="matched-model-report",
        )

        self.assertEqual(len(report.runs), 2)
        self.assertEqual(
            {run.model for run in report.runs},
            {"2DWavelengthDependent", "2DDustMean"},
        )

    def test_explicit_models_request_cross_product(self):
        case = self.make_case()
        report = run_synthetic_wavelength_robustness_matrix(
            [case],
            models=("2DWavelengthDependent", "2DSeparable"),
            lightcurve_factory=lambda _: object(),
            fit_runner=lambda _lc, _kwargs, item: self.truth_outputs(item),
        )

        self.assertEqual(len(report.runs), 2)

    def test_report_round_trip_preserves_unknown_fields(self):
        case = self.make_case()
        run = self.make_success(case)
        aggregate = aggregate_synthetic_wavelength_robustness_runs([run])
        report = SyntheticWavelengthRobustnessReport(
            report_id="round-trip",
            cases=(case,),
            runs=(run,),
            aggregate=aggregate,
        )
        payload = report.to_dict()
        payload["future_report_field"] = {"retained": True}
        rebuilt = SyntheticWavelengthRobustnessReport.from_mapping(payload)

        self.assertEqual(
            rebuilt.extra_fields["future_report_field"], {"retained": True}
        )
        self.assertEqual(
            rebuilt.schema_version,
            SYNTHETIC_WAVELENGTH_ROBUSTNESS_SCHEMA_VERSION,
        )
        self.assertFalse(rebuilt.automatic_model_selection_applied)
        json.dumps(rebuilt.to_dict(), allow_nan=False)

    def test_empty_matrix_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            run_synthetic_wavelength_robustness_matrix([])


if __name__ == "__main__":
    unittest.main()
