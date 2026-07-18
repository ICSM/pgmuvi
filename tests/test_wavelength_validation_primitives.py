import json
import math
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
from pgmuvi.wavelength_validation import (
    WAVELENGTH_VALIDATION_SCHEMA_VERSION,
    RecoveryMetricDirection,
    WavelengthRecoveryMetric,
    WavelengthValidationAggregate,
    WavelengthValidationFailureExpectation,
    WavelengthValidationPhase,
    WavelengthValidationProvenance,
    WavelengthValidationRun,
    WavelengthValidationScenario,
    WavelengthValidationSourceKind,
    WavelengthValidationTruth,
    as_wavelength_validation_aggregate,
    as_wavelength_validation_run,
    as_wavelength_validation_scenario,
)


class TestValidationTruthAndProvenance(unittest.TestCase):
    def test_truth_round_trip_preserves_unknown_fields(self):
        payload = {
            "generating_model": "2DDustMean",
            "truth_kind": "mean_and_covariance",
            "physical_wavelengths": [0.5, 1.0, 2.0],
            "band_labels": ["g", "J", "K"],
            "temporal_parameters": {"period": 600.0},
            "wavelength_mean_parameters": {"alpha": 1.5},
            "future_truth_field": {"retained": True},
        }
        record = WavelengthValidationTruth.from_mapping(payload)
        encoded = record.to_dict()

        self.assertEqual(record.schema_version, WAVELENGTH_VALIDATION_SCHEMA_VERSION)
        self.assertEqual(encoded["extra_fields"]["future_truth_field"], {"retained": True})
        self.assertEqual(
            WavelengthValidationTruth.from_mapping(encoded).to_dict(), encoded
        )
        json.dumps(encoded)

    def test_truth_requires_matching_band_and_wavelength_lengths(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            WavelengthValidationTruth(
                generating_model="2DSeparable",
                truth_kind="covariance",
                physical_wavelengths=(0.5, 1.0),
                band_labels=("g",),
            )

    def test_truth_rejects_non_physical_wavelengths(self):
        with self.assertRaisesRegex(ValueError, "finite positive"):
            WavelengthValidationTruth(
                generating_model="2DSeparable",
                truth_kind="covariance",
                physical_wavelengths=(0.5, math.inf),
                band_labels=("g", "K"),
            )

    def test_provenance_normalizes_non_finite_values(self):
        provenance = WavelengthValidationProvenance(
            package_commit="9c47e5f",
            dependencies={"numpy": "test"},
            configuration={"loss": math.inf, "score": math.nan},
        )
        payload = provenance.to_dict()
        self.assertIsNone(payload["configuration"]["loss"])
        self.assertIsNone(payload["configuration"]["score"])
        json.dumps(payload, allow_nan=False)


class TestValidationScenario(unittest.TestCase):
    def _truth(self):
        return WavelengthValidationTruth(
            generating_model="2DPowerLawMean",
            truth_kind="mean_and_covariance",
            physical_wavelengths=(0.5, 1.0, 2.0),
            band_labels=("g", "J", "K"),
        )

    def test_synthetic_scenario_round_trip(self):
        scenario = WavelengthValidationScenario(
            scenario_id="d1-powerlaw-moderate",
            phase=WavelengthValidationPhase.D1_SYNTHETIC_RECOVERY,
            source_kind=WavelengthValidationSourceKind.SYNTHETIC,
            description="Moderate power-law wavelength dependence.",
            truth=self._truth(),
            sampling_configuration={"n_cycles": 4.0},
            fit_configuration={"fit_strategy": "consensus"},
            tags=("nominal", "power_law"),
        )
        payload = scenario.to_dict()
        rebuilt = WavelengthValidationScenario.from_mapping(payload)

        self.assertTrue(payload["advisory_only"])
        self.assertEqual(rebuilt, scenario)
        self.assertIs(as_wavelength_validation_scenario(scenario), scenario)
        json.dumps(payload)

    def test_synthetic_scenario_requires_truth(self):
        with self.assertRaisesRegex(ValueError, "require a truth record"):
            WavelengthValidationScenario(
                scenario_id="missing-truth",
                phase=WavelengthValidationPhase.D1_SYNTHETIC_RECOVERY,
                source_kind=WavelengthValidationSourceKind.SYNTHETIC,
                description="Invalid synthetic scenario.",
            )

    def test_observed_scenario_may_omit_truth(self):
        scenario = WavelengthValidationScenario(
            scenario_id="cit6",
            phase=WavelengthValidationPhase.D3_REPRESENTATIVE_LPV,
            source_kind=WavelengthValidationSourceKind.OBSERVED,
            description="Representative LPV source.",
        )
        self.assertIsNone(scenario.truth)

    def test_expected_failure_is_classified_and_ineligible(self):
        expectation = WavelengthValidationFailureExpectation(
            failure_code="insufficient_usable_bands",
            stage=ExecutionStage.PRECONDITION,
            technical_outcome=TechnicalOutcome.SKIPPED,
            comparison_eligibility=ComparisonEligibility.INELIGIBLE,
            acceptable_exception_types=("ValueError",),
        )
        payload = expectation.to_dict()
        self.assertEqual(payload["failure_stage"], "precondition")
        self.assertEqual(payload["technical_outcome"], "skipped")
        json.dumps(payload)

    def test_expected_failure_cannot_be_comparison_eligible(self):
        with self.assertRaisesRegex(ValueError, "comparison-eligible"):
            WavelengthValidationFailureExpectation(
                failure_code="optimizer_failure",
                stage=ExecutionStage.OPTIMIZATION,
                comparison_eligibility=ComparisonEligibility.ELIGIBLE,
            )


class TestRecoveryMetricAndRun(unittest.TestCase):
    def _status(self, technical_outcome=TechnicalOutcome.COMPLETED):
        return WavelengthAttemptStatus(
            disposition=AttemptDisposition.ATTEMPTED,
            execution_stage=ExecutionStage.COMPLETED,
            technical_outcome=technical_outcome,
            diagnostic_validity=DiagnosticValidity.VALID,
            scientific_usability=ScientificUsability.USABLE,
            comparison_eligibility=ComparisonEligibility.ELIGIBLE,
        )

    def test_metric_records_dimension_component_and_threshold(self):
        metric = WavelengthRecoveryMetric(
            name="wavelength_scale_factor_error",
            value=1.4,
            truth_value=1.0,
            passed=True,
            direction=RecoveryMetricDirection.LOWER_IS_BETTER,
            threshold={"maximum": 2.0},
            model="2D",
            parameter="mixture_scales",
            ard_dimension="wavelength_frequency",
            component_index=0,
            limitations=("descriptive recovery metric",),
        )
        payload = metric.to_dict()
        self.assertEqual(payload["ard_dimension"], "wavelength_frequency")
        self.assertEqual(payload["component_index"], 0)
        self.assertEqual(payload["direction"], "lower_is_better")
        json.dumps(payload)

    def test_run_round_trip_preserves_status_metrics_and_unknown_fields(self):
        payload = {
            "run_id": "d1-powerlaw-moderate-seed7-2DPowerLawMean",
            "scenario_id": "d1-powerlaw-moderate",
            "model": "2DPowerLawMean",
            "seed": 7,
            "status": self._status().to_dict(),
            "metrics": [
                {
                    "name": "period_relative_error",
                    "value": 0.03,
                    "truth_value": 600.0,
                    "passed": True,
                }
            ],
            "future_run_field": [1, 2, 3],
        }
        run = WavelengthValidationRun.from_mapping(payload)
        encoded = run.to_dict()

        self.assertEqual(run.status.technical_outcome, TechnicalOutcome.COMPLETED)
        self.assertEqual(run.metrics[0].name, "period_relative_error")
        self.assertEqual(encoded["extra_fields"]["future_run_field"], [1, 2, 3])
        self.assertEqual(WavelengthValidationRun.from_mapping(encoded).to_dict(), encoded)
        self.assertIs(as_wavelength_validation_run(run), run)
        json.dumps(encoded)

    def test_failure_record_requires_failed_status(self):
        with self.assertRaisesRegex(ValueError, "failed technical outcome"):
            WavelengthValidationRun(
                run_id="bad-run",
                scenario_id="scenario",
                model="2D",
                status=self._status(),
                failure={
                    "failure_code": "optimization_error",
                    "failure_stage": "optimization",
                    "exception_message": "failed",
                },
            )


class TestValidationAggregate(unittest.TestCase):
    def test_aggregate_round_trip_is_failure_aware_and_advisory_only(self):
        aggregate = WavelengthValidationAggregate(
            aggregate_id="d1-nominal-summary",
            phase=WavelengthValidationPhase.D1_SYNTHETIC_RECOVERY,
            scenario_ids=("scenario-a",),
            run_ids=("run-a", "run-b"),
            status_counts={"completed": 1, "failed": 1},
            metric_summaries={"period_relative_error": {"median": 0.04}},
            failure_summaries={"optimization": 1},
        )
        payload = aggregate.to_dict()
        rebuilt = WavelengthValidationAggregate.from_mapping(payload)

        self.assertEqual(payload["n_runs"], 2)
        self.assertTrue(payload["advisory_only"])
        self.assertFalse(payload["automatic_model_selection_applied"])
        self.assertEqual(rebuilt, aggregate)
        self.assertIs(as_wavelength_validation_aggregate(aggregate), aggregate)
        json.dumps(payload)

    def test_aggregate_rejects_automatic_model_selection(self):
        with self.assertRaisesRegex(ValueError, "cannot apply automatic"):
            WavelengthValidationAggregate(
                aggregate_id="invalid",
                phase=WavelengthValidationPhase.D1_SYNTHETIC_RECOVERY,
                automatic_model_selection_applied=True,
            )

    def test_records_reject_selection_claims_in_extension_fields(self):
        with self.assertRaisesRegex(ValueError, "model-selection claims"):
            WavelengthRecoveryMetric(
                name="invalid",
                extra_fields={"selected_model": "2D"},
            )
        with self.assertRaisesRegex(ValueError, "model-selection claims"):
            WavelengthValidationAggregate(
                aggregate_id="invalid-summary",
                phase=WavelengthValidationPhase.D1_SYNTHETIC_RECOVERY,
                model_summaries={"winner": "2D"},
            )


if __name__ == "__main__":
    unittest.main()
