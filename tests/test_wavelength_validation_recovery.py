import json
import unittest

from pgmuvi.wavelength_status import (
    ComparisonEligibility,
    ExecutionStage,
    TechnicalOutcome,
    WavelengthFailureRecord,
)
from pgmuvi.wavelength_validation_recovery import (
    DEFAULT_SYNTHETIC_RECOVERY_THRESHOLDS,
    SUPPORTED_SYNTHETIC_RECOVERY_MODELS,
    SyntheticWavelengthRecoveryReport,
    SyntheticWavelengthRecoveryThresholds,
    aggregate_synthetic_wavelength_recovery_runs,
    build_synthetic_wavelength_recovery_metrics,
    evaluate_synthetic_wavelength_recovery,
    run_synthetic_wavelength_recovery,
    run_synthetic_wavelength_recovery_matrix,
)
from pgmuvi.wavelength_validation_synthetic import (
    make_synthetic_wavelength_validation_case,
)


class RecoveryCaseMixin:
    def make_case(self, **overrides):
        kwargs = {
            "scenario_id": "d1-recovery-test",
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
            "seed": 17,
        }
        kwargs.update(overrides)
        return make_synthetic_wavelength_validation_case(**kwargs)

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


class TestSyntheticRecoveryThresholds(unittest.TestCase):
    def test_defaults_match_frozen_d1_gates(self):
        thresholds = DEFAULT_SYNTHETIC_RECOVERY_THRESHOLDS

        self.assertEqual(thresholds.period_relative_error, 0.15)
        self.assertEqual(thresholds.wavelength_lengthscale_factor_error, 4.0)
        self.assertEqual(thresholds.mean_normalized_rmse("strong"), 0.15)
        self.assertEqual(thresholds.mean_normalized_rmse("weak"), 0.25)
        json.dumps(thresholds.to_dict(), allow_nan=False)

    def test_thresholds_require_positive_finite_values(self):
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            SyntheticWavelengthRecoveryThresholds(period_relative_error=0.0)

    def test_factor_threshold_cannot_be_below_one(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            SyntheticWavelengthRecoveryThresholds(
                wavelength_lengthscale_factor_error=0.9
            )


class TestSyntheticRecoveryMetrics(RecoveryCaseMixin, unittest.TestCase):
    def metric_map(self, metrics):
        return {metric.name: metric for metric in metrics}

    def test_exact_truth_outputs_pass_available_recovery_metrics(self):
        case = self.make_case(xtransform="minmax")
        outputs = self.truth_outputs(case)
        metrics = build_synthetic_wavelength_recovery_metrics(
            case,
            "2DWavelengthDependent",
            fitted_period=outputs["fitted_period"],
            fitted_wavelength_lengthscale=outputs[
                "fitted_wavelength_lengthscale"
            ],
            fitted_mean_by_band=outputs["fitted_mean_by_band"],
            boundary_hits=outputs["boundary_hits"],
        )
        by_name = self.metric_map(metrics)

        self.assertTrue(
            by_name["coordinate_round_trip_max_abs_error"].passed
        )
        self.assertTrue(by_name["period_relative_error"].passed)
        self.assertTrue(
            by_name["wavelength_lengthscale_factor_error"].passed
        )
        self.assertTrue(by_name["mean_law_normalized_rmse"].passed)
        self.assertEqual(by_name["ard_boundary_hit_count"].value, 0)
        self.assertTrue(by_name["ard_boundary_labels_valid"].passed)

    def test_missing_fit_outputs_are_explicitly_unavailable(self):
        case = self.make_case()
        by_name = self.metric_map(
            build_synthetic_wavelength_recovery_metrics(
                case, "2DWavelengthDependent"
            )
        )

        self.assertTrue(
            by_name["coordinate_round_trip_max_abs_error"].available
        )
        self.assertFalse(by_name["period_relative_error"].available)
        self.assertIsNone(by_name["period_relative_error"].passed)
        self.assertFalse(
            by_name["wavelength_lengthscale_factor_error"].available
        )
        self.assertFalse(by_name["mean_law_normalized_rmse"].available)

    def test_period_and_lengthscale_errors_use_relative_and_factor_forms(self):
        case = self.make_case()
        truth = case.scenario.truth
        metrics = build_synthetic_wavelength_recovery_metrics(
            case,
            "2DSeparable",
            fitted_period=125.0,
            fitted_wavelength_lengthscale=(
                5.0
                * truth.wavelength_covariance_parameters[
                    "wavelength_lengthscale"
                ]
            ),
        )
        by_name = self.metric_map(metrics)

        self.assertAlmostEqual(by_name["period_relative_error"].value, 0.25)
        self.assertFalse(by_name["period_relative_error"].passed)
        self.assertAlmostEqual(
            by_name["wavelength_lengthscale_factor_error"].value, 5.0
        )
        self.assertFalse(
            by_name["wavelength_lengthscale_factor_error"].passed
        )

    def test_mean_metric_is_prediction_space_not_parameter_space(self):
        case = self.make_case()
        truth = case.scenario.truth.noiseless_summary["mean_by_band"]
        shifted = [value + 20.0 for value in truth]
        by_name = self.metric_map(
            build_synthetic_wavelength_recovery_metrics(
                case,
                "2DWavelengthDependent",
                fitted_mean_by_band=shifted,
            )
        )

        metric = by_name["mean_law_normalized_rmse"]
        self.assertFalse(metric.passed)
        self.assertEqual(metric.parameter, "mean_module")
        self.assertEqual(metric.threshold["maximum"], 0.15)

    def test_joint_sm_metrics_cover_parameter_component_and_dimension(self):
        case = self.make_case(
            generating_model="2D",
            mean_kind="constant",
            covariance_kind="joint_spectral_mixture_ard",
        )
        truth = case.scenario.truth.wavelength_covariance_parameters
        diagnostics = {
            "parameters": {
                "mixture_means": {
                    "raw_input_coordinate_values": truth["mixture_means"]
                },
                "mixture_scales": {
                    "raw_input_coordinate_values": truth["mixture_scales"]
                },
            }
        }
        metrics = build_synthetic_wavelength_recovery_metrics(
            case,
            "2D",
            sm_ard_diagnostics=diagnostics,
        )
        factor_metrics = [
            metric for metric in metrics if metric.name == "sm_ard_factor_error"
        ]
        absolute_metrics = [
            metric
            for metric in metrics
            if metric.name == "sm_ard_absolute_error"
        ]

        self.assertEqual(len(factor_metrics), 4)
        self.assertEqual(len(absolute_metrics), 4)
        self.assertEqual(
            {metric.parameter for metric in factor_metrics},
            {"mixture_means", "mixture_scales"},
        )
        self.assertEqual(
            {metric.ard_dimension for metric in factor_metrics},
            {"temporal_frequency", "wavelength_frequency"},
        )
        self.assertTrue(
            all(
                metric.passed
                for metric in factor_metrics
                if metric.available
            )
        )
        zero_frequency = [
            metric
            for metric in factor_metrics
            if metric.parameter == "mixture_means"
            and metric.ard_dimension == "wavelength_frequency"
        ][0]
        self.assertFalse(zero_frequency.available)
        self.assertEqual(zero_frequency.truth_value, 0.0)

    def test_boundary_metric_accepts_only_maintained_dimension_labels(self):
        case = self.make_case()
        valid = build_synthetic_wavelength_recovery_metrics(
            case,
            "2D",
            boundary_hits=(
                {
                    "parameter_name": "mixture_scales",
                    "dimension_name": "wavelength_frequency",
                },
            ),
        )
        invalid = build_synthetic_wavelength_recovery_metrics(
            case,
            "2D",
            boundary_hits=(
                {
                    "parameter_name": "mixture_scales",
                    "dimension_name": "time_frequency",
                },
            ),
        )

        self.assertTrue(self.metric_map(valid)["ard_boundary_labels_valid"].passed)
        self.assertFalse(
            self.metric_map(invalid)["ard_boundary_labels_valid"].passed
        )


class TestSyntheticRecoveryRun(RecoveryCaseMixin, unittest.TestCase):
    def test_evaluator_builds_eligible_advisory_run(self):
        case = self.make_case()
        run = evaluate_synthetic_wavelength_recovery(
            case,
            "2DWavelengthDependent",
            **self.truth_outputs(case),
        )
        payload = run.to_dict()

        self.assertEqual(run.status.technical_outcome, TechnicalOutcome.COMPLETED)
        self.assertEqual(
            run.status.comparison_eligibility,
            ComparisonEligibility.ELIGIBLE,
        )
        self.assertTrue(payload["advisory_only"])
        self.assertNotIn("selected_model", payload)
        self.assertEqual(run.seed, 17)
        json.dumps(payload, allow_nan=False)

    def test_evaluator_preserves_classified_failure(self):
        case = self.make_case()
        failure = WavelengthFailureRecord(
            failure_code="synthetic_recovery_fit_failed",
            stage=ExecutionStage.OPTIMIZATION,
            substage="fit_execution",
            exception_type="RuntimeError",
            message="optimizer failed",
            diagnostics={},
        )
        run = evaluate_synthetic_wavelength_recovery(
            case,
            "2D",
            failure=failure,
        )

        self.assertEqual(run.status.technical_outcome, TechnicalOutcome.FAILED)
        self.assertEqual(run.status.execution_stage, ExecutionStage.OPTIMIZATION)
        self.assertEqual(run.failure.failure_code, "synthetic_recovery_fit_failed")

    def test_default_runner_extracts_period_lengthscale_and_mean(self):
        import torch

        case = self.make_case(xtransform="minmax")
        truth = case.scenario.truth
        time_scale = truth.coordinate_transforms["time"]["scale"]
        wavelength_scale = truth.coordinate_transforms["wavelength"]["scale"]
        expected_mean = truth.noiseless_summary["mean_by_band"]
        expected_period = truth.temporal_parameters["fundamental_period"]
        expected_lengthscale = truth.wavelength_covariance_parameters[
            "wavelength_lengthscale"
        ]

        class FakeMean:
            def parameters(self):
                return iter(())

            def __call__(self, _inputs):
                return torch.tensor(expected_mean, dtype=torch.float64)

        class FakeTimeKernel:
            period_length = torch.tensor(
                expected_period / time_scale, dtype=torch.float64
            )

        class FakeWavelengthBase:
            lengthscale = torch.tensor(
                expected_lengthscale / wavelength_scale, dtype=torch.float64
            )

        class FakeWavelengthKernel:
            base_kernel = FakeWavelengthBase()

        class FakeCovariance:
            kernels = (FakeTimeKernel(), FakeWavelengthKernel())

        class FakeModel:
            mean_module = FakeMean()
            covar_module = FakeCovariance()

        class FakeLightcurve:
            model = FakeModel()
            consensus_diagnostics = {}
            fit_history = [{"success": True}]

            def fit(self, **fit_kwargs):
                self.fit_kwargs = dict(fit_kwargs)
                return [1.0, 0.5]

            def get_parameter_workflow_summary(self):
                return {"available": True, "applied": []}

        lightcurve = FakeLightcurve()
        run = run_synthetic_wavelength_recovery(
            case,
            "2DWavelengthDependent",
            lightcurve_factory=lambda _: lightcurve,
        )
        metrics = {metric.name: metric for metric in run.metrics}

        self.assertEqual(run.status.technical_outcome, TechnicalOutcome.COMPLETED)
        self.assertTrue(metrics["period_relative_error"].passed)
        self.assertTrue(
            metrics["wavelength_lengthscale_factor_error"].passed
        )
        self.assertTrue(metrics["mean_law_normalized_rmse"].passed)
        self.assertEqual(lightcurve.fit_kwargs["fit_strategy"], "consensus")
        self.assertTrue(run.parameter_workflow["available"])

    def test_runner_uses_injected_outputs_without_gpytorch(self):
        case = self.make_case()
        sentinel = object()
        seen = {}

        def fit_runner(lightcurve, fit_kwargs, received_case):
            seen["lightcurve"] = lightcurve
            seen["fit_kwargs"] = dict(fit_kwargs)
            seen["case"] = received_case
            return self.truth_outputs(received_case)

        run = run_synthetic_wavelength_recovery(
            case,
            "2DDustMean",
            fit_kwargs={"training_iter": 12, "miniter": 4},
            lightcurve_factory=lambda _: sentinel,
            fit_runner=fit_runner,
        )

        self.assertIs(seen["lightcurve"], sentinel)
        self.assertIs(seen["case"], case)
        self.assertEqual(seen["fit_kwargs"]["model"], "2DDustMean")
        self.assertEqual(seen["fit_kwargs"]["fit_strategy"], "consensus")
        self.assertEqual(
            seen["fit_kwargs"]["time_kernel_type"], "quasi_periodic"
        )
        self.assertTrue(seen["fit_kwargs"]["use_acf"])
        self.assertEqual(run.status.technical_outcome, TechnicalOutcome.COMPLETED)

    def test_2d_defaults_preserve_independent_ard_initialization(self):
        case = self.make_case(
            generating_model="2D",
            mean_kind="constant",
            covariance_kind="joint_spectral_mixture_ard",
        )
        seen = {}

        def fit_runner(_lightcurve, fit_kwargs, received_case):
            seen.update(fit_kwargs)
            return self.truth_outputs(received_case)

        run_synthetic_wavelength_recovery(
            case,
            "2D",
            lightcurve_factory=lambda _: object(),
            fit_runner=fit_runner,
        )

        self.assertNotIn("fit_strategy", seen)
        self.assertTrue(seen["use_best_band_init"])
        self.assertEqual(seen["num_mixtures"], 1)

    def test_fit_exception_is_recorded_and_matrix_can_continue(self):
        case = self.make_case()

        def fail(*_args):
            raise RuntimeError("deliberate fit failure")

        run = run_synthetic_wavelength_recovery(
            case,
            "2DSeparable",
            lightcurve_factory=lambda _: object(),
            fit_runner=fail,
        )

        self.assertEqual(run.status.technical_outcome, TechnicalOutcome.FAILED)
        self.assertEqual(run.status.execution_stage, ExecutionStage.OPTIMIZATION)
        self.assertEqual(run.failure.exception_type, "RuntimeError")

    def test_stop_on_error_reraises_fit_exception(self):
        case = self.make_case()

        def fail(*_args):
            raise RuntimeError("deliberate fit failure")

        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            run_synthetic_wavelength_recovery(
                case,
                "2DSeparable",
                lightcurve_factory=lambda _: object(),
                fit_runner=fail,
                stop_on_error=True,
            )

    def test_invalid_fit_runner_payload_is_diagnostics_failure(self):
        case = self.make_case()
        run = run_synthetic_wavelength_recovery(
            case,
            "2DSeparable",
            lightcurve_factory=lambda _: object(),
            fit_runner=lambda *_args: "invalid payload",
        )

        self.assertEqual(run.status.execution_stage, ExecutionStage.DIAGNOSTICS)
        self.assertEqual(
            run.failure.failure_code, "synthetic_recovery_diagnostics_failed"
        )

    def test_unsupported_model_is_rejected(self):
        case = self.make_case()
        self.assertNotIn("2DAchromatic", SUPPORTED_SYNTHETIC_RECOVERY_MODELS)
        with self.assertRaisesRegex(ValueError, "model must be one of"):
            run_synthetic_wavelength_recovery(
                case,
                "2DAchromatic",
                lightcurve_factory=lambda _: object(),
                fit_runner=lambda *_args: {},
            )


class TestSyntheticRecoveryAggregate(RecoveryCaseMixin, unittest.TestCase):
    def make_success(self, case, model="2DWavelengthDependent"):
        return evaluate_synthetic_wavelength_recovery(
            case,
            model,
            **self.truth_outputs(case),
        )

    def make_failure(self, case, model="2D"):
        failure = WavelengthFailureRecord(
            failure_code="synthetic_recovery_fit_failed",
            stage=ExecutionStage.OPTIMIZATION,
            substage="fit_execution",
            exception_type="RuntimeError",
            message="failed",
            diagnostics={},
        )
        return evaluate_synthetic_wavelength_recovery(
            case, model, failure=failure
        )

    def test_aggregate_preserves_successes_failures_and_metric_availability(self):
        case = self.make_case()
        aggregate = aggregate_synthetic_wavelength_recovery_runs(
            [self.make_success(case), self.make_failure(case)]
        )
        payload = aggregate.to_dict()

        self.assertEqual(aggregate.n_runs, 2)
        self.assertEqual(payload["status_counts"]["completed"], 1)
        self.assertEqual(payload["status_counts"]["failed"], 1)
        self.assertEqual(payload["failure_summaries"]["n_failures"], 1)
        period = payload["metric_summaries"]["period_relative_error"]
        self.assertEqual(period["n_available"], 1)
        self.assertEqual(period["n_unavailable"], 1)
        self.assertFalse(payload["automatic_model_selection_applied"])
        self.assertNotIn("winner", payload)

    def test_aggregate_gates_use_matching_generating_model_runs(self):
        case = self.make_case()
        matched = self.make_success(case, "2DWavelengthDependent")
        mismatched_failure = self.make_failure(case, "2D")
        aggregate = aggregate_synthetic_wavelength_recovery_runs(
            [matched, mismatched_failure]
        )
        extras = aggregate.extra_fields
        gates = extras["d1_gate_summary"]

        self.assertEqual(extras["matched_truth_run_ids"], [matched.run_id])
        self.assertTrue(gates["all_evaluated_gates_passed"])
        self.assertTrue(
            gates["gates"]["matched_run_completion"]["passed"]
        )
        self.assertEqual(aggregate.failure_summaries["n_failures"], 1)

    def test_matrix_runner_returns_typed_json_safe_report(self):
        first = self.make_case(scenario_id="matrix-one", seed=1)
        second = self.make_case(scenario_id="matrix-two", seed=2)

        def fit_runner(_lightcurve, _fit_kwargs, case):
            return self.truth_outputs(case)

        report = run_synthetic_wavelength_recovery_matrix(
            [first, second],
            models=("2DWavelengthDependent", "2DSeparable"),
            lightcurve_factory=lambda _: object(),
            fit_runner=fit_runner,
            report_id="matrix-report",
        )
        payload = report.to_dict()

        self.assertEqual(len(report.runs), 4)
        self.assertEqual(report.aggregate.n_runs, 4)
        self.assertFalse(payload["automatic_model_selection_applied"])
        self.assertNotIn("selected_model", payload)
        json.dumps(payload, allow_nan=False)

    def test_report_round_trip_preserves_unknown_fields(self):
        case = self.make_case()
        run = self.make_success(case)
        aggregate = aggregate_synthetic_wavelength_recovery_runs([run])
        report = SyntheticWavelengthRecoveryReport(
            report_id="round-trip",
            runs=(run,),
            aggregate=aggregate,
        )
        payload = report.to_dict()
        payload["future_report_field"] = {"retained": True}
        rebuilt = SyntheticWavelengthRecoveryReport.from_mapping(payload)

        self.assertEqual(
            rebuilt.extra_fields["future_report_field"], {"retained": True}
        )
        json.dumps(rebuilt.to_dict(), allow_nan=False)

    def test_empty_matrix_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            run_synthetic_wavelength_recovery_matrix([])


if __name__ == "__main__":
    unittest.main()
