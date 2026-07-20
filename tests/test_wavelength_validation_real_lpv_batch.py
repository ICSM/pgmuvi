"""Failure-aware batch contracts for representative observed-LPV validation."""

from __future__ import annotations

import json
import unittest

import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.wavelength_validation_real_lpv import (
    DEFAULT_REPRESENTATIVE_LPV_MODELS,
    RepresentativeLPVBatchReport,
    RepresentativeLPVSourceSpecification,
    run_representative_lpv_validation_batch,
    validate_representative_lpv_source_manifest,
)


class BatchFixtures:
    @staticmethod
    def lightcurve(name):
        x = torch.tensor(
            [
                [0.0, 0.55],
                [1.0, 0.55],
                [0.0, 1.25],
                [1.0, 1.25],
            ],
            dtype=torch.float64,
        )
        y = torch.tensor(
            [10.0, 11.0, 8.0, 9.0],
            dtype=torch.float64,
        )
        yerr = torch.tensor(
            [0.2, 0.2, 0.3, 0.3],
            dtype=torch.float64,
        )
        return Lightcurve(
            x,
            y,
            yerr=yerr,
            band=np.asarray(
                ["instrument-v", "instrument-v", "instrument-j", "instrument-j"],
                dtype=str,
            ),
            xtransform=None,
            center_time=False,
            check_sampling=False,
            max_samples=None,
            max_samples_per_band=None,
            name=name,
        )

    @staticmethod
    def successful_workflow(model="2DDustMean"):
        return {
            "kind": "period_independent_wavelength_advisory_workflow",
            "advisory_only": True,
            "automatic_model_selection_applied": False,
            "selected_model": None,
            "fit_quality_ranking_status": "single_valid_candidate",
            "fit_quality_ranking_available": False,
            "only_valid_model": model,
            "top_ranked_model": None,
            "model_kernel_config_report": {
                "period_independent_diagnostics": {
                    "kind": (
                        "period_independent_wavelength_structure_diagnostics"
                    ),
                    "n_usable_observational_channels": 2,
                    "n_distinct_physical_wavelengths": 2,
                },
            },
            "run_report": {
                "model_kernel_config_results": [
                    {
                        "model_kernel_config_id": f"{model}-config",
                        "model": model,
                        "status": "passed",
                        "fit_success": True,
                        "fit_failed": False,
                        "technical_outcome": "completed",
                        "diagnostic_validity": "valid",
                        "scientific_usability": "usable",
                        "comparison_eligibility": "eligible",
                        "warning_count": 0,
                        "warnings": [],
                        "consensus_success": True,
                        "consensus_period": 410.0,
                        "consensus_frequency": 1.0 / 410.0,
                        "n_accepted_bands": 2,
                        "n_rejected_bands": 0,
                        "fit_kwargs": {
                            "model": model,
                            "fit_strategy": "consensus",
                            "time_kernel_type": "quasi_periodic",
                            "learn_additional_noise": True,
                        },
                        "training_nrmse_by_target_scale": 0.15,
                        "n_sm_ard_boundary_hits": 0,
                        "sm_ard_boundary_hits": [],
                    }
                ],
            },
            "quality_report": {
                "score_kind": "training_residual_fit_quality",
                "fit_quality_ranking_status": "single_valid_candidate",
                "fit_quality_ranking_available": False,
                "n_with_fit_quality": 1,
                "only_valid_model": model,
                "top_ranked_model": None,
                "ranked_results": [],
            },
            "advisory_conclusions": [],
            "unresolved_ambiguities": [
                {
                    "code": "single_valid_candidate",
                    "summary": (
                        "Only one candidate supplied usable fit-quality "
                        "diagnostics."
                    ),
                }
            ],
        }


class TestRepresentativeLPVSourceSpecification(unittest.TestCase):
    def test_source_specification_round_trip_is_json_safe(self):
        specification = RepresentativeLPVSourceSpecification(
            source_id="lpv-a",
            source_path="inputs/lpv-a.csv",
            description="Representative oxygen-rich LPV.",
            sample_role="oxygen_rich",
            selection_reason="Public source with broad wavelength coverage.",
            seed=17,
            metadata={"catalog": "public-example"},
        )

        payload = specification.to_dict()
        rebuilt = RepresentativeLPVSourceSpecification.from_mapping(payload)

        self.assertEqual(rebuilt.to_dict(), payload)
        self.assertEqual(payload["seed"], 17)
        json.dumps(payload, allow_nan=False)

    def test_required_manifest_fields_cannot_be_empty(self):
        required_fields = {
            "source_id": "lpv-a",
            "source_path": "inputs/lpv-a.csv",
            "description": "Representative LPV.",
            "sample_role": "validation",
            "selection_reason": "Representative wavelength coverage.",
        }

        for field_name in required_fields:
            with self.subTest(field_name=field_name):
                values = dict(required_fields)
                values[field_name] = ""
                with self.assertRaisesRegex(ValueError, field_name):
                    RepresentativeLPVSourceSpecification(**values)

    def test_seed_must_be_nonnegative(self):
        with self.assertRaisesRegex(ValueError, "seed"):
            RepresentativeLPVSourceSpecification(
                source_id="lpv-a",
                source_path="inputs/lpv-a.csv",
                description="Representative LPV.",
                sample_role="validation",
                selection_reason="Representative wavelength coverage.",
                seed=-1,
            )


class TestRepresentativeLPVManifestValidation(unittest.TestCase):
    def test_manifest_rejects_duplicate_source_ids(self):
        manifest = [
            {
                "source_id": "duplicate",
                "source_path": "inputs/a.csv",
                "description": "Source A.",
                "sample_role": "validation",
                "selection_reason": "Reason A.",
            },
            {
                "source_id": "duplicate",
                "source_path": "inputs/b.csv",
                "description": "Source B.",
                "sample_role": "validation",
                "selection_reason": "Reason B.",
            },
        ]

        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_representative_lpv_source_manifest(manifest)

    def test_manifest_preserves_input_order(self):
        normalized = validate_representative_lpv_source_manifest(
            [
                {
                    "source_id": "lpv-b",
                    "source_path": "inputs/b.csv",
                    "description": "Source B.",
                    "sample_role": "carbon_rich",
                    "selection_reason": "Carbon-rich representative.",
                },
                {
                    "source_id": "lpv-a",
                    "source_path": "inputs/a.csv",
                    "description": "Source A.",
                    "sample_role": "oxygen_rich",
                    "selection_reason": "Oxygen-rich representative.",
                },
            ]
        )

        self.assertEqual(
            tuple(item.source_id for item in normalized),
            ("lpv-b", "lpv-a"),
        )


class TestRepresentativeLPVBatchRunner(BatchFixtures, unittest.TestCase):
    def setUp(self):
        self.manifest = [
            RepresentativeLPVSourceSpecification(
                source_id="good-source",
                source_path="inputs/good.csv",
                description="Source expected to complete.",
                sample_role="validation",
                selection_reason="Exercises successful D3 reporting.",
                seed=11,
            ),
            RepresentativeLPVSourceSpecification(
                source_id="load-failure",
                source_path="inputs/missing.csv",
                description="Source expected to fail while loading.",
                sample_role="failure_boundary",
                selection_reason="Exercises setup-failure continuation.",
                seed=12,
            ),
            RepresentativeLPVSourceSpecification(
                source_id="workflow-failure",
                source_path="inputs/workflow.csv",
                description="Source expected to fail during workflow.",
                sample_role="failure_boundary",
                selection_reason="Exercises workflow-failure continuation.",
                seed=13,
            ),
        ]

    def test_batch_continues_after_source_failures(self):
        loader_calls = []
        workflow_calls = []

        def source_loader(specification):
            loader_calls.append(specification.source_id)
            if specification.source_id == "load-failure":
                raise FileNotFoundError("Synthetic missing-source failure.")
            return self.lightcurve(specification.source_id)

        def workflow_runner(lightcurve, **kwargs):
            workflow_calls.append(
                {
                    "name": lightcurve.name,
                    "kwargs": dict(kwargs),
                }
            )
            if lightcurve.name == "workflow-failure":
                raise RuntimeError("Synthetic workflow failure.")
            return self.successful_workflow()

        report = run_representative_lpv_validation_batch(
            self.manifest,
            source_loader=source_loader,
            workflow_runner=workflow_runner,
            output_root="validation_outputs/d3_real_lpv",
        )

        self.assertEqual(
            loader_calls,
            ["good-source", "load-failure", "workflow-failure"],
        )
        self.assertEqual(
            [item["name"] for item in workflow_calls],
            ["good-source", "workflow-failure"],
        )

        payload = report.to_dict()
        self.assertEqual(payload["n_sources"], 3)
        self.assertEqual(payload["n_completed_sources"], 1)
        self.assertEqual(payload["n_failed_sources"], 2)
        self.assertEqual(
            payload["source_status_counts"],
            {
                "completed": 1,
                "failed": 2,
            },
        )

        rows = {
            row["source_id"]: row
            for row in payload["source_results"]
        }

        self.assertEqual(rows["good-source"]["status"], "completed")
        self.assertIsNotNone(
            rows["good-source"]["validation_report"]
        )

        self.assertEqual(rows["load-failure"]["status"], "failed")
        self.assertEqual(
            rows["load-failure"]["failure"]["stage"],
            "source_loading",
        )
        self.assertEqual(
            rows["load-failure"]["failure"]["exception_type"],
            "FileNotFoundError",
        )

        self.assertEqual(
            rows["workflow-failure"]["failure"]["stage"],
            "advisory_workflow",
        )
        self.assertEqual(
            rows["workflow-failure"]["failure"]["exception_type"],
            "RuntimeError",
        )

    def test_batch_uses_explicit_model_order_and_output_paths(self):
        captured = []

        def source_loader(specification):
            return self.lightcurve(specification.source_id)

        def workflow_runner(lightcurve, **kwargs):
            captured.append(
                {
                    "name": lightcurve.name,
                    "kwargs": dict(kwargs),
                }
            )
            return self.successful_workflow()

        report = run_representative_lpv_validation_batch(
            self.manifest[:1],
            source_loader=source_loader,
            workflow_runner=workflow_runner,
            output_root="validation_outputs/d3_real_lpv",
        )

        self.assertEqual(
            tuple(captured[0]["kwargs"]["include_models"]),
            DEFAULT_REPRESENTATIVE_LPV_MODELS,
        )
        self.assertTrue(
            captured[0]["kwargs"]["include_2d_baseline"]
        )

        row = report.to_dict()["source_results"][0]
        self.assertEqual(
            row["source_output_dir"],
            "validation_outputs/d3_real_lpv/sources/good-source",
        )
        self.assertEqual(
            row["source_report_path"],
            (
                "validation_outputs/d3_real_lpv/sources/"
                "good-source/report.json"
            ),
        )

    def test_batch_report_is_advisory_nonselecting_and_json_safe(self):
        def source_loader(specification):
            return self.lightcurve(specification.source_id)

        def workflow_runner(lightcurve, **kwargs):
            del lightcurve, kwargs
            return self.successful_workflow()

        report = run_representative_lpv_validation_batch(
            self.manifest[:1],
            source_loader=source_loader,
            workflow_runner=workflow_runner,
            output_root="validation_outputs/d3_real_lpv",
        )

        self.assertIsInstance(report, RepresentativeLPVBatchReport)

        payload = report.to_dict()
        self.assertTrue(payload["advisory_only"])
        self.assertFalse(
            payload["automatic_model_selection_applied"]
        )
        self.assertIsNone(payload["selected_model"])
        self.assertFalse(payload["writes_outputs"])
        self.assertFalse(payload["automatic_constraints_applied"])
        self.assertFalse(payload["automatic_initialization_applied"])

        rebuilt = RepresentativeLPVBatchReport.from_mapping(payload)
        self.assertEqual(rebuilt.to_dict(), payload)
        json.dumps(payload, allow_nan=False)

    def test_batch_does_not_mutate_manifest_objects(self):
        before = [item.to_dict() for item in self.manifest]

        def source_loader(specification):
            return self.lightcurve(specification.source_id)

        def workflow_runner(lightcurve, **kwargs):
            del lightcurve, kwargs
            return self.successful_workflow()

        run_representative_lpv_validation_batch(
            self.manifest,
            source_loader=source_loader,
            workflow_runner=workflow_runner,
            output_root="validation_outputs/d3_real_lpv",
        )

        after = [item.to_dict() for item in self.manifest]
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
