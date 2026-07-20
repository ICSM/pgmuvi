"""Reproducibility contracts for representative LPV D3 execution."""

from __future__ import annotations

import json
import random
import unittest

import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.wavelength_validation_real_lpv import (
    RepresentativeLPVBatchReport,
    RepresentativeLPVSourceSpecification,
    build_representative_lpv_runtime_environment,
    run_representative_lpv_validation_batch,
)


def _lightcurve(name):
    return Lightcurve(
        torch.tensor(
            [
                [0.0, 0.55],
                [1.0, 0.55],
                [0.0, 1.25],
                [1.0, 1.25],
            ],
            dtype=torch.float64,
        ),
        torch.tensor(
            [10.0, 11.0, 8.0, 9.0],
            dtype=torch.float64,
        ),
        yerr=torch.tensor(
            [0.2, 0.2, 0.3, 0.3],
            dtype=torch.float64,
        ),
        band=np.asarray(
            [
                "instrument-v",
                "instrument-v",
                "instrument-j",
                "instrument-j",
            ],
            dtype=str,
        ),
        xtransform=None,
        center_time=False,
        check_sampling=False,
        max_samples=None,
        max_samples_per_band=None,
        name=name,
    )


def _workflow_report():
    return {
        "kind": "period_independent_wavelength_advisory_workflow",
        "advisory_only": True,
        "automatic_model_selection_applied": False,
        "selected_model": None,
        "fit_quality_ranking_status": "unavailable",
        "fit_quality_ranking_available": False,
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
            "model_kernel_config_results": [],
        },
        "quality_report": {
            "fit_quality_ranking_status": "unavailable",
            "fit_quality_ranking_available": False,
            "n_with_fit_quality": 0,
            "top_ranked_model": None,
            "ranked_results": [],
        },
    }


class TestRepresentativeLPVRuntimeEnvironment(unittest.TestCase):
    def test_runtime_environment_is_compact_and_json_safe(self):
        payload = build_representative_lpv_runtime_environment()

        required = (
            "python_version",
            "python_implementation",
            "python_executable",
            "platform",
            "pgmuvi_version",
            "numpy_version",
            "torch_version",
            "gpytorch_version",
            "cuda_available",
        )
        for key in required:
            with self.subTest(key=key):
                self.assertIn(key, payload)

        json.dumps(payload, allow_nan=False)


class TestRepresentativeLPVSourceSeed(unittest.TestCase):
    def setUp(self):
        self.manifest = (
            RepresentativeLPVSourceSpecification(
                source_id="seeded-source",
                source_path="inputs/source.csv",
                description="Seeded representative LPV.",
                sample_role="reproducibility",
                selection_reason="Exercises deterministic source execution.",
                seed=135,
            ),
        )

    def _run_and_capture(self):
        draws = {}

        def loader(specification):
            draws["loader"] = (
                random.random(),
                float(np.random.random()),
                float(torch.rand(())),
            )
            return _lightcurve(specification.source_id)

        def workflow_runner(lightcurve, **kwargs):
            del lightcurve, kwargs
            draws["workflow"] = (
                random.random(),
                float(np.random.random()),
                float(torch.rand(())),
            )
            return _workflow_report()

        report = run_representative_lpv_validation_batch(
            self.manifest,
            source_loader=loader,
            workflow_runner=workflow_runner,
            workflow_kwargs={
                "base_fit_kwargs": {
                    "fit_strategy": "consensus",
                    "time_kernel_type": "quasi_periodic",
                    "learn_additional_noise": True,
                },
            },
        )
        return draws, report

    def test_same_source_seed_repeats_loader_and_workflow_draws(self):
        first_draws, first_report = self._run_and_capture()
        second_draws, second_report = self._run_and_capture()

        self.assertEqual(first_draws, second_draws)

        first_row = first_report.to_dict()["source_results"][0]
        second_row = second_report.to_dict()["source_results"][0]

        self.assertEqual(first_row["execution_seed"], 135)
        self.assertTrue(first_row["seed_applied"])
        self.assertEqual(
            first_row["seed_scope"],
            "source_loading_and_advisory_workflow",
        )
        self.assertEqual(first_row, second_row)

    def test_batch_restores_caller_rng_states(self):
        random.seed(90210)
        np.random.seed(90210)
        torch.manual_seed(90210)

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()

        self._run_and_capture()

        self.assertEqual(random.getstate(), python_state)

        restored_numpy_state = np.random.get_state()
        self.assertEqual(
            restored_numpy_state[0],
            numpy_state[0],
        )
        np.testing.assert_array_equal(
            restored_numpy_state[1],
            numpy_state[1],
        )
        self.assertEqual(
            restored_numpy_state[2:],
            numpy_state[2:],
        )
        self.assertTrue(
            torch.equal(
                torch.random.get_rng_state(),
                torch_state,
            )
        )


class TestRepresentativeLPVExecutionProvenance(unittest.TestCase):
    def test_batch_report_retains_workflow_and_runtime_provenance(self):
        specification = RepresentativeLPVSourceSpecification(
            source_id="provenance-source",
            source_path="inputs/source.csv",
            description="Representative LPV provenance source.",
            sample_role="reproducibility",
            selection_reason="Exercises execution provenance.",
            seed=7,
        )
        workflow_configuration = {
            "base_fit_kwargs": {
                "fit_strategy": "consensus",
                "time_kernel_type": "quasi_periodic",
                "learn_additional_noise": True,
                "training_iter": 500,
                "miniter": 100,
            },
            "make_plots": False,
        }

        report = run_representative_lpv_validation_batch(
            (specification,),
            source_loader=lambda item: _lightcurve(item.source_id),
            workflow_runner=lambda lightcurve, **kwargs: (
                _workflow_report()
            ),
            workflow_kwargs=workflow_configuration,
        )

        payload = report.to_dict()
        self.assertEqual(
            payload["workflow_configuration"],
            workflow_configuration,
        )
        self.assertIn(
            "python_version",
            payload["runtime_environment"],
        )
        self.assertIn(
            "torch_version",
            payload["runtime_environment"],
        )

        rebuilt = RepresentativeLPVBatchReport.from_mapping(payload)
        self.assertEqual(rebuilt.to_dict(), payload)
        json.dumps(payload, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
