from __future__ import annotations

from pathlib import Path
from unittest import mock
import unittest

import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.single_source_analysis import (
    preserve_single_source_analysis_state,
    resolve_single_source_period_component_configuration,
)
from pgmuvi.wavelength_constraint_tutorial import (
    load_wavelength_constraint_tutorial_lightcurve,
)


class TestConsensusParameterWorkflowControl(unittest.TestCase):
    def _small_lightcurve(self):
        time = np.linspace(0.0, 300.0, 24)
        xdata = np.vstack(
            [
                np.column_stack(
                    [time, np.full_like(time, wavelength)]
                )
                for wavelength in (0.8, 1.2)
            ]
        )
        ydata = np.concatenate(
            [
                2.0 + np.sin(2.0 * np.pi * time / 120.0)
                for _ in (0.8, 1.2)
            ]
        )
        return Lightcurve(
            xdata,
            ydata,
            yerr=np.full_like(ydata, 0.05),
            band=np.asarray(["A"] * 24 + ["B"] * 24),
            center_time=False,
            check_sampling=False,
            max_samples=None,
        )

    def test_fit_core_forwards_disabled_flag_to_consensus_dispatch(self):
        lightcurve = self._small_lightcurve()
        with mock.patch.object(
            lightcurve,
            "_consensus_fit",
            return_value={"loss": [1.0]},
        ) as consensus_fit:
            lightcurve._fit_core(
                model="2DWavelengthDependent",
                fit_strategy="consensus",
                use_parameter_workflow=False,
                training_iter=1,
                miniter=0,
            )
        self.assertFalse(
            consensus_fit.call_args.kwargs[
                "use_parameter_workflow"
            ]
        )

    def test_fit_core_forwards_enabled_flag_to_consensus_dispatch(self):
        lightcurve = self._small_lightcurve()
        with mock.patch.object(
            lightcurve,
            "_consensus_fit",
            return_value={"loss": [1.0]},
        ) as consensus_fit:
            lightcurve._fit_core(
                model="2DWavelengthDependent",
                fit_strategy="consensus_multicomp",
                use_parameter_workflow=True,
                training_iter=1,
                miniter=0,
            )
        self.assertTrue(
            consensus_fit.call_args.kwargs[
                "use_parameter_workflow"
            ]
        )

    def test_public_parameter_context_preview_applies_nothing(self):
        lightcurve = self._small_lightcurve()
        context = lightcurve.get_parameter_estimation_context()
        self.assertTrue(context.wavelength_diagnostics.available)
        self.assertIsNone(lightcurve.parameter_workflow_result)
        self.assertFalse(lightcurve.is_fitted)

    def test_bundled_source_enabled_disabled_consensus_comparison(self):
        repository = Path(__file__).resolve().parents[1]
        source = repository / "examples/data/10131+3049.csv"
        configuration = (
            resolve_single_source_period_component_configuration(
                ls_num_components=3,
                gp_num_components=2,
            )
        )

        lightcurves = []
        summaries = []
        results = []
        for workflow_enabled in (True, False):
            with preserve_single_source_analysis_state(
                seed=174,
                default_dtype=torch.float64,
                working_directory=repository,
            ):
                lightcurve, _ = (
                    load_wavelength_constraint_tutorial_lightcurve(
                        source,
                        max_samples_per_observational_channel=50,
                        check_sampling=True,
                        sampling_kwargs=None,
                        name=(
                            "workflow enabled"
                            if workflow_enabled
                            else "workflow disabled"
                        ),
                    )
                )
                result = lightcurve.fit(
                    model="2DWavelengthDependent",
                    fit_strategy=configuration["fit_strategy"],
                    time_kernel_type=configuration[
                        "time_kernel_type"
                    ],
                    wavelength_kernel_type="rbf",
                    num_mixtures=configuration["fit_kwargs"].get(
                        "num_mixtures"
                    ),
                    constraint_set="LPV",
                    training_iter=1,
                    miniter=0,
                    optim="Adam",
                    lr=0.03,
                    learn_additional_noise=True,
                    use_parameter_workflow=workflow_enabled,
                    verbose=False,
                )
            self.assertTrue(lightcurve.is_fitted)
            self.assertIn("loss", result)
            self.assertTrue(
                lightcurve.get_fit_history()[-1]["success"]
            )
            lightcurves.append(lightcurve)
            results.append(result)
            summaries.append(
                lightcurve.get_period_summary(
                    n_peaks=2,
                    prefer_fitted_psd=True,
                )
            )

        self.assertTrue(
            lightcurves[0].get_parameter_workflow_report()[
                "available"
            ]
        )
        self.assertFalse(
            lightcurves[1].get_parameter_workflow_report()[
                "available"
            ]
        )
        self.assertEqual(
            lightcurves[0].consensus_diagnostics[
                "consensus_frequencies"
            ],
            lightcurves[1].consensus_diagnostics[
                "consensus_frequencies"
            ],
        )
        self.assertEqual(
            summaries[0].component_diagnostics.n_components,
            2,
        )
        self.assertEqual(
            summaries[1].component_diagnostics.n_components,
            2,
        )
        self.assertEqual(len(results[0]["loss"]), 1)
        self.assertEqual(len(results[1]["loss"]), 1)

    def test_bundled_source_control_reloads_are_bitwise_identical(self):
        repository = Path(__file__).resolve().parents[1]
        source = repository / "examples/data/10131+3049.csv"
        loaded = []

        for name in ("enabled control", "disabled control"):
            with preserve_single_source_analysis_state(
                seed=174,
                default_dtype=torch.float64,
                working_directory=repository,
            ):
                lightcurve, summary = (
                    load_wavelength_constraint_tutorial_lightcurve(
                        source,
                        max_samples_per_observational_channel=50,
                        check_sampling=True,
                        sampling_kwargs=None,
                        name=name,
                    )
                )
            loaded.append((lightcurve, summary))

        enabled, enabled_summary = loaded[0]
        disabled, disabled_summary = loaded[1]
        controlled_keys = (
            "check_sampling",
            "sampling_kwargs",
            "n_rows_original",
            "n_rows_excluded_by_validity_policy",
            "n_rows_eligible",
            "n_rows_before_sampling_quality_filter",
            "n_rows_removed_by_sampling_quality_filter",
            "n_rows_retained",
        )
        for key in controlled_keys:
            self.assertEqual(
                enabled_summary[key],
                disabled_summary[key],
            )

        self.assertEqual(enabled.xdata.dtype, torch.float64)
        self.assertEqual(disabled.xdata.dtype, torch.float64)
        self.assertTrue(torch.equal(enabled.xdata, disabled.xdata))
        self.assertTrue(torch.equal(enabled.ydata, disabled.ydata))
        self.assertTrue(torch.equal(enabled.yerr, disabled.yerr))
        self.assertTrue(
            np.array_equal(
                enabled.observational_channel_labels,
                disabled.observational_channel_labels,
            )
        )


if __name__ == "__main__":
    unittest.main()
