from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import torch
from gpytorch.constraints import Interval

from pgmuvi.lightcurve import (
    Lightcurve,
    _reconcile_consensus_spectral_mixture_scale_constraint,
)
from pgmuvi.single_source_analysis import (
    preserve_single_source_analysis_state,
    resolve_single_source_period_component_configuration,
)
from pgmuvi.wavelength_constraint_tutorial import (
    load_wavelength_constraint_tutorial_lightcurve,
)


class TestConsensusMulticompConstraintReconciliation(unittest.TestCase):
    def _lightcurve_with_spectral_mixture_model(self):
        time = np.linspace(0.0, 500.0, 20)
        xdata = np.vstack(
            [
                np.column_stack(
                    [time, np.full_like(time, wavelength)]
                )
                for wavelength in (0.6, 1.2)
            ]
        )
        ydata = np.concatenate(
            [
                2.0
                + np.sin(2.0 * np.pi * time / 120.0)
                + 0.3 * np.sin(2.0 * np.pi * time / 45.0)
                for _ in (0.6, 1.2)
            ]
        )
        lightcurve = Lightcurve(
            xdata,
            ydata,
            yerr=np.full_like(ydata, 0.05),
            band=np.asarray(["A"] * 20 + ["B"] * 20),
            center_time=False,
            check_sampling=False,
            max_samples=None,
        )
        lightcurve.set_model(
            "2DWavelengthDependent",
            time_kernel_type="spectral_mixture",
            wavelength_kernel_type="rbf",
            num_mixtures=2,
        )
        return lightcurve

    def test_expands_scale_interval_without_clipping_consensus_guess(self):
        lightcurve = self._lightcurve_with_spectral_mixture_model()
        kernel = lightcurve.model.covar_module.kernels[0]
        kernel.register_constraint(
            "raw_mixture_scales",
            Interval(
                torch.tensor([[[1.0e-6]]], dtype=torch.float64),
                torch.tensor(
                    [[[1.535080428963092e-4]]],
                    dtype=torch.float64,
                ),
            ),
            replace=True,
        )
        requested = torch.tensor(
            [
                [[8.439161306553325e-5]],
                [[2.2262447272708517e-4]],
            ],
            dtype=torch.float64,
        )
        guess = {
            "covar_module.kernels.0.mixture_scales": requested
        }

        diagnostics = (
            _reconcile_consensus_spectral_mixture_scale_constraint(
                lightcurve.model,
                guess,
            )
        )

        self.assertTrue(diagnostics["adjusted"])
        self.assertEqual(
            diagnostics["status"],
            "expanded_to_include_consensus_guess",
        )
        self.assertGreater(
            diagnostics["requested_maximum"],
            diagnostics["old_upper_bound"][0],
        )
        self.assertGreater(
            diagnostics["new_upper_bound"][0],
            diagnostics["requested_maximum"],
        )
        self.assertAlmostEqual(
            diagnostics["old_lower_bound"][0],
            diagnostics["new_lower_bound"][0],
        )

        lightcurve.set_hypers(dict(guess))
        realized = kernel.mixture_scales.detach()
        self.assertTrue(torch.isfinite(realized).all())
        self.assertTrue(torch.allclose(realized, requested))

    def test_does_not_change_an_interval_that_already_contains_guess(self):
        lightcurve = self._lightcurve_with_spectral_mixture_model()
        kernel = lightcurve.model.covar_module.kernels[0]
        requested = torch.full(
            (2, 1, 1),
            1.0e-4,
            dtype=torch.float64,
        )
        kernel.register_constraint(
            "raw_mixture_scales",
            Interval(
                torch.tensor([[[1.0e-6]]], dtype=torch.float64),
                torch.tensor([[[3.0e-4]]], dtype=torch.float64),
            ),
            replace=True,
        )
        diagnostics = (
            _reconcile_consensus_spectral_mixture_scale_constraint(
                lightcurve.model,
                {
                    "covar_module.kernels.0.mixture_scales":
                        requested
                },
            )
        )
        self.assertFalse(diagnostics["adjusted"])
        self.assertEqual(
            diagnostics["status"],
            "already_contains_consensus_guess",
        )
        self.assertEqual(
            diagnostics["old_upper_bound"],
            diagnostics["new_upper_bound"],
        )

    def test_bundled_source_notebook_path_reaches_fitted_state(self):
        repository = Path(__file__).resolve().parents[1]
        source = repository / "examples/data/10131+3049.csv"
        configuration = (
            resolve_single_source_period_component_configuration(
                ls_num_components=3,
                gp_num_components=2,
            )
        )

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
                    name="two-component parameter-workflow regression",
                )
            )
            fit_kwargs = {
                "model": "2DWavelengthDependent",
                "fit_strategy": configuration["fit_strategy"],
                "time_kernel_type": configuration[
                    "time_kernel_type"
                ],
                "wavelength_kernel_type": "rbf",
                "constraint_set": "LPV",
                "training_iter": 1,
                "miniter": 0,
                "optim": "Adam",
                "lr": 0.03,
                "learn_additional_noise": True,
                "use_parameter_workflow": True,
                "verbose": False,
            }
            fit_kwargs.update(configuration["fit_kwargs"])
            result = lightcurve.fit(**fit_kwargs)

        self.assertTrue(lightcurve.is_fitted)
        self.assertIn("loss", result)
        latest_fit = lightcurve.get_fit_history()[-1]
        self.assertTrue(latest_fit["success"])
        self.assertFalse(latest_fit["failed"])
        reconciliation = getattr(
            lightcurve,
            "_consensus_multicomp_scale_constraint_reconciliation",
        )
        self.assertTrue(reconciliation["adjusted"])
        self.assertGreater(
            reconciliation["new_upper_bound"][0],
            reconciliation["requested_maximum"],
        )
        summary = lightcurve.get_period_summary(
            n_peaks=2,
            prefer_fitted_psd=True,
        )
        self.assertTrue(summary["is_multicomponent"])
        self.assertEqual(
            summary.component_diagnostics.n_components,
            2,
        )


if __name__ == "__main__":
    unittest.main()
