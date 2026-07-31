"""Tests for user-controlled LS/GP component counts and GP summary plots."""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pgmuvi.lightcurve import (
    ComponentDiagnosticsResult,
    Lightcurve,
    PeriodPeakResult,
    PeriodSummaryResult,
)
from pgmuvi.single_source_analysis import (
    resolve_single_source_period_component_configuration,
)


class TestPeriodComponentConfiguration(unittest.TestCase):
    def test_single_component_uses_quasi_periodic_consensus(self):
        configuration = (
            resolve_single_source_period_component_configuration(
                ls_num_components=4,
                gp_num_components=1,
            )
        )
        self.assertEqual(configuration["ls_num_components"], 4)
        self.assertEqual(configuration["gp_num_components"], 1)
        self.assertEqual(configuration["fit_strategy"], "consensus")
        self.assertEqual(
            configuration["time_kernel_type"],
            "quasi_periodic",
        )
        self.assertEqual(configuration["fit_kwargs"], {})

    def test_multicomponent_gp_count_is_independent_of_ls_count(self):
        configuration = (
            resolve_single_source_period_component_configuration(
                ls_num_components=2,
                gp_num_components=4,
            )
        )
        self.assertEqual(configuration["ls_num_components"], 2)
        self.assertEqual(configuration["gp_num_components"], 4)
        self.assertEqual(
            configuration["fit_strategy"],
            "consensus_multicomp",
        )
        self.assertEqual(
            configuration["time_kernel_type"],
            "spectral_mixture",
        )
        self.assertEqual(
            configuration["fit_kwargs"]["num_mixtures"],
            4,
        )
        self.assertEqual(
            configuration["fit_kwargs"]["max_components_per_band"],
            2,
        )

    def test_component_counts_must_be_positive_integers(self):
        for keyword, value in (
            ("ls_num_components", 0),
            ("gp_num_components", -1),
            ("ls_num_components", True),
            ("gp_num_components", 1.5),
        ):
            arguments = {
                "ls_num_components": 2,
                "gp_num_components": 2,
            }
            arguments[keyword] = value
            with self.subTest(keyword=keyword, value=value):
                with self.assertRaises((TypeError, ValueError)):
                    resolve_single_source_period_component_configuration(
                        **arguments
                    )


class TestPeriodSummaryAxisAndComponents(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    @staticmethod
    def _lightcurve():
        time = torch.linspace(0.0, 10.0, 20, dtype=torch.float64)
        flux = torch.sin(time)
        return Lightcurve(
            time,
            flux,
            center_time=False,
            check_sampling=False,
            max_samples=None,
        )

    @staticmethod
    def _spectral_summary():
        frequency = np.logspace(-3.0, -1.0, 400)
        means = np.asarray([0.01, 0.025])
        scales = np.asarray([0.0012, 0.0020])
        weights = np.asarray([1.0, 0.6])
        components = [
            weight
            * np.exp(-0.5 * ((frequency - mean) / scale) ** 2)
            for mean, scale, weight in zip(
                means,
                scales,
                weights,
                strict=True,
            )
        ]
        psd = np.sum(components, axis=0)
        peaks = [
            PeriodPeakResult(
                rank=1,
                frequency=means[0],
                period=1.0 / means[0],
                height=1.0,
                prominence=1.0,
                area_fraction=0.6,
                interval_frequency=(0.008, 0.012),
                interval_period=(1.0 / 0.012, 1.0 / 0.008),
                coherence_proxy=10.0,
            ),
            PeriodPeakResult(
                rank=2,
                frequency=means[1],
                period=1.0 / means[1],
                height=0.6,
                prominence=0.5,
                area_fraction=0.4,
                interval_frequency=(0.021, 0.029),
                interval_period=(1.0 / 0.029, 1.0 / 0.021),
                coherence_proxy=8.0,
            ),
        ]
        diagnostics = ComponentDiagnosticsResult(
            component_periods=1.0 / means,
            component_frequencies=means,
            component_weights=weights,
            component_period_scales=np.asarray([10.0, 4.0]),
            component_frequency_scales=scales,
            n_components=2,
            kernel_family="SpectralMixtureKernel",
        )
        return PeriodSummaryResult(
            method="spectral_mixture_psd_peak",
            backend="spectral_mixture",
            kernel_family="SpectralMixtureKernel",
            time_kernel_family="SpectralMixtureKernel",
            n_peaks_detected=2,
            n_peaks_analyzed=2,
            n_peaks_requested=2,
            dominant_period=100.0,
            dominant_frequency=0.01,
            peaks=peaks,
            freq_grid=frequency,
            psd=psd,
            component_diagnostics=diagnostics,
        )

    def test_spectral_summary_can_use_period_axis_and_component_curves(self):
        lightcurve = self._lightcurve()
        summary = self._spectral_summary()
        figure, axis = lightcurve.plot_period_summary(
            summary=summary,
            show=False,
            x_axis="period",
            log_x=True,
            log_y=False,
            show_components=True,
            max_peaks_to_mark=2,
        )
        self.assertEqual(axis.get_xlabel(), "Period")
        self.assertEqual(axis.get_xscale(), "log")
        labels = [line.get_label() for line in axis.lines]
        self.assertIn("Summed PSD", labels)
        self.assertIn("SM component 1", labels)
        self.assertIn("SM component 2", labels)
        self.assertGreater(len(figure.axes), 1)

    def test_component_curves_are_optional(self):
        lightcurve = self._lightcurve()
        summary = self._spectral_summary()
        figure, axis = lightcurve.plot_period_summary(
            summary=summary,
            show=False,
            x_axis="period",
            log_y=False,
            show_components=False,
        )
        labels = [line.get_label() for line in axis.lines]
        self.assertIn("Summed PSD", labels)
        self.assertFalse(
            any(label.startswith("SM component") for label in labels)
        )

    def test_explicit_period_summary_has_no_quantitative_y_axis(self):
        lightcurve = self._lightcurve()
        peak = PeriodPeakResult(
            rank=1,
            frequency=0.01,
            period=100.0,
            interval_frequency=(1.0 / 120.0, 1.0 / 80.0),
            interval_period=(80.0, 120.0),
        )
        summary = PeriodSummaryResult(
            method="explicit_period_parameter",
            backend="explicit_period",
            dominant_period=100.0,
            dominant_frequency=0.01,
            peaks=[peak],
            freq_grid=None,
            psd=None,
            interval_definition=(
                "coherence_proxy_from_rbf_lengthscale"
            ),
        )
        figure, axis = lightcurve.plot_period_summary(
            summary=summary,
            show=False,
            x_axis="period",
            log_x=True,
        )
        self.assertEqual(axis.get_xlabel(), "Period")
        self.assertEqual(axis.get_xscale(), "log")
        self.assertEqual(axis.get_ylabel(), "")
        self.assertEqual(axis.get_yticks().size, 0)
        lower, upper = axis.get_xlim()
        self.assertLess(lower, 100.0)
        self.assertLess(100.0, upper)

    def test_multicomp_consensus_can_request_fitted_psd_summary(self):
        lightcurve = self._lightcurve()
        lightcurve.set_model("1D", num_mixtures=2)
        lightcurve.consensus_diagnostics = {
            "fit_strategy": "consensus_multicomp",
            "consensus_success": True,
            "consensus_periods": [100.0, 40.0],
            "consensus_period_widths": [10.0, 4.0],
            "fitted_mixture_periods": [98.0, 41.0],
            "initialized_mixture_periods": [101.0, 39.0],
            "consensus_component_strengths": [1.0, 0.5],
            "multicomponent_period_summaries": [
                {
                    "component_index": 0,
                    "consensus_period": 100.0,
                    "consensus_period_width": 10.0,
                    "fitted_mixture_period": 98.0,
                    "initialized_mixture_period": 101.0,
                    "consensus_component_strength": 1.0,
                    "source_cluster_id": 0,
                    "member_bands": ["A", "B"],
                },
                {
                    "component_index": 1,
                    "consensus_period": 40.0,
                    "consensus_period_width": 4.0,
                    "fitted_mixture_period": 41.0,
                    "initialized_mixture_period": 39.0,
                    "consensus_component_strength": 0.5,
                    "source_cluster_id": 1,
                    "member_bands": ["A", "B"],
                },
            ],
        }
        summary = lightcurve.get_period_summary(
            n_peaks=2,
            prefer_fitted_psd=True,
        )
        self.assertTrue(summary["is_multicomponent"])
        self.assertIsNotNone(summary["freq_grid"])
        self.assertIsNotNone(summary["psd"])
        self.assertEqual(
            summary["method"],
            "consensus_multicomp_spectral_mixture_psd",
        )
        self.assertEqual(summary["component_periods"], [100.0, 40.0])
        self.assertEqual(
            summary.component_diagnostics.n_components,
            2,
        )

    def test_invalid_period_summary_axis_is_rejected(self):
        lightcurve = self._lightcurve()
        summary = self._spectral_summary()
        with self.assertRaisesRegex(ValueError, "x_axis"):
            lightcurve.plot_period_summary(
                summary=summary,
                show=False,
                x_axis="wavelength",
            )




    def test_explicit_period_summary_is_compact_point_interval_plot(self):
        lightcurve = self._lightcurve()
        peak = PeriodPeakResult(
            rank=1,
            frequency=0.01,
            period=100.0,
            interval_frequency=(1.0 / 120.0, 1.0 / 80.0),
            interval_period=(80.0, 120.0),
        )
        summary = PeriodSummaryResult(
            method="explicit_period_parameter",
            backend="explicit_period",
            dominant_period=100.0,
            dominant_frequency=0.01,
            peaks=[peak],
            freq_grid=None,
            psd=None,
            interval_definition=(
                "coherence_proxy_from_rbf_lengthscale"
            ),
        )

        figure, axis = lightcurve.plot_period_summary(
            summary=summary,
            show=False,
            x_axis="period",
            log_x=True,
        )

        width, height = figure.get_size_inches()
        self.assertAlmostEqual(width, 9.0, places=6)
        self.assertLess(height, 2.6)
        self.assertEqual(
            axis.get_title(),
            "Fitted GP temporal period",
        )
        self.assertNotIn(
            "explicit_period_parameter",
            axis.get_title(),
        )
        self.assertIsNone(axis.get_legend())
        self.assertEqual(len(axis.patches), 0)
        self.assertEqual(axis.get_yticks().size, 0)
        self.assertLess(
            axis.get_ylim()[1] - axis.get_ylim()[0],
            0.5,
        )
        rendered_text = " ".join(
            text.get_text() for text in axis.texts
        )
        self.assertIn("Period = 100", rendered_text)
        self.assertIn("coherence-proxy interval", rendered_text)
        self.assertNotIn("Dominant period:", rendered_text)

    def test_explicit_period_renderer_does_not_change_psd_renderer(self):
        lightcurve = self._lightcurve()
        summary = self._spectral_summary()
        figure, axis = lightcurve.plot_period_summary(
            summary=summary,
            show=False,
            x_axis="period",
            log_y=False,
            show_components=True,
        )
        self.assertEqual(axis.get_ylabel(), "PSD")
        self.assertIsNotNone(axis.get_legend())
        labels = [line.get_label() for line in axis.lines]
        self.assertIn("Summed PSD", labels)
        self.assertIn("SM component 1", labels)
        self.assertIn("SM component 2", labels)


class TestPeriodDiagnosticWrapperContract(unittest.TestCase):
    def test_wrapper_forwards_shared_summary_without_acf_peak_api(self):
        import inspect
        from unittest.mock import patch

        signature = inspect.signature(
            Lightcurve.plot_period_diagnostic_comparison
        )
        self.assertIn("period_summary", signature.parameters)
        self.assertIn("period_summary_kwargs", signature.parameters)
        self.assertNotIn(
            "acf_minimum_prominence",
            signature.parameters,
        )

        lightcurve = Lightcurve(
            torch.linspace(0.0, 10.0, 12, dtype=torch.float64),
            torch.ones(12, dtype=torch.float64),
            center_time=False,
            check_sampling=False,
            max_samples=None,
        )
        shared_summary = object()
        expected_result = {"status": "forwarded"}

        with patch(
            "pgmuvi.period_diagnostic_comparison."
            "plot_period_diagnostic_comparison",
            return_value=expected_result,
        ) as mocked:
            result = lightcurve.plot_period_diagnostic_comparison(
                period_summary=shared_summary,
                period_summary_kwargs={
                    "x_axis": "period",
                    "show_components": True,
                },
                show=False,
                strict=True,
            )

        self.assertIs(result, expected_result)
        mocked.assert_called_once()
        kwargs = mocked.call_args.kwargs
        self.assertIs(kwargs["period_summary"], shared_summary)
        self.assertEqual(
            kwargs["period_summary_kwargs"]["x_axis"],
            "period",
        )
        self.assertTrue(
            kwargs["period_summary_kwargs"]["show_components"]
        )
        self.assertNotIn("acf_minimum_prominence", kwargs)

if __name__ == "__main__":
    unittest.main()
