"""Tests for LS/curve-only-ACF/shared-GP period comparison."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve
import pgmuvi.period_diagnostic_comparison as comparison_module
from pgmuvi.period_diagnostic_comparison import (
    build_period_feature_matches,
    plot_period_diagnostic_comparison,
    register_period_diagnostic_evidence,
)


def _peak(rank, period, interval):
    return SimpleNamespace(
        rank=rank,
        period=period,
        frequency=1.0 / period,
        interval_period=interval,
        area_fraction=1.0 / rank,
        prominence=0.5 / rank,
        coherence_proxy=5.0 / rank,
    )


class _FakeLightcurve:
    def __init__(self):
        self.xdata = torch.tensor(
            [
                [0.0, 0.5],
                [1.0, 0.5],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=torch.float64,
        )
        self.ydata = torch.tensor(
            [1.0, 2.0, 3.0, 4.0],
            dtype=torch.float64,
        )
        self.yerr = torch.full((4,), 0.1, dtype=torch.float64)
        self.observational_channel_labels = np.asarray(
            ["A", "A", "B", "B"],
            dtype=str,
        )
        self.ndim = 2
        self.model = object()
        self.likelihood = object()
        self.results = {"loss": [3.0, 2.0, 1.0]}
        self.period_summary_calls = 0
        self.period_summary_kwargs = []
        self._summary = SimpleNamespace(
            dominant_period=100.0,
            dominant_frequency=0.01,
            peaks=[
                _peak(1, 100.0, (90.0, 112.0)),
                _peak(2, 50.0, (45.0, 56.0)),
                _peak(3, 220.0, (190.0, 250.0)),
            ],
            component_diagnostics=SimpleNamespace(
                component_periods=np.asarray(
                    [101.0, 49.0, 205.0, 410.0]
                ),
                component_frequencies=np.asarray(
                    [1.0 / 101.0, 1.0 / 49.0, 1.0 / 205.0, 1.0 / 410.0]
                ),
                component_weights=np.asarray([0.4, 0.3, 0.2, 0.1]),
            ),
        )

    def get_period_summary(self):
        return self._summary

    def plot_period_summary(self, *, summary=None, show=False, **kwargs):
        self.period_summary_calls += 1
        self.period_summary_kwargs.append(dict(kwargs))
        fig, axes = plt.subplots(2, 1)
        return fig, axes


def _evidence():
    frequency_grid = np.linspace(0.002, 0.05, 120)
    power_grid = (
        np.exp(-0.5 * ((frequency_grid - 0.01) / 0.001) ** 2)
        + 0.7
        * np.exp(-0.5 * ((frequency_grid - 0.02) / 0.0015) ** 2)
    )
    lag = np.linspace(0.0, 300.0, 61)
    acf = (
        np.cos(2.0 * np.pi * lag / 100.0)
        + 0.4 * np.cos(2.0 * np.pi * lag / 50.0)
    )
    rows = []
    for channel, wavelength in (("A", 0.5), ("B", 1.0)):
        rows.append(
            {
                "observational_channel": channel,
                "physical_wavelengths": [wavelength],
                "status": "available",
                "lomb_scargle": {
                    "frequency_grid": frequency_grid,
                    "power_grid": power_grid,
                    "peak_periods": [100.0, 50.0, 200.0],
                    "peak_frequencies": [0.01, 0.02, 0.005],
                    "peak_significant": [True, True, False],
                    "peak_powers": [1.0, 0.7, 0.2],
                },
                "acf": {
                    "lag": lag,
                    "acf": acf,
                    "interpretation": (
                        "comparison_curve_only_no_peak_identification"
                    ),
                },
            }
        )
    return {
        "rows": rows,
        "all_channels_retained_for_consensus": True,
        "consensus_period": 100.0,
    }


class TestPublicLightcurveAPI(unittest.TestCase):
    def test_lightcurve_exposes_registration_and_plot_methods(self):
        self.assertTrue(
            hasattr(Lightcurve, "register_period_diagnostic_evidence")
        )
        self.assertTrue(
            hasattr(Lightcurve, "plot_period_diagnostic_comparison")
        )
        self.assertTrue(
            hasattr(Lightcurve, "plot_lomb_scargle_periodogram")
        )


class TestLsGpFeatureMatching(unittest.TestCase):
    def test_all_gp_peaks_seed_distinct_features(self):
        gp_peaks = [
            {"rank": 1, "period": 100.0, "frequency": 0.01},
            {"rank": 2, "period": 50.0, "frequency": 0.02},
            {"rank": 3, "period": 220.0, "frequency": 1.0 / 220.0},
        ]
        features = build_period_feature_matches(
            ls_candidates=[
                {"rank": 1, "period": 98.0, "frequency": 1.0 / 98.0},
                {"rank": 2, "period": 51.0, "frequency": 1.0 / 51.0},
            ],
            gp_psd_peaks=gp_peaks,
            tolerance=0.15,
            harmonic_orders=(0.5, 1.0, 2.0, 3.0),
        )
        self.assertGreaterEqual(len(features), len(gp_peaks))
        self.assertEqual(
            sum(len(feature["gp_psd_peaks"]) for feature in features),
            3,
        )
        self.assertEqual(
            sum(len(feature["ls_candidates"]) for feature in features),
            2,
        )
        self.assertTrue(
            all("acf_candidates" not in feature for feature in features)
        )

    def test_matching_does_not_use_candidate_rank(self):
        features = build_period_feature_matches(
            ls_candidates=[
                {"rank": 99, "period": 100.0, "frequency": 0.01},
            ],
            gp_psd_peaks=[
                {"rank": 7, "period": 101.0, "frequency": 1.0 / 101.0},
            ],
            tolerance=0.15,
            harmonic_orders=(1.0,),
        )
        self.assertEqual(len(features), 1)
        self.assertEqual(len(features[0]["ls_candidates"]), 1)


class TestComparisonPrerequisites(unittest.TestCase):
    def test_missing_diagnostics_returns_empty_and_warns(self):
        lightcurve = _FakeLightcurve()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = plot_period_diagnostic_comparison(
                lightcurve,
                show=False,
            )
        self.assertEqual(result, {})
        self.assertTrue(
            any("Lomb--Scargle" in str(item.message) for item in caught)
        )

    def test_missing_fit_returns_empty_and_warns(self):
        lightcurve = _FakeLightcurve()
        register_period_diagnostic_evidence(lightcurve, _evidence())
        lightcurve.results = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = plot_period_diagnostic_comparison(
                lightcurve,
                show=False,
            )
        self.assertEqual(result, {})
        self.assertTrue(
            any("completed GP fit" in str(item.message) for item in caught)
        )

    def test_stale_registered_evidence_is_rejected(self):
        lightcurve = _FakeLightcurve()
        register_period_diagnostic_evidence(lightcurve, _evidence())
        lightcurve.ydata = lightcurve.ydata + 1.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = plot_period_diagnostic_comparison(
                lightcurve,
                show=False,
            )
        self.assertEqual(result, {})
        self.assertTrue(any("stale" in str(item.message) for item in caught))


class TestComparisonPlots(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_every_valid_channel_is_plotted(self):
        lightcurve = _FakeLightcurve()
        register_period_diagnostic_evidence(lightcurve, _evidence())
        result = plot_period_diagnostic_comparison(
            lightcurve,
            show=False,
        )
        self.assertEqual(set(result), {"A", "B"})
        self.assertTrue(
            all(record["status"] == "plotted" for record in result.values())
        )
        self.assertEqual(lightcurve.period_summary_calls, 1)

    def test_acf_is_curve_only_and_has_no_candidates(self):
        lightcurve = _FakeLightcurve()
        register_period_diagnostic_evidence(lightcurve, _evidence())
        result = plot_period_diagnostic_comparison(
            lightcurve,
            show=False,
        )
        for record in result.values():
            self.assertNotIn("acf_candidates", record)
            self.assertEqual(
                record["acf_interpretation"],
                "comparison_curve_only_no_peak_identification",
            )
            labels = [
                line.get_label()
                for line in record["comparison_axes"][1].lines
            ]
            self.assertFalse(any(label.startswith("ACF #") for label in labels))

    def test_all_gp_peaks_components_and_ls_candidates_are_preserved(self):
        lightcurve = _FakeLightcurve()
        register_period_diagnostic_evidence(lightcurve, _evidence())
        result = plot_period_diagnostic_comparison(
            lightcurve,
            show=False,
        )
        for record in result.values():
            self.assertEqual(len(record["gp_psd_peaks"]), 3)
            self.assertEqual(len(record["gp_kernel_components"]), 4)
            self.assertEqual(len(record["ls_candidates"]), 3)
            self.assertEqual(
                record["gp_period_scope"],
                "shared_2d_temporal_kernel",
            )

    def test_period_summary_is_one_shared_period_axis_figure(self):
        lightcurve = _FakeLightcurve()
        register_period_diagnostic_evidence(lightcurve, _evidence())
        result = plot_period_diagnostic_comparison(
            lightcurve,
            show=False,
        )
        records = list(result.values())
        self.assertEqual(lightcurve.period_summary_calls, 1)
        self.assertEqual(
            lightcurve.period_summary_kwargs[0]["x_axis"],
            "period",
        )
        self.assertTrue(lightcurve.period_summary_kwargs[0]["log_x"])
        self.assertTrue(
            lightcurve.period_summary_kwargs[0]["show_components"]
        )
        self.assertIs(
            records[0]["period_summary_figure"],
            records[1]["period_summary_figure"],
        )
        self.assertTrue(all(record["period_summary_shared"] for record in records))


class TestCentralLombScarglePlotReuse(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_comparison_uses_shared_plotter_for_every_channel(self):
        lightcurve = _FakeLightcurve()
        register_period_diagnostic_evidence(lightcurve, _evidence())
        with mock.patch.object(
            comparison_module,
            "plot_lomb_scargle_periodogram",
            wraps=comparison_module.plot_lomb_scargle_periodogram,
        ) as plotter:
            result = comparison_module.plot_period_diagnostic_comparison(
                lightcurve,
                show=False,
            )
        self.assertEqual(set(result), {"A", "B"})
        self.assertEqual(plotter.call_count, 2)
        for call in plotter.call_args_list:
            self.assertEqual(call.kwargs["x_axis"], "period")
            self.assertEqual(call.kwargs["x_scale"], "log")
            self.assertEqual(call.kwargs["y_scale"], "linear")
            self.assertEqual(len(call.kwargs["candidates"]), 3)


if __name__ == "__main__":
    unittest.main()
