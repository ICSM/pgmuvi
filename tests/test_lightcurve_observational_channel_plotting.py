"""Regression tests for channel-aware 2-D light-curve plotting."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve


class _PredictiveDistribution:
    def __init__(self, x_values):
        self.mean = torch.zeros_like(x_values[:, 0])

    def confidence_region(self):
        return self.mean - 0.2, self.mean + 0.2


class _Model:
    def __call__(self, x_values):
        return x_values


class _Likelihood:
    def __call__(self, model_output):
        return _PredictiveDistribution(model_output)


class TestObservationalChannelPlotting(unittest.TestCase):
    def setUp(self):
        xdata = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [3.0, 1.0],
                [0.0, 2.0],
                [1.0, 2.0],
            ]
        )
        ydata = torch.tensor([1.0, 1.2, 0.9, 1.1, 2.0, 2.2])
        yerr = torch.full((6,), 0.1)
        channels = np.array(["A", "A", "B", "B", "C", "C"])
        self.lightcurve = Lightcurve(
            xdata,
            ydata,
            yerr=yerr,
            band=channels,
            max_samples=None,
        )

    def tearDown(self):
        plt.close("all")

    def test_data_only_plot_keeps_shared_wavelength_channels_distinct(self):
        figures = self.lightcurve._plot_data_only(show=False)

        first_labels = figures[0].axes[0].get_legend_handles_labels()[1]
        second_labels = figures[1].axes[0].get_legend_handles_labels()[1]
        self.assertEqual(first_labels, ["A", "B"])
        self.assertEqual(second_labels, ["C"])

    def test_data_only_plot_supports_missing_uncertainties(self):
        lightcurve = Lightcurve(
            self.lightcurve.xdata.detach().clone(),
            self.lightcurve.ydata.detach().clone(),
            band=np.asarray(
                self.lightcurve.observational_channel_labels,
                dtype=str,
            ),
            max_samples=None,
        )

        self.assertFalse(hasattr(lightcurve, "yerr"))

        figures = lightcurve._plot_data_only(show=False)

        first_labels = figures[0].axes[0].get_legend_handles_labels()[1]
        second_labels = figures[1].axes[0].get_legend_handles_labels()[1]
        self.assertEqual(first_labels, ["A", "B"])
        self.assertEqual(second_labels, ["C"])

    def test_fitted_plot_labels_channels_and_does_not_write_when_disabled(self):
        self.lightcurve.model = _Model()
        self.lightcurve.likelihood = _Likelihood()
        x_fine = torch.linspace(0.0, 3.0, 8)

        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(tmpdir)
                figures = self.lightcurve._plot_2d(
                    x_fine,
                    show=False,
                    save=False,
                )
                self.assertEqual(list(Path(tmpdir).iterdir()), [])
            finally:
                os.chdir(old_cwd)

        labels = figures[0].axes[0].get_legend_handles_labels()[1]
        self.assertIn("Predictive mean", labels)
        self.assertIn("95% predictive interval", labels)
        self.assertIn("A", labels)
        self.assertIn("B", labels)
        self.assertNotIn("Observed Data", labels)


if __name__ == "__main__":
    unittest.main()
