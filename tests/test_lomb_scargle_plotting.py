"""Tests for the maintained Lomb--Scargle plotting contract."""

from __future__ import annotations

import unittest
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.lomb_scargle_plotting import (
    plot_lomb_scargle_periodogram,
)


class TestLombScarglePlottingContract(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_defaults_are_period_log_x_and_linear_y(self):
        frequency = np.asarray([0.5, 0.25, 0.1])
        power = np.asarray([1.0, 3.0, 2.0])
        fig, ax = plot_lomb_scargle_periodogram(
            frequency,
            power,
            show=False,
        )
        self.assertIsNotNone(fig)
        self.assertEqual(ax.get_xscale(), "log")
        self.assertEqual(ax.get_yscale(), "linear")
        self.assertEqual(ax.get_xlabel(), "Period")
        np.testing.assert_allclose(
            ax.lines[0].get_xdata(),
            np.asarray([2.0, 4.0, 10.0]),
        )

    def test_frequency_axis_is_an_explicit_override(self):
        _, ax = plot_lomb_scargle_periodogram(
            [0.1, 0.2, 0.3],
            [1.0, 2.0, 1.5],
            x_axis="frequency",
            show=False,
        )
        self.assertEqual(ax.get_xscale(), "linear")
        self.assertEqual(ax.get_xlabel(), "Frequency")

    def test_all_candidates_and_reference_components_are_marked(self):
        candidates = [
            {"rank": 1, "period": 100.0, "frequency": 0.01},
            {"rank": 2, "period": 50.0, "frequency": 0.02},
            {"rank": 3, "period": 200.0, "frequency": 0.005},
        ]
        _, ax = plot_lomb_scargle_periodogram(
            np.linspace(0.003, 0.03, 40),
            np.linspace(0.2, 1.0, 40),
            candidates=candidates,
            reference_periods={
                "Injected component 1": 100.0,
                "Injected component 2": 50.0,
            },
            show=False,
        )
        labels = [line.get_label() for line in ax.lines]
        self.assertIn("LS #1", labels)
        self.assertIn("LS #2", labels)
        self.assertIn("LS #3", labels)
        self.assertIn("Injected component 1", labels)
        self.assertIn("Injected component 2", labels)

    def test_multiple_periodograms_can_share_one_axes(self):
        fig, ax = plot_lomb_scargle_periodogram(
            [0.01, 0.02, 0.03],
            [1.0, 2.0, 1.0],
            label="Default",
            show=False,
        )
        returned_fig, returned_ax = plot_lomb_scargle_periodogram(
            [0.01, 0.02, 0.03],
            [0.8, 2.2, 1.1],
            ax=ax,
            label="Alternative",
            linestyle="--",
            show=False,
        )
        self.assertIs(returned_fig, fig)
        self.assertIs(returned_ax, ax)
        self.assertEqual(ax.get_xscale(), "log")
        labels = [line.get_label() for line in ax.lines]
        self.assertIn("Default", labels)
        self.assertIn("Alternative", labels)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "same number"):
            plot_lomb_scargle_periodogram(
                [0.1, 0.2],
                [1.0],
                show=False,
            )


class TestLightcurveLombScarglePlotting(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    @staticmethod
    def _lightcurve():
        time = torch.linspace(0.0, 300.0, 80, dtype=torch.float64)
        flux = torch.sin(2.0 * torch.pi * time / 60.0)
        error = torch.full_like(time, 0.05)
        return Lightcurve(time, flux, yerr=error, max_samples=None)

    def test_freq_only_call_populates_plot_cache(self):
        lightcurve = self._lightcurve()
        lightcurve.fit_LS(freq_only=True)
        fig, ax = lightcurve.plot_lomb_scargle_periodogram(show=False)
        self.assertIsNotNone(fig)
        self.assertEqual(ax.get_xlabel(), "Period")
        self.assertEqual(ax.get_xscale(), "log")

    def test_combined_call_preserves_all_requested_candidates(self):
        lightcurve = self._lightcurve()
        peak_frequency, significant, _, _ = lightcurve.fit_LS(
            freq_only=False,
            num_peaks=3,
            return_full=True,
        )
        _, ax = lightcurve.plot_lomb_scargle_periodogram(show=False)
        labels = [line.get_label() for line in ax.lines]
        expected = min(3, len(peak_frequency))
        self.assertEqual(
            sum(label.startswith("LS #") for label in labels),
            expected,
        )
        self.assertEqual(len(significant), expected)

    def test_missing_cache_exits_gracefully(self):
        lightcurve = self._lightcurve()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig, ax = lightcurve.plot_lomb_scargle_periodogram(show=False)
        self.assertIsNone(fig)
        self.assertIsNone(ax)
        self.assertTrue(
            any("no cached full" in str(item.message) for item in caught)
        )


if __name__ == "__main__":
    unittest.main()
