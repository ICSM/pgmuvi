"""Tests for Lightcurve.plot and its y-scale helpers."""
import matplotlib

matplotlib.use("Agg")
import unittest

import numpy as np
import torch

from pgmuvi.lightcurve import (
    Lightcurve,
)


class TestYscaleAndYlim(unittest.TestCase):
    """Unit tests for Lightcurve._yscale_and_ylim static helper."""

    def _call(self, y_vals, yscale="auto", ylim=None):
        return Lightcurve._yscale_and_ylim(np.array(y_vals), yscale, ylim)

    # --- scale selection ---

    def test_auto_all_positive_large_range_selects_log(self):
        # max/min = 1000 > 100
        scale, _ = self._call([1.0, 10.0, 1000.0])
        self.assertEqual(scale, "log")

    def test_auto_all_positive_small_range_selects_linear(self):
        # max/min = 10 < 100
        scale, _ = self._call([1.0, 5.0, 10.0])
        self.assertEqual(scale, "linear")

    def test_auto_mixed_signs_selects_linear(self):
        scale, _ = self._call([-1.0, 1.0, 100.0])
        self.assertEqual(scale, "linear")

    def test_auto_zero_in_data_selects_linear(self):
        scale, _ = self._call([0.0, 1.0, 1000.0])
        self.assertEqual(scale, "linear")

    def test_explicit_log_respected(self):
        scale, _ = self._call([1.0, 2.0], yscale="log")
        self.assertEqual(scale, "log")

    def test_explicit_linear_respected(self):
        scale, _ = self._call([1.0, 1000.0], yscale="linear")
        self.assertEqual(scale, "linear")

    # --- limit computation (auto ylim) ---

    def test_linear_auto_lim_adds_padding(self):
        # data [0, 10]: range=10, padding=1 → [-1, 11]
        _, lim = self._call([0.0, 10.0], yscale="linear")
        self.assertIsNotNone(lim)
        self.assertAlmostEqual(lim[0], -1.0)
        self.assertAlmostEqual(lim[1], 11.0)

    def test_identical_values_nonzero_uses_magnitude_padding(self):
        # all values are 5: padding = 0.1 * 5 = 0.5
        _, lim = self._call([5.0, 5.0, 5.0], yscale="linear")
        self.assertIsNotNone(lim)
        self.assertAlmostEqual(lim[0], 4.5)
        self.assertAlmostEqual(lim[1], 5.5)

    def test_identical_values_zero_fallback_padding(self):
        # all values are 0: base=1, padding=0.1
        _, lim = self._call([0.0, 0.0, 0.0], yscale="linear")
        self.assertIsNotNone(lim)
        self.assertAlmostEqual(lim[0], -0.1)
        self.assertAlmostEqual(lim[1], 0.1)

    def test_log_auto_lim_is_positive(self):
        scale, lim = self._call([1.0, 10.0, 1000.0])
        self.assertEqual(scale, "log")
        self.assertIsNotNone(lim)
        self.assertGreater(lim[0], 0.0)
        self.assertGreater(lim[1], lim[0])

    def test_log_forced_nonpositive_data_returns_none_lim(self):
        # Forced log but data has zero: cannot compute valid limits
        scale, lim = self._call([0.0, 1.0, 1000.0], yscale="log")
        self.assertEqual(scale, "log")
        self.assertIsNone(lim)

    def test_log_forced_negative_data_returns_none_lim(self):
        scale, lim = self._call([-5.0, 1.0, 1000.0], yscale="log")
        self.assertEqual(scale, "log")
        self.assertIsNone(lim)

    # --- explicit ylim ---

    def test_explicit_ylim_linear_used_as_is(self):
        _, lim = self._call([1.0, 5.0], yscale="linear", ylim=[0.0, 10.0])
        self.assertEqual(lim, [0.0, 10.0])

    def test_explicit_ylim_log_positive_lower_used_as_is(self):
        _, lim = self._call([1.0, 1000.0], yscale="log", ylim=[0.5, 2000.0])
        self.assertEqual(lim, [0.5, 2000.0])

    def test_explicit_ylim_log_nonpositive_lower_returns_none(self):
        # Lower bound of 0 is invalid on a log axis
        _, lim = self._call([1.0, 1000.0], yscale="log", ylim=[0.0, 2000.0])
        self.assertIsNone(lim)

    def test_explicit_ylim_log_negative_lower_returns_none(self):
        _, lim = self._call([1.0, 1000.0], yscale="log", ylim=[-1.0, 2000.0])
        self.assertIsNone(lim)

    def test_explicit_ylim_auto_log_nonpositive_lower_returns_none(self):
        # auto selects log (large range, all positive), but caller passes
        # ylim with a non-positive lower bound
        _, lim = self._call([1.0, 1000.0], yscale="auto", ylim=[-1.0, 2000.0])
        self.assertIsNone(lim)


class TestPlotYscaleValidation(unittest.TestCase):
    """Test that plot() raises ValueError for invalid yscale values."""

    def setUp(self):
        xdata = torch.as_tensor([1.0, 2.0, 3.0, 4.0])
        ydata = torch.as_tensor([1.0, 2.0, 1.0, 2.0])
        self.lc = Lightcurve(xdata, ydata)

    def test_invalid_yscale_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.lc.plot(yscale="symlog")

    def test_invalid_yscale_message_is_informative(self):
        with self.assertRaisesRegex(ValueError, "yscale must be one of"):
            self.lc.plot(yscale="bad_value")


class TestPlotWithoutFit(unittest.TestCase):
    """Test that plot() works without a prior GP fit."""

    def setUp(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        xdata = torch.as_tensor([1.0, 2.0, 3.0, 4.0])
        ydata = torch.as_tensor([1.0, 2.0, 1.0, 2.0])
        self.lc = Lightcurve(xdata, ydata)

    def tearDown(self):
        self.plt.close("all")

    def test_plot_returns_figure_without_fit(self):
        fig = self.lc.plot(show=False)
        self.assertIsNotNone(fig)



class TestMultibandPlotWithoutFit(unittest.TestCase):
    """Test that plot() handles multiband (2-D) data correctly before fitting."""

    def setUp(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        # Two wavelengths (0.5 and 1.5), two observations each
        self.xdata_2d = torch.as_tensor(
            [[1.0, 0.5], [2.0, 0.5], [3.0, 1.5], [4.0, 1.5]], dtype=torch.float32
        )
        self.ydata_2d = torch.as_tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        self.lc_2d = Lightcurve(self.xdata_2d, self.ydata_2d)

    def tearDown(self):
        self.plt.close("all")

    def test_multiband_plot_returns_list_of_figures(self):
        """plot() should return a list of figures for multiband data."""
        result = self.lc_2d.plot(show=False)
        self.assertIsInstance(result, list)

    def test_multiband_plot_one_figure_per_wavelength(self):
        """There should be one figure per unique wavelength."""
        figs = self.lc_2d.plot(show=False)
        self.assertEqual(len(figs), 2)

    def test_multiband_plot_figures_are_matplotlib_figures(self):
        """Each element in the returned list should be a Figure."""
        import matplotlib.figure
        figs = self.lc_2d.plot(show=False)
        for fig in figs:
            self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_multiband_plot_with_yerr(self):
        """Multiband plot should work when yerr is provided."""
        yerr_2d = torch.as_tensor([0.1, 0.1, 0.1, 0.1], dtype=torch.float32)
        lc_2d_err = Lightcurve(self.xdata_2d, self.ydata_2d, yerr=yerr_2d)
        figs = lc_2d_err.plot(show=False)
        self.assertIsInstance(figs, list)
        self.assertEqual(len(figs), 2)


class TestPlotNPred(unittest.TestCase):
    """Test the n_pred parameter of Lightcurve.plot()."""

    def setUp(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        xdata = torch.as_tensor([1.0, 2.0, 3.0, 4.0])
        ydata = torch.as_tensor([1.0, 2.0, 1.0, 2.0])
        self.lc_1d = Lightcurve(xdata, ydata)
        xdata_2d = torch.as_tensor(
            [[1.0, 0.5], [2.0, 0.5], [3.0, 1.5], [4.0, 1.5]], dtype=torch.float32
        )
        ydata_2d = torch.as_tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        self.lc_2d = Lightcurve(xdata_2d, ydata_2d)

    def tearDown(self):
        self.plt.close("all")

    def test_1d_plot_accepts_n_pred(self):
        """plot() with n_pred returns a figure for 1D data."""
        fig = self.lc_1d.plot(show=False, n_pred=50)
        self.assertIsNotNone(fig)

    def test_2d_plot_accepts_n_pred(self):
        """plot() with n_pred returns a list of figures for 2D data."""
        figs = self.lc_2d.plot(show=False, n_pred=50)
        self.assertIsInstance(figs, list)

    def test_invalid_n_pred_not_int_raises(self):
        """plot() raises ValueError when n_pred is not an integer."""
        with self.assertRaises(ValueError):
            self.lc_1d.plot(show=False, n_pred=100.5)

    def test_invalid_n_pred_bool_raises(self):
        """plot() raises ValueError when n_pred is a bool."""
        with self.assertRaises(ValueError):
            self.lc_1d.plot(show=False, n_pred=True)

    def test_invalid_n_pred_zero_raises(self):
        """plot() raises ValueError when n_pred is 0."""
        with self.assertRaises(ValueError):
            self.lc_1d.plot(show=False, n_pred=0)

    def test_invalid_n_pred_negative_raises(self):
        """plot() raises ValueError when n_pred is negative."""
        with self.assertRaises(ValueError):
            self.lc_1d.plot(show=False, n_pred=-10)

    def test_invalid_n_pred_one_raises(self):
        """plot() raises ValueError when n_pred is 1 (< 2)."""
        with self.assertRaises(ValueError):
            self.lc_1d.plot(show=False, n_pred=1)

    def test_n_pred_numpy_int_accepted(self):
        """plot() accepts a numpy integer for n_pred."""
        fig_small = self.lc_1d.plot(show=False, n_pred=np.int32(10))
        fig_large = self.lc_1d.plot(show=False, n_pred=np.int32(50))
        self.assertIsNotNone(fig_small)
        self.assertIsNotNone(fig_large)
