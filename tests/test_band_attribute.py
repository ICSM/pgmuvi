"""Tests for the optional ``band`` attribute of :class:`Lightcurve`."""

import unittest

import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve


def _make_1d():
    """Return minimal 1-D (xdata, ydata, yerr) tensors."""
    t = torch.linspace(0, 10, 20)
    y = torch.sin(t)
    yerr = torch.full_like(y, 0.1)
    return t, y, yerr


def _make_2d():
    """Return minimal 2-D (xdata, ydata, yerr) tensors with two bands."""
    t = torch.linspace(0, 10, 10)
    wl1 = torch.full((10,), 1.0)
    wl2 = torch.full((10,), 2.0)
    t2 = torch.cat([t, t])
    wl = torch.cat([wl1, wl2])
    x = torch.stack([t2, wl], dim=1)
    y = torch.sin(t2)
    yerr = torch.full_like(y, 0.1)
    return x, y, yerr


class TestBandAttributeNone(unittest.TestCase):
    """band defaults to None when not provided."""

    def test_1d_no_band(self):
        t, y, yerr = _make_1d()
        lc = Lightcurve(t, y, yerr=yerr)
        self.assertIsNone(lc.band)

    def test_2d_no_band(self):
        x, y, yerr = _make_2d()
        lc = Lightcurve(x, y, yerr=yerr)
        self.assertIsNone(lc.band)


class TestBandAttribute1D(unittest.TestCase):
    """band storage for 1-D light curves."""

    def test_single_band_label(self):
        t, y, yerr = _make_1d()
        lc = Lightcurve(t, y, yerr=yerr, band=["V"])
        self.assertIsNotNone(lc.band)
        np.testing.assert_array_equal(lc.band, np.array(["V"]))

    def test_wrong_length_raises(self):
        t, y, yerr = _make_1d()
        with self.assertRaises(ValueError):
            Lightcurve(t, y, yerr=yerr, band=["V", "R"])


class TestBandAttribute2D(unittest.TestCase):
    """band storage for 2-D (multiband) light curves."""

    def test_two_band_labels(self):
        x, y, yerr = _make_2d()
        lc = Lightcurve(x, y, yerr=yerr, band=["V", "R"])
        np.testing.assert_array_equal(lc.band, np.array(["V", "R"]))

    def test_band_stored_as_numpy_strings(self):
        x, y, yerr = _make_2d()
        lc = Lightcurve(x, y, yerr=yerr, band=["W1", "W2"])
        self.assertIsInstance(lc.band, np.ndarray)
        self.assertTrue(
            np.issubdtype(lc.band.dtype, np.str_),
            msg=f"Expected str dtype, got {lc.band.dtype}",
        )

    def test_wrong_length_raises(self):
        x, y, yerr = _make_2d()
        with self.assertRaises(ValueError):
            Lightcurve(x, y, yerr=yerr, band=["V"])

    def test_wrong_length_too_many_raises(self):
        x, y, yerr = _make_2d()
        with self.assertRaises(ValueError):
            Lightcurve(x, y, yerr=yerr, band=["V", "R", "I"])

    def test_band_none_explicit(self):
        x, y, yerr = _make_2d()
        lc = Lightcurve(x, y, yerr=yerr, band=None)
        self.assertIsNone(lc.band)

    def test_non_2d_band_array_raises(self):
        x, y, yerr = _make_2d()
        with self.assertRaises(ValueError):
            Lightcurve(x, y, yerr=yerr, band=[["V", "R"]])


class TestBandFromCsv(unittest.TestCase):
    """from_csv passes band through to the constructor."""

    def setUp(self):
        import tempfile
        import os

        self._tmpdir = tempfile.mkdtemp()
        csv_path = os.path.join(self._tmpdir, "lc.csv")
        with open(csv_path, "w") as f:
            f.write("time,band,flux,err\n")
            for i in range(5):
                f.write(f"{float(i)},1.0,{float(i)},0.1\n")
            for i in range(5):
                f.write(f"{float(i)},2.0,{float(i)},0.1\n")
        self._csv_path = csv_path

    def test_from_csv_passes_band(self):
        lc = Lightcurve.from_csv(self._csv_path, band=["A", "B"])
        np.testing.assert_array_equal(lc.band, np.array(["A", "B"]))

    def test_from_csv_no_band_is_none(self):
        lc = Lightcurve.from_csv(self._csv_path)
        self.assertIsNone(lc.band)


if __name__ == "__main__":
    unittest.main()
