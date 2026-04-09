"""Tests for the optional ``band`` attribute of :class:`Lightcurve`."""

import os
import tempfile
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


class TestBandFromCsvNumeric(unittest.TestCase):
    """from_csv with numeric wavelength column: band is None unless provided."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        csv_path = os.path.join(self._tmpdir, "lc_num.csv")
        with open(csv_path, "w") as f:
            f.write("time,band,flux,err\n")
            for i in range(5):
                f.write(f"{float(i)},1.0,{float(i)},0.1\n")
            for i in range(5):
                f.write(f"{float(i)},2.0,{float(i)},0.1\n")
        self._csv_path = csv_path

    def test_numeric_band_col_no_labels(self):
        """Numeric wavelength column → xdata is 2-D, band stays None."""
        lc = Lightcurve.from_csv(self._csv_path)
        self.assertIsNone(lc.band)
        self.assertEqual(lc.ndim, 2)

    def test_explicit_band_kwarg_still_works(self):
        """Callers can still pass band= explicitly with a numeric wavelength col."""
        lc = Lightcurve.from_csv(self._csv_path, band=["A", "B"])
        np.testing.assert_array_equal(lc.band, np.array(["A", "B"]))


class TestBandFromCsvStringLabels(unittest.TestCase):
    """from_csv auto-ingests string band labels and maps them to indices."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        csv_path = os.path.join(self._tmpdir, "lc_str.csv")
        with open(csv_path, "w") as f:
            f.write("time,band,flux,err\n")
            for i in range(5):
                f.write(f"{float(i)},V,{float(i)},0.1\n")
            for i in range(5):
                f.write(f"{float(i)},R,{float(i)},0.1\n")
        self._csv_path = csv_path

    def test_string_band_auto_ingested(self):
        """String band column is auto-detected and stored in lc.band."""
        lc = Lightcurve.from_csv(self._csv_path)
        self.assertIsNotNone(lc.band)
        np.testing.assert_array_equal(lc.band, np.array(["V", "R"]))

    def test_string_band_2d_xdata(self):
        """String band column produces 2-D xdata."""
        lc = Lightcurve.from_csv(self._csv_path)
        self.assertEqual(lc.ndim, 2)

    def test_string_band_indices_are_floats(self):
        """The wavelength axis (xdata[:,1]) contains 0-based float indices."""
        lc = Lightcurve.from_csv(self._csv_path)
        unique_wl = lc.xdata[:, 1].unique().sort().values
        np.testing.assert_array_almost_equal(
            unique_wl.numpy(), [0.0, 1.0]
        )

    def test_string_band_explicit_kwarg_overrides(self):
        """Caller-provided band= takes precedence over auto-detection."""
        lc = Lightcurve.from_csv(self._csv_path, band=["g", "r"])
        np.testing.assert_array_equal(lc.band, np.array(["g", "r"]))

    def test_single_string_band_is_1d(self):
        """A string band column with only one unique value → 1-D lightcurve."""
        csv_path = os.path.join(self._tmpdir, "lc_single.csv")
        with open(csv_path, "w") as f:
            f.write("time,band,flux\n")
            for i in range(5):
                f.write(f"{float(i)},V,{float(i)}\n")
        lc = Lightcurve.from_csv(csv_path)
        self.assertEqual(lc.ndim, 1)
        np.testing.assert_array_equal(lc.band, np.array(["V"]))


class TestBandFromTable(unittest.TestCase):
    """from_table ingests string band labels when xdata is 2-D (multiband)."""

    def _make_table_2d(self, band_values=None):
        """Astropy Table where 'x' is a 2-D (N x 2) column (time + wavelength)."""
        from astropy.table import Table

        band_values = band_values or ["V"] * 5 + ["R"] * 5
        # time column repeated for two bands
        t = list(range(5)) * 2
        # wavelength column: 1.0 for V, 2.0 for R
        wl = [1.0] * 5 + [2.0] * 5
        x2d = np.column_stack([t, wl])  # shape (10, 2)
        y = [float(i) for i in range(10)]
        return Table(
            {
                "x": x2d,
                "y": y,
                "yerr": [0.1] * 10,
                "band": band_values,
            }
        )

    def test_explicit_bandcol_2d(self):
        """Explicit bandcol with 2-D xcol → band labels stored."""
        tab = self._make_table_2d()
        lc = Lightcurve.from_table(tab, xcol="x", ycol="y", bandcol="band")
        np.testing.assert_array_equal(lc.band, np.array(["V", "R"]))

    def test_auto_detect_string_band_col_2d(self):
        """String column named 'band' is auto-detected when xcol is 2-D."""
        tab = self._make_table_2d()
        lc = Lightcurve.from_table(tab, xcol="x", ycol="y")
        np.testing.assert_array_equal(lc.band, np.array(["V", "R"]))

    def test_numeric_band_col_not_auto_ingested(self):
        """A numeric 'band' column alongside a 2-D xcol is NOT ingested."""
        from astropy.table import Table

        t = list(range(5)) * 2
        wl = [1.0] * 5 + [2.0] * 5
        x2d = np.column_stack([t, wl])
        tab = Table(
            {
                "x": x2d,
                "y": [float(i) for i in range(10)],
                "band": wl,  # numeric, not string
            }
        )
        lc = Lightcurve.from_table(tab, xcol="x", ycol="y")
        self.assertIsNone(lc.band)

    def test_1d_xcol_no_band_ingested(self):
        """With a 1-D xcol, band labels are NOT ingested even if column exists."""
        from astropy.table import Table

        tab = Table(
            {
                "x": list(range(10)),
                "y": [float(i) for i in range(10)],
                "band": ["V"] * 5 + ["R"] * 5,
            }
        )
        # 1-D xcol → 1-D lightcurve → band detection is skipped
        lc = Lightcurve.from_table(tab, xcol="x", ycol="y")
        self.assertIsNone(lc.band)

    def test_explicit_band_kwarg_takes_precedence(self):
        """Caller-provided band= takes precedence over auto-detection."""
        tab = self._make_table_2d()
        lc = Lightcurve.from_table(
            tab, xcol="x", ycol="y", bandcol="band", band=["g", "r"]
        )
        np.testing.assert_array_equal(lc.band, np.array(["g", "r"]))

    def test_no_band_col_is_none(self):
        """Table with no string band column → lc.band is None."""
        from astropy.table import Table

        tab = Table(
            {
                "x": list(range(10)),
                "y": [float(i) for i in range(10)],
            }
        )
        lc = Lightcurve.from_table(tab, xcol="x", ycol="y")
        self.assertIsNone(lc.band)


if __name__ == "__main__":
    unittest.main()
