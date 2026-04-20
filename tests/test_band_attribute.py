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


# Per-row band label arrays for _make_2d() (20 rows: 10 band-1, 10 band-2)
_BAND_2D_VR = np.array(["V"] * 10 + ["R"] * 10)
_BAND_2D_W1W2 = np.array(["W1"] * 10 + ["W2"] * 10)


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
    """band storage for 2-D (multiband) light curves.

    For a 20-row lightcurve (10 rows per band), band must have 20 elements.
    """

    def test_two_band_labels(self):
        x, y, yerr = _make_2d()
        lc = Lightcurve(x, y, yerr=yerr, band=_BAND_2D_VR)
        np.testing.assert_array_equal(lc.band, _BAND_2D_VR)

    def test_band_stored_as_numpy_strings(self):
        x, y, yerr = _make_2d()
        lc = Lightcurve(x, y, yerr=yerr, band=_BAND_2D_W1W2)
        self.assertIsInstance(lc.band, np.ndarray)
        self.assertTrue(
            np.issubdtype(lc.band.dtype, np.str_),
            msg=f"Expected str dtype, got {lc.band.dtype}",
        )

    def test_wrong_length_raises(self):
        """Fewer labels than rows → ValueError."""
        x, y, yerr = _make_2d()
        with self.assertRaises(ValueError):
            Lightcurve(x, y, yerr=yerr, band=["V"])

    def test_wrong_length_too_many_raises(self):
        """More labels than rows → ValueError."""
        x, y, yerr = _make_2d()
        # 21 labels for a 20-row lightcurve
        with self.assertRaises(ValueError):
            Lightcurve(x, y, yerr=yerr, band=["V"] * 11 + ["R"] * 10)

    def test_band_none_explicit(self):
        x, y, yerr = _make_2d()
        lc = Lightcurve(x, y, yerr=yerr, band=None)
        self.assertIsNone(lc.band)

    def test_non_2d_band_array_raises(self):
        x, y, yerr = _make_2d()
        with self.assertRaises(ValueError):
            Lightcurve(x, y, yerr=yerr, band=[["V", "R"]])

    def test_nonfinite_rows_are_dropped_from_band(self):
        x, y, yerr = _make_2d()
        x[3, 0] = torch.nan
        y[12] = torch.inf
        lc = Lightcurve(x, y, yerr=yerr, band=_BAND_2D_VR)
        expected = np.delete(_BAND_2D_VR, [3, 12])
        np.testing.assert_array_equal(lc.band, expected)


class TestBandFromCsvNumericWavelength(unittest.TestCase):
    """from_csv with a numeric 'wavelength' column: 2-D xdata, band=None."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        csv_path = os.path.join(self._tmpdir, "lc_num_wl.csv")
        with open(csv_path, "w") as f:
            f.write("time,wavelength,flux,err\n")
            for i in range(5):
                f.write(f"{float(i)},550.0,{float(i)},0.1\n")
            for i in range(5):
                f.write(f"{float(i)},650.0,{float(i)},0.1\n")
        self._csv_path = csv_path

    def test_numeric_wavelength_col_2d(self):
        """Numeric 'wavelength' column → 2-D xdata, band stays None."""
        lc = Lightcurve.from_csv(self._csv_path)
        self.assertIsNone(lc.band)
        self.assertEqual(lc.ndim, 2)

    def test_explicit_band_kwarg_still_works(self):
        """Callers can pass per-row band= explicitly alongside a numeric wavelength col."""
        per_row = ["A"] * 5 + ["B"] * 5
        lc = Lightcurve.from_csv(self._csv_path, band=per_row)
        np.testing.assert_array_equal(lc.band, np.array(per_row))


class TestBandFromCsvNumericBandColNotWavelength(unittest.TestCase):
    """A column named 'band' with numeric values is NOT a wavelength column.

    'band' is in _WAVELENGTH_ID_COLUMN_NAMES, not _WAVELENGTH_COLUMN_NAMES,
    so it is never used for xdata[:, 1] by auto-detection.
    """

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        csv_path = os.path.join(self._tmpdir, "lc_num_band.csv")
        with open(csv_path, "w") as f:
            f.write("time,band,flux,err\n")
            for i in range(5):
                f.write(f"{float(i)},1.0,{float(i)},0.1\n")
            for i in range(5):
                f.write(f"{float(i)},2.0,{float(i)},0.1\n")
        self._csv_path = csv_path

    def test_numeric_band_col_gives_1d(self):
        """Numeric 'band' col is not in _WAVELENGTH_COLUMN_NAMES → 1-D lightcurve."""
        lc = Lightcurve.from_csv(self._csv_path)
        self.assertIsNone(lc.band)
        self.assertEqual(lc.ndim, 1)


class TestBandFromCsvStringBandNoWavelength(unittest.TestCase):
    """String band-ID column without a numeric wavelength → 1-D lightcurve.

    The float wavelength column (from _WAVELENGTH_COLUMN_NAMES) is required
    to create a 2-D lightcurve.  A string band-ID column alone produces a
    1-D lightcurve with a single stored band label from the file.
    """

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        csv_path = os.path.join(self._tmpdir, "lc_str.csv")
        with open(csv_path, "w") as f:
            f.write("time,band,flux,err\n")
            for i in range(10):
                f.write(f"{float(i)},V,{float(i)},0.1\n")
        self._csv_path = csv_path

    def test_string_band_no_wavelength_is_1d(self):
        """String 'band' column without numeric wavelength → 1-D lightcurve."""
        lc = Lightcurve.from_csv(self._csv_path)
        self.assertEqual(lc.ndim, 1)

    def test_string_band_auto_populates_for_1d(self):
        """band is auto-populated for 1-D lightcurves when a band-ID column exists."""
        lc = Lightcurve.from_csv(self._csv_path)
        np.testing.assert_array_equal(lc.band, np.array(["V"]))

    def test_single_string_band_still_1d(self):
        """A string band column with one unique value → 1-D, band populated."""
        csv_path = os.path.join(self._tmpdir, "lc_single.csv")
        with open(csv_path, "w") as f:
            f.write("time,band,flux\n")
            for i in range(5):
                f.write(f"{float(i)},V,{float(i)}\n")
        lc = Lightcurve.from_csv(csv_path)
        self.assertEqual(lc.ndim, 1)
        np.testing.assert_array_equal(lc.band, np.array(["V"]))

    def test_mixed_string_bands_warn_and_leave_band_unset(self):
        """Mixed 1-D band labels warn and do not auto-populate band."""
        csv_path = os.path.join(self._tmpdir, "lc_mixed.csv")
        with open(csv_path, "w") as f:
            f.write("time,band,flux\n")
            for i in range(5):
                f.write(f"{float(i)},V,{float(i)}\n")
            for i in range(5, 10):
                f.write(f"{float(i)},R,{float(i)}\n")
        with self.assertWarnsRegex(
            UserWarning, "multiple distinct non-empty labels"
        ):
            lc = Lightcurve.from_csv(csv_path)
        self.assertEqual(lc.ndim, 1)
        self.assertIsNone(lc.band)


class TestBandFromCsvBothColumnsIndependent(unittest.TestCase):
    """Numeric wavelength AND string band-ID columns are populated independently.

    When a CSV has both a numeric wavelength column (for xdata[:,1]) and a
    string band-ID column (for lc.band), both are used independently.
    """

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        csv_path = os.path.join(self._tmpdir, "lc_both.csv")
        with open(csv_path, "w") as f:
            f.write("time,wavelength,band,flux\n")
            for i in range(5):
                f.write(f"{float(i)},550.0,V,{float(i)}\n")
            for i in range(5):
                f.write(f"{float(i)},650.0,R,{float(i)}\n")
        self._csv_path = csv_path

    def test_both_2d_xdata(self):
        """Numeric 'wavelength' column → 2-D xdata."""
        lc = Lightcurve.from_csv(self._csv_path)
        self.assertEqual(lc.ndim, 2)

    def test_both_band_populated(self):
        """String 'band' column → lc.band populated with per-row labels."""
        lc = Lightcurve.from_csv(self._csv_path)
        np.testing.assert_array_equal(
            lc.band, np.array(["V"] * 5 + ["R"] * 5)
        )

    def test_numeric_wavelength_in_xdata(self):
        """xdata[:,1] contains the numeric wavelength values, not indices."""
        lc = Lightcurve.from_csv(self._csv_path)
        unique_wl = lc.xdata[:, 1].unique().sort().values
        np.testing.assert_array_almost_equal(unique_wl.numpy(), [550.0, 650.0])

    def test_explicit_band_kwarg_overrides_auto(self):
        """Caller-provided band= takes precedence over auto-detection."""
        per_row = ["g"] * 5 + ["r"] * 5
        lc = Lightcurve.from_csv(self._csv_path, band=per_row)
        np.testing.assert_array_equal(lc.band, np.array(per_row))


class TestWavelengthIdColumnNames(unittest.TestCase):
    """_WAVELENGTH_ID_COLUMN_NAMES is used for string band-ID auto-detection."""

    def _write_csv(self, path, band_col_name, bands):
        """Write a CSV with a numeric 'wavelength' column AND a string band-ID col."""
        unique = list(dict.fromkeys(bands))
        with open(path, "w") as f:
            f.write(f"time,wavelength,{band_col_name},flux\n")
            for i, b in enumerate(bands):
                wl = float(unique.index(b) + 1)
                f.write(f"{float(i)},{wl},{b},{float(i)}\n")

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def test_filter_col_auto_detected(self):
        """Column named 'filter' is auto-detected via _WAVELENGTH_ID_COLUMN_NAMES."""
        path = os.path.join(self._tmpdir, "filter.csv")
        bands = ["V", "V", "R", "R", "R"]
        self._write_csv(path, "filter", bands)
        lc = Lightcurve.from_csv(path)
        np.testing.assert_array_equal(lc.band, np.array(bands))

    def test_filtername_col_auto_detected(self):
        """Column named 'filtername' is auto-detected via _WAVELENGTH_ID_COLUMN_NAMES."""
        path = os.path.join(self._tmpdir, "filtername.csv")
        bands = ["g", "g", "r", "r"]
        self._write_csv(path, "filtername", bands)
        lc = Lightcurve.from_csv(path)
        np.testing.assert_array_equal(lc.band, np.array(bands))

    def test_filter_name_col_auto_detected(self):
        """Column named 'filter_name' is auto-detected via _WAVELENGTH_ID_COLUMN_NAMES."""
        path = os.path.join(self._tmpdir, "filter_name.csv")
        bands = ["W1", "W1", "W2", "W2"]
        self._write_csv(path, "filter_name", bands)
        lc = Lightcurve.from_csv(path)
        np.testing.assert_array_equal(lc.band, np.array(bands))

    def test_numeric_wavelength_and_string_band_id_are_independent(self):
        """Numeric wavelength (xdata[:,1]) and string band-ID (lc.band) are
        populated independently when both columns are present."""
        path = os.path.join(self._tmpdir, "both.csv")
        bands = ["V", "V", "R", "R"]
        with open(path, "w") as f:
            f.write("time,wavelength,band,flux\n")
            for i, b in enumerate(bands):
                wl = 550.0 if b == "V" else 650.0
                f.write(f"{float(i)},{wl},{b},{float(i)}\n")
        lc = Lightcurve.from_csv(path)
        # Numeric wavelength column → 2-D xdata
        self.assertEqual(lc.ndim, 2)
        # String band-ID column → lc.band populated per-row
        np.testing.assert_array_equal(lc.band, np.array(bands))

    def test_string_band_id_col_not_used_for_xdata(self):
        """String band-ID column alone (no numeric wavelength) remains 1-D."""
        path = os.path.join(self._tmpdir, "id_only.csv")
        with open(path, "w") as f:
            f.write("time,band,flux\n")
            for i in range(4):
                b = "V" if i < 2 else "R"
                f.write(f"{float(i)},{b},{float(i)}\n")
        with self.assertWarnsRegex(
            UserWarning, "multiple distinct non-empty labels"
        ):
            lc = Lightcurve.from_csv(path)
        # No numeric wavelength column → 1-D
        self.assertEqual(lc.ndim, 1)
        # Mixed labels for 1-D input leave band unset.
        self.assertIsNone(lc.band)

    def test_band_id_names_not_in_wavelength_column_names(self):
        """'band' and 'filter' are NOT in _WAVELENGTH_COLUMN_NAMES."""
        self.assertNotIn("band", Lightcurve._WAVELENGTH_COLUMN_NAMES)
        self.assertNotIn("filter", Lightcurve._WAVELENGTH_COLUMN_NAMES)

    def test_band_id_names_content(self):
        """_WAVELENGTH_ID_COLUMN_NAMES contains the required entries."""
        for name in ("band", "filter", "filtername", "filter_name"):
            self.assertIn(name, Lightcurve._WAVELENGTH_ID_COLUMN_NAMES)


class TestWavelengthIdColumnNamesFromTable(unittest.TestCase):
    """from_table auto-detects string band IDs via _WAVELENGTH_ID_COLUMN_NAMES."""

    def _make_table_2d_with_filter(self, col_name, band_values):
        from astropy.table import Table

        t = list(range(len(band_values)))
        wl = [1.0 if v == band_values[0] else 2.0 for v in band_values]
        x2d = np.column_stack([t, wl])
        return Table(
            {
                "x": x2d,
                "y": [float(i) for i in range(len(band_values))],
                col_name: band_values,
            }
        )

    def test_filter_col_auto_detected(self):
        """Column named 'filter' is auto-detected."""
        band_values = ["V", "V", "R", "R"]
        tab = self._make_table_2d_with_filter("filter", band_values)
        lc = Lightcurve.from_table(tab, xcol="x", ycol="y")
        np.testing.assert_array_equal(lc.band, np.array(band_values))

    def test_filtername_col_auto_detected(self):
        """Column named 'filtername' is auto-detected."""
        band_values = ["g", "g", "r", "r"]
        tab = self._make_table_2d_with_filter("filtername", band_values)
        lc = Lightcurve.from_table(tab, xcol="x", ycol="y")
        np.testing.assert_array_equal(lc.band, np.array(band_values))

    def test_filter_name_col_auto_detected(self):
        """Column named 'filter_name' is auto-detected."""
        band_values = ["W1", "W1", "W2", "W2"]
        tab = self._make_table_2d_with_filter("filter_name", band_values)
        lc = Lightcurve.from_table(tab, xcol="x", ycol="y")
        np.testing.assert_array_equal(lc.band, np.array(band_values))


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
        """Explicit bandcol with 2-D xcol → per-row band labels stored."""
        tab = self._make_table_2d()
        lc = Lightcurve.from_table(tab, xcol="x", ycol="y", bandcol="band")
        np.testing.assert_array_equal(lc.band, np.array(["V"] * 5 + ["R"] * 5))

    def test_auto_detect_string_band_col_2d(self):
        """String column named 'band' is auto-detected when xcol is 2-D."""
        tab = self._make_table_2d()
        lc = Lightcurve.from_table(tab, xcol="x", ycol="y")
        np.testing.assert_array_equal(lc.band, np.array(["V"] * 5 + ["R"] * 5))

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
        per_row = ["g"] * 5 + ["r"] * 5
        lc = Lightcurve.from_table(
            tab, xcol="x", ycol="y", bandcol="band", band=per_row
        )
        np.testing.assert_array_equal(lc.band, np.array(per_row))

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
