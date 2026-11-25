"""Tests for Lightcurve CSV and table input/output."""
import os
import shutil
import tempfile
import unittest
import warnings

import numpy as np
import torch

from pgmuvi.lightcurve import (
    InputHelpers,
    Lightcurve,
)


class TestFromCSV(unittest.TestCase):
    """Tests for the InputHelpers.from_csv classmethod."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _write_csv(self, filename, content):
        path = os.path.join(self.tmpdir, filename)
        with open(path, "w") as f:
            f.write(content)
        return path

    # ------------------------------------------------------------------
    # Auto-detection tests
    # ------------------------------------------------------------------

    def test_autodetect_standard_names(self):
        """Auto-detection with canonical lowercase column names."""
        path = self._write_csv(
            "standard.csv",
            "x,y,yerr\n1.0,2.0,0.1\n2.0,3.0,0.2\n3.0,4.0,0.3\n",
        )
        lc = Lightcurve.from_csv(path)
        self.assertIsInstance(lc, Lightcurve)
        self.assertEqual(len(lc.xdata), 3)

    def test_autodetect_jd_magnitude(self):
        """Auto-detection with JD / Magnitude column names (sample data style)."""
        path = self._write_csv(
            "jd_mag.csv",
            "JD,Magnitude\n2450000.0,1.5\n2450001.0,1.6\n2450002.0,1.4\n",
        )
        lc = Lightcurve.from_csv(path)
        self.assertIsInstance(lc, Lightcurve)
        self.assertEqual(len(lc.xdata), 3)
        # yerr should be None when no uncertainty column is present
        self.assertFalse(hasattr(lc, "_yerr_raw"))

    def test_autodetect_case_insensitive(self):
        """Column name matching must be case-insensitive."""
        path = self._write_csv(
            "mixed_case.csv",
            "Time,Flux,Error\n0.0,1.0,0.05\n1.0,2.0,0.05\n2.0,1.5,0.05\n",
        )
        lc = Lightcurve.from_csv(path)
        self.assertIsInstance(lc, Lightcurve)
        self.assertEqual(len(lc.xdata), 3)
        self.assertTrue(hasattr(lc, "_yerr_raw"))

    def test_autodetect_no_yerr_column(self):
        """When no uncertainty column exists, yerr should not be set."""
        path = self._write_csv(
            "no_yerr.csv",
            "time,flux\n0.0,1.0\n1.0,2.0\n2.0,1.5\n",
        )
        lc = Lightcurve.from_csv(path)
        self.assertIsInstance(lc, Lightcurve)
        self.assertFalse(hasattr(lc, "_yerr_raw"))

    # ------------------------------------------------------------------
    # Explicit column name tests
    # ------------------------------------------------------------------

    def test_explicit_column_names(self):
        """Explicit xcol/ycol/yerrcol should override auto-detection."""
        path = self._write_csv(
            "explicit.csv",
            "date,signal,noise\n1.0,10.0,0.5\n2.0,11.0,0.5\n3.0,9.0,0.5\n",
        )
        lc = Lightcurve.from_csv(path, xcol="date", ycol="signal", yerrcol="noise")
        self.assertIsInstance(lc, Lightcurve)
        self.assertEqual(len(lc.xdata), 3)
        self.assertTrue(hasattr(lc, "_yerr_raw"))

    def test_explicit_xcol_missing_raises(self):
        """Explicitly specified xcol that does not exist should raise ValueError."""
        path = self._write_csv(
            "missing_x.csv",
            "time,flux\n0.0,1.0\n1.0,2.0\n",
        )
        with self.assertRaises(ValueError):
            Lightcurve.from_csv(path, xcol="nonexistent")

    def test_explicit_ycol_missing_raises(self):
        """Explicitly specified ycol that does not exist should raise ValueError."""
        path = self._write_csv(
            "missing_y.csv",
            "time,flux\n0.0,1.0\n1.0,2.0\n",
        )
        with self.assertRaises(ValueError):
            Lightcurve.from_csv(path, ycol="nonexistent")

    def test_explicit_yerrcol_missing_raises(self):
        """Explicitly specified yerrcol that does not exist should raise ValueError."""
        path = self._write_csv(
            "missing_yerr.csv",
            "time,flux\n0.0,1.0\n1.0,2.0\n",
        )
        with self.assertRaises(ValueError):
            Lightcurve.from_csv(path, yerrcol="nonexistent")

    def test_autodetect_x_fails_raises(self):
        """When x column cannot be auto-detected, a ValueError should be raised."""
        path = self._write_csv(
            "ambiguous.csv",
            "alpha,beta\n0.0,1.0\n1.0,2.0\n",
        )
        with self.assertRaises(ValueError):
            Lightcurve.from_csv(path)

    def test_autodetect_y_fails_raises(self):
        """When y column cannot be auto-detected, a ValueError should be raised."""
        path = self._write_csv(
            "no_y.csv",
            "time,alpha\n0.0,1.0\n1.0,2.0\n",
        )
        with self.assertRaises(ValueError):
            Lightcurve.from_csv(path)

    # ------------------------------------------------------------------
    # Data integrity tests
    # ------------------------------------------------------------------

    def test_data_values_preserved(self):
        """Values read from the CSV must match those in the Lightcurve."""
        path = self._write_csv(
            "values.csv",
            "x,y,yerr\n1.0,10.0,0.1\n2.0,20.0,0.2\n3.0,30.0,0.3\n",
        )
        lc = Lightcurve.from_csv(path)
        expected_x = torch.as_tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        expected_y = torch.as_tensor([10.0, 20.0, 30.0], dtype=torch.float32)
        expected_yerr = torch.as_tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        self.assertTrue(torch.allclose(lc._xdata_raw, expected_x))
        self.assertTrue(torch.allclose(lc._ydata_raw, expected_y))
        self.assertTrue(torch.allclose(lc._yerr_raw, expected_yerr))

    # ------------------------------------------------------------------
    # InputHelpers class attribute tests
    # ------------------------------------------------------------------

    def test_inputhelpers_is_base_of_lightcurve(self):
        """Lightcurve should be a subclass of InputHelpers."""
        self.assertTrue(issubclass(Lightcurve, InputHelpers))

    def test_column_name_lists_accessible(self):
        """Column name candidate lists should be accessible from Lightcurve."""
        self.assertIn("jd", Lightcurve._X_COLUMN_NAMES)
        self.assertIn("magnitude", Lightcurve._Y_COLUMN_NAMES)
        self.assertIn("err", Lightcurve._YERR_COLUMN_NAMES)
        self.assertIn("wavelength", Lightcurve._WAVELENGTH_COLUMN_NAMES)
        self.assertIn("wave", Lightcurve._WAVELENGTH_COLUMN_NAMES)
        # "band" and "filter" were moved to _WAVELENGTH_ID_COLUMN_NAMES
        self.assertIn("band", Lightcurve._WAVELENGTH_ID_COLUMN_NAMES)
        self.assertIn("filter", Lightcurve._WAVELENGTH_ID_COLUMN_NAMES)
        self.assertIn("filtername", Lightcurve._WAVELENGTH_ID_COLUMN_NAMES)
        self.assertIn("filter_name", Lightcurve._WAVELENGTH_ID_COLUMN_NAMES)

    # ------------------------------------------------------------------
    # 2D / multiband lightcurve tests
    # ------------------------------------------------------------------

    def test_autodetect_multiband_2d(self):
        """A wavelength column with multiple values triggers a 2D lightcurve."""
        path = self._write_csv(
            "multiband.csv",
            "time,flux,wavelength\n"
            "1.0,1.5,0.5\n2.0,1.6,0.5\n3.0,1.4,0.5\n"
            "1.0,2.5,1.5\n2.0,2.6,1.5\n3.0,2.4,1.5\n",
        )
        lc = Lightcurve.from_csv(path)
        self.assertIsInstance(lc, Lightcurve)
        # xdata should be 2D: (N, 2) with time in col 0, wavelength in col 1
        self.assertEqual(lc._xdata_raw.dim(), 2)
        self.assertEqual(lc._xdata_raw.shape[1], 2)
        self.assertEqual(len(lc.ydata), 6)

    def test_autodetect_single_wavelength_is_1d(self):
        """A wavelength column with only one unique value → 1D lightcurve."""
        path = self._write_csv(
            "single_band.csv",
            "time,flux,wavelength\n1.0,1.5,0.5\n2.0,1.6,0.5\n3.0,1.4,0.5\n",
        )
        lc = Lightcurve.from_csv(path)
        self.assertIsInstance(lc, Lightcurve)
        # Only one wavelength → treat as 1D
        self.assertEqual(lc._xdata_raw.dim(), 1)

    def test_explicit_wavelcol(self):
        """Explicit wavelcol parameter creates a 2D lightcurve."""
        path = self._write_csv(
            "explicit_band.csv",
            "time,flux,band\n1.0,1.5,1.0\n2.0,1.6,1.0\n3.0,1.4,2.0\n",
        )
        lc = Lightcurve.from_csv(path, xcol="time", ycol="flux", wavelcol="band")
        self.assertIsInstance(lc, Lightcurve)
        self.assertEqual(lc._xdata_raw.dim(), 2)
        self.assertEqual(lc._xdata_raw.shape[1], 2)

    def test_explicit_wavelcol_missing_raises(self):
        """Explicitly specified wavelcol that does not exist should raise ValueError."""
        path = self._write_csv(
            "missing_wavelcol.csv",
            "time,flux\n0.0,1.0\n1.0,2.0\n",
        )
        with self.assertRaises(ValueError):
            Lightcurve.from_csv(path, wavelcol="nonexistent")

    def test_xcol_as_list_creates_2d(self):
        """xcol as a list of column names stacks columns into 2D xdata."""
        path = self._write_csv(
            "xcol_list.csv",
            "time,band,flux\n1.0,0.5,1.5\n2.0,0.5,1.6\n3.0,1.5,1.4\n",
        )
        lc = Lightcurve.from_csv(path, xcol=["time", "band"], ycol="flux")
        self.assertIsInstance(lc, Lightcurve)
        self.assertEqual(lc._xdata_raw.dim(), 2)
        self.assertEqual(lc._xdata_raw.shape[1], 2)

    def test_xcol_as_single_element_list_is_1d(self):
        """xcol as a single-element list returns a 1D lightcurve."""
        path = self._write_csv(
            "xcol_single_list.csv",
            "time,flux\n1.0,1.5\n2.0,1.6\n3.0,1.4\n",
        )
        lc = Lightcurve.from_csv(path, xcol=["time"], ycol="flux")
        self.assertIsInstance(lc, Lightcurve)
        self.assertEqual(lc._xdata_raw.dim(), 1)

    def test_xcol_list_with_missing_col_raises(self):
        """A column listed in xcol that is absent raises ValueError."""
        path = self._write_csv(
            "bad_list.csv",
            "time,flux\n1.0,1.5\n2.0,1.6\n",
        )
        with self.assertRaises(ValueError):
            Lightcurve.from_csv(path, xcol=["time", "nonexistent"], ycol="flux")

    def test_2d_data_values_preserved(self):
        """Check that 2D xdata contains correct time and wavelength values."""
        path = self._write_csv(
            "2d_values.csv",
            "time,wavelength,flux\n"
            "1.0,0.5,10.0\n2.0,0.5,20.0\n"
            "1.0,1.5,30.0\n2.0,1.5,40.0\n",
        )
        lc = Lightcurve.from_csv(path)
        expected_time = torch.as_tensor([1.0, 2.0, 1.0, 2.0], dtype=torch.float32)
        expected_wave = torch.as_tensor([0.5, 0.5, 1.5, 1.5], dtype=torch.float32)
        self.assertTrue(torch.allclose(lc._xdata_raw[:, 0], expected_time))
        self.assertTrue(torch.allclose(lc._xdata_raw[:, 1], expected_wave))

    # ------------------------------------------------------------------
    # Sample data file
    # ------------------------------------------------------------------

    def test_sample_csv(self):
        """from_csv should work with the bundled AlfOriAAVSO_Vband.csv."""
        from pgmuvi.datasets import get_path

        sample = get_path("AlfOriAAVSO_Vband.csv")
        lc = Lightcurve.from_csv(sample)
        self.assertIsInstance(lc, Lightcurve)
        self.assertGreater(len(lc.xdata), 0)

    # ------------------------------------------------------------------
    # NaN dropping tests
    # ------------------------------------------------------------------

    def test_nan_in_y_dropped(self):
        """Rows with NaN in y should be dropped with a warning."""
        path = self._write_csv(
            "nan_y.csv",
            "x,y,yerr\n1.0,2.0,0.1\n2.0,nan,0.2\n3.0,4.0,0.3\n",
        )
        with self.assertWarns(UserWarning):
            lc = Lightcurve.from_csv(path)
        self.assertEqual(len(lc.xdata), 2)

    def test_nan_in_x_dropped(self):
        """Rows with NaN in x should be dropped with a warning."""
        path = self._write_csv(
            "nan_x.csv",
            "x,y,yerr\n1.0,2.0,0.1\nnan,3.0,0.2\n3.0,4.0,0.3\n",
        )
        with self.assertWarns(UserWarning):
            lc = Lightcurve.from_csv(path)
        self.assertEqual(len(lc.xdata), 2)

    def test_nan_in_yerr_dropped(self):
        """Rows with NaN in yerr should be dropped with a warning."""
        path = self._write_csv(
            "nan_yerr.csv",
            "x,y,yerr\n1.0,2.0,nan\n2.0,3.0,0.2\n3.0,4.0,0.3\n",
        )
        with self.assertWarns(UserWarning):
            lc = Lightcurve.from_csv(path)
        self.assertEqual(len(lc.xdata), 2)

    def test_no_nan_no_warning(self):
        """When there are no NaN values, no warning should be raised."""
        path = self._write_csv(
            "no_nan.csv",
            "x,y,yerr\n1.0,2.0,0.1\n2.0,3.0,0.2\n3.0,4.0,0.3\n",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lc = Lightcurve.from_csv(path)
        self.assertEqual(len(lc.xdata), 3)

    def test_nan_in_2d_x_dropped(self):
        """Rows with NaN in either column of a 2D x should be dropped."""
        path = self._write_csv(
            "nan_2d_x.csv",
            "time,wavelength,flux\n"
            "1.0,0.5,10.0\nnan,1.5,20.0\n3.0,0.5,30.0\n3.0,1.5,40.0\n",
        )
        with self.assertWarns(UserWarning):
            lc = Lightcurve.from_csv(path)
        # One row dropped → 3 rows remain
        self.assertEqual(len(lc.ydata), 3)

    def test_nan_in_wave_changes_dimensionality(self):
        """NaN in wavelength column: after dropping, only one wavelength remains → 1D."""
        path = self._write_csv(
            "nan_wave_single.csv",
            "time,wavelength,flux\n"
            "1.0,0.5,10.0\n2.0,nan,20.0\n3.0,0.5,30.0\n",
        )
        with self.assertWarns(UserWarning):
            lc = Lightcurve.from_csv(path)
        # After dropping the NaN row, only wavelength 0.5 remains → 1D
        self.assertEqual(lc._xdata_raw.dim(), 1)
        self.assertEqual(len(lc.xdata), 2)

    def test_all_nan_raises(self):
        """When all rows contain NaN, a ValueError should be raised."""
        path = self._write_csv(
            "all_nan.csv",
            "x,y,yerr\nnan,nan,nan\nnan,nan,nan\n",
        )
        with self.assertRaises(ValueError):
            Lightcurve.from_csv(path)


class TestToCSV(unittest.TestCase):
    """Tests for the Lightcurve.to_csv method."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_to_csv_roundtrip_with_optional_columns(self):
        """2-D lightcurve with yerr+band should round-trip through CSV."""
        time = torch.as_tensor([1.0, 2.0, 1.0, 2.0], dtype=torch.float32)
        wavelength = torch.as_tensor([550.0, 550.0, 650.0, 650.0], dtype=torch.float32)
        x = torch.stack([time, wavelength], dim=1)
        y = torch.as_tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32)
        yerr = torch.as_tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
        band = np.array(["V", "V", "R", "R"])
        lc = Lightcurve(x, y, yerr=yerr, band=band)

        path = os.path.join(self.tmpdir, "with_optional.csv")
        self.assertIsNone(lc.to_csv(path))

        with open(path) as fh:
            header = fh.readline().strip()
        self.assertEqual(header, "time,wavelength,flux,flux_error,band")

        lc2 = Lightcurve.from_csv(path)
        self.assertEqual(lc2.ndim, 2)
        self.assertTrue(torch.allclose(lc2.xdata, x))
        self.assertTrue(torch.allclose(lc2.ydata, y))
        self.assertTrue(torch.allclose(lc2.yerr, yerr))
        np.testing.assert_array_equal(lc2.band, band)

    def test_to_csv_roundtrip_without_optional_columns(self):
        """1-D lightcurve without yerr/band should omit optional columns."""
        x = torch.as_tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        y = torch.as_tensor([10.0, 20.0, 30.0], dtype=torch.float32)
        lc = Lightcurve(x, y)

        path = os.path.join(self.tmpdir, "minimal.csv")
        lc.to_csv(path)

        with open(path) as fh:
            header = fh.readline().strip()
        self.assertEqual(header, "time,wavelength,flux")

        lc2 = Lightcurve.from_csv(path)
        self.assertEqual(lc2.ndim, 1)
        self.assertTrue(torch.allclose(lc2.xdata, x))
        self.assertTrue(torch.allclose(lc2.ydata, y))
        self.assertFalse(hasattr(lc2, "_yerr_raw"))

    def test_to_csv_default_filename(self):
        """Default output name should be lightcurve.csv."""
        x = torch.as_tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        y = torch.as_tensor([4.0, 5.0, 6.0], dtype=torch.float32)
        lc = Lightcurve(x, y)

        old_cwd = os.getcwd()
        try:
            os.chdir(self.tmpdir)
            self.assertIsNone(lc.to_csv())
        finally:
            os.chdir(old_cwd)

        expected = os.path.join(self.tmpdir, "lightcurve.csv")
        self.assertTrue(os.path.exists(expected))

    def test_to_csv_raises_for_multidimensional_y(self):
        """CSV export requires 1-D ydata."""
        x = torch.as_tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        y = torch.as_tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        lc = Lightcurve(x, y)
        lc.register_buffer(
            "_ydata_raw",
            torch.as_tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
        )
        path = os.path.join(self.tmpdir, "bad_y_shape.csv")
        with self.assertRaises(ValueError):
            lc.to_csv(path)
        self.assertFalse(os.path.exists(path))

    def test_to_csv_raises_for_mismatched_y_length(self):
        """CSV export requires matching x/y lengths."""
        x = torch.as_tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        y = torch.as_tensor([4.0, 5.0, 6.0], dtype=torch.float32)
        lc = Lightcurve(x, y)
        lc.register_buffer("_ydata_raw", torch.as_tensor([4.0, 5.0], dtype=torch.float32))
        path = os.path.join(self.tmpdir, "bad_y_len.csv")
        with self.assertRaises(ValueError):
            lc.to_csv(path)
        self.assertFalse(os.path.exists(path))

    def test_to_csv_raises_for_mismatched_yerr_length(self):
        """CSV export requires matching x/yerr lengths."""
        x = torch.as_tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        y = torch.as_tensor([4.0, 5.0, 6.0], dtype=torch.float32)
        yerr = torch.as_tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        lc = Lightcurve(x, y, yerr=yerr)
        lc.register_buffer("_yerr_raw", torch.as_tensor([0.1, 0.2], dtype=torch.float32))
        path = os.path.join(self.tmpdir, "bad_yerr_len.csv")
        with self.assertRaises(ValueError):
            lc.to_csv(path)
        self.assertFalse(os.path.exists(path))


class TestFromTable(unittest.TestCase):
    """Tests for the Lightcurve.from_table classmethod NaN dropping behaviour."""

    def _make_table(self, x_vals, y_vals, yerr_vals=None):
        """Create an astropy Table with x, y, and optional yerr columns."""
        from astropy.table import Table

        cols = {"x": x_vals, "y": y_vals}
        if yerr_vals is not None:
            cols["yerr"] = yerr_vals
        return Table(cols)

    def test_nan_in_y_dropped(self):
        """Rows with NaN in y should be dropped with a warning."""
        tab = self._make_table(
            [1.0, 2.0, 3.0],
            [10.0, float("nan"), 30.0],
        )
        with self.assertWarns(UserWarning):
            lc = Lightcurve.from_table(tab, xcol="x", ycol="y")
        self.assertEqual(len(lc.xdata), 2)

    def test_nan_in_x_dropped(self):
        """Rows with NaN in x should be dropped with a warning."""
        tab = self._make_table(
            [1.0, float("nan"), 3.0],
            [10.0, 20.0, 30.0],
        )
        with self.assertWarns(UserWarning):
            lc = Lightcurve.from_table(tab, xcol="x", ycol="y")
        self.assertEqual(len(lc.xdata), 2)

    def test_nan_in_yerr_dropped(self):
        """Rows with NaN in yerr should be dropped with a warning."""
        tab = self._make_table(
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
            yerr_vals=[0.1, float("nan"), 0.3],
        )
        with self.assertWarns(UserWarning):
            lc = Lightcurve.from_table(tab, xcol="x", ycol="y", yerrcol="yerr")
        self.assertEqual(len(lc.xdata), 2)

    def test_no_nan_no_warning(self):
        """When there are no NaN values, no warning should be raised."""
        tab = self._make_table([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lc = Lightcurve.from_table(tab, xcol="x", ycol="y")
        self.assertEqual(len(lc.xdata), 3)

    def test_all_nan_raises(self):
        """When all rows contain NaN, a ValueError should be raised."""
        tab = self._make_table(
            [float("nan"), float("nan")],
            [float("nan"), float("nan")],
        )
        with self.assertRaises(ValueError):
            Lightcurve.from_table(tab, xcol="x", ycol="y")

    def test_valid_rows_preserved(self):
        """Values in non-NaN rows should be preserved correctly."""
        tab = self._make_table(
            [1.0, 2.0, 3.0],
            [10.0, float("nan"), 30.0],
            yerr_vals=[0.1, 0.2, 0.3],
        )
        with self.assertWarns(UserWarning):
            lc = Lightcurve.from_table(tab, xcol="x", ycol="y", yerrcol="yerr")
        expected_x = torch.as_tensor([1.0, 3.0], dtype=torch.float32)
        expected_y = torch.as_tensor([10.0, 30.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(lc._xdata_raw, expected_x))
        self.assertTrue(torch.allclose(lc._ydata_raw, expected_y))
