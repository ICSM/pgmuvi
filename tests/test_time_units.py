import unittest

import torch

from pgmuvi.lightcurve import Lightcurve


class TestTimeUnits(unittest.TestCase):
    """Tests for the time_units parameter of Lightcurve."""

    def setUp(self):
        self.xdata_days = torch.as_tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        self.ydata = torch.as_tensor([1.0, 2.0, 1.0, 2.0], dtype=torch.float32)

    def test_none_units_no_conversion(self):
        """time_units=None leaves xdata unchanged (assumed days)."""
        lc = Lightcurve(self.xdata_days, self.ydata, time_units=None)
        self.assertTrue(torch.allclose(lc.xdata, self.xdata_days))

    def test_days_string_no_conversion(self):
        """time_units='d' leaves xdata unchanged."""
        lc = Lightcurve(self.xdata_days, self.ydata, time_units="d")
        self.assertTrue(torch.allclose(lc.xdata, self.xdata_days))

    def test_seconds_string_conversion(self):
        """time_units='s' converts seconds to days (divide by 86400)."""
        xdata_seconds = self.xdata_days * 86400.0
        lc = Lightcurve(xdata_seconds, self.ydata, time_units="s")
        self.assertTrue(torch.allclose(lc.xdata, self.xdata_days, atol=1e-4))

    def test_hours_string_conversion(self):
        """time_units='hr' converts hours to days (divide by 24)."""
        xdata_hours = self.xdata_days * 24.0
        lc = Lightcurve(xdata_hours, self.ydata, time_units="hr")
        self.assertTrue(torch.allclose(lc.xdata, self.xdata_days, atol=1e-4))

    def test_astropy_unit_conversion(self):
        """time_units=astropy.units.s converts seconds to days."""
        import astropy.units as u
        xdata_seconds = self.xdata_days * 86400.0
        lc = Lightcurve(xdata_seconds, self.ydata, time_units=u.s)
        self.assertTrue(torch.allclose(lc.xdata, self.xdata_days, atol=1e-4))

    def test_invalid_units_raises(self):
        """time_units with non-time units raises ValueError."""
        with self.assertRaises(ValueError):
            Lightcurve(self.xdata_days, self.ydata, time_units="m")

    def test_2d_only_time_column_converted(self):
        """For 2D xdata, only the time column (col 0) is converted."""
        import astropy.units as u
        # Build 2D xdata: time in seconds, wavelength in arbitrary units
        time_seconds = torch.as_tensor(
            [86400.0, 172800.0, 259200.0, 345600.0], dtype=torch.float32
        )
        wavelengths = torch.as_tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float32)
        xdata_2d = torch.stack([time_seconds, wavelengths], dim=1)
        lc = Lightcurve(xdata_2d, self.ydata, time_units=u.s)
        # Time column should now be in days
        self.assertTrue(
            torch.allclose(lc.xdata[:, 0], self.xdata_days, atol=1e-4)
        )
        # Wavelength column should be unchanged
        self.assertTrue(torch.allclose(lc.xdata[:, 1], wavelengths))

    def test_from_table_time_units(self):
        """from_table passes time_units through to the constructor."""
        from astropy.table import Table
        t = Table()
        t["time"] = (self.xdata_days * 86400.0).numpy()
        t["flux"] = self.ydata.numpy()
        t["flux_err"] = self.ydata.numpy()
        lc = Lightcurve.from_table(
            t, xcol="time", ycol="flux", yerrcol="flux_err", time_units="s"
        )
        self.assertTrue(torch.allclose(lc.xdata, self.xdata_days, atol=1e-4))

    def test_numpy_array_input_with_units(self):
        """time_units works when xdata is a numpy array (not just a tensor)."""
        xdata_seconds_np = (self.xdata_days * 86400.0).numpy()
        lc = Lightcurve(xdata_seconds_np, self.ydata, time_units="s")
        self.assertTrue(torch.allclose(lc.xdata, self.xdata_days, atol=1e-4))

    def test_list_input_with_units(self):
        """time_units works when xdata is a plain Python list."""
        xdata_seconds_list = (self.xdata_days * 86400.0).tolist()
        lc = Lightcurve(xdata_seconds_list, self.ydata, time_units="s")
        self.assertTrue(torch.allclose(lc.xdata, self.xdata_days, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
