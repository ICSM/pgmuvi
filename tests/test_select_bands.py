"""Tests for :meth:`Lightcurve.select_bands`."""

import unittest

import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_2d(n_per_band=5, wl1=550.0, wl2=650.0, label1="V", label2="R"):
    """Two-band 2-D lightcurve with per-row band labels."""
    t = torch.linspace(0.0, 10.0, n_per_band)
    t2 = torch.cat([t, t])
    wl = torch.cat(
        [torch.full((n_per_band,), wl1), torch.full((n_per_band,), wl2)]
    )
    x = torch.stack([t2, wl], dim=1)
    y = torch.sin(t2)
    yerr = torch.full_like(y, 0.1)
    band = np.array([label1] * n_per_band + [label2] * n_per_band)
    return Lightcurve(x, y, yerr=yerr, band=band)


def _make_3band(n_per_band=4):
    """Three-band 2-D lightcurve with per-row band labels."""
    wls = [1.0, 2.0, 3.0]
    labels = ["g", "r", "i"]
    t = torch.linspace(0.0, 5.0, n_per_band)
    xs, ys, bands = [], [], []
    for wl, lbl in zip(wls, labels):
        xs.append(torch.stack([t, torch.full((n_per_band,), wl)], dim=1))
        ys.append(torch.zeros(n_per_band))
        bands.extend([lbl] * n_per_band)
    x = torch.cat(xs)
    y = torch.cat(ys)
    band = np.array(bands)
    return Lightcurve(x, y, band=band)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSelectBandsBasic(unittest.TestCase):
    """Basic single-band selection by string and float."""

    def setUp(self):
        self.lc = _make_2d()

    def test_select_by_string_returns_lightcurve(self):
        result = self.lc.select_bands(["V"])
        self.assertIsInstance(result, Lightcurve)

    def test_select_by_string_correct_rows(self):
        result = self.lc.select_bands(["V"])
        # All returned wavelength values must equal wl1 (550 nm).
        wl = result.xdata[:, 1]
        self.assertTrue((wl == 550.0).all())

    def test_select_by_string_correct_length(self):
        result = self.lc.select_bands(["V"])
        self.assertEqual(len(result.xdata), 5)

    def test_select_by_float_correct_rows(self):
        result = self.lc.select_bands([650.0])
        wl = result.xdata[:, 1]
        self.assertTrue((wl == 650.0).all())
        self.assertEqual(len(result.xdata), 5)

    def test_select_by_int_correct_rows(self):
        """Integer values should be promoted to float for matching."""
        lc = _make_2d(wl1=1.0, wl2=2.0)
        result = lc.select_bands([1])
        self.assertEqual(len(result.xdata), 5)
        self.assertTrue((result.xdata[:, 1] == 1.0).all())

    def test_band_attribute_preserved(self):
        result = self.lc.select_bands(["V"])
        self.assertIsNotNone(result.band)
        self.assertTrue(np.all(result.band == "V"))

    def test_band_length_matches_xdata(self):
        result = self.lc.select_bands(["R"])
        self.assertEqual(len(result.band), len(result.xdata))

    def test_yerr_preserved(self):
        result = self.lc.select_bands(["V"])
        self.assertIsNotNone(result.yerr)
        self.assertEqual(len(result.yerr), len(result.xdata))

    def test_ydata_correct(self):
        result = self.lc.select_bands(["R"])
        expected_y = self.lc.ydata[5:]  # second half of 10-row LC
        torch.testing.assert_close(result.ydata, expected_y)

    def test_name_inherited(self):
        self.lc.name = "MyStar"
        result = self.lc.select_bands(["V"])
        self.assertEqual(result.name, "MyStar")


class TestSelectBandsInputTypes(unittest.TestCase):
    """Verify that tuple, ndarray and np.integer/np.float64 inputs work."""

    def setUp(self):
        self.lc = _make_2d()

    def test_tuple_input(self):
        """A tuple of selectors should work identically to a list."""
        result = self.lc.select_bands(("V",))
        self.assertEqual(len(result.xdata), 5)
        self.assertTrue((result.xdata[:, 1] == 550.0).all())

    def test_ndarray_string_input(self):
        """A numpy string array of selectors should work."""
        result = self.lc.select_bands(np.array(["V"]))
        self.assertEqual(len(result.xdata), 5)

    def test_ndarray_float_input(self):
        """A numpy float array of selectors should work."""
        result = self.lc.select_bands(np.array([550.0]))
        self.assertEqual(len(result.xdata), 5)
        self.assertTrue((result.xdata[:, 1] == 550.0).all())

    def test_np_float64_element(self):
        """numpy.float64 scalar elements should be accepted."""
        result = self.lc.select_bands([np.float64(650.0)])
        self.assertEqual(len(result.xdata), 5)
        self.assertTrue((result.xdata[:, 1] == 650.0).all())

    def test_np_int_element(self):
        """numpy.int64 scalar elements should be accepted."""
        lc = _make_2d(wl1=1.0, wl2=2.0)
        result = lc.select_bands([np.int64(1)])
        self.assertEqual(len(result.xdata), 5)


class TestSelectBandsMultiple(unittest.TestCase):
    """Selecting multiple bands at once."""

    def setUp(self):
        self.lc = _make_3band()

    def test_select_two_strings(self):
        result = self.lc.select_bands(["g", "r"])
        self.assertEqual(len(result.xdata), 8)  # 2 × 4

    def test_select_two_floats(self):
        result = self.lc.select_bands([1.0, 3.0])
        unique_wl = result.xdata[:, 1].unique().sort().values.tolist()
        self.assertEqual(unique_wl, [1.0, 3.0])
        self.assertEqual(len(result.xdata), 8)

    def test_select_mixed_string_and_float(self):
        """Mix one string selector and one float selector."""
        result = self.lc.select_bands(["g", 3.0])
        unique_wl = result.xdata[:, 1].unique().sort().values.tolist()
        self.assertEqual(unique_wl, [1.0, 3.0])
        self.assertEqual(len(result.xdata), 8)

    def test_select_all_bands_returns_full_lc(self):
        result = self.lc.select_bands(["g", "r", "i"])
        self.assertEqual(len(result.xdata), len(self.lc.xdata))

    def test_band_label_subset_correct(self):
        result = self.lc.select_bands(["r", "i"])
        np.testing.assert_array_equal(
            np.unique(result.band), np.array(["i", "r"])
        )


class TestSelectBandsErrors(unittest.TestCase):
    """Error conditions."""

    def test_raises_for_1d_lightcurve(self):
        t = torch.linspace(0, 10, 20)
        y = torch.sin(t)
        lc = Lightcurve(t, y)
        with self.assertRaises(ValueError):
            lc.select_bands(["V"])

    def test_raises_for_string_when_band_none(self):
        lc = _make_2d()
        lc.band = None
        with self.assertRaises(ValueError):
            lc.select_bands(["V"])

    def test_raises_for_unsupported_type(self):
        lc = _make_2d()
        with self.assertRaises(TypeError):
            lc.select_bands([["V"]])  # nested list element

    def test_raises_for_bare_string_input(self):
        """A bare string like bands='V' must raise TypeError.

        Without this guard the string would be iterated as characters,
        silently giving wrong results.
        """
        lc = _make_2d()
        with self.assertRaises(TypeError):
            lc.select_bands("V")

    def test_raises_for_bare_multichar_string(self):
        """A multi-character bare string like bands='W1' must raise TypeError."""
        lc = _make_2d(label1="W1", label2="W2")
        with self.assertRaises(TypeError):
            lc.select_bands("W1")

    def test_raises_for_nan_selector(self):
        """np.nan as a float selector is not meaningful and must raise ValueError."""
        lc = _make_2d()
        with self.assertRaises(ValueError):
            lc.select_bands([np.nan])

    def test_raises_for_float_nan_selector(self):
        """float('nan') must also raise ValueError."""
        lc = _make_2d()
        with self.assertRaises(ValueError):
            lc.select_bands([float("nan")])

    def test_nonexistent_band_raises_value_error(self):
        """Selecting a label that is not present raises ValueError (no rows match)."""
        lc = _make_2d()
        # "Z" is not in the data — no rows match → ValueError from __init__.
        with self.assertRaises(ValueError):
            lc.select_bands(["Z"])

    def test_negative_wavelength_no_match_raises(self):
        """A negative wavelength not in the data yields no rows → ValueError."""
        lc = _make_2d()
        with self.assertRaises(ValueError):
            lc.select_bands([-1.0])

    def test_float_selector_no_band_required(self):
        """Float selection works even when self.band is None."""
        lc = _make_2d()
        lc.band = None
        result = lc.select_bands([550.0])
        self.assertEqual(len(result.xdata), 5)
        self.assertIsNone(result.band)


class TestSelectBandsNoBand(unittest.TestCase):
    """select_bands when the Lightcurve has no band attribute."""

    def test_float_select_no_band_attr(self):
        """Float-only selector works when band is None."""
        t = torch.linspace(0, 10, 10)
        wl = torch.cat([torch.ones(5), torch.full((5,), 2.0)])
        x = torch.stack([t, wl], dim=1)
        y = torch.zeros(10)
        lc = Lightcurve(x, y)
        result = lc.select_bands([1.0])
        self.assertEqual(len(result.xdata), 5)
        self.assertIsNone(result.band)


if __name__ == "__main__":
    unittest.main()
