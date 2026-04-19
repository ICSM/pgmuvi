"""Tests for :meth:`Lightcurve.select_bands`.

These tests reflect the redesigned API where bands are identified exclusively
through the ``band`` attribute.  Wavelength-based selection is no longer
supported.
"""

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
    """Basic single-band selection by string label."""

    def setUp(self):
        self.lc = _make_2d()

    def test_select_single_band_returns_lightcurve(self):
        result = self.lc.select_bands(["V"])
        self.assertIsInstance(result, Lightcurve)

    def test_select_single_band_correct_length(self):
        result = self.lc.select_bands(["V"])
        self.assertEqual(len(result.xdata), 5)

    def test_select_single_band_uses_band_not_wavelength(self):
        """Selection is based on self.band, not on wavelength values."""
        # Both "V" and "R" rows have different wavelengths (550 vs 650).
        # Selecting by label "V" must return exactly the V-band rows.
        result = self.lc.select_bands(["V"])
        np.testing.assert_array_equal(result.band, np.array(["V"] * 5))

    def test_select_other_band(self):
        result = self.lc.select_bands(["R"])
        self.assertEqual(len(result.xdata), 5)
        np.testing.assert_array_equal(result.band, np.array(["R"] * 5))

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

    def test_xtransform_inherited(self):
        result = self.lc.select_bands(["V"])
        self.assertEqual(result.xtransform, self.lc.xtransform)

    def test_ytransform_inherited(self):
        result = self.lc.select_bands(["V"])
        self.assertEqual(result.ytransform, self.lc.ytransform)


class TestSelectBandsInputContainerTypes(unittest.TestCase):
    """Verify that tuple and numpy.ndarray string inputs work."""

    def setUp(self):
        self.lc = _make_2d()

    def test_tuple_input(self):
        """A tuple of string labels should work identically to a list."""
        result = self.lc.select_bands(("V",))
        self.assertEqual(len(result.xdata), 5)
        np.testing.assert_array_equal(result.band, np.array(["V"] * 5))

    def test_ndarray_string_input(self):
        """A numpy string array of labels should work."""
        result = self.lc.select_bands(np.array(["V"]))
        self.assertEqual(len(result.xdata), 5)
        np.testing.assert_array_equal(result.band, np.array(["V"] * 5))


class TestSelectBandsMultiple(unittest.TestCase):
    """Selecting multiple bands at once."""

    def setUp(self):
        self.lc = _make_3band()

    def test_select_two_bands(self):
        result = self.lc.select_bands(["g", "r"])
        self.assertEqual(len(result.xdata), 8)  # 2 × 4

    def test_select_two_bands_correct_labels(self):
        result = self.lc.select_bands(["g", "r"])
        np.testing.assert_array_equal(
            np.unique(result.band), np.array(["g", "r"])
        )

    def test_select_all_bands_returns_full_lc(self):
        result = self.lc.select_bands(["g", "r", "i"])
        self.assertEqual(len(result.xdata), len(self.lc.xdata))

    def test_band_label_subset_correct(self):
        result = self.lc.select_bands(["r", "i"])
        np.testing.assert_array_equal(
            np.unique(result.band), np.array(["i", "r"])
        )

    def test_select_is_or_based(self):
        """OR semantics: rows matching any requested label are included."""
        result_single_g = self.lc.select_bands(["g"])
        result_single_r = self.lc.select_bands(["r"])
        result_both = self.lc.select_bands(["g", "r"])
        self.assertEqual(
            len(result_both.xdata),
            len(result_single_g.xdata) + len(result_single_r.xdata),
        )


class TestSelectBandsErrors(unittest.TestCase):
    """Error conditions."""

    def test_raises_for_bare_string_input(self):
        """A bare string like bands='V' must raise TypeError."""
        lc = _make_2d()
        with self.assertRaises(TypeError):
            lc.select_bands("V")

    def test_raises_for_bare_multichar_string(self):
        """A multi-character bare string like bands='W1' must raise TypeError."""
        lc = _make_2d(label1="W1", label2="W2")
        with self.assertRaises(TypeError):
            lc.select_bands("W1")

    def test_raises_when_band_is_none(self):
        """select_bands requires self.band to be set."""
        lc = _make_2d()
        lc.band = None
        with self.assertRaises(ValueError):
            lc.select_bands(["V"])

    def test_raises_for_1d_lightcurve_no_band(self):
        """1-D lightcurve with band=None raises ValueError."""
        t = torch.linspace(0, 10, 20)
        y = torch.sin(t)
        lc = Lightcurve(t, y)
        with self.assertRaises(ValueError):
            lc.select_bands(["V"])

    def test_selects_full_1d_lightcurve_with_single_band_label(self):
        """A 1-D lightcurve with band=['V'] is returned in full for ['V']."""
        t = torch.linspace(0, 10, 20)
        y = torch.sin(t)
        yerr = torch.full_like(y, 0.1)
        lc = Lightcurve(t, y, yerr=yerr, band=["V"])

        selected = lc.select_bands(["V"])

        self.assertTrue(torch.equal(selected.xdata, lc.xdata))
        self.assertTrue(torch.equal(selected.ydata, lc.ydata))
        self.assertTrue(torch.equal(selected.yerr, lc.yerr))
        self.assertEqual(list(selected.band), ["V"])

    def test_raises_for_nonmatching_label_on_1d_single_band_lightcurve(self):
        """A non-matching label on a 1-D single-band lightcurve raises."""
        t = torch.linspace(0, 10, 20)
        y = torch.sin(t)
        lc = Lightcurve(t, y, band=["V"])
        with self.assertRaises(ValueError):
            lc.select_bands(["R"])

    def test_raises_for_nonexistent_label(self):
        """Requesting a label that is not in band raises ValueError."""
        lc = _make_2d()
        with self.assertRaises(ValueError):
            lc.select_bands(["Z"])

    def test_raises_for_float_selector(self):
        """Numeric float selectors are rejected with TypeError."""
        lc = _make_2d()
        with self.assertRaises(TypeError):
            lc.select_bands([650.0])

    def test_raises_for_int_selector(self):
        """Numeric int selectors are rejected with TypeError."""
        lc = _make_2d(wl1=1.0, wl2=2.0)
        with self.assertRaises(TypeError):
            lc.select_bands([1])

    def test_raises_for_np_float64_selector(self):
        """numpy.float64 selectors are rejected with TypeError."""
        lc = _make_2d()
        with self.assertRaises(TypeError):
            lc.select_bands([np.float64(650.0)])

    def test_raises_for_np_int64_selector(self):
        """numpy.int64 selectors are rejected with TypeError."""
        lc = _make_2d(wl1=1.0, wl2=2.0)
        with self.assertRaises(TypeError):
            lc.select_bands([np.int64(1)])

    def test_raises_for_nan_selector(self):
        """np.nan as a selector is rejected with TypeError (float)."""
        lc = _make_2d()
        with self.assertRaises(TypeError):
            lc.select_bands([np.nan])

    def test_raises_for_float_nan_selector(self):
        """float('nan') is also rejected with TypeError."""
        lc = _make_2d()
        with self.assertRaises(TypeError):
            lc.select_bands([float("nan")])

    def test_raises_for_nested_list_element(self):
        """A nested list element raises TypeError."""
        lc = _make_2d()
        with self.assertRaises(TypeError):
            lc.select_bands([["V"]])

    def test_raises_for_none_element(self):
        """None element raises TypeError."""
        lc = _make_2d()
        with self.assertRaises(TypeError):
            lc.select_bands([None])

    def test_raises_for_mixed_string_and_float(self):
        """Mixed string and float selectors are rejected."""
        lc = _make_3band()
        with self.assertRaises(TypeError):
            lc.select_bands(["g", 3.0])

    def test_raises_for_bytes_element(self):
        """bytes elements are rejected with TypeError."""
        lc = _make_2d()
        with self.assertRaises(TypeError):
            lc.select_bands([b"V"])

    def test_all_missing_labels_raises_value_error(self):
        """All selectors present but none matching raises ValueError."""
        lc = _make_2d()
        with self.assertRaises(ValueError):
            lc.select_bands(["Z", "X"])


class TestSelectBandsContainerTypes(unittest.TestCase):
    """Verify that unsupported container types are rejected."""

    def setUp(self):
        self.lc = _make_2d()

    def test_raises_for_integer_input(self):
        """An integer is not a valid container."""
        with self.assertRaises(TypeError):
            self.lc.select_bands(5)

    def test_raises_for_set_input(self):
        """A set is not an accepted container type."""
        with self.assertRaises(TypeError):
            self.lc.select_bands({"V"})

    def test_raises_for_dict_input(self):
        """A dict is not an accepted container type."""
        with self.assertRaises(TypeError):
            self.lc.select_bands({"V": 1})

    def test_raises_for_generator_input(self):
        """A generator expression is not an accepted container type."""
        with self.assertRaises(TypeError):
            self.lc.select_bands(x for x in ["V"])

    def test_list_input_accepted(self):
        """list is an accepted container type."""
        result = self.lc.select_bands(["V"])
        self.assertIsInstance(result, Lightcurve)

    def test_tuple_input_accepted(self):
        """tuple is an accepted container type."""
        result = self.lc.select_bands(("V",))
        self.assertIsInstance(result, Lightcurve)

    def test_ndarray_input_accepted(self):
        """numpy.ndarray is an accepted container type."""
        result = self.lc.select_bands(np.array(["V"]))
        self.assertIsInstance(result, Lightcurve)


class TestSelectBandsBandPreservation(unittest.TestCase):
    """Verify that the returned Lightcurve correctly preserves band."""

    def test_returned_band_is_subset_of_original(self):
        lc = _make_3band()
        result = lc.select_bands(["g", "i"])
        # Only "g" and "i" labels should appear
        self.assertTrue(all(b in ("g", "i") for b in result.band))

    def test_returned_band_matches_xdata_length(self):
        lc = _make_2d()
        result = lc.select_bands(["V"])
        self.assertEqual(len(result.band), len(result.xdata))

    def test_band_attribute_type_preserved(self):
        lc = _make_2d()
        result = lc.select_bands(["V"])
        self.assertIsInstance(result.band, np.ndarray)


if __name__ == "__main__":
    unittest.main()
