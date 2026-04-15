"""Tests for :meth:`Lightcurve.drop_bands`."""

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


class TestDropBandsSingle(unittest.TestCase):
    """Drop a single band."""

    def setUp(self):
        self.lc = _make_2d()

    def test_returns_lightcurve(self):
        result = self.lc.drop_bands(["V"])
        self.assertIsInstance(result, Lightcurve)

    def test_dropped_band_absent(self):
        result = self.lc.drop_bands(["V"])
        self.assertNotIn("V", result.band)

    def test_remaining_band_present(self):
        result = self.lc.drop_bands(["V"])
        self.assertTrue(np.all(result.band == "R"))

    def test_row_count_correct(self):
        result = self.lc.drop_bands(["V"])
        self.assertEqual(len(result.xdata), 5)

    def test_does_not_mutate_original(self):
        original_len = len(self.lc.xdata)
        self.lc.drop_bands(["V"])
        self.assertEqual(len(self.lc.xdata), original_len)

    def test_ydata_correct(self):
        result = self.lc.drop_bands(["V"])
        # The "R" band is the second half of the original data.
        expected_y = self.lc.ydata[5:]
        torch.testing.assert_close(result.ydata, expected_y)


class TestDropBandsMultiple(unittest.TestCase):
    """Drop multiple bands at once."""

    def setUp(self):
        self.lc = _make_3band()

    def test_drop_two_bands_correct_count(self):
        result = self.lc.drop_bands(["g", "r"])
        self.assertEqual(len(result.xdata), 4)

    def test_drop_two_bands_only_remaining_band_present(self):
        result = self.lc.drop_bands(["g", "r"])
        self.assertTrue(np.all(result.band == "i"))

    def test_drop_all_but_one_correct_band(self):
        result = self.lc.drop_bands(["r", "i"])
        np.testing.assert_array_equal(np.unique(result.band), ["g"])


class TestDropBandsEdgeCases(unittest.TestCase):
    """Edge cases: all removed, nonexistent bands."""

    def test_drop_all_bands_raises_value_error(self):
        lc = _make_2d()
        with self.assertRaises(ValueError) as ctx:
            lc.drop_bands(["V", "R"])
        self.assertIn("All rows were removed", str(ctx.exception))

    def test_drop_nonexistent_band_returns_copy(self):
        lc = _make_2d()
        result = lc.drop_bands(["Z"])
        self.assertEqual(len(result.xdata), len(lc.xdata))
        np.testing.assert_array_equal(result.band, lc.band)

    def test_drop_mix_existing_and_nonexistent(self):
        lc = _make_2d()
        # "Z" does not exist; only "V" rows should be removed.
        result = lc.drop_bands(["V", "Z"])
        self.assertTrue(np.all(result.band == "R"))
        self.assertEqual(len(result.xdata), 5)


class TestDropBandsPreservesAttributes(unittest.TestCase):
    """Verify that metadata and arrays are propagated correctly."""

    def setUp(self):
        self.lc = _make_2d()
        self.lc.name = "MyStar"

    def test_name_preserved(self):
        result = self.lc.drop_bands(["V"])
        self.assertEqual(result.name, "MyStar")

    def test_xtransform_preserved(self):
        result = self.lc.drop_bands(["V"])
        self.assertEqual(result.xtransform, self.lc.xtransform)

    def test_ytransform_preserved(self):
        result = self.lc.drop_bands(["V"])
        self.assertEqual(result.ytransform, self.lc.ytransform)

    def test_band_attribute_present(self):
        result = self.lc.drop_bands(["V"])
        self.assertIsNotNone(result.band)

    def test_band_length_matches_xdata(self):
        result = self.lc.drop_bands(["V"])
        self.assertEqual(len(result.band), len(result.xdata))

    def test_yerr_preserved(self):
        result = self.lc.drop_bands(["V"])
        self.assertIsNotNone(result.yerr)
        self.assertEqual(len(result.yerr), len(result.xdata))


class TestDropBandsInputValidation(unittest.TestCase):
    """Input validation — types and element checks."""

    def setUp(self):
        self.lc = _make_2d()

    def test_bare_string_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.lc.drop_bands("V")

    def test_bare_multichar_string_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.lc.drop_bands("VR")

    def test_tuple_input_accepted(self):
        result = self.lc.drop_bands(("V",))
        self.assertEqual(len(result.xdata), 5)

    def test_ndarray_string_input_accepted(self):
        result = self.lc.drop_bands(np.array(["V"]))
        self.assertEqual(len(result.xdata), 5)

    def test_float_element_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.lc.drop_bands([550.0])

    def test_int_element_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.lc.drop_bands([1])

    def test_np_float64_element_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.lc.drop_bands([np.float64(550.0)])

    def test_np_int64_element_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.lc.drop_bands([np.int64(1)])

    def test_bytes_element_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.lc.drop_bands([b"V"])

    def test_none_element_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.lc.drop_bands([None])

    def test_nested_list_element_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.lc.drop_bands([["V"]])

    def test_band_none_raises_value_error(self):
        self.lc.band = None
        with self.assertRaises(ValueError):
            self.lc.drop_bands(["V"])


if __name__ == "__main__":
    unittest.main()
