"""Tests for Lightcurve.merge() and Lightcurve.concat()."""

import os
import tempfile
import unittest
import warnings

import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_2d_lc(wavelengths, n_per_band=10, band_labels=None, seed=0):
    """Return a 2-D Lightcurve with one band per wavelength.

    Parameters
    ----------
    wavelengths : list of float
        Wavelength values; one band per entry.
    n_per_band : int
        Number of rows per band.
    band_labels : list of str or None
        Band label strings.  Auto-generated from wavelength if None.
    seed : int
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    if band_labels is None:
        band_labels = [f"B{i}" for i in range(len(wavelengths))]

    xs, ys, yerrs, bands = [], [], [], []
    for wl, bl in zip(wavelengths, band_labels):
        t = np.sort(rng.uniform(0, 10, n_per_band))
        y = rng.normal(0, 1, n_per_band)
        ye = rng.uniform(0.05, 0.15, n_per_band)
        wl_col = np.full(n_per_band, float(wl))
        xs.append(np.column_stack([t, wl_col]))
        ys.append(y)
        yerrs.append(ye)
        bands.extend([bl] * n_per_band)

    x = torch.tensor(np.vstack(xs), dtype=torch.float32)
    y = torch.tensor(np.concatenate(ys), dtype=torch.float32)
    ye = torch.tensor(np.concatenate(yerrs), dtype=torch.float32)
    band = np.array(bands, dtype=str)
    return Lightcurve(x, y, yerr=ye, band=band)


def _make_1d_lc(n=15, seed=0):
    """Return a plain 1-D Lightcurve with no band attribute."""
    rng = np.random.default_rng(seed)
    t = torch.tensor(
        np.sort(rng.uniform(0, 10, n)), dtype=torch.float32
    )
    y = torch.tensor(rng.normal(0, 1, n), dtype=torch.float32)
    ye = torch.tensor(rng.uniform(0.05, 0.15, n), dtype=torch.float32)
    return Lightcurve(t, y, yerr=ye)


def _make_csv(wavelengths, n_per_band=8, band_labels=None, seed=1):
    """Write a CSV file and return the path.

    Always includes at least two distinct wavelengths so that
    ``Lightcurve.from_csv`` auto-detects a 2-D lightcurve.  If only one
    wavelength is requested, a second dummy wavelength is appended (and
    later bands beyond those listed are dropped by the callers that only
    care about specific bands).
    """
    rng = np.random.default_rng(seed)
    if band_labels is None:
        band_labels = [f"B{i}" for i in range(len(wavelengths))]

    # Ensure at least two distinct wavelengths for 2-D auto-detection.
    # from_csv only creates a 2-D lightcurve when more than one unique
    # wavelength value is present; if the caller only needs one real band,
    # we add a sentinel band at a well-separated wavelength.
    _DUMMY_WL_OFFSET = 1000.0  # large enough to never collide with real bands
    all_wls = list(wavelengths)
    all_bls = list(band_labels)
    if len(all_wls) < 2:
        _dummy_wl = max(all_wls) + _DUMMY_WL_OFFSET
        _dummy_bl = "_dummy_"
        all_wls.append(_dummy_wl)
        all_bls.append(_dummy_bl)

    rows = []
    for wl, bl in zip(all_wls, all_bls):
        for _ in range(n_per_band):
            t = rng.uniform(0, 10)
            flux = rng.normal(1, 0.1)
            err = rng.uniform(0.01, 0.05)
            rows.append(f"{t:.6f},{flux:.6f},{err:.6f},{wl:.1f},{bl}")

    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w") as fh:
        fh.write("time,flux,flux_error,wavelength,band\n")
        fh.write("\n".join(rows))
    return path


# ===========================================================================
# Tests for merge()
# ===========================================================================


class TestMergeSuccess(unittest.TestCase):
    """1. Successful merge of disjoint bands."""

    def test_merge_disjoint_bands(self):
        lc_self = _make_2d_lc([1.0, 2.0], band_labels=["V", "R"])
        lc_other = _make_2d_lc([3.0], band_labels=["I"])
        result = lc_self.merge(lc_other)

        self.assertIsInstance(result, Lightcurve)
        self.assertEqual(result.ndim, 2)
        unique_bands = set(np.unique(result.band))
        self.assertEqual(unique_bands, {"V", "R", "I"})

    def test_merge_preserves_row_order(self):
        """Self rows come first, then other rows."""
        lc_self = _make_2d_lc([1.0], n_per_band=5, band_labels=["V"])
        lc_other = _make_2d_lc([2.0], n_per_band=7, band_labels=["R"])
        result = lc_self.merge(lc_other)

        n_self = len(lc_self._xdata_raw)
        np.testing.assert_array_equal(
            result.band[:n_self], np.full(n_self, "V")
        )
        np.testing.assert_array_equal(
            result.band[n_self:], np.full(7, "R")
        )

    def test_merge_result_is_new_object(self):
        lc_self = _make_2d_lc([1.0], band_labels=["V"])
        lc_other = _make_2d_lc([2.0], band_labels=["R"])
        result = lc_self.merge(lc_other)
        self.assertIsNot(result, lc_self)
        self.assertIsNot(result, lc_other)

    def test_merge_result_is_unfitted(self):
        lc_self = _make_2d_lc([1.0], band_labels=["V"])
        lc_other = _make_2d_lc([2.0], band_labels=["R"])
        result = lc_self.merge(lc_other)
        # A freshly constructed Lightcurve should have no model
        self.assertFalse(
            getattr(result, "_Lightcurve__SET_MODEL_CALLED", False)
        )


class TestMergeDuplicateBandRaises(unittest.TestCase):
    """2. Duplicate band → raises ValueError."""

    def test_duplicate_band_raises(self):
        lc_self = _make_2d_lc([1.0], band_labels=["V"])
        lc_other = _make_2d_lc([2.0], band_labels=["V"])
        with self.assertRaises(ValueError):
            lc_self.merge(lc_other)


class TestMergeDuplicateWavelengthRaises(unittest.TestCase):
    """3. Duplicate wavelength → raises ValueError."""

    def test_duplicate_wavelength_raises(self):
        lc_self = _make_2d_lc([1.0], band_labels=["V"])
        lc_other = _make_2d_lc([1.0], band_labels=["R"])
        with self.assertRaises(ValueError):
            lc_self.merge(lc_other)


class TestMergeOnConflictSkip(unittest.TestCase):
    """4. on_conflict='skip' skips the entire conflicting band."""

    def test_skip_duplicate_band(self):
        lc_self = _make_2d_lc([1.0, 2.0], band_labels=["V", "R"])
        # "V" duplicates, "I" is new
        lc_other = _make_2d_lc([3.0, 4.0], band_labels=["V", "I"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = lc_self.merge(lc_other, on_conflict="skip")

        # "V" should have been skipped, "I" should be present
        unique_bands = set(np.unique(result.band))
        self.assertIn("I", unique_bands)
        # The duplicate "V" from other was skipped; only self's "V" rows remain
        v_count = int((result.band == "V").sum())
        self.assertEqual(v_count, 10)  # only self's 10 rows
        # A warning should have been emitted
        self.assertTrue(
            any("skip" in str(warning.message).lower()
                or "V" in str(warning.message)
                for warning in w)
        )

    def test_skip_leaves_non_conflicting_bands(self):
        lc_self = _make_2d_lc([1.0], band_labels=["V"])
        lc_other = _make_2d_lc([1.0, 2.0], band_labels=["V", "R"])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = lc_self.merge(lc_other, on_conflict="skip")
        unique_bands = set(np.unique(result.band))
        self.assertIn("R", unique_bands)

    def test_skip_never_partially_includes_band(self):
        """NEVER partially include rows from a skipped band."""
        lc_self = _make_2d_lc([1.0], n_per_band=5, band_labels=["V"])
        lc_other = _make_2d_lc([1.0], n_per_band=8, band_labels=["V"])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = lc_self.merge(lc_other, on_conflict="skip")
        # Only self rows should remain
        self.assertEqual(len(result._xdata_raw), 5)


class TestMerge1DWithoutWavelengthRaises(unittest.TestCase):
    """5. 1-D merge without wavelength → raises."""

    def test_1d_merge_no_wavelength_raises(self):
        lc_self = _make_2d_lc([1.0], band_labels=["V"])
        lc_other = _make_1d_lc()
        with self.assertRaises(ValueError):
            lc_self.merge(lc_other, band="R")


class TestMerge1DWithWavelength(unittest.TestCase):
    """6. 1-D merge with wavelength works."""

    def test_1d_merge_with_wavelength(self):
        lc_self = _make_2d_lc([1.0], band_labels=["V"])
        lc_other = _make_1d_lc()
        result = lc_self.merge(lc_other, band="R", wavelength=2.0)

        self.assertIsInstance(result, Lightcurve)
        self.assertEqual(result.ndim, 2)
        unique_bands = set(np.unique(result.band))
        self.assertIn("V", unique_bands)
        self.assertIn("R", unique_bands)

    def test_1d_merge_adds_wavelength_column(self):
        lc_self = _make_2d_lc([1.0], band_labels=["V"])
        lc_other = _make_1d_lc(n=12)
        result = lc_self.merge(lc_other, band="R", wavelength=5.0)

        r_mask = result.band == "R"
        wl_vals = result._xdata_raw[
            torch.as_tensor(r_mask, dtype=torch.bool), 1
        ]
        self.assertTrue((wl_vals == 5.0).all())


class TestMergeInvalidOther(unittest.TestCase):
    """Extra: merge rejects lists and non-Lightcurve inputs."""

    def test_list_raises_type_error(self):
        lc_self = _make_2d_lc([1.0], band_labels=["V"])
        lc_other = _make_2d_lc([2.0], band_labels=["R"])
        with self.assertRaises(TypeError):
            lc_self.merge([lc_other])

    def test_wrong_type_raises_type_error(self):
        lc_self = _make_2d_lc([1.0], band_labels=["V"])
        with self.assertRaises(TypeError):
            lc_self.merge(42)

    def test_self_must_be_2d(self):
        lc_1d = _make_1d_lc()
        lc_other = _make_2d_lc([1.0], band_labels=["V"])
        with self.assertRaises(ValueError):
            lc_1d.merge(lc_other)

    def test_2d_other_with_wavelength_raises(self):
        lc_self = _make_2d_lc([1.0], band_labels=["V"])
        lc_other = _make_2d_lc([2.0], band_labels=["R"])
        with self.assertRaises(ValueError):
            lc_self.merge(lc_other, wavelength=2.0)


class TestMergeFromCsv(unittest.TestCase):
    """merge() accepts a CSV path for other."""

    def setUp(self):
        self.csv_path = _make_csv([3.0], band_labels=["I"])

    def tearDown(self):
        os.unlink(self.csv_path)

    def test_merge_csv_path(self):
        lc_self = _make_2d_lc([1.0, 2.0], band_labels=["V", "R"])
        result = lc_self.merge(self.csv_path, on_conflict="skip")
        unique_bands = set(np.unique(result.band))
        self.assertIn("I", unique_bands)


# ===========================================================================
# Tests for concat()
# ===========================================================================


class TestConcatMixedInputs(unittest.TestCase):
    """7. concat with mixed inputs works."""

    def test_concat_two_lightcurves(self):
        lc1 = _make_2d_lc([1.0], band_labels=["V"])
        lc2 = _make_2d_lc([2.0], band_labels=["R"])
        result = Lightcurve.concat([lc1, lc2])

        self.assertIsInstance(result, Lightcurve)
        self.assertEqual(result.ndim, 2)
        unique_bands = set(np.unique(result.band))
        self.assertEqual(unique_bands, {"V", "R"})

    def test_concat_three_lightcurves(self):
        lc1 = _make_2d_lc([1.0], band_labels=["V"])
        lc2 = _make_2d_lc([2.0], band_labels=["R"])
        lc3 = _make_2d_lc([3.0], band_labels=["I"])
        result = Lightcurve.concat([lc1, lc2, lc3])
        unique_bands = set(np.unique(result.band))
        self.assertEqual(unique_bands, {"V", "R", "I"})

    def test_concat_preserves_order(self):
        lc1 = _make_2d_lc([1.0], n_per_band=5, band_labels=["V"])
        lc2 = _make_2d_lc([2.0], n_per_band=7, band_labels=["R"])
        result = Lightcurve.concat([lc1, lc2])

        n1 = len(lc1._xdata_raw)
        np.testing.assert_array_equal(
            result.band[:n1], np.full(n1, "V")
        )
        np.testing.assert_array_equal(
            result.band[n1:], np.full(7, "R")
        )

    def test_concat_csv_and_lightcurve(self):
        csv_path = _make_csv([1.0], band_labels=["V"])
        try:
            lc2 = _make_2d_lc([2.0], band_labels=["R"])
            result = Lightcurve.concat([csv_path, lc2])
            unique_bands = set(np.unique(result.band))
            self.assertIn("V", unique_bands)
            self.assertIn("R", unique_bands)
        finally:
            os.unlink(csv_path)

    def test_concat_bare_string_single_item(self):
        """A bare str is treated as one CSV item, not char iterable."""
        csv_path = _make_csv([1.0], band_labels=["V"])
        try:
            result = Lightcurve.concat(csv_path)
            self.assertIsInstance(result, Lightcurve)
        finally:
            os.unlink(csv_path)

    def test_concat_result_is_new_object(self):
        lc1 = _make_2d_lc([1.0], band_labels=["V"])
        lc2 = _make_2d_lc([2.0], band_labels=["R"])
        result = Lightcurve.concat([lc1, lc2])
        self.assertIsNot(result, lc1)
        self.assertIsNot(result, lc2)


class TestConcatDuplicateBands(unittest.TestCase):
    """8. concat with duplicate bands → raises / skips."""

    def test_duplicate_band_raises(self):
        lc1 = _make_2d_lc([1.0], band_labels=["V"])
        lc2 = _make_2d_lc([2.0], band_labels=["V"])
        with self.assertRaises(ValueError):
            Lightcurve.concat([lc1, lc2])

    def test_duplicate_band_skip(self):
        lc1 = _make_2d_lc([1.0], band_labels=["V"])
        lc2 = _make_2d_lc([2.0], band_labels=["V"])
        lc3 = _make_2d_lc([3.0], band_labels=["R"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = Lightcurve.concat([lc1, lc2, lc3], on_conflict="skip")
        unique_bands = set(np.unique(result.band))
        self.assertIn("V", unique_bands)
        self.assertIn("R", unique_bands)
        self.assertTrue(any(w))

    def test_duplicate_wavelength_raises(self):
        lc1 = _make_2d_lc([1.0], band_labels=["V"])
        lc2 = _make_2d_lc([1.0], band_labels=["R"])
        with self.assertRaises(ValueError):
            Lightcurve.concat([lc1, lc2])

    def test_empty_items_raises(self):
        with self.assertRaises(ValueError):
            Lightcurve.concat([])

    def test_wrong_type_raises(self):
        with self.assertRaises(TypeError):
            Lightcurve.concat([42])


# ===========================================================================
# Invariant tests
# ===========================================================================


class TestInvariantBandMatchesRowCount(unittest.TestCase):
    """9. band always matches row count after merge / concat."""

    def test_merge_invariant(self):
        lc1 = _make_2d_lc([1.0, 2.0], n_per_band=10, band_labels=["V", "R"])
        lc2 = _make_2d_lc([3.0], n_per_band=12, band_labels=["I"])
        result = lc1.merge(lc2)
        self.assertEqual(len(result._xdata_raw), len(result.band))
        self.assertEqual(len(result._ydata_raw), len(result.band))
        if hasattr(result, "_yerr_raw"):
            self.assertEqual(len(result._yerr_raw), len(result.band))

    def test_concat_invariant(self):
        lc1 = _make_2d_lc([1.0], n_per_band=5, band_labels=["V"])
        lc2 = _make_2d_lc([2.0], n_per_band=7, band_labels=["R"])
        lc3 = _make_2d_lc([3.0], n_per_band=9, band_labels=["I"])
        result = Lightcurve.concat([lc1, lc2, lc3])
        self.assertEqual(len(result._xdata_raw), len(result.band))
        self.assertEqual(len(result._ydata_raw), len(result.band))
        if hasattr(result, "_yerr_raw"):
            self.assertEqual(len(result._yerr_raw), len(result.band))

    def test_merge_skip_invariant(self):
        lc1 = _make_2d_lc([1.0, 2.0], n_per_band=6, band_labels=["V", "R"])
        lc2 = _make_2d_lc([2.0, 3.0], n_per_band=4, band_labels=["R", "I"])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = lc1.merge(lc2, on_conflict="skip")
        self.assertEqual(len(result._xdata_raw), len(result.band))
        self.assertEqual(len(result._ydata_raw), len(result.band))


class TestResultAlways2D(unittest.TestCase):
    """10. Returned object is always 2-D."""

    def test_merge_result_is_2d(self):
        lc1 = _make_2d_lc([1.0], band_labels=["V"])
        lc2 = _make_2d_lc([2.0], band_labels=["R"])
        result = lc1.merge(lc2)
        self.assertEqual(result.ndim, 2)

    def test_merge_1d_other_result_is_2d(self):
        lc1 = _make_2d_lc([1.0], band_labels=["V"])
        lc_1d = _make_1d_lc()
        result = lc1.merge(lc_1d, band="R", wavelength=3.0)
        self.assertEqual(result.ndim, 2)

    def test_concat_result_is_2d(self):
        lc1 = _make_2d_lc([1.0], band_labels=["V"])
        lc2 = _make_2d_lc([2.0], band_labels=["R"])
        result = Lightcurve.concat([lc1, lc2])
        self.assertEqual(result.ndim, 2)


if __name__ == "__main__":
    unittest.main()
